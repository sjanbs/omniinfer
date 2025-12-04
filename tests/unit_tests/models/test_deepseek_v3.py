import pytest
import torch
import weakref
from pytest_mock import MockerFixture
import logging
from typing import Dict
from unittest.mock import patch

from transformers import PretrainedConfig
from vllm.config import set_current_vllm_config, get_layers_from_vllm_config
from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadataBuilder)
from vllm.attention import AttentionType, get_attn_backend
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs
from vllm.v1.worker.block_table import MultiGroupBlockTable
from vllm.utils import is_pin_memory_available
from vllm.platforms import current_platform
from vllm.v1.utils import bind_kv_cache
from vllm.model_executor.model_loader.utils import set_default_torch_dtype, process_weights_after_loading

from omni.models.config_loader.loader import model_extra_config
from omni.models.deepseek.deepseek_v3 import DeepseekV3ForCausalLM
from omni.adaptors.vllm.forward_context import set_forward_context
from omni.layers.attention.backend.attention import AscendAttentionState

from omni.layers.attention.backend.mla import AscendMLADecodeMetadata, AscendMLAMetadata

@pytest.fixture(scope="module")
def base_config():
    config = PretrainedConfig(
        attention_bias=False,
        hidden_size=512,
        num_attention_heads=8,
        num_key_value_heads=8,
        num_nextn_predict_layers=1,
        num_hidden_layers=2,
        intermediate_size=256,
        hidden_act="silu",
        rms_norm_eps=1e-6,
        rope_theta=10000.0,
        max_position_embeddings=2048,
        model_type="deepseek_v3",
        n_routed_experts=4,
        n_shared_experts=1,
        moe_intermediate_size=256,
        num_experts_per_tok=2,
        routed_scaling_factor=1.0,
        first_k_dense_replace=1,
        moe_layer_freq=1,
        q_lora_rank=512,
        kv_lora_rank=512,
        qk_nope_head_dim=128,
        qk_rope_head_dim=64,
        v_head_dim=128,
        topk_method="noaux_tc",
        scoring_func="sigmoid",
        seq_aux=True,
        norm_topk_prob=True,
        n_group=1,
        topk_group=1,
        tie_word_embeddings=False,
        torch_dtype=torch.bfloat16,
        use_cache=True,
        vocab_size=10000,
        rope_scaling={
                        "beta_fast": 32,
                        "beta_slow": 1,
                        "factor": 40,
                        "mscale": 1.0,
                        "mscale_all_dim": 1.0,
                        "original_max_position_embeddings": 4096,
                        "type": "yarn"
                    },
    )
    return config

class Test_DeepseekV3ForCausalLM():

    # Initialize in initialize_kv_cache
    kv_caches: list[torch.Tensor] = []
    attn_metadata_builders: list[AttentionMetadataBuilder] = []
    attn_backends: list[type[AttentionBackend]] = []
    device = torch.device(f"{current_platform.device_type}:0")
    mc2_mask = torch.zeros(64, dtype=torch.bool, device=current_platform.device_type)

    def init_input_block_table(self, vllm_config):
        self.block_table = MultiGroupBlockTable(
            max_num_reqs=vllm_config.scheduler_config.max_num_seqs,
            max_model_len=vllm_config.model_config.max_model_len,
            max_num_batched_tokens=vllm_config.scheduler_config.max_num_batched_tokens,
            pin_memory=is_pin_memory_available(),
            device=self.device,
            block_size=vllm_config.cache_config.block_size,
        )

    def init_attn_backends(self, vllm_config):
        for i, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            assert isinstance(kv_cache_spec, AttentionSpec), "Only AttentionSpec is supported for now."
            attn_backend_i = get_attn_backend(
                kv_cache_spec.head_size,
                vllm_config.model_config.dtype,
                kv_cache_spec.dtype,
                kv_cache_spec.block_size,
                vllm_config.model_config.is_attention_free,
                use_mla=kv_cache_spec.use_mla,
            )
            if attn_backend_i is None:
                error_msg = (
                    f"Error with get_attn_backend: {kv_cache_spec.head_size=}, "
                    f"{vllm_config.model_config.dtype=}, {kv_cache_spec.dtype=}, "
                    f"{kv_cache_spec.block_size=}, "
                    f"{vllm_config.model_config.is_attention_free=}, "
                    f"{kv_cache_spec.use_mla=}")
                logging.error(error_msg)
            assert attn_backend_i is not None, "Non-Attention backend is not supported by V1."

            logging.info(f"[init_attn_backends] {vllm_config.compilation_config.full_cuda_graph=}")
            if vllm_config.compilation_config.full_cuda_graph:
                attn_backend_name = attn_backend_i.__name__
                flash_attn_version = get_flash_attn_version()
                assert not (attn_backend_name != "FlashAttentionBackend" or flash_attn_version != 3), (
                            f"full_cuda_graph is only supported with "
                            f"FA3. Current attention backend is "
                            f"{attn_backend_name}, FlashAttention version is "
                            f"{flash_attn_version}.")

            # block_table_i = self.block_table[i]
            # attn_metadata_builder_i = attn_backend_i.get_builder_cls()(
            #     weakref.proxy(self), kv_cache_spec, block_table_i)
            self.attn_backends.append(attn_backend_i)
            # self.attn_metadata_builders.append(attn_metadata_builder_i)
        logging.info(f"[init_attn_backends] {self.attn_backends=}")
        logging.info(f"[init_attn_backends] {self.attn_metadata_builders=}")

    def get_kv_cache_groups(self):
        grouped_layer_names = [list(self.kv_cache_spec.keys())]
        self.kv_cache_group_specs = create_kv_cache_group_specs(self.kv_cache_spec, grouped_layer_names)
        logging.info(f"return {self.kv_cache_group_specs=}")

    def initialize_kv_cache(self, vllm_config):
        # TODO:
        num_blocks = 128
        kv_caches: Dict[str, torch.Tensor] = {}

        for i, kv_cache_group in enumerate(self.kv_cache_group_specs):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                assert isinstance(kv_cache_spec, AttentionSpec), "Only AttentionSpec is supported for now."
                # adapted for Pangu 72Bv2
                hf_config = vllm_config.model_config.hf_config
                v_channels = getattr(hf_config, "v_channels", None)
                if v_channels is None:
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, 
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, 
                        kv_cache_spec.head_size)
                else:
                    kv_cache_shape = self.attn_backends[i].get_kv_cache_shape(
                        num_blocks, 
                        kv_cache_spec.block_size,
                        kv_cache_spec.num_kv_heads, 
                        kv_cache_spec.head_size,
                        v_channels)

                kv_caches[layer_name] = self.attn_backends[i].init_kv_cache_each_layer(
                    kv_cache_shape, 
                    vllm_config.model_config.dtype,
                    self.device,
                    vllm_config.model_config,
                    False)

        # bind kv cache to Attention and runner.kv_caches
        bind_kv_cache(
            kv_caches,
            vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def init_kv_cache(self, vllm_config):
        # self.init_input_block_table(vllm_config)
        layers = get_layers_from_vllm_config(vllm_config, Attention)
        block_size = vllm_config.cache_config.block_size
        use_mla = vllm_config.model_config.use_mla
        self.kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            logging.info(f"{layer_name=}, {attn_module=}, {attn_module.attn_type=}, {attn_module.sliding_window=}")
            assert attn_module.attn_type == AttentionType.DECODER
            if attn_module.sliding_window is not None:
                self.kv_cache_spec[layer_name] = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=vllm_config.model_config.dtype,
                    sliding_window=attn_module.sliding_window,
                    use_mla=use_mla)
            else:
                self.kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=vllm_config.model_config.dtype,
                    use_mla=use_mla)
            logging.info(f"{layer_name=}, {self.kv_cache_spec[layer_name].type_id=}, {self.kv_cache_spec[layer_name].page_size_bytes=}")
        # get Dict[str, KVCacheSpec]
        logging.info(f"return {self.kv_cache_spec=}")
        self.get_kv_cache_groups()
        self.init_attn_backends(vllm_config)
        # alloc kv cache tensor
        self.initialize_kv_cache(vllm_config)

    def generate_activate_mask(self, actual_seqs_num, batch_size):
        self.mc2_mask.zero_()
        self.mc2_mask[:actual_seqs_num].fill_(True)

    def build_dummy_metadata(self, vllm_config, num_tokens, max_batch_size, model):
        input_positions = torch.zeros(max_batch_size,
                                  dtype=torch.int64,
                                  device=self.device)
        slot_mapping = torch.zeros(max_batch_size,
                                dtype=torch.int64,
                                device=self.device)
        graph_block_tables = torch.zeros((max_batch_size, (vllm_config.model_config.max_model_len + vllm_config.cache_config.block_size - 1) // vllm_config.cache_config.block_size))
        block_table = graph_block_tables.to(
            device=self.device,
            dtype=torch.int32
        )

        seq_lens = torch.ones(max_batch_size, dtype=torch.long, device=self.device, pin_memory=True) * 2
        first_layer_ind = model.model.start_layer
        if isinstance(model.model.layers[first_layer_ind].self_attn, torch.nn.ModuleList):
            cos, sin = model.model.layers[first_layer_ind].self_attn[0].rotary_emb.get_cos_sin(input_positions)
        else:
            cos, sin = model.model.layers[first_layer_ind].self_attn.rotary_emb.get_cos_sin(input_positions)
        best_topk = None
        self.generate_activate_mask(0, max_batch_size)
        decode_metadata = AscendMLADecodeMetadata(
                input_positions=input_positions,
                block_table=block_table,
                seq_lens=seq_lens,
                mc2_mask=self.mc2_mask,
                cos=cos,
                sin=sin,
                best_topk=best_topk)
        return AscendMLAMetadata(  # type: ignore
            num_actual_tokens=num_tokens,
            slot_mapping=slot_mapping,
            num_decodes=num_tokens,
            num_decode_tokens=num_tokens,
            num_prefills=0,
            attn_mask=None,
            attn_state=AscendAttentionState.DecodeOnly,
            prefill=None,
            decode=decode_metadata,
            omni_cache=None
        )

    def forward_dummy(self, vllm_config, model, num_tokens: int):
        input_ids, inputs_embeds = self.input_ids[:num_tokens], None
        intermediate_tensors = None
        positions = self.positions[:num_tokens]
        attn_metadata = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            attn_metadata_builded = self.build_dummy_metadata(vllm_config, num_tokens, 64, model)
            # if self.enable_torchair_graph_mode:
            #     builder.mark_static_for_attn_metadata(attn_metadata_i) # graph need builder? is neccessary?
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_builded
        model_kwargs = {
            "kv_caches": self.kv_caches,
            "attn_metadata": attn_metadata,
            "selected_indices": None
        }
        with set_forward_context(attn_metadata, vllm_config):
            forward_results = model(
                input_ids=input_ids,
                positions=positions,
                intermediate_tensors=intermediate_tensors,
                inputs_embeds=inputs_embeds,
                **model_kwargs
            )

    def test_init_DeepseekV3ForCausalLM(self, mocker: MockerFixture, vllm_config, dist_init):
        logging.info(f"{vllm_config.model_config.use_mla=}, {id(vllm_config)=}")

        # Optional: update model_extra_config
        model_extra_config.task_config.decode_gear_list = [64]

        # initialize model
        class MockDpGroup:
            def __init__(self):
                self.world_size = 2

        device_config = vllm_config.device_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(vllm_config.model_config.dtype):
            with target_device:
                with set_current_vllm_config(vllm_config, check_compile=False):
                    with patch("omni.layers.attention.deepseek_mla.get_dp_group", return_value=MockDpGroup()):
                        model = DeepseekV3ForCausalLM(vllm_config=vllm_config)
            process_weights_after_loading(model, vllm_config.model_config, target_device)
        model.eval()

        assert isinstance(
            model,
            DeepseekV3ForCausalLM,
        )

        # quantization

        # init runner params
        self.max_num_tokens = vllm_config.scheduler_config.max_num_batched_tokens
        self.max_num_reqs=vllm_config.scheduler_config.max_num_seqs
        self.input_ids = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int32,
                                     device=self.device)
        self.positions = torch.zeros(self.max_num_tokens,
                                     dtype=torch.int64,
                                     device=self.device)
        self.seq_lens = torch.zeros(self.max_num_reqs,
                                    dtype=torch.int32,
                                    device=self.device)
        self.slot_mapping = torch.zeros(self.max_num_tokens,
                                        dtype=torch.int64,
                                        device=self.device)

        # init kv_cache
        self.init_kv_cache(vllm_config)

        # test forward dummy
        self.forward_dummy(vllm_config, model, 64)

        # test forward normal


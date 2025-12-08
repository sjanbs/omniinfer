import torch
import logging
import weakref
from typing import Dict
import numpy as np

from vllm.attention.backends.abstract import (AttentionBackend,
                                              AttentionMetadataBuilder)
from vllm.config import set_current_vllm_config, get_layers_from_vllm_config, CompilationLevel
from vllm.attention.layer import Attention
from vllm.attention.utils.fa_utils import get_flash_attn_version
from vllm.attention import AttentionType, get_attn_backend
from vllm.v1.kv_cache_interface import (AttentionSpec, FullAttentionSpec,
                                        KVCacheConfig, KVCacheSpec,
                                        SlidingWindowSpec)
from vllm.v1.core.kv_cache_utils import create_kv_cache_group_specs
from vllm.utils import is_pin_memory_available, supports_dynamo
from vllm.platforms import current_platform
from vllm.v1.utils import bind_kv_cache
from vllm.model_executor.model_loader.utils import set_default_torch_dtype, process_weights_after_loading
from vllm.v1.worker.gpu_input_batch import InputBatch
from omni.adaptors.vllm.forward_context import set_forward_context
from omni.layers.attention.backend.attention import AscendAttentionState
from omni.layers.attention.backend.attention_dummy_builder import DummyAttentionMetadataBuilder

from omni.adaptors.vllm.worker.npu_model_runner import GraphCompileConfiguration, mark_static_for_graph_default

__origin_get_device_properties__ = torch.npu.get_device_properties
class NPUDeviceProperties:
    def __init__(self, device):
        self.properties = __origin_get_device_properties__(device)
        self.multi_processor_count = self.properties.multi_processor_count \
            if hasattr(self.properties, 'multi_processor_count') else 0

def get_device_properties(device):
    return NPUDeviceProperties(device)

class MockRunner:
    def __init__(self, vllm_config, device):
        self.vllm_config = vllm_config
        self.kv_caches: list[torch.Tensor] = []
        self.attn_metadata_builders: list[AttentionMetadataBuilder] = []
        self.attn_backends: list[type[AttentionBackend]] = []
        self.device = device
        torch.npu.set_device(device)
        self.block_size = vllm_config.cache_config.block_size
        self.scheduler_config = vllm_config.scheduler_config
        self.model_config = vllm_config.model_config
        self.max_model_len = self.model_config.max_model_len
        self.max_num_reqs=self.vllm_config.scheduler_config.max_num_seqs
        self.speculative_config = vllm_config.speculative_config
        self.use_spec_decode = False
        if self.speculative_config:
            self.use_spec_decode = True
        self.omni_cache = None
        self.attn_mask = None
        self.attn_state = None
        self.model_mark_static = False
        self.dummy_model_mark_static = False
        self.drafter_mark_static = False
        self.dummy_drafter_mark_static = False
        self.pin_memory = is_pin_memory_available()
        self.mc2_mask = torch.zeros(vllm_config.scheduler_config.max_batch_size, dtype=torch.bool, device=current_platform.device_type)
        self.decode_gear_list = self.vllm_config.npu_compilation_config.decode_gear_list
        if not self.use_spec_decode:
            self.max_batch_size = self.max_num_reqs
        elif not self.speculative_config.enable_adaptive:
            self.max_batch_size = self.max_num_reqs * (1 + self.speculative_config.num_speculative_tokens)
        else:
            if self.decode_gear_list is None or len(self.decode_gear_list) == 0:
                raise RuntimeError("When enable adaptive speculative decoding, decode_gear_list must be set.")
            self.max_batch_size = self.decode_gear_list[0]

        self.max_num_tokens = self.vllm_config.scheduler_config.max_num_batched_tokens
        self.slot_mapping_cpu = torch.zeros(self.max_num_tokens,
                                            dtype=torch.int64,
                                            device="cpu",
                                            pin_memory=is_pin_memory_available())
        self.slot_mapping_np = self.slot_mapping_cpu.numpy()
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

        num_tokens_per_reqs_decode = 1 if not self.use_spec_decode else (1 + self.speculative_config.num_speculative_tokens)
        self.graph_block_tables = np.zeros(
            (self.max_num_reqs * num_tokens_per_reqs_decode,
             (self.model_config.max_model_len + self.block_size - 1) // self.block_size),
            dtype=np.int32)
        self.positions_cpu = torch.zeros(self.max_num_tokens,
                                         dtype=torch.int64,
                                         device="cpu",
                                         pin_memory=self.pin_memory)
        self.seq_lens_cpu = torch.zeros(self.max_num_reqs,
                                        dtype=torch.int32,
                                        device="cpu",
                                        pin_memory=self.pin_memory)
        self.seq_lens_np = self.seq_lens_cpu.numpy()
        self.input_batch = InputBatch(
            max_num_reqs=self.max_num_reqs,
            max_model_len=self.max_model_len,
            max_num_batched_tokens=self.max_num_tokens,
            device=self.device,
            pin_memory=self.pin_memory,
            vocab_size=self.model_config.hf_text_config.vocab_size,
            block_size=self.block_size,
        )

        self.enable_torchair_graph_mode = (
                    self.vllm_config.npu_compilation_config.level > CompilationLevel.NO_COMPILATION and supports_dynamo())

        logging.info(f"MockRunner: enable_torchair_graph_mode: {self.enable_torchair_graph_mode}")

        torch.cuda.get_device_properties = get_device_properties

    def init_model(self, model_cls):
        device_config = self.vllm_config.device_config
        target_device = torch.device(device_config.device)
        with set_default_torch_dtype(self.vllm_config.model_config.dtype):
            with target_device:
                with set_current_vllm_config(self.vllm_config, check_compile=False):
                    self.model = model_cls(vllm_config=self.vllm_config)
            # quantization
            process_weights_after_loading(self.model, self.vllm_config.model_config, target_device)
        self.model.eval()
        return self.model

    def init_attn_backends(self):
        for i, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            kv_cache_spec = kv_cache_group_spec.kv_cache_spec
            assert isinstance(kv_cache_spec, AttentionSpec), "Only AttentionSpec is supported for now."
            attn_backend_i = get_attn_backend(
                kv_cache_spec.head_size,
                self.vllm_config.model_config.dtype,
                kv_cache_spec.dtype,
                kv_cache_spec.block_size,
                self.vllm_config.model_config.is_attention_free,
                use_mla=kv_cache_spec.use_mla,
            )
            if attn_backend_i is None:
                error_msg = (
                    f"Error with get_attn_backend: {kv_cache_spec.head_size=}, "
                    f"{self.vllm_config.model_config.dtype=}, {kv_cache_spec.dtype=}, "
                    f"{kv_cache_spec.block_size=}, "
                    f"{self.vllm_config.model_config.is_attention_free=}, "
                    f"{kv_cache_spec.use_mla=}")
                logging.error(error_msg)
            assert attn_backend_i is not None, "Non-Attention backend is not supported by V1."

            logging.info(f"[init_attn_backends] {self.vllm_config.compilation_config.full_cuda_graph=}")
            if self.vllm_config.compilation_config.full_cuda_graph:
                attn_backend_name = attn_backend_i.__name__
                flash_attn_version = get_flash_attn_version()
                assert not (attn_backend_name != "FlashAttentionBackend" or flash_attn_version != 3), (
                            f"full_cuda_graph is only supported with "
                            f"FA3. Current attention backend is "
                            f"{attn_backend_name}, FlashAttention version is "
                            f"{flash_attn_version}.")

            block_table_i = self.input_batch.block_table[i]
            attn_metadata_builder_i = attn_backend_i.get_builder_cls()(
                weakref.proxy(self), kv_cache_spec, block_table_i)
            self.attn_backends.append(attn_backend_i)
            self.attn_metadata_builders.append(attn_metadata_builder_i)
        logging.info(f"[init_attn_backends] {self.attn_backends=}")
        logging.info(f"[init_attn_backends] {self.attn_metadata_builders=}")

    def get_kv_cache_groups(self):
        grouped_layer_names = [list(self.kv_cache_spec.keys())]
        self.kv_cache_group_specs = create_kv_cache_group_specs(self.kv_cache_spec, grouped_layer_names)
        logging.info(f"return {self.kv_cache_group_specs=}")

    def initialize_kv_cache(self):
        # TODO:
        num_blocks = 128
        kv_caches: Dict[str, torch.Tensor] = {}

        for i, kv_cache_group in enumerate(self.kv_cache_group_specs):
            kv_cache_spec = kv_cache_group.kv_cache_spec
            for layer_name in kv_cache_group.layer_names:
                assert isinstance(kv_cache_spec, AttentionSpec), "Only AttentionSpec is supported for now."
                # adapted for Pangu 72Bv2
                hf_config = self.vllm_config.model_config.hf_config
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
                    self.vllm_config.model_config.dtype,
                    self.device,
                    self.vllm_config.model_config,
                    False)

        # bind kv cache to Attention and runner.kv_caches
        bind_kv_cache(
            kv_caches,
            self.vllm_config.compilation_config.static_forward_context,
            self.kv_caches)

    def init_kv_cache(self):
        layers = get_layers_from_vllm_config(self.vllm_config, Attention)
        block_size = self.vllm_config.cache_config.block_size
        use_mla = self.vllm_config.model_config.use_mla
        self.kv_cache_spec: dict[str, KVCacheSpec] = {}
        for layer_name, attn_module in layers.items():
            logging.info(f"{layer_name=}, {attn_module=}, {attn_module.attn_type=}, {attn_module.sliding_window=}")
            assert attn_module.attn_type == AttentionType.DECODER
            if attn_module.sliding_window is not None:
                self.kv_cache_spec[layer_name] = SlidingWindowSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.vllm_config.model_config.dtype,
                    sliding_window=attn_module.sliding_window,
                    use_mla=use_mla)
            else:
                self.kv_cache_spec[layer_name] = FullAttentionSpec(
                    block_size=block_size,
                    num_kv_heads=attn_module.num_kv_heads,
                    head_size=attn_module.head_size,
                    dtype=self.vllm_config.model_config.dtype,
                    use_mla=use_mla)
            logging.info(f"{layer_name=}, {self.kv_cache_spec[layer_name].type_id=}, {self.kv_cache_spec[layer_name].page_size_bytes=}")
        # get Dict[str, KVCacheSpec]
        logging.info(f"return {self.kv_cache_spec=}")
        self.get_kv_cache_groups()
        self.init_attn_backends()
        # alloc kv cache tensor
        self.initialize_kv_cache()

    @torch.inference_mode()
    def _dummy_run(self, num_tokens: int, total_steps: int = 1):
        input_ids, inputs_embeds = self.input_ids[:num_tokens], None
        intermediate_tensors = None
        positions = self.positions[:num_tokens]

        # No kv_caches: profile run
        if not self.kv_caches:
            logging.debug("Start running profile dummy.")
            with set_forward_context(None, self.vllm_config):
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                )
            return forward_results
        self.attn_state = AscendAttentionState.DecodeOnly
        attn_metadata = {}
        for kv_cache_group_id, kv_cache_group_spec in enumerate(self.kv_cache_group_specs):
            builder = self.attn_metadata_builders[kv_cache_group_id]
            assert isinstance(builder, DummyAttentionMetadataBuilder), f"{builder} does not implement DummyAttentionMetadataBuilder"
            attn_metadata_i = builder.build_dummy(num_tokens, self.max_batch_size)
            if self.enable_torchair_graph_mode:
                builder.mark_static_for_attn_metadata(attn_metadata_i)
            for layer_name in kv_cache_group_spec.layer_names:
                attn_metadata[layer_name] = attn_metadata_i
        model_kwargs = {
            "kv_caches": self.kv_caches,
            "attn_metadata": attn_metadata,
            "selected_indices": None
        }
        with set_forward_context(attn_metadata, self.vllm_config):
            use_compile = self.enable_torchair_graph_mode
            for _ in range(total_steps):
                if use_compile:
                    logging.debug("Start running dummy compiled model.")
                    if not self.dummy_model_mark_static:
                        if isinstance(self.model, GraphCompileConfiguration):
                            self.model.mark_static_for_graph(input_ids, positions, attn_metadata, self.kv_caches)
                        else:
                            mark_static_for_graph_default(input_ids, inputs_embeds, positions, self.kv_caches)
                        self.dummy_model_mark_static = True
                else:
                    logging.debug("Start running dummy eager model.")
                forward_results = self.model(
                    input_ids=input_ids,
                    positions=positions,
                    intermediate_tensors=intermediate_tensors,
                    inputs_embeds=inputs_embeds,
                    **model_kwargs
                )

        return forward_results

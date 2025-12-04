import pytest
import torch
from pytest_mock import MockerFixture
import logging
from unittest.mock import patch

from transformers import PretrainedConfig
from vllm.platforms import current_platform

from omni.models.config_loader.loader import model_extra_config
from omni.models.deepseek.deepseek_v3 import DeepseekV3ForCausalLM

from .base import MockRunner

class Test_DeepseekV3ForCausalLM():

    @pytest.fixture(scope="class")
    def base_config(self):
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

    @pytest.fixture(autouse=True)
    def setup(self, vllm_config):
        logging.info(f"Setting up Test_DeepseekV3ForCausalLM class {id(vllm_config)=}")
        self.vllm_config = vllm_config
        # Optional: update vllm_config

        # Optional: update model_extra_config
        model_extra_config.task_config.decode_gear_list = [vllm_config.scheduler_config.max_batch_size]

        self.device = torch.device(f"{current_platform.device_type}:0")
        self.mock_runner = MockRunner(vllm_config, self.device)

    def test_init_DeepseekV3ForCausalLM(self, mocker: MockerFixture, dist_init):
        logging.info(f"{self.vllm_config.model_config.use_mla=}, {id(self.vllm_config)=}")
        assert self.vllm_config.model_config.use_mla == True

        # initialize model
        class MockDpGroup:
            def __init__(self):
                self.world_size = 2
        with patch("omni.layers.attention.deepseek_mla.get_dp_group", return_value=MockDpGroup()):
            self.model = self.mock_runner.init_model(DeepseekV3ForCausalLM)
        assert isinstance(
            self.model,
            DeepseekV3ForCausalLM,
        )

        # init kv_cache
        self.mock_runner.init_kv_cache()

        # test forward dummy
        forward_results = self.mock_runner.forward_dummy(self.vllm_config.scheduler_config.max_batch_size)
        logging.info(f"{forward_results[0].shape=}, {forward_results[1].shape=}")
        assert forward_results[0].shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.hidden_size])
        assert forward_results[1].shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.vocab_size])

        # test forward normal

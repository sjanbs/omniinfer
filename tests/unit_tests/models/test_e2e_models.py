from omni.adaptors.vllm.patches import model_patch
import pytest
import os
import torch
import torch.multiprocessing as mp
import tempfile
from unittest.mock import patch

from vllm.platforms import current_platform
from vllm.config import CompilationLevel
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.config import set_current_vllm_config

from omni.models.config_loader.loader import model_extra_config

from .base import MockRunner
from .utils import creat_vllm_config
from .registry import HF_EXAMPLE_MODELS

class Test_e2e_models():
    def init_distributed(self, local_rank, world_size: int):
        with set_current_vllm_config(self.vllm_config, check_compile=False):
            init_distributed_environment(
                world_size=world_size,
                rank=local_rank,
                distributed_init_method=f"file://{self.temp_file_path}",
                local_rank=local_rank,
                backend="hccl",
            )
            initialize_model_parallel(world_size, 1)

    def _model_runner(self, local_rank: int, world_size: int, model_info, enable_graph):
        self.vllm_config = creat_vllm_config(model_info.hf_config)
        print(f"{world_size=}, {local_rank=}, {self.vllm_config.model_config.use_mla=}, {id(self.vllm_config)=}")
        assert self.vllm_config.model_config.use_mla == True

        # Optional: update vllm_config
        self.vllm_config.npu_compilation_config.decode_gear_list = [self.vllm_config.scheduler_config.max_batch_size]

        # Optional: update model_extra_config
        model_extra_config.task_config.decode_gear_list = [self.vllm_config.scheduler_config.max_batch_size]

        torch.npu.set_device(local_rank)
        self.device = torch.device(f"{current_platform.device_type}:0")

        self.init_distributed(local_rank, world_size)

        self.mock_runner = MockRunner(self.vllm_config, self.device)

        # initialize model
        with model_info.init_patch_context():
            self.model = self.mock_runner.init_model(model_info.model_cls)
        assert isinstance(
            self.model,
            model_info.model_cls,
        )

        # enable graph compile
        if enable_graph:
            self.vllm_config.npu_compilation_config.level = CompilationLevel.DYNAMO_AS_IS

        # profile run
        self.mock_runner._dummy_run(self.vllm_config.scheduler_config.max_batch_size)

        # init kv_cache
        self.mock_runner.init_kv_cache()

        # test forward dummy
        forward_results = self.mock_runner._dummy_run(self.vllm_config.scheduler_config.max_batch_size)
        print(f"{forward_results[0].shape=}, {forward_results[1].shape=}")
        assert forward_results[0].shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.hidden_size])
        assert forward_results[1].shape == torch.Size([self.vllm_config.scheduler_config.max_batch_size, self.vllm_config.model_config.hf_config.vocab_size])

    @pytest.mark.parametrize("enable_graph", [False, True])
    @pytest.mark.parametrize("world_size", [1])
    @pytest.mark.parametrize("model_arch", HF_EXAMPLE_MODELS.get_supported_archs())
    def test_model(self, world_size: int, model_arch: str, enable_graph: bool):
        model_info = HF_EXAMPLE_MODELS.get_hf_info(model_arch)

        with tempfile.NamedTemporaryFile(delete=False) as tfile:
            self.temp_file_path = tfile.name

        try:
            mp.spawn(
                self._model_runner,
                args=(world_size, model_info, enable_graph),
                nprocs=world_size,
            )
        finally:
            if os.path.exists(self.temp_file_path):
                os.remove(self.temp_file_path)

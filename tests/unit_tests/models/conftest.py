from omni.adaptors.vllm.patches import model_patch

import tempfile
import logging
from collections import Counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock, patch

import pytest
import torch
from vllm.config import CacheConfig, ParallelConfig, CompilationLevel, CompilationConfig, SchedulerConfig, DeviceConfig, set_current_vllm_config
from vllm.distributed import (cleanup_dist_env_and_memory,
                              init_distributed_environment,
                              initialize_model_parallel)
from vllm.platforms import current_platform
from omni.adaptors.vllm.compilation.compile_config import NPUCompilationConfig

@pytest.fixture
def vllm_config(base_config):
    model_config = SimpleNamespace(
        hf_config=base_config,
        hf_text_config=base_config,
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
        use_mla=True,
        quant_config=None,
        max_model_len=2048,
        is_attention_free=False,
    )

    parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        )
    cache_config = CacheConfig()
    cache_config.block_size = 128
    compilation_config = MagicMock(spec=CompilationConfig)
    compilation_config.level = 0
    compilation_config.full_cuda_graph = False
    # compilation_config.custom_ops = ["none", "+rms_norm", "+rotary_embedding"]
    compilation_config.custom_ops = ['all']
    compilation_config.enabled_custom_ops = MagicMock(spec=Counter[str])
    compilation_config.disabled_custom_ops = MagicMock(spec=Counter[str])
    # compilation_config.static_forward_context = MagicMock(spec=dict[str, Any])
    compilation_config.static_forward_context = {}
    npu_compilation_config = MagicMock(spec=NPUCompilationConfig)
    npu_compilation_config.level = CompilationLevel.NO_COMPILATION

    scheduler_config = MagicMock(spec=SchedulerConfig)
    scheduler_config.max_num_seqs = 4
    scheduler_config.max_num_batched_tokens = 2048
    scheduler_config.preemption_mode = None

    device_config = DeviceConfig(device=current_platform.device_type)

    vllm_config = Mock()
    vllm_config.model_config = model_config
    vllm_config.cache_config = cache_config
    vllm_config.quant_config = None
    vllm_config.parallel_config = parallel_config
    vllm_config.compilation_config = compilation_config
    vllm_config.npu_compilation_config = npu_compilation_config
    vllm_config.scheduler_config = scheduler_config
    vllm_config.device_config = device_config
    return vllm_config

@pytest.fixture
def dist_init(vllm_config):
    with set_current_vllm_config(vllm_config, check_compile=False):
        temp_file = tempfile.mkstemp()[1]
        init_distributed_environment(
            world_size=1,
            rank=0,
            distributed_init_method=f"file://{temp_file}",
            local_rank=0,
            backend="hccl",
        )
        initialize_model_parallel(1, 1)
    yield
    cleanup_dist_env_and_memory()

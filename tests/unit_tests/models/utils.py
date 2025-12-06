from collections import Counter
from types import SimpleNamespace
from typing import Any
from unittest.mock import MagicMock, Mock

import torch
from vllm.config import CacheConfig, ParallelConfig, CompilationLevel, CompilationConfig, SchedulerConfig, DeviceConfig
from vllm.platforms import current_platform
from omni.adaptors.vllm.compilation.compile_config import NPUCompilationConfig

def creat_vllm_config(base_config):
    model_config = SimpleNamespace(
        hf_config=base_config,
        hf_text_config=base_config,
        tensor_parallel_size=1,
        dtype=torch.bfloat16,
        use_mla=True,
        quant_config=None,
        max_model_len=2048,
        is_attention_free=False,
        disable_cascade_attn = True,
    )

    parallel_config=ParallelConfig(
            pipeline_parallel_size=1,
            tensor_parallel_size=1,
            data_parallel_size=1,
        )
    cache_config = CacheConfig()
    cache_config.block_size = 128
    compilation_config = CompilationConfig()
    compilation_config.level = 0
    compilation_config.full_cuda_graph = False
    compilation_config.custom_ops = ['all']
    compilation_config.enabled_custom_ops = MagicMock(spec=Counter[str])
    compilation_config.disabled_custom_ops = MagicMock(spec=Counter[str])
    compilation_config.static_forward_context = {}
    npu_compilation_config = NPUCompilationConfig()
    npu_compilation_config.level = CompilationLevel.NO_COMPILATION
    npu_compilation_config.use_ge_graph_cached = False

    scheduler_config = SchedulerConfig()
    scheduler_config.max_num_seqs = 4
    scheduler_config.max_num_batched_tokens = 2048
    scheduler_config.max_batch_size = 4
    scheduler_config.preemption_mode = None
    scheduler_config.chunked_prefill_enabled = False


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
    vllm_config.speculative_config = None
    return vllm_config

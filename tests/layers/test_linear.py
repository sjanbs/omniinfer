import os
import pytest
import torch
import torch.multiprocessing as mp
import tempfile
from contextlib import contextmanager
from typing import Callable, Any, List, Tuple
from unittest.mock import MagicMock

from omni.layers.linear import AscendMergedColumnParallelLinear, AscendRowParallelLinear
from omni.models.config_loader.loader import model_extra_config
from vllm.distributed import (
    init_distributed_environment,
    initialize_model_parallel,
    tensor_model_parallel_reduce_scatter
)
from vllm.utils import update_environment_variables
from vllm.config import VllmConfig, DeviceConfig


TEST_SEED = 0
torch.manual_seed(TEST_SEED)

PARAM_CASES = [
    (torch.float32, 64, 16, 3),
    (torch.bfloat16, 64, 16, 3),
    (torch.float32, 128, 32, 5),
    (torch.float64, 32, 8, 1),
    (torch.bfloat16, 256, 64, 8),
]

PARAM_IDS = [
    f"{str(dtype)}_{in_sz}_{out_sz}_b{batch}"
    for dtype, in_sz, out_sz, batch in PARAM_CASES
]

@contextmanager
def distributed_context(local_rank: int, world_size: int, temp_file_path: str):
    """
    Context manager to handle setting up and tearing down the 
    distributed environment inside a worker process.
    """
    torch.npu.set_device(local_rank)
    os.environ["GLOO_SOCKET_IFNAME"] = "lo"
    
    # Generic Environment Setup
    update_environment_variables({
        "RANK": str(local_rank),
        "LOCAL_RANK": str(local_rank),
        "WORLD_SIZE": str(world_size),
        "MASTER_ADDR": "127.0.0.1",
        "MASTER_PORT": "29500",
    })

    # Initialize Distributed Backend
    init_distributed_environment(
        distributed_init_method=f"file://{temp_file_path}",
        rank=local_rank,
        local_rank=local_rank,
        world_size=world_size,
        backend="hccl",
    )
    
    # Initialize Model Parallelism (Tensor Parallel)
    initialize_model_parallel(tensor_model_parallel_size=world_size)
    
    try:
        yield
    finally:
        torch.distributed.destroy_process_group()

def _worker_wrapper(local_rank: int, world_size: int, temp_file_path: str, 
                    func: Callable, args: Tuple, kwargs: dict):
    """
    Wraps the actual test logic with the distributed context.
    """
    model_extra_config.parall_config.dense_mlp_tp_size = world_size
    
    with distributed_context(local_rank, world_size, temp_file_path):
        func(local_rank, world_size, *args, **kwargs)

def run_distributed_test(num_processes: int, func: Callable, *args, **kwargs):
    """
    Spawns processes and manages the rendezvous file.
    Usage: run_distributed_test(2, my_test_logic_function, arg1, arg2)
    """
    with tempfile.NamedTemporaryFile(delete=False) as tfile:
        temp_file_path = tfile.name

    try:
        mp.spawn(
            _worker_wrapper,
            args=(num_processes, temp_file_path, func, args, kwargs),
            nprocs=num_processes,
        )
    finally:
        if os.path.exists(temp_file_path):
            os.remove(temp_file_path)


def _shard_merged_weight(full_weight: torch.Tensor, output_sizes: List[int], 
                         tp_size: int, tp_rank: int) -> torch.Tensor:
    shards = []
    start = 0
    for size in output_sizes:
        per_partition = size // tp_size
        block = full_weight[start:start + size]
        shards.append(block[tp_rank * per_partition:(tp_rank + 1) * per_partition])
        start += size
    return torch.cat(shards, dim=0)

# Tests start here

def _logic_merged_linear_spawn(local_rank, world_size, full_weight, input_tensor, 
                               input_size, output_sizes, dtype):
    """Core logic for testing AscendMergedColumnParallelLinear in distributed mode."""
    device = torch.device("npu")
    local_input = input_tensor.to(device)
    
    layer = AscendMergedColumnParallelLinear(
        input_size=input_size,
        output_sizes=output_sizes,
        bias=False,
        gather_output=True,
        tp_size=world_size,
        tp_rank=local_rank,
        params_dtype=dtype,
        prefix="test_merged_spawn",
    ).to(device)

    layer.weight_loader(layer.weight, full_weight.to(device))
    out, bias = layer(local_input)
    
    assert bias is None

    # 1. Calculate Golden Result (Standard MM)
    expected_full_standard = torch.matmul(local_input, full_weight.to(device).T)
    
    # 2. Simulate TP Interleaving
    partition_sizes = [s // world_size for s in output_sizes]
    logical_parts = torch.split(expected_full_standard, output_sizes, dim=-1)
    
    sharded_parts = []
    for part, p_size in zip(logical_parts, partition_sizes):
        sharded_parts.append(torch.split(part, p_size, dim=-1))
        
    interleaved_chunks = []
    for rank in range(world_size):
        for part_idx in range(len(output_sizes)):
            interleaved_chunks.append(sharded_parts[part_idx][rank])
            
    expected_full_interleaved = torch.cat(interleaved_chunks, dim=-1)

    assert out.shape == expected_full_interleaved.shape
    assert torch.allclose(out, expected_full_interleaved, atol=1e-4, rtol=1e-5)

class AscendMMRSModel(torch.nn.Module):
    def __init__(self, hidden_size=16, dtype=torch.bfloat16, device=None):
        super().__init__()
        self.dtype = dtype
        self.device = torch.device(f"npu:{torch.npu.current_device()}") if device == "npu" else torch.device(device)
        self.hidden_size = hidden_size
        weight_shape = (self.hidden_size * 2, hidden_size)
        self.gate_proj = torch.nn.Parameter(
            torch.empty(weight_shape, dtype=self.dtype, device=self.device),
            requires_grad=False,
        )
        torch.nn.init.normal_(self.gate_proj, std=0.02)

    def forward(self, hidden_states):
        view = hidden_states.reshape(-1, self.hidden_size)
        permute = self.gate_proj.permute(1, 0)
        mm = torch.mm(view, permute)
        reduce_scatter = tensor_model_parallel_reduce_scatter(mm, dim=0)
        return reduce_scatter

def _logic_tp_smoke(local_rank, world_size, test_model_cls, batch_size, 
                           seq_len, hidden_size, dtype):
    """Core logic for testing the async TP pass model."""
    # Setup minimal vLLM config (if needed for internal checks)
    vllm_config = VllmConfig()
    vllm_config.device_config = DeviceConfig(device=torch.device("npu"))

    model = test_model_cls(hidden_size, dtype=dtype, device="npu")

    hidden_states = torch.randn((batch_size * seq_len, hidden_size),
                                dtype=dtype, device="npu",
                                requires_grad=False)

    compiled_model = model 
    compiled_model(hidden_states)
    
    # If no crash, we assume pass (assertions can be added for output validity)
    assert True

@pytest.mark.parametrize("dtype,input_size,output_size,batch", PARAM_CASES, ids=PARAM_IDS)
def test_ascend_row_parallel_linear_basic(dtype, input_size, output_size, batch):
    """Single process test for RowParallelLinear."""
    torch.manual_seed(TEST_SEED)
    full_weight = torch.randn(output_size, input_size, dtype=dtype)

    layer0 = AscendRowParallelLinear(
        input_size, output_size, tp_size=1, tp_rank=0, bias=False,
        input_is_parallel=False, skip_bias_add=False, params_dtype=dtype,
        reduce_results=False, quant_config=None, prefix="test_linear",
    )
    
    layer0.weight_loader(layer0.weight, full_weight.clone())
    input_tensor = torch.randn(batch, input_size, dtype=dtype)
    out0, out_bias0 = layer0(input_tensor)
    expected_out0 = torch.matmul(input_tensor, full_weight.T)

    assert torch.allclose(out0, expected_out0, atol=1e-6, rtol=1e-5)
    assert out_bias0 is None

@pytest.mark.parametrize("tp_rank", [0, 1])
def test_ascend_merged_column_parallel_linear_sharding(tp_rank: int):
    """Single process test for MergedColumn sharding logic."""
    torch.manual_seed(TEST_SEED)
    input_size = 4
    output_sizes = [6, 10]
    tp_size = 2
    batch = 5
    dtype = torch.float32

    full_weight = torch.randn(sum(output_sizes), input_size, dtype=dtype)
    layer = AscendMergedColumnParallelLinear(
        input_size=input_size, output_sizes=output_sizes, bias=False,
        gather_output=False, tp_size=tp_size, tp_rank=tp_rank,
        params_dtype=dtype, prefix="test_merged_shard",
    )
    layer.weight_loader(layer.weight, full_weight.clone())

    input_tensor = torch.randn(batch, input_size, dtype=dtype)
    expected_weight = _shard_merged_weight(full_weight, output_sizes, tp_size, tp_rank)
    expected_out = torch.matmul(input_tensor, expected_weight.T)

    out, bias = layer(input_tensor)

    assert bias is None
    assert out.shape == expected_out.shape
    assert torch.allclose(out, expected_out, atol=1e-6, rtol=1e-5)

@pytest.mark.parametrize("test_model", [AscendMMRSModel])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_async_tp_smoke(test_model, batch_size, seq_len, hidden_size, dtype):
    """Distributed test: spawns processes to test FX pass / Model."""
    num_processes = 2
    run_distributed_test(
        num_processes,
        _logic_tp_smoke,
        # Args passed to worker:
        test_model, batch_size, seq_len, hidden_size, dtype
    )

@pytest.mark.parametrize("dtype", [torch.float32])
def test_ascend_merged_column_parallel_linear_gather_output_tp_spawn(dtype):
    """Distributed test: spawns processes to test merged linear gather."""
    num_processes = 2
    input_size = 8
    output_sizes = [20, 40]
    # important: num_processes must | any output sizes
    batch = 3

    full_weight = torch.randn(sum(output_sizes), input_size, dtype=dtype)
    input_tensor = torch.randn(batch, input_size, dtype=dtype)

    run_distributed_test(
        num_processes,
        _logic_merged_linear_spawn,
        # Args passed to worker:
        full_weight, input_tensor, input_size, output_sizes, dtype
    )

# ... existing imports ...
# Ensure you import the RowParallelLinear class you provided (or define it in the test if it's not in the library yet)
from omni.layers.linear import RowParallelLinear 

# --- New Test Logic for RowParallelLinear ---

def _logic_row_parallel_distributed(local_rank, world_size, 
                                    input_size, output_size, batch_size, dtype,
                                    input_is_parallel, skip_bias_add, reduce_results):
    """
    Distributed worker logic for RowParallelLinear.
    
    Concept:
    Y = XW + b
    In Row Parallelism, we split X and W along the 'k' (input) dimension.
    Y = (X1 * W1) + (X2 * W2) + ... + b
    """
    device = torch.device("npu")
    # BF16 requires significantly looser tolerances for distributed reductions
    if dtype == torch.bfloat16:
        atol, rtol = 1e-2, 1e-2  # ~0.01 tolerance
    elif dtype == torch.float16:
        atol, rtol = 5e-3, 5e-3
    else: # float32
        atol, rtol = 1e-5, 1e-5
    # 1. Setup Golden Model
    golden = torch.nn.Linear(input_size, output_size, bias=True).to(dtype).to(device)
    
    # 2. Setup Unit Under Test
    layer = RowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        bias=True,
        input_is_parallel=input_is_parallel,
        skip_bias_add=skip_bias_add,
        params_dtype=dtype,
        reduce_results=reduce_results,
        prefix="test_row_p"
    ).to(device)

    # 3. Manually Shard Weights
    part_size = input_size // world_size
    start = local_rank * part_size
    end = (local_rank + 1) * part_size
    
    with torch.no_grad():
        layer.weight.data.copy_(golden.weight.data[:, start:end])
        if local_rank == 0:
            layer.bias.data.copy_(golden.bias.data)
        else:
            layer.bias.data.zero_()

    # 4. Prepare Input
    full_input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    
    if input_is_parallel:
        layer_input = full_input[..., start:end].detach().clone()
    else:
        layer_input = full_input.detach().clone()

    # 5. Run Forward
    out, out_bias = layer(layer_input)

    # Calculate Expected Partial Components
    # Slice the weight for this rank: [Output_Size, Input_Shard_Size]
    weight_shard = golden.weight.data[:, start:end]
    input_shard = full_input[..., start:end]
    
    # Pure Partial MatMul (No bias added yet)
    expected_partial_matmul = torch.matmul(input_shard, weight_shard.T)

    if reduce_results:
        # --- Case A: All-Reduce was performed ---
        # The output 'out' should be the FULL sum across all ranks
        
        # Calculate expected full output (XW + b)
        expected_full = golden(full_input) 
        
        if skip_bias_add:
            # If we skipped adding bias, 'out' should match XW (full matmul without bias)
            expected_matmul_only = torch.matmul(full_input, golden.weight.T)
            
            # print(f"expected_matmul_only = {expected_matmul_only}", flush=True)
            # print(f"out = {out}", flush=True)
            assert torch.allclose(out, expected_matmul_only, atol=atol, rtol=rtol), \
                f"Rank {local_rank}: Full MatMul mismatch (skip_bias_add=True)"

            # And out_bias should be returned
            if local_rank == 0:
                assert torch.allclose(out_bias, golden.bias, atol=atol, rtol=rtol)
        else:
            # Standard case: out = XW + b
            assert torch.allclose(out, expected_full, atol=atol, rtol=rtol), \
                f"Rank {local_rank}: Full output mismatch"
                
    else:
        # --- Case B: NO Reduce performed (Partial Results) ---
        # Valid combination: reduce_results=False AND skip_bias_add=True
        
        # 'out' should be exactly the partial matmul (X_shard * W_shard)
        assert torch.allclose(out, expected_partial_matmul, atol=atol, rtol=rtol), \
            f"Rank {local_rank}: Partial MatMul mismatch (reduce_results=False)"
            
        # Check that bias was returned correctly so it can be reduced later
        if local_rank == 0:
            assert torch.allclose(out_bias, golden.bias, atol=atol, rtol=rtol), \
                "Rank 0: Bias should be returned in out_bias"
        else:
            # On other ranks, out_bias is usually None or zero-filled depending on impl.
            # Omni/vLLM usually returns None on non-zero ranks if skip_bias_add=True
            if out_bias is not None:
                assert torch.allclose(out_bias, torch.zeros_like(out_bias)), \
                    f"Rank {local_rank}: Bias should be None or Zero"

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_is_parallel", [True, False])
@pytest.mark.parametrize("skip_bias_add", [True, False])
def test_row_parallel_linear_distributed(dtype, input_is_parallel, skip_bias_add):
    """
    Tests RowParallelLinear in a 2-GPU distributed setting.
    """
    num_processes = 2
    input_size = 32
    output_size = 16
    batch_size = 4
    reduce_results = True # Standard usage

    run_distributed_test(
        num_processes,
        _logic_row_parallel_distributed,
        # Args:
        input_size, output_size, batch_size, dtype,
        input_is_parallel, skip_bias_add, reduce_results
    )

@pytest.mark.parametrize("dtype", [torch.float32])
def test_row_parallel_no_reduce(dtype):
    """
    Tests RowParallelLinear with reduce_results=False.
    This is often used when feeding into another specific layer or custom optimization.
    """
    num_processes = 2
    input_size = 32
    output_size = 16
    batch_size = 4
    
    # Fixed flags for this specific test case
    input_is_parallel = True 
    skip_bias_add = True
    reduce_results = False

    run_distributed_test(
        num_processes,
        _logic_row_parallel_distributed,
        # Args:
        input_size, output_size, batch_size, dtype,
        input_is_parallel, skip_bias_add, reduce_results
    )
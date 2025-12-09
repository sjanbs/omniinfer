import os
import pytest
import torch
import torch.multiprocessing as mp
import tempfile
import traceback
import sys
import importlib
from typing import Callable, Any, List, Tuple

# Top-level imports for pytest parameterization
from omni.layers.linear import (
    AscendMergedColumnParallelLinear,
    AscendRowParallelLinear,
    DP2TPRowParallelLinear,
    RowParallelLinear,
    Tp2DpAndTpRowParallelLinear,
)
from omni.models.config_loader.loader import model_extra_config

from .distributed_test_common import distributed_worker_pool, _persistent_worker_loop
TEST_SEED = 0

def _shard_merged_weight(full_weight, output_sizes, tp_size, tp_rank):
    shards = []
    start = 0
    for size in output_sizes:
        per_partition = size // tp_size
        block = full_weight[start:start + size]
        shards.append(block[tp_rank * per_partition:(tp_rank + 1) * per_partition])
        start += size
    return torch.cat(shards, dim=0)

# --- Logic Functions ---

def _logic_merged_linear_spawn(local_rank, world_size, full_weight, input_tensor, 
                               input_size, output_sizes, dtype):
    # CRITICAL: Import the class locally to ensure we use the reloaded version
    from omni.layers.linear import AscendMergedColumnParallelLinear
    
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

    expected_full_standard = torch.matmul(local_input, full_weight.to(device).T)
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

def _logic_tp_smoke(local_rank, world_size, test_model_cls, batch_size, 
                            seq_len, hidden_size, dtype):
    model = test_model_cls(hidden_size, dtype=dtype, device="npu")
    hidden_states = torch.randn((batch_size * seq_len, hidden_size),
                                dtype=dtype, device="npu",
                                requires_grad=False)
    model(hidden_states)
    assert True

def _logic_row_parallel_distributed(local_rank, world_size, 
                                    input_size, output_size, batch_size, dtype,
                                    input_is_parallel, skip_bias_add, reduce_results):
    # Import locally to use reloaded module
    from omni.layers.linear import RowParallelLinear
    device = torch.device("npu")
    
    if dtype == torch.bfloat16: atol, rtol = 1e-2, 1e-2 
    elif dtype == torch.float16: atol, rtol = 5e-3, 5e-3
    else: atol, rtol = 1e-5, 1e-5

    golden = torch.nn.Linear(input_size, output_size, bias=True).to(dtype).to(device)
    
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

    part_size = input_size // world_size
    start = local_rank * part_size
    end = (local_rank + 1) * part_size
    
    with torch.no_grad():
        layer.weight.data.copy_(golden.weight.data[:, start:end])
        if local_rank == 0:
            layer.bias.data.copy_(golden.bias.data)
        else:
            layer.bias.data.zero_()

    full_input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    if input_is_parallel:
        layer_input = full_input[..., start:end].detach().clone()
    else:
        layer_input = full_input.detach().clone()

    out, out_bias = layer(layer_input)

    if reduce_results:
        expected_full = golden(full_input) 
        if skip_bias_add:
            expected_matmul_only = torch.matmul(full_input, golden.weight.T)
            assert torch.allclose(out, expected_matmul_only, atol=atol, rtol=rtol)
            if local_rank == 0:
                assert torch.allclose(out_bias, golden.bias, atol=atol, rtol=rtol)
        else:
            assert torch.allclose(out, expected_full, atol=atol, rtol=rtol)
    else:
        weight_shard = golden.weight.data[:, start:end]
        input_shard = full_input[..., start:end]
        expected_partial = torch.matmul(input_shard, weight_shard.T)
        assert torch.allclose(out, expected_partial, atol=atol, rtol=rtol)

def _logic_dp2tp_row_parallel_linear(local_rank, world_size, dtype, batch_size,
                                     q_len, num_heads, v_head_dim, output_size):
    from omni.layers.linear import DP2TPRowParallelLinear

    device = torch.device("npu")
    input_size = num_heads * v_head_dim
    layer = DP2TPRowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        tp_size=world_size,
        tp_rank=local_rank,
        bias=False,
        input_is_parallel=False,
        params_dtype=dtype,
        reduce_results=True,
        quant_config=None,
        prefix="test_dp2tp",
    ).to(device)

    full_weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    layer.weight_loader(layer.weight, full_weight.detach().clone())

    torch.manual_seed(TEST_SEED + local_rank + 1)
    input_tensor = torch.randn(batch_size * q_len, input_size, dtype=dtype, device=device)

    out, out_bias = layer(input_tensor, batch_size, q_len, num_heads, v_head_dim)
    assert out_bias is None

    expected = torch.matmul(input_tensor, full_weight.T)
    atol, rtol = (1e-5, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert out.shape == expected.shape
    assert torch.allclose(out, expected, atol=atol, rtol=rtol)


def _logic_dp2tp_row_parallel_linear_par_no_reduce(local_rank, world_size, dtype,
                                                   batch_size, q_len, num_heads,
                                                   v_head_dim, output_size):
    from omni.layers.linear import DP2TPRowParallelLinear

    device = torch.device("npu")
    input_size = num_heads * v_head_dim
    input_size_per_partition = input_size // world_size
    start = local_rank * input_size_per_partition
    end = (local_rank + 1) * input_size_per_partition

    layer = DP2TPRowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        tp_size=world_size,
        tp_rank=local_rank,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=True,
        params_dtype=dtype,
        reduce_results=False,
        quant_config=None,
        prefix="test_dp2tp_par_no_reduce",
    ).to(device)

    golden_weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    with torch.no_grad():
        layer.weight.data.copy_(golden_weight[:, start:end])
        layer.bias.data.copy_(torch.randn(output_size, dtype=dtype, device=device))

    full_input = torch.randn(batch_size * q_len, input_size, dtype=dtype, device=device)
    shard_input = full_input[..., start:end].contiguous()

    out, out_bias = layer(shard_input, batch_size, q_len, num_heads, v_head_dim)

    expected_partial = torch.matmul(shard_input, golden_weight[:, start:end].T)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out, expected_partial, atol=atol, rtol=rtol)
    assert torch.allclose(out_bias, layer.bias, atol=atol, rtol=rtol)


def _logic_tp2dp_and_tp_row_parallel_linear(local_rank, world_size, dtype,
                                            batch_size, input_size, output_size):
    from omni.layers.linear import Tp2DpAndTpRowParallelLinear

    device = torch.device("npu")
    assert batch_size % world_size == 0

    layer = Tp2DpAndTpRowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        tp_size=world_size,
        tp_rank=local_rank,
        bias=False,
        input_is_parallel=False,
        params_dtype=dtype,
        reduce_results=True,
        quant_config=None,
        prefix="test_tp2dp_tp",
    ).to(device)

    full_weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    layer.weight_loader(layer.weight, full_weight.detach().clone())

    input_tensor = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    out, out_bias = layer(input_tensor)
    assert out_bias is None

    full_expected = torch.matmul(input_tensor, full_weight.T)
    expected_chunks = torch.tensor_split(full_expected, world_size, dim=0)
    expected_local = expected_chunks[local_rank]

    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert out.shape == expected_local.shape
    assert torch.allclose(out, expected_local, atol=atol, rtol=rtol)


def _logic_tp2dp_and_tp_row_parallel_linear_par_no_reduce(local_rank, world_size, dtype,
                                                          batch_size, input_size, output_size):
    from omni.layers.linear import Tp2DpAndTpRowParallelLinear

    device = torch.device("npu")
    input_size_per_partition = input_size // world_size
    start = local_rank * input_size_per_partition
    end = (local_rank + 1) * input_size_per_partition

    layer = Tp2DpAndTpRowParallelLinear(
        input_size=input_size,
        output_size=output_size,
        tp_size=world_size,
        tp_rank=local_rank,
        bias=True,
        input_is_parallel=True,
        skip_bias_add=True,
        params_dtype=dtype,
        reduce_results=False,
        quant_config=None,
        prefix="test_tp2dp_tp_par_no_reduce",
    ).to(device)

    golden_weight = torch.randn(output_size, input_size, dtype=dtype, device=device)
    with torch.no_grad():
        layer.weight.data.copy_(golden_weight[:, start:end])
        layer.bias.data.copy_(torch.randn(output_size, dtype=dtype, device=device))

    full_input = torch.randn(batch_size, input_size, dtype=dtype, device=device)
    shard_input = full_input[..., start:end].contiguous()
    out, out_bias = layer(shard_input)

    expected_partial = torch.matmul(shard_input, golden_weight[:, start:end].T)
    atol, rtol = (1e-6, 1e-5) if dtype == torch.float32 else (1e-3, 1e-3)
    assert torch.allclose(out, expected_partial, atol=atol, rtol=rtol)
    assert torch.allclose(out_bias, layer.bias, atol=atol, rtol=rtol)

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
        from vllm.distributed import tensor_model_parallel_reduce_scatter
        view = hidden_states.reshape(-1, self.hidden_size)
        permute = self.gate_proj.permute(1, 0)
        mm = torch.mm(view, permute)
        reduce_scatter = tensor_model_parallel_reduce_scatter(mm, dim=0)
        return reduce_scatter

# --- Parameterized Tests ---

PARAM_CASES = [
    (torch.float32, 64, 16, 3),
    (torch.bfloat16, 64, 16, 3),
    (torch.float32, 128, 32, 5),
    (torch.float64, 32, 8, 1),
    (torch.bfloat16, 256, 64, 8),
]
PARAM_IDS = [f"{str(d)}_{i}_{o}_b{b}" for d, i, o, b in PARAM_CASES]

@pytest.mark.parametrize("dtype,input_size,output_size,batch", PARAM_CASES, ids=PARAM_IDS)
def test_ascend_row_parallel_linear_basic(dtype, input_size, output_size, batch):
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
    torch.manual_seed(TEST_SEED)
    input_size = 4; output_sizes = [6, 10]; tp_size = 2; batch = 5; dtype = torch.float32
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

# --- Distributed Tests ---

@pytest.mark.parametrize("test_model", [AscendMMRSModel])
@pytest.mark.parametrize("batch_size", [8])
@pytest.mark.parametrize("seq_len", [16])
@pytest.mark.parametrize("hidden_size", [16])
@pytest.mark.parametrize("dtype", [torch.bfloat16])
def test_async_tp_smoke(distributed_worker_pool, test_model, batch_size, seq_len, hidden_size, dtype):
    distributed_worker_pool(
        _logic_tp_smoke, 
        test_model, batch_size, seq_len, hidden_size, dtype
    )

@pytest.mark.parametrize("dtype", [torch.float32])
def test_ascend_merged_column_parallel_linear_gather_output_tp_spawn(distributed_worker_pool, dtype):
    input_size = 8
    output_sizes = [20, 40]
    batch = 3
    torch.manual_seed(TEST_SEED)
    full_weight = torch.randn(sum(output_sizes), input_size, dtype=dtype)
    input_tensor = torch.randn(batch, input_size, dtype=dtype)
    distributed_worker_pool(
        _logic_merged_linear_spawn,
        full_weight, input_tensor, input_size, output_sizes, dtype
    )

@pytest.mark.parametrize("dtype", [torch.bfloat16, torch.float16])
@pytest.mark.parametrize("input_is_parallel", [True, False])
@pytest.mark.parametrize("skip_bias_add", [True, False])
def test_row_parallel_linear_distributed(distributed_worker_pool, dtype, input_is_parallel, skip_bias_add):
    input_size = 32
    output_size = 16
    batch_size = 4
    reduce_results = True
    distributed_worker_pool(
        _logic_row_parallel_distributed,
        input_size, output_size, batch_size, dtype,
        input_is_parallel, skip_bias_add, reduce_results
    )

@pytest.mark.parametrize("dtype", [torch.float32])
def test_row_parallel_no_reduce(distributed_worker_pool, dtype):
    input_size = 32
    output_size = 16
    batch_size = 4
    distributed_worker_pool(
        _logic_row_parallel_distributed,
        input_size, output_size, batch_size, dtype,
        input_is_parallel=True, skip_bias_add=True, reduce_results=False
    )

@pytest.mark.parametrize("dtype", [torch.float32])
def test_dp2tp_row_parallel_linear_distributed(distributed_worker_pool, dtype):
    batch_size = 2
    q_len = 1
    num_heads = 2
    v_head_dim = 4
    output_size = 6
    distributed_worker_pool(
        _logic_dp2tp_row_parallel_linear,
        dtype, batch_size, q_len, num_heads, v_head_dim, output_size
    )


@pytest.mark.parametrize("dtype", [torch.float32])
def test_tp2dp_and_tp_row_parallel_linear_distributed(distributed_worker_pool, dtype):
    batch_size = 4
    input_size = 8
    output_size = 10
    distributed_worker_pool(
        _logic_tp2dp_and_tp_row_parallel_linear,
        dtype, batch_size, input_size, output_size
    )


@pytest.mark.parametrize("dtype", [torch.float32])
def test_dp2tp_row_parallel_linear_par_no_reduce(distributed_worker_pool, dtype):
    batch_size = 2
    q_len = 1
    num_heads = 2
    v_head_dim = 4
    output_size = 6
    distributed_worker_pool(
        _logic_dp2tp_row_parallel_linear_par_no_reduce,
        dtype, batch_size, q_len, num_heads, v_head_dim, output_size
    )


@pytest.mark.parametrize("dtype", [torch.float32])
def test_tp2dp_and_tp_row_parallel_linear_par_no_reduce(distributed_worker_pool, dtype):
    batch_size = 4
    input_size = 8
    output_size = 10
    distributed_worker_pool(
        _logic_tp2dp_and_tp_row_parallel_linear_par_no_reduce,
        dtype, batch_size, input_size, output_size
    )

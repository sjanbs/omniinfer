import importlib.util
import os
import pathlib
import sys
import types
from types import SimpleNamespace
import torch
import pytest
from omni.layers.moe.fused_moe import fused_moe

@pytest.fixture(scope="module")
def fused_module():
    project_root = pathlib.Path(__file__).resolve().parents[3]
    module_path = project_root / "omni" / "layers" / "moe" / "fused_moe" / "fused_moe.py"

    # Mock torch_npu with minimal behaviors
    mock_torch_npu = types.ModuleType("torch_npu")

    def _as_tensor(value, device=None, dtype=None):
        return torch.as_tensor(value, device=device, dtype=dtype)

    class TensorWrapper(list):
        @property
        def device(self):
            return self[0].device

        @property
        def shape(self):
            return self[0].shape

        def to(self, *args, **kwargs):
            return TensorWrapper([self[0].to(*args, **kwargs)])

        def unsqueeze(self, dim):
            return self[0].unsqueeze(dim)

    def npu_moe_gating_top_k_softmax(gating_output, k):
        batch = gating_output.shape[0]
        weights = torch.full((batch, k), 1.0)
        ids = torch.arange(k, dtype=torch.int32).unsqueeze(0).repeat(batch, 1)
        row_idx = torch.arange(batch * k, dtype=torch.int32)
        return weights, ids, row_idx

    def npu_grouped_matmul(x, weight, **kwargs):
        base = x[0][0] if isinstance(x[0], TensorWrapper) else x[0]
        return TensorWrapper([base])

    def npu_swiglu(x):
        return x

    def npu_moe_finalize_routing(out, *args, **kwargs):
        if isinstance(out, (list, tuple)):
            return out[0]
        if isinstance(out, TensorWrapper):
            return out[0]
        return out

    def npu_moe_compute_expert_tokens(expert_idx, n):
        return torch.arange(n, dtype=torch.int32)

    def npu_moe_init_routing(hidden_states, row_idx, expert_idx, active_num):
        return hidden_states, row_idx.view(-1), expert_idx.view(-1)

    def npu_moe_init_routing_v2(hidden_states, expert_idx, scale=None, active_num=None, **kwargs):
        expanded_x_idx = torch.arange(expert_idx.numel(), dtype=torch.int32)
        expert_tokens = torch.ones(expert_idx.numel(), dtype=torch.int32)
        dynamic_scale = torch.ones(expert_idx.numel())
        return hidden_states, expanded_x_idx, expert_tokens, dynamic_scale

    def npu_dequant_swiglu_quant(gate_up_proj, weight_scale, activation_scale, **kwargs):
        return gate_up_proj, activation_scale if torch.is_tensor(activation_scale) else torch.as_tensor(activation_scale)

    def npu_grouped_matmul_swiglu_quant_v2(sorted_tokens, weights, scales, pertoken_scale, expert_tokens):
        return TensorWrapper([sorted_tokens]), pertoken_scale

    def npu_dynamic_quant(x):
        return x, torch.ones(x.shape[0])

    def npu_moe_re_routing(gathered_tokens, tokens_per_expert_group, per_token_scales=None):
        idxs = torch.arange(gathered_tokens.shape[0], dtype=torch.int32)
        tokens_per_local_expert = tokens_per_expert_group.view(-1)
        return gathered_tokens, per_token_scales, idxs, tokens_per_local_expert

    def npu_grouped_matmul_finalize_routing(*args, **kwargs):
        return args[0]

    def npu_moe_distribute_dispatch_v2(**kwargs):
        x = kwargs["x"]
        expand_x = torch.zeros_like(x)
        dynamic_scale = torch.ones(x.shape[0])
        expand_idx = torch.arange(x.shape[0], dtype=torch.int32)
        expert_token_nums = torch.ones(1, dtype=torch.int64)
        ep_recv_counts = torch.ones(1, dtype=torch.int64)
        tp_recv_counts = torch.ones(1, dtype=torch.int64)
        return expand_x, dynamic_scale, expand_idx, expert_token_nums, ep_recv_counts, tp_recv_counts

    def npu_moe_distribute_combine_v2(**kwargs):
        return kwargs["expand_x"]

    def npu_prefetch(*args, **kwargs):
        return None

    mock_torch_npu.npu_moe_gating_top_k_softmax = npu_moe_gating_top_k_softmax
    mock_torch_npu.npu_grouped_matmul = npu_grouped_matmul
    mock_torch_npu.npu_swiglu = npu_swiglu
    mock_torch_npu.npu_moe_finalize_routing = npu_moe_finalize_routing
    mock_torch_npu.npu_moe_compute_expert_tokens = npu_moe_compute_expert_tokens
    mock_torch_npu.npu_moe_init_routing = npu_moe_init_routing
    mock_torch_npu.npu_moe_init_routing_v2 = npu_moe_init_routing_v2
    mock_torch_npu.npu_dequant_swiglu_quant = npu_dequant_swiglu_quant
    mock_torch_npu.npu_grouped_matmul_swiglu_quant_v2 = npu_grouped_matmul_swiglu_quant_v2
    mock_torch_npu.npu_dynamic_quant = npu_dynamic_quant
    mock_torch_npu.npu_moe_re_routing = npu_moe_re_routing
    mock_torch_npu.npu_grouped_matmul_finalize_routing = npu_grouped_matmul_finalize_routing
    mock_torch_npu.npu_moe_distribute_dispatch_v2 = npu_moe_distribute_dispatch_v2
    mock_torch_npu.npu_moe_distribute_combine_v2 = npu_moe_distribute_combine_v2
    mock_torch_npu.npu_prefetch = npu_prefetch

    sys.modules["torch_npu"] = mock_torch_npu

    # Mock vllm platform and distributed helpers
    fake_ep_group = SimpleNamespace(world_size=2, rank_in_group=0, device_group=None)
    fake_world_group = SimpleNamespace(world_size=2, rank_in_group=0)
    fake_dp_group = SimpleNamespace(world_size=1)

    vllm_platforms = types.ModuleType("vllm.platforms")
    vllm_platforms.current_platform = SimpleNamespace(device_type="cpu")

    def get_ep_group():
        return fake_ep_group

    def get_world_group():
        return fake_world_group

    def get_dp_group():
        return fake_dp_group

    vllm_distributed = types.ModuleType("vllm.distributed")
    vllm_distributed.get_ep_group = get_ep_group
    vllm_distributed.get_world_group = get_world_group
    vllm_distributed.get_dp_group = get_dp_group

    sys.modules["vllm.platforms"] = vllm_platforms
    sys.modules["vllm.distributed"] = vllm_distributed
    sys.modules["vllm.forward_context"] = types.ModuleType("vllm.forward_context")
    sys.modules["vllm.forward_context"].get_forward_context = lambda: SimpleNamespace(attn_metadata=None)
    torch.distributed.all_to_all_single = lambda output, input, *args, **kwargs: output.copy_(input)
    original_zeros_like = torch.zeros_like
    original_empty_like = torch.empty_like

    def _zeros_like(inp, *args, **kwargs):
        while isinstance(inp, TensorWrapper):
            inp = inp[0]
        return original_zeros_like(inp, *args, **kwargs)

    def _empty_like(inp, *args, **kwargs):
        while isinstance(inp, TensorWrapper):
            inp = inp[0]
        return original_empty_like(inp, *args, **kwargs)

    torch.zeros_like = _zeros_like
    torch.empty_like = _empty_like

    # Mock config loader
    operator_opt_config = SimpleNamespace(
        moe_multi_stream_tune=False,
        experts_pruning=False,
        enable_gmm_swiglu_quant=False,
        cast_w2_scale_f32=False,
        gmm_nz=False,
        new_w4_op=False,
        attn_prefetch=0,
        prefill_enable_long_seq=False,
        enable_kv_rmsnorm_rope_cache=False,
        shared_experts_to_gmm=False,
    )
    parall_config = SimpleNamespace(redundancy_shared_expert_num=0, attn_dies=0, o_proj_tp_size=1)
    task_config = SimpleNamespace(enable_omni_placement=False, enable_attn_ffn_disaggregation=False,
                                  decode_gear_list=[16])
    model_extra = SimpleNamespace(operator_opt_config=operator_opt_config, parall_config=parall_config,
                                  task_config=task_config)
    sys.modules["omni.models.config_loader.loader"] = types.ModuleType("loader")
    sys.modules["omni.models.config_loader.loader"].model_extra_config = model_extra
    os.environ.setdefault("ROLE", "prefill")

    # Mock ConditionalTNGScope
    sys.modules["omni.layers.utils"] = types.ModuleType("utils")
    class ConditionalTNGScope:
        def __init__(self, *args, **kwargs):
            pass
        def __enter__(self):
            return self
        def __exit__(self, exc_type, exc, tb):
            return False
    sys.modules["omni.layers.utils"].ConditionalTNGScope = ConditionalTNGScope

    # Mock torchair
    tng_scope = types.SimpleNamespace(npu_stream_switch=lambda *args, **kwargs: ConditionalTNGScope())
    sys.modules["torchair"] = types.ModuleType("torchair")
    sys.modules["torchair"].scope = tng_scope

    # Import module
    spec = importlib.util.spec_from_file_location("fused_moe", module_path)
    fused_moe = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(fused_moe)  # type: ignore[arg-type]

    return fused_moe


def test_fused_topk(fused_module):
    gating = torch.randn(2, 4)
    weights, ids, rows = fused_module.fused_topk(gating, topk=2, renormalize=True)
    assert weights.shape == (2, 2)
    assert ids.shape == (2, 2)
    assert rows.numel() == 4


def test_grouped_topk_softmax(fused_module):
    hidden = torch.randn(3, 2)
    gating = torch.tensor([[1.0, 0.0, 0.5, -0.5, 2.0, 1.0]])
    weights, ids, rows = fused_module.grouped_topk(hidden, gating, topk=2, renormalize=True,
                                                   num_expert_group=2, topk_group=1)
    assert weights.shape[1] == 2
    assert ids.shape == weights.shape
    assert rows.numel() == ids.numel()


def test_fused_experts_allgather_ep_warmup(fused_module):
    hidden_states = torch.ones(2, 3)
    w1 = torch.ones(1, 3, 3)
    w2 = torch.ones(1, 3, 3)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    row_idx = torch.zeros(2, 1, dtype=torch.int32)
    output = fused_module.fused_experts_allgather_ep(hidden_states, w1, w2, topk_weights, topk_ids, row_idx,
                                                     warm_up=True, n_routed_experts=1, local_expert_indices=[0])
    assert output.shape == hidden_states.shape


def test_fused_experts_alltoall_ep_warmup(fused_module, monkeypatch):
    hidden_states = torch.ones(2, 2)
    w1 = torch.ones(1, 2, 2)
    w2 = torch.ones(1, 2, 2)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    row_idx = torch.zeros(2, 1, dtype=torch.int32)

    def fake_all_to_all_single(output, input, *args, **kwargs):
        if hasattr(input, "__iter__") and not isinstance(input, torch.Tensor):
            input = list(input)[0]
        output.copy_(input)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)

    output = fused_module.fused_experts_alltoall_ep(hidden_states, w1, w2, topk_weights, topk_ids, row_idx,
                                                    warm_up=True)
    assert output.shape == hidden_states.shape


def test_fused_experts_ep_best_alltoall(fused_module, monkeypatch):
    hidden_states = torch.ones(2, 2)
    w1 = torch.ones(1, 2, 2)
    w2 = torch.ones(1, 2, 2)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    row_idx = torch.zeros(2, 1, dtype=torch.int32)

    original_grouped_matmul = fused_module.torch_npu.npu_grouped_matmul
    monkeypatch.setattr(fused_module.torch_npu, "npu_grouped_matmul", lambda x, weight, **kwargs: x[0])
    def fake_all_to_all_single(output, input, *args, **kwargs):
        if hasattr(input, "__iter__") and not isinstance(input, torch.Tensor):
            input = list(input)[0]
        output.copy_(input)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)

    output = fused_module.fused_experts_ep_best_alltoall(hidden_states, w1, w2, topk_weights, topk_ids, row_idx)
    monkeypatch.setattr(fused_module.torch_npu, "npu_grouped_matmul", original_grouped_matmul)
    assert output.shape == hidden_states.shape


def test_fused_experts_allgather_ep_a3_prefill(fused_module):
    class Layer:
        def __init__(self):
            self.weight_num_bits = 8
            self.w13_weight = torch.ones(1, 3, 3)
            self.w13_weight_scale = torch.ones(1, 3)
            self.w2_weight = torch.ones(1, 3, 3)
            self.w2_weight_scale = torch.ones(1, 3)
            self.moe_layer_idx = 0
            self.w2_weight_scale = torch.ones(1, 3)
            self.w13_weight_scale = torch.ones(1, 3)
            self.w2_weight = torch.ones(1, 3, 3)
            self.w2_weight_scale = torch.ones(1, 3)
            self.w13_weight = torch.ones(1, 3, 3)
            self.w13_weight_scale = torch.ones(1, 3)
            self.w2_weight = torch.ones(1, 3, 3)
            self.w2_weight_scale = torch.ones(1, 3)
            self.planner = SimpleNamespace(record_activation=lambda *args, **kwargs: None)
    layer = Layer()
    hidden_states = torch.ones(2, 3)
    pertoken_scale = torch.ones(2)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    output = fused_module.fused_experts_allgather_ep_a3(layer, hidden_states, pertoken_scale, topk_weights,
                                                        topk_ids, n_routed_experts=1, is_prefill=True,
                                                        max_num_deployed_expert_per_rank=1)
    assert output.shape[0] == hidden_states.shape[0]


def test_gmm_expert_weight8(fused_module):
    class Layer:
        def __init__(self):
            self.weight_num_bits = 8
            self.w13_weight = torch.ones(1, 2, 2)
            self.w13_weight_scale = torch.ones(1, 2)
            self.w2_weight = torch.ones(1, 2, 2)
            self.w2_weight_scale = torch.ones(1, 2)
    layer = Layer()
    x = torch.ones(1, 2)
    expert_tokens = torch.ones(1, dtype=torch.int64)
    pertoken_scale = torch.ones(1)
    out = fused_module.gmm_expert(layer, x, expert_tokens, dynamic_scale=pertoken_scale)
    assert out.shape[0] == x.shape[0]


def test_moe_infer_fusion(fused_module):
    class Layer:
        def __init__(self):
            self.w13_weight = torch.ones(1, 2, 2)
            self.w2_weight = torch.ones(1, 2, 2)
            self.weight_num_bits = 8
            self.w13_weight_scale = torch.ones(1, 2)
            self.w2_weight_scale = torch.ones(1, 2)
            self.planner = SimpleNamespace(record_activation=lambda *args, **kwargs: None)
    layer = Layer()
    x = torch.ones(2, 2)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    topk_weight = torch.ones(2, 1)
    hidden_states, gathered, weights, row_idx = fused_module.moe_infer_fusion(layer, x, topk_ids, topk_weight,
                                                                             warm_up=False, is_prefill=True,
                                                                             comm_group=None)
    assert hidden_states.shape == x.shape
    assert gathered.shape[1] == x.shape[1]


def test_shared_expert_quant_forward(fused_module):
    class Gate:
        def __init__(self):
            self.weight = torch.ones(1, 2, 2)
            self.weight_scale = torch.ones(2)
    class Down:
        def __init__(self):
            self.weight = torch.ones(1, 2, 2)
            self.weight_scale = torch.ones(2)
    class Layer:
        def __init__(self):
            self.gate_up_proj = Gate()
            self.down_proj = Down()
    layer = Layer()
    sorted_tokens = torch.ones(1, 2)
    expert_tokens = torch.ones(1, dtype=torch.int64)
    out = fused_module.shared_expert_quant_forward(layer, sorted_tokens, expert_tokens, torch.bfloat16, dynamic_scale=torch.ones(1))
    assert out.shape[0] == sorted_tokens.shape[0]


def test_moe_expert_quant_forward_weight8(fused_module):
    class Layer:
        def __init__(self):
            self.quant_mode = True
            self.weight_num_bits = 8
            self.w13_weight = torch.ones(1, 2, 2)
            self.w13_weight_scale = torch.ones(1, 2)
            self.w2_weight = torch.ones(1, 2, 2)
            self.w2_weight_scale = torch.ones(1, 2)
    layer = Layer()
    sorted_tokens = torch.ones(1, 2)
    expert_tokens = torch.ones(1, dtype=torch.int64)
    out = fused_module.moe_expert_quant_forward(layer, sorted_tokens, expert_tokens, torch.bfloat16, dynamic_scale=torch.ones(1))
    assert out.shape[0] == sorted_tokens.shape[0]


def test_set_fake_expand_x_and_speculative(fused_module):
    fused_module.fake_expand_x.clear()
    fused_module.set_num_speculative_tokens(1)
    fused_module.set_fake_expand_x(2, [2, 2])
    assert 2 in fused_module.fake_expand_x
    fused_module.set_fake_expand_x(4, [4, 2])
    assert 4 in fused_module.fake_expand_x


def test_fused_experts_moe_dispatch_combine(fused_module, monkeypatch):
    class Layer:
        def __init__(self):
            self.tp_size = 1
            self.quant_mode = True
            self.moe_all_to_all_group_name = "g1"
            self.moe_rs_group_name = "g2"
            self.w13_weight = torch.ones(1, 2, 2)
            self.w13_weight_scale = torch.ones(1, 2)
            self.w2_weight = torch.ones(1, 2, 2)
            self.w2_weight_scale = torch.ones(1, 2)
            self.moe_layer_idx = 0
            self.planner = SimpleNamespace(record_activation=lambda *args, **kwargs: None)
    layer = Layer()
    hidden_states = torch.ones(2, 2)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)

    monkeypatch.setattr(fused_module, "moe_expert_quant_forward", lambda *args, **kwargs: args[1])

    output = fused_module.fused_experts_moe_dispatch_combine(layer, hidden_states, topk_weights, topk_ids,
                                                             max_num_deployed_expert=2, is_prefill=True,
                                                             is_route_expert=True)
    assert output.shape == hidden_states.shape


def test_static_routing(fused_module):
    hidden_states = torch.randn(4, 2)
    indices = fused_module.static_routing(hidden_states)
    assert len(indices) == hidden_states.shape[0]


def test_shared_expert_alltoall_ep(fused_module, monkeypatch):
    def fake_all_to_all_single(output, input, **kwargs):
        output.copy_(input)
    monkeypatch.setattr(torch.distributed, "all_to_all_single", fake_all_to_all_single)

    hidden_states = torch.ones(2, 2)
    expert = torch.nn.Linear(2, 2, bias=False)
    torch.nn.init.constant_(expert.weight, 1.0)
    output = fused_module.shared_expert_alltoall_ep(hidden_states, expert, warm_up=False)
    assert output.shape == hidden_states.shape


def test_fused_experts_allgather_ep_a2(fused_module):
    class Layer:
        def __init__(self):
            self.weight_num_bits = 8
            self.w13_weight = torch.ones(1, 2, 2)
            self.w13_weight_scale = torch.ones(1, 2)
            self.w2_weight = torch.ones(1, 2, 2)
            self.w2_weight_scale = torch.ones(1, 2)
            self.moe_layer_idx = 0
            self.planner = SimpleNamespace(record_activation=lambda *args, **kwargs: None)
    layer = Layer()
    hidden_states = torch.ones(2, 2)
    pertoken_scale = torch.ones(2)
    topk_weights = torch.ones(2, 1)
    topk_ids = torch.zeros(2, 1, dtype=torch.int32)
    smooth_scale = torch.ones(1, 1)
    output = fused_module.fused_experts_allgather_ep_a2(layer, hidden_states, pertoken_scale, topk_weights,
                                                        topk_ids, n_routed_experts=1, is_prefill=True,
                                                        max_num_deployed_expert_per_rank=1, smooth_scale=smooth_scale)
    assert output.shape[0] == hidden_states.shape[0]
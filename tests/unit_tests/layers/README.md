> 参考路径：`tests/unit_tests/layers/` 下现有的 UT 代码，如 `test_mla.py`、`test_fused_moe.py`、`test_linear.py` 等。

## 1. 选一个最接近的示例
- **算子/模块类型**：注意选择与目标模块相同的模式。例如：
  - 解码路径、前填充路径和权重预处理相关测试可参考 `test_mla.py`。
  - 专家路由、分块 GEMM 相关测试可参考 `test_fused_moe.py`。
  - 线性层、Embedding、LayerNorm 则可参考 `test_linear.py`、`test_vocab_embeding.py`、`test_layernorm.py`。

## 2. 复用固定装置（fixtures）
- **分布式池**：需要 NPU/HCCL 的分布式测试时直接使用 `distributed_worker_pool` fixture，它会自动在模块级启动持久化工作进程。
- **常见张量生成**：`distributed_test_common.py` 提供的辅助函数或 pytest 的参数化模式可以减少样板代码。

## 3. 按照“驱动 + 逻辑”拆分
- **逻辑函数**：放在文件顶层，负责在设备上运行真实算子逻辑；记得在函数内部导入 `omni.layers` 并重置随机种子。
- **测试驱动**：pytest 函数只做参数构造、mock/patch、调用逻辑函数（或分布式池）。
- 在 `test_mla.py` 中，`_run_mla_decode_logic`、`_run_mla_prefill_logic` 这类函数就是逻辑层；对应的 `test_forward_decode_graph_and_prolog_init`、`test_forward_prefill` 等是驱动层，便于对输入输出形状、调用次数、参数传递进行断言。

## 4. 善用 Mock 断言交互
- 对不需要真实计算的模块使用 `unittest.mock`：
  - 在 `test_fused_moe.py` 中，使用 `mock.patch.object` 检查专家路由的 EP 组是否按预期调用。
  - 在 `test_mla.py` 中，通过 `patch` 替换 `parallel_gather`、`mla_prolog`，只校验张量尺寸和调用次数，而不依赖底层实现。
- Mock 时务必保留关键输入输出的形状断言，以确保接口契约稳定。

## 5. 关注张量形状与参数
- 绝大多数 UT 通过构造最小化张量来验证：
  - 形状变换是否正确（如解码/前填充的 `[batch, seq, num_heads, head_dim]`）。
  - 参数下传是否完整（例如 `kv_scale_mode`、`expert_layer_map`）。
- 结合 `torch.randn` 的小尺寸张量或 `torch.arange` 的可预测数据，便于在断言中直接比较。

## 6. 提供图模式与非图模式覆盖
- 对同时支持 Graph Mode 的模块，参考 `test_forward_decode_graph_and_prolog_init` 中的处理方式：
  - 使用 `mock.patch` 捕获 `init_graph_mode`、`build_graph` 的调用。
  - 为图模式和非图模式各写一条参数化用例或分支，确保两条路径都有覆盖。

## 7. 运行与跳过策略
- 本仓库的 CI 可能缺少 `vllm`、NPU 驱动等依赖。可以在本地具备依赖的环境执行：
  - `python -m pytest tests/unit_tests/layers -k '<your_test_name>'`
- 如果需要在无依赖环境下确保用例至少能被收集，可：
  - 使用 Mock 替换外部依赖。
  - 在开头添加条件跳过（`pytest.importorskip("vllm")`）。

## 8. Review 自查清单
- [ ] 逻辑函数是否在顶层定义并重置了随机种子？
- [ ] 是否在逻辑函数内部导入了目标模块？
- [ ] 是否覆盖了图模式/非图模式、解码/前填充等关键分支？
- [ ] Mock 是否校验了形状、调用次数、关键参数？
- [ ] 测试名称是否清晰表达覆盖场景？

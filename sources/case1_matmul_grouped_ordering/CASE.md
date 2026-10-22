# 案例 1：GEMM 的 program-id 重映射（grouped ordering / L2 swizzle）

| 维度 | 内容 |
|---|---|
| 算子类型 | 稠密矩阵乘 C = A·B（tiled GEMM，fp32 输入，fp32 累加） |
| 来源 | Triton 官方教程 `03-matrix-multiplication`（"L2 Cache Optimizations" 一节），Triton 内置 matmul、`torch.compile` 的 Inductor GEMM 模板、FlashAttention-2 的 head/seq 调度都使用同类重映射 |
| 优化方式 | 只改变 `pid → (pid_m, pid_n)` 的整数映射：行主序遍历改为 `GROUP_M` 行一组的"分组列主序"，使同时驻留的 program 复用 L2 中的 A/B 行块。tile 内计算、K 循环、mask、store 完全不变 |
| 优化方的等价性依据 | "每个 tile 都恰好被某个 program 算一次，算法不变"——即新映射是网格上的**双射**，且 tile 体逐字相同。教程未做任何浮点论证，因为不需要 |
| 等价性强度 | **逐位相等**（bitwise）。两侧对同一 tile 执行完全相同的指令序列（同一 TTGIR、同一 reduction 树、同一 MMA 顺序），所以差异仅在"哪个 program 算哪个 tile"，输出 bit 不变 |

## 1. 前后代码差异

`before.py`：

```python
pid_m = pid // num_pid_n
pid_n = pid % num_pid_n
```

`after.py`：

```python
num_pid_in_group = GROUP_M * num_pid_n
group_id        = pid // num_pid_in_group
first_pid_m     = group_id * GROUP_M
group_size_m    = min(num_pid_m - first_pid_m, GROUP_M)   # 最后一组可能不满
pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
pid_n = (pid % num_pid_in_group) // group_size_m
```

对应 TTIR（`ir/after_matmul_grouped.ttir.mlir`）新增的全部是标量 `arith.divsi / remsi / minsi / muli / addi / subi`；
`diff ir/before_*.ttir.mlir ir/after_*.ttir.mlir` 除 SSA 重命名外，只有这一段标量整数子图不同，
`scf.for` 循环体、`tt.dot ... inputPrecision = tf32`、`tt.load/tt.store` 完全一致。

## 2. 等价性建模的影响因素

1. **等价性是网格级的，不是 program 级的。** 对同一 `pid`，before 和 after 计算的是不同 tile，逐 program 比较必然失败。正确的陈述是：
   `⋃_{pid} effect_before(pid) = ⋃_{pid} effect_after(pid)`，其中 `effect(pid)` 是该 program 对全局内存的写集合 `{(addr, value)}`。
2. **需要证明的性质全是整数性质：**
   - 值域：`0 ≤ pid_m < num_pid_m ∧ 0 ≤ pid_n < num_pid_n`；
   - 单射：`pid ≠ pid' ⇒ (pid_m, pid_n) ≠ (pid_m', pid_n')`；
   - 定义域与值域基数相等（`grid = num_pid_m * num_pid_n`）⇒ 双射。
   `min()` 处理的尾组是唯一的非平凡点。
3. **写集合两两不相交**（不同 tile 写不同的 C 区域，由 `c_mask` 和 tile 不重叠保证），因此 program 的执行顺序无关，不需要原子操作或内存模型推理。
4. **整数溢出是建模的一部分。** Triton 索引默认 i32，`pid * BLOCK_M`、`offs_am[:, None] * stride_am` 在超大矩阵下可能溢出；证明必须在定宽二进制补码算术下做（见 `scripts/verify_case1_bijection.py`），或把"无溢出"写成前置条件。
5. **`% M` / `% N` 的回绕读取**（`offs_am = (...) % M`）使越界行读到合法但"错误"的数据，再被 `c_mask` 丢弃；这意味着**读集合**不等价（after 和 before 读同样的地址集合，但若比较"读了什么"会误报），spec 只能约束写集合。
6. **自动调优让问题更大：** 若 before/after 还选了不同 `BLOCK_*` / `num_warps`，则 tile 体不同，逐位相等立即失效，退化到案例 0 §4 所述"模 reduction 重结合相等"。

## 3. 优化方的判定逻辑（关注点）

- 是否忽略浮点精度：**不涉及**，因为 tile 体相同，不需要任何浮点假设。这是五个案例里唯一"完全不需要讨论浮点"的案例。
- 三角函数等：不涉及。
- 判定方式：代码评审 + 性能测试；教程中正确性测试用 `torch.allclose(..., atol=1e-2, rtol=0)` 与 cuBLAS 比较——这个容差是为 TF32 / fp16 留的，与本优化无关。

## 4. 建议的验证范围与层级

| 项 | 建议 |
|---|---|
| 层级 | **TTIR**（甚至 TTIR 中的标量整数子图即可）。TTGIR 的 layout / 流水线对本案例无影响，因为两侧 TTGIR 的 tile 体相同 |
| 范围 | 网格级双射 + tile 体结构等价（alpha 等价）+ 写集合不相交 |
| 证明工具 | SMT（bit-vector 理论）证双射：`scripts/verify_case1_bijection.py` 在 16 位模型、`num_pid_m, num_pid_n ≤ 64, GROUP_M ≤ 8` 下 Z3 约 90 s 证完四条性质；32 位模型未在 10 min 内结束，说明除法 bit-blasting 是瓶颈，可改为整数理论 + 无溢出前置条件 |
| spec 形式 | 关系式：`∀ tile (i, j) ∈ [0, num_pid_m) × [0, num_pid_n). ∃! pid. map_after(pid) = (i, j)`；再加 `body_before ≡_α body_after`。不需要输出值的数值 spec |

## 5. 证据

- `ir/SUMMARY.txt`：两侧 TTIR 均为 2 load / 1 store / 1 dot / 1 for；TTGIR 均含 `blocked, dot_op, nvidia_mma, slice, swizzled_shared` 五种 layout、5 个 `convert_layout`、6 个 async copy（`num_stages=3` 流水线）。
- `cases/NUMERIC_RESULTS.md`：M,N,K = 100,72,50（非块大小整数倍）、`GROUP_M=3`（不整除 `num_pid_m=4`）下，解释器输出 **bitwise 相等**。
- `scripts/verify_case1_bijection.py`：Z3 证明输出见上。

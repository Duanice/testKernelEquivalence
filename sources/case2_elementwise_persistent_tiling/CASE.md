# 案例 2：逐元素算子的分块改变 + persistent kernel

| 维度 | 内容 |
|---|---|
| 算子类型 | 逐元素（elementwise）融合算子 `out = relu(x + y)`，代表所有 pointwise / 激活 / bias-add 类算子 |
| 来源 | Triton 教程 `01-vector-add`（before）；persistent kernel 写法见教程 `09-persistent-matmul`、Triton 内置 `tl.range`+`tl.num_programs` 惯用法、Inductor 的 persistent reduction 模板（after） |
| 优化方式 | (a) tile 大小 1024 → 2048（每线程 8 元素，利于 128-bit 向量化访存）；(b) 网格从 `cdiv(n, BLOCK)` 个 program 改为固定少量（≈SM 数）program，每个 program 以 grid-stride 循环处理多个 tile |
| 优化方的等价性依据 | "每个元素被恰好处理一次，且处理函数逐元素相同"。所有自动调优框架（`@triton.autotune` 搜 `BLOCK`、Inductor 的 config 搜索）都默认接受这一点 |
| 等价性强度 | **逐位相等**：逐元素运算没有跨元素的浮点依赖，分块方式不影响每个输出 bit |

## 1. 前后代码差异

- before：`offsets = pid * BLOCK + arange(BLOCK)`，单 tile。
- after：`for tile in range(pid, num_tiles, num_programs): offsets = tile * BLOCK + arange(BLOCK)`，多 tile；TTIR 中多出一个 `scf.for`，循环上界依赖运行时的 `tt.get_num_programs`。
- 两侧的逐元素运算子图一致：`arith.addf` + `arith.maxnumf`（`tl.maximum` 的 NaN 语义见下）。

## 2. 等价性建模的影响因素

1. **不能直接按相同 program ID 逐一比较。** 例如，实际启动 8 个 program 时，before 的 program 0 写入 `[0,1024)`，after 的 program 0 写入 `[0,2048) ∪ [16384,18432) ∪ …`。两者承担的工作范围不同，因此应比较 **整个 grid**最终写入的地址和值，而不能要求相同编号的 program 行为一致。
2. 程序会写哪些位置，取决于输入大小和启动配置；只看 kernel 内部代码还不够。
**写集合是运行时参数的函数，依赖这些参数变化。**
 `grid` 由 host 端 `launch()` 计算（`min(num_programs, cdiv(n, BLOCK))`），**不在 kernel IR 里**。所以 "两个 Triton 算子" 的表示必须包含 launch 侧：`(kernel, grid 公式, constexpr 取值, 参数前置条件)`，否则 spec 不完整。
3. **覆盖性证明是整数区间推理：** `⋃_tile [tile·BLOCK, (tile+1)·BLOCK) ∩ [0, n) = [0, n)` 且 `tile = pid + k·num_programs` 遍历 `[0, num_tiles)` 恰好一次。前者是 case 1 的双射问题换了形态。
4. **mask 语义：** `tl.load(..., mask)` 未给 `other` 时被 mask 的元素值未定义（TTIR `tt.load %ptr, %mask` 无第三操作数），但同一 mask 下的 `tt.store` 不写回这些元素，所以未定义值不逃逸——spec 要能表达"仅比较被写的地址"。
5. **i32 溢出：** `tile * BLOCK` 在 `n ≥ 2^31` 时溢出；before 中 `pid * BLOCK` 同样会。两侧溢出行为不同（after 的 tile 索引范围相同但循环变量以 i32 累加），需要显式前置条件 `n < 2^31`。
6. **NaN 语义不是"显然"的：** `tl.maximum(x + y, 0.0)` 在 TTIR 里是 `arith.maxnumf`（NaN 被另一操作数替换），而 `tl.where(v > 0, v, 0)` 会把 NaN 变成 0——这两种"relu"在真实数上相同、在 IEEE 上不同。若 after 顺手改写了这一句，逐位相等立即失效。本案例保持同一写法。

## 3. 优化方的判定逻辑

- 是否忽略浮点精度：不需要——没有跨元素累加。但优化方通常**没有意识到**这是为什么安全；一旦同样的分块改变用在含 `tl.sum` 的 kernel 上，相同的思路会给出不同 bit（案例 0 §4）。
- 容差测试：教程用 `torch.max(torch.abs(out_torch - out_triton))` 应为 0，本案例确实为 0。

## 4. 建议的验证范围与层级

| 项 | 建议 |
|---|---|
| 层级 | **TTIR + launch 侧元信息**。TTGIR 只改变 `sizePerThread`/`warpsPerCTA`，对逐元素结果无影响 |
| 范围 | 网格级写集合相等；每个被写地址对应的值由同一逐元素函数计算（可用 SMT 对标量函数证明等价，或直接做 TTIR 子图同构） |
| 证明策略 | 把 kernel 抽象为 `addr ↦ value` 的部分函数，分解为 "索引函数等价（整数）" + "值函数等价（标量浮点，这里是语法相同）" |
| spec 形式 | `∀ i ∈ [0, n). out[i] = relu(x[i] + y[i])`，其中 relu 以 `maxnumf` 语义定义；前置条件：`x, y, out` 互不重叠、`n < 2^31` |

## 5. 证据

- `ir/SUMMARY.txt`：before 0 个 `scf.for`，after 1 个；两侧均 1 个 fp 运算，layout 只有 `blocked`。
- `cases/NUMERIC_RESULTS.md`：`n = 10007`（非 2 的幂，触发 mask 尾块），before `BLOCK=1024`（10 个 program）vs after `BLOCK=2048`、3 个 persistent program，输出 **bitwise 相等**。

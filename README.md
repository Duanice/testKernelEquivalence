# testKernelEquivalence

一个可以运行的 **Triton → Triton 数学等价性最小原型**，聚焦「张量表达式 → 自动标量化 → Z3」。

**已跑通：两组正确模型返回 UNSAT；两个故意改错的模型返回 SAT，且反例经独立的 Fraction 精确计算回放成立。**
实际耗时、反例位置与输出值见 [结果报告](results/REPORT.md)，完整输入见 [results.json](results/results.json)。

输入案例使用用户本地案例集的 case1 与 case2，当前工作树文件逐字复制到 [sources/](sources/)。
这次没有实现源码/TTIR 解析器：**从 Triton 到张量表达式由人工完成；从张量表达式到标量公式和 Z3 查询由程序完成。**
结论依赖人工模型忠实于源码，并不是自动认证任意两个 Triton 源文件。

## 运行

只需要 Python（建议 3.10+）和 Z3，不需要 GPU、Triton、PyTorch 或 NumPy。

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python verify.py
.venv/bin/python -m unittest -v
```

运行会生成/更新 `results/`。也可指定新目录：

```bash
.venv/bin/python verify.py --out-dir work/rerun --timeout-ms 10000
```

四个查询中任何一个未取得预期结果、SAT 的反例未能精确回放、源码快照被修改，脚本都会失败。
`UNKNOWN`/超时不是证明，也不是通过。不支持的 IR 操作直接报错，不自动替换成假设。
原始 Triton 文件仅作为人工模型的可审阅输入保存，核心验证脚本不会导入它们。

## 选取了哪些程序

| 查询 | 原始案例 / 改动 | 固定实例 | 预期 |
|---|---|---|---|
| `case2_equivalent` | 普通 tiled `relu(x+y)` vs persistent grid-stride kernel | n=77；before BLOCK=8；after BLOCK=16、2 个 program | UNSAT |
| `case2_wrong_relu` | 同上，但 after 的 maximum 故意改成 minimum | 同上 | SAT |
| `case1_equivalent` | row-major GEMM vs grouped GEMM；另选不同 K 分块大小 | M=33、N=17、K=19；BM=BN=16；BK 从 16 改为 32；GROUP_M=2 | UNSAT |
| `case1_missing_term` | 同上，但 after 的 A 读取 mask 从 K 改成 K-1，漏掉最后一项 | 同上 | SAT |

原案例通过 `launch()` 参数允许这些配置；它们不是原文件 `KERNELS` 编译清单里的默认配置。
本次没有编译或执行任何 Triton kernel，也没有实测上述配置在某块 GPU 上的兼容性。

- case2 触发尾块 mask 和 persistent 的多次迭代。
- case1 触发 M/N/K 三个维度的尾块、最后一个不满的 group，以及不同的归约分组。
- 两个负例都有对应的 [Triton 源码副本](mutants/)，与模型中的局部变更一致。**它们是故意注入的错误，不是发现原案例有 bug。**

## 核心机制

```text
sources/.../before.py / after.py
     │ 人工建立各自的索引、tile、mask、算子图
     ▼
case_models.py → Tensor + Store + Kernel
     │ tensor_model.py: Scalarizer.at / Kernel.output
     ▼
输出位置的标量公式 before(i, j)、after(i, j)
     │ verify.py: 有效输出位置 ∧ before != after
     ▼
Z3 → UNSAT / SAT / UNKNOWN
     │ SAT 时提取完整输入与输出坐标
     ▼
replay.py → 独立的 Fraction CPU 算法回放
```

1. **输入共享。** 每个输入 buffer 对应一个 `Int → Real` 未解释函数。before 与 after 引用相同函数；相同地址读取到同一值。输入值没有被固定成随机样本。
2. **算子有语义。** 加乘生成 Z3 实数算术；min/max 生成 `If`；`dot` 沿 tile 的 K 轴完整展开成乘积之和。mask 外明确 `other=0` 的读取返回零；未指定 `other` 的读取保留为未定义符号。
3. **以最终输出坐标对齐。** `Kernel.output` 根据输出坐标选取写它的 tile，再以 tile 内坐标调用标量化器。两侧不要求同一个 program id 计算同一块。
4. **检查写覆盖。** 对固定 launch 的所有整数位置枚举，要求每个合法输出恰好被写一次。遗漏或重复写直接拒绝，不允许两个空程序得到虚假的等价结论。此检查不是任意形状下的符号双射证明，也不是通用内存安全检查。
5. **搜索反例。** 求解 `0 <= i < M ∧ 0 <= j < N ∧ before(i,j) != after(i,j)`。UNSAT 意味着这个固定实例的实数模型在所有输入取值上等价；SAT 给出反例。

具体公式和源码对应关系见 [标量化过程](docs/scalarization.md)。

## 如何查看证据

| 文件 | 用途 |
|---|---|
| [tensor_model.py](tensor_model.py) | 小型张量 IR、自动标量化、输出坐标对齐、写覆盖检查 |
| [case_models.py](case_models.py) | 人工提取的 before/after 模型；需要重点审阅的可信前端 |
| [verify.py](verify.py) | 实际查询、状态处理、反例提取及报告生成 |
| [replay.py](replay.py) | 不使用 IR/Z3 公式的独立精确计算 |
| [test_verifier.py](test_verifier.py) | 四个求解结果、导出查询重跑、独立数值校验、覆盖拒绝、UNKNOWN 分支 |
| `results/*.formulas.txt` | 标量化器实际生成的两侧公式，未手写最终答案 |
| `results/*.smt2` | 包含声明、断言、check-sat 的自包含 SMT-LIB 查询 |
| `results/*.model.txt`、`results/results.json` | SAT 模型、全部输入、精确反例输出、版本与代码哈希 |
| [sources/manifest.json](sources/manifest.json) | 来源 HEAD、当前工作树快照哈希、负例生成说明 |

如果环境中有 Z3 命令行，可独立运行：

```bash
z3 results/case1_equivalent.smt2
z3 results/case1_missing_term.smt2
```

没有 CLI 时，单元测试也会重新解析并求解全部 `.smt2`，不会依赖原来的 Python 表达式对象。
测试中的 UNKNOWN 使用受控注入，验证状态处理；不宣称在本案例上观测到了求解器的能力边界。

## 结论适用范围

这次验证成功支持的结论是：**对这两组人工提取的、包含分块与 mask 的张量计算模型，可以自动标量化，并让 Z3 证明实数等价或找到可回放的结构性错误。**

- 固定形状、固定 grid/constexpr、连续且互不重叠的输入/输出，索引无整数溢出。
- K=19 的所有真实归约项都保留；扩展到 tile 的 32 项通过 mask 补零。**没有把长归约截断为前几项，也没有证明任意 K。**
- `tl.dot` 按精确实数乘加解释，dtype 转换视为恒等；忽略 TF32、FMA 舍入、FP32 舍入、NaN/Inf、溢出、signed zero。
- 不验证编译器 lowering、GPU 线程同步/竞态、别名、任意 stride、硬件执行或浮点误差界。Fraction 回放验证数学反例，不能替代 GPU 执行。
- 原 cases 的原有数值检查/编译证据没有在本项目重跑，也没有当作本次证明依据。
- UNSAT 是 Z3 的求解结果，未另外导出可独立检查的证明证书。
- 快照哈希能发现源码漂移，但不能证明人工提取正确。修改 kernel 后必须审查并更新对应模型。

## 参考

- [Gimlet: Formally Verifying AI-Generated GPU Kernels](https://gimletlabs.ai/blog/formally-verifying-ai-generated-kernels)：借鉴标量化与 SMT 验证路线，本仓库是独立最小实现。
- [Triton matrix multiplication tutorial](https://triton-lang.org/main/getting-started/tutorials/03-matrix-multiplication.html)：case1 的原始算法背景。
- [Z3 arithmetic](https://microsoft.github.io/z3guide/docs/theories/Arithmetic/) 与 [uninterpreted functions](https://microsoft.github.io/z3guide/docs/logic/Uninterpreted-functions-and-constants/)：实数语义及符号输入。

# 实际验证结果

Python 3.14.6；Z3 4.16.0；每次查询限时 10000 ms。

以下结果属于人工提取的张量模型：固定形状、实数语义、完整归约展开。未执行 Triton/GPU。

| 查询 | 实际返回 | 求解耗时 / s | 反例精确回放 |
|---|---|---:|---|
| case2_equivalent | UNSAT | 0.096212 | — |
| case2_wrong_relu | SAT | 0.086722 | True |
| case1_equivalent | UNSAT | 0.000090 | — |
| case1_missing_term | SAT | 0.002093 | True |

**case2_wrong_relu**：输出位置 `(0,)`，before = `1/2`，after = `0`。
完整输入在 [results.json](results.json)，求解器模型在 [case2_wrong_relu.model.txt](case2_wrong_relu.model.txt)。

**case1_missing_term**：输出位置 `(32, 16)`，before = `114`，after = `108`。
完整输入在 [results.json](results.json)，求解器模型在 [case1_missing_term.model.txt](case1_missing_term.model.txt)。

耗时是本次机器上的观测值，不是性能基准；UNSAT 是求解结果，没有导出独立可检查的证明证书。

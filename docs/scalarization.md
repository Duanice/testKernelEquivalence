# 从两个 Triton 程序到一个 Z3 查询

## 1. 前端手工保留什么

输入程序保存在 `sources/case1_matmul_grouped_ordering/` 和
`sources/case2_elementwise_persistent_tiling/`。快照包含源目录已有的未提交编辑，源目录未改动。
两个模型分别遍历各自 launch，而不是先假定它们都实现同一个目标函数。

| Triton 源码结构 | 手工模型 | 自动标量化 |
|---|---|---|
| `tl.arange(0, BLOCK)` | tile shape 和 tile 内坐标 | 访问某一输出时传入局部标量下标 |
| `tl.load(ptr + offsets, mask, other)` | `load(name, shape, address, mask, other)` | 符号输入函数读取，必要时生成 If |
| `x + y`、`tl.maximum` | `elementwise` 节点 | 标量加法、比较与 If |
| `tl.dot(a,b,acc)` | `add(acc,dot(a,b))` | 对 dot 的 K 维完整展开 |
| GEMM 的 K 循环 | 固定配置下展开为多个 dot 节点 | 递归标量化各个 dot，保留求和分组 |
| grouped pid / persistent loop | 分别构造 `Store(origin,value,writer)` | 按输出坐标选择对应 Store |
| `tl.store(...,mask=...)` | 有效输出区域及覆盖检查 | 只在合法输出域内比较结果 |

缓存提示、num_warps、device、torch 包装不影响本模型里的数学值，没有纳入 IR。
`Store` 只支持这些案例使用的矩形 tile 和全局边界 store mask；不是任意指针写入的通用表示。
所有模型固定连续布局：A 行 stride=K、B/C 行 stride=N；不覆盖一般 stride。

## 2. Case2：不同 grid 需要按地址对齐

本次 n=77：

- before：BLOCK=8，10 个 program，每个处理一块。
- after：BLOCK=16，2 个 program；pid=0 处理 tile 0、2、4，pid=1 处理 tile 1、3。

输出地址 i 对应 tile 内位置 `i-origin`。load 地址经过代入变为 `X(i)`、`Y(i)`。
两侧有效输出的值都是：

```text
If(X(i) + Y(i) >= 0, X(i) + Y(i), 0)
```

脚本实际构建的是保留各自 Store 选择条件的表达式，导出的 formulas 中仍可看见不同 tile 边界。
查询为 `0 <= i < 77 AND before(i) != after(i)`。
在最大值被改成最小值的负例中，这一查询可满足。

## 3. Case1：dot 具体怎样变成标量

本次矩阵为 `A[33,19] × B[19,17]`，输出为 `C[33,17]`。
before 的 BK=16，展开为两个 dot；after 的 BK=32，展开为一个 dot。
BM=BN=16，输出 tile 网格为 3×2。

| pid | before 的 tile | after 的 tile（GROUP_M=2） |
|---:|---|---|
| 0 | (0,0) | (0,0) |
| 1 | (0,1) | (1,0) |
| 2 | (1,0) | (0,1) |
| 3 | (1,1) | (1,1) |
| 4 | (2,0) | (2,0) |
| 5 | (2,1) | (2,1) |

因此比较相同 pid 会比较不同输出块。标量化器使用共同的输出 `(i,j)` 找到各自 tile。
合法输出位置保证源码中的 `% M`、`% N` 回绕后仍是 i、j。

记 `p_q(i,j) = A(i*19 + q) * B(q*17 + j)`。
这只是展示公式的缩写，程序里来自两个独立的 load 节点和 dot 节点：

```text
before(i,j) = (0 + sum(q=0..15, p_q))
                  + sum(q=16..31, q<19 ? p_q : 0)

after(i,j)  = 0 + sum(q=0..31, q<19 ? p_q : 0)
```

实际 Z3 公式不包含未定义的 `sum`：标量化器用 Python 的固定整数循环生成每一项，再生成 Z3 加法。
输入 A、B 是实数值函数；乘法和加法是 Z3 自带的实数运算，**没有把 dot 声明成未解释函数**。

检查：

```text
0 <= i < 33
0 <= j < 17
before(i,j) != after(i,j)
```

两个正确模型返回 UNSAT。负例把 after 的 A load mask 改成 `q<18`，因此漏掉 `p_18`；
Z3 可以选择最后一项非零的输入作为反例。

## 4. 为什么 SAT 后还要回放

`verify.py` 从 Z3 模型提取所有 A/B 或 X/Y 元素，保存为精确分数字符串。
`replay.py` 根据两套 launch 和索引循环直接计算整个输出，使用 Python `Fraction`，不导入张量 IR。
验证反例位置的结果同时满足：

1. 回放 before == Z3 before。
2. 回放 after == Z3 after。
3. 回放 before != 回放 after。

这样可以发现反例提取、地址对齐或模型求值的实现错误。
不过两份实现都由人根据源码写出，仍不能消除前端建模错误；自动 TTIR 提取是另一个阶段。

## 5. 不应把结果扩大到哪里

这四个实例没有 exp/sqrt；不需要给这些函数加公理，也不会把这类抽象下的 SAT 当成真实反例。
验证的是固定形状的全部实数输入，而不是对少量输入做测试。归约的全部 19 项都参与。
只在 CPU 上运行了数学模型，未运行 Triton 解释器或 GPU；忽略 TF32/FP32 等浮点效应。
本次结果不能证明任意模型、任意 shape 或一般 AI 生成 kernel 都能高效验证。

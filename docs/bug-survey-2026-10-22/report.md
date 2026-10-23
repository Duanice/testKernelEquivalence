**Triton 算子改写验证的 GitHub 缺陷调研**

调研日期：2026 年 10 月 22 日（America/Los_Angeles；采集时间为 UTC 10 月 23 日）。范围为 Triton、vLLM、FlagGems，另核实用户提供的 Unsloth、Liger、SGLang 案例。

这批记录支持一个具体研究动机：Triton 算子的正确性依赖输入布局、分块配置、启动方式和机器整数语义，输出差分测试也可能漏掉非法访问。但它不足以证明某类 bug 在整个生态中的占比、某种验证方法的覆盖率，或论文的新颖性。用户提供的“55 个、37/18、76%”缺少完整逐条清单，本轮不沿用这些数字。

本轮建立 **36 条主样本缺陷记录，来自 34 个 issue/PR**；另有 **4 条补充记录，来自 3 个 issue/PR**。阅读了 17 个 PR 的关键生产代码 diff，其中包括一个去重的回移 PR。没有执行 GPU 复现；只在本地复核了若干整数和数值公式。

[逐条案例与证据](case_catalog.md) · [CSV 数据](cases.csv) · [结构化记录](records.json) · [统计结果](stats.json) · [检索日志](query_log.json) · [原始证据快照](source_snapshots.json) · [CPU 算术核对](arithmetic_checks.json)

**1. 样本和计数方法**

九个检索查询返回 157 条结果，URL 去重后为 151 条；再结合指定案例，按与 kernel 正确性、配置和内存语义的相关性选择深入阅读对象。检索包含 overflow、stride、mask、grid 等关键词，因此是有目的的案例收集，存在明显选择偏差，不是随机样本或全量审计。检索式、结果 URL 和选择后的记录均已保存。

以独立缺陷或同根因缺陷族计数，不以链接数量计数。FlagGems #5234 的整数溢出与遗漏 feature、vLLM #52576 的越界读与跨量化组，分别拆成两条；Unsloth #13830 在补充集拆成地址溢出和旧编译器兼容性两条。FlagGems #7539 与上游 #6978 合并计一次。同一缺陷的 issue、修复 PR、回移及 vendored 副本不重复计数。

证据 A 表示读了正文及关键生产代码修复 diff；B 表示依据 issue 中的机制解释、复现代码、日志或 IR 节选，未另审生产修复 diff。A 不等于已合并或已独立复现。主样本为 A 类 15 条、B 类 21 条；其中与已合并 PR 对应的主样本只有 F01、F02，其他条目须保留各自的开放状态和证据限制。

| 仓库 | 主样本缺陷记录 |
|---|---:|
| FlagGems | 13 |
| vLLM | 12 |
| Triton | 11 |
| 合计 | 36 |

| 所在层和状态 | 数量 |
|---|---:|
| kernel、算子库或 host 封装层 | 27 |
| 编译器问题报告 | 7 |
| 编译器数值契约存在争议 | 1 |
| 根因和归属未定 | 1 |

Triton 仓库中的 `triton_kernels` 是算子库：#12368、#12370、#12365 均归应用层。vLLM #62697 只保留 Triton reader 的 pattern A 为根因未定；其 pattern B 在报告中定位到 HIP cache writer，不纳入 Triton 主样本。Triton #6647 的评论讨论了归约重物化与浮点契约，不将它直接升级为已确认误编译。

**2. 应用层分类结果**

以下 27 条采用互斥的主要根因分类；shape、layout、dtype 等触发标签允许重叠。

| 主要根因 | 数量 | 代表案例 |
|---|---:|---|
| 尾块、mask、padding 范围 | 7 | [vLLM #52576](https://github.com/vllm-project/vllm/issues/52576)、[FlagGems #7118](https://github.com/flagos-ai/FlagGems/issues/7118) |
| 索引、stride、布局映射 | 6 | [FlagGems #5374](https://github.com/flagos-ai/FlagGems/pull/5374)、[vLLM #55109](https://github.com/vllm-project/vllm/pull/55109) |
| 整数中间值或地址溢出 | 5 | [FlagGems #5234](https://github.com/flagos-ai/FlagGems/pull/5234)、[vLLM #48398](https://github.com/vllm-project/vllm/pull/48398) |
| 覆盖遗漏或启动 grid 约束 | 4 | [FlagGems #5225](https://github.com/flagos-ai/FlagGems/pull/5225)、[FlagGems #6978](https://github.com/flagos-ai/FlagGems/pull/6978) |
| 数值、dtype、非有限值 | 3 | [FlagGems #7038](https://github.com/flagos-ai/FlagGems/pull/7038)、[vLLM #57094](https://github.com/vllm-project/vllm/pull/57094) |
| tile 与量化组不匹配 | 1 | [vLLM #52576](https://github.com/vllm-project/vllm/issues/52576) 的另一缺陷 |
| 算子模式或参考语义错误 | 1 | [FlagGems #7542](https://github.com/flagos-ai/FlagGems/issues/7542) |

其中 **23/27（85.2%）的报告明确给出了 shape、分块/调优配置或 stride/layout 触发条件**。这只描述所选案例；不是“85.2% 的真实 bug 都是配置相关”，也不是“这些 bug 仅在列出的条件下发生”。本轮不把翻转维度（F01）、dtype与数值范围（F12）、训练模式（F13）、数据分布（V10）单独计作上述三类条件，比例中明确排除了这四条。

本轮尾块/mask 类最多，不能据此确认“整数溢出是最大类”。不同关键词、去重规则和采样窗口都会改变排名。与数值有关的错误也可能受 shape 影响：vLLM #57094 的 NaN 正是由 CUDA graph padding 产生的零长度行触发。

**3. 对原有研究设想最有影响的发现**

**host 启动和初始化必须进入验证对象。** vLLM #49087 同时修改 grid、kernel 的 program 编号解码，以及 `zeros_like` 到 `empty_like` 的初始化。FlagGems #5225 则修复固定 grid 与 autotune tile 的不一致。只给验证器两份函数体 TTIR，不给 grid、参数约束和初始内存状态，无法完整表达这些问题。[vLLM #49087](https://github.com/vllm-project/vllm/pull/49087)；[FlagGems #5225](https://github.com/flagos-ai/FlagGems/pull/5225)。

**常规测试漏检的机制有直接案例，而非只有比例。** #5225 报告五种候选中四种会漏写，但首次调优时全部候选轮流写同一输出，覆盖完整的候选可能掩盖缺陷；缓存某个较小 tile 后才暴露。#55109 报告平台 dispatcher 的测试可能走另一实现，只有直接运行 Triton 路径和非连续视图才能暴露故障。这比笼统说“测试覆盖不到”更适合作为论文动机。[FlagGems #5225](https://github.com/flagos-ai/FlagGems/pull/5225)；[vLLM #55109](https://github.com/vllm-project/vllm/pull/55109)。

**溢出要检查中间表达式，阈值不是统一的输入元素数。** FlagGems pdist 的 `N*(N-1)` 在 N=46342 时为 2147534622，超过有符号 int32；输入 `(46342,4)` 只有 185368 个元素，尽管输出约有十亿个 pair。index_put 的 index×stride 和 randn 的 pid×BLOCK×4 则是地址中间值溢出。在乘法之后才转 int64 无法恢复丢失的高位。[FlagGems #5234](https://github.com/flagos-ai/FlagGems/pull/5234)；[#6892](https://github.com/flagos-ai/FlagGems/pull/6892)；[#7269](https://github.com/flagos-ai/FlagGems/issues/7269)。

建模上可保留两层：用 bit-vector 精确表达机器运算，先证明与实际访存相关的中间地址计算不发生改变逻辑地址的溢出，再在满足桥接条件的部分使用无界整数和 Presburger 方法。两个符号量相乘（如符号index×符号stride）并不属于一般Presburger算术，仍需限制参数、定界或引入额外引理；int64也不是无限范围。不是所有整数都应禁止回绕，例如 RNG 计数器可能有意采用模算术；也不应将永不访存的 masked lane 上每个中间回绕都直接报为程序错误。

**输出相等不能替代安全检查。** vLLM #52576 的 packed weight load 在 K 尾块越界，而对应 scale 被 mask 为零，报告中的数值结果因此可能正常。lifting 应保留每次访存的对象、地址、mask 和依赖，先产生安全义务，不能因为数值表达式最终乘零而将访问事实丢掉。Triton 库 #12365 还显示错误 mask 会越过行边界，造成多个 program 写同一 scale 地址；首阶段可限制为无竞争子集，并把竞争自由写成前提或检查项。[vLLM #52576](https://github.com/vllm-project/vllm/issues/52576)；[Triton #12365](https://github.com/triton-lang/triton/issues/12365)。

**验证应量化合法配置，并证明配置选择满足前提。** SGLang #39626 明确是手动提供 BK64/group32，官方 tuner 会排除此组合。相应任务可以是证明 `BK>0` 且 `group_k % BK == 0` 对该单scale-per-tile实现成立，验证不合法配置被拒绝，或验证新实现真正支持跨组缩放。不能把每个可填写的配置都默认当成程序承诺支持的配置。[SGLang #39626](https://github.com/sgl-project/sglang/issues/39626)。

**修复对与优化对需要不同预期。** 有缺陷旧版与修复版在触发输入上通常应为 NOT_EQ，适合检验反例发现能力。正确的两种优化实现才应为 EQ。FlagGems #6978/#7539 的 capped grid + grid-stride 是有价值的正例候选，但原程序在超限 grid 上无法执行：共同合法域上可验证等价；扩展到原来不能启动的 shape，应证明新版满足逻辑 spec 和后端约束。index_copy 若允许重复目标索引，还须处理写竞争或限定索引唯一，不能直接宣称任意输入唯一写入。[FlagGems #6978](https://github.com/flagos-ai/FlagGems/pull/6978)；[回移 #7539](https://github.com/flagos-ai/FlagGems/pull/7539)。

**4. 编译器边界与数值边界**

七条编译器问题报告分别涉及 pipeline/GDC 顺序（#11905）、warp specialization 副作用（#12395）、归约跨循环改写（#12348）、嵌套循环 flatten（#12192）、前端 loop carry（#12104）、浮点转换舍入（#5925）、AMD lowering（#11378）。它们不全位于 TTGIR：#12104 发生在前端生成 TTIR 时；#11378 的后续评论将线索指向 lowering 时的 region simplification。#11905 虽已关闭，但没有已读评论说明修复原因，不能据关闭状态宣称已解决。

如果研究以输入 TTIR 为起点，则应写清“在可信源码到 TTIR 转换及目标编译链假设下”的保证，或另设 translation validation。验证高层两份 TTIR 不能自动保证后续所有 backend 编译 pass 正确，也无法修复一个已在前端丢失的语义。各条证据见案例表中的 T05–T11。

Liger #1490 的一行修复很适合作为 mask 反例，但它涉及 `-inf`、tanh 和 softmax。最小例子 V=3、softcap=1：错误 padding 使分母从3变成 `3+exp(-1)`，损失由 log(3) 变为约1.2142833。纯实数加未解释函数不能无条件还原这个机制；应明确使用有效lane语义、扩展实数或浮点模型。[Liger #1490](https://github.com/linkedin/Liger-Kernel/pull/1490)。

因此目前不能支持“方法覆盖不到约28%”这种性能数字。应先实现验证器，再在冻结、去重且划分清楚的基准上统计 PROVED、COUNTEREXAMPLE、UNKNOWN、UNSUPPORTED；数值类需要浮点语义，参考语义错误需要可信 spec，都不等于永远无法验证。

**5. 建议第一批验证案例**

| 优先级 | 案例 | 要验证的内容 | 预期 |
|---|---|---|---|
| 1 | FlagGems #5225 | host grid、autotune tile 与完整写覆盖 | 错误配置给反例；修复覆盖所有允许配置 |
| 2 | vLLM #49087 | floor/ceil grid、尾块、输出初始化 | N=257等输入区分旧版与新版 |
| 3 | FlagGems #6978（#7539回移） | capped grid、grid-stride、后端grid乘积限制 | 合法域等价；扩展域满足spec与启动条件 |
| 4 | vLLM #48398 | int32乘法前扩位，机器地址与逻辑地址关系 | 构造地址反例；在明确界限内证明修复 |
| 5 | vLLM #52576 第1项 | packed weight尾块load的安全义务 | 即使输出相等也能报告越界 |

这组覆盖核心证明义务，并有已读的小规模代码差异。Liger softcap和Triton batched split-K放到数值模型、多kernel语义扩展阶段。Unsloth #13830也可用于溢出，但它同时包含硬件/编译器版本fallback，作为最小第一例不如vLLM #48398单纯。

建议将问题表述为：**对受支持 Triton 子集及其 host 启动描述，在明确的 shape、layout、配置集合和后端契约下，验证改写的可观察结果关系，并检查有效访存、写覆盖及所需的竞争和整数地址义务。** 首阶段使用实数计算值与精确机器整数地址，数值舍入和编译器正确性分别限定范围。样本已证明这些义务各有真实报告支撑；尚未证明通用自动 lifting 的可行性、验证器覆盖率或论文贡献强度。

**GitHub 缺陷逐条目录**

采集日期：2026-10-22（America/Los_Angeles）。全部记录均未在本轮执行 GPU 复现。A=已读正文和关键生产代码 diff，B=报告/复现/日志/IR 节选。开放 PR 的修复及报告中的根因不等于维护者已确认。

[返回分析报告](report.md) · [CSV](cases.csv) · [JSON](records.json)

**主样本：36 条，34 个 issue/PR**
| ID与来源 | 层/根因 | 触发条件 | 证据与状态 |
|---|---|---|---|
| [F01 FlagGems #5374](https://github.com/flagos-ai/FlagGems/pull/5374) | 应用层：索引/stride/布局 | 仅翻转部分前导维度；示例 shape=(20,320,15)、仅 dim=1 | A；已合并 |
| [F02 FlagGems #6978](https://github.com/flagos-ai/FlagGems/pull/6978) | 应用层：覆盖/启动grid | Ascend grid 各轴乘积 >65535；(200,481) 得96200 | A；已合并 |
| [F03 FlagGems #5225](https://github.com/flagos-ai/FlagGems/pull/5225) | 应用层：覆盖/启动grid | autotune tile 与固定(64,128) launch grid 不匹配；缓存赢家后可暴露 | A；开放PR |
| [F04 FlagGems #5234](https://github.com/flagos-ai/FlagGems/pull/5234) | 应用层：整数溢出 | N>=46342 时 N*(N-1) 中间乘积越过有符号32位 | A；开放PR |
| [F05 FlagGems #5234](https://github.com/flagos-ai/FlagGems/pull/5234) | 应用层：覆盖/启动grid | feature M>1024，但仅处理 capped BLOCK_M 单个tile | A；开放PR |
| [F06 FlagGems #6892](https://github.com/flagos-ai/FlagGems/pull/6892) | 应用层：整数溢出 | int32 index * stride 超过2^31-1；large strided index_put | A；开放PR |
| [F07 FlagGems #7269](https://github.com/flagos-ai/FlagGems/issues/7269) | 应用层：整数溢出 | randn pid*BLOCK*4 的有符号32位地址溢出 | B；open issue |
| [F08 FlagGems #7091](https://github.com/flagos-ai/FlagGems/issues/7091) | 应用层：索引/stride/布局 | index 为 expand(124) 的零stride视图，仅有一个int64存储元素 | B；open issue |
| [F09 FlagGems #7118](https://github.com/flagos-ai/FlagGems/issues/7118) | 应用层：尾块/mask | argmin k1路径非整tile，最小输入shape=(1,) | B；open issue |
| [F10 FlagGems #7120](https://github.com/flagos-ai/FlagGems/issues/7120) | 应用层：尾块/mask | reflect pad长度20、左右各3；生成kernel读mask错误 | B；open issue |
| [F11 FlagGems #7396](https://github.com/flagos-ai/FlagGems/issues/7396) | 应用层：索引/stride/布局 | slice_backward grad_output为零stride或非连续视图 | B；open issue |
| [F12 FlagGems #7038](https://github.com/flagos-ai/FlagGems/pull/7038) | 应用层：数值/dtype | float64先降为float32；exp(90)或exp(700)错误得到inf | A；开放PR |
| [F13 FlagGems #7542](https://github.com/flagos-ai/FlagGems/issues/7542) | 应用层：参考/模式语义 | train=False但使用training输入梯度公式 | B；open issue |
| [V01 vllm #49087](https://github.com/vllm-project/vllm/pull/49087) | 应用层：覆盖/启动grid | hidden_dim不是256倍数，或小于256 | A；开放PR |
| [V02 vllm #52576](https://github.com/vllm-project/vllm/issues/52576) | 应用层：尾块/mask | GPTQ/AWQ K=96,BK=64等尾块，packed weights无mask读取 | B；open issue |
| [V03 vllm #52576](https://github.com/vllm-project/vllm/issues/52576) | 应用层：tile/量化组 | block-FP8 BK=256/512、group_k=128，单tile跨独立scale组 | B；open issue |
| [V04 vllm #48398](https://github.com/vllm-project/vllm/pull/48398) | 应用层：整数溢出 | DeepGEMM ep_gather source_token_index * stride超过int32 | A；开放PR |
| [V05 vllm #59407](https://github.com/vllm-project/vllm/pull/59407) | 应用层：整数溢出 | MoE-LoRA num_tokens*top_k*N达到32位地址边界 | A；开放PR |
| [V06 vllm #55109](https://github.com/vllm-project/vllm/pull/55109) | 应用层：索引/stride/布局 | 窄化block_table视图的stride(0)>shape[-1]且多请求 | A；开放PR |
| [V07 vllm #57167](https://github.com/vllm-project/vllm/pull/57167) | 应用层：索引/stride/布局 | topk_weights为wide[:,:top_k]，列stride=1但行stride不匹配 | A；开放PR |
| [V08 vllm #61803](https://github.com/vllm-project/vllm/issues/61803) | 应用层：尾块/mask | 最后KV tile的page-table load超出有效页列 | B；open issue |
| [V09 vllm #47323](https://github.com/vllm-project/vllm/pull/47323) | 应用层：尾块/mask | Hopper-swizzled MXFP4 scales且EVEN_K=False | A；开放PR |
| [V10 vllm #60104](https://github.com/vllm-project/vllm/pull/60104) | 应用层：数值/dtype | 后续tile的极大logit超出首tile估计的exp参考 | A；开放PR |
| [V11 vllm #57094](https://github.com/vllm-project/vllm/pull/57094) | 应用层：数值/dtype | CUDA graph padding产生seq_len=0行，0/0 NaN污染batch量化scale | A；开放PR |
| [V12 vllm #62697](https://github.com/vllm-project/vllm/issues/62697) | 归属未定：根因未定 | RDNA pattern A：padded head>=128、KV block>=32等读路径故障 | B；open issue |
| [T01 triton #12368](https://github.com/triton-lang/triton/issues/12368) | 应用层：尾块/mask | persistent ragged-K未pad且slice末尾非BK整除，误读下一slice | B；open issue |
| [T02 triton #12370](https://github.com/triton-lang/triton/issues/12370) | 应用层：索引/stride/布局 | batched split_k>1且每batch输出scale不同 | B；open issue |
| [T03 triton #12365](https://github.com/triton-lang/triton/issues/12365) | 应用层：尾块/mask | 非persistent fused SwiGLU+MX输出，N=576使输出scale最后tile不满 | B；open issue |
| [T04 triton #6647](https://github.com/triton-lang/triton/issues/6647) | 契约争议：编译器数值契约 | num_stages由1改2；reduce重物化采用不同布局/顺序 | B；closed issue |
| [T05 triton #11905](https://github.com/triton-lang/triton/issues/11905) | 编译器报告：pipeline/同步 | H20 PDL+num_stages=2，gdc_wait位于K循环内 | B；closed issue |
| [T06 triton #12395](https://github.com/triton-lang/triton/issues/12395) | 编译器报告：副作用划分 | Hopper warp_specialize=True、num_warps=4，loop内atomic/gdc副作用 | B；open issue |
| [T07 triton #12348](https://github.com/triton-lang/triton/issues/12348) | 编译器报告：归约改写 | OptimizeThreadLocality改写update与reduce combiner不一致的loop | B；open issue |
| [T08 triton #12192](https://github.com/triton-lang/triton/issues/12192) | 编译器报告：循环改写 | flatten=True时outer loop carry在inner迭代提前更新 | B；open issue |
| [T09 triton #12104](https://github.com/triton-lang/triton/issues/12104) | 编译器报告：前端loop carry | prev与cur初始共享SSA值，loop内prev=cur且cur更新 | B；open issue |
| [T10 triton #5925](https://github.com/triton-lang/triton/issues/5925) | 编译器报告：浮点转换 | A100 f16->f8e5m2 round-to-nearest-even，9误舍入为10 | B；open issue |
| [T11 triton #11378](https://github.com/triton-lang/triton/issues/11378) | 编译器报告：lowering | gfx1151 guarded scf.for；num_warps影响错值或fault | B；open issue |

**补充样本：4 条，3 个 issue/PR；不混入主样本比例**

| ID与来源 | 层/根因 | 触发条件 | 证据与状态 |
|---|---|---|---|
| [S01 unsloth #13830](https://github.com/unslothai/unsloth/pull/13830) | 应用层：整数溢出 | FP8 GEMV weights >2^31元素，row*stride的32位中间溢出 | A；已合并 |
| [S02 unsloth #13830](https://github.com/unslothai/unsloth/pull/13830) | 编译器报告：lowering | Blackwell+Triton3.3，BLOCK_N=4/8或warp组合 | A；已合并 |
| [S03 Liger-Kernel #1490](https://github.com/linkedin/Liger-Kernel/pull/1490) | 应用层：尾块/mask | softcap启用且vocabulary尾tile有padding | A；已合并 |
| [S04 sglang #39626](https://github.com/sgl-project/sglang/issues/39626) | 应用层：tile/量化组 | 手工BK64、quant group32；BK32正确 | B；open issue |

**逐条限制和验证适用性**
各条的 fit 是研究适用性判断，不是验证器实测结果。精确复现条件与代码以保存的 source_snapshots.json 为准。

**F01 — [fix flip op accuracy test failure on mthreads backend](https://github.com/flagos-ai/FlagGems/pull/5374)**

已读已合并 diff：改为逐维解码和 flip_mask；host 新增 shape/stride/mask 参数，不能只看 kernel。

固定修复分支快照：[`410c43ac8e2a`](https://github.com/flagos-ai/FlagGems/commit/410c43ac8e2a15309dc4bc4c3ef7cb7e3af0b550)；本轮记录的 base SHA：`f7c55cb2fe1fb90fc713eafa6f63d7cbb73453a9`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**F02 — [[KMCompiler][Ascend] fix(index_copy): cap launch grid at the coreDim limit  ](https://github.com/flagos-ai/FlagGems/pull/6978)**

已合并；#7539 为同一修复的回移，只计一次。旧程序超限不能执行，需区分共同合法域等价与新域扩展；index_copy 重复索引还需额外约束。

固定修复分支快照：[`46ffd90d5446`](https://github.com/flagos-ai/FlagGems/commit/46ffd90d54462b1fbb364bc43ea64a77d85c674b)；本轮记录的 base SHA：`bb84d2292dde57b100310d990317b2853ff5cdc6`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**F03 — [Fix im2col launch grid to match autotuned tile sizes](https://github.com/flagos-ai/FlagGems/pull/5225)**

PR 仍开放。已读 host grid 和 kernel num_row_tiles diff；正文报告5个候选4个漏写，首轮候选共同写缓冲区可掩盖错误。

固定修复分支快照：[`2e6860b96933`](https://github.com/flagos-ai/FlagGems/commit/2e6860b969335bfda44fd0d692d7a6eb5577d393)；本轮记录的 base SHA：`75cf3ea559e4e5e3d9d1fec3e989ca4dab6e0af0`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**F04 — [Fix pdist int32 pair-count overflow and 1024-feature truncation](https://github.com/flagos-ai/FlagGems/pull/5234)**

同一 PR 的第1个独立缺陷。输入示例(46342,4)仅185368元素；不能将所有int32溢出描述为输入元素数>2^31。

固定修复分支快照：[`2a826a4b2880`](https://github.com/flagos-ai/FlagGems/commit/2a826a4b288010676f93c7538614a8408ef6acdc)；本轮记录的 base SHA：`75cf3ea559e4e5e3d9d1fec3e989ca4dab6e0af0`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**F05 — [Fix pdist int32 pair-count overflow and 1024-feature truncation](https://github.com/flagos-ai/FlagGems/pull/5234)**

同一 PR 的第2个缺陷。已读新增 NUM_TILES 循环；为归约遗漏项，归到覆盖类而非整数溢出。

固定修复分支快照：[`2a826a4b2880`](https://github.com/flagos-ai/FlagGems/commit/2a826a4b288010676f93c7538614a8408ef6acdc)；本轮记录的 base SHA：`75cf3ea559e4e5e3d9d1fec3e989ca4dab6e0af0`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**F06 — [fix(index_put): prevent int32 offset overflow in index_put_ and _index_put_impl_](https://github.com/flagos-ai/FlagGems/pull/6892)**

与 issue#6411、PR#6413 是同一缺陷族；本轮计一次。须在乘法前扩位，仅最终指针扩位不够。

固定修复分支快照：[`32f067cf226f`](https://github.com/flagos-ai/FlagGems/commit/32f067cf226f8b9cc2f73581b88807fc73120346)；本轮记录的 base SHA：`e905c8fb6205d2d9a35d60cb67421ad2e2dfc72c`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**F07 — [[Bug] randn output address overflows int32 above 2^31 elements](https://github.com/flagos-ai/FlagGems/issues/7269)**

正文有已捕获IR的描述及无分配算术复现。RNG计数器和输出地址是不同算术；本轮未检查IR附件或执行GPU。

**F08 — [Illegal memory access in index_select_kernel when using non-contiguous tensors](https://github.com/flagos-ai/FlagGems/issues/7091)**

正文含最小复现和sanitizer记录；按连续索引读取导致越界。

**F09 — [Illegal memory accesses in argmin_kernel_opt_k1 when not using inpus that are a multiple of the block dimeniosn](https://github.com/flagos-ai/FlagGems/issues/7118)**

正文指出省略load/store mask，并附越界写日志。

**F10 — [Illegal memory accesses _pad_jit_function due to incorrect masking](https://github.com/flagos-ai/FlagGems/issues/7120)**

正文有sanitizer记录；未审查生成的padding源码，精确错误谓词待复现。

**F11 — [[BUG] slice_backward ignores zero-stride grad_output, corrupting every sliced .sum().backward()](https://github.com/flagos-ai/FlagGems/issues/7396)**

直接slice_backward复现支持stride缺陷；正文较宽的所有autograd路径断言不外推。建议的修复是host contiguous()，并非kernel-only。

**F12 — [Fix float64 precision and overflow in exp](https://github.com/flagos-ai/FlagGems/pull/7038)**

已读dtype分支diff。纯实数模型会抹掉此类错误。

固定修复分支快照：[`5ee1670a926a`](https://github.com/flagos-ai/FlagGems/commit/5ee1670a926a944ba9e4391170b3818bd982f2ce)；本轮记录的 base SHA：`0d95c20928ae2002d7d51f6d2df3307eb11bd8a3`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**F13 — [[Bug]native_batch_norm_backward computes the training-mode input gradient when train=False](https://github.com/flagos-ai/FlagGems/issues/7542)**

正文给出32元素复现与两种梯度公式；需要模式契约和可信参考，不能仅从两段相同错误代码发现。

**V01 — [[Bugfix][Kernel][MoE] Zero-expert identity kernel silently drops output for hidden_dim not a multiple of 256](https://github.com/vllm-project/vllm/pull/49087)**

开放PR；已读floor-grid改二维ceil-grid及zeros_like改empty_like，host初始化是语义的一部分。

固定修复分支快照：[`5756f78e2213`](https://github.com/vllm-project/vllm/commit/5756f78e221307852d93d8cece4e371d82d53247)；本轮记录的 base SHA：`4635cc3e8f609755ad45f9fc2cdb54fa7552038e`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**V02 — [[Bug]: Triton MoE and block-FP8 GEMMs mishandle the K tile — an out-of-bounds weight read, and wrong scales when a tile spans two quantization groups](https://github.com/vllm-project/vllm/issues/52576)**

该issue第1个独立缺陷；sanitizer报告越界而数值可被零scale掩盖，不能从输出一致推出访存安全。

**V03 — [[Bug]: Triton MoE and block-FP8 GEMMs mishandle the K tile — an out-of-bounds weight read, and wrong scales when a tile spans two quantization groups](https://github.com/vllm-project/vllm/issues/52576)**

该issue第2个独立缺陷。默认配置处于安全域；需证明可选配置约束或拒绝无效配置，不应声称所有config都必须支持。

**V04 — [[Bugfix] Fix int32 address overflow in DeepGEMM ep_gather Triton kernel](https://github.com/vllm-project/vllm/pull/48398)**

已读4行差异；乘法前转int64。正文有H100大batch启动复现，PR仍开放。

固定修复分支快照：[`9f8d9372c3de`](https://github.com/vllm-project/vllm/commit/9f8d9372c3de40a8f8f9525436e62bab2a78960f)；本轮记录的 base SHA：`5c0c987c03593a1c05b8728fb265c94deb3d0a4c`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**V05 — [[Bugfix][LoRA][MoE] Use int64 token offsets in MoE-LoRA Triton kernels](https://github.com/vllm-project/vllm/pull/59407)**

多个同模式kernel合并为一个缺陷族；已读各生产文件diff，正文只为部分路径报告GPU验证。

固定修复分支快照：[`407b660f517e`](https://github.com/vllm-project/vllm/commit/407b660f517ea3010594d690cce85aa5e4e1ce2e)；本轮记录的 base SHA：`9d61ade7f657833df27e43c4561455523c78c8fb`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**V06 — [[Bugfix] Fix DeepSeek V4 K-cache gather for strided block tables](https://github.com/vllm-project/vllm/pull/55109)**

已读使用真实row stride及host传参diff；连续和单请求控制可通过。

固定修复分支快照：[`f4a087a1d982`](https://github.com/vllm-project/vllm/commit/f4a087a1d9827279a8877b3afffc8698ddc2401c)；本轮记录的 base SHA：`ab905a885dfbfc60a2c02286cc9c608c93884de3`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**V07 — [[Bugfix][MoE] Reject non-contiguous routing tensors](https://github.com/vllm-project/vllm/pull/57167)**

只将Triton routing weights缺陷纳入本条；同PR CUDA topk_ids旁支不另计。修复添加host拒绝，作者未本地跑GPU测试。

固定修复分支快照：[`3b831781d3cd`](https://github.com/vllm-project/vllm/commit/3b831781d3cd13c8ab1e59ae165316434fbe0d4d)；本轮记录的 base SHA：`25b0add7b8a1c944d5c4e364f2de6aa82497a2ad`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**V08 — [[Bug]: triton unified_attention reads block_table out-of-bounds past the used columns](https://github.com/vllm-project/vllm/issues/61803)**

正文含代码和sanitizer日志；#61810/#62779等修复候选与本条去重，未据PR标题确认最终修复。

**V09 — [Fix OOB scale reads in Triton MXFP4 matmul kernel causing NaN outputs](https://github.com/vllm-project/vllm/pull/47323)**

已读vLLM vendored triton_kernels补丁脚本；根因在算子库，不是Triton编译器，未额外计上游副本。

固定修复分支快照：[`98e927c7f859`](https://github.com/vllm-project/vllm/commit/98e927c7f859e4a787fd9c87b136bccd21ab3718)；本轮记录的 base SHA：`00eb7cefa31e32585fb419db5bb945f6a42480fe`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**V10 — [[Bugfix][Sampler] Stabilize standalone Triton top-p normalization](https://github.com/vllm-project/vllm/pull/60104)**

已读在线rescale与ties调整；纯实数会忽略exp溢出；本条不计入shape/config/layout条件比例。

固定修复分支快照：[`d2320617b64c`](https://github.com/vllm-project/vllm/commit/d2320617b64cd371f8285065bdcc7240f32f67a1)；本轮记录的 base SHA：`08a65e287e55099255ce178c5b5f58ce42dfd461`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**V11 — [[Bugfix][Attention] Avoid NaN from empty seq_len in Triton MLA decode (CUDA graph pad)](https://github.com/vllm-project/vllm/pull/57094)**

已读空行输出保护diff；明确说明数值类也可能依赖shape与执行模式。

固定修复分支快照：[`d5e5deb21139`](https://github.com/vllm-project/vllm/commit/d5e5deb211397a436b88a03d26b4790f9a215511)；本轮记录的 base SHA：`8e2beaa3bff595577a069d49d96bde9074294ec2`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**V12 — [[Bug][ROCm] `ROCM_ATTN` returns wrong attention (and can fault the GPU) on RDNA for head size ≥ 96 with KV block size ≠ 16, and for MHA with head size 192/256](https://github.com/vllm-project/vllm/issues/62697)**

只保留Triton reader的pattern A；正文pattern B定位到HIP cache writer，排除主样本。A尚未定因，不强归用户层或编译器。

**T01 — [`triton_kernels.matmul`'s persistent kernel accepts unpadded ragged-K input because its "At least one input must be padded!" check can never fail, silently giving wrong results instead of rejecting it](https://github.com/triton-lang/triton/issues/12368)**

triton_kernels库层。正文有persistent/nonpersistent对照；缺陷同时涉及dispatcher padding前提失效。

**T02 — [[triton_kernels] Batched split-K matmul silently mishandles per-batch output flexpoint scales](https://github.com/triton-lang/triton/issues/12370)**

triton_kernels库层；reduce未按batch偏移scale，actual_scale[1:]还可能未写。端到端涉及多个kernel及atomic统计。

**T03 — [Non-persistent `triton_kernels.matmul` with a fused SwiGLU masks the MX output scales with the pre-activation `N` instead of `N // 2`, silently writing past each row's scales](https://github.com/triton-lang/triton/issues/12365)**

triton_kernels库层；mask使用输入N而非yN=N/2，可跨行写、越界和写冲突，需安全及竞争检查。

**T04 — [Reduction is duplicated in TTIR -> TTGIR with num_stages>1 causing strange inconsistencies](https://github.com/triton-lang/triton/issues/6647)**

已读正文与全部5条评论。社区对是否违反数值契约存在讨论，不作为已确认编译器误编译。

**T05 — [Pipelined dot-input load appears before gdc_wait and produces incorrect results with PDL](https://github.com/triton-lang/triton/issues/11905)**

正文含TTGIR/PTX节选及对照；issue closed但无评论说明，不能据关闭推断已修复。

**T06 — [[Hopper] `warp_specialize=True` places loop ops by who uses their result, silently dropping or repeating atomics and `gdc_*`](https://github.com/triton-lang/triton/issues/12395)**

正文称副作用丢失或重复，TTGIR warp-specialization相关；不将每一种atomic重复计数。

**T07 — [In a loop, `acc -= tl.sum(x, 1)` silently adds the sum instead of subtracting it](https://github.com/triton-lang/triton/issues/12348)**

正文的减sum示例应为-512却得到+512；已读报告内pass定位及候选修复，未独立审查编译器源文件。

**T08 — [[FuseNestedLoops] flatten=True advances outer loop-carried indices on inner iterations](https://github.com/triton-lang/triton/issues/12192)**

正文及贡献者评论确认；FuseNestedLoops变换，不能笼统都叫TTGIR layout问题。

**T09 — [Loop-carry detection skips `prev = cur` when both start as the same value, silently keeping the pre-loop value instead of carrying it](https://github.com/triton-lang/triton/issues/12104)**

Python前端loop-carry发现缺陷，发生在TTIR形成阶段；比较两个已生成TTIR不能补足源码到TTIR保证。

**T10 — [RTNE semantics are not respected by `tt.fp_to_fp` going from f16 to f8e5m2 on A100](https://github.com/triton-lang/triton/issues/5925)**

正文只有短数值解释，无已读修复diff或评论；数值转换lowering报告，不归索引类。

**T11 — [[AMD] scf.for with runtime trip count miscompiled on gfx1151](https://github.com/triton-lang/triton/issues/11378)**

正文最初怀疑ISA，后续贡献者称3.8不复现且指向#10496 greedy region simplification；根因按后续评论保留为lowering相关。

**S01 — [FP8 rowwise GEMV: int64 row offsets past 2^31 elements, and skip it on Blackwell with Triton 3.3](https://github.com/unslothai/unsloth/pull/13830)**

同PR第1个缺陷；已合并diff含kernel前扩位和host LONG_INDEXING选择。恰好2^31元素在报告中仍正确。

固定修复分支快照：[`79c995eedb0f`](https://github.com/unslothai/unsloth/commit/79c995eedb0fbd673b9150fb0764134f6ccdf72e)；本轮记录的 base SHA：`e1cdbf4350004c45eebb4564e93ade446a907779`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**S02 — [FP8 rowwise GEMV: int64 row offsets past 2^31 elements, and skip it on Blackwell with Triton 3.3](https://github.com/unslothai/unsloth/pull/13830)**

同PR第2个缺陷；已读的是host版本/硬件fallback，不是编译器根因修复，不能宣称两个问题都只改kernel。

固定修复分支快照：[`79c995eedb0f`](https://github.com/unslothai/unsloth/commit/79c995eedb0fbd673b9150fb0764134f6ccdf72e)；本轮记录的 base SHA：`e1cdbf4350004c45eebb4564e93ade446a907779`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**S03 — [Fix softcap padding in cross entropy](https://github.com/linkedin/Liger-Kernel/pull/1490)**

已合并1行diff：softcap后重置越界lane为-inf。实数模型不能直接表示-inf，需显式有效lane或扩展数值语义。

固定修复分支快照：[`a652e16c71e6`](https://github.com/linkedin/Liger-Kernel/commit/a652e16c71e6b8d5c6693ea3dd8eeeab7fe08d81)；本轮记录的 base SHA：`70fe0333fef1052441b7825132e03c767b98fcb1`。PR base/head 是采集时状态；PR diff 对应 GitHub 的合并基准比较，不把该 base SHA 直接宣称为原始 bug 引入点。

**S04 — [[Bug] Block-FP8 Triton matmul silently accepts an unsupported K tile and returns wrong results](https://github.com/sgl-project/sglang/issues/39626)**

官方tuner会过滤非法组合；issue未报告默认serving触发，建议validator拒绝或重写支持跨group缩放。

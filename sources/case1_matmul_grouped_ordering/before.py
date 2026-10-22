"""Case 1 / before: tiled GEMM with the naive row-major program-id -> tile mapping.

Source: Triton tutorial 03-matrix-multiplication (the "naive" ordering it contrasts
with the grouped ordering). Each program computes one BLOCK_M x BLOCK_N tile of C.
"""

import triton
import triton.language as tl

# 矩阵乘法：C = A @ B。
# 输入 A：M × K 的矩阵
# 输入 B：K × N 的矩阵
# 输出 C：M × N 的矩阵
# C 中每个元素都是 A 的一行与 B 的一列，对应元素相乘后求和
# 代码把计算分给多个 GPU program：
# - 把 C 切成 BLOCK_M × BLOCK_N 的小块。
# - 每个 program 负责一块，按行给这些块编号。
# - 每轮沿 K 维处理 BLOCK_K 个位置，把乘积累加到 acc。
# - 完成所有 K 维计算后，把 acc 写入 C 的对应位置。
# 调用 launch(a, b)，就会启动这些 program，并返回结果矩阵 C。

@triton.jit
def matmul_rowmajor_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,#？？
):
    # tl.constexpr 表示：这个参数的值，在编译当前版本的 kernel 时就必须确定。
    # 其他参数在这里通常作为运行时参数传入
    pid = tl.program_id(axis=0)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # --- only this mapping differs from after.py ---------------------------
    pid_m = pid // num_pid_n
    pid_n = pid % num_pid_n
    # -----------------------------------------------------------------------

    # A 的行下标，假设pid_m = 1，BLOCK_M = 4，M = 10
    # offs_am = (1 * 4 + [0, 1, 2, 3]) % 10
    #     = [4, 5, 6, 7]。 表示要读取 A 的第 4、5、6、7 行。
    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    
    # B 的列下标
    # 假设pid_n = 2，BLOCK_N = 4，N = 10
    # offs_bn = (2 * 4 + [0, 1, 2, 3]) % 10
    #     = [8, 9, 0, 1]。 表示要读取 B 的第 8、9、0、1 列。（但最后只把对应第 8、9 列的结果写入 C。）
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    
    # A 的列下标，同时也是 B 的行下标
    # BLOCK_K = 4
    # offs_k = [0, 1, 2, 3]
    offs_k = tl.arange(0, BLOCK_K)

    # 假设stride_am = 8，stride_ak = 1 
    # A[i, k] 的地址 = a_ptr + i × 行步长 + k × 列步长
    # offs_am变成[[4],
    #            [5],
    #            [6],
    #            [7]]
    # 
    # offs_am[:, None] * 8    广播相加   offs_k[None, :] * 1 再加上A 的起始指针：
    # [[32],                           [[0, 1, 2, 3]]
    # [40],
    # [48],
    # [56]]
    # a_ptrs = a_ptr + [[32, 33, 34, 35],
    #              [40, 41, 42, 43],
    #              [48, 49, 50, 51],
    #              [56, 57, 58, 59]]
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)

    # 总共需要处理 ceil(K / BLOCK_K) 块，k 表示第几轮
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < K - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        # 把本轮 A 小块 × B 小块的结果，小矩阵乘法，加到 acc 中
        acc = tl.dot(a, b, acc)
        # 下一轮：A 的读取位置向右移动 BLOCK_K 列。当前 program 的循环只沿 K 维向右移动
        a_ptrs += BLOCK_K * stride_ak 
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def launch(a, b, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16):
    # BLOCK_M = 32   # 每个 program 计算 C 的 32 行
    # BLOCK_N = 32   # 每个 program 计算 C 的 32 列
    # BLOCK_K = 16   # 沿乘法的 K 维，每轮处理 16 个位置
    import torch

    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_rowmajor_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K,
    )
    return c


# (fn, signature, constexprs, compile options) consumed by scripts/dump_ir.py
_SIG = {
    "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
    "M": "i32", "N": "i32", "K": "i32",
    "stride_am": "i32", "stride_ak": "i32", "stride_bk": "i32", "stride_bn": "i32",
    "stride_cm": "i32", "stride_cn": "i32",
    "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
}
KERNELS = [
    ("matmul_rowmajor", matmul_rowmajor_kernel, _SIG,
     {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32}, {"num_warps": 4, "num_stages": 3}),
]

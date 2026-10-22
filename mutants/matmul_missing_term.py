# Deliberately incorrect negative control; not a bug in the original case.
"""Case 1 / after: the same tiled GEMM with "grouped ordering" (L2-cache swizzle)
of program ids, exactly as in Triton tutorial 03-matrix-multiplication.

Only the pid -> (pid_m, pid_n) mapping changes. The per-tile computation is
textually identical to before.py.
"""


# 为什么分块行？为了让不同 program 更容易复用读过的数据
    # 如： 计算 C00：需要 A 的第0块行、B 的第0块列
    #     计算 C10：需要 A 的第1块行、B 的第0块列
    #                                 ↑
    #                         用到了相同的 B 数据
    # 因此，after 把它们分配给相邻编号的 program：
    # before 的 pid：      after 的 pid（两块行一组）：
    # 0   1   2           0   2   4
    # 3   4   5           1   3   5
    # 这样，编号相近的 program 有机会复用 L2 缓存中的 B 数据，减少从显存重复读取。
    # 实际执行顺序仍由 GPU 调度

    # ？？但是before的实现难道不能编号相近的 program 复用 L2 缓存中的 A 数据吗？
    # re:能。对。after 的目的，是在较小范围内兼顾 A、B 两边的数据复用。
    # before 横着处理完整个块行，之后才换行；如果这一行很长，等下一块行再次需要之前的 B 数据时，
    # 它们可能已经离开缓存。分组让这部分 B 数据更早被再次使用，同时仍保留组内复用 A 的机会。
    # 所以两者都有缓存复用，分组是在调整复用方式，并不保证所有情况下都更快。

    # ？？？所以实际上 before 和 after只是改变了“某个program应该计算哪一小块”，对吗？---对，只改变pid → (pid_m, pid_n) 的对应关系
    # 验证时，应检查：前后是否都把每个 C 块恰好计算一次，并且对应块的计算相同，而不是要求相同 pid 的 program 做相同的事。

import triton
import triton.language as tl


@triton.jit
def matmul_grouped_kernel(
    a_ptr, b_ptr, c_ptr,
    M, N, K,
    stride_am, stride_ak,
    stride_bk, stride_bn,
    stride_cm, stride_cn,
    BLOCK_M: tl.constexpr, BLOCK_N: tl.constexpr, BLOCK_K: tl.constexpr,
    GROUP_M: tl.constexpr,
):
    pid = tl.program_id(axis=0)
    num_pid_m = tl.cdiv(M, BLOCK_M)
    num_pid_n = tl.cdiv(N, BLOCK_N)
    # --- only this mapping differs from before.py --------------------------
    num_pid_in_group = GROUP_M * num_pid_n # 一个完整的组里有多少个 program。组内块行数 × 块列数
    group_id = pid // num_pid_in_group  # 第几组
    first_pid_m = group_id * GROUP_M # 这一组的第一个块行的 pid_m
    group_size_m = min(
        num_pid_m - first_pid_m,  # 从本组起始位置到最后，还剩多少个块行
        GROUP_M  # 每组最多放多少个块行
        ) # 当前这一组实际包含多少个块行，因为最后一组可能凑不满 GROUP_M 行。
    pid_m = first_pid_m + ((pid % num_pid_in_group) % group_size_m)
    pid_n = (pid % num_pid_in_group) // group_size_m
    # -----------------------------------------------------------------------

    offs_am = (pid_m * BLOCK_M + tl.arange(0, BLOCK_M)) % M
    offs_bn = (pid_n * BLOCK_N + tl.arange(0, BLOCK_N)) % N
    offs_k = tl.arange(0, BLOCK_K)
    a_ptrs = a_ptr + (offs_am[:, None] * stride_am + offs_k[None, :] * stride_ak)
    b_ptrs = b_ptr + (offs_k[:, None] * stride_bk + offs_bn[None, :] * stride_bn)

    acc = tl.zeros((BLOCK_M, BLOCK_N), dtype=tl.float32)
    for k in range(0, tl.cdiv(K, BLOCK_K)):
        a = tl.load(a_ptrs, mask=offs_k[None, :] < (K - 1) - k * BLOCK_K, other=0.0)
        b = tl.load(b_ptrs, mask=offs_k[:, None] < K - k * BLOCK_K, other=0.0)
        acc = tl.dot(a, b, acc)#
        a_ptrs += BLOCK_K * stride_ak
        b_ptrs += BLOCK_K * stride_bk

    offs_cm = pid_m * BLOCK_M + tl.arange(0, BLOCK_M)
    offs_cn = pid_n * BLOCK_N + tl.arange(0, BLOCK_N)
    c_ptrs = c_ptr + stride_cm * offs_cm[:, None] + stride_cn * offs_cn[None, :]
    c_mask = (offs_cm[:, None] < M) & (offs_cn[None, :] < N)
    tl.store(c_ptrs, acc, mask=c_mask)


def launch(a, b, BLOCK_M=32, BLOCK_N=32, BLOCK_K=16, GROUP_M=8):# 把 C 的每 8 个块行划成一组，组内按列给 program 编号
    import torch

    M, K = a.shape
    K2, N = b.shape
    assert K == K2
    c = torch.empty((M, N), dtype=torch.float32, device=a.device)
    grid = (triton.cdiv(M, BLOCK_M) * triton.cdiv(N, BLOCK_N),)
    matmul_grouped_kernel[grid](
        a, b, c, M, N, K,
        a.stride(0), a.stride(1), b.stride(0), b.stride(1), c.stride(0), c.stride(1),
        BLOCK_M=BLOCK_M, BLOCK_N=BLOCK_N, BLOCK_K=BLOCK_K, GROUP_M=GROUP_M,
    )
    return c


_SIG = {
    "a_ptr": "*fp32", "b_ptr": "*fp32", "c_ptr": "*fp32",
    "M": "i32", "N": "i32", "K": "i32",
    "stride_am": "i32", "stride_ak": "i32", "stride_bk": "i32", "stride_bn": "i32",
    "stride_cm": "i32", "stride_cn": "i32",
    "BLOCK_M": "constexpr", "BLOCK_N": "constexpr", "BLOCK_K": "constexpr",
    "GROUP_M": "constexpr",
}
KERNELS = [
    ("matmul_grouped", matmul_grouped_kernel, _SIG,
     {"BLOCK_M": 64, "BLOCK_N": 64, "BLOCK_K": 32, "GROUP_M": 8}, {"num_warps": 4, "num_stages": 3}),
]

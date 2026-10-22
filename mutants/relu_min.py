# Deliberately incorrect negative control; not a bug in the original case.
"""Case 2 / after: "persistent" kernel. A fixed number of programs (≈ number of SMs)
loops over tiles with a grid-stride, and the tile size is changed (1024 -> 2048,
with 8 elements per thread for wider vectorised accesses).

The per-program footprint is now a strided *set* of tiles, so program-by-program
comparison with before.py is meaningless; only the whole-grid effect matches.
"""

import triton
import triton.language as tl


@triton.jit
def add_relu_persistent_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    num_programs = tl.num_programs(axis=0)
    num_tiles = tl.cdiv(n_elements, BLOCK)
    for tile in range(pid, num_tiles, num_programs): #当前 program 从编号为 pid 的块开始，每次跳 num_programs 块。
        offsets = tile * BLOCK + tl.arange(0, BLOCK)
        mask = offsets < n_elements
        x = tl.load(x_ptr + offsets, mask=mask)
        y = tl.load(y_ptr + offsets, mask=mask)
        out = tl.minimum(x + y, 0.0)
        tl.store(out_ptr + offsets, out, mask=mask)

def launch(x, y, BLOCK=2048, num_programs=8):
    import torch

    out = torch.empty_like(x)
    n = x.numel()
    grid = (min(num_programs, triton.cdiv(n, BLOCK)),)
    add_relu_persistent_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


# 这是给项目的离线编译工具准备的配置，告诉它“编译哪个 kernel、参数是什么类型、用什么配置”。
_SIG = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32", "BLOCK": "constexpr"}
KERNELS = [
    ("add_relu_persistent", add_relu_persistent_kernel, _SIG, {"BLOCK": 2048}, {"num_warps": 8}),
]

# 优化方式
# (a) tile 大小 1024 → 2048（每线程 8 元素，利于 128-bit 向量化访存）；
# (b) 网格从 cdiv(n, BLOCK) 个 program 改为固定少量（≈SM 数）program，每个 program 以 grid-stride 循环处理多个 tile

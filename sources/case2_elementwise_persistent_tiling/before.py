"""Case 2 / before: one program per BLOCK-sized tile (Triton tutorial 01-vector-add
style), computing out = relu(x + y) element-wise.

grid = cdiv(n, BLOCK). Each program touches exactly one contiguous tile.
"""

import triton
import triton.language as tl


@triton.jit
def add_relu_tiled_kernel(x_ptr, y_ptr, out_ptr, n_elements, BLOCK: tl.constexpr):
    pid = tl.program_id(axis=0)
    offsets = pid * BLOCK + tl.arange(0, BLOCK)
    mask = offsets < n_elements
    x = tl.load(x_ptr + offsets, mask=mask)
    y = tl.load(y_ptr + offsets, mask=mask)
    out = tl.maximum(x + y, 0.0)
    tl.store(out_ptr + offsets, out, mask=mask)


def launch(x, y, BLOCK=1024):
    import torch

    out = torch.empty_like(x)
    n = x.numel()
    grid = (triton.cdiv(n, BLOCK),)
    add_relu_tiled_kernel[grid](x, y, out, n, BLOCK=BLOCK)
    return out


_SIG = {"x_ptr": "*fp32", "y_ptr": "*fp32", "out_ptr": "*fp32", "n_elements": "i32", "BLOCK": "constexpr"}
KERNELS = [
    ("add_relu_tiled", add_relu_tiled_kernel, _SIG, {"BLOCK": 1024}, {"num_warps": 4}),
]

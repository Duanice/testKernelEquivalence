"""Manual frontends for the source snapshots. No Python/TTIR parser is claimed.

Keep the two launch/index constructions separate: comparing equal program IDs
would be wrong for both cases. Pair values by their final output coordinates.
"""

from tensor_model import Kernel, Store, dot, elementwise, full, load


def cdiv(n, block):
    return (n + block - 1) // block


def relu_before(n=77, block=8):
    stores = []
    for pid in range(cdiv(n, block)):
        base = pid * block
        x = load("X", (block,), lambda lane, base=base: base + lane,
                 lambda lane, base=base: base + lane < n)
        y = load("Y", (block,), lambda lane, base=base: base + lane,
                 lambda lane, base=base: base + lane < n)
        value = elementwise("max", elementwise("add", x, y), full((block,), 0))
        stores.append(Store((base,), value, f"pid={pid}"))
    return Kernel("relu_tiled", (n,), stores, {"X": n, "Y": n}, {"BLOCK": block})


def relu_after(n=77, block=16, programs=2, mutant=False):
    stores = []
    grid = min(programs, cdiv(n, block))
    for pid in range(grid):
        for tile in range(pid, cdiv(n, block), grid):
            base = tile * block
            x = load("X", (block,), lambda lane, base=base: base + lane,
                     lambda lane, base=base: base + lane < n)
            y = load("Y", (block,), lambda lane, base=base: base + lane,
                     lambda lane, base=base: base + lane < n)
            # Negative control: change only this operation, matching mutants/relu_min.py.
            value = elementwise("min" if mutant else "max", elementwise("add", x, y),
                                full((block,), 0))
            stores.append(Store((base,), value, f"pid={pid},tile={tile}"))
    return Kernel("relu_persistent" + ("_mutant" if mutant else ""), (n,), stores,
                  {"X": n, "Y": n}, {"BLOCK": block, "programs": grid})


def matmul_before(m=33, n=17, k=19, bm=16, bn=16, bk=16):
    stores = []
    tiles_n = cdiv(n, bn)
    for pid in range(cdiv(m, bm) * tiles_n):
        tile_m, tile_n = pid // tiles_n, pid % tiles_n
        acc = full((bm, bn), 0)
        for step in range(cdiv(k, bk)):
            start = step * bk
            a = load("A", (bm, bk),
                     lambda r, q, tm=tile_m, s=start: ((tm * bm + r) % m) * k + s + q,
                     lambda r, q, s=start: q < k - s, other=0)
            b = load("B", (bk, bn),
                     lambda q, c, tn=tile_n, s=start: (s + q) * n + (tn * bn + c) % n,
                     lambda q, c, s=start: q < k - s, other=0)
            acc = elementwise("add", acc, dot(a, b))
        stores.append(Store((tile_m * bm, tile_n * bn), acc, f"pid={pid}"))
    return Kernel("matmul_rowmajor", (m, n), stores, {"A": m * k, "B": k * n},
                  {"M": m, "N": n, "K": k, "BLOCK_M": bm, "BLOCK_N": bn, "BLOCK_K": bk})


def matmul_after(m=33, n=17, k=19, bm=16, bn=16, bk=32, group=2, mutant=False):
    stores = []
    tiles_m, tiles_n = cdiv(m, bm), cdiv(n, bn)
    group_width = group * tiles_n
    for pid in range(tiles_m * tiles_n):
        first_m = (pid // group_width) * group
        size_m = min(tiles_m - first_m, group)
        tile_m = first_m + (pid % group_width) % size_m
        tile_n = (pid % group_width) // size_m
        acc = full((bm, bn), 0)
        for step in range(cdiv(k, bk)):
            start = step * bk
            # Negative control: omit the last true reduction term via A's load mask.
            limit_a = k - (1 if mutant else 0)
            a = load("A", (bm, bk),
                     lambda r, q, tm=tile_m, s=start: ((tm * bm + r) % m) * k + s + q,
                     lambda r, q, s=start: q < limit_a - s, other=0)
            b = load("B", (bk, bn),
                     lambda q, c, tn=tile_n, s=start: (s + q) * n + (tn * bn + c) % n,
                     lambda q, c, s=start: q < k - s, other=0)
            acc = elementwise("add", acc, dot(a, b))
        stores.append(Store((tile_m * bm, tile_n * bn), acc, f"pid={pid}"))
    return Kernel("matmul_grouped" + ("_mutant" if mutant else ""), (m, n), stores,
                  {"A": m * k, "B": k * n},
                  {"M": m, "N": n, "K": k, "BLOCK_M": bm, "BLOCK_N": bn,
                   "BLOCK_K": bk, "GROUP_M": group})


def comparisons():
    return [
        ("case2_equivalent", relu_before(), relu_after(), "unsat"),
        ("case2_wrong_relu", relu_before(), relu_after(mutant=True), "sat"),
        ("case1_equivalent", matmul_before(), matmul_after(), "unsat"),
        ("case1_missing_term", matmul_before(), matmul_after(mutant=True), "sat"),
    ]

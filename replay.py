"""Independent CPU replay of the fixed launches, using fractions.Fraction.

Does not import the tensor IR or evaluate Z3 expressions. These are mathematical
kernel executions, not Triton interpreter or GPU executions.
"""

from fractions import Fraction


def replay_relu(inputs, n, block, programs=None, mutant=False):
    out = [None] * n
    num_tiles = (n + block - 1) // block
    grid = num_tiles if programs is None else min(programs, num_tiles)
    for pid in range(grid):
        tiles = [pid] if programs is None else range(pid, num_tiles, grid)
        for tile in tiles:
            for lane in range(block):
                index = tile * block + lane
                if index < n:
                    if out[index] is not None:
                        raise ValueError("Duplicate replay write")
                    value = inputs["X"][index] + inputs["Y"][index]
                    out[index] = min(value, 0) if mutant else max(value, 0)
    if any(x is None for x in out):
        raise ValueError("Missing replay writes")
    return out


def replay_matmul(inputs, cfg, grouped=False, mutant=False):
    m, n, k = (cfg[key] for key in ("M", "N", "K"))
    bm, bn, bk = (cfg[key] for key in ("BLOCK_M", "BLOCK_N", "BLOCK_K"))
    rows, cols = (m + bm - 1) // bm, (n + bn - 1) // bn
    out = [None] * (m * n)
    for pid in range(rows * cols):
        if grouped:
            group = cfg["GROUP_M"]
            first_row = pid // (group * cols) * group
            group_rows = min(rows - first_row, group)
            tile_row = first_row + pid % (group * cols) % group_rows
            tile_col = pid % (group * cols) // group_rows
        else:
            tile_row, tile_col = divmod(pid, cols)
        for r in range(bm):
            for c in range(bn):
                row, col = tile_row * bm + r, tile_col * bn + c
                if row >= m or col >= n:
                    continue
                acc = Fraction(0)
                for step in range((k + bk - 1) // bk):
                    block_sum = Fraction(0)
                    for lane in range(bk):
                        q = step * bk + lane
                        a = inputs["A"][(row % m) * k + q] if q < k - int(mutant) else Fraction(0)
                        b = inputs["B"][q * n + col % n] if q < k else Fraction(0)
                        block_sum += a * b
                    acc += block_sum
                addr = row * n + col
                if out[addr] is not None:
                    raise ValueError("Duplicate replay write")
                out[addr] = acc
    if any(x is None for x in out):
        raise ValueError("Missing replay writes")
    return out


def replay_pair(name, before, after, inputs):
    mutant = name in {"case2_wrong_relu", "case1_missing_term"}
    if name.startswith("case2_"):
        n = before.shape[0]
        return (replay_relu(inputs, n, before.config["BLOCK"]),
                replay_relu(inputs, n, after.config["BLOCK"], after.config["programs"], mutant))
    return (replay_matmul(inputs, before.config),
            replay_matmul(inputs, after.config, grouped=True, mutant=mutant))

"""Small, manually populated tensor IR and automatic exact-real scalarization."""

from dataclasses import dataclass
from itertools import product
from math import prod

import z3


@dataclass(eq=False)
class Tensor:
    shape: tuple
    op: str
    args: tuple = ()
    data: object = None


def full(shape, value=0):
    return Tensor(shape, "full", data=value)


def load(name, shape, address, mask=None, other=None):
    return Tensor(shape, "load", data=(name, address, mask, other))


def elementwise(op, left, right):
    if op not in {"add", "mul", "max", "min"} or left.shape != right.shape:
        raise ValueError("Unsupported operation or inconsistent tensor shapes")
    return Tensor(left.shape, op, (left, right))


def dot(left, right):
    if len(left.shape) != 2 or len(right.shape) != 2 or left.shape[1] != right.shape[0]:
        raise ValueError("dot requires compatible rank-2 tensors")
    return Tensor((left.shape[0], right.shape[1]), "dot", (left, right))


class Scalarizer:
    """Each input buffer is one shared Int -> Real uninterpreted function.

    shortcut: casts/rounding and integer overflow are outside this real-valued IR;
    add typed semantics before claiming IEEE or machine-integer equivalence.
    """

    def __init__(self):
        self.inputs = {}
        self.dot_terms = 0

    def input(self, name):
        if name not in self.inputs:
            self.inputs[name] = z3.Function(name, z3.IntSort(), z3.RealSort())
        return self.inputs[name]

    def at(self, tensor, indices):
        if len(indices) != len(tensor.shape):
            raise ValueError("Incorrect rank for scalarization")
        if tensor.op == "full":
            return z3.RealVal(tensor.data)
        if tensor.op == "load":
            name, address, mask, other = tensor.data
            addr = z3.simplify(z3.IntVal(0) + address(*indices))
            condition = z3.simplify(z3.And(True if mask is None else mask(*indices)))
            # Inactive masked loads without `other` are undefined, never zero-filled.
            fallback = (z3.RealVal(other) if other is not None else
                        z3.Function("masked_undefined_" + name, z3.IntSort(), z3.RealSort())(addr))
            if z3.is_false(condition):
                return fallback
            value = self.input(name)(addr)
            return value if z3.is_true(condition) else z3.If(condition, value, fallback)
        if tensor.op == "dot":
            left, right = tensor.args
            row, col = indices
            self.dot_terms += left.shape[1]
            return z3.Sum([self.at(left, (row, k)) * self.at(right, (k, col))
                           for k in range(left.shape[1])])
        if tensor.op not in {"add", "mul", "max", "min"}:
            raise ValueError(f"Unsupported IR operation: {tensor.op}")
        left, right = (self.at(arg, indices) for arg in tensor.args)
        if tensor.op == "add":
            return left + right
        if tensor.op == "mul":
            return left * right
        if tensor.op == "max":
            return z3.If(left >= right, left, right)
        return z3.If(left <= right, left, right)


@dataclass
class Store:
    origin: tuple
    value: Tensor
    writer: str


@dataclass
class Kernel:
    name: str
    shape: tuple
    stores: list
    inputs: dict
    config: dict

    def coverage(self):
        """Exhaust the fixed integer launch, not tensor values; masked stores discarded."""
        owners = {}
        for store in self.stores:
            for local in product(*(range(d) for d in store.value.shape)):
                coord = tuple(o + k for o, k in zip(store.origin, local))
                if not all(0 <= k < n for k, n in zip(coord, self.shape)):
                    continue
                if coord in owners:
                    raise ValueError(f"Multiple writers at {coord}: {owners[coord]}, {store.writer}")
                owners[coord] = store.writer
        if len(owners) != prod(self.shape):
            raise ValueError(f"Incomplete writes: {len(owners)}/{prod(self.shape)}")
        return {"outputs": len(owners), "stores": len(self.stores), "unique_writes": True}

    def output(self, scalarizer, indices):
        self.coverage()
        result = z3.RealVal(0)
        for store in self.stores:
            local = tuple(i - o for i, o in zip(indices, store.origin))
            active = z3.And(*(z3.And(k >= 0, k < n)
                              for k, n in zip(local, store.value.shape)))
            result = z3.If(active, scalarizer.at(store.value, local), result)
        return result

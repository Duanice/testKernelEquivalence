"""Checks the encoder against independent arithmetic and rejects vacuous proofs."""

from fractions import Fraction
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import z3

from case_models import comparisons, matmul_after, matmul_before, relu_after, relu_before
from replay import replay_pair
from tensor_model import Scalarizer, Tensor, full, load
from verify import check_sources, exact_fraction, verify


class VerifierTests(unittest.TestCase):
    def test_expected_verdicts_witnesses_and_exported_queries(self):
        check_sources()
        with tempfile.TemporaryDirectory() as directory:
            for item in comparisons():
                with self.subTest(case=item[0]):
                    row = verify(*item, Path(directory), 10000)
                    self.assertTrue(row["passed"])
                    if row["verdict"] == "sat":
                        self.assertTrue(row["counterexample"]["exact_replay_passed"])
                    solver = z3.Solver()
                    solver.set(timeout=10000)
                    solver.add(z3.parse_smt2_file(str(Path(directory) / f"{item[0]}.smt2")))
                    self.assertEqual(str(solver.check()), item[3])

    def test_scalarization_matches_independent_replay(self):
        pairs = [("case2_equivalent", relu_before(n), relu_after(n)) for n in (1, 16, 17, 77)]
        pairs += [("case1_equivalent", matmul_before(m, n, k), matmul_after(m, n, k))
                  for m, n, k in ((1, 1, 1), (17, 3, 17), (33, 17, 19))]
        for name, before, after in pairs:
            inputs = {key: [Fraction((i % 7) - 3, 3) for i in range(size)]
                      for key, size in before.inputs.items()}
            expected_pair = replay_pair(name, before, after, inputs)
            self.assertEqual(*expected_pair)
            for kernel, expected in zip((before, after), expected_pair):
                scalarizer = Scalarizer()
                coords = [(0,) * len(kernel.shape), tuple(n - 1 for n in kernel.shape)]
                for coord in coords:
                    with self.subTest(kernel=kernel.name, shape=kernel.shape, coordinate=coord):
                        expr = kernel.output(scalarizer, tuple(z3.IntVal(x) for x in coord))
                        substitutions = [(scalarizer.input(key)(i), z3.RealVal(str(value)))
                                         for key, values in inputs.items() for i, value in enumerate(values)]
                        actual = exact_fraction(z3.simplify(z3.substitute(expr, *substitutions)))
                        address = 0
                        for i, n in zip(coord, kernel.shape):
                            address = address * n + i
                        self.assertEqual(actual, expected[address])

    def test_coverage_rejects_missing_and_duplicate_writes(self):
        missing = relu_after()
        missing.stores.pop()
        with self.assertRaisesRegex(ValueError, "Incomplete"):
            missing.coverage()
        duplicate = relu_before()
        duplicate.stores.append(duplicate.stores[0])
        with self.assertRaisesRegex(ValueError, "Multiple"):
            duplicate.coverage()

    def test_masks_and_unsupported_ops_are_not_silently_assumed(self):
        scalarizer = Scalarizer()
        padded = load("X", (4,), lambda i: i, lambda i: i < 3, other=0)
        self.assertEqual(exact_fraction(scalarizer.at(padded, (3,))), 0)
        undefined = load("X", (4,), lambda i: i, lambda i: i < 3)
        self.assertIn("masked_undefined", str(scalarizer.at(undefined, (3,))))
        with self.assertRaisesRegex(ValueError, "Unsupported"):
            scalarizer.at(Tensor((1,), "exp", (full((1,), 0),)), (0,))

    def test_unknown_is_not_success(self):
        with tempfile.TemporaryDirectory() as directory:
            with patch.object(z3.Solver, "check", return_value=z3.unknown), \
                 patch.object(z3.Solver, "reason_unknown", return_value="injected timeout"):
                result = verify(*comparisons()[0], Path(directory), 1)
        self.assertEqual(result["verdict"], "unknown")
        self.assertEqual(result["reason_unknown"], "injected timeout")
        self.assertFalse(result["passed"])


if __name__ == "__main__":
    unittest.main()

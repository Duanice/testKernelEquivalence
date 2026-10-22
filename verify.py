"""Run four fixed-shape equivalence queries and export independently replayable evidence."""

import argparse
from fractions import Fraction
import hashlib
import json
from pathlib import Path
import platform
import time

import z3

from case_models import comparisons
from replay import replay_pair
from tensor_model import Scalarizer

ROOT = Path(__file__).resolve().parent


def check_sources():
    manifest = json.loads((ROOT / "sources/manifest.json").read_text())
    for item in manifest["files"]:
        actual = hashlib.sha256((ROOT / item["snapshot"]).read_bytes()).hexdigest()
        if actual != item["sha256"]:
            raise ValueError(f"Source snapshot changed: {item['snapshot']}; review the manual model first")
    return manifest


def exact_fraction(value):
    if not z3.is_rational_value(value):
        raise ValueError(f"Cannot replay non-rational witness exactly: {value}")
    return Fraction(value.numerator_as_long(), value.denominator_as_long())


def verify(name, before, after, expected, out_dir, timeout_ms):
    if before.shape != after.shape or before.inputs != after.inputs:
        raise ValueError("Both kernels must share output shape and input buffers")
    coverage = {"before": before.coverage(), "after": after.coverage()}
    scalarizer = Scalarizer()
    indices = tuple(z3.Int(f"out_index_{d}") for d in range(len(before.shape)))
    started = time.perf_counter()
    left = before.output(scalarizer, indices)
    right = after.output(scalarizer, indices)
    encoding_seconds = time.perf_counter() - started
    solver = z3.Solver()
    solver.set(timeout=timeout_ms)
    solver.add(*(z3.And(index >= 0, index < extent) for index, extent in zip(indices, before.shape)))
    solver.add(left != right)
    # Emit the actual query before solver.check, without inserting an expected verdict.
    query = f"; {name}: exact-real model, fixed shape {before.shape}\n"
    query += f"(set-option :timeout {timeout_ms})\n" + solver.to_smt2()
    (out_dir / f"{name}.smt2").write_text(query)
    (out_dir / f"{name}.formulas.txt").write_text(
        "BEFORE (unsimplified scalar formula)\n" + left.sexpr() +
        "\n\nAFTER (unsimplified scalar formula)\n" + right.sexpr() + "\n")
    started = time.perf_counter()
    verdict = solver.check()
    solve_seconds = time.perf_counter() - started
    record = {
        "name": name, "expected": expected, "verdict": str(verdict),
        "passed": str(verdict) == expected,
        "shape": before.shape, "before_config": before.config, "after_config": after.config,
        "coverage": coverage, "encoding_seconds": encoding_seconds,
        "solve_seconds": solve_seconds, "dot_terms_emitted": scalarizer.dot_terms,
        "semantics": "exact reals; all input values at this fixed shape; manually extracted tensor models",
        "mutation": ({"case2_wrong_relu": "after: maximum -> minimum",
                      "case1_missing_term": "after: A load mask K -> K-1"}.get(name)),
    }
    if verdict == z3.unknown:
        record["reason_unknown"] = solver.reason_unknown()
    elif verdict == z3.sat:
        model = solver.model()
        coord = tuple(model.eval(i, model_completion=True).as_long() for i in indices)
        inputs = {key: [exact_fraction(model.eval(scalarizer.input(key)(i), model_completion=True))
                        for i in range(size)] for key, size in before.inputs.items()}
        before_values, after_values = replay_pair(name, before, after, inputs)
        address = 0
        for i, size in zip(coord, before.shape):
            address = address * size + i
        ref_value, opt_value = before_values[address], after_values[address]
        z3_left = exact_fraction(model.eval(left, model_completion=True))
        z3_right = exact_fraction(model.eval(right, model_completion=True))
        replay_ok = ref_value != opt_value and ref_value == z3_left and opt_value == z3_right
        record["passed"] = record["passed"] and replay_ok
        record["counterexample"] = {
            "coordinate": coord, "before": str(ref_value), "after": str(opt_value),
            "z3_before": str(z3_left), "z3_after": str(z3_right), "exact_replay_passed": replay_ok,
            "inputs": {key: [str(x) for x in values] for key, values in inputs.items()},
        }
        (out_dir / f"{name}.model.txt").write_text(str(model) + "\n")
    return record


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out-dir", type=Path, default=ROOT / "results")
    parser.add_argument("--timeout-ms", type=int, default=10000)
    args = parser.parse_args()
    if args.timeout_ms <= 0:
        parser.error("--timeout-ms must be positive")
    manifest = check_sources()
    args.out_dir.mkdir(parents=True, exist_ok=True)
    # No stale witnesses if a rerun changes SAT into UNKNOWN/UNSAT.
    for name, *_ in comparisons():
        (args.out_dir / f"{name}.model.txt").unlink(missing_ok=True)
    records = []
    for item in comparisons():
        record = verify(*item, args.out_dir, args.timeout_ms)
        records.append(record)
        print(f"{record['name']}: {record['verdict'].upper()} "
              f"({record['solve_seconds']:.6f}s), passed={record['passed']}", flush=True)
    report = {
        "python": platform.python_version(), "platform": platform.platform(),
        "z3": z3.get_version_string(), "timeout_ms": args.timeout_ms,
        "source_repo_head": manifest["source_repo_head"],
        "implementation_sha256": {name: hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
                                  for name in ["tensor_model.py", "case_models.py", "verify.py", "replay.py",
                                               "sources/manifest.json"]},
        "all_passed": all(row["passed"] for row in records), "comparisons": records,
    }
    (args.out_dir / "results.json").write_text(json.dumps(report, indent=2, ensure_ascii=False) + "\n")
    lines = ["# 实际验证结果", "",
             f"Python {report['python']}；Z3 {report['z3']}；每次查询限时 {args.timeout_ms} ms。", "",
             "以下结果属于人工提取的张量模型：固定形状、实数语义、完整归约展开。未执行 Triton/GPU。", "",
             "| 查询 | 实际返回 | 求解耗时 / s | 反例精确回放 |", "|---|---|---:|---|"]
    for row in records:
        witness = row.get("counterexample")
        replay = str(witness["exact_replay_passed"]) if witness else "—"
        lines.append(f"| {row['name']} | {row['verdict'].upper()} | {row['solve_seconds']:.6f} | {replay} |")
        if row["verdict"] == "unknown":
            lines.append(f"\nUNKNOWN 原因：{row['reason_unknown']}\n")
    for row in records:
        if "counterexample" in row:
            witness = row["counterexample"]
            lines += ["", f"**{row['name']}**：输出位置 `{witness['coordinate']}`，"
                      f"before = `{witness['before']}`，after = `{witness['after']}`。",
                      f"完整输入在 [results.json](results.json)，求解器模型在 [{row['name']}.model.txt]({row['name']}.model.txt)。"]
    lines += ["", "耗时是本次机器上的观测值，不是性能基准；UNSAT 是求解结果，没有导出独立可检查的证明证书。", ""]
    (args.out_dir / "REPORT.md").write_text("\n".join(lines))
    return 0 if report["all_passed"] else 1


if __name__ == "__main__":
    raise SystemExit(main())

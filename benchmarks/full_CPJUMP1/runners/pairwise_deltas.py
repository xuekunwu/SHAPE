#!/usr/bin/env python3
"""Paired differences between runs of one full_CPJUMP1 experiment, on their saved bootstrap draws.

Every run of an experiment is scored on the same evaluation units with the same draws
(evaluation/full_metrics_draws.npz), so the difference of two runs is a paired comparison with its own standard
error: the shared verdict rule (benchmarks/_shared/full_metrics.verdict) with the second run in the role of the
reference. The verdict names are the contract's (GAIN / SOURCE_SHORTCUT_SUSPECTED / LOSS / NO_RESOLVABLE_DIFFERENCE);
read them as "A differs from B", not as a gain over a control. Runs that differ in code or reasoning settings are
descriptive pairs, not a strict comparison.

    python pairwise_deltas.py --run <experiment dir> --pair shape_rerun_e2e_21f63d2b:shape --out pairwise.json

Writes <out> (JSON) and <out stem>.md. Needs the repo on PYTHONPATH (benchmarks._shared)."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from benchmarks._shared import full_metrics as fm


def pair(run: Path, a: str, b: str) -> dict:
    """Run `a` against run `b`: per verdict metric the main and guard values of both, the paired delta, its SE, its
    95% interval, the guard delta and the verdict."""
    draws = {name: fm.load(run / name / "evaluation" / "full_metrics_draws.npz") for name in (a, b)}
    rows = {}
    for metric in fm.METRICS:
        v = fm.verdict(draws[a], draws[b], metric)
        key = f"{fm.SPLITS[0]}/{fm._metric_key(metric)}"
        rows[metric] = {"a_main": float(draws[a]["points"][key]), "b_main": float(draws[b]["points"][key]),
                        "delta": float(v["delta"]), "se": float(v["se"]), "ci95": [float(x) for x in v["ci95"]],
                        "guard_delta": float(v["guard_delta"]), "verdict": v["verdict"]}
    return {"a": a, "b": b, "metrics": rows}


def markdown(pairs: list[dict]) -> str:
    lines = ["# Paired differences between runs (same evaluation units, same bootstrap draws)", "",
             "Read as \"A differs from B\". Runs differ in code or reasoning settings: descriptive, not strict.", "",
             "| A | B | metric | A (main) | B (main) | A - B | SE | 95% CI of A - B | guard A - B | verdict |",
             "| --- | --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for p in pairs:
        for metric, r in p["metrics"].items():
            lines.append(f"| {p['a']} | {p['b']} | {metric} | {r['a_main']:.4f} | {r['b_main']:.4f} | "
                         f"{r['delta']:+.4f} | {r['se']:.4f} | [{r['ci95'][0]:+.4f}, {r['ci95'][1]:+.4f}] | "
                         f"{r['guard_delta']:+.4f} | {r['verdict']} |")
    return "\n".join(lines) + "\n"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[0])
    ap.add_argument("--run", required=True, type=Path, help="the experiment directory holding the run dirs")
    ap.add_argument("--pair", action="append", required=True, metavar="A:B", help="run A against run B (repeatable)")
    ap.add_argument("--out", required=True, type=Path, help="output JSON; a .md with the same stem is written too")
    args = ap.parse_args(argv)
    pairs = []
    for spec in args.pair:
        a, _, b = spec.partition(":")
        if not a or not b:
            ap.error(f"--pair needs A:B, got {spec!r}")
        pairs.append(pair(args.run, a, b))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(pairs, indent=2, sort_keys=True) + "\n", encoding="utf-8", newline="\n")
    args.out.with_suffix(".md").write_text(markdown(pairs), encoding="utf-8", newline="\n")
    print(markdown(pairs))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

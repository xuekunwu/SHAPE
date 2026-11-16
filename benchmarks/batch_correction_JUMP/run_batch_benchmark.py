"""The JUMP batch-correction benchmark: raw plus five comparison arms on one Arevalo scenario.

    python benchmarks/batch_correction_JUMP/run_batch_benchmark.py
        prepare  --base JUMP --output PREP --scenario scenario_N --microscope-config CSV --arevalo-cache DIR
        control  --prep PREP
        suite    --prep PREP --output RUN [--model gpt-5.6-luna] [--max-rounds 8] [--noise-band 0.002] [--cores N]
        report   --prep PREP --output RUN          (rewrite the report of a finished suite)
        shape-only --prep PREP --output RUN [--model gpt-5.6-luna] [--max-rounds 8] [--noise-band 0.002] [--cores N]
                 (SHAPE alone, no gpt_luna/react_luna_harmony -- a model-matrix ablation, not the official
                 6-arm roster; scored into OUT/SHAPE_ONLY_REPORT.json, never BENCHMARK_REPORT.json)
        react    --prep PREP --output OUT [--compare-run RUN] [--model gpt-5.6-luna] [--noise-band 0.002] [--cores N]
                 (the ReAct arm alone on a prep whose control stage has run, scored into OUT/REACT_REPORT.json;
                 with --compare-run, at the per-round cap that run's SHAPE arm recorded)
        effort-sweep --prep PREP --output OUT [--efforts none low medium high xhigh] [--max-tokens 32000]
                 [--timeout 900] [--model gpt-5.6-luna] [--max-rounds 8] [--noise-band 0.002] [--cores N]
                 (the SHAPE arm alone at each reasoning-effort level, scored into OUT/SWEEP_REPORT.json)

The implementation is in runners/ (contract, config, prepare, control, the arms, score, suite, effort_sweep).
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.batch_correction_JUMP.runners.config import DEFAULT  # noqa: E402


def main() -> None:
    p = argparse.ArgumentParser(prog="run_batch_benchmark")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--base", required=True, help="JUMP root containing inputs/")
    pr.add_argument("--output", required=True)
    pr.add_argument("--scenario", default="scenario_2")
    pr.add_argument("--microscope-config", required=True)
    pr.add_argument("--arevalo-cache", required=True, help="resumable preprocessing cache root")
    pr.add_argument("--seed", type=int, default=DEFAULT.split_seed)
    sub.add_parser("control").add_argument("--prep", required=True)
    su = sub.add_parser("suite")
    su.add_argument("--prep", required=True)
    su.add_argument("--output", required=True)
    su.add_argument("--model", default=DEFAULT.model)
    su.add_argument("--max-rounds", type=int, default=DEFAULT.max_rounds)
    su.add_argument("--noise-band", type=float, default=DEFAULT.noise_band)
    su.add_argument("--cores", type=int, default=DEFAULT.cores)
    rp = sub.add_parser("report")
    rp.add_argument("--prep", required=True)
    rp.add_argument("--output", required=True)
    so = sub.add_parser("shape-only")
    so.add_argument("--prep", required=True)
    so.add_argument("--output", required=True)
    so.add_argument("--model", default=DEFAULT.model)
    so.add_argument("--max-rounds", type=int, default=DEFAULT.max_rounds)
    so.add_argument("--noise-band", type=float, default=DEFAULT.noise_band)
    so.add_argument("--cores", type=int, default=DEFAULT.cores)
    ra = sub.add_parser("react")
    ra.add_argument("--prep", required=True)
    ra.add_argument("--output", required=True)
    ra.add_argument("--model", default=DEFAULT.model)
    ra.add_argument("--noise-band", type=float, default=DEFAULT.noise_band)
    ra.add_argument("--cores", type=int, default=DEFAULT.cores)
    ra.add_argument("--compare-run", help="the run compared with: its SHAPE arm's per-round cap is used")
    es = sub.add_parser("effort-sweep")
    es.add_argument("--prep", required=True)
    es.add_argument("--output", required=True)
    es.add_argument("--efforts", nargs="+", default=None, help="levels to run (default: all five)")
    es.add_argument("--max-tokens", type=int, default=None, help="per-reply token cap, reasoning included")
    es.add_argument("--timeout", type=int, default=None, help="per-request timeout in seconds")
    es.add_argument("--model", default=DEFAULT.model)
    es.add_argument("--max-rounds", type=int, default=DEFAULT.max_rounds)
    es.add_argument("--noise-band", type=float, default=DEFAULT.noise_band)
    es.add_argument("--cores", type=int, default=DEFAULT.cores)
    a = p.parse_args()

    if a.cmd == "prepare":
        from benchmarks.batch_correction_JUMP.runners.prepare import prepare

        out = Path(a.output)
        out.mkdir(parents=True, exist_ok=True)
        prepare(Path(a.base), out, scenario=a.scenario, microscope_config=Path(a.microscope_config),
                arevalo_cache=Path(a.arevalo_cache), seed=a.seed)
    elif a.cmd == "control":
        from benchmarks.batch_correction_JUMP.runners.control import control

        control(Path(a.prep))
    elif a.cmd == "suite":
        from benchmarks.batch_correction_JUMP.runners.suite import run_suite

        report = run_suite(Path(a.prep), Path(a.output), model=a.model, max_rounds=a.max_rounds,
                           noise_band=a.noise_band, cores=a.cores)
        print(json.dumps({"report": str(Path(a.output) / "BENCHMARK_REPORT.json"), "roster": report["roster"]}))
    elif a.cmd == "shape-only":
        from benchmarks.batch_correction_JUMP.runners.suite import run_shape_only

        report = run_shape_only(Path(a.prep), Path(a.output), model=a.model, max_rounds=a.max_rounds,
                                noise_band=a.noise_band, cores=a.cores)
        print(json.dumps({"report": str(Path(a.output) / "SHAPE_ONLY_REPORT.json"), "roster": report["roster"]}))
    elif a.cmd == "report":
        from benchmarks.batch_correction_JUMP.runners.suite import write_report

        report = write_report(Path(a.prep), Path(a.output))
        print(json.dumps({"report": str(Path(a.output) / "BENCHMARK_REPORT.json"), "roster": report["roster"]}))
    elif a.cmd == "react":
        from benchmarks.batch_correction_JUMP.runners.llm_arms import run_react_only

        report = run_react_only(Path(a.prep), Path(a.output), model=a.model, noise_band=a.noise_band, cores=a.cores,
                                compare_run=Path(a.compare_run) if a.compare_run else None)
        print(json.dumps({"report": str(Path(a.output) / "REACT_REPORT.json"), "row": report["row"]}, default=str))
    elif a.cmd == "effort-sweep":
        from benchmarks.batch_correction_JUMP.runners import effort_sweep as sweep

        report = sweep.run_sweep(Path(a.prep), Path(a.output), model=a.model, max_rounds=a.max_rounds,
                                 noise_band=a.noise_band, cores=a.cores, efforts=tuple(a.efforts or sweep.EFFORTS),
                                 max_tokens=a.max_tokens or sweep.MAX_TOKENS, timeout=a.timeout or sweep.TIMEOUT_SECONDS)
        print(json.dumps({"report": str(Path(a.output) / "SWEEP_REPORT.json"),
                          "levels": [r["effort"] for r in report["rows"]]}))


if __name__ == "__main__":
    main()

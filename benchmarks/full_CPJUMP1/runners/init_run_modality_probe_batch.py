#!/usr/bin/env python
"""Batch init_run_modality_probe.py for all 7 arms of a ModalityProbe-family dataset. Arm/model pairs mirror
the real CPJUMP1 final/seed2 runs exactly (benchmarks/full_CPJUMP1/results/final/<name>/CONFIG.json: arm,
model) -- same backend + reasoner combination, just pointed at the given dataset's prep instead of CPJUMP1's.

    cd /path/to/SHAPE
    .venv/bin/python benchmarks/full_CPJUMP1/runners/init_run_modality_probe_batch.py \
        --dataset ModalityProbe2 --prep data/CPJUMP1_modality_probe2/prepared \
        --out-root benchmarks/full_CPJUMP1/results/modality_probe2 [--cores N]

Writes <out-root>/<name>/CONFIG.json for each; skips (prints, does not raise) any name whose output dir
already exists, so the script is safe to re-run after partially completing. Then run stages yourself, per
arm: python -m benchmarks.full_CPJUMP1.harness.cli stage segment --prep PREP --run RUN
"""
from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

ASSETS_ROOT = Path("data/CPJUMP1/assets")

# (output name, arm, model) -- matches results/final/<name>/CONFIG.json exactly
RUNS = [
    ("control", "control", None),
    ("biomni_luna", "biomni", "gpt-5.6-luna"),
    ("biomni_sonnet55", "biomni", "claude-sonnet-5-5"),
    ("cc_sonnet55", "claude", "claude-sonnet-5-5"),
    ("codex_luna", "codex", "gpt-5.6-luna"),
    ("shape_luna", "shape", "gpt-5.6-luna"),
    ("shape_sonnet55", "shape", "claude-sonnet-5-5"),
]


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dataset", choices=("ModalityProbe", "ModalityProbe2", "ModalityProbe3"), default="ModalityProbe")
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--out-root", type=Path, required=True)
    ap.add_argument("--cores", type=int, default=8, help="initial CONFIG.json cores; edit per-stage as needed")
    ap.add_argument("--skip", nargs="*", default=[], help="arm output names already handled elsewhere")
    a = ap.parse_args()

    for name, arm, model in RUNS:
        if name in a.skip:
            print(f"[skip] {name} (--skip)")
            continue
        out = a.out_root / name
        if out.exists():
            print(f"[skip] {out} already exists")
            continue
        cmd = [sys.executable, "-m", "benchmarks.full_CPJUMP1.runners.init_run_modality_probe",
               "--arm", arm, "--dataset", a.dataset, "--prep", str(a.prep), "--out", str(out),
               "--assets-root", str(ASSETS_ROOT), "--cores", str(a.cores)]
        if model:
            cmd += ["--model", model]
        print(f"[init] {name}: {' '.join(cmd)}")
        subprocess.run(cmd, check=True)


if __name__ == "__main__":
    main()

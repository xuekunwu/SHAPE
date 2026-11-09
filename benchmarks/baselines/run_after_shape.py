"""Orchestrator: wait for the in-flight SHAPE runs to finish, then run the CC + Codex baselines on
BOTH benchmarks (segmentation + cropper) at iters=10 (agent decides when to stop early).

  CC    = claude-sonnet-5    Codex = gpt-5.6-sol
  seg   -> calls the shared Cellpose-SAM service (--sam-url)   cropper -> uses the GT masks (no SAM)

Runs sequentially so they don't oversubscribe the box or the APIs. Pure IO/subprocess (no GPU) --
best run in a durable shell (it may wait hours for SHAPE to finish):

    python -m benchmarks.baselines.run_after_shape \
        --wait-dir benchmarks/segmentation_NeurIPS22-CellSeg/results/2026-09-19_113551_concurrent \
        --wait-dir benchmarks/cropper_NeurIPS22-CellSeg/results/2026-09-19_125305_concurrent \
        --limit 1000 --iters 10 --sam-url http://gpu-node.example.net:8137
"""
from __future__ import annotations

import argparse
import glob
import os
import subprocess
import sys
import time
from pathlib import Path

_REPO = Path(__file__).resolve().parents[2]


def _done(run_dir: str, target: int) -> bool:
    n = len(glob.glob(os.path.join(run_dir, "images", "*", "metrics.json")))
    return n >= target or bool(glob.glob(os.path.join(run_dir, "*summary*.json"))
                               and n > 0 and not _still_running(run_dir))


def _still_running(run_dir: str) -> bool:
    # a run is "running" if its parent benchmark still has live worker processes; cheap proxy:
    return False  # rely on the metrics-count / summary check


def wait_for(dirs: list[str], target: int, poll: int = 180) -> None:
    if not dirs:
        return
    while True:
        counts = {d: len(glob.glob(os.path.join(d, "images", "*", "metrics.json"))) for d in dirs}
        summ = {d: bool(glob.glob(os.path.join(d, "*summary*.json"))) for d in dirs}
        done = all(counts[d] >= target or summ[d] for d in dirs)
        print(f"[after-shape] waiting for SHAPE: "
              + ", ".join(f"{Path(d).name}={counts[d]}/{target}{'(summary)' if summ[d] else ''}"
                          for d in dirs), flush=True)
        if done:
            print("[after-shape] SHAPE runs complete -> launching baselines.", flush=True)
            return
        time.sleep(poll)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--wait-dir", action="append", default=[],
                    help="SHAPE run dir(s) to wait for (repeatable). Empty = start immediately.")
    ap.add_argument("--wait-target", type=int, default=1000, help="metrics.json count = done.")
    ap.add_argument("--data-root", default="data/NeurIPS22-CellSeg")
    ap.add_argument("--limit", type=int, default=1000)
    ap.add_argument("--iters", type=int, default=10)
    ap.add_argument("--timeout", type=int, default=7200, help="per-iteration agent timeout (s).")
    ap.add_argument("--sam-url", default="http://gpu-node.example.net:8137")
    ap.add_argument("--price-in", type=float, default=None, help="gpt-5.6-sol input $/1M (optional).")
    ap.add_argument("--price-out", type=float, default=None, help="gpt-5.6-sol output $/1M (optional).")
    ap.add_argument("--only", default=None, help="comma filter, e.g. 'claude-code,codex' or 'seg'.")
    args = ap.parse_args()

    wait_for(args.wait_dir, args.wait_target)

    # (agent, model, benchmark, extra flags). seg calls SAM; cropper uses GT masks.
    runs = [
        ("claude-code", "claude-sonnet-5", "segmentation", ["--sam-url", args.sam_url]),
        ("claude-code", "claude-sonnet-5", "cropper", []),
        ("codex", "gpt-5.6-sol", "segmentation", ["--sam-url", args.sam_url]),
        ("codex", "gpt-5.6-sol", "cropper", []),
    ]
    if args.only:
        keys = [k.strip() for k in args.only.split(",")]
        runs = [r for r in runs if any(k in (r[0], r[2], r[2][:3]) for k in keys)]

    for agent, model, bench, extra in runs:
        cmd = [sys.executable, "-m", "benchmarks.baselines.run_baseline",
               "--agent", agent, "--model", model, "--benchmark", bench,
               "--data-root", args.data_root, "--limit", str(args.limit),
               "--iters", str(args.iters), "--timeout", str(args.timeout)] + extra
        if agent == "codex" and args.price_in is not None and args.price_out is not None:
            cmd += ["--price-in", str(args.price_in), "--price-out", str(args.price_out)]
        print(f"\n[after-shape] === {agent} x {bench} ({model}) ===\n  {' '.join(cmd)}", flush=True)
        subprocess.run(cmd, cwd=str(_REPO))
    print("[after-shape] ALL BASELINES DONE.", flush=True)


if __name__ == "__main__":
    main()

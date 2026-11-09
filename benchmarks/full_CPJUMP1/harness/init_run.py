"""Create a CPJUMP1 run dir + CONFIG.json WITHOUT running any stage, so the five stages can
then be driven ONE AT A TIME (`cli stage <name> --prep --run`) on whatever node suits each
(GPU for segment/embed, CPU for crop/batch/cluster). This is exactly the config `cli run`
writes at cli.py:259-272, minus the preflight gate and the monolithic run_stages() call --
the single-stage worker re-checks commit/assets/env per stage anyway.

  python -m benchmarks.full_CPJUMP1.harness.init_run \
      --arm shape --prep PREP --out RUN --assets-root ASSETS [--smoke] [--cores 8] [--model ...]

Then: cli stage segment --prep PREP --run RUN ; cli stage crop ... ; etc.
"""
from __future__ import annotations

import argparse
import math
import time
from pathlib import Path

from .cli import ARMS, ARM_MODELS, DEFAULT_MODEL, SMOKE_POLICY, reasoning_record, shape_policy, smoke_images
from .data import load_prepared, write_json
from .pipeline import CPJUMP1, ssl_epoch_budget


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True, help="fresh run dir")
    ap.add_argument("--assets-root", type=Path, required=True)
    ap.add_argument("--model", default=None)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--budget-hours", type=float, default=11)
    ap.add_argument("--train-cells-per-site", type=int, default=128)
    ap.add_argument("--inference-chunk", type=int, default=2048)
    ap.add_argument("--smoke", action="store_true")
    a = ap.parse_args()

    if a.out.exists():
        raise FileExistsError("use a fresh output directory; partial runs are never reused as evidence")
    if not math.isfinite(a.budget_hours) or a.budget_hours <= 0:
        raise ValueError("budget-hours must be positive and finite")
    model = a.model or ARM_MODELS.get(a.arm, DEFAULT_MODEL)

    rows = load_prepared(a.prep, verify_pixels=True)
    if a.smoke:
        rows = smoke_images(rows, CPJUMP1.usages)
    a.out.mkdir(parents=True)
    (a.out / "logs").mkdir()

    config = {"arm": a.arm, "model": model, "seed": a.seed, "cores": a.cores,
              "deadline": time.time() + a.budget_hours * 3600, "budget_hours": a.budget_hours,
              "train_cells_per_site": a.train_cells_per_site, "inference_chunk": a.inference_chunk,
              "images": [r["image_id"] for r in rows], "smoke": a.smoke,
              "assets_root": str(a.assets_root)}
    config["ssl_epoch_budget"] = ssl_epoch_budget(rows, a.train_cells_per_site, CPJUMP1.usages)
    if a.arm == "shape":
        config["shape_policy"] = shape_policy(a.cores)
    config["reasoning"] = reasoning_record(a.arm)
    if a.smoke:
        config["smoke_policy"] = dict(SMOKE_POLICY)
    write_json(a.out / "CONFIG.json", config)
    print(f"[init] {a.out}/CONFIG.json written: arm={a.arm} images={len(config['images'])} "
          f"smoke={a.smoke} model={model}")
    print(f"[init] now run stages: python -m benchmarks.full_CPJUMP1.harness.cli stage "
          f"<segment|crop|embed|batch|cluster> --prep {a.prep} --run {a.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python
"""init_run.py for the ModalityProbe / ModalityProbe2 datasets: identical to harness/init_run.py except it
loads via modality_probe_data.load_prepared (not the CRISPR-manifest-gated data.load_prepared -- init_run.py
imports that module directly, not through Dataset.loader(), so it cannot be reused unmodified for a
different dataset) and writes config["dataset"] explicitly (absent that key, dataset(config) in pipeline.py
defaults to CPJUMP1, which would then read this dataset's 600 images as if they were the CRISPR 1000-set).

  python -m benchmarks.full_CPJUMP1.runners.init_run_modality_probe \
      --arm shape --dataset ModalityProbe2 --prep data/CPJUMP1_modality_probe2/prepared --out RUN \
      --assets-root ASSETS [--cores 8] [--model ...]

Then: cli stage segment --prep PREP --run RUN ; cli stage crop ... ; etc. (same cli.py, same stages --
both datasets' .cli point at the same benchmarks.full_CPJUMP1.harness.cli module).
"""
from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path

sys.path.insert(0, ".")
from benchmarks.full_CPJUMP1.harness.cli import ARMS, ARM_MODELS, DEFAULT_MODEL, SMOKE_POLICY, reasoning_record, shape_policy, smoke_images
from benchmarks.full_CPJUMP1.harness.data import write_json
from benchmarks.full_CPJUMP1.harness.modality_probe_data import load_prepared
from benchmarks.full_CPJUMP1.harness.pipeline import MODALITY_PROBE, MODALITY_PROBE2, MODALITY_PROBE3, ssl_epoch_budget

SPECS = {"ModalityProbe": MODALITY_PROBE, "ModalityProbe2": MODALITY_PROBE2, "ModalityProbe3": MODALITY_PROBE3}


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--arm", choices=ARMS, required=True)
    ap.add_argument("--dataset", choices=sorted(SPECS), default="ModalityProbe")
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
    spec = SPECS[a.dataset]

    rows = load_prepared(a.prep, verify_pixels=True)
    if a.smoke:
        rows = smoke_images(rows, spec.usages)
    a.out.mkdir(parents=True)
    (a.out / "logs").mkdir()

    config = {"arm": a.arm, "model": model, "seed": a.seed, "cores": a.cores, "dataset": a.dataset,
              "deadline": time.time() + a.budget_hours * 3600, "budget_hours": a.budget_hours,
              "train_cells_per_site": a.train_cells_per_site, "inference_chunk": a.inference_chunk,
              "images": [r["image_id"] for r in rows], "smoke": a.smoke,
              "assets_root": str(a.assets_root)}
    config["ssl_epoch_budget"] = ssl_epoch_budget(rows, a.train_cells_per_site, spec.usages)
    if a.arm == "shape":
        config["shape_policy"] = shape_policy(a.cores)
    config["reasoning"] = reasoning_record(a.arm)
    if a.smoke:
        config["smoke_policy"] = dict(SMOKE_POLICY)
    write_json(a.out / "CONFIG.json", config)
    print(f"[init] {a.out}/CONFIG.json written: arm={a.arm} dataset={a.dataset} images={len(config['images'])} "
          f"smoke={a.smoke} model={model}")
    print(f"[init] now run stages: python -m benchmarks.full_CPJUMP1.harness.cli stage "
          f"<segment|crop|embed|batch|cluster> --prep {a.prep} --run {a.out}")


if __name__ == "__main__":
    main()

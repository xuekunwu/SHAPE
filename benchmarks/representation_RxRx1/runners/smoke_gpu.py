"""Diagnostic real-DINOv3 training and resume; no labels, no well-mAP result."""
from __future__ import annotations

import argparse
import json
import os
import socket
from pathlib import Path

import torch

from benchmarks.representation_RxRx1.runners import control
from benchmarks.representation_RxRx1.runners.epoch_budget import FixedEpochVerifier, shared_budget
from benchmarks.representation_RxRx1.runners.prepare import check_assets, load, outside_repo, tool_input


def smoke_cells(cells):
    """At most two plates, two train/four val wells per plate, eight cells per site."""
    plates = sorted(set(cells.loc[cells.split == "train", "plate"]))[:2]
    indices = []
    for plate in plates:
        for split, n_wells in (("train", 2), ("val", 4)):
            rows = cells[(cells.plate == plate) & (cells.split == split)]
            wells = sorted(set(rows.well))[:n_wells]
            rows = rows[rows.well.isin(wells)]
            indices.extend(rows.groupby(["well", "site"], sort=True).head(8).index)
    subset = cells.loc[sorted(indices)]
    if (subset.split == "train").sum() < 2 or not (subset.split == "val").any():
        raise SystemExit("NOT_READY: smoke needs at least two training cells and validation cells")
    return subset


def run(root: Path, run_dir: Path) -> dict:
    if not torch.cuda.is_available():
        raise SystemExit("NOT_READY: GPU smoke needs CUDA")
    expected = os.environ.get("RXRX_EXPECTED_COMMIT")
    if not expected or control.git_commit() != expected:
        raise SystemExit("INVALID_REMOTE_SYNC: GPU smoke requires the intended commit")
    outside_repo(run_dir)
    if run_dir.exists():
        raise SystemExit("NOT_READY: use a fresh smoke run directory")
    summary, cells = load(root)
    budget = shared_budget(root, summary, cells)
    subset = smoke_cells(cells)
    crop_array, meta = tool_input(summary, subset)
    control.fs_retry(run_dir.mkdir, parents=True)
    record = {"agent": "gpu_smoke", "diagnostic_only": True, "commit": control.git_commit(),
              "expected_commit": expected, "host": socket.gethostname(),
              "slurm_job": os.environ.get("SLURM_JOB_ID"), "gpu": torch.cuda.get_device_name(),
              "training_budget": budget, "smoke_train_cells": int((subset.split == "train").sum()),
              "smoke_val_cells": int((subset.split == "val").sum()), "calls": []}
    control.write_json(run_dir / "SMOKE.json", record)
    checkpoint = None
    for name in ("shape", "react_resume"):
        verifier = FixedEpochVerifier(budget)
        config = verifier.defaults()
        if checkpoint:
            config.update(init_from="parent", init_checkpoint=checkpoint)
        kwargs = {**verifier.trial_params(config), "output_subdir": f"smoke_{name}", "batch_size": 8}
        result = control.analyzer().execute(crop_array=crop_array, cell_metadata=[dict(m) for m in meta],
                                            query_cache_dir=str(run_dir / "cache"), **kwargs)
        epochs = (result.get("training") or {}).get("epochs_trained", 0)
        training_config = result.get("training_config") or {}
        if (result.get("error") or not 0 < epochs <= budget["max_epochs"]
                or training_config.get("max_epochs_hard_cap") != budget["max_epochs"]
                or not result.get("resume_best") or not Path(result["resume_best"]).is_file()
                or not result.get("adata_path") or not Path(result["adata_path"]).is_file()):
            raise SystemExit(f"SMOKE_FAILED: {name}: invalid training/checkpoint or epoch cap: {result.get('error')}")
        if checkpoint and not (training_config.get("resume") or {}).get("epoch", 0) > 0:
            raise SystemExit("SMOKE_FAILED: second call did not resume training state")
        record["calls"].append({"name": name, "kwargs": kwargs,
                               **{k: result.get(k) for k in control.TOOL_FACTS}})
        control.write_json(run_dir / "SMOKE.json", record)
        checkpoint = result["resume_best"]
    (run_dir / "SMOKE_COMPLETE").write_text(expected + "\n", encoding="utf-8")
    print(f"SMOKE_OK commit={expected} max_epochs={budget['max_epochs']} run={run_dir}", flush=True)
    return record


def main():
    from benchmarks.representation_RxRx1.runners.config import DEFAULT, PREP_SETS, BenchmarkConfig

    parser = argparse.ArgumentParser()
    parser.add_argument("--run", required=True)
    parser.add_argument("--cell-type", default=DEFAULT.cell_type)
    parser.add_argument("--prep-set", default=DEFAULT.prep_set, choices=sorted(PREP_SETS))
    args = parser.parse_args()
    if args.cell_type not in PREP_SETS[args.prep_set]:
        parser.error(f"--cell-type {args.cell_type} has no {args.prep_set} prep")
    os.environ.update(check_assets(Path(DEFAULT.output_root)))
    run(BenchmarkConfig(cell_type=args.cell_type, prep_set=args.prep_set).root(), Path(args.run))


if __name__ == "__main__":
    main()

"""Cohort vs per-image segmentation A/B on the SAME CPJUMP1 images.

Answers: does the segmentation cohort (calibration subset -> consensus levers -> the rest run
cohort-mode with no/low LLM) save tokens WITHOUT losing mask quality, vs the per-image path that
runs a full LLM VRR on every image? Runs both the real harness paths (_segmentation_per_image with
the engine on every image, and _segmentation_cohort) on one fixed image subset, snapshots the token
usage around each, and compares the delivered masks head-to-head (cell count, foreground IoU) plus
the recorded admission verdict and call count per image.

Needs a remote SAM pool ($SHAPE_SAM_SERVER_URL) and the LLM gateway (repo .env). No GPU here.

  SHAPE_SAM_SERVER_URL=http://IP:8137,http://IP:8138,... \
  python -m benchmarks.full_CPJUMP1.runners.cohort_probe --n 24 \
     --out benchmarks/full_CPJUMP1/results/cohort_probe
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path

import numpy as np
import tifffile


def token_snapshot() -> dict:
    from shape.engine import usage
    return dict(usage.snapshot())


def run_side(kind: str, prep: Path, run: Path, config: dict, rows: list[dict]):
    """Run one segmentation path; return (delta token snapshot, wall seconds). kind: 'perimage'|'cohort'."""
    from benchmarks.full_CPJUMP1.harness.pipeline import (_segmentation_per_image, _segmentation_cohort, engine_for)
    if run.exists():
        shutil.rmtree(run)
    (run / "logs").mkdir(parents=True)
    engine = engine_for(config, "segment")        # fresh shape vision engine under the stage env
    before, t0 = token_snapshot(), time.perf_counter()
    (_segmentation_per_image if kind == "perimage" else _segmentation_cohort)(prep, run, config, rows, engine)
    wall = time.perf_counter() - t0
    after = token_snapshot()
    delta = {k: int(after.get(k, 0) or 0) - int(before.get(k, 0) or 0)
             for k in ("prompt_tokens", "completion_tokens", "total_tokens", "calls")}
    return delta, round(wall, 1)


def read_records(run: Path) -> dict:
    """Per-image {image_id: {cell_count, n_calls, admission_run, mask}} from the side's records."""
    out = {}
    seg = read_json(run / "segmentation.json")
    masks = {r["image_id"]: r for r in seg}
    for ident, r in masks.items():
        rec = None
        for name in ("SEARCH.json", "COHORT_SEARCH.json"):
            p = run / "segment" / ident / name
            if p.exists():
                rec = read_json(p)
        adm = (rec or {}).get("admission") or {}
        out[ident] = {"cell_count": r.get("n_cells"), "mask": r.get("mask"),
                      "n_calls": (rec or {}).get("n_calls", 0), "admission_run": adm.get("rule") == "RUN"}
    return out


def foreground_iou(a: Path, b: Path) -> float:
    """Binary foreground IoU of two instance masks (instance IDs differ; compare cell-vs-background)."""
    ma = np.asarray(tifffile.imread(a)) > 0
    mb = np.asarray(tifffile.imread(b)) > 0
    inter = int(np.logical_and(ma, mb).sum())
    union = int(np.logical_or(ma, mb).sum())
    return inter / union if union else 1.0


def read_json(p: Path):
    from benchmarks.full_CPJUMP1.harness.data import read_json as rj
    return rj(p)


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", type=Path, default=Path("data/CPJUMP1/prepared"))
    ap.add_argument("--n", type=int, default=24, help="images in the A/B subset")
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/full_CPJUMP1/results/cohort_probe"))
    a = ap.parse_args(argv)
    if not os.environ.get("SHAPE_SAM_SERVER_URL"):
        raise SystemExit("set SHAPE_SAM_SERVER_URL to the remote SAM pool")

    from benchmarks.full_CPJUMP1.harness.cli import load_prepared
    from benchmarks.full_CPJUMP1.harness.pipeline import stage_env
    os.environ.update(stage_env("shape", "segment"))     # SHAPE_REASONING_EFFORT / SHAPE_LLM_TIMEOUT

    rows_all = load_prepared(a.prep.resolve())
    rows = rows_all[:: max(1, len(rows_all) // a.n)][: a.n]     # a fixed spread across the plates
    ids = [r["image_id"] for r in rows]
    config = {"arm": "shape", "model": a.model, "cores": a.cores, "deadline": time.time() + 4 * 3600,
              "images": ids}
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"[cohort-probe] {len(rows)} images | model={a.model} | SAM={os.environ['SHAPE_SAM_SERVER_URL'][:40]}...",
          flush=True)

    per_tok, per_wall = run_side("perimage", a.prep.resolve(), a.out / "perimage", config, rows)
    print(f"[cohort-probe] per-image done: {per_tok['total_tokens']} tok, {per_wall}s", flush=True)
    coh_tok, coh_wall = run_side("cohort", a.prep.resolve(), a.out / "cohort", config, rows)
    print(f"[cohort-probe] cohort done: {coh_tok['total_tokens']} tok, {coh_wall}s", flush=True)

    per, coh = read_records(a.out / "perimage"), read_records(a.out / "cohort")
    per_image = []
    for ident in ids:
        p, c = per.get(ident, {}), coh.get(ident, {})
        iou = foreground_iou(Path(p["mask"]), Path(c["mask"])) if p.get("mask") and c.get("mask") else None
        per_image.append({"image_id": ident, "iou": None if iou is None else round(iou, 4),
                          "cells_perimage": p.get("cell_count"), "cells_cohort": c.get("cell_count"),
                          "calls_perimage": p.get("n_calls"), "calls_cohort": c.get("n_calls"),
                          "adm_perimage": p.get("admission_run"), "adm_cohort": c.get("admission_run")})
    ious = [r["iou"] for r in per_image if r["iou"] is not None]
    cell_rel = [abs(r["cells_perimage"] - r["cells_cohort"]) / max(1, r["cells_perimage"])
                for r in per_image if r["cells_perimage"] and r["cells_cohort"] is not None]
    report = {
        "n_images": len(ids),
        "tokens": {"perimage": per_tok, "cohort": coh_tok,
                   "savings_x": round(per_tok["total_tokens"] / max(1, coh_tok["total_tokens"]), 2),
                   "calls_savings_x": round(per_tok["calls"] / max(1, coh_tok["calls"]), 2)},
        "wall_seconds": {"perimage": per_wall, "cohort": coh_wall},
        "quality": {
            "mask_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
            "mask_iou_median": round(float(np.median(ious)), 4) if ious else None,
            "mask_iou_min": round(float(np.min(ious)), 4) if ious else None,
            "cell_count_rel_diff_mean": round(float(np.mean(cell_rel)), 4) if cell_rel else None,
            "admission_run_perimage": round(np.mean([r["adm_perimage"] for r in per_image]), 4),
            "admission_run_cohort": round(np.mean([r["adm_cohort"] for r in per_image]), 4),
        },
        "per_image": per_image,
    }
    out = a.out / "report.json"
    json.dump(report, open(out, "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2, default=str))
    print(f"[cohort-probe] -> {out}", flush=True)


if __name__ == "__main__":
    main()

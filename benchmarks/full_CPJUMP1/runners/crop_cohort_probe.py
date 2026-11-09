"""Cohort vs per-image CROP A/B on the SAME CPJUMP1 images (the crop analogue of cohort_probe.py).

Reuses an existing run's delivered segmentation masks (so no re-segmentation), then crops the same
image subset two ways: _cropping_per_image (a full crop VRR on every image) and _cropping_cohort
(calibration subset's VRR -> consensus crop levers -> the rest cropped with no LLM). Snapshots token
usage around each and compares the delivered crops head-to-head: retained cell count per image and
crop-box agreement (mean IoU of each matched cell's crop_bbox).

Needs a remote SAM pool ($SHAPE_SAM_SERVER_URL -- segmentation masks are reused but the tool still
constructs) and the LLM gateway (repo .env).

  SHAPE_SAM_SERVER_URL=... python -m benchmarks.full_CPJUMP1.runners.crop_cohort_probe \
     --seg-run benchmarks/full_CPJUMP1/results/shape --n 40 \
     --out benchmarks/full_CPJUMP1/results/crop_cohort_probe
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import time
from pathlib import Path


def token_snapshot() -> dict:
    from shape.engine import usage
    return dict(usage.snapshot())


def box_iou(a, b) -> float:
    """IoU of two [r0, c0, r1, c1] crop boxes."""
    if not a or not b:
        return 0.0
    ar0, ac0, ar1, ac1 = a
    br0, bc0, br1, bc1 = b
    ir0, ic0, ir1, ic1 = max(ar0, br0), max(ac0, bc0), min(ar1, br1), min(ac1, bc1)
    inter = max(0, ir1 - ir0) * max(0, ic1 - ic0)
    ua = (ar1 - ar0) * (ac1 - ac0) + (br1 - br0) * (bc1 - bc0) - inter
    return inter / ua if ua else 1.0


def run_side(kind: str, prep: Path, run: Path, config: dict, rows: list[dict], seg_rows: list[dict]):
    from benchmarks.full_CPJUMP1.harness.data import read_json, write_json
    from benchmarks.full_CPJUMP1.harness.pipeline import _cropping_per_image, _cropping_cohort, engine_for
    tok_file = run / "_probe_tokens.json"
    if (run / "cells.json").exists() and tok_file.exists():   # reuse a finished side (the per-image side is slow)
        saved = read_json(tok_file)
        print(f"[crop-probe] reusing existing {kind} side ({saved['tokens']['total_tokens']} tok)", flush=True)
        return saved["tokens"], saved["wall"]
    if run.exists():
        shutil.rmtree(run)
    (run / "logs").mkdir(parents=True)
    write_json(run / "segmentation.json", seg_rows)       # reuse the existing delivered masks (absolute paths)
    before, t0 = token_snapshot(), time.perf_counter()
    if kind == "perimage":
        _cropping_per_image(prep, run, config, rows)
    else:
        _cropping_cohort(prep, run, config, rows, engine_for(config, "crop"))
    wall = round(time.perf_counter() - t0, 1)
    after = token_snapshot()
    delta = {k: int(after.get(k, 0) or 0) - int(before.get(k, 0) or 0)
             for k in ("prompt_tokens", "completion_tokens", "total_tokens", "calls")}
    write_json(tok_file, {"tokens": delta, "wall": wall})
    return delta, wall


def cells_by_image(run: Path) -> dict:
    from benchmarks.full_CPJUMP1.harness.data import read_json
    out: dict[str, dict] = {}
    for c in read_json(run / "cells.json"):
        out.setdefault(c["image_id"], {})[c["cell_id"]] = c.get("crop_bbox")
    return out


def main(argv=None) -> None:
    import numpy as np
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", type=Path, default=Path("data/CPJUMP1/prepared"))
    ap.add_argument("--seg-run", type=Path, default=Path("benchmarks/full_CPJUMP1/results/shape"),
                    help="a run whose segmentation.json + masks to reuse")
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--model", default="gpt-5.6-luna")
    ap.add_argument("--cores", type=int, default=8)
    ap.add_argument("--out", type=Path, default=Path("benchmarks/full_CPJUMP1/results/crop_cohort_probe"))
    a = ap.parse_args(argv)
    if not os.environ.get("SHAPE_SAM_SERVER_URL"):
        raise SystemExit("set SHAPE_SAM_SERVER_URL to the remote SAM pool")

    from benchmarks.full_CPJUMP1.harness.cli import load_prepared
    from benchmarks.full_CPJUMP1.harness.data import read_json
    from benchmarks.full_CPJUMP1.harness.pipeline import stage_env
    os.environ.update(stage_env("shape", "crop"))

    seg = {r["image_id"]: r for r in read_json(a.seg_run / "segmentation.json")}
    rows_all = [r for r in load_prepared(a.prep.resolve()) if r["image_id"] in seg]
    rows = rows_all[:: max(1, len(rows_all) // a.n)][: a.n]       # a fixed spread
    ids = [r["image_id"] for r in rows]
    seg_rows = [seg[i] for i in ids]
    config = {"arm": "shape", "model": a.model, "cores": a.cores, "deadline": time.time() + 4 * 3600,
              "images": ids}
    a.out.mkdir(parents=True, exist_ok=True)
    print(f"[crop-probe] {len(rows)} images | model={a.model}", flush=True)

    per_tok, per_wall = run_side("perimage", a.prep.resolve(), a.out / "perimage", config, rows, seg_rows)
    print(f"[crop-probe] per-image done: {per_tok['total_tokens']} tok, {per_wall}s", flush=True)
    coh_tok, coh_wall = run_side("cohort", a.prep.resolve(), a.out / "cohort", config, rows, seg_rows)
    print(f"[crop-probe] cohort done: {coh_tok['total_tokens']} tok, {coh_wall}s", flush=True)

    per, coh = cells_by_image(a.out / "perimage"), cells_by_image(a.out / "cohort")
    per_image, ious, count_rel = [], [], []
    for ident in ids:
        pc, cc = per.get(ident, {}), coh.get(ident, {})
        shared = set(pc) & set(cc)
        box_ious = [box_iou(pc[k], cc[k]) for k in shared]
        iou = float(np.mean(box_ious)) if box_ious else None
        if iou is not None:
            ious.append(iou)
        if pc:
            count_rel.append(abs(len(pc) - len(cc)) / max(1, len(pc)))
        per_image.append({"image_id": ident, "cells_perimage": len(pc), "cells_cohort": len(cc),
                          "box_iou_mean": None if iou is None else round(iou, 4)})
    cohort_rec = read_json(a.out / "cohort" / "crop" / "COHORT_RECORD.json")
    report = {
        "n_images": len(ids),
        "tokens": {"perimage": per_tok, "cohort": coh_tok,
                   "savings_x": round(per_tok["total_tokens"] / max(1, coh_tok["total_tokens"]), 2),
                   "calls_savings_x": round(per_tok["calls"] / max(1, coh_tok["calls"]), 2)},
        "wall_seconds": {"perimage": per_wall, "cohort": coh_wall},
        "cohort_counts": cohort_rec.get("counts"),
        "quality": {
            "crop_box_iou_mean": round(float(np.mean(ious)), 4) if ious else None,
            "crop_box_iou_median": round(float(np.median(ious)), 4) if ious else None,
            "crop_box_iou_min": round(float(np.min(ious)), 4) if ious else None,
            "cell_count_rel_diff_mean": round(float(np.mean(count_rel)), 4) if count_rel else None,
        },
        "per_image": per_image,
    }
    json.dump(report, open(a.out / "report.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "per_image"}, indent=2, default=str))
    print(f"[crop-probe] -> {a.out}/report.json", flush=True)


if __name__ == "__main__":
    main()

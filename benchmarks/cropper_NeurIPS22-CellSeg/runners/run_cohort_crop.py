"""Per-image crop VLRR vs the e2e CROP COHORT, on the same NeurIPS22-CellSeg GT-masked cells.

Two arms on one fixed image set, same GT masks, same shared start margin, same engine:

  per_image : every image runs its own crop VLRR (run_shape_crop.run_shape_arm -- CropVerifier +
              StepVerifier) and delivers its own tuned margin. This is the cropper benchmark's
              `shape` arm and stands in for the cs-cm per-image crop (its crop_trajectory is a
              post-hoc best-observed re-selection ON TOP of exactly this kind of search).
  cohort    : the e2e crop-cohort idea -- a calibration subset runs the full per-image VLRR, the
              CONSENSUS of their delivered margins is applied to every remaining image in ONE no-LLM
              cropper call (the crop analogue of _cropping_cohort; here the single tuned lever is the
              margin, so the consensus is the median delivered margin).

Readouts are the benchmark's own kept-only mechanism metrics (run_shape_crop._qc_values_for_result):
neighbor_contamination (lower better) and target_retention (higher better), plus tokens and wall time
per image. The question: does the cohort keep the per-image quality while spending far fewer tokens?

  SHAPE_SAM_SERVER_URL not needed (crop is CPU + the triage LLM). Needs the LLM gateway (repo .env).
  python -m ... is awkward (hyphenated dir); run by path:
    python benchmarks/cropper_NeurIPS22-CellSeg/runners/run_cohort_crop.py --n 24 --calib 6 \
       --out benchmarks/cropper_NeurIPS22-CellSeg/results/cohort_ab
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
_REPO = HERE.parents[2]
sys.path.insert(0, str(_REPO))

import numpy as np  # noqa: E402

import run_shape_crop as rsc  # noqa: E402  -- sibling module (hyphenated dir)
from pairing import _find_image_mask_pairs  # noqa: E402
from shape.agent.verification.vision_engine import build_vision_engine  # noqa: E402
from shape.engine import usage as token_usage  # noqa: E402
from shape.tools.single_cell_cropper.tool import SEGMENTER_DEFAULT_MIN_SIZE, Single_Cell_Cropper_Tool  # noqa: E402

NAIVE_MARGIN_FRAC = 0.20          # the shared fair-start margin both arms begin from (run_shape_crop)
INITIAL_MARGIN = 0                # run_shape_arm's starting padding (px); the VLRR tunes it


def _readouts(result: dict) -> dict:
    """The benchmark's kept-only mechanism readouts of one delivered crop result."""
    cont = rsc._qc_values_for_result(result, "neighbor_contamination", kept_only=True)
    ret = rsc._qc_values_for_result(result, "target_retention", kept_only=True)
    return {"n_crops": len(ret),
            "contamination_mean": float(np.mean(cont)) if cont else None,
            "retention_mean": float(np.mean(ret)) if ret else None,
            "retention_min": float(np.min(ret)) if ret else None}


def _delivered_margin(resolution) -> float:
    """The margin (padding px) the per-image VLRR delivered -- its last trial's params."""
    try:
        return float(resolution.trials[-1][0].get("padding", INITIAL_MARGIN))
    except Exception:
        return float(INITIAL_MARGIN)


def _per_image(tool, img, mask, out_dir, engine) -> tuple[dict, float, float]:
    """One image's full crop VLRR; returns (readouts, delivered_margin, seconds). Tokens are measured
    per-ARM (a global counter the concurrent workers share; per-image deltas would race)."""
    t0 = time.perf_counter()
    resolution = rsc.run_shape_arm(tool, img, mask, out_dir, SEGMENTER_DEFAULT_MIN_SIZE,
                                   INITIAL_MARGIN, NAIVE_MARGIN_FRAC, engine)
    return _readouts(rsc._converged_result(resolution)), _delivered_margin(resolution), time.perf_counter() - t0


def _cohort_call(tool, img, mask, out_dir, margin: float) -> tuple[dict, float]:
    """One image cropped at the consensus margin, NO LLM; returns (readouts, seconds)."""
    t0 = time.perf_counter()
    result = tool.execute(original_image=str(img), nuclei_mask=str(mask), source_image_id=img.stem,
                          group="default", min_area=SEGMENTER_DEFAULT_MIN_SIZE, output_format="png",
                          margin=int(round(margin)), margin_frac=NAIVE_MARGIN_FRAC,
                          margin_mode="fraction", recenter=False, query_cache_dir=str(out_dir))
    return _readouts(result), time.perf_counter() - t0


def _agg(records: list[dict], tokens_total: int, wall: float) -> dict:
    def m(key):
        xs = [r[key] for r in records if r.get(key) is not None]
        return round(statistics.mean(xs), 4) if xs else None
    return {"n_images": len(records), "contamination_mean": m("contamination_mean"),
            "retention_mean": m("retention_mean"), "retention_min": m("retention_min"),
            "tokens_total": tokens_total,
            "tokens_per_image": round(tokens_total / max(1, len(records)), 1),
            "wall_seconds": round(wall, 1)}


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data-dir", type=Path, default=Path("data/NeurIPS22-CellSeg"))
    ap.add_argument("--label-suffix", default="_label")
    ap.add_argument("--label-ext", default=".tiff")
    ap.add_argument("--n", type=int, default=24, help="images in the A/B set")
    ap.add_argument("--calib", type=int, default=6, help="cohort calibration subset size")
    ap.add_argument("--workers", type=int, default=16, help="concurrent crop VLRR workers (LLM-bound)")
    ap.add_argument("--model", default=None, help="triage/VLRR LLM (default: $LLM_MODEL or gpt-5.6-luna)")
    ap.add_argument("--out", type=Path, default=Path("benchmarks/cropper_NeurIPS22-CellSeg/results/cohort_ab"))
    a = ap.parse_args(argv)

    import os
    model = a.model or os.environ.get("LLM_MODEL") or "gpt-5.6-luna"
    pairs = _find_image_mask_pairs(a.data_dir, a.label_suffix, a.label_ext)[: a.n]
    if len(pairs) < a.calib + 1:
        raise SystemExit(f"only {len(pairs)} image/mask pairs; need > --calib {a.calib}")
    a.out.mkdir(parents=True, exist_ok=True)
    engine = build_vision_engine(model)
    tool = Single_Cell_Cropper_Tool()
    print(f"[cohort-ab] {len(pairs)} images | calib {a.calib} | workers {a.workers} | model {model}", flush=True)

    # ARM 1 -- per-image VLRR on every image, concurrently (LLM-bound). Tokens measured per-ARM.
    from concurrent.futures import ThreadPoolExecutor
    done = [0]
    def one_perimage(i_img_mask):
        i, img, mask = i_img_mask
        ro, margin, sec = _per_image(tool, img, mask, a.out / "per_image" / img.stem, engine)
        done[0] += 1
        print(f"[per_image] {done[0]}/{len(pairs)} {img.stem} margin={margin:.1f} "
              f"cont={ro['contamination_mean']} ret={ro['retention_mean']}", flush=True)
        return {**ro, "margin": margin, "seconds": sec, "image": img.stem}
    tok0, t0 = token_usage.total_tokens(), time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        per_image = list(pool.map(one_perimage, [(i, p[0], p[1]) for i, p in enumerate(pairs)]))
    per_image_tokens, per_image_wall = token_usage.total_tokens() - tok0, time.perf_counter() - t0

    # ARM 2 -- cohort: calibration images reuse their per-image result; consensus margin for the rest.
    consensus_margin = float(statistics.median([per_image[i]["margin"] for i in range(a.calib)]))
    print(f"[cohort] consensus margin (median of {a.calib} calibration VLRRs) = {consensus_margin:.1f}", flush=True)
    # the calibration subset's cost is its share of ARM-1's tokens (it IS the per-image VLRR on those).
    cohort_tokens = round(per_image_tokens * a.calib / max(1, len(pairs)))
    cdone = [0]
    def one_cohort(i_img_mask):
        i, img, mask = i_img_mask
        if i < a.calib:
            rec = {k: per_image[i][k] for k in ("contamination_mean", "retention_mean", "retention_min", "image")}
            rec["seconds"] = per_image[i]["seconds"]
        else:
            ro, sec = _cohort_call(tool, img, mask, a.out / "cohort" / img.stem, consensus_margin)
            rec = {**ro, "seconds": sec, "image": img.stem}
        cdone[0] += 1
        return rec
    tc0 = time.perf_counter()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        cohort = list(pool.map(one_cohort, [(i, p[0], p[1]) for i, p in enumerate(pairs)]))
    cohort_wall = per_image_wall * a.calib / max(1, len(pairs)) + (time.perf_counter() - tc0)

    pa, co = _agg(per_image, per_image_tokens, per_image_wall), _agg(cohort, cohort_tokens, cohort_wall)
    report = {"model": model, "n_images": len(pairs), "calibration": a.calib,
              "consensus_margin": consensus_margin, "per_image": pa, "cohort": co,
              "token_savings_x": round(pa["tokens_total"] / max(1, co["tokens_total"]), 2),
              "per_image_records": per_image, "cohort_records": cohort}
    json.dump(report, open(a.out / "report.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if not k.endswith("_records")}, indent=2, default=str))
    print(f"[cohort-ab] -> {a.out}/report.json", flush=True)


if __name__ == "__main__":
    main()

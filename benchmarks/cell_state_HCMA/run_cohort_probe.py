"""Cohort-mode token/time probe: segment a BATCH of HCMA images in ONE solve() call so the
segmentation cohort fires (calibration on a few images -> consensus levers -> the rest run
cohort-mode with NO agent round / NO LLM). Contrasts with the per-image benchmark runner,
which calls solve() once per image (len(specs)==1) and so NEVER triggers the cohort, paying
a full VLRR + vision-judge loop on every image.

Records, from solve()'s cohort record + a global token snapshot:
  * parameter_source counts: how many images ran calibration / cohort / per_image_vrr
  * total tokens + wall time, and (approx) the share spent on the calibrated/routed images
    (the only ones that call the LLM) vs the free cohort images.

Needs a running SAM service (GPU) -- set SHAPE_SAM_SERVER_URL to it.

  SHAPE_SAM_SERVER_URL=http://vm-host.example.net:8137,http://vm-host.example.net:8138 \
  python -m benchmarks.cell_state_HCMA.run_cohort_probe --n 100 --image-workers 16 \
     --out-dir benchmarks/cell_state_HCMA/results/cohort_probe
"""
from __future__ import annotations

import argparse, json, os, time
from pathlib import Path

import pandas as pd


def _token_snapshot():
    from shape.engine import usage as token_usage
    return token_usage.snapshot()


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--manifest", default="benchmarks/cell_state_HCMA/results/hcma_configA_manifest.parquet",
                    help="Config-A image manifest (image_path column)")
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--image-workers", type=int, default=16)
    ap.add_argument("--max-steps", type=int, default=8)
    ap.add_argument("--model", default=None, help="LLM (default: $LLM_MODEL or gpt-5.6-luna)")
    ap.add_argument("--no-cohort", action="store_true",
                    help="disable the cohort (per-image VRR) for an A/B token comparison")
    ap.add_argument("--out-dir", default="benchmarks/cell_state_HCMA/results/cohort_probe")
    a = ap.parse_args()
    os.makedirs(a.out_dir, exist_ok=True)

    if not os.environ.get("SHAPE_SAM_SERVER_URL"):
        raise SystemExit("set SHAPE_SAM_SERVER_URL to a running SAM service (GPU node)")
    model = a.model or os.environ.get("LLM_MODEL") or "gpt-5.6-luna"

    man = pd.read_parquet(a.manifest)
    paths = man["image_path"].head(a.n).tolist()
    images = [{"path": str(p), "image_id": Path(p).stem, "group": "default"} for p in paths]
    print(f"[cohort-probe] {len(images)} images | cohort={'OFF' if a.no_cohort else 'ON'} | "
          f"image_workers={a.image_workers} | model={model}", flush=True)

    from shape.solver import solve
    snap0 = _token_snapshot(); t0 = time.perf_counter()
    result = solve(
        question="Segment every cell in each image; deliver one instance mask per image.",
        images=images, llm_engine_name=model, api_key=os.environ.get("LLM_API_KEY"),
        max_steps=a.max_steps, query_cache_dir=str(Path(a.out_dir) / "shape_cache"),
        enabled_tools=["Cell_Segmenter_Tool"], image_workers=a.image_workers,
        segmentation_cohort=not a.no_cohort)
    wall = time.perf_counter() - t0
    snap1 = _token_snapshot()

    def d(k): return int(snap1.get(k, 0) or 0) - int(snap0.get(k, 0) or 0)
    tokens = {k: d(k) for k in ("prompt_tokens", "completion_tokens", "total_tokens", "calls")}

    # the cohort record is attached to the segmentation step's verification entry
    cohort = None
    for v in (result.get("verification_summary") or []) if isinstance(result, dict) else []:
        if isinstance(v, dict) and v.get("cohort"):
            cohort = v["cohort"]; break
    counts = (cohort or {}).get("counts")
    # how many images actually hit the LLM = calibration + per_image_vrr (cohort images are free)
    llm_images = (counts.get("calibration", 0) + counts.get("per_image_vrr", 0)) if counts else None

    report = {"n_images": len(images), "cohort_enabled": not a.no_cohort,
              "image_workers": a.image_workers, "model": model,
              "wall_seconds": round(wall, 1), "tokens": tokens,
              "tokens_per_image": {k: round(v / max(1, len(images)), 1) for k, v in tokens.items()},
              "cohort_counts": counts,
              "images_that_called_llm": llm_images,
              "images_free_cohort": (counts.get("cohort", 0) if counts else None),
              "cohort_groups": [{k: g.get(k) for k in ("group", "calibration_size", "routing")}
                                for g in (cohort or {}).get("groups", [])] if cohort else None}
    out = Path(a.out_dir) / ("report_cohort.json" if not a.no_cohort else "report_perimage.json")
    json.dump(report, open(out, "w"), indent=2, default=str)
    print(json.dumps(report, indent=2, default=str))
    print(f"[cohort-probe] -> {out}", flush=True)


if __name__ == "__main__":
    main()

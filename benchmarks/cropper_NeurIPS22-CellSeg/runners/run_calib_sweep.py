"""Calibration-size (calib) scaling curve for the crop cohort, on one image set.

Runs, over the SAME fixed image+GT-mask set:
  - per-image crop VLRR on EVERY image (run_cohort_crop._per_image) -> each image's own tuned,
    triaged crop (contamination/retention) + its delivered margin;
  - one cohort-mode crop on EVERY image at the CONSENSUS margin (median of all delivered margins),
    no LLM, no per-cell triage.
Then it SWEEPS calib = {0, 2, 4, ...} by re-aggregation (free): at calib k the first k images use
their per-image result (they are the calibrated ones) and the rest use their cohort-mode result,
so contamination(k) traces how fast the cohort recovers the per-image quality as calibration grows,
and tokens(k) = per-image-arm tokens x k/N. The curve's SHAPE is the point: if the crop value is a
GLOBAL lever (margin) the consensus transfers it and quality is flat in k (leverage); if it is a
PER-CELL decision (triage) only the k calibrated images improve and quality is linear in k (none).

Image sources (--pairs):
  neurips : data/NeurIPS22-CellSeg/{images,labels} GT masks (pairing._find_image_mask_pairs)
  cpjump  : a full_CPJUMP1 run's segmentation.json -> (prepared image, delivered mask) pairs

  python benchmarks/cropper_NeurIPS22-CellSeg/runners/run_calib_sweep.py --pairs neurips --n 300 \
     --workers 16 --out benchmarks/cropper_NeurIPS22-CellSeg/results/calib_sweep_neurips
"""
from __future__ import annotations

import argparse
import json
import statistics
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(HERE.parents[2]))

import run_cohort_crop as rcc  # noqa: E402  -- reuse _per_image / _cohort_call / _readouts
from shape.engine import usage as token_usage  # noqa: E402
from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool  # noqa: E402


def _pairs_neurips(n: int) -> list[tuple[Path, Path]]:
    from pairing import _find_image_mask_pairs
    return _find_image_mask_pairs(Path("data/NeurIPS22-CellSeg"), "_label", ".tiff")[:n]


def _pairs_cpjump(seg_run: Path, n: int) -> list[tuple[Path, Path]]:
    """(prepared image, delivered mask) pairs from a full_CPJUMP1 run's segmentation.json."""
    from benchmarks.full_CPJUMP1.harness.cli import load_prepared
    from benchmarks.full_CPJUMP1.harness.data import read_json, within
    prep = Path("data/CPJUMP1/prepared")
    rows = {r["image_id"]: r for r in load_prepared(prep)}
    seg = read_json(seg_run / "segmentation.json")
    pairs = []
    for s in seg[:n]:
        row = rows.get(s["image_id"])
        if row and Path(s["mask"]).exists():
            pairs.append((within(prep / "public", row["file"]), Path(s["mask"])))
    return pairs


def main(argv=None) -> None:
    import os
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pairs", choices=("neurips", "cpjump"), required=True)
    ap.add_argument("--seg-run", type=Path, default=Path("benchmarks/full_CPJUMP1/results/shape"),
                    help="--pairs cpjump: the run whose segmentation.json + masks to crop")
    ap.add_argument("--n", type=int, default=300)
    ap.add_argument("--workers", type=int, default=16)
    ap.add_argument("--calibs", default="0,2,4,8,16,28,64,128,256",
                    help="comma-separated calib points to plot")
    ap.add_argument("--model", default=None)
    ap.add_argument("--out", type=Path, required=True)
    a = ap.parse_args(argv)
    model = a.model or os.environ.get("LLM_MODEL") or "gpt-5.6-luna"
    pairs = _pairs_neurips(a.n) if a.pairs == "neurips" else _pairs_cpjump(a.seg_run, a.n)
    if len(pairs) < 2:
        raise SystemExit(f"only {len(pairs)} pairs")
    a.out.mkdir(parents=True, exist_ok=True)
    calibs = sorted({int(c) for c in a.calibs.split(",") if int(c) <= len(pairs)} | {len(pairs)})

    from shape.agent.verification.vision_engine import build_vision_engine
    engine = build_vision_engine(model)
    tool = Single_Cell_Cropper_Tool()
    print(f"[sweep] {a.pairs} {len(pairs)} images | workers {a.workers} | calibs {calibs} | model {model}", flush=True)

    # per-image VLRR on all, concurrent; tokens measured per-arm.
    prog = [0]
    def one_pi(ix):
        i, (img, mask) = ix
        ro, margin, sec = rcc._per_image(tool, img, mask, a.out / "per_image" / img.stem, engine)
        prog[0] += 1
        if prog[0] % 20 == 0:
            print(f"[per_image] {prog[0]}/{len(pairs)}", flush=True)
        return {"i": i, "image": img.stem, "margin": margin, **ro}
    t0, tok0 = time.perf_counter(), token_usage.total_tokens()
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        pi = sorted(pool.map(one_pi, list(enumerate(pairs))), key=lambda r: r["i"])
    pi_tokens, pi_wall = token_usage.total_tokens() - tok0, time.perf_counter() - t0

    # cohort-mode on all, at the consensus margin (median of all delivered margins), no LLM.
    consensus = float(statistics.median([r["margin"] for r in pi]))
    print(f"[sweep] consensus margin (median of all {len(pi)} per-image margins) = {consensus:.2f}", flush=True)
    def one_co(ix):
        i, (img, mask) = ix
        ro, _sec = rcc._cohort_call(tool, img, mask, a.out / "cohort" / img.stem, consensus)
        return {"i": i, **ro}
    with ThreadPoolExecutor(max_workers=a.workers) as pool:
        co = sorted(pool.map(one_co, list(enumerate(pairs))), key=lambda r: r["i"])

    # sweep calib by re-aggregation.
    def agg(vals):
        xs = [v for v in vals if v is not None]
        return round(statistics.mean(xs), 5) if xs else None
    curve = []
    N = len(pairs)
    for k in calibs:
        cont = agg([(pi if i < k else co)[i]["contamination_mean"] for i in range(N)])
        ret = agg([(pi if i < k else co)[i]["retention_mean"] for i in range(N)])
        curve.append({"calib": k, "contamination_mean": cont, "retention_mean": ret,
                      "tokens": round(pi_tokens * k / N), "frac_calibrated": round(k / N, 4)})
    report = {"dataset": a.pairs, "n_images": N, "model": model, "consensus_margin": consensus,
              "per_image_tokens_total": pi_tokens, "per_image_wall_s": round(pi_wall, 1),
              "per_image_contamination": agg([r["contamination_mean"] for r in pi]),
              "cohort0_contamination": agg([r["contamination_mean"] for r in co]),
              "curve": curve}
    json.dump(report, open(a.out / "sweep.json", "w"), indent=2, default=str)
    print(json.dumps({k: v for k, v in report.items() if k != "curve"}, indent=2, default=str))
    print("calib,contamination,retention,tokens")
    for pt in curve:
        print(f"  {pt['calib']},{pt['contamination_mean']},{pt['retention_mean']},{pt['tokens']}")
    print(f"[sweep] -> {a.out}/sweep.json", flush=True)


if __name__ == "__main__":
    main()

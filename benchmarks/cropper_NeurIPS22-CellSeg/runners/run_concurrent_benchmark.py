"""Concurrent cropper benchmark — MaskCrop (baseline) vs SHAPE (agentic VLRR) single-cell
cropping on NeurIPS22-CellSeg, in the SAME shape as segmentation_NeurIPS22-CellSeg's
run_concurrent_benchmark.py: a ProcessPoolExecutor over images, one worker per image, and the
same standard result layout:

  results/<ts>_concurrent/
    run_params.json         — every knob this run used (auto-dumped at launch)
    run_summary.json        — macro crop-quality + efficiency + cost
    cell_image_metrics.csv  — one row per image
    images/<id>/            — per-image crops (maskcrop/ + shape/) + metrics.json

Metric (GT-free crop quality, the cropper's own QC):
  * neighbor_contamination  (LOWER = better; fraction of crop pixels belonging to OTHER cells)
  * target_retention        (HIGHER = better; fraction of the target cell kept in-frame)
The agentic gain is the per-image contamination REDUCTION (maskcrop - shape); the SHAPE arm's
VLRR loop tightens/recenters each crop (and triages bad ones) toward less contamination while
holding retention. GPU is used ONCE (main pre-segments the Cellpose-SAM masks, cached to
--masks-dir); the workers are CPU + the mask-check triage VLM only.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from multiprocessing import get_context
from pathlib import Path

_THIS_DIR = Path(__file__).resolve().parent
_REPO = _THIS_DIR.parents[2]


def _load_local_config():
    import importlib.util
    spec = importlib.util.spec_from_file_location("_cropper_cfg", _THIS_DIR / "cropper_config.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.BenchmarkConfig()


# --------------------------------------------------------------------------- #
# crop-quality extraction from a tool result's on-disk metadata
# --------------------------------------------------------------------------- #
def _read_crop_metadata(result: dict) -> list:
    """The final crop list (each with a `qc` dict) for a tool.execute() result."""
    import json as _json
    p = (result or {}).get("cell_crops_metadata_path") if isinstance(result, dict) else None
    if not p or not os.path.exists(p):
        # fall back to the inline metadata if the tool returned it
        cm = (result or {}).get("cell_metadata") if isinstance(result, dict) else None
        return cm or []
    try:
        d = _json.load(open(p))
    except Exception:
        return []
    return d if isinstance(d, list) else (d.get("cell_metadata") or [])


def _qc_means(crops: list) -> dict:
    """Mean of each GT-free QC field over the KEPT crops (a crop the triage marked
    vision_reject=True is dropped from the population)."""
    kept = [c for c in crops if not c.get("vision_reject")]
    out = {"n_crops": len(crops), "n_kept": len(kept), "n_rejected": len(crops) - len(kept)}
    for k in ("neighbor_contamination", "target_retention", "occupancy",
              "target_dominance", "single_object", "mask_containment"):
        vals = [c["qc"][k] for c in kept if c.get("qc") and k in c["qc"]]
        out[k] = float(sum(vals) / len(vals)) if vals else None
    return out


def _run_shape_arm_capped(tool, image_path, mask_path, out_dir, min_area, initial_margin,
                          naive_margin_frac, engine, montage_grid, max_iters):
    """SHAPE agentic crop arm with COST KNOBS: a bigger triage montage grid (more crops per
    VLM montage -> fewer montage calls, the dominant token cost on crowded images) and a lower
    VLRR iteration cap. Mirrors run_shape_crop.run_shape_arm otherwise."""
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.agent.verification.verifiers.crop import CropVerifier, adapt_crop

    def tool_runner(params):
        return tool.execute(
            original_image=str(image_path), nuclei_mask=str(mask_path),
            min_area=min_area, margin=params["padding"], margin_frac=naive_margin_frac,
            margin_mode="fraction", recenter=False, query_cache_dir=str(out_dir),
            cell_overrides=params.get("cell_overrides", {}))

    initial_params = {"padding": float(initial_margin), "cell_overrides": {}}
    initial_result = tool_runner(initial_params)
    verifier = CropVerifier()
    verifier.llm_engine = engine
    verifier.triage_montage_grid = tuple(montage_grid)
    return StepVerifier(max_iters=max_iters).resolve_stage(
        verifier=verifier, adapter=adapt_crop, tool_runner=tool_runner,
        initial_params=initial_params, initial_result=initial_result)


def _crop_iterations(tc):
    """Ordered VLRR iterations from the on-disk tool_cache. A metadata file and its triage file
    share a hash suffix (``..._<hash>.json``); pair them by that hash and order by write time so
    step N in the trace is the Nth crop->triage round (glob-sort would order by hash, not time)."""
    import glob as _glob
    import re as _re
    meta, tri = {}, {}
    for mp in _glob.glob(str(tc / "cell_crops_metadata_*.json")):
        m = _re.search(r"cell_crops_metadata_([0-9a-fA-F]+)\.json$", mp)
        if m:
            meta[m.group(1)] = mp
    for tp in _glob.glob(str(tc / "cell_crops_triage_trace_*.json")):
        m = _re.search(r"cell_crops_triage_trace_([0-9a-fA-F]+)\.json$", tp)
        if m:
            tri[m.group(1)] = tp
    order = sorted(meta, key=lambda h: os.path.getmtime(meta[h]))
    return [(meta[h], tri.get(h)) for h in order]


def _build_crop_trace(out_dir, resolution, image_id, final_metrics) -> dict:
    """Per-image AGENTIC trace of the crop VLRR loop, in the SAME schema as segmentation's
    (``shape.neurips.concurrent.trace.v1``) so both benchmarks' traces read uniformly. The crop
    loop is MaskCrop -> CropVerifier(triage VLM) -> retune -> rerun. Each step carries REAL
    per-module input & output (mirroring segmentation's planner/executor/verifier detail), all
    reconstructed from the on-disk tool_cache: the executor = the cropper tool (input: the crop
    config that ran; output: per-crop QC), the verifier = the triage screen + VLM (input: what it
    judged; output: per-crop keep/reject verdicts with reasons)."""
    def _load(p):
        try:
            return json.load(open(p)) if p else None
        except Exception:
            return None

    def _mean(xs):
        xs = [x for x in xs if isinstance(x, (int, float))]
        return round(sum(xs) / len(xs), 4) if xs else None

    tc = Path(out_dir) / "tool_cache"
    steps, trajectory = [], []
    total_judged = total_rejected = 0
    for i, (mp, tp) in enumerate(_crop_iterations(tc)):
        crops = _load(mp) or []
        if isinstance(crops, dict):
            crops = crops.get("cell_metadata", [])
        qc = lambda c: c.get("qc") or {}  # noqa: E731
        c0 = crops[0] if crops else {}
        # executor INPUT: the crop configuration this iteration ran with
        exec_in = {"n_cells": len(crops), "margin_mode": c0.get("margin_mode"),
                   "recenter": c0.get("recenter"),
                   "mean_inset_px": _mean([qc(c).get("contamination_inset_px") for c in crops])}
        # executor OUTPUT: real per-crop QC + aggregates (the tool's result for this iteration)
        exec_out = {
            "n_crops": len(crops),
            "mean_neighbor_contamination": _mean([qc(c).get("neighbor_contamination") for c in crops]),
            "mean_target_retention": _mean([qc(c).get("target_retention") for c in crops]),
            "mean_occupancy": _mean([qc(c).get("occupancy") for c in crops]),
            "per_crop_qc": [{"cell_id": c.get("cell_id"),
                             "neighbor_contamination": qc(c).get("neighbor_contamination"),
                             "target_retention": qc(c).get("target_retention"),
                             "occupancy": qc(c).get("occupancy")} for c in crops],
        }
        # verifier INPUT/OUTPUT: the triage screen + VLM montage, and its per-crop verdicts
        verdicts = (_load(tp) or {}).get("verdicts", []) if tp else []
        montage = next((v.get("mask_check", {}).get("montage_path") for v in verdicts
                        if v.get("mask_check", {}).get("montage_path")), None)
        ver_in = {"n_judged": len(verdicts),
                  "n_sent_to_vlm": sum(1 for v in verdicts if v.get("mask_check", {}).get("sent_to_vlm")),
                  "montage_path": montage}
        # verifier OUTPUT: per-crop verdict WITH its full decision basis -- the terse `reasoning`
        # string is only a label, so surface the three real stages: the deterministic screen
        # (which arms fired + the QC metrics), the VLM verdict (cell_present/absent + confidence,
        # populated only for crops actually sent to it), and the geometry gate + final decision.
        ver_out = {
            "n_rejected": sum(1 for v in verdicts if v.get("reject")),
            "n_flagged_by_screen": sum(1 for v in verdicts
                                       if (v.get("data_screen") or {}).get("verdict") == "flagged"),
            "verdicts": [{
                "cell_id": v.get("cell_id"),
                "verdict": v.get("verdict"),            # final keep / reject
                "reject": v.get("reject"),
                "problem": v.get("problem"),
                "final": v.get("final"),                # {kept, reason}
                "data_screen": v.get("data_screen"),    # arms_fired + solidity/area/texture metrics
                "vlm": v.get("mask_check"),                  # {sent_to_vlm, verdict, confidence, montage_path}
                "geometry": v.get("geometry"),          # inset_px, neighbor_contamination, retention
                "reasoning": v.get("reasoning"),
            } for v in verdicts],
        }
        total_judged += len(verdicts)
        total_rejected += ver_out["n_rejected"]
        steps.append({"step": i, "tool": "Single_Cell_Cropper_Tool",
                      "modules": {"executor": {"input": exec_in, "output": exec_out},
                                  "verifier": {"input": ver_in, "output": ver_out}}})
        trajectory.append({"iteration": i, "n_crops": len(crops),
                           "mean_contamination": exec_out["mean_neighbor_contamination"],
                           "n_rejected": ver_out["n_rejected"]})

    n_iters = len(steps)
    # Mirror segmentation's `activity` block with crop-equivalent fields (segmenter->cropper).
    activity = {
        "n_steps_executed": n_iters,
        "tool_calls": {"Single_Cell_Cropper_Tool": n_iters},
        "tool_calls_total": n_iters,
        "cropper_calls": n_iters,
        "plan_rejections": 0,
        "verifier_iterations_total": n_iters,
        "verifier_rounds_by_stage": {"crop": n_iters},
        "vision_judgements": total_judged,
        "escalations": 0,
        "crops_retuned": max(0, n_iters - 1),
        "crops_rejected_total": total_rejected,
        "contamination_trajectory": trajectory,
    }

    return {
        "schema": "shape.neurips.concurrent.trace.v1",
        "image_id": image_id,
        "question": ("Crop every segmented cell into a clean single-cell image, minimizing "
                     "neighbor-cell contamination while retaining the target cell."),
        "pipeline": ("MaskCrop -> CropVerifier(triage VLM gate) -> Retune -> Rerun, "
                     "repeated per VLRR iteration"),
        "stop_reason": getattr(resolution, "status", None),
        "n_steps": n_iters,
        "activity": activity,
        "final_metrics": final_metrics,
        "steps": steps,
    }


# --------------------------------------------------------------------------- #
# worker: one image, one process
# --------------------------------------------------------------------------- #
def run_one_image(job: dict) -> dict:
    for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS",
               "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_v] = "1"
    os.environ["SHAPE_FAST_FIGURES"] = "1"

    sys.path.insert(0, str(_THIS_DIR))
    import mask_cache as mq
    import run_shape_crop as rsc
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
    from shape.agent.verification.vision_engine import build_vision_engine
    from shape.engine import usage as token_usage

    image_path = Path(job["image_path"])
    image_id = image_path.stem
    masks_dir = Path(job["masks_dir"])
    img_out = Path(job["run_root"]) / "images" / image_id
    img_out.mkdir(parents=True, exist_ok=True)
    row = {"image_name": image_id, "status": "ok", "error": ""}
    try:
        pred_mask_path = masks_dir / f"{image_id}_pred.tiff"
        if not pred_mask_path.exists():   # pre-caching missed it -> segment now (rare)
            mq._load_or_segment_mask(image_path, masks_dir, {}, job["batch_size"],
                                     job["flow_threshold"], job["cellprob_threshold"])
        tool = Single_Cell_Cropper_Tool()
        engine = build_vision_engine(job["model"], job.get("api_key"))

        t0 = time.perf_counter()
        snap0 = token_usage.snapshot()
        # ARM A: MaskCrop baseline (deterministic, no VLM)
        a_res = rsc.run_maskcrop_arm(tool, image_path, pred_mask_path, img_out / "maskcrop",
                                     job["min_area"], job["margin"], job["maskcrop_margin_frac"])
        a = _qc_means(_read_crop_metadata(a_res))
        # ARM B: SHAPE agentic VLRR crop optimisation (+ in-loop triage VLM), COST-CAPPED
        # (bigger montage grid -> fewer VLM montage calls; fewer VLRR iterations).
        b_reso = _run_shape_arm_capped(
            tool, image_path, pred_mask_path, img_out / "shape",
            job["min_area"], job["margin"], job["maskcrop_margin_frac"], engine,
            job["montage_grid"], job["crop_max_iters"])
        b_res = getattr(b_reso, "result", None) or {}
        b = _qc_means(_read_crop_metadata(b_res))
        # per-image AGENTIC trace of the crop VLRR loop (+ pull the verify/outer-loop
        # counts out of it so the CSV row and the run summary can aggregate them)
        crop_activity = {}
        try:
            _trace = _build_crop_trace(img_out / "shape", b_reso, image_id, b)
            (img_out / "trace.json").write_text(
                json.dumps(_trace, indent=2, default=str), encoding="utf-8")
            crop_activity = _trace.get("activity", {})
        except Exception:
            pass
        runtime = time.perf_counter() - t0
        snap1 = token_usage.snapshot()

        def _d(k):
            return int(snap1.get(k, 0) or 0) - int(snap0.get(k, 0) or 0)
        tokens = _d("total_tokens"); tok_in = _d("input_tokens") or _d("prompt_tokens")
        tok_out = _d("output_tokens") or _d("completion_tokens"); tok_cached = _d("cached_tokens")

        cont_a, cont_b = a.get("neighbor_contamination"), b.get("neighbor_contamination")
        reduction = (cont_a - cont_b) if (cont_a is not None and cont_b is not None) else None
        metrics = {
            "image_name": image_id,
            "n_crops_maskcrop": a["n_crops"], "n_crops_shape_kept": b["n_kept"],
            "n_shape_rejected": b["n_rejected"],
            "contamination_maskcrop": cont_a, "contamination_shape": cont_b,
            "contamination_reduction": reduction,
            "retention_maskcrop": a.get("target_retention"),
            "retention_shape": b.get("target_retention"),
            "occupancy_maskcrop": a.get("occupancy"), "occupancy_shape": b.get("occupancy"),
            "shape_vlrr_status": getattr(b_reso, "status", None),
            # verify / outer-loop accounting (the crop VLRR ran MaskCrop -> triage-verify
            # -> retune -> rerun; one verifier pass per outer-loop iteration):
            "vlrr_iters": int(crop_activity.get("n_steps_executed", 0) or 0),        # outer loop
            "crops_retuned": int(crop_activity.get("crops_retuned", 0) or 0),        # max(0, iters-1)
            "vision_judgements": int(crop_activity.get("vision_judgements", 0) or 0),  # crops sent to VLM
        }
        (img_out / "metrics.json").write_text(
            json.dumps({**metrics, "runtime_seconds": runtime, "tokens": tokens},
                       indent=2, default=str), encoding="utf-8")
        row.update({**metrics, "runtime_seconds": round(runtime, 3),
                    "tokens": tokens, "tokens_input": tok_in, "tokens_output": tok_out,
                    "tokens_cached": tok_cached, "_metrics": metrics})
    except Exception as exc:
        import traceback
        row["status"] = "error"; row["error"] = f"{type(exc).__name__}: {exc}"
        (img_out / "error.txt").write_text(traceback.format_exc(), encoding="utf-8")
    return row


_CSV_COLUMNS = [
    "image_name", "status",
    "n_crops_maskcrop", "n_crops_shape_kept", "n_shape_rejected",
    "contamination_maskcrop", "contamination_shape", "contamination_reduction",
    "retention_maskcrop", "retention_shape", "occupancy_maskcrop", "occupancy_shape",
    "shape_vlrr_status", "vlrr_iters", "crops_retuned", "vision_judgements", "runtime_seconds",
    "tokens", "tokens_input", "tokens_output", "tokens_cached", "cost_usd", "error",
]


def _write_csv(rows, path):
    import csv
    with open(path, "w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=_CSV_COLUMNS, extrasaction="ignore")
        w.writeheader(); w.writerows(rows)


def _mean(vals):
    vals = [v for v in vals if v is not None]
    return float(sum(vals) / len(vals)) if vals else None


def _cost(ti, to, tc, pin, pout, pc):
    if pin is None:
        return None
    return round((ti or 0) / 1e6 * pin + (to or 0) / 1e6 * pout + (tc or 0) / 1e6 * (pc or pin), 4)


def main() -> None:
    try:
        from dotenv import load_dotenv
        load_dotenv(_REPO / ".env")
    except Exception:
        pass
    cfg = _load_local_config()
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    default_root = os.environ.get("SHAPE_BENCH_DATA_ROOT")
    ap.add_argument("--data-root", type=Path,
                    default=(Path(default_root) / "NeurIPS22-CellSeg") if default_root else None,
                    help="NeurIPS22-CellSeg root with images/ + labels/.")
    ap.add_argument("--image-subdir", default="images")
    ap.add_argument("--label-subdir", default="labels")
    ap.add_argument("--label-suffix", default=cfg.label_suffix)
    ap.add_argument("--label-ext", default=cfg.label_ext)
    ap.add_argument("--output-root", type=Path, default=_THIS_DIR.parent / "results")
    ap.add_argument("--masks-dir", type=Path, default=None,
                    help="Where Cellpose-SAM _pred.tiff masks are cached (default: <run>/pred_masks).")
    ap.add_argument("--masks-from-seg-run", type=Path, default=None,
                    help="Reuse the AGENTIC masks from a segmentation run dir "
                         "(results/<ts>_concurrent) instead of re-segmenting: converts each "
                         "image's final cell_mask_<id>_a<N>.npy -> <id>_pred.tiff. No GPU.")
    ap.add_argument("--masks-from-gt", action="store_true",
                    help="Crop on the GROUND-TRUTH masks (data/labels/<id>_label.tiff) instead "
                         "of a predicted segmentation: measures crop quality on ideal masks. "
                         "No GPU, no segmentation.")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--limit", type=int)
    ap.add_argument("--model", default=cfg.llm_model)
    ap.add_argument("--min-area", type=int, default=cfg.default_min_area)
    ap.add_argument("--margin", type=int, default=cfg.default_margin)
    ap.add_argument("--maskcrop-margin-frac", type=float, default=cfg.default_maskcrop_margin_frac)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--montage-grid", default="6x6",
                    help="Triage montage grid RxC (crops per VLM montage). Bigger = fewer "
                         "montage calls = cheaper on crowded images. Default 6x6 (was 3x3).")
    ap.add_argument("--crop-max-iters", type=int, default=3,
                    help="Max crop-VLRR iterations (the inner verify->retune loop cap). "
                         "Default 3 -- measured: no image uses >3 rounds (round 2 does ~99%% of "
                         "the contamination reduction, round 3 the rest); 2 conceded 38%% early.")
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--price-in", type=float, default=None)
    ap.add_argument("--price-out", type=float, default=None)
    ap.add_argument("--price-cached", type=float, default=None)
    args = ap.parse_args()
    if args.data_root is None:
        ap.error("--data-root required (or $SHAPE_BENCH_DATA_ROOT).")

    run_root = args.output_root / (time.strftime("%Y-%m-%d_%H%M%S") + "_concurrent")
    run_root.mkdir(parents=True, exist_ok=True)
    masks_dir = args.masks_dir or (run_root / "pred_masks")
    masks_dir.mkdir(parents=True, exist_ok=True)

    image_dir = args.data_root / args.image_subdir
    label_dir = args.data_root / args.label_subdir
    exts = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
    imgs = sorted(p for p in glob.glob(str(image_dir / "*"))
                  if os.path.splitext(p)[1].lower() in exts and not p.endswith("_seg"))
    pairs = []
    for ip in imgs:
        stem = Path(ip).stem
        gt = label_dir / f"{stem}{args.label_suffix}{args.label_ext}"
        pairs.append((Path(ip), gt if gt.exists() else None))
    if args.limit:
        pairs = pairs[:args.limit]
    if not pairs:
        ap.error(f"no images under {image_dir}")

    api_key = os.environ.get("LLM_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    # run_params.json — self-document the run (same convention as segmentation)
    try:
        import subprocess
        git = subprocess.run(["git", "-C", str(_THIS_DIR), "rev-parse", "HEAD"],
                             capture_output=True, text=True, timeout=5).stdout.strip() or None
    except Exception:
        git = None
    (run_root / "run_params.json").write_text(json.dumps({
        "timestamp": run_root.name, "argv": sys.argv, "git_commit": git,
        "task": "cropper_NeurIPS22-CellSeg (concurrent, MaskCrop vs SHAPE)",
        "n_images": len(pairs), "args": {k: (str(v) if isinstance(v, Path) else v)
                                         for k, v in vars(args).items()},
        "model": args.model, "masks_dir": str(masks_dir),
        "pricing_usd_per_1M": {"input": args.price_in, "output": args.price_out,
                               "cached_input": args.price_cached},
        "env": {k: os.environ.get(k) for k in ("LLM_BASE_URL", "SHAPE_REASONING_EFFORT")},
    }, indent=2, default=str), encoding="utf-8")

    # Masks: GT / reuse a segmentation run's agentic masks (no GPU) / pre-segment (GPU once).
    if args.masks_from_gt:
        import numpy as _np, tifffile as _tiff
        print(f"[cropper] cropping on GROUND-TRUTH masks (no segmentation)", flush=True)
        n_ok = 0
        for ip, gt in pairs:
            out_tiff = masks_dir / f"{ip.stem}_pred.tiff"
            if out_tiff.exists():
                n_ok += 1; continue
            if gt is None or not Path(gt).exists():
                print(f"  [miss] no GT label for {ip.stem}", flush=True); continue
            try:
                m = _tiff.imread(str(gt)).astype(_np.uint16)
                _tiff.imwrite(str(out_tiff), m)
                n_ok += 1
            except Exception as e:
                print(f"  [conv-err] {ip.stem}: {e}", flush=True)
        print(f"[cropper] staged {n_ok}/{len(pairs)} GT masks -> {masks_dir}", flush=True)
    elif args.masks_from_seg_run:
        import numpy as _np, tifffile as _tiff
        seg_imgs = args.masks_from_seg_run / "images"
        print(f"[cropper] reusing agentic masks from {args.masks_from_seg_run} (no re-segmentation)",
              flush=True)
        n_ok = 0
        for ip, _gt in pairs:
            out_tiff = masks_dir / f"{ip.stem}_pred.tiff"
            if out_tiff.exists():
                n_ok += 1; continue
            npys = sorted(glob.glob(str(seg_imgs / ip.stem / "shape_cache" /
                                        "output_visualizations" / f"cell_mask_{ip.stem}_a*.npy")))
            if not npys:
                print(f"  [miss] no seg mask for {ip.stem}", flush=True); continue
            try:
                m = _np.load(npys[-1]).astype(_np.uint16)   # highest attempt = accepted mask
                _tiff.imwrite(str(out_tiff), m)
                n_ok += 1
            except Exception as e:
                print(f"  [conv-err] {ip.stem}: {e}", flush=True)
        print(f"[cropper] converted {n_ok}/{len(pairs)} agentic masks -> {masks_dir}", flush=True)
    else:
        print(f"[cropper] pre-segmenting {len(pairs)} Cellpose-SAM masks -> {masks_dir}", flush=True)
        import mask_cache as mq
        model_holder: dict = {}
        for i, (ip, _gt) in enumerate(pairs, 1):
            if not (masks_dir / f"{ip.stem}_pred.tiff").exists():
                try:
                    mq._load_or_segment_mask(ip, masks_dir, model_holder, args.batch_size,
                                             args.flow_threshold, args.cellprob_threshold)
                except Exception as e:
                    print(f"  [seg-err] {ip.stem}: {e}", flush=True)
            if i % 50 == 0:
                print(f"  segmented {i}/{len(pairs)}", flush=True)

    jobs = [{"image_path": str(ip), "run_root": str(run_root), "masks_dir": str(masks_dir),
             "model": args.model, "api_key": api_key, "min_area": args.min_area,
             "margin": args.margin, "maskcrop_margin_frac": args.maskcrop_margin_frac,
             "batch_size": args.batch_size, "flow_threshold": args.flow_threshold,
             "cellprob_threshold": args.cellprob_threshold,
             "montage_grid": tuple(int(x) for x in str(args.montage_grid).lower().split("x")),
             "crop_max_iters": args.crop_max_iters} for ip, _ in pairs]

    print(f"[cropper] {len(jobs)} image(s) x {args.workers} workers -> {run_root}", flush=True)
    t0 = time.perf_counter()
    rows = []
    ctx = get_context("spawn")
    with ProcessPoolExecutor(max_workers=max(1, args.workers), mp_context=ctx) as ex:
        futs = {ex.submit(run_one_image, j): j["image_path"] for j in jobs}
        for i, fut in enumerate(as_completed(futs), 1):
            try:
                r = fut.result()
            except Exception as exc:
                r = {"image_name": Path(futs[fut]).stem, "status": "worker_crash",
                     "error": f"{type(exc).__name__}: {exc}"}
            rows.append(r)
            print(f"  [{i}/{len(jobs)}] {r['image_name']}: {r['status']} "
                  f"cont {r.get('contamination_maskcrop')}->{r.get('contamination_shape')}", flush=True)
    rows.sort(key=lambda r: r["image_name"])

    for r in rows:
        r["cost_usd"] = _cost(r.get("tokens_input"), r.get("tokens_output"),
                              r.get("tokens_cached"), args.price_in, args.price_out, args.price_cached)
    _write_csv(rows, run_root / "cell_image_metrics.csv")

    scored = [r for r in rows if r.get("_metrics")]
    from collections import Counter as _Counter
    _iters = [int(r.get("vlrr_iters") or 0) for r in scored]
    agentic_activity_totals = {
        "crop_max_iters": args.crop_max_iters,                 # the outer-loop cap
        # OUTER LOOP: MaskCrop -> triage-verify -> retune -> rerun iterations actually run
        "outer_loop_iters_total": sum(_iters),
        "outer_loop_iters_mean": round(sum(_iters) / len(_iters), 3) if _iters else 0.0,
        "outer_loop_iters_by_count": {str(k): v for k, v in sorted(_Counter(_iters).items())},
        # VERIFY: the triage/verifier ran once per outer-loop iteration (so total == outer loop);
        # `vision_judgements_total` is how many crops were actually sent to the VLM.
        "verify_rounds_total": sum(_iters),
        "crops_retuned_total": sum(int(r.get("crops_retuned") or 0) for r in scored),
        "vision_judgements_total": sum(int(r.get("vision_judgements") or 0) for r in scored),
        "vlrr_status_breakdown": dict(_Counter(r.get("shape_vlrr_status") for r in scored)),
    }
    summary = {
        "dataset": "NeurIPS22-CellSeg", "task": "cropper (concurrent, MaskCrop vs SHAPE)",
        "workers": args.workers, "model": args.model,
        "n_images": len(rows), "n_scored": len(scored),
        "n_errors": sum(1 for r in rows if r["status"] in ("error", "worker_crash")),
        "wall_clock_seconds": round(time.perf_counter() - t0, 2),
        "crop_quality_macro": {
            "contamination_maskcrop": _mean([r.get("contamination_maskcrop") for r in scored]),
            "contamination_shape": _mean([r.get("contamination_shape") for r in scored]),
            "contamination_reduction": _mean([r.get("contamination_reduction") for r in scored]),
            "retention_maskcrop": _mean([r.get("retention_maskcrop") for r in scored]),
            "retention_shape": _mean([r.get("retention_shape") for r in scored]),
            "n_crops_maskcrop_total": sum(int(r.get("n_crops_maskcrop") or 0) for r in scored),
            "n_crops_shape_kept_total": sum(int(r.get("n_crops_shape_kept") or 0) for r in scored),
            "n_shape_rejected_total": sum(int(r.get("n_shape_rejected") or 0) for r in scored),
        },
        "agentic_activity_totals": agentic_activity_totals,
        "cost": {
            "model": args.model,
            "price_per_1M": {"input": args.price_in, "output": args.price_out,
                             "cached_input": args.price_cached},
            "tokens_total": sum(int(r.get("tokens") or 0) for r in rows),
            "tokens_input_total": sum(int(r.get("tokens_input") or 0) for r in rows),
            "tokens_output_total": sum(int(r.get("tokens_output") or 0) for r in rows),
            "cost_usd_total": (round(sum(r["cost_usd"] for r in rows if r.get("cost_usd")), 4)
                               if args.price_in is not None else None),
        },
        "csv": str(run_root / "cell_image_metrics.csv"),
    }
    (run_root / "run_summary.json").write_text(json.dumps(summary, indent=2, default=str),
                                               encoding="utf-8")
    m = summary["crop_quality_macro"]
    aat = summary["agentic_activity_totals"]
    print(f"[done] {len(scored)}/{len(rows)} scored | contamination "
          f"maskcrop={m['contamination_maskcrop']:.4f} -> shape={m['contamination_shape']:.4f} "
          f"(reduction {m['contamination_reduction']:+.4f}) | retention "
          f"{m['retention_maskcrop']:.3f}->{m['retention_shape']:.3f}", flush=True)
    print(f"[done] outer-loop iters: total={aat['outer_loop_iters_total']} "
          f"mean={aat['outer_loop_iters_mean']} by_count={aat['outer_loop_iters_by_count']} "
          f"(cap {aat['crop_max_iters']}) | retunes={aat['crops_retuned_total']} "
          f"vlm_judged={aat['vision_judgements_total']} | status={aat['vlrr_status_breakdown']}",
          flush=True)
    print(f"[done] summary -> {run_root / 'run_summary.json'}", flush=True)


if __name__ == "__main__":
    main()

"""Cellpose 3 (cyto3) baseline arm on the 1000-image cardiac organoid set.

Runs in the cellpose3 conda env (cellpose 3.1.x). Settings follow the benchmark's Cellpose3 arm
(benchmarks/segmentation_NeurIPS22-CellSeg/runners/cellseg_arms.py at 30f799d9): model cyto3, diameter
estimated by the size model (diameter=None), grayscale channels [0, 0], the image itself when 2-D or its
highest-variance channel. One call per image, no agent, no LLM.

Per image: images/<image_id>/mask.tif (uint16 label map), record.json; appended to records.jsonl.
At the end: cellpose3.csv (same layout as the other baseline per-image tables, AP at 0.50 ... 0.95),
per_image_ap_full_thresholds_cellpose3.csv (AP at 0.05 ... 1.00), run_meta.json.
Images that raise are recorded with delivered=false and AP 0 (the study convention) and counted.
"""
import argparse
import csv
import hashlib
import json
import os
import platform
import sys
import time

import numpy as np
import tifffile

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_ap as ia  # noqa: E402

ARM = "cellpose3"
MODEL_TYPE = "cyto3"


def read_image(path):
    """Raw image as H x W or H x W x C; tifffile first, PIL for files whose bytes are not TIFF (some .tif files are JPEG)."""
    try:
        arr = tifffile.imread(path)
    except Exception:
        from PIL import Image
        with Image.open(path) as im:
            arr = np.asarray(im)
    arr = np.asarray(arr).squeeze()
    if arr.ndim == 3 and arr.shape[0] <= 4 < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 4:
        arr = arr[..., :3]
    return arr


def gray(img):
    img = np.asarray(img).squeeze()
    if img.ndim == 3 and img.shape[0] <= 4 < img.shape[-1]:
        img = np.moveaxis(img, 0, -1)
    if img.ndim == 2:
        return img.astype(np.float32)
    stds = [float(img[..., c].std()) for c in range(img.shape[-1])]
    return img[..., int(np.argmax(stds))].astype(np.float32)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out-root", required=True)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--resume", action="store_true", help="keep existing error-free records, rerun the rest")
    args = ap.parse_args()
    from cellpose import core, models, version as cp_version
    import torch
    pairs = json.load(open(args.manifest))["pairs"]
    if args.limit:
        pairs = pairs[:args.limit]
    out = args.out_root
    os.makedirs(os.path.join(out, "images"), exist_ok=True)
    gpu = bool(core.use_gpu())
    model = models.Cellpose(gpu=gpu, model_type=MODEL_TYPE)
    weights = os.path.join(os.path.expanduser("~"), ".cellpose", "models", MODEL_TYPE)
    meta = {"arm": ARM, "model_type": MODEL_TYPE, "cellpose": cp_version, "torch": torch.__version__, "cuda": torch.version.cuda,
            "gpu": gpu, "gpu_name": torch.cuda.get_device_name(0) if gpu else None, "host": platform.node(),
            "python": sys.executable, "numpy": np.__version__, "job_id": os.environ.get("SLURM_JOB_ID"),
            "weights_path": weights, "weights_sha256": ia.sha256_file(weights) if os.path.isfile(weights) else None,
            "size_model_sha256": ia.sha256_file(weights.replace(MODEL_TYPE, "size_%s.npy" % MODEL_TYPE)) if os.path.isfile(weights.replace(MODEL_TYPE, "size_%s.npy" % MODEL_TYPE)) else None,
            "manifest": args.manifest, "manifest_sha256": ia.sha256_file(args.manifest), "n_images": len(pairs),
            "settings": {"diameter": None, "channels": [0, 0], "input": "grayscale, highest-variance channel if multi-channel"},
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "resume": args.resume, "image_reader": "tifffile, PIL fallback"}
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=2)
    records, full_rows, failures = [], [], 0
    rec_path = os.path.join(out, "records.jsonl")
    previous = {}
    if args.resume and os.path.isfile(rec_path):
        for line in open(rec_path):
            r = json.loads(line)
            if not r.get("error"):
                previous[r["image_id"]] = r
        os.replace(rec_path, rec_path + ".before_resume")
        print("resume: keeping %d error-free records" % len(previous), flush=True)
    for i, p in enumerate(pairs):
        image_id = p["image_id"]
        d = os.path.join(out, "images", image_id)
        os.makedirs(d, exist_ok=True)
        if image_id in previous:
            rec = previous[image_id]
            with open(rec_path, "a") as fh:
                fh.write(json.dumps(rec) + "\n")
            records.append(rec)
            full_rows.extend((image_id, ARM, k, v["true_positive"], v["false_positive"], v["false_negative"], v["average_precision"]) for k, v in rec["thresholds_full"].items())
            continue
        if ia.sha256_file(p["image"]) != p["image_sha256"] or ia.sha256_file(p["label"]) != p["label_sha256"]:
            raise RuntimeError("input hash mismatch for " + image_id)
        gt = ia.load_label(p["label"])
        t0 = time.time()
        rec = {"image_id": image_id, "arm": ARM, "started_unix": t0, "delivered": False, "error": "", "mask_sha256": "",
               "image_sha256": p["image_sha256"], "ground_truth_sha256": p["label_sha256"],
               "image_height_px": int(gt.shape[0]), "image_width_px": int(gt.shape[1])}
        try:
            img = gray(read_image(p["image"]))
            if img.shape != gt.shape:
                raise RuntimeError("image shape %s != label shape %s" % (img.shape, gt.shape))
            masks, _flows, _styles, diams = model.eval(img, diameter=None, channels=[0, 0])
            masks = np.asarray(masks)
            if masks.max() > 65535:
                raise RuntimeError("more than 65535 labels")
            mpath = os.path.join(d, "mask.tif")
            tifffile.imwrite(mpath, masks.astype(np.uint16))
            rec["native_seconds"] = time.time() - t0
            rec["estimated_diameter"] = float(np.asarray(diams).ravel()[0])
            rec["mask_sha256"] = ia.sha256_file(mpath)
            rec["delivered"] = True
            ev = ia.evaluate(gt, masks)
        except Exception as exc:  # recorded, counted, AP 0 by convention
            failures += 1
            rec["error"] = "%s: %s" % (type(exc).__name__, exc)
            ev = {"ground_truth_cell_count": int(len(np.unique(gt)) - (1 if (gt == 0).any() else 0)), "predicted_cell_count": 0,
                  "thresholds": {"%.2f" % t: {"true_positive": 0, "false_positive": 0, "false_negative": int(len(np.unique(gt)) - (1 if (gt == 0).any() else 0)),
                                             "average_precision": 0.0, "precision": 0.0, "recall": 0.0, "f1": 0.0, "mean_matched_iou": 0.0} for t in ia.FULL_THRESHOLDS}}
        rec["seconds"] = time.time() - t0
        th = ev["thresholds"]
        std = {k: th[k] for k in ("%.2f" % t for t in ia.STANDARD_THRESHOLDS)}
        rec.update(ground_truth_cell_count=ev["ground_truth_cell_count"], predicted_cell_count=ev["predicted_cell_count"],
                   ap50=std["0.50"]["average_precision"], ap75=std["0.75"]["average_precision"],
                   map_50_95=float(np.mean([v["average_precision"] for v in std.values()])),
                   precision_at_50=std["0.50"]["precision"], recall_at_50=std["0.50"]["recall"], f1_at_50=std["0.50"]["f1"],
                   mean_matched_iou_at_50=std["0.50"]["mean_matched_iou"],
                   cell_count_error_fraction=abs(ev["predicted_cell_count"] - ev["ground_truth_cell_count"]) / max(1, ev["ground_truth_cell_count"]),
                   tool_calls=1, native_calls=1, llm_calls=0, tokens=0, llm_seconds=0, _thresholds=std, thresholds_full=th)
        json.dump(rec, open(os.path.join(d, "record.json"), "w"), indent=2)
        with open(rec_path, "a") as fh:
            fh.write(json.dumps(rec) + "\n")
        records.append(rec)
        full_rows.extend((image_id, ARM, k, v["true_positive"], v["false_positive"], v["false_negative"], v["average_precision"]) for k, v in th.items())
        if i % 25 == 0 or i == len(pairs) - 1:
            print("%d/%d %s map=%.3f ap50=%.3f pred=%d gt=%d %.1fs%s" % (i + 1, len(pairs), image_id[:40], rec["map_50_95"], rec["ap50"],
                  rec["predicted_cell_count"], rec["ground_truth_cell_count"], rec["seconds"], " ERROR " + rec["error"] if rec["error"] else ""), flush=True)
    # per-image table in the layout of the other baseline tables
    cols = ["image_id", "arm", "evaluation_status", "delivered", "source_run", "run_kind", "run_status", "image_height_px", "image_width_px",
            "image_sha256", "ground_truth_sha256", "ground_truth_cell_count", "predicted_cell_count", "ap50", "ap75", "map_50_95",
            "precision_at_50", "recall_at_50", "f1_at_50", "mean_matched_iou_at_50", "cell_count_error_fraction", "seconds", "native_seconds",
            "llm_seconds", "tool_calls", "native_calls", "llm_calls", "tokens", "started_unix", "mask_sha256", "error", "estimated_diameter"]
    metrics = ["true_positive", "false_positive", "false_negative", "average_precision", "precision", "recall", "f1", "mean_matched_iou"]
    for t in ia.STANDARD_THRESHOLDS:
        cols += ["iou_%03d_%s" % (int(round(t * 100)), m) for m in metrics]
    run_name = os.path.basename(os.path.normpath(out))
    with open(os.path.join(out, "cellpose3.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(cols)
        for r in records:
            row = [r["image_id"], ARM, "success" if r["delivered"] else "failed", str(r["delivered"]).lower(), run_name, "full", "COMPLETE",
                   r["image_height_px"], r["image_width_px"], r["image_sha256"], r["ground_truth_sha256"], r["ground_truth_cell_count"],
                   r["predicted_cell_count"], r["ap50"], r["ap75"], r["map_50_95"], r["precision_at_50"], r["recall_at_50"], r["f1_at_50"],
                   r["mean_matched_iou_at_50"], r["cell_count_error_fraction"], r["seconds"], r.get("native_seconds", ""), 0, 1, 1, 0, 0,
                   r["started_unix"], r["mask_sha256"], r["error"], r.get("estimated_diameter", "")]
            for t in ia.STANDARD_THRESHOLDS:
                row += [r["_thresholds"]["%.2f" % t][m] for m in metrics]
            w.writerow(row)
    with open(os.path.join(out, "per_image_ap_full_thresholds_cellpose3.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "arm", "iou", "tp", "fp", "fn", "ap"]); w.writerows(full_rows)
    meta.update(ended_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), n_delivered=sum(r["delivered"] for r in records), n_failed=failures,
                mean_map_50_95=float(np.mean([r["map_50_95"] for r in records])), mean_ap50=float(np.mean([r["ap50"] for r in records])),
                total_seconds=float(sum(r["seconds"] for r in records)))
    json.dump(meta, open(os.path.join(out, "run_meta.json"), "w"), indent=2)
    print("SUMMARY", json.dumps({k: meta[k] for k in ("n_images", "n_delivered", "n_failed", "mean_map_50_95", "mean_ap50", "total_seconds", "gpu_name")}), flush=True)


if __name__ == "__main__":
    main()

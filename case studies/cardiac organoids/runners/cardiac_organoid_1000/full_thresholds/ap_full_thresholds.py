"""Recompute AP at IoU thresholds 0.05 ... 1.00 for every arm of the 1000-image cardiac organoid study
from the delivered masks, verifying each mask's sha256 against the exported per-image table and the
recomputed AP at 0.50 ... 0.95 against the stored values.

Inputs:
  EXPORT/per_image/*.csv            the exported per-image tables (mask_sha256, iou_0XX_average_precision, ...)
  EXPORT/launch_code/run_manifest.json   the 1000 pairs with ground-truth label paths
  OPT1/shape/shard_XXX/run/images/<id>/mask.tif       SHAPE masks
  react.csv attempt_dir/mask.tif                      ReAct masks
  NONLLM/<arm>/records.jsonl -> attempt_dir/mask.tif  baseline masks
Outputs (OUT):
  per_image_ap_full_thresholds.csv  image_id, arm, iou, tp, fp, fn, ap   (7 arms x 1000 images x 20 thresholds)
  mean_ap_full_thresholds.csv       arm, iou, n, mean_ap, sd_ap, sem_ap
  verification.json                 counts of verified hashes and AP matches, any mismatches
"""
import csv
import json
import os
import sys
from multiprocessing import Pool

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_ap as ia  # noqa: E402

SCRATCH = os.environ["SCRATCH"]
EXPORT = os.path.join(SCRATCH, "export_hard1000_opt1_30f799d9")
OPT1 = os.path.join(SCRATCH, "shape_benchmark/results/cardiac_organoids_segmentation/hard1000_opt1_30f799d9_20261024")
GROUP = "/scratch/groups/your-group/your-user/shape_benchmark/results/cardiac_organoids_segmentation"
NONLLM = {"cellpose_sam": GROUP + "/nonllm_full_00fad589", "shape_seg_co": GROUP + "/nonllm_full_00fad589",
          "organoid_id": GROUP + "/nonllm_full_00fad589", "stardist": GROUP + "/nonllm_full_00fad589",
          "microsam": GROUP + "/nonllm_full_00fad589_microsam"}
ARM_CSV = {"shape": "opt1_shape_cohort_bounded.csv", "react": "react.csv", "shape_seg_co": "shape_seg_co.csv",
           "microsam": "microsam.csv", "cellpose_sam": "cellpose_sam.csv", "organoid_id": "organoid_id.csv",
           "stardist": "stardist.csv"}
OUT = os.environ.get("AP_OUT") or (sys.argv[1] if len(sys.argv) > 1 else os.path.join(SCRATCH, "ap_full_thresholds_20261024"))


def read_csv(path):
    with open(path, encoding="utf-8", newline="") as fh:
        return {r["image_id"]: r for r in csv.DictReader(fh)}


def truthy(v):
    return str(v).strip().lower() in ("true", "1", "yes")


def mask_path(arm, row, nonllm_records):
    if arm == "shape":
        return os.path.join(OPT1, "shape", "shard_%03d" % int(row["shard_index"]), "run", "images", row["image_id"], "mask.tif")
    if arm == "react":
        return os.path.join(row["attempt_dir"], "mask.tif")
    return os.path.join(nonllm_records[arm][row["image_id"]]["attempt_dir"], "mask.tif")


def work(task):
    arm, image_id, label_path, mpath, delivered, expected_sha, expected_ap, expected_gt = task
    gt = ia.load_label(label_path)
    res = {"arm": arm, "image_id": image_id, "delivered": delivered, "sha_ok": None, "ap_ok": None, "note": ""}
    if not delivered:
        # the study counts an image without a delivered mask as AP 0 at every threshold
        res["rows"] = [(image_id, arm, "%.2f" % t, 0, 0, int(gt.max() > 0 and len(np.unique(gt)) - 1), 0.0) for t in ia.FULL_THRESHOLDS]
        res["note"] = "no delivered mask; AP 0 by convention"
        return res
    if not os.path.isfile(mpath):
        res["note"] = "mask file missing: " + mpath
        return res
    res["sha_ok"] = ia.sha256_file(mpath) == expected_sha
    pred = ia.load_label(mpath)
    ev = ia.evaluate(gt, pred)
    if expected_gt is not None and ev["ground_truth_cell_count"] != expected_gt:
        res["note"] += "gt count %d != stored %d; " % (ev["ground_truth_cell_count"], expected_gt)
    if expected_ap:
        diffs = [abs(ev["thresholds"]["%.2f" % t]["average_precision"] - expected_ap["%.2f" % t]) for t in ia.STANDARD_THRESHOLDS]
        res["ap_ok"] = max(diffs) < 1e-9
        if not res["ap_ok"]:
            res["note"] += "AP mismatch max %.3g; " % max(diffs)
    res["rows"] = [(image_id, arm, k, v["true_positive"], v["false_positive"], v["false_negative"], v["average_precision"])
                   for k, v in ev["thresholds"].items()]
    return res


def main():
    os.makedirs(OUT, exist_ok=True)
    manifest = json.load(open(os.path.join(EXPORT, "launch_code", "run_manifest.json")))
    labels = {p["image_id"]: p["label"] for p in manifest["pairs"]}
    assert len(labels) == 1000
    nonllm_records = {}
    for arm, root in NONLLM.items():
        with open(os.path.join(root, arm, "records.jsonl")) as fh:
            nonllm_records[arm] = {r["image_id"]: r for r in map(json.loads, fh) if r["image_id"] in labels}
        assert len(nonllm_records[arm]) == 1000, (arm, len(nonllm_records[arm]))
    tasks = []
    for arm, fname in ARM_CSV.items():
        table = read_csv(os.path.join(EXPORT, "per_image", fname))
        assert set(table) == set(labels), arm
        for image_id, row in table.items():
            delivered = truthy(row.get("delivered", "true"))
            exp_ap = {}
            for t in ia.STANDARD_THRESHOLDS:
                v = row.get("iou_%03d_average_precision" % int(round(t * 100)), "")
                if v != "":
                    exp_ap["%.2f" % t] = float(v)
            if len(exp_ap) != len(ia.STANDARD_THRESHOLDS):
                exp_ap = {}  # rows without the per-threshold breakdown (reused ReAct rows): no AP check
            gt_count = int(row["ground_truth_cell_count"]) if row.get("ground_truth_cell_count", "") != "" else None
            mp = mask_path(arm, row, nonllm_records) if delivered else ""
            tasks.append((arm, image_id, labels[image_id], mp, delivered, row["mask_sha256"], exp_ap, gt_count))
    print("tasks", len(tasks), flush=True)
    with Pool(int(os.environ.get("SLURM_CPUS_PER_TASK", "4"))) as pool:
        results = pool.map(work, tasks, chunksize=8)
    rows, ver = [], {"n": len(results), "sha_checked": 0, "sha_ok": 0, "ap_checked": 0, "ap_ok": 0,
                     "not_delivered": 0, "missing_mask": 0, "problems": []}
    for r in results:
        if "rows" in r:
            rows.extend(r["rows"])
        if not r["delivered"]:
            ver["not_delivered"] += 1
        if r["sha_ok"] is not None:
            ver["sha_checked"] += 1; ver["sha_ok"] += int(r["sha_ok"])
        if r["ap_ok"] is not None:
            ver["ap_checked"] += 1; ver["ap_ok"] += int(r["ap_ok"])
        if "missing" in r["note"]:
            ver["missing_mask"] += 1
        if r["note"] and "by convention" not in r["note"] or r["sha_ok"] is False or r["ap_ok"] is False:
            ver["problems"].append({k: r[k] for k in ("arm", "image_id", "sha_ok", "ap_ok", "note")})
    with open(os.path.join(OUT, "per_image_ap_full_thresholds.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["image_id", "arm", "iou", "tp", "fp", "fn", "ap"]); w.writerows(rows)
    by = {}
    for image_id, arm, iou, tp, fp, fn, ap in rows:
        by.setdefault((arm, iou), []).append(ap)
    with open(os.path.join(OUT, "mean_ap_full_thresholds.csv"), "w", newline="") as fh:
        w = csv.writer(fh); w.writerow(["arm", "iou", "n", "mean_ap", "sd_ap", "sem_ap"])
        for arm in ARM_CSV:
            for t in ia.FULL_THRESHOLDS:
                a = np.array(by.get((arm, "%.2f" % t), []), dtype=float)
                w.writerow([arm, "%.2f" % t, len(a), "%.6f" % a.mean(), "%.6f" % a.std(ddof=1), "%.6f" % (a.std(ddof=1) / np.sqrt(len(a)))])
    json.dump(ver, open(os.path.join(OUT, "verification.json"), "w"), indent=2)
    print(json.dumps({k: v for k, v in ver.items() if k != "problems"}), flush=True)
    print("problems:", len(ver["problems"]))
    for p in ver["problems"][:20]:
        print("  ", p)
    for arm in ARM_CSV:
        print(arm, " ".join("%.3f" % np.mean(by[(arm, "%.2f" % t)]) for t in ia.FULL_THRESHOLDS))


if __name__ == "__main__":
    main()

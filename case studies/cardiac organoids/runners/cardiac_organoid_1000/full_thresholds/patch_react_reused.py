"""Fill in the ReAct rows whose exported table has no attempt_dir (rows reused from an earlier run of the
same source): locate each image's mask.tif by its recorded mask_sha256 under the ReAct run roots, score it
at IoU 0.05 ... 1.00 and replace the placeholder rows in the full-threshold tables. Light work (9 images)."""
import csv
import glob
import json
import os
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import instance_ap as ia  # noqa: E402

SCRATCH = os.environ["SCRATCH"]
OUT = os.environ.get("AP_OUT") or os.path.join(SCRATCH, "ap_full_thresholds_20261024")
EXPORT = os.path.join(SCRATCH, "export_hard1000_opt1_30f799d9")
ROOTS = [os.path.join(SCRATCH, "shape_benchmark/results/cardiac_organoids_segmentation", d) for d in
         ("react_full_826227be_20261021", "full_826227be_20261021", "hard1000_826227be_20261022")]

ver = json.load(open(os.path.join(OUT, "verification.json")))
todo = [p for p in ver["problems"] if p["arm"] == "react" and "missing" in p["note"]]
with open(os.path.join(EXPORT, "per_image", "react.csv"), encoding="utf-8", newline="") as fh:
    react = {r["image_id"]: r for r in csv.DictReader(fh)}
labels = {p["image_id"]: p["label"] for p in json.load(open(os.path.join(EXPORT, "launch_code", "run_manifest.json")))["pairs"]}
new_rows, found = {}, []
for p in todo:
    iid, want = p["image_id"], react[p["image_id"]]["mask_sha256"]
    cands = []
    for root in ROOTS:
        cands += glob.glob(os.path.join(root, "react", "shard_*", "run", "react", "images", iid, "attempt_*", "mask.tif"))
        cands += glob.glob(os.path.join(root, "react", "shard_*", "run", "react", "images", iid, "mask.tif"))
    hit = [c for c in cands if ia.sha256_file(c) == want]
    if len(hit) != 1:
        print("NOT FOUND" if not hit else "AMBIGUOUS", iid, len(cands), "candidates", hit[:3])
        continue
    ev = ia.evaluate(ia.load_label(labels[iid]), ia.load_label(hit[0]))
    std = [ev["thresholds"]["%.2f" % t]["average_precision"] for t in ia.STANDARD_THRESHOLDS]
    ok = abs(np.mean(std) - float(react[iid]["map_50_95"])) < 1e-9 and abs(std[0] - float(react[iid]["ap50"])) < 1e-9 and abs(std[5] - float(react[iid]["ap75"])) < 1e-9
    print(iid[:45], "mask", hit[0].split("cardiac_organoids_segmentation/")[1][:60], "map %.3f stored %s ok=%s" % (np.mean(std), react[iid]["map_50_95"], ok))
    new_rows[iid] = [(iid, "react", k, v["true_positive"], v["false_positive"], v["false_negative"], v["average_precision"]) for k, v in ev["thresholds"].items()]
    found.append({"image_id": iid, "mask": hit[0], "mask_sha256": want, "stored_metrics_reproduced": bool(ok)})
if len(new_rows) != len(todo):
    sys.exit("not all reused ReAct masks were found; tables unchanged")
path = os.path.join(OUT, "per_image_ap_full_thresholds.csv")
rows = list(csv.reader(open(path, newline="")))
header, body = rows[0], [r for r in rows[1:] if not (r[1] == "react" and r[0] in new_rows)]
for iid in new_rows:
    body.extend(new_rows[iid])
with open(path, "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(header); w.writerows(body)
by = {}
for r in body:
    by.setdefault((r[1], r[2]), []).append(float(r[6]))
arms = ["shape", "react", "shape_seg_co", "microsam", "cellpose_sam", "organoid_id", "stardist"]
with open(os.path.join(OUT, "mean_ap_full_thresholds.csv"), "w", newline="") as fh:
    w = csv.writer(fh); w.writerow(["arm", "iou", "n", "mean_ap", "sd_ap", "sem_ap"])
    for arm in arms:
        for t in ia.FULL_THRESHOLDS:
            a = np.array(by[(arm, "%.2f" % t)]); w.writerow([arm, "%.2f" % t, len(a), "%.6f" % a.mean(), "%.6f" % a.std(ddof=1), "%.6f" % (a.std(ddof=1) / np.sqrt(len(a)))])
ver["reused_react_rows_located_by_mask_sha256"] = found
ver["missing_mask"] = 0
ver["problems"] = [p for p in ver["problems"] if not (p["arm"] == "react" and "missing" in p["note"])]
ver["sha_checked"] += len(found); ver["sha_ok"] += len(found)
json.dump(ver, open(os.path.join(OUT, "verification.json"), "w"), indent=2)
print("rows", len(body), "react", " ".join("%.3f" % np.mean(by[("react", "%.2f" % t)]) for t in ia.FULL_THRESHOLDS))

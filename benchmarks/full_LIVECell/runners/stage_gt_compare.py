"""Read-only: segmentation and crop GT scores of several arms of a LIVECell experiment, paired against the first arm.
usage: lc_stage_compare.py PREP EXP OUT.json arm_dir [arm_dir ...]   (arm_dir relative to EXP; the first is the base)"""
import json
import sys
from multiprocessing import Pool
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from benchmarks.full_LIVECell.harness import score as S
from benchmarks.full_LIVECell.harness.data import load_prepared
from benchmarks.full_CPJUMP1.harness.data import read_json

PREP, EXP, OUT = Path(sys.argv[1]), Path(sys.argv[2]), Path(sys.argv[3])
ARMS = sys.argv[4:]
index = {r["image_id"]: r for r in read_json(PREP / S.GT_INDEX)}
SEG = {a: {r["image_id"]: r for r in read_json(EXP / a / "segmentation.json")} for a in ARMS}
CELLS = {}
for a in ARMS:
    by = {}
    for c in read_json(EXP / a / "cells.json"):
        by.setdefault(c["image_id"], []).append(c)
    CELLS[a] = by


def one(args):
    arm, ident = args
    gt = S.gt_labels(PREP, index[ident])
    o = S.overlap(gt, tifffile.imread(SEG[arm][ident]["mask"]))
    row = {"one": 1, **S.segmentation_scores(o), **S.crop_scores(o, gt, CELLS[arm].get(ident, [])),
           "n_gt_without_pixels": 0}
    return arm, ident, row


if __name__ == "__main__":
    images = load_prepared(PREP)
    labels = pd.DataFrame(read_json(PREP / "private" / "labels.json"))
    draws = S.GTDraws(S.gt_groups(labels, images))
    jobs = [(a, i) for a in ARMS for i in sorted(SEG[a])]
    with Pool(8) as p:
        res = p.map(one, jobs, chunksize=8)
    tables = {a: [] for a in ARMS}
    for arm, ident, r in res:
        tables[arm].append({"image_id": ident, **r})
    readings = {a: draws.reading(tables[a]) for a in ARMS}
    base, out = ARMS[0], {}
    for fam, metrics in (("segmentation", S.SEGMENTATION), ("crop", S.CROP)):
        for name in draws.groups:
            for m in metrics:
                row = {a: readings[a][fam][name][m][0] for a in ARMS}
                v0, d0 = readings[base][fam][name][m]
                for a in ARMS[1:]:
                    v, d = readings[a][fam][name][m]
                    diff = (d - d0)[np.isfinite(d - d0)]
                    row[f"{a}-{base}"] = (None if v is None or v0 is None else
                                          {"delta": v - v0, "se": float(np.std(diff, ddof=1)),
                                           "ci95": [float(x) for x in np.percentile(diff, [2.5, 97.5])]})
                out[f"{fam}|{name}|{m}"] = row
    extra = {}
    for a in ARMS[1:]:
        fa, fb = pd.DataFrame(tables[a]).set_index("image_id"), pd.DataFrame(tables[base]).set_index("image_id")
        d = fa["ap50"] - fb["ap50"]
        extra[a] = {"ap50_images_better": int((d > 1e-9).sum()), "ap50_images_worse": int((d < -1e-9).sum()),
                    "ap50_images_equal": int((d.abs() <= 1e-9).sum()),
                    "cells_delivered": {x: int(pd.DataFrame(tables[x])["n_delivered"].sum()) for x in ARMS}}
    OUT.write_text(json.dumps({"metrics": out, "paired": extra}, indent=1, sort_keys=True), encoding="utf-8")
    print("written", OUT)

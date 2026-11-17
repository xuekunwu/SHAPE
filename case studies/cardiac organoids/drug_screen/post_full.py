"""QA evidence of a run of full_driver.py (pass criteria V2-V5).  usage: python post_full.py NAME   (env SMOKE_ROOT)"""
import json, os, sys
from collections import Counter
from pathlib import Path
import numpy as np
import tifffile

ROOT, name = Path(os.environ["SMOKE_ROOT"]), sys.argv[1]
run, prep = ROOT / "runs" / name, ROOT / "prepared"
rd = lambda p: json.loads(Path(p).read_text())
config = rd(run / "CONFIG.json")
rows = {r["image_id"]: r for r in rd(prep / "public" / "images.json") if r["image_id"] in set(config["images"])}
out = dict(run=name, mode=config["mode"], n_images=len(rows), usage_images=dict(Counter(r["usage"] for r in rows.values())),
           plate_wells=len({r["well"] for r in rows.values()}))
seg = rd(run / "segmentation.json")
out["segment"] = dict(n_masks=len(seg), n_objects=sum(s["n_cells"] for s in seg), parameter_source=dict(Counter(s["parameter_source"] for s in seg)),
                      with_qc_error=sum("defaults_qc_error" in s for s in seg))
cohort = rd(run / "segment_COHORT.json")
out["segment"]["cohort"] = dict(counts=cohort["counts"], ungrouped=len(cohort["ungrouped"]), organoid_selective_review=cohort.get("organoid_selective_review"),
                                groups=[dict(size=len(g["images"]), calibration=g.get("calibration"), routing=g.get("routing"),
                                             stable=(g.get("consensus") or {}).get("stable"), selected=(g.get("consensus") or {}).get("selected_params"))
                                        for g in cohort["groups"]])
cc = rd(run / "crop" / "COHORT_RECORD.json")          # the harness crop cohort (`pipeline._cropping_cohort`)
out["crop"] = dict(counts=cc.get("counts"), calibration_mode=cc.get("calibration_mode"), ungrouped=len(cc.get("ungrouped", [])),
                   groups=[dict(size=len(g["images"]), calibration=g.get("calibration"), routed=g.get("routed"),
                               stable=(g.get("consensus") or {}).get("stable"), cohort_levers=g.get("cohort_levers"))
                           for g in cc.get("groups", [])])
cells = rd(run / "cells.json")
excluded = rd(run / "excluded_cells.json")
bad, shapes = [], Counter()
for c in cells:
    a = tifffile.imread(c["crop_path"])
    shapes[f"{a.shape[0]}ch {a.dtype}"] += 1
    r0, c0, r1, c1 = c["crop_bbox"]
    if a.ndim != 3 or a.shape[0] != 2 or a.shape[1] != a.shape[2] or (r1 - r0) != (c1 - c0) or not (0 <= r0 < r1 <= 904 and 0 <= c0 < c1 <= 1224):
        bad.append(c["cell_id"])
out["crops"] = dict(n=len(cells), excluded=len(excluded), objects=out["segment"]["n_objects"], shapes=dict(shapes), bad=bad[:5], n_bad=len(bad),
                    usage=dict(Counter(c["usage"] for c in cells)), images_with_crops=len({c["image_id"] for c in cells}))
emb = np.load(run / "embedding.npy")
sel = rd(run / "SSL_SELECTION.json")
ck = sel.get("checkpoint")
out["embed"] = dict(shape=list(emb.shape), finite=bool(np.isfinite(emb).all()), rows_equal_crops=bool(len(emb) == len(cells)),
                    selection_mode=sel.get("mode"), checkpoint_exists=bool(ck and Path(ck).is_file()), n_development_cells=sel.get("n_development_cells"))
out["usage"] = {}
for s in ("segment", "crop", "embed"):
    c = rd(run / f"{s}_COMPLETE.json")
    out["usage"][s] = dict(seconds=round(c["seconds"]), total_tokens=c["usage"]["total_tokens"], calls=c["usage"]["calls"],
                           tokens_per_image=round(c["usage"]["total_tokens"] / max(1, len(rows))), usage_complete=c["usage"].get("usage_complete"))
(run / "qa").mkdir(exist_ok=True)
(run / "qa" / "summary.json").write_text(json.dumps(out, indent=2))
print("POST_FULL", json.dumps(out)[:6000])

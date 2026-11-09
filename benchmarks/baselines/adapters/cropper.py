"""Cropper benchmark adapter (NeurIPS22-CellSeg). The agent is given images + their instance masks
and must choose, per labeled cell, a crop bounding box that captures the target cell while
minimizing neighbor-cell contamination. Scored GT-free with our crop-QC (mirrors SHAPE, which runs
its cropper on the same GT masks).
"""
from __future__ import annotations

import glob
import json
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile

from .base import BenchmarkAdapter, ScoreResult
from ..metrics.crop_qc import image_crop_metrics, macro_summary

_IMG_EXTS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")

_TASK = """# Task: single-cell cropping

You are given microscopy images in `./inputs/` and their instance segmentation masks in
`./inputs/masks/<id>_mask.tiff` (each cell = a unique positive integer label, 0 = background).

For **every** image and **every** labeled cell, choose a CROP bounding box that captures the whole
target cell while minimizing **neighbor contamination** -- pixels of OTHER cells inside the box.

Deliver `./deliver/crops/<id>.json`: a JSON list, one entry per cell label in that image's mask:

    [{"cell_label": 5, "bbox": [minr, minc, maxr, maxc]}, ...]

`bbox` is in pixel coordinates, half-open `[minr:maxr, minc:maxc]` (row/col). Include every label.

## How you are scored (GT-free, from the mask you were given)
- **neighbor_contamination** (LOWER better) = other-cell foreground pixels in the box / all
  foreground in the box.
- **target_retention** (HIGHER better) = target-cell pixels in the box / the target cell's total area.

A tight bounding box around a cell captures it fully (retention ~1); shrinking/centering the box
reduces overlap with neighbors. Cover every cell. You may use any method (regionprops bounding
boxes are a strong start); search online if useful.

## Definition of done
`./deliver/crops/<id>.json` exists for every image, listing a bbox for every labeled cell.
"""


class CropperAdapter(BenchmarkAdapter):
    id = "cropper"

    def __init__(self, data_root: Path, image_subdir="images", label_subdir="labels",
                 label_suffix="_label", label_ext=".tiff"):
        super().__init__(data_root)
        self.image_dir = self.data_root / image_subdir
        self.label_dir = self.data_root / label_subdir
        self.label_suffix = label_suffix
        self.label_ext = label_ext

    def select_items(self, limit=None, seed=0):
        imgs = sorted(p for p in glob.glob(str(self.image_dir / "*"))
                      if os.path.splitext(p)[1].lower() in _IMG_EXTS)
        return [Path(p).stem for p in imgs][:limit] if limit else [Path(p).stem for p in imgs]

    def _src_image(self, stem):
        for e in _IMG_EXTS:
            for cand in (self.image_dir / f"{stem}{e}", self.image_dir / f"{stem}{e.upper()}"):
                if cand.exists():
                    return str(cand)
        return None

    def _mask(self, stem):
        p = self.label_dir / f"{stem}{self.label_suffix}{self.label_ext}"
        return tifffile.imread(str(p)) if p.exists() else None

    def prepare_workdir(self, workdir, items):
        (workdir / "inputs" / "masks").mkdir(parents=True, exist_ok=True)
        (workdir / "deliver" / "crops").mkdir(parents=True, exist_ok=True)
        for stem in items:
            src = self._src_image(stem)
            if src:
                shutil.copy2(src, workdir / "inputs" / Path(src).name)
            mp = self.label_dir / f"{stem}{self.label_suffix}{self.label_ext}"
            if mp.exists():
                shutil.copy2(mp, workdir / "inputs" / "masks" / f"{stem}_mask.tiff")
        (workdir / "TASK.md").write_text(_TASK, encoding="utf-8")

    def task_prompt(self, workdir, items):
        return ("Read ./TASK.md and complete it. For every image in ./inputs/ and every labeled "
                "cell in ./inputs/masks/<id>_mask.tiff, write a crop bbox to "
                "./deliver/crops/<id>.json. Cover every cell; minimize neighbor contamination while "
                "keeping the whole target cell. Work autonomously.")

    def score(self, workdir, items):
        crop_dir = workdir / "deliver" / "crops"
        per_item, n_ok = {}, 0
        for stem in items:
            mask = self._mask(stem)
            if mask is None:
                continue
            cp = crop_dir / f"{stem}.json"
            crops = []
            if cp.exists():
                try:
                    crops = json.load(open(cp))
                    n_ok += 1
                except Exception:
                    crops = []
            per_item[stem] = image_crop_metrics(mask, crops if isinstance(crops, list) else [])
        return ScoreResult(summary=macro_summary(per_item), per_item=per_item,
                           n_delivered=n_ok, n_expected=len(items))

    def feedback_prompt(self, score, history: list | None = None):
        s = score.summary
        worst = sorted(score.per_item.items(),
                       key=lambda kv: kv[1]["mean_neighbor_contamination"], reverse=True)[:5]
        lines = "\n".join(
            f"  - {k}: contamination={v['mean_neighbor_contamination']:.3f}, "
            f"retention={v['mean_target_retention']:.3f}, coverage={v['coverage']:.2f}" for k, v in worst)
        # Trajectory feedback (score = retention - contamination): show it by round + the
        # last-round delta, and tell the agent not to stop while it is still improving.
        scs = [float(h["summary"].get("score", 0) or 0) for h in (history or [])]
        if len(scs) >= 2:
            delta = scs[-1] - scs[-2]
            trend = "STILL IMPROVING -- keep tightening" if delta > 0.002 else "no gain last round"
            traj = (f"Score by round so far: {', '.join(f'{a:.3f}' for a in scs)} "
                    f"(last round {delta:+.3f} -- {trend}).")
        elif scs:
            traj = f"That was your first tuned round (score {scs[-1]:.3f}); try to push it higher."
        else:
            traj = ""
        return (
            f"Your crops scored mean_neighbor_contamination={s.get('mean_neighbor_contamination', 0):.3f} "
            f"(lower better), mean_target_retention={s.get('mean_target_retention', 0):.3f} (higher "
            f"better), coverage={s.get('coverage', 0):.2f}, score={s.get('score', 0):.3f} over "
            f"{score.n_delivered}/{score.n_expected}. {traj}\nWeakest images (high contamination = "
            f"boxes overlapping neighbors):\n{lines}\n\n"
            "IMPROVE: tighten boxes toward each cell (a tight bbox + small margin), ensure every "
            "cell is covered (retention near 1), and rewrite ./deliver/crops/<id>.json.\n\n"
            "Do NOT stop while the score is still rising. ONLY create an empty file ./deliver/STOP "
            "if a full round produced NO score improvement (it has clearly plateaued).")

    def materialize(self, deliver_dir, items, out_dir, score):
        """Per-image images/<id>/{metrics.json, deliverable_crop.png (mask + crop boxes)}."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        import matplotlib.patches as mpatches
        from skimage.color import label2rgb

        crop_dir = Path(deliver_dir) / "crops"
        for stem in items:
            imgdir = Path(out_dir) / "images" / stem
            imgdir.mkdir(parents=True, exist_ok=True)
            m = score.per_item.get(stem, {})
            (imgdir / "metrics.json").write_text(
                json.dumps({"image_id": stem, **m}, indent=2, default=str), encoding="utf-8")
            mask = self._mask(stem)
            cp = crop_dir / f"{stem}.json"
            if mask is None or not cp.exists():
                continue
            try:
                crops = json.load(open(cp))
                fig, ax = plt.subplots(figsize=(6, 6))
                ax.imshow(label2rgb(mask.astype("int64"), bg_label=0))
                for c in (crops if isinstance(crops, list) else []):
                    minr, minc, maxr, maxc = c["bbox"]
                    ax.add_patch(mpatches.Rectangle((minc, minr), maxc - minc, maxr - minr,
                                                    fill=False, edgecolor="white", lw=0.6))
                ax.set_title(f"{stem}  contam={m.get('mean_neighbor_contamination', 0):.3f}  "
                             f"retention={m.get('mean_target_retention', 0):.3f}  "
                             f"cells={m.get('n_cells')}", fontsize=9)
                ax.axis("off")
                fig.savefig(imgdir / "deliverable_crop.png", dpi=130, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

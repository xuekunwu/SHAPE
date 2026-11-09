"""Segmentation benchmark adapter (NeurIPS22-CellSeg). The agent gets microscopy images and must
deliver an instance-labeled mask per image; we score instance-AP against held-out GT. The agent
is told the task + output contract only -- it must discover/install the tool (Cellpose-SAM) itself.
"""
from __future__ import annotations

import glob
import os
import shutil
from pathlib import Path

import numpy as np
import tifffile

from .base import BenchmarkAdapter, ScoreResult
from ..metrics.instance_ap import image_metrics, macro_summary

_IMG_EXTS = (".bmp", ".png", ".tif", ".tiff", ".jpg", ".jpeg")

_TASK = """# Task: cell instance segmentation

You are given microscopy images of cells in `./inputs/`. For **every** image `<id>.<ext>` there,
produce an **instance segmentation mask**: a 2-D integer array the same height x width as the
input, where background is 0 and each distinct cell has its own unique positive integer id
(1, 2, 3, ...). Touching cells must get DIFFERENT ids.

Save each mask as `./deliver/masks/<id>_pred.tiff` (uint16, lossless TIFF, same H x W as input).
`<id>` is the input filename without extension (e.g. `inputs/cell_00007.bmp` -> `deliver/masks/cell_00007_pred.tiff`).

## How you are scored (you do NOT have the ground truth)
Instance Average Precision: each predicted cell is matched to a ground-truth cell by mask IoU;
AP = TP / (TP + FP + FN) averaged over IoU thresholds 0.50..0.95. So both the **number of cells**
and the **boundary accuracy** matter. Over-segmenting (splitting one cell into many) and
under-segmenting (merging cells) both hurt.

Stay inside ./inputs (read) and ./deliver (write) -- do not read, modify, or explore anything else
on the machine.
"""

# No SAM service provided -> the agent discovers/installs the tool itself.
_TOOLS_INSTALL = """
## Tools
Use any open-source method. For microscopy cell segmentation, a strong modern choice is
**Cellpose** (the "Cellpose-SAM" / cpsam model). Search online for how to install and run it, pick
sensible parameters, and apply it to every input. Install whatever you need (pip, etc.).
"""

# A SAM service IS provided -> MANDATORY: call it; installing/running anything locally is forbidden.
_SAM_SECTION = """
## Tools -- REQUIRED: use the provided Cellpose-SAM service (do NOT install or run anything locally)
A Cellpose-SAM inference server is running at **{url}** on the shared filesystem. You MUST use it.
Do NOT pip-install cellpose/torch, do NOT run segmentation or training locally, do NOT spawn heavy
compute -- just POST each image path + a mask output path; the server segments and writes the
instance mask (uint16 TIFF) to disk. Minimal client:

    import pickle, urllib.request
    def segment(img_path, mask_out, url="{url}"):
        body = pickle.dumps({{"img_path": img_path,
                              "kwargs": {{"diameter": 30, "channels": [0, 0],
                                          "flow_threshold": 0.4, "cellprob_threshold": 0.0,
                                          "min_size": 15}},
                              "mask_out": mask_out}})
        req = urllib.request.Request(url + "/eval_path", data=body,
                                     headers={{"Content-Type": "application/octet-stream"}})
        return pickle.loads(urllib.request.urlopen(req, timeout=600).read())
        # -> {{"mask_path": ..., "n_cells": int, "shape": [H, W]}}

Call `segment("inputs/<name>", "deliver/masks/<id>_pred.tiff")` for EVERY image. The service handles
concurrent requests, so segment ALL images in PARALLEL with a thread pool (~32 workers, e.g.
`concurrent.futures.ThreadPoolExecutor(32)`) instead of one at a time -- that is the fast way.
Then TUNE the kwargs (diameter to the cell size, flow/cellprob thresholds, min_size) across re-runs
to maximize instance AP (raise diameter for larger cells; lower cellprob_threshold to catch faint
cells, raise it to drop noise). The whole job is just concurrent calls to this service -- nothing else.
"""

_TASK_DONE = """
## Definition of done
`./deliver/masks/<id>_pred.tiff` exists for every image in `./inputs/`, each a valid instance mask.
"""


class SegmentationAdapter(BenchmarkAdapter):
    id = "segmentation"

    def __init__(self, data_root: Path, image_subdir="images", label_subdir="labels",
                 label_suffix="_label", label_ext=".tiff", sam_url=None):
        super().__init__(data_root)
        self.image_dir = self.data_root / image_subdir
        self.label_dir = self.data_root / label_subdir
        self.label_suffix = label_suffix
        self.label_ext = label_ext
        self.sam_url = sam_url            # if set, the agent CALLS this Cellpose-SAM service

    def select_items(self, limit=None, seed=0) -> list[str]:
        imgs = sorted(p for p in glob.glob(str(self.image_dir / "*"))
                      if os.path.splitext(p)[1].lower() in _IMG_EXTS)
        stems = [Path(p).stem for p in imgs]
        return stems[:limit] if limit else stems

    def _src_image(self, stem: str) -> str | None:
        for e in _IMG_EXTS:
            for cand in (self.image_dir / f"{stem}{e}", self.image_dir / f"{stem}{e.upper()}"):
                if cand.exists():
                    return str(cand)
        return None

    def prepare_workdir(self, workdir: Path, items: list[str]) -> None:
        (workdir / "inputs").mkdir(parents=True, exist_ok=True)
        (workdir / "deliver" / "masks").mkdir(parents=True, exist_ok=True)
        for stem in items:
            src = self._src_image(stem)
            if src:
                shutil.copy2(src, workdir / "inputs" / Path(src).name)
        tools = _SAM_SECTION.format(url=self.sam_url) if self.sam_url else _TOOLS_INSTALL
        (workdir / "TASK.md").write_text(_TASK + tools + _TASK_DONE, encoding="utf-8")

    def task_prompt(self, workdir: Path, items: list[str]) -> str:
        return ("Read ./TASK.md and complete it. Segment every image in ./inputs/ and write one "
                "instance mask per image to ./deliver/masks/<id>_pred.tiff (uint16 TIFF). Do not "
                "stop until every input has a mask. Work autonomously; install what you need.")

    def _gt(self, stem: str):
        p = self.label_dir / f"{stem}{self.label_suffix}{self.label_ext}"
        return tifffile.imread(str(p)) if p.exists() else None

    def score(self, workdir: Path, items: list[str]) -> ScoreResult:
        mask_dir = workdir / "deliver" / "masks"
        per_item, n_ok = {}, 0
        for stem in items:
            gt = self._gt(stem)
            if gt is None:
                continue
            mp = mask_dir / f"{stem}_pred.tiff"
            if not mp.exists():
                # missing delivery scores as all-false-negatives
                per_item[stem] = image_metrics(np.zeros_like(gt), gt)
                continue
            try:
                pred = tifffile.imread(str(mp))
                per_item[stem] = image_metrics(pred, gt)
                n_ok += 1
            except Exception:
                per_item[stem] = image_metrics(np.zeros_like(gt), gt)
        return ScoreResult(summary=macro_summary(per_item), per_item=per_item,
                           n_delivered=n_ok, n_expected=len(items))

    def materialize(self, deliver_dir, items, out_dir, score):
        """SHAPE-parallel per-image channel: out_dir/images/<id>/{metrics.json, deliverable_seg.png}
        (raw | predicted labels | pred=cyan/GT=magenta overlay). deliver_dir holds the chosen
        iteration's masks (<id>_pred.tiff)."""
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from skimage.color import label2rgb
        from skimage.segmentation import find_boundaries
        from skimage.io import imread

        mask_dir = Path(deliver_dir) / "masks"
        for stem in items:
            imgdir = Path(out_dir) / "images" / stem
            imgdir.mkdir(parents=True, exist_ok=True)
            m = score.per_item.get(stem, {})
            gt_path = self.label_dir / f"{stem}{self.label_suffix}{self.label_ext}"
            (imgdir / "metrics.json").write_text(
                __import__("json").dumps({"image_id": stem, "ground_truth_path": str(gt_path),
                                          **m}, indent=2, default=str), encoding="utf-8")
            mp = mask_dir / f"{stem}_pred.tiff"
            src = self._src_image(stem)
            if not mp.exists() or not src:
                continue
            try:
                img = imread(src)
                pred = tifffile.imread(str(mp)).astype("int64")
                gt = self._gt(stem)
                base = img if img.ndim == 3 else np.stack([img] * 3, -1)
                base = base.astype(np.float32)
                base = (base - base.min()) / (base.ptp() + 1e-8)
                panels = [("input", "raw"), ("prediction", "label")]
                if gt is not None:
                    panels.append(("overlay: pred=cyan, GT=magenta", "overlay"))
                fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 4.4))
                axes = np.atleast_1d(axes).ravel()
                for ax, (title, kind) in zip(axes, panels):
                    if kind == "raw":
                        ax.imshow(base)
                    elif kind == "label":
                        ax.imshow(label2rgb(pred, bg_label=0))
                    else:
                        ov = base.copy()
                        ov[find_boundaries(pred, mode="outer")] = [0.0, 1.0, 1.0]
                        ov[find_boundaries(gt.astype("int64"), mode="outer")] = [1.0, 0.0, 1.0]
                        ax.imshow(ov)
                    ax.set_title(title, fontsize=8); ax.axis("off")
                fig.suptitle(f"{stem}  AP50={m.get('ap50', 0):.3f}  F1@50={m.get('f1_at_50', 0):.3f}  "
                             f"pred={m.get('predicted_cell_count')} gt={m.get('ground_truth_cell_count')}",
                             fontsize=9)
                fig.savefig(imgdir / "deliverable_seg.png", dpi=130, bbox_inches="tight")
                plt.close(fig)
            except Exception:
                pass

    def feedback_prompt(self, score: ScoreResult, history: list | None = None) -> str:
        s = score.summary
        worst = sorted(score.per_item.items(), key=lambda kv: kv[1]["ap50"])[:5]
        lines = "\n".join(
            f"  - {k}: AP50={v['ap50']:.3f}, predicted {v['predicted_cell_count']} cells vs "
            f"GT {v['ground_truth_cell_count']}" for k, v in worst)
        # Trajectory feedback: show the AP50 by round + the last-round delta, so the agent
        # can tell whether it is still climbing -- and is told NOT to stop while it is.
        aps = [float(h["summary"].get("ap50", 0) or 0) for h in (history or [])]
        if len(aps) >= 2:
            delta = aps[-1] - aps[-2]
            trend = "STILL IMPROVING -- keep tuning" if delta > 0.005 else \
                    "no meaningful gain last round" if delta <= 0.005 else ""
            traj = (f"AP50 by round so far: {', '.join(f'{a:.3f}' for a in aps)} "
                    f"(last round {delta:+.3f} -- {trend}).")
        elif aps:
            traj = (f"That was your FIRST tuned round (AP50 {aps[-1]:.3f}, up from the ~0.70 "
                    f"default-parameter baseline). There is almost always more to gain from "
                    f"further tuning -- do not stop yet.")
        else:
            traj = ""
        return (
            f"Your masks scored macro AP50={s.get('ap50', 0):.3f}, AP75={s.get('ap75', 0):.3f}, "
            f"mAP(50-95)={s.get('map_50_95', 0):.3f} over {score.n_delivered}/{score.n_expected} "
            f"delivered. {traj}\nYour weakest images (cell-count mismatch usually means over/"
            f"under-segmentation):\n{lines}\n\n"
            "Diagnose and IMPROVE: adjust the model/params (cell diameter, flow/cellprob "
            "thresholds, normalization) or method, re-run on ALL inputs, and overwrite the masks in "
            "./deliver/masks/. Try a genuinely different setting each round (a bigger diameter "
            "sweep, per-image adaptation, different normalization) rather than a tiny tweak.\n\n"
            "Do NOT stop while AP50 is still rising. ONLY create an empty file ./deliver/STOP if a "
            "full round of tuning produced NO AP50 improvement (the score has clearly plateaued).")

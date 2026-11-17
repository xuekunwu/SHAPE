"""Single-cell crop QC -- the benchmark's copy of SHAPE's GT-free crop metrics. Given the instance mask and a
crop bounding box for a target cell, measure how clean the crop is:

  neighbor_contamination = (foreground pixels of OTHER cells in the box) / (all foreground in box)
  target_retention       = (target-cell pixels in the box) / (target cell's total area)

Lower contamination + higher retention = a cleaner single-cell crop. Mirrors
shape/tools/single_cell_cropper/tool.py.
"""
from __future__ import annotations

import numpy as np

_EPS = 1e-9


def crop_qc(labeled_mask: np.ndarray, bbox, cell_label: int) -> dict:
    """bbox = [minr, minc, maxr, maxc] (half-open). Returns the per-crop QC dict."""
    minr, minc, maxr, maxc = (int(round(x)) for x in bbox)
    H, W = labeled_mask.shape[:2]
    minr, minc = max(0, minr), max(0, minc)
    maxr, maxc = min(H, maxr), min(W, maxc)
    if maxr <= minr or maxc <= minc:
        return {"neighbor_contamination": 1.0, "target_retention": 0.0, "occupancy": 0.0,
                "target_dominance": 0.0, "valid": False}
    crop = labeled_mask[minr:maxr, minc:maxc]
    crop_area = crop.shape[0] * crop.shape[1]
    fg = int(np.count_nonzero(crop))
    other = int(np.count_nonzero((crop != 0) & (crop != cell_label)))
    target_in_crop = int(np.count_nonzero(crop == cell_label))
    total_target = int(np.count_nonzero(labeled_mask == cell_label)) or 1
    return {
        "neighbor_contamination": other / (fg + _EPS),
        "target_retention": min(1.0, target_in_crop / total_target),
        "occupancy": target_in_crop / crop_area,
        "target_dominance": (fg - other) / (fg + _EPS),
        "valid": True,
    }


def naive_bbox(labeled_mask: np.ndarray, cell_label: int, margin_frac: float = 0.20):
    """The naive MaskCrop reference (SHAPE's arm A): a cell's tight bounding box grown by
    margin_frac of its equivalent diameter. Returns [minr, minc, maxr, maxc] or None."""
    ys, xs = np.where(labeled_mask == cell_label)
    if ys.size == 0:
        return None
    area = ys.size
    m = margin_frac * (4.0 * area / np.pi) ** 0.5
    return [ys.min() - m, xs.min() - m, ys.max() + 1 + m, xs.max() + 1 + m]


def image_crop_metrics(labeled_mask: np.ndarray, crops: list[dict], margin_frac=0.20) -> dict:
    """Per-image crop QC for the agent's crops, PLUS the naive-MaskCrop reference so the output is
    directly comparable to SHAPE (contamination_maskcrop -> contamination_shape -> reduction)."""
    labels = [int(l) for l in np.unique(labeled_mask) if l]
    # agent's crops (== SHAPE's "shape" arm)
    agent = [crop_qc(labeled_mask, c["bbox"], int(c["cell_label"])) for c in crops if "bbox" in c]
    # naive MaskCrop reference (== SHAPE's "maskcrop" arm), one tight+margin box per labeled cell
    naive = [crop_qc(labeled_mask, nb, l) for l in labels
             if (nb := naive_bbox(labeled_mask, l, margin_frac)) is not None]

    def mean(rows, k):
        return float(np.mean([r[k] for r in rows])) if rows else 0.0

    c_shape = mean(agent, "neighbor_contamination")
    c_mask = mean(naive, "neighbor_contamination")
    return {
        "n_crops": len(agent),
        "n_cells": len(labels),
        "coverage": len({int(c["cell_label"]) for c in crops if "cell_label" in c} & set(labels)) / (len(labels) or 1),
        # SHAPE-comparable keys (naive MaskCrop arm vs the agent's crops):
        "contamination_maskcrop": c_mask,
        "contamination_shape": c_shape,
        "contamination_reduction": c_mask - c_shape,
        "retention_maskcrop": mean(naive, "target_retention"),
        "retention_shape": mean(agent, "target_retention"),
        "occupancy_maskcrop": mean(naive, "occupancy"),
        "occupancy_shape": mean(agent, "occupancy"),
        # baseline-native aliases (kept for continuity):
        "mean_neighbor_contamination": c_shape,
        "mean_target_retention": mean(agent, "target_retention"),
        "mean_occupancy": mean(agent, "occupancy"),
        "mean_target_dominance": mean(agent, "target_dominance"),
    }


def macro_summary(per_image: dict) -> dict:
    keys = ["contamination_maskcrop", "contamination_shape", "contamination_reduction",
            "retention_maskcrop", "retention_shape", "mean_neighbor_contamination",
            "mean_target_retention", "mean_occupancy", "coverage"]
    vals = {k: float(np.mean([m.get(k, 0.0) for m in per_image.values()]))
            for k in keys if per_image}
    vals["n_images"] = len(per_image)
    # single scalar the loop maximizes: retention minus contamination (cleaner + complete crops).
    vals["score"] = round(vals.get("mean_target_retention", 0) - vals.get("mean_neighbor_contamination", 0), 4)
    return vals

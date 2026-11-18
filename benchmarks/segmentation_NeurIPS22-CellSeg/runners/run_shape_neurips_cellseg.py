#!/usr/bin/env python3
"""NeurIPS22-CellSeg benchmark: the FULL SHAPE agent vs Cellpose-SAM alone.

The benchmark feeds the image set to the real agent (`shape.solver.solve` -- planner /
executor / verifier / memory, LLM + VLRR active) with a counting query that keeps it at
segmentation, then scores the agent's per-image masks against ground-truth instance masks.
It also runs Cellpose-SAM ALONE (the tool, default params, no agent) as the comparison
foil, and plots agent-vs-tool accuracy + runtime. Labels are used only at scoring time;
LLM nondeterminism is part of what is measured (no deterministic mode).

Data layout (defaults; both subfolders can be overridden):
    <data-root>/images/<name>.{png,bmp,tif,tiff,jpg}   # input images
    <data-root>/labels/<name>_label.tiff               # ground-truth instance masks

Random subset (the test set is ~1000 images):
    --sample-frac 0.2        # seeded random 20%
    --seed 0                 # reproducible draw
    --limit 50               # hard cap, applied after sampling

Example:
    python benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_shape_neurips_cellseg.py \
        --data-root /scratch/users/your-user/shape_benchmark/NeurIPS22-CellSeg \
        --sample-frac 0.2 --seed 0
"""

from __future__ import annotations

import argparse
import json
import os
import random
import sys
import time
from pathlib import Path

import numpy as np

from shape.solver import solve
from shape.engine import usage as token_usage
from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "_shared"))   # benchmarks/_shared
import segmentation_metrics as sm  # noqa: E402  -- benchmarks/_shared/ module (efficiency metrics)


IOU_THRESHOLDS = tuple(round(value, 2) for value in np.arange(0.50, 1.00, 0.05))
# NeurIPS22-CellSeg images come in mixed formats; labels live in a separate folder.
_IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}


# --------------------------------------------------------------------------- #
# Ground-truth / prediction loading + instance-segmentation evaluation
# --------------------------------------------------------------------------- #
def _read_image_array(path: str) -> np.ndarray:
    """Read a TIFF/PNG/BMP label image as a raw integer array (no rescaling)."""
    try:
        from skimage.io import imread
        return np.asarray(imread(path))
    except Exception:
        from PIL import Image
        with Image.open(path) as image:
            return np.asarray(image)


def load_label_mask(path) -> np.ndarray:
    """Load an instance-label map from .npy (segmenter predictions) or an image file
    (.tif/.tiff/.png/.bmp ground truth), returned as a 2-D integer array."""
    path = str(path)
    if path.lower().endswith(".npy"):
        loaded = np.load(path, allow_pickle=True)
        if isinstance(loaded, np.ndarray) and loaded.shape == () and loaded.dtype == object:
            payload = loaded.item()
            if isinstance(payload, dict):
                for key in ("masks", "mask", "labels"):
                    if key in payload:
                        return np.asarray(payload[key]).squeeze()
        return np.asarray(loaded).squeeze()
    arr = np.asarray(_read_image_array(path)).squeeze()
    if arr.ndim == 3:                       # a label image stored as RGB -> 1st channel
        arr = arr[..., 0]
    return arr


def _instance_iou_matrix(ground_truth: np.ndarray, prediction: np.ndarray) -> np.ndarray:
    """Return pairwise IoU for non-background instance labels."""
    from scipy import sparse

    ground_truth = np.asarray(ground_truth).squeeze()
    prediction = np.asarray(prediction).squeeze()
    if ground_truth.shape != prediction.shape:
        raise ValueError(
            f"Ground-truth shape {ground_truth.shape} != prediction shape {prediction.shape}"
        )
    gt_labels, gt_inverse = np.unique(ground_truth, return_inverse=True)
    pred_labels, pred_inverse = np.unique(prediction, return_inverse=True)
    overlap = sparse.coo_matrix(
        (
            np.ones(ground_truth.size, dtype=np.int64),
            (gt_inverse.ravel(), pred_inverse.ravel()),
        ),
        shape=(len(gt_labels), len(pred_labels)),
    ).toarray()
    gt_keep = gt_labels != 0
    pred_keep = pred_labels != 0
    intersections = overlap[np.ix_(gt_keep, pred_keep)].astype(np.float64)
    gt_areas = overlap[gt_keep, :].sum(axis=1, keepdims=True)
    pred_areas = overlap[:, pred_keep].sum(axis=0, keepdims=True)
    unions = gt_areas + pred_areas - intersections
    return np.divide(
        intersections, unions,
        out=np.zeros_like(intersections), where=unions > 0,
    )


def _match_instances(iou: np.ndarray, threshold: float) -> tuple[int, list[float]]:
    """Maximum-cardinality matching at an IoU threshold, then maximum total IoU."""
    from scipy.optimize import linear_sum_assignment

    if not iou.size:
        return 0, []
    valid = iou >= threshold
    cardinality_bonus = min(iou.shape) + 1.0
    rows, cols = linear_sum_assignment(-(valid.astype(np.float64) * cardinality_bonus + iou))
    accepted = valid[rows, cols]
    return int(accepted.sum()), iou[rows[accepted], cols[accepted]].tolist()


def evaluate_instance_segmentation(prediction_path: str, ground_truth_path: Path) -> dict:
    """Compute benchmark-only instance segmentation metrics (AP/precision/recall/F1
    across IoU 0.50-0.95), matching the LIVECell benchmark's definitions."""
    prediction = load_label_mask(prediction_path)
    ground_truth = load_label_mask(ground_truth_path)
    iou = _instance_iou_matrix(ground_truth, prediction)
    n_gt, n_pred = iou.shape
    by_threshold = {}
    for threshold in IOU_THRESHOLDS:
        true_positive, matched_ious = _match_instances(iou, threshold)
        false_positive = n_pred - true_positive
        false_negative = n_gt - true_positive
        precision = true_positive / max(1, true_positive + false_positive)
        recall = true_positive / max(1, true_positive + false_negative)
        f1 = 2 * precision * recall / max(1e-12, precision + recall)
        average_precision = true_positive / max(
            1, true_positive + false_positive + false_negative)
        by_threshold[f"{threshold:.2f}"] = {
            "true_positive": true_positive,
            "false_positive": false_positive,
            "false_negative": false_negative,
            "average_precision": average_precision,
            "precision": precision,
            "recall": recall,
            "f1": f1,
            "matched_ious": matched_ious,
        }
    at_50 = by_threshold["0.50"]
    all_matched = at_50["matched_ious"]
    return {
        "ap_definition": "TP / (TP + FP + FN) for instance masks without confidence scores",
        "ground_truth_path": str(ground_truth_path),
        "ground_truth_cell_count": n_gt,
        "predicted_cell_count": n_pred,
        "ap50": at_50["average_precision"],
        "ap75": by_threshold["0.75"]["average_precision"],
        "map_50_95": float(np.mean([v["average_precision"] for v in by_threshold.values()])),
        "precision_at_50": at_50["precision"],
        "recall_at_50": at_50["recall"],
        "f1_at_50": at_50["f1"],
        "mean_matched_iou_at_50": float(np.mean(all_matched)) if all_matched else 0.0,
        "cell_count_error_fraction": abs(n_pred - n_gt) / max(1, n_gt),
        "thresholds": by_threshold,
    }


def aggregate_segmentation_metrics(per_image: list[dict]) -> dict:
    """Micro-aggregate instance counts across images and report dataset metrics."""
    if not per_image:
        return {"evaluable": False, "reason": "No images had ground-truth masks."}
    aggregate_thresholds = {}
    for threshold in IOU_THRESHOLDS:
        key = f"{threshold:.2f}"
        entries = [item["thresholds"][key] for item in per_image]
        tp = sum(item["true_positive"] for item in entries)
        fp = sum(item["false_positive"] for item in entries)
        fn = sum(item["false_negative"] for item in entries)
        precision = tp / max(1, tp + fp)
        recall = tp / max(1, tp + fn)
        aggregate_thresholds[key] = {
            "true_positive": tp, "false_positive": fp, "false_negative": fn,
            "average_precision": tp / max(1, tp + fp + fn),
            "precision": precision, "recall": recall,
            "f1": 2 * precision * recall / max(1e-12, precision + recall),
        }
    at_50 = aggregate_thresholds["0.50"]
    return {
        "evaluable": True,
        "ap_definition": "TP / (TP + FP + FN) for instance masks without confidence scores",
        "n_images": len(per_image),
        "ap50": at_50["average_precision"],
        "ap75": aggregate_thresholds["0.75"]["average_precision"],
        "map_50_95": float(np.mean(
            [v["average_precision"] for v in aggregate_thresholds.values()])),
        "precision_at_50": at_50["precision"],
        "recall_at_50": at_50["recall"],
        "f1_at_50": at_50["f1"],
        "mean_predicted_cells": float(np.mean(
            [item["predicted_cell_count"] for item in per_image])),
        "mean_ground_truth_cells": float(np.mean(
            [item["ground_truth_cell_count"] for item in per_image])),
        "mean_cell_count_error_fraction": float(np.mean(
            [item["cell_count_error_fraction"] for item in per_image])),
        "thresholds": aggregate_thresholds,
    }


def _select_images(images, sample_frac, limit, seed):
    """Optionally draw a seeded random fraction, then apply a hard cap."""
    selected = list(images)
    if sample_frac is not None and 0.0 < float(sample_frac) < 1.0:
        k = max(1, round(float(sample_frac) * len(selected)))
        selected = sorted(random.Random(seed).sample(selected, k), key=lambda p: p.name)
    if limit is not None:
        selected = selected[:limit]
    return selected


def run_cellpose_baseline(segmenter, image_path: Path, gt_path: Path,
                          baseline_cache: Path) -> dict:
    """Run the Cellpose-SAM tool ALONE on the raw image and score it.

    This is the no-agent baseline: one call with the tool's default parameters, no
    preprocessing VLRR, no mask-measured diameter, no vision judge, no retune. Returns
    its instance-segmentation benchmark (or a reason), wall-clock runtime, and cell
    count -- the foil that the SHAPE agent's numbers are compared against.

    On current main the segmenter's former internal auto-preprocess path has been
    removed, so this baseline always receives the source image directly.
    """
    baseline_cache.mkdir(parents=True, exist_ok=True)
    t0 = time.perf_counter()
    try:
        result = segmenter.execute(image=str(image_path),
                                   query_cache_dir=str(baseline_cache),
                                   image_id=image_path.stem)
    except Exception as exc:  # the tool itself crashed on this image
        return {"benchmark": {"evaluable": False, "reason": f"baseline error: {exc}"},
                "runtime_seconds": time.perf_counter() - t0, "cell_count": 0}
    runtime = time.perf_counter() - t0
    cell_count = result.get("cell_count", 0) if isinstance(result, dict) else 0
    mask_path = result.get("mask_path") if isinstance(result, dict) else None
    if mask_path and gt_path.is_file():
        try:
            benchmark = evaluate_instance_segmentation(mask_path, gt_path)
            benchmark["image_name"] = image_path.stem
        except Exception as exc:
            benchmark = {"evaluable": False, "reason": str(exc)}
    else:
        benchmark = {"evaluable": False,
                     "reason": "no mask" if not mask_path else "Ground-truth mask not found."}
    return {"benchmark": benchmark, "runtime_seconds": runtime, "cell_count": cell_count}


def _nature_style() -> None:
    """Apply Nature-journal matplotlib defaults (Arial 7pt, thin axes, no top/right
    spines, 300 dpi). Called at the top of every figure-generating function."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif",
        "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7,
        "xtick.labelsize": 6, "ytick.labelsize": 6, "legend.fontsize": 6,
        "axes.linewidth": 0.5, "xtick.major.width": 0.5, "ytick.major.width": 0.5,
        "axes.spines.top": False, "axes.spines.right": False,
        "savefig.dpi": 300, "figure.dpi": 300,
    })


_COMPARE_METRICS = [("ap50", "AP50"), ("ap75", "AP75"), ("map_50_95", "mAP"),
                    ("f1_at_50", "F1@50"), ("recall_at_50", "Recall@50")]
_MACRO_KEYS = ["ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50", "f1_at_50"]
_BASE_COLOR, _SHAPE_COLOR = "#9ecae1", "#3182bd"


def macro_metrics(per_image: list[dict]) -> dict:
    """Macro average: the mean of the PER-IMAGE metrics, every image weighted equally.

    Preferred over the micro aggregate (which pools all cells and is dominated by a few
    high-count images) for a dataset whose cell counts span orders of magnitude -- it
    reflects typical performance across diverse images, so a big win on a small image
    (e.g. a rescued off-scale field) is not drowned out.
    """
    if not per_image:
        return {k: 0.0 for k in _MACRO_KEYS} | {"n_images": 0, "evaluable": False}
    out = {k: float(np.mean([float(im.get(k, 0) or 0) for im in per_image]))
           for k in _MACRO_KEYS}
    out["n_images"] = len(per_image)
    out["evaluable"] = True
    return out


def _efficiency_panel(ax, eff: dict) -> None:
    """SHAPE agent-efficiency panel: retune decision-quality bars (0-1) + the per-image cost
    profile as text (it is SHAPE-only -- Cellpose alone makes no tool calls)."""
    rate_keys = [("positive_tool_call_rate", "positive retune"),
                 ("wasted_tool_call_rate", "wasted retune")]
    xv = np.arange(len(rate_keys))
    vals = [float(eff.get(k) or 0) for k, _ in rate_keys]
    bars = ax.bar(xv, vals, width=0.5, color=[_SHAPE_COLOR, _BASE_COLOR])
    ax.set_xticks(xv)
    ax.set_xticklabels([lbl for _, lbl in rate_keys])
    ax.set_ylim(0, 1.05)
    ax.set_xlim(-0.6, 4.6)
    ax.set_ylabel("Rate (of retunes)")
    ax.set_title("Agent efficiency (SHAPE)")
    for rect in bars:
        ax.annotate(f"{rect.get_height():.2f}",
                    (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                    ha="center", va="bottom", fontsize=5)

    def g(k, fmt="{:.2f}"):
        v = eff.get(k)
        return "n/a" if v is None else fmt.format(v)
    lines = [
        f"tool calls / img:   {g('tool_calls_per_image')}",
        f"retunes / img:      {g('retunes_per_image')}",
        f"tokens / img:       {g('token_cost_per_image', '{:.0f}')}",
        f"time / img:         {g('time_per_image_s', '{:.1f}')} s",
        f"AP50 gain / call:   {g('ap50_gain_per_tool_call', '{:.3f}')}",
        f"AP50 gain / 1k tok: {g('ap50_gain_per_1k_tokens', '{:.4f}')}",
        f"AP50 gain / min:    {g('ap50_gain_per_minute', '{:.3f}')}",
    ]
    ax.text(0.40, 0.98, "\n".join(lines), transform=ax.transAxes, fontsize=6,
            va="top", ha="left", family="monospace")


def plot_comparison(base_agg: dict, shape_agg: dict,
                    base_mean_runtime: float, shape_mean_runtime: float,
                    out_path: Path, efficiency: dict | None = None) -> None:
    """Comparison figure (Nature style, 300 dpi): TOP = accuracy + runtime (Cellpose-SAM vs
    SHAPE); BOTTOM = the SHAPE agent-efficiency panel (retune quality + per-image cost)."""
    _nature_style()
    import matplotlib.pyplot as plt

    fig = plt.figure(figsize=(7.2, 4.4))
    gs = fig.add_gridspec(2, 2, width_ratios=[3, 1], height_ratios=[1, 1],
                          hspace=0.7, wspace=0.3)
    ax_acc = fig.add_subplot(gs[0, 0])
    ax_rt = fig.add_subplot(gs[0, 1])
    ax_eff = fig.add_subplot(gs[1, :])
    x = np.arange(len(_COMPARE_METRICS))
    width = 0.38
    base_vals = [float(base_agg.get(k, 0) or 0) for k, _ in _COMPARE_METRICS]
    shape_vals = [float(shape_agg.get(k, 0) or 0) for k, _ in _COMPARE_METRICS]
    b1 = ax_acc.bar(x - width / 2, base_vals, width, label="Cellpose-SAM", color=_BASE_COLOR)
    b2 = ax_acc.bar(x + width / 2, shape_vals, width, label="SHAPE (agentic)", color=_SHAPE_COLOR)
    ax_acc.set_xticks(x)
    ax_acc.set_xticklabels([lbl for _, lbl in _COMPARE_METRICS])
    ax_acc.set_ylabel("Score")
    ax_acc.set_ylim(0, 1.05)
    ax_acc.legend(frameon=False, loc="upper right")
    ax_acc.set_title("Instance segmentation accuracy (macro: per-image mean)")
    for bars in (b1, b2):
        for rect in bars:
            ax_acc.annotate(f"{rect.get_height():.2f}",
                            (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                            ha="center", va="bottom", fontsize=5)

    rt_vals = [base_mean_runtime, shape_mean_runtime]
    rb = ax_rt.bar([0, 1], rt_vals, width=0.6, color=[_BASE_COLOR, _SHAPE_COLOR])
    ax_rt.set_xticks([0, 1])
    ax_rt.set_xticklabels(["Cellpose-SAM", "SHAPE"], rotation=20, ha="right")
    ax_rt.set_ylabel("Mean runtime (s / image)")
    ax_rt.set_title("Runtime")
    for rect in rb:
        ax_rt.annotate(f"{rect.get_height():.1f}s",
                       (rect.get_x() + rect.get_width() / 2, rect.get_height()),
                       ha="center", va="bottom", fontsize=5)

    _efficiency_panel(ax_eff, efficiency or {})

    fig.suptitle("Cellpose-SAM vs SHAPE on NeurIPS22-CellSeg", fontsize=8)
    fig.savefig(out_path, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# Pull the agent's outputs out of its memory for scoring
# --------------------------------------------------------------------------- #
def _collect_masks_by_image(memory) -> dict:
    """image_id -> predicted mask path, from the agent's segmentation step(s).

    A multi-image segmentation step records ``result={"per_image": [{image_id, mask_path,
    ...}]}``; a single-image one records a flat dict. Either way we pull the predicted
    label-mask path keyed by image so it can be scored against that image's GT mask."""
    masks: dict = {}
    for action in memory.get_actions(llm_safe=False):
        result = action.get("result")
        if not isinstance(result, dict):
            continue
        entries = (result["per_image"]
                   if isinstance(result.get("per_image"), list) else [result])
        for entry in entries:
            if isinstance(entry, dict) and entry.get("mask_path") and entry.get("image_id"):
                masks[entry["image_id"]] = entry["mask_path"]
    return masks


def _iter_stages(verification_summary):
    """Every stage dict across a (flat or per_image) verification_summary."""
    for record in verification_summary or []:
        if not isinstance(record, dict):
            continue
        groups = (record.get("per_image") if isinstance(record.get("per_image"), list)
                  else [record])
        for grp in groups:
            if isinstance(grp, dict):
                for stage in (grp.get("stages") or []):
                    yield stage


def _summarize_agent_activity(verification_summary) -> dict:
    """Aggregate the agent's VLRR activity from solve()'s verification_summary so the eval
    reflects how much the system actually reasoned (within-stage retunes, vision judging,
    escalations) -- not just final outputs."""
    activity = {"records": len(verification_summary or []), "segmentation_retuned": 0,
                "vision_judged": 0, "escalations": 0}
    for stage in _iter_stages(verification_summary):
        trials = stage.get("trials") or []
        if stage.get("stage") == "segmentation" and (stage.get("n_trials") or 0) > 1:
            activity["segmentation_retuned"] += 1
        if any("[vision]" in str(t.get("notes", "")) for t in trials):
            activity["vision_judged"] += 1
        action = stage.get("action")
        escalated = isinstance(action, str) and action.startswith("escalate")
        if escalated:
            activity["escalations"] += 1
    return activity


def _image_trace(verification_summary) -> dict:
    """Per-image escalation, the vision judge's first image profile and a compact stage trace (for
    run_summary['agent_trace']), so a past run's VLRR decisions are recoverable without re-running."""
    escalated = False
    judge: dict = {}
    stages_trace: list = []
    for stage in _iter_stages(verification_summary):
        trials = stage.get("trials") or []
        for t in trials:
            if not judge and t.get("judge_cell_size"):
                judge = {"cell_size": t.get("judge_cell_size"),
                         "cell_density": t.get("judge_cell_density")}
        if str(stage.get("action") or "").startswith("escalate"):
            escalated = True
        stages_trace.append({
            "stage": stage.get("stage"), "action": stage.get("action"),
            "n_trials": stage.get("n_trials"),
            "trials": [{
                "param_delta": t.get("param_delta"),
                "judge_cell_size": t.get("judge_cell_size"),
                "judge_cell_density": t.get("judge_cell_density"),
                "note": str(t.get("notes") or "")[:160],
            } for t in trials],
        })
    return {"escalated": escalated, "judge_profile": judge, "stages": stages_trace}


def _safe_ap50(mask_path, gt_path):
    """AP50 of one mask vs GT, or None on any failure (for the per-call efficiency trajectory)."""
    try:
        return float(evaluate_instance_segmentation(mask_path, gt_path).get("ap50"))
    except Exception:
        return None


# Default query: the professional, correctly-scoped Prompt.txt (states the
# segmentation-only boundary explicitly -- do NOT crop/cluster/analyze), read at
# runtime so it stays in sync with the file on disk. `--prompt-file` overrides it.
_PROMPT_FILE = Path(__file__).resolve().parents[1] / "Prompt.txt"

# Task 083: this benchmark (the "shape_main" arm) tests the SEGMENTOR only.
# Image_Preprocessor_Tool is intentionally excluded from `enabled_tools` so the LLM
# planner can never pick it as an explicit step. Current main also removed the
# segmenter's former internal auto-preprocess branch, so no hidden preprocessing
# substitution is possible in any benchmark arm.
_SHAPE_MAIN_TOOLS = [Cell_Segmenter_Tool.__name__]


def _default_question() -> str:
    # The task prompt is a benchmark input: a missing Prompt.txt is an error (FileNotFoundError), as it is for the
    # gpt arm, never a silent simpler question (which once ran the shape and react arms on the wrong task).
    return _PROMPT_FILE.read_text(encoding="utf-8")


def run_batch(image_dir: Path, label_dir: Path, output_root: Path, *,
              label_suffix: str = "_label", label_ext: str = ".tiff",
              sample_frac: float | None = None, limit: int | None = None, seed: int = 0,
              model: str = "gpt-4o", run_baseline: bool = True,
              image_names: set[str] | None = None, max_steps: int = 8,
              verify_refine: bool = False, question: str | None = None) -> dict:
    effective_question = question if question is not None else _default_question()
    all_images = sorted(
        p for p in image_dir.iterdir()
        if p.is_file() and p.suffix.lower() in _IMAGE_EXTS
        and not p.stem.endswith(label_suffix))
    if not all_images:
        raise ValueError(f"No images ({sorted(_IMAGE_EXTS)}) found in {image_dir}")
    if image_names:
        # restrict to an explicit set of image stems (e.g. the worst-N by a baseline)
        all_images = [p for p in all_images if p.stem in image_names]
        missing = image_names - {p.stem for p in all_images}
        if missing:
            print(f"[batch] WARNING: {len(missing)} listed images not found: "
                  f"{sorted(missing)[:5]}{' ...' if len(missing) > 5 else ''}")
        if not all_images:
            raise ValueError("None of the listed images were found in the image dir.")
    total_available = len(all_images)
    images = _select_images(all_images, sample_frac, limit, seed)
    if not images:
        raise ValueError("Sampling selected 0 images; check --sample-frac/--limit.")

    output_root.mkdir(parents=True, exist_ok=True)
    print(f"[batch] {len(images)}/{total_available} images selected "
          f"(sample_frac={sample_frac}, limit={limit}, seed={seed})")

    def _gt(image_path: Path) -> Path:
        return label_dir / f"{image_path.stem}{label_suffix}{label_ext}"

    # Provider-aware key: gpt-* -> LLM_API_KEY, claude-* -> ANTHROPIC_API_KEY,
    # glm-* -> ZHIPUAI_API_KEY (so --model glm-5.1 picks up the right env var).
    from shape.engine.factory import provider_of, resolve_api_key
    api_key = resolve_api_key(model)
    if not api_key:
        key_name = {"openai": "LLM_API_KEY", "anthropic": "ANTHROPIC_API_KEY",
                    "glm": "ZHIPUAI_API_KEY"}.get(provider_of(model),
                                                  "the provider's API key env var")
        print(f"[batch] WARNING: {key_name} not set -- the agent's LLM planner/verifier "
              "cannot run. This benchmark evaluates the FULL agent and needs an API key.")
    print("[batch] Cellpose-SAM baseline: " + ("ON" if run_baseline else "OFF"))
    segmenter = Cell_Segmenter_Tool() if run_baseline else None

    baseline_results: list[dict] = []
    baseline_times: list[float] = []
    shape_results: list[dict] = []
    shape_times: list[float] = []
    all_verification: list = []
    agent_trace: dict = {}                      # image_id -> compact per-stage VLRR trace
    efficiency_records: list = []               # per-image records for segmentation_metrics

    # Per image: Cellpose-SAM ALONE -> cellpose/<id>/ ; the FULL agent -> shape/<id>/ ,
    # so each image's masks AND every retune overlay live in its own folder.
    for index, image_path in enumerate(images, start=1):
        image_id = image_path.stem
        gt_path = _gt(image_path)

        base_ap50 = None
        if run_baseline:
            b = run_cellpose_baseline(segmenter, image_path, gt_path,
                                      output_root / "cellpose" / image_id)
            baseline_times.append(b["runtime_seconds"])
            base_ap50 = b["benchmark"].get("ap50")
            if base_ap50 is not None:
                baseline_results.append(b["benchmark"])
            print(f"  [{index}/{len(images)}] cellpose {image_id}: "
                  f"AP50={b['benchmark'].get('ap50', float('nan')):.3f} "
                  f"cells={b['cell_count']} t={b['runtime_seconds']:.1f}s")

        t0 = time.perf_counter()
        tok0 = token_usage.total_tokens()
        solve_kwargs = dict(
            question=effective_question,
            images=[{"path": str(image_path), "group": "default"}],
            llm_engine_name=model, api_key=api_key, max_steps=max_steps,
            query_cache_dir=str(output_root / "shape" / image_id),
            verify_refine=verify_refine,
            enabled_tools=_SHAPE_MAIN_TOOLS,  # task 083: segmentor only, see comment above
        )
        result = solve(**solve_kwargs)
        shape_times.append(time.perf_counter() - t0)
        ver = result.get("verification_summary") or []
        all_verification.extend(ver)
        try:     # the full per-image record (agentic rounds, plans, assessments, final choice)
            (output_root / "shape" / image_id).mkdir(parents=True, exist_ok=True)
            (output_root / "shape" / image_id / "verification_summary.json").write_text(
                json.dumps(ver, indent=1, default=str), encoding="utf-8")
        except OSError as exc:
            print(f"  [trace] {image_id}: verification summary not written ({exc})")
        trace = _image_trace(ver)                   # did THIS image escalate?
        agent_trace[image_id] = trace["stages"]
        prediction = _collect_masks_by_image(result["memory"]).get(image_id)
        if prediction and gt_path.is_file():
            try:
                m = evaluate_instance_segmentation(prediction, gt_path)
                m["image_name"] = image_id
                m["seconds"] = round(shape_times[-1], 2)   # per-image wall-clock (efficiency metrics)
                m["escalated"] = trace["escalated"]
                m["judge_profile"] = trace["judge_profile"]
                shape_results.append(m)
                attempts = sm.attempt_masks_for(prediction)
                efficiency_records.append({
                    "image_id": image_id, "baseline_ap50": base_ap50,
                    "call_ap50": [a for a in (_safe_ap50(p, gt_path) for p in attempts)
                                  if a is not None],
                    "tokens": token_usage.total_tokens() - tok0, "seconds": m["seconds"]})
                print(f"  [{index}/{len(images)}] shape    {image_id}: AP50={m['ap50']:.3f} "
                      f"recall={m['recall_at_50']:.3f} pred={m['predicted_cell_count']} "
                      f"gt={m['ground_truth_cell_count']} t={shape_times[-1]:.1f}s"
                      + (" ESCALATED" if trace["escalated"] else ""))
            except Exception as exc:
                print(f"  [seg-eval] {image_id}: {exc}")
        elif not prediction:
            print(f"  [{index}/{len(images)}] shape    {image_id}: no mask produced")

    shape_total_runtime = float(sum(shape_times))
    n_images = max(1, len(images))
    shape_macro = macro_metrics(shape_results)

    summary = {
        "dataset": "NeurIPS22-CellSeg",
        "task": "agentic_segmentation",
        "image_count": len(images),
        "total_available": total_available,
        "sampling": {"sample_frac": sample_frac, "limit": limit, "seed": seed},
        "verify_refine": verify_refine,  # task 044: which arm of the same-job A/B this is
        "metric_type": "macro (per-image mean)",
        "predicted_mask_images": len(shape_results),
        "shape_runtime_seconds_total": shape_total_runtime,
        "segmentation_evaluation": {"per_image": shape_results, "macro": shape_macro},
        "agent_activity": _summarize_agent_activity(all_verification),
        "agent_trace": agent_trace,
        "efficiency_metrics": sm.compute_metrics(efficiency_records),
        # 083 fix round 1: persist the per-image records `efficiency_metrics` was
        # computed FROM (not just that band-level aggregate), so a paired per-image
        # comparison across two runs (main vs v8d) can decompose all 6 efficiency
        # metrics per image instead of only at the band aggregate.
        "efficiency_records": efficiency_records,
    }

    # ---- Cellpose-SAM (tool alone) vs SHAPE (agentic) comparison + figure (macro) ----
    if run_baseline:
        base_macro = macro_metrics(baseline_results)

        def _mean(xs):
            return float(np.mean(xs)) if xs else 0.0

        comparison = {
            "metric_type": "macro (per-image mean)",
            "cellpose_baseline": {
                **{k: base_macro.get(k) for k in _MACRO_KEYS},
                "n_evaluable": base_macro.get("n_images", 0),
                "mean_runtime_seconds": _mean(baseline_times),
                "total_runtime_seconds": float(sum(baseline_times)),
            },
            "shape_agentic": {
                **{k: shape_macro.get(k) for k in _MACRO_KEYS},
                "n_evaluable": shape_macro.get("n_images", 0),
                "mean_runtime_seconds": shape_total_runtime / n_images,
                "total_runtime_seconds": shape_total_runtime,
            },
        }
        summary["comparison"] = comparison
        summary["baseline_evaluation"] = {"per_image": baseline_results, "macro": base_macro}
        if base_macro.get("evaluable") and shape_macro.get("evaluable"):
            fig_path = output_root / (
                time.strftime("%Y-%m-%d_%H%M") + "_comparison_shape_vs_cellpose.png")
            try:
                plot_comparison(
                    base_macro, shape_macro,
                    comparison["cellpose_baseline"]["mean_runtime_seconds"],
                    comparison["shape_agentic"]["mean_runtime_seconds"], fig_path,
                    efficiency=summary.get("efficiency_metrics"))
                summary["comparison"]["figure_path"] = str(fig_path)
                print(f"Comparison figure saved to {fig_path}")
            except Exception as exc:
                print(f"Comparison figure failed: {exc}")

    summary_path = output_root / "run_summary.json"
    with open(summary_path, "w", encoding="utf-8") as handle:
        json.dump(summary, handle, indent=2, default=str)
    if shape_macro.get("evaluable"):
        print(f"Segmentation (macro): AP50={shape_macro['ap50']:.4f}, "
              f"AP75={shape_macro['ap75']:.4f}, mAP={shape_macro['map_50_95']:.4f}, "
              f"F1@0.50={shape_macro['f1_at_50']:.4f} (n={shape_macro['n_images']})")
    else:
        print("Segmentation: not evaluable (no images scored)")
    if run_baseline and "comparison" in summary:
        b, s = summary["comparison"]["cellpose_baseline"], summary["comparison"]["shape_agentic"]
        print("Comparison macro (Cellpose-SAM alone -> SHAPE agentic):")
        print(f"  AP50   {b['ap50']:.4f} -> {s['ap50']:.4f}")
        print(f"  F1@50  {b['f1_at_50']:.4f} -> {s['f1_at_50']:.4f}")
        print(f"  Recall {b['recall_at_50']:.4f} -> {s['recall_at_50']:.4f}")
        print(f"  Runtime {b['mean_runtime_seconds']:.1f}s -> "
              f"{s['mean_runtime_seconds']:.1f}s per image")
    eff = summary["efficiency_metrics"]

    def _f(v):
        return "n/a" if v is None else f"{v:.3f}"

    print(f"Efficiency: tool_calls/img={_f(eff['tool_calls_per_image'])} "
          f"positive_rate={_f(eff['positive_tool_call_rate'])} "
          f"wasted_rate={_f(eff['wasted_tool_call_rate'])} "
          f"AP50_gain/call={_f(eff['ap50_gain_per_tool_call'])} "
          f"time/img={_f(eff['time_per_image_s'])}s "
          f"tokens/img={_f(eff['token_cost_per_image'])}")
    print(f"Summary saved to {summary_path}")
    return summary


def main() -> None:
    # Local per-benchmark hyperparameter config is the single source of truth for the
    # default model (named cellseg_config so a sibling benchmark's config.py can't shadow it).
    sys.path.insert(0, str(Path(__file__).resolve().parent))
    from cellseg_config import DEFAULT as _cfg  # noqa: PLC0415
    parser = argparse.ArgumentParser(
        description="Agentic segmentation benchmark on NeurIPS22-CellSeg: the FULL SHAPE "
                    "agent vs Cellpose-SAM alone.")
    parser.add_argument(
        "--data-root", type=Path,
        default=(Path(os.environ["SHAPE_BENCH_DATA_ROOT"]) / "NeurIPS22-CellSeg"
                 if os.environ.get("SHAPE_BENCH_DATA_ROOT") else None),
        required=not os.environ.get("SHAPE_BENCH_DATA_ROOT"),
        help="Dataset root containing images/ and labels/ subfolders."
             "Defaults to $SHAPE_BENCH_DATA_ROOT/NeurIPS22-CellSeg.")
    parser.add_argument("--image-dir", type=Path,
                        help="Override images dir (default <data-root>/images).")
    parser.add_argument("--label-dir", type=Path,
                        help="Override labels dir (default <data-root>/labels).")
    parser.add_argument("--label-suffix", default="_label",
                        help="Label filename suffix: <stem><suffix><ext> (default _label).")
    parser.add_argument("--label-ext", default=".tiff",
                        help="Label file extension (default .tiff).")
    parser.add_argument("--output-root", "--out-root", dest="output_root", type=Path,
                        default=Path("solver_cache/NeurIPS22-CellSeg"))
    parser.add_argument("--sample-frac", type=float,
                        help="Seeded random fraction of images to run, e.g. 0.2 for 20%%.")
    parser.add_argument("--limit", type=int,
                        help="Hard cap on #images (applied after sampling).")
    parser.add_argument("--seed", type=int, default=0,
                        help="Seed for reproducible random sampling.")
    parser.add_argument("--image-list",
                        help="Run only these images. Either a path to a file with one "
                             "image name (stem) per line (# comments ok), or a comma-"
                             "separated list of stems. Overrides random sampling.")
    parser.add_argument("--prompt-file", type=Path,
                        help="Path to a text file with the agent's query prompt. Default: "
                             "read benchmarks/segmentation_NeurIPS22-CellSeg/Prompt.txt at "
                             "runtime (the professional, segmentation-only-scoped prompt).")
    parser.add_argument("--model", default=_cfg.llm_model,
                        help="LLM the agent uses (planner + vision verifier). "
                             "Default from cellseg_config.py (llm_model).")
    parser.add_argument("--baseline", action=argparse.BooleanOptionalAction, default=True,
                        help="Also run Cellpose-SAM alone (default params, no agent) and "
                             "compare accuracy + runtime against SHAPE. Use --no-baseline "
                             "to skip (halves segmentation time).")
    parser.add_argument("--max-steps", type=int, default=8,
                        help="Max agent steps.")
    parser.add_argument("--verify-refine", action=argparse.BooleanOptionalAction,
                        default=False,
                        help="Task 044: segmenter Verify refinement (adds scale_mismatch "
                             "evidence; drops flow_error from the vision prompt only). "
                             "Default off (base behavior); pass --verify-refine for the "
                             "refined arm of the same-job A/B.")
    args = parser.parse_args()
    image_dir = args.image_dir or (args.data_root / "images")
    label_dir = args.label_dir or (args.data_root / "labels")
    image_names = None
    if args.image_list:
        p = Path(args.image_list)
        if p.is_file():
            names = p.read_text(encoding="utf-8").splitlines()
        else:
            names = args.image_list.split(",")          # comma-separated stems
        # strip inline '# ...' comments (e.g. "cell_00151  # ap50=0") and whitespace; drop blanks
        names = [ln.split("#", 1)[0].strip() for ln in names]
        image_names = {Path(n).stem for n in names if n}
        print(f"[batch] image-list: running {len(image_names)} named images")
    question = args.prompt_file.read_text(encoding="utf-8") if args.prompt_file else None
    run_batch(
        image_dir, label_dir, args.output_root,
        label_suffix=args.label_suffix, label_ext=args.label_ext,
        sample_frac=args.sample_frac, limit=args.limit, seed=args.seed,
        model=args.model or _cfg.llm_model, run_baseline=args.baseline,
        image_names=image_names, max_steps=args.max_steps or 8,
        verify_refine=args.verify_refine,
        question=question)


if __name__ == "__main__":
    main()

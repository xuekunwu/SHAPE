"""Verifier-invoked color/class filter for cell segmentation.

Some specimens (classically STAINED BLOOD SMEARS / platelet fields) contain more than one
object CLASS distinguished only by COLOR: a sea of pink/red red-blood-cells that are NOT the
counting target, plus a sparse population of deep-purple white-blood-cells / platelets that
IS. Cellpose-SAM segments every round object, so it grossly over-counts on these images.

This module lets a VLM DECIDE (never a hardcoded threshold) whether one whole color population
is a non-target class and should be dropped:

  1. Candidacy (deterministic, data-driven): split the per-mask mean (B-R) by Otsu on THIS
     image's own distribution. A real 2-cluster split with a small minority => candidate.
  2. Evidence: render a COLOR-PRESERVING image (raw + thin outlines). The saved cell_overlay
     recolors every mask a random color, destroying the pink/purple signal, so we render our
     own -- see results/color_filter_experiment.md.
  3. Decision (VLM): the model sees the evidence image + the cluster percentages and returns
     {none | keep_high_br | keep_low_br}. Keep-biased; balance-gated (filter only when the
     KEPT population is a small minority -- else the two colors are the same class counted
     together, e.g. a smear whose GT counts RBCs too).
  4. Apply: keep the chosen cluster; the cut is Otsu on this image's B-R (never a fixed number).

Validated post-hoc on NeurIPS22-CellSeg (gpt-5.6-luna): full-1000 macro AP50 0.7645 -> 0.8301
(+0.066), 82/1000 filtered, 0 collateral damage. See results/color_filter_experiment.md.
"""
from __future__ import annotations

import io
import logging
from typing import Literal, Optional

import numpy as np
from pydantic import BaseModel

from shape.engine.errors import NoEngineError
from shape.utils.schemas import ExecutionError

logger = logging.getLogger("shape.tools.cell_segmenter.color_class_filter")

# Candidacy: at least this many masks, and the KEPT minority must be at most this fraction of
# objects for a keep_* action to be honored (population-balance rule -- the separator between
# "few purple targets in a pink RBC sea" and "two comparably-abundant stains of one class").
_MIN_MASKS = 4
_MINORITY_MAX_FRAC = 0.10
# The KEPT (target) population must have at least this many masks. A kept cluster of 1-2 is
# almost never a real sparse target population -- it is a mis-split (e.g. the VLM firing on a
# fluorescence image whose B-R happens to be bimodal, collapsing 110 cells to 1). Blood-smear
# platelet/WBC minorities are typically 5-15, so this vetoes the false positives without
# touching the real cases.
_MINORITY_MIN_COUNT = 3


def _to_rgb(img) -> np.ndarray:
    a = np.asarray(img)
    if a.ndim == 2:
        return np.stack([a, a, a], -1).astype(np.float32)
    if a.shape[2] >= 3:
        return a[..., :3].astype(np.float32)
    return np.repeat(a[..., :1], 3, -1).astype(np.float32)


def _per_mask_br(img_rgb: np.ndarray, mask: np.ndarray) -> dict[int, float]:
    """Each mask's mean (B - R), by mask id in increasing order. One pass finds every mask's bounding box
    (`ndimage.find_objects`); a mask's pixels are gathered inside its box, in the image's row order -- the
    same values in the same order as a full-image `B[mask == i]`, so the same means to the last bit, without
    scanning the whole image once per mask."""
    from scipy import ndimage

    R, B = img_rgb[..., 0], img_rgb[..., 2]
    out = {}
    for i, box in enumerate(ndimage.find_objects(mask), start=1):
        if box is None:
            continue
        inside = mask[box] == i
        out[i] = float(B[box][inside].mean() - R[box][inside].mean())
    return out


def color_cluster_stats(img, mask) -> Optional[dict]:
    """Otsu-split the per-mask (B-R) distribution. Returns None when the image is NOT a
    color-class candidate (too few masks, or no separable 2-cluster split)."""
    from skimage.filters import threshold_otsu
    img_rgb = _to_rgb(img)
    br = _per_mask_br(img_rgb, mask)
    if len(br) < _MIN_MASKS:
        return None
    vals = np.array(list(br.values()), dtype=np.float64)
    if float(vals.ptp()) < 1e-3:
        return None
    try:
        cut = float(threshold_otsu(vals))
    except Exception:
        return None
    hi = vals[vals > cut]
    lo = vals[vals <= cut]
    if len(hi) == 0 or len(lo) == 0:
        return None
    n = len(vals)
    return {
        "br": br, "cut": cut, "n": n,
        "hi_n": int(len(hi)), "lo_n": int(len(lo)),
        "hi_frac": float(len(hi) / n), "lo_frac": float(len(lo) / n),
        "hi_mean": float(hi.mean()), "lo_mean": float(lo.mean()),
    }


def render_evidence_png(img, mask) -> bytes:
    """Raw image (percentile-normalized) + thin dark mask outlines -- preserves the ORIGINAL
    cell colors so the VLM can make a color-class call (the random-color overlay cannot)."""
    from skimage.segmentation import find_boundaries
    from PIL import Image
    x = _to_rgb(img)
    lo, hi = np.percentile(x, 1), np.percentile(x, 99)
    x = np.clip((x - lo) / (hi - lo + 1e-8), 0, 1)
    u = (x * 255).astype(np.uint8)
    u[find_boundaries(mask, mode="outer")] = [30, 30, 30]
    buf = io.BytesIO()
    Image.fromarray(u).save(buf, format="PNG")
    return buf.getvalue()


class ColorClassVerdict(BaseModel):
    """Does ONE color population correspond to a NON-TARGET object type to remove? KEEP-
    everything (action='none') is the default; only pick keep_* on clear mixed-class evidence
    AND when the KEPT population is a small minority (see the balance rule in the prompt)."""
    scene: str                                    # what the image shows + which objects are the target cells
    is_mixed_class: bool                          # True only if a whole color population is a non-target object type
    action: Literal["none", "keep_high_br", "keep_low_br"]
    confidence: Literal["high", "low"]
    reason: str


_PROMPT = (
    "You are triaging a CELL SEGMENTATION result for a counting task. The image shows every "
    "segmented object outlined on the ORIGINAL image (colors are real). The tool tends to "
    "OVER-SEGMENT: in some specimens (classically STAINED BLOOD SMEARS) it outlines every round "
    "object, but only ONE color class is the actual target cell (e.g. the deep-purple white "
    "blood cells / platelets), while the many pink/red objects are RED BLOOD CELLS that must NOT "
    "be counted.\n\n"
    "The segmented objects split into TWO color populations by mean (Blue - Red) intensity "
    "(data-driven, this image only):\n{clusters}\n\n"
    "Decide whether ONE whole color population is a NON-TARGET object type that should be removed:\n"
    "  - action='none'  -> KEEP everything (the DEFAULT).\n"
    "  - action='keep_high_br' -> keep only the HIGHER (B-R) population (purple/blue), remove the "
    "lower (pink/red) one.\n"
    "  - action='keep_low_br'  -> keep only the LOWER (B-R) population, remove the higher one.\n\n"
    "KEEP-EVERYTHING is the overwhelming default. Only filter on clear, un-second-guessable "
    "evidence that a whole color population is a non-target object class.\n\n"
    "DECISIVE RULE ON POPULATION BALANCE: filtering is appropriate ONLY when the population you "
    "would KEEP is a SMALL MINORITY (roughly <10% of all objects) -- a few sparse target cells in "
    "a DOMINANT background of the other color (the classic smear: a handful of purple WBC/"
    "platelets among a sea of pink RBC). If the two color populations are COMPARABLY ABUNDANT "
    "(neither is a small minority -- e.g. 20/80 or 40/60), they are almost certainly the SAME "
    "target cell type stained at different intensities and are counted TOGETHER -- choose 'none' "
    "EVEN IF the field looks like a smear. Removing a large, comparably-abundant population "
    "deletes true target cells. Check the percentages above before deciding.\n\n"
    "Report scene, is_mixed_class, action, confidence (high/low), and one sentence of reason that "
    "STATES the kept-population percentage and whether it is a small minority."
)


def _clusters_text(stats: dict) -> str:
    return (f"  total segmented objects: {stats['n']}\n"
            f"  HIGH (B-R): n={stats['hi_n']} ({stats['hi_frac']:.1%} of objects), "
            f"mean B-R={stats['hi_mean']:+.1f} (bluer/purpler)\n"
            f"  LOW  (B-R): n={stats['lo_n']} ({stats['lo_frac']:.1%} of objects), "
            f"mean B-R={stats['lo_mean']:+.1f} (redder/pinker)\n"
            f"  Otsu cut at B-R={stats['cut']:+.1f}")


def decide(engine, img, mask, stats: Optional[dict] = None) -> dict:
    """VLM color-class decision. Returns a dict trace: {candidate, action, applied, confidence,
    scene, reason, ...}. `applied` is the action actually honored (balance-gated + confidence-
    gated); 'none' when not a candidate / no engine / low confidence / balance rule vetoes."""
    info = {"candidate": False, "action": "none", "applied": "none",
            "confidence": None, "scene": "", "reason": ""}
    if engine is None:                    # a system error, never "keep all masks"
        raise NoEngineError("color-class decide: no vision engine is configured")
    if stats is None:
        stats = color_cluster_stats(img, mask)
    if stats is None:
        return info
    info["candidate"] = True
    info.update({"hi_frac": stats["hi_frac"], "lo_frac": stats["lo_frac"],
                 "n_masks": stats["n"], "cut": stats["cut"]})
    try:
        prompt = _PROMPT.format(clusters=_clusters_text(stats))
        ev = render_evidence_png(img, mask)
        resp = engine.generate([prompt, ev], response_format=ColorClassVerdict)
        rec = resp.get("content") if isinstance(resp, dict) else resp
        if rec is None or not hasattr(rec, "action"):
            raise ExecutionError(f"color-class decide: malformed answer ({type(rec).__name__})")
        info.update({"action": rec.action, "confidence": rec.confidence,
                     "scene": getattr(rec, "scene", ""), "reason": getattr(rec, "reason", "")})
        # confidence gate + balance gate: honor keep_* only when high-confidence AND the KEPT
        # population is a small minority (the data-driven separator, not a color threshold) AND
        # that minority has enough masks to be a real population (not a 1-2 mask mis-split).
        kept_frac = stats["hi_frac"] if rec.action == "keep_high_br" else (
            stats["lo_frac"] if rec.action == "keep_low_br" else 1.0)
        kept_n = stats["hi_n"] if rec.action == "keep_high_br" else (
            stats["lo_n"] if rec.action == "keep_low_br" else stats["n"])
        if (rec.action != "none" and rec.confidence == "high"
                and kept_frac <= _MINORITY_MAX_FRAC and kept_n >= _MINORITY_MIN_COUNT):
            info["applied"] = rec.action
        elif rec.action != "none" and kept_frac > _MINORITY_MAX_FRAC:
            info["balance_veto"] = True
        elif rec.action != "none" and kept_n < _MINORITY_MIN_COUNT:
            info["min_count_veto"] = True
    except ExecutionError:
        raise
    except Exception as exc:  # an execution error, never a silent "keep all masks"
        raise ExecutionError(f"color-class decide failed: {type(exc).__name__}: {exc}") from exc
    return info


def apply(img, mask, action: str, stats: Optional[dict] = None) -> np.ndarray:
    """Keep only the chosen color cluster; cut = Otsu on this image's per-mask B-R."""
    if action == "none":
        return mask
    if stats is None:
        stats = color_cluster_stats(img, mask)
    if stats is None:
        return mask
    br, cut = stats["br"], stats["cut"]
    if action == "keep_high_br":
        keep = [i for i, v in br.items() if v > cut]
    elif action == "keep_low_br":
        keep = [i for i, v in br.items() if v <= cut]
    else:
        return mask
    return np.where(np.isin(mask, keep), mask, 0).astype(mask.dtype)


def filter_mask(engine, img, mask) -> tuple[np.ndarray, dict]:
    """Orchestrate: candidacy -> VLM decision -> apply. Returns (possibly-filtered mask, trace)."""
    stats = color_cluster_stats(img, mask)
    info = decide(engine, img, mask, stats)
    new_mask = apply(img, mask, info["applied"], stats) if info["applied"] != "none" else mask
    return new_mask, info

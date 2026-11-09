"""Verifier 3: Segmentation (Cellpose-SAM, tunable).

Key upgrade over a flat checklist: `flow_error` is a GATE, not just one more
metric. High flow error means Cellpose-SAM does not fit this image at all, so
tuning diameter/threshold is futile. The gate decides whether we are even
allowed to retune here, or must escalate upstream / declare tool mismatch.

`SegmentationAgenticVerifier` (the end of this module) is Cell_Segmenter_Tool's agentic VRR
(`base.AgenticVerifier`; CARD.md "Agentic VRR"). `SegmentationVerifier` stays the linear loop (the
organoid segmenter) and lends the agentic loop its
readouts, QC score and diagnosis.
"""

from __future__ import annotations

from collections.abc import Mapping
import inspect
import io
import json
import math
import os
import re
import time
from typing import Any, Optional, Literal
from urllib.request import urlopen

import numpy as np
from pydantic import BaseModel, Field

from shape.engine.errors import NoEngineError, is_fatal_llm_error
from shape.tools.cell_segmenter import color_class_filter as ccf
from shape.utils.logger import logger
from ..initial_params import diameter_correction_needed, diameter_from_mask_area
from ..loaders import _load_gray, _uncovered_foreground
from shape.utils.schemas import ExecutionError, Hypothesis, VerifyReport, VLRRResult, WaveMove, WaveSelection

from .. import base as vrr_base
from ..base import (EFFECTIVE_DEFAULTS, INPUT, PARAM_BOUNDS, _SEVERITY_RANK, AgenticVerifier, BaseVerifier,
                    apply_bounded, normalize, numeric, weighted_score)


def adapt_segmentation(result: dict, upstream=None, image_path=None):
    """Segmenter result dict -> ((masks, image, flow_error), context).

    Folded out of the shared adapters bridge (Phase 2b): the segmentation stage's
    native-result adaptation lives with its verifier now. Loads the saved label mask
    (.npy) and reads the tool-computed flow_error. `upstream` may carry image-loading
    quality flags (sharpness_low/background_high) used to route a high flow_error
    upstream. `image_path` (the segmentation input) lets us estimate a deterministic
    under-detection (missed-cell) cue from uncovered foreground texture.
    """
    mask_path = result.get("mask_path") or result.get("organoid_mask_path")
    masks = None
    if mask_path and os.path.exists(mask_path):
        try:
            masks = np.load(mask_path)
        except Exception as e:
            logger.warning(f"adapt_segmentation: failed to load mask {mask_path}: {e}")
    if masks is None:
        masks = np.zeros((1, 1), dtype=np.int32)
    # flow_error is the VLRR segmentation gate. The segmenter tools always emit it,
    # so a MISSING value on a real (non-empty) mask means the gate silently defaults
    # to 0.0 and is inactive for this pass -- make that visible instead of hiding it.
    if result.get("flow_error") is None and masks.any():
        logger.warning("adapt_segmentation: result has no flow_error; the VLRR flow "
                       "gate is inactive for this pass (mask=%s)", mask_path)
    flow_error = float(result.get("flow_error", 0.0) or 0.0)
    coverage_qc: dict = {}
    if image_path:
        img = _load_gray(image_path)
        if img is not None:
            stats = _uncovered_foreground(img, masks)
            if stats is not None:
                coverage_qc.update(stats)
    context = {
        "upstream": dict(upstream or {}),
        "overlay_qc": dict(result.get("overlay_qc") or {}),
        "coverage_qc": coverage_qc,
        # overlay PNG (colored masks on the image) for the vision-LLM judge
        "overlay_path": result.get("overlay_path") or result.get("output_path"),
        # raw image + mask paths for the color/class-filter retune proposer (it needs the
        # ORIGINAL RGB for the per-mask B-R, which the random-color overlay destroys).
        "image_path": image_path,
        "mask_path": result.get("mask_path") or result.get("organoid_mask_path"),
        # 044 scope A: the diameter this pass ACTUALLY used (tool.py:537
        # result["parameters"]["diameter"]; None when the tool auto-detected -- see
        # tool.py:369), threaded through so `quantify` can compute scale_mismatch.
        "diameter_used": (result.get("parameters") or {}).get("diameter"),
        "parameters": dict(result.get("parameters") or {}),
    }
    # mask_qc ignores the image arg; pass masks as a harmless placeholder
    return (masks, masks, flow_error), context


def _scale_mismatch(mask_area_median: float, diameter_used) -> float | None:
    """Threshold-free scale-mismatch EVIDENCE for the vision judge (044 BRIEF sec A):
    measured typical cell size -- REUSING the existing, tested
    `initial_params.diameter_from_mask_area` on the median mask area, do NOT re-derive --
    divided by the `diameter` actually used for this pass. ~1.0 = matches; >>1 = the used
    diameter under-estimated the true cell size (masks too small for the cells); <<1 =
    over-estimated. No hard threshold (sec 3): the ratio itself is the evidence, the LLM
    weighs it.

    Returns None (the metric is OMITTED, never a misleading ~1.0) when `diameter_used`
    is not a captured numeric value -- None or "auto" (the cellpose auto-detect path,
    tool.py:369, where no diameter was actually applied to compare against). The
    cell_count==0 case is handled by the caller never reaching here (mask_qc's
    zero-mask branch has no `mask_area_median` to pass).
    """
    if diameter_used is None or (isinstance(diameter_used, str)
                                  and diameter_used.strip().lower() == "auto"):
        return None
    try:
        used = float(diameter_used)
    except (TypeError, ValueError):
        return None
    if used <= 0:
        return None
    return diameter_from_mask_area(mask_area_median) / used


def mask_qc(masks: np.ndarray, image: np.ndarray, flow_error: float,
            diameter_used=None) -> dict[str, float]:
    # Vectorized: one np.unique pass for per-label areas instead of a full-image scan
    # per mask -- the per-mask version is O(n_masks * pixels) and was the dominant cost
    # on dense fields (thousands of masks on multi-megapixel images).
    labels, counts = np.unique(masks, return_counts=True)
    keep = labels != 0
    ids = labels[keep]
    if ids.size == 0:
        return {"cell_count": 0.0, "flow_error": flow_error,
                "foreground_coverage": 0.0, "tiny_mask_fraction": 0.0,
                "large_mask_fraction": 0.0, "mask_area_CV": 0.0,
                "border_mask_fraction": 0.0, "mean_solidity": 0.0}
    areas = counts[keep].astype(float)
    med = np.median(areas)
    border_ids = set(np.unique(np.concatenate(
        [masks[0], masks[-1], masks[:, 0], masks[:, -1]]))) - {0}
    solidity = _mean_solidity(masks, ids, areas)
    out = {
        "cell_count": float(ids.size),
        "flow_error": float(flow_error),
        "mask_area_median": float(med),
        "mask_area_CV": float(areas.std() / (areas.mean() + 1e-9)),
        "tiny_mask_fraction": float((areas < 0.3 * med).mean()),
        "large_mask_fraction": float((areas > 3.0 * med).mean()),
        "foreground_coverage": float((masks > 0).mean()),
        "border_mask_fraction": float(len(border_ids) / ids.size),
        "mean_solidity": solidity,
    }
    sm = _scale_mismatch(med, diameter_used)
    if sm is not None:
        out["scale_mismatch"] = sm
    return out


def _mean_solidity(masks, ids, areas) -> float:
    # Vectorized: ndimage.find_objects returns every label's bounding box in ONE pass,
    # so solidity = area / bbox_area without an O(pixels) np.where per mask.
    from scipy import ndimage

    objs = ndimage.find_objects(masks)
    vals = []
    for area, i in zip(areas, ids):
        idx = int(i) - 1
        sl = objs[idx] if 0 <= idx < len(objs) else None
        if sl is None:
            continue
        bb = (sl[0].stop - sl[0].start) * (sl[1].stop - sl[1].start)
        vals.append(float(area) / (bb + 1e-9))
    return float(np.mean(vals)) if vals else 0.0


class SegmentationVerifier(BaseVerifier):
    name = "segmentation"
    # Opt-in to the vision-LLM tie-breaker. `llm_engine` (a multimodal ChatOpenAI)
    # is injected by the integration layer when enable_llm_verifier is on; when it
    # is None the verifier stays purely deterministic.
    uses_llm = True
    llm_engine = None
    # The tool this verifier governs; its CARD.md (loaded via the base `tuning_card()`) is
    # injected into the vision prompt so the LLM reasons problem -> parameter from
    # documented semantics rather than a hard-coded knob table.
    tool_dir = "cell_segmenter"
    # 044 (VRR refinement of the working 040 loop): OFF by default so an unset flag is
    # code-path-identical to base `origin/main @ 6087637` -- set to True (per-instance,
    # mirroring `llm_engine`/`tool_name` above) by `Verifier.resolve_managed_stage` when
    # the run opts in. Gates BOTH: (A) scale_mismatch computed in `quantify`, and (B)
    # flow_error filtered from the vision prompt only (`adjudicate`'s vision call below).
    verify_refine = False

    # Shape-quality gates, config-declared (metric, threshold, op) instead of buried
    # as inline literals in localize(). These are soft PRIORS: they generate the
    # candidate "triggered" complaints, but the vision judge may tighten the verdict
    # (never loosen, per _clamp_action) -- reasoning over evidence, not a fixed rule.
    # ONLY gates with a matching lever in _diagnose belong here (observable fault ->
    # identifiable lever -> matching expected_effect). border_mask_fraction and
    # mask_area_CV were removed: neither maps to a lever, so a standalone trigger fell
    # into the noisy-FP fallback (cellprob/min_size up, expected tiny_mask_fraction down)
    # -- an unrelated move whose unmet expectation immediately conceded the stage. They
    # are still COMPUTED in mask_qc and still passed to the vision judge as EVIDENCE
    # (see the `border`/CV normalisation in the evidence block below); they just no
    # longer auto-trigger a retune.
    _SHAPE_GATES = (
        ("tiny_mask_fraction", 0.15, ">"),
        ("large_mask_fraction", 0.10, ">"),
        ("mean_solidity", 0.35, "<"),
    )

    def __init__(self, flow_gate: float = 0.4, miss_gate: float = 0.10,
                 count_gate: float = 0.15, accept_score: float = 0.80):
        self.flow_gate = flow_gate
        # accept a first pass without retuning when its mask-quality score is at least
        # this high AND the only complaint is the (noisy) under-detection cue -- which
        # over-fires on diffuse/fluorescent texture (e.g. 0.9 "missed" at recall 0.94),
        # wasting a retune the reconcile step then invalidates. The vision judge can
        # still tighten back to a retune if it actually sees missed cells.
        self.accept_score = accept_score
        # deterministic under-detection backstop: fraction of uncovered foreground
        # above which we retune for missed cells even if mask shapes look fine.
        # 0.10 separates real under-detections (~0.12-0.14) from clean runs (<0.07).
        self.miss_gate = miss_gate
        # COUNT-based companion gate for CROWDED fields: the area gate misses cases
        # where many small cells are dropped (low area, high count). Trigger when the
        # estimated missed-cell count exceeds this fraction of the total.
        self.count_gate = count_gate

    def quantify(self, tool_output, context) -> dict[str, float]:
        masks, image, flow_error = tool_output
        # verify_refine gates threading the used diameter at all -- OFF, `mask_qc` gets
        # diameter_used=None and never adds scale_mismatch (base-identical).
        diameter_used = context.get("diameter_used") if self.verify_refine else None
        metrics = mask_qc(masks, image, flow_error, diameter_used)
        overlay_qc = context.get("overlay_qc", {})
        metrics["overlay_is_black"] = float(bool(overlay_qc.get("overlay_is_black")))
        metrics["coverage_implausible"] = float(bool(overlay_qc.get("coverage_implausible")))
        coverage_qc = context.get("coverage_qc", {})
        metrics["missed_foreground_fraction"] = float(
            coverage_qc.get("missed_foreground_fraction", 0.0) or 0.0)
        metrics["missed_blob_size_ratio"] = float(
            coverage_qc.get("missed_blob_size_ratio", 0.0) or 0.0)
        metrics["count_miss_fraction"] = float(
            coverage_qc.get("count_miss_fraction", 0.0) or 0.0)
        return metrics

    def _under_detected(self, m) -> bool:
        """Under-detection by EITHER cue: a large uncovered foreground AREA, or many
        cell-sized uncovered regions by COUNT (crowded fields where the area stays low)."""
        return (m.get("missed_foreground_fraction", 0.0) > self.miss_gate
                or m.get("count_miss_fraction", 0.0) > self.count_gate)

    def localize(self, metrics, context):
        triggered = []
        if metrics.get("overlay_is_black") or metrics.get("coverage_implausible"):
            triggered.append("visual_output_invalid")
            return "tool_mismatch", triggered, []
        if metrics["flow_error"] > self.flow_gate:
            triggered.append("flow_gate")
            up = context.get("upstream", {})
            if (up.get("sharpness_low") or up.get("background_high")):
                return "upstream", triggered, []        # bad input drives bad flow
            return "tool_mismatch", triggered, []        # clean image, model unfit
        if metrics["cell_count"] == 0:
            # nothing detected at all: maximal under-detection, not a low-solidity
            # quality issue (all shape metrics are 0). Route to lowering the
            # threshold; the loop escalates if that fails to recover cells.
            return "this_step", ["no_detection"], []
        for k, thr, op in self._SHAPE_GATES:
            v = metrics[k]
            if (op == ">" and v > thr) or (op == "<" and v < thr):
                triggered.append(k)
        # missed cells: uncovered foreground by area, or many missed by count (crowded)
        if self._under_detected(metrics):
            triggered.append("under_detection")
        return "this_step", triggered, []

    def _color_filter_proposal(self, metrics, context, score):
        """Propose a color/class-filter retune when the mask is a color-separable candidate AND
        the vision judge confirms one whole color population is a non-target object class.
        Returns a VLRRResult(param_delta={color_class_filter: keep_high_br|keep_low_br}) or None
        (stay on the normal retune/accept path). Cheap deterministic candidacy (bimodal per-mask
        B-R) gates the VLM so it runs only on candidates; the judge's own balance rule then
        vetoes comparably-abundant populations (same-class stains counted together)."""
        engine = getattr(self, "llm_engine", None)
        mask_path = (context or {}).get("mask_path")
        image_path = (context or {}).get("image_path")
        if engine is None or not mask_path or not image_path:
            return None
        try:
            import os as _os
            import numpy as _np
            from skimage.io import imread as _imread
            from shape.tools.cell_segmenter import color_class_filter as _ccf
            if not (_os.path.exists(mask_path) and _os.path.exists(image_path)):
                return None
            mask = _np.load(mask_path).astype(_np.int32)
            rgb = _imread(image_path)
            stats = _ccf.color_cluster_stats(rgb, mask)
            if stats is None:
                return None  # not color-separable -> normal path (no VLM call)
            info = _ccf.decide(engine, rgb, mask, stats)
            action = info.get("applied", "none")
            if action == "none":
                return None
            notes = (f"[color-filter] keep={action} "
                     f"(hi {stats['hi_frac']:.1%}/lo {stats['lo_frac']:.1%}): a color "
                     f"population is non-target -> drop it (param retune cannot). "
                     f"{str(info.get('reason', ''))[:120]}")
            # judge_severity="major": the vision judge (ccf.decide, which sees the ORIGINAL
            # colors) found a whole non-target color population -> this UNFILTERED attempt is
            # bad REGARDLESS of its (deceptively high) mask-shape score, so keep-best ranks it
            # below the filtered attempt (which comes back judge "none"). Without this, the
            # 1678 uniform round RBC masks score ~0.95 on shape QC and keep-best keeps them.
            return VLRRResult("WARNING", score, {}, action="retune_here",
                              param_delta={"color_class_filter": action},
                              expected_effect={"cell_count": "down"},
                              judge_severity="major",
                              judge_problem="non_target_color_class",
                              notes=notes[:300])
        except ExecutionError:
            raise
        except Exception as exc:  # unreadable optional image evidence produces no proposal
            logger.warning("color-filter proposal failed (%s); normal path", exc)
            return None

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        parts = {
            "tiny": normalize(metrics["tiny_mask_fraction"], 0.0, 0.15),
            "large": normalize(metrics["large_mask_fraction"], 0.0, 0.10),
            "solidity": normalize(metrics["mean_solidity"], 0.80, 0.35),
            "border": normalize(metrics["border_mask_fraction"], 0.0, 0.40),
        }
        score = weighted_score(parts, {k: 1.0 for k in parts})
        # A zero-cell result (no detections, or an errored rerun whose mask loads as
        # empty) otherwise scores ~0.75 because tiny/large/border are all 0 -- which
        # would let it win _best over a real segmentation. Force it to the floor so
        # the rerun loop always keeps the usable result.
        if metrics["cell_count"] == 0:
            score = 0.0

        if locus == "upstream":
            return VLRRResult("FAIL", score, {}, action="conceded", fault="upstream",
                              upstream_target="image_loading",
                              notes="High flow_error traced to poor input quality; "
                                    "fix upstream before any seg tuning.")
        if locus == "tool_mismatch":
            return VLRRResult("FAIL", score, {}, action="conceded", fault="ceiling",
                              notes="High flow_error on a clean image: Cellpose-SAM "
                                    "unfit here. Consider different channel/model/SAM prompt.")
        # Already color/class-filtered this pass (the vision judge saw the ORIGINAL colors and
        # confirmed the removed population is a non-target class): ACCEPT. Do NOT re-litigate it
        # as under-detection -- the removed RBCs read as "missed cells" to the coverage QC, which
        # would wrongly retune to recover them and revert the class correction. judge_severity=
        # "none" ranks this above the pre-filter attempt (which the color proposal marked "major").
        if (context.get("parameters") or {}).get("color_class_filter"):
            return VLRRResult("PASS", score, {}, action="accept", judge_severity="none",
                              notes="color/class filter applied + vision-confirmed; accept "
                                    "(removed population is a non-target color class).")
        # this_step quality zone: build the deterministic recommendation first...
        if not triggered:
            det = VLRRResult("PASS", score, {}, action="accept")
        elif set(triggered) <= {"under_detection"} and score >= self.accept_score:
            # Masks are already high quality and the ONLY complaint is the noisy
            # under-detection cue -> accept instead of spending a retune that almost
            # always gets invalidated. (The vision judge below can still tighten this
            # back to a retune if it genuinely sees many missed cells.)
            det = VLRRResult("PASS", score, {}, action="accept",
                             notes=f"masks good (score {score:.2f}); under-detection cue "
                                   "not corroborated -> accept without retune.")
        else:
            delta, expect, why = self._diagnose(metrics, context)
            decision = "FAIL" if score < self.warn_score else "WARNING"
            det = VLRRResult(decision, score, {}, action="retune_here",
                             param_delta=delta, expected_effect=expect, notes=why)

        # COLOR/CLASS-FILTER retune proposal (VLRR action): on a color-separable over-segmented
        # scene (blood smear: a purple WBC/platelet minority among a pink RBC majority) a param
        # retune cannot help -- no diameter/threshold drops the non-target color class. A vision
        # judge (which sees the ORIGINAL colors) decides whether to drop one color population;
        # if so we retune with color_class_filter so keep-best + reconcile apply. Gated on a
        # cheap deterministic candidacy (bimodal per-mask B-R) so the VLM runs only on candidates.
        cf = self._color_filter_proposal(metrics, context, score)
        if cf is not None:
            return cf

        # ...then let a vision LLM judge the overlay and confirm/override the
        # accept-vs-retune decision and its direction. Falls back to `det` on any
        # failure or when no engine is wired (purely deterministic).
        #
        # Consulted on EVERY attempt (not once per stage): each re-segmentation
        # produces a new overlay, so judging each one (a) traces the visual reasoning
        # behind every parameter change and (b) catches under-detection the
        # deterministic mask QC misses -- e.g. a dense image where individual masks
        # look fine but many cells are uncovered, which the overlay makes obvious.
        if getattr(self, "llm_engine", None) is not None:
            from ..llm_recommend import vision_recommend_segmentation
            return vision_recommend_segmentation(self.llm_engine, context, metrics, det,
                                                 card=self.tuning_card(),
                                                 verify_refine=self.verify_refine)
        return det

    def _diagnose(self, m, context=None):
        tiny, large = m["tiny_mask_fraction"], m["large_mask_fraction"]
        if m["cell_count"] == 0:
            return self._diagnose_zero_detection(context or {})
        if self._under_detected(m):
            return self._diagnose_under_detection(m)
        if large > 0.10 and m["cell_count"] and m["mean_solidity"] < 0.35:
            return ({"diameter": "down", "flow_threshold": "up"},
                    {"large_mask_fraction": "down", "mean_solidity": "up"},
                    "Merged cells -> shrink diameter, raise flow_threshold.")
        if large > 0.10:
            return ({"diameter": "down", "cellprob_threshold": "down"},
                    {"cell_count": "up", "large_mask_fraction": "down"},
                    "Under-segmentation -> shrink diameter, lower cellprob.")
        if tiny > 0.15:
            return ({"cellprob_threshold": "up", "min_size": "up"},
                    {"tiny_mask_fraction": "down"},
                    "Over-segmentation -> raise cellprob, raise min_size.")
        return ({"cellprob_threshold": "up", "min_size": "up"},
                {"tiny_mask_fraction": "down"},
                "Noisy false positives -> raise cellprob, raise min_size.")

    def _diagnose_zero_detection(self, context):
        params = context.get("parameters", {}) or {}
        expect = {"cell_count": "up"}
        rescue_enabled = (
            getattr(self, "tool_name", None) == "Organoid_Segmenter_Tool"
            or os.getenv("SHAPE_ZERO_DETECTION_RESCUE", "").strip().lower() in {
                "1", "true", "yes", "on"
            }
        )
        if not rescue_enabled:
            return ({"cellprob_threshold": "down"}, expect,
                    "No cells detected -> lower cellprob_threshold to detect cells.")

        def _num(name, default):
            try:
                return float(params.get(name, default))
            except (TypeError, ValueError):
                return default

        cellprob = _num("cellprob_threshold", 0.0)
        # Cellpose resizes the image by 30 / diameter; null (or 30) is the native scale the
        # CO checkpoint was trained at. Probe native first, then one enlarged (15) and one
        # reduced (60) scale, in the order of the tool's own rescue plan. Omitted parameters
        # keep the tool contract, so the first rescue step changes only the threshold.
        diameter = _num("diameter", None)
        native = diameter is None or diameter == 30.0
        niter = _num("niter", 0.0)
        augment = bool(params.get("augment", False))
        tile_overlap = _num("tile_overlap", 0.0)

        if not augment and not native and (diameter < 30.0 or diameter > 60.0):
            return ({"diameter": None, "cellprob_threshold": -3.0}, expect,
                    "No cells detected away from the native scale -> restore the native scale.")
        if not augment and native:
            if cellprob > -2.0:
                return ({"cellprob_threshold": -2.0}, expect,
                        "No cells detected -> try a broad probability-threshold rescue.")
            if cellprob > -3.0:
                return ({"cellprob_threshold": -4.0}, expect,
                        "No cells detected at -2 -> try a very permissive threshold.")
            if cellprob <= -3.5:
                return ({"diameter": 15.0, "cellprob_threshold": -3.0}, expect,
                        "No cells detected after threshold sweep -> test an enlarged image (diameter 15).")
            return ({"diameter": 60.0, "cellprob_threshold": -3.0}, expect,
                    "No cells detected after enlarged-scale probe -> test a reduced image (diameter 60).")
        if not augment and diameter < 60.0:
            return ({"diameter": 60.0, "cellprob_threshold": -3.0}, expect,
                    "No cells detected -> test a reduced image (diameter 60).")
        if not augment:
            return ({"augment": True, "cellprob_threshold": -3.0}, expect,
                    "No cells detected across diameter probes -> add test-time augmentation.")
        if niter < 500.0:
            return ({"niter": 500, "cellprob_threshold": -3.0}, expect,
                    "No cells detected with augmentation -> raise dynamics iterations.")
        if tile_overlap < 0.25:
            return ({"tile_overlap": 0.25, "cellprob_threshold": -3.0}, expect,
                    "No cells detected after deeper dynamics -> increase tile overlap.")
        return ({"diameter": None, "cellprob_threshold": -5.0}, expect,
                "No cells detected after recall probes -> return to the native scale with maximum permissive threshold.")

    def _diagnose_under_detection(self, m):
        """Pick the Cellpose-SAM knob for missed cells from the missed-blob size.

        large missed (ratio>=1.8) -> flow can't span big cells: raise niter
            (max iterations) and bsize (max resize);
        small missed (ratio<=0.6) -> rescale too coarse: lower diameter;
        normal-sized but flow-rejected (flow_error elevated) -> raise flow_threshold;
        otherwise faint cells below the prob cut -> lower cellprob_threshold.
        """
        ratio = m.get("missed_blob_size_ratio", 1.0) or 1.0
        expect = {"cell_count": "up", "missed_foreground_fraction": "down"}
        if ratio >= 1.8:
            return ({"niter": "up"}, expect,
                    "Large cells missed -> raise niter (max iterations) so flow-following spans them.")
        if ratio <= 0.6:
            return ({"diameter": "down"}, expect,
                    "Small cells missed -> lower diameter (finer rescale).")
        if m.get("flow_error", 0.0) >= 0.25:
            return ({"flow_threshold": "up"}, expect,
                    "Well-formed cells rejected by flow QC -> raise flow_threshold.")
        return ({"cellprob_threshold": "down", "augment": True}, expect,
                "Faint cells missed -> lower cellprob_threshold + test-time augmentation.")


# ---- the agentic VRR (`SegmentationAgenticVerifier`; CARD.md "Agentic VRR") ---------------------------------
# The loop itself is `base.AgenticVerifier`; below is only the segmenter's own part: trial 0 (the input
# image and its preflight read), the target (from the request, else the CARD's default convention),
# SegmentationVerifier's readouts and QC score of every call with the color populations of its masks, the measured
# problems of a call (`measured_problems`: the admission's checks of the call at the defaults, with the colour
# evidence of a colour minority), the system's reference call (arm A's corrections of the default call: the measured
# cell scale, the colour rule) that is the default delivery unless a call with fewer measured problems overrides it,
# the agent's assessment of each overlay (the trial's `judge_severity`, the progress signal), the guard, the SAM
# server's replicas as the concurrency cap, the images attached to the prompts and the fallback
# (SegmentationVerifier's diagnosis).
#
# flow_error is measured (SegmentationVerifier's readouts: kept in the trial metrics, the trials and the logs) but is
# no guard, no admission check, no input of the fallback's moves and not shown to the agent: task 043 found it
# anti-correlated with AP50 (oriented rho = -0.35), and task 044 already dropped it from the vision evidence. The
# linear loop keeps its flow gate.

# the admission's scale check is arm A's, the linear loop's diameter correction (`integration.verify_managed_tool`,
# `initial_params.diameter_correction_needed`): mask_diameter_px off from the diameter used by this factor or more
SCALE_FACTOR = inspect.signature(diameter_correction_needed).parameters["factor"].default
# the admission's colour gates are arm A's, `color_class_filter.decide`'s gates on the population it keeps: a color
# population of at most `ccf._MINORITY_MAX_FRAC` and at least `ccf._MINORITY_MIN_COUNT` of a call's masks; its
# colour evidence, the masks of each population on the raw image (`_color_evidence`):
COLOR_TILES = 8             # masks shown per color population, evenly spaced in its (B - R) order
COLOR_TILE_PX = 96          # side of a mask's tile in the image
COLOR_TILE_MIN_PX = 32      # a tile's square: 2 x the mask's longer box side, at least this, clipped to the image
PROMPT_IMAGE_PX = 1024      # longest side of an image in a round prompt
FINAL_IMAGE_PX = 512        # longest side of each overlay in the final-choice prompt
INPUT_STATUS = "INPUT"
QC_READOUTS = ("cell_count", "foreground_coverage", "mask_area_median", "mask_diameter_px", "mask_area_CV",
               "tiny_mask_fraction", "large_mask_fraction", "border_mask_fraction", "mean_solidity",
               "missed_foreground_fraction", "count_miss_fraction", "missed_blob_size_ratio")
HIDDEN_READOUTS = ("flow_error",)    # measured, never shown to the agent (the flow_error note above)
# A call with the colour filter is measured on the filter's own split (`quantify`, `measured_problems`). Measured on
# its kept masks, the colour-filtered reference default read the removed population as missed cells and its kept
# class re-split into a "colour minority" (QC round 4, departure 6), so the unfiltered call and the call keeping the
# majority (red-cell) population could override it. The readouts SegmentationVerifier's under-detection gates read
# (`_under_detected`), measured on such a call's masks before the filter (the masks it removed count as covered):
UNDER_DETECTION_READOUTS = ("missed_foreground_fraction", "count_miss_fraction")
# ... and with them, from the same masks, the size of the blobs they leave uncovered:
PREFILTER_READOUTS = UNDER_DETECTION_READOUTS + ("missed_blob_size_ratio",)
COLOR_FILTERS = {"keep_high_br": "high", "keep_low_br": "low"}   # a color_class_filter -> the population it keeps
# the target: a request's section headed "## Target" (any case), up to its next H1 / H2 heading or a "---" rule;
# without one, the CARD's section:
TARGET_HEADING = re.compile(r"^##[ \t]+target[ \t]*$", re.IGNORECASE | re.MULTILINE)
TARGET_END = re.compile(r"^(#{1,2}[ \t]|-{3,}[ \t]*$)", re.MULTILINE)
DEFAULT_TARGET_SECTION = "## Default target convention"


class CallAssessment(VerifyReport):
    """The agent's reading of one finished call's overlay (VerifyReport: problem, severity, confidence)."""
    call: str


class SegPlan(BaseModel):
    """A round's plan: the shared ``WavePlan`` (analysis, hypotheses, moves, deliver) with the agent's assessment
    of each attached overlay, asked before the hypotheses and the moves (a subclass of WavePlan would put it
    last)."""
    analysis: str
    assessments: list[CallAssessment] = Field(
        default_factory=list, description="one assessment per call whose overlay is attached for assessment")
    hypotheses: list[Hypothesis] = Field(default_factory=list, description="your current hypothesis list")
    focus_levers: list[str] = Field(
        default_factory=list,
        description="the levers this round's moves may change, when you narrow (empty = no focus)")
    moves: list[WaveMove] = Field(description="the calls of this round; none = satisfied, stop")
    deliver: Optional[str] = Field(
        None, description="with no moves: the finished call to deliver; it must meet the final rule")


class SegChoice(BaseModel):
    """The final choice (the shared ``FinalChoice``'s fields), with the agent's assessment of each admissible call it
    has not assessed yet asked first: the override of the reference default reads the assessments."""
    assessments: list[CallAssessment] = Field(
        default_factory=list, description="one assessment per call the final question lists as not assessed yet")
    call: str = Field(description="the chosen call id")
    reasoning: str
    cited_readouts: list[str] = Field(default_factory=list, description="the readouts the choice rests on")


class PairwiseComparison(BaseModel):
    preferred_call: str
    baseline_call: str
    confidence: Literal['low','medium','high']
    reason: str = Field(description='Visible whole-cell benefits and trade-offs, including missed/merged/fragmented cells and background')
    cited_readouts: list[str]
    loss_explained: bool = Field(description='Whether any >25% loss of count or foreground coverage is visually justified')
    loss_explanation: str


class BalancedSegChoice(SegChoice):
    comparison: PairwiseComparison


class BalancedSegPlan(SegPlan):
    comparison: Optional[PairwiseComparison] = Field(default=None,
        description='Required when stopping with a nonbaseline delivery; compare against the original default')


from ..segmentation_adaptive import ContinuationEvidence


class GainSegPlan(SegPlan):
    continuation: Optional[ContinuationEvidence] = Field(default=None,
        description='Evidence-based continuation after assessing finished calls; unknown expected gain means keep searching')


class GainBalancedSegPlan(BalancedSegPlan):
    continuation: Optional[ContinuationEvidence] = Field(default=None,
        description='Evidence-based continuation after assessing finished calls; unknown expected gain means keep searching')


def _image_array(path: str):
    try:
        from skimage.io import imread

        return np.asarray(imread(path))
    except Exception as exc:  # noqa: BLE001 -- an unreadable image leaves the record without QC
        logger.warning("segmentation agentic: cannot read %s (%s)", path, exc)
        return None


def input_record(base_kwargs: Mapping[str, Any]) -> dict:
    """Trial 0: the input image as a record (not a tool call) -- its path and id, size, channels,
    dtype and the image QC (``preprocessing.absolute_quality_metrics`` on the grey image)."""
    from .preprocessing import absolute_quality_metrics

    path = (base_kwargs or {}).get("image")
    record: dict[str, Any] = {"status": INPUT_STATUS, "image": path, "image_id": (base_kwargs or {}).get("image_id"),
                              "image_qc": {}, "image_shape": None, "channels": None, "dtype": None}
    arr = _image_array(path) if path else None
    if arr is not None:
        record.update(image_shape=[int(s) for s in arr.shape[:2]], dtype=str(arr.dtype),
                      channels=int(arr.shape[2]) if arr.ndim == 3 else 1)
    gray = _load_gray(path) if path else None
    if gray is not None:
        bit_range = (float(np.iinfo(arr.dtype).max)
                     if arr is not None and np.issubdtype(arr.dtype, np.integer) else 255.0)
        record["image_qc"] = {k: v for k, v in absolute_quality_metrics(gray, bit_range).items()
                              if isinstance(v, float) and math.isfinite(v)}
    return record


def _png(image, max_px: int) -> bytes | None:
    """A display PNG of `image` (a path or an array): per-channel 1-99 percentile to uint8, longest side
    at most `max_px`."""
    from PIL import Image

    arr = _image_array(image) if isinstance(image, str) else np.asarray(image)
    if arr is None or arr.size == 0:
        return None
    arr = np.asarray(arr, dtype=np.float64)
    if arr.ndim == 3 and arr.shape[2] > 3:
        arr = arr[..., :3]
    lo, hi = np.percentile(arr, 1), np.percentile(arr, 99)
    arr = np.clip((arr - lo) / (hi - lo if hi > lo else 1.0), 0, 1) * 255
    img = Image.fromarray(arr.astype(np.uint8).squeeze())
    img.thumbnail((max_px, max_px))
    buf = io.BytesIO()
    img.save(buf, format="PNG")
    return buf.getvalue()


def sam_concurrency() -> int:
    """Calls that may run at the same time: the SAM server's replicas (``/health``, summed over the
    servers in $SHAPE_SAM_SERVER_URL); 1 without a server (the in-process model runs one call at a time)."""
    urls = [u.strip().rstrip("/") for u in os.environ.get("SHAPE_SAM_SERVER_URL", "").split(",") if u.strip()]
    total = 0
    for url in urls:
        try:
            with urlopen(f"{url}/health", timeout=10) as response:
                total += max(1, int(json.loads(response.read()).get("replicas") or 1))
        except Exception:  # noqa: BLE001 -- an unreachable server counts as one slot
            total += 1
    return max(1, total)


class _Recording:
    """An engine that keeps the error its `generate` raised (re-raised: `color_class_filter.decide` catches it, so
    the error is the failed try's reason)."""

    def __init__(self, engine, observer=None, phase=None, attempt=1):
        self.engine, self.error = engine, None
        self.observer, self.phase, self.attempt = observer, phase, attempt

    def generate(self, *args, **kwargs):
        response, started = None, time.perf_counter()
        try:
            response = self.engine.generate(*args, **kwargs)
            return response
        except Exception as exc:
            self.error = exc
            raise
        finally:
            if self.observer is not None:
                self.observer.record_llm_call(
                    self.phase, self.attempt, time.perf_counter() - started, response,
                    args[0] if args else kwargs.get("content", []),
                    "error" if self.error else "returned", type(self.error).__name__ if self.error else None)


class SegmentationAgenticVerifier(AgenticVerifier):
    """Agentic VRR for Cell_Segmenter_Tool (``base.AgenticVerifier``, StepVerifier wave mode; CARD.md
    "Agentic VRR"), the tool's verifier in ``stage_map`` (``verifier_by_tool``).

    Trial 0 is the input image (``input_record``), a measurement and not a tool call; the agent sees the
    image itself and the preflight read of it (``_preflight``: ``image_context.read_image_context_profile``,
    once per image). A call's readouts and QC score (``qc_score``) are SegmentationVerifier's, with the
    color populations of its masks (``_color_populations``: ``color_class_filter.color_cluster_stats``,
    deterministic). The task text states the target (``target``): the "## Target" section of the user's request
    (``request``, set by ``Verifier.resolve_managed_stage``: the solve() query, else the stage's sub_goal), else
    the CARD's default convention. The admission
    (``admission_call``): round 1 is the tool at its CARD defaults (`default`); a measured problem of that call --
    it failed, the guard, or ``admission_problems`` (``measured_problems``: the linear loop's triggers without its
    flow gate, arm A's scale check and arm A's colour gates -- a colour minority, whose colour evidence, the masks
    of each color population on the raw image, ``_color_evidence``, is then attached after the input image) --
    starts the tuning, else the default call is delivered as it is (``admit_defaults``: the linear loop's
    deterministic admission; the agent is not asked). After a RUN, the system runs one reference call from the
    default call before the agent's first round (``reference_moves``) with every arm A correction that holds on the
    default call: the measured diameter when the scale check holds, and arm A's colour rule
    (``color_class_filter.decide``, asked on a colour minority) when it applies. The agent paces its own search,
    broad then narrow as it judges (stated in every round prompt, ``schedule_text``): it may name
    ``focus_levers`` in a plan, and that round's moves change only them; once a round has run under a focus, every
    later plan names one again, never widened (the shared layer's narrowing, ``base._schedule_problem``). The
    reference call is the default delivery; the agent's final choice overrides it only with strictly fewer measured
    problems and a better assessment of its own (``override_allowed``; a call with the colour filter is measured on
    the filter's own split: ``measured_problems``); the final question asks the assessment of each admissible call
    not assessed yet (``SegChoice``). Without a reference call that finished and the guard admits, the default
    call is the default delivery under the same rule when the guard admits it (``reference_default``). With each
    plan the agent assesses the overlays of the calls just finished (``SegPlan.assessments``); an
    assessment is recorded as that trial's ``judge_severity`` / ``judge_problem`` and is the progress
    signal (``plateau_after_plan``: ``patience`` rounds in a row without a better assessed severity).
    Where a rule chooses -- the fallback's final choice, an agent choice still inadmissible after the retry --
    the reference default is delivered when there is one; else, as in ``least_bad`` (no admissible call), the
    call with the better (assessment, QC score). The
    guard: cells and a valid overlay (flow_error is measured, never a guard and never shown: the note above
    ``SCALE_FACTOR``). The concurrency cap is the SAM server's replicas (``sam_concurrency``). The fallback
    moves: SegmentationVerifier's diagnosis of the best call, turned into values, then arm A's diameter
    correction of that call (``_scale_problem``).
    """

    name = "segmentation"
    tool_dir = "cell_segmenter"
    score_key = "qc_score"
    admission_call = True                 # round 1 = the tool at its defaults; the admission decides on it
    plateau_after_plan = True
    narrowing = True                      # the shared narrowing: the agent's focus, never widened
    key_readouts = QC_READOUTS
    plan_schema = SegPlan
    choice_schema = SegChoice
    tool_phrase = "cell-segmentation tool (Cellpose-SAM)"
    efficiency_defaults = None        # None: segmentation_policy.LEGACY (the historical stop rules); see its switches
    unit = "image"
    value_examples = '24, 0.6, true, null, "keep_high_br"'
    cite_text = "Cite the evidence your choice rests on (overlays, QC readouts, your assessments)."
    repair_note = " (assessments and hypotheses may be left empty)"   # + the base's focus note
    verify_refine = False                 # set by resolve_managed_stage; not used by this loop
    request: str | None = None            # set by resolve_managed_stage: the solve() query, else the sub_goal
    # set by the segmentation cohort (`segmentation_cohort`, through `verifier_options`):
    cohort_levers: dict | None = None     # the cohort mode: the group's consensus levers (`cohort_moves`)
    start_params: dict | None = None      # a routed image: its group's consensus as the tool defaults
    input_record = staticmethod(input_record)

    def __init__(self, *, gain_policy='legacy', native_gain_noise=.001, **kwargs):
        from ..balanced_policy import selected_policy, POLICY
        if gain_policy not in ('legacy', 'evidence_v1'):
            raise ValueError('unknown segmentation gain policy')
        if not math.isfinite(native_gain_noise) or native_gain_noise < 0:
            raise ValueError('native gain noise must be finite and nonnegative')
        self.gain_policy, self.native_gain_noise = gain_policy, float(native_gain_noise)
        self.continuation_events = []
        self._continuation_evidence = None
        self.balanced_vrr = selected_policy() == POLICY
        if self.balanced_vrr:
            # Set before the base wraps the schema for four-stage VRR.
            self.plan_schema, self.choice_schema = BalancedSegPlan, BalancedSegChoice
        if self.gain_policy == 'evidence_v1':
            self.plan_schema = GainBalancedSegPlan if self.balanced_vrr else GainSegPlan
        super().__init__(**kwargs)
        from ..segmentation_policy import verifier_policy
        self.efficiency_policy = verifier_policy(self.efficiency_defaults)
        self.efficiency_events: list[dict] = []
        self.assessments: dict[str, dict] = {}     # call -> the agent's assessment of its overlay
        self.preflight: dict | str = "unavailable (not read yet)"   # the input's preflight read (`_preflight`)
        self.color_evidence: dict | None = None    # the mask ids each population shows (`_color_evidence`)
        self.color_decision: dict | str | None = None   # the colour rule's decision on the default call
        self._evidence: tuple | None = None        # (label, PNG) of the colour evidence, attached after the input
        self._overlays: dict[int, str] = {}        # trial index -> the call's overlay PNG
        self._masks: dict[int, str] = {}           # trial index -> the call's mask (.npy)
        self._colors: dict[int, Any] = {}          # trial index -> the color populations of its masks
        self._prefilter_colors: dict[int, Any] = {}   # trial index -> those of its masks before the colour filter
        self._linear = SegmentationVerifier()      # readouts, QC score and diagnosis only (no engine)
        self._input_path: str | None = None
        self._input_png: bytes | None = None
        self._rgb = None                           # the raw input image, for the color populations
        self._attempts: dict = {}                  # the finished calls, as last bound

    # ---- measurements ---------------------------------------------------------------------------------
    @staticmethod
    def make_adapter(base_kwargs):
        """The native result is the verifier input; the context carries the image path."""
        image_path = (base_kwargs or {}).get("image")
        return lambda result, upstream: (result, {"image_path": image_path, "upstream": dict(upstream or {})})

    def concurrency(self) -> int:
        return sam_concurrency()

    def quantify(self, tool_output, context) -> dict[str, float]:
        """Trial 0: the image QC, and the preflight read of the raw image (`_preflight`). A call:
        SegmentationVerifier's readouts of its mask (mask QC, overlay QC, uncovered foreground), its
        mask_diameter_px (`initial_params.diameter_from_mask_area` of its median mask area, as the linear loop
        measures the cells) and the color populations of its masks (`_color_populations`); the tool's seconds
        are the call's report. A call whose colour filter acted (the tool's `prefilter_mask_path`, its masks
        before the filter): the under-detection readouts (`UNDER_DETECTION_READOUTS`) measured on those masks, so
        the masks the filter removed count as covered (the other readouts stay on the kept masks), and the color
        populations of those masks (`_prefilter_colors`, the split the filter made)."""
        result = tool_output if isinstance(tool_output, Mapping) else {}
        index = int(context.get("trial_index") or 0)
        if index == 0 or result.get("status") == INPUT_STATUS:
            self._input_path = result.get("image")
            self.preflight = self._preflight(self._input_path)
            return numeric(result.get("image_qc") or {})
        if result.get("seconds") is not None:
            self._reports[index] = {"seconds": result["seconds"]}
        vinput, ctx = adapt_segmentation(dict(result), context.get("upstream"), image_path=context.get("image_path"))
        self._overlays[index] = ctx.get("overlay_path")
        self._masks[index] = ctx.get("mask_path")
        self._colors[index] = self._color_populations(vinput[0])
        metrics = self._linear.quantify(vinput, ctx)
        if metrics.get("mask_area_median"):                  # a call with masks
            metrics["mask_diameter_px"] = diameter_from_mask_area(metrics["mask_area_median"])
        prefilter = result.get("prefilter_mask_path")
        if prefilter and os.path.exists(prefilter):
            masks = np.load(prefilter)
            self._prefilter_colors[index] = self._color_populations(masks)
            gray = _load_gray(context.get("image_path")) if context.get("image_path") else None
            coverage = (_uncovered_foreground(gray, masks) if gray is not None else None) or {}
            metrics.update({k: float(coverage.get(k, 0.0) or 0.0) for k in PREFILTER_READOUTS})
        return metrics

    def _preflight(self, path: str | None) -> dict | str:
        """The preflight read of the raw input image: `image_context.read_image_context_profile` (the vision
        model's specimen / object-class read made before any segmentation) with this verifier's engine; its
        process cache makes it one read per image. Its specimen context, visible object classes, provisional
        target hint and confidence -- or "unavailable (<why>)": no engine, an unreadable image, a failed or
        an unconfident read (the function's conservative fallback, always of low confidence)."""
        if not path:
            return "unavailable (no input image)"
        if self.llm_engine is None:
            return "unavailable (no VLM engine available)"
        try:
            from shape.utils.image_processor import ImageProcessor
            from ..image_context import read_image_context_profile

            profile = read_image_context_profile(ImageProcessor.load_image(path),
                                                 engine=_Recording(self.llm_engine, self, "preflight"))
        except Exception as exc:  # noqa: BLE001 -- information only: a read that fails is shown as unavailable
            if is_fatal_llm_error(exc):
                self.reasoner_failures += 1
                exc.shape_question = "preflight"
                raise
            if isinstance(exc, ExecutionError):
                from shape.utils.schemas import QuestionFailed
                self.reasoner_failures += 1
                self.question_failed(QuestionFailed(str(exc), label="preflight", cause=str(exc)))
                if "malformed preflight answer" in str(exc):
                    return "unavailable (malformed VLM response)"
            return f"unavailable ({type(exc).__name__}: {exc})"[:300]
        if profile.confidence == "low":
            return f"unavailable ({'; '.join(profile.limitations) or 'no confident read'})"
        return {"specimen_context": profile.specimen_context,
                "visible_object_classes": list(profile.visible_object_classes),
                "provisional_target_hint": profile.provisional_target_hint, "confidence": profile.confidence}

    def _color_populations(self, masks) -> Any:
        """`color_class_filter.color_cluster_stats` of a call's masks on the raw input image (deterministic, no
        model): the Otsu split of the per-mask mean (blue - red) intensity -- the objects, the cut and each
        population's masks, fraction and mean (the per-mask values left out); None without a split (fewer than
        4 masks, or no spread)."""
        if self._rgb is None and self._input_path:
            self._rgb = _image_array(self._input_path)
        masks = np.asarray(masks)
        if self._rgb is None or masks.shape[:2] != self._rgb.shape[:2]:
            return "unavailable (no raw image of the mask's size)"
        from shape.tools.cell_segmenter.color_class_filter import color_cluster_stats

        stats = color_cluster_stats(self._rgb, masks)
        if stats is None:
            return None
        return {"objects": stats["n"], "cut_b_minus_r": round(stats["cut"], 2),
                "high": {"masks": stats["hi_n"], "fraction": round(stats["hi_frac"], 4),
                         "mean_b_minus_r": round(stats["hi_mean"], 2)},
                "low": {"masks": stats["lo_n"], "fraction": round(stats["lo_frac"], 4),
                        "mean_b_minus_r": round(stats["lo_mean"], 2)}}

    def localize(self, metrics, context):
        return "this_step", [], []

    def qc_score(self, metrics: Mapping[str, float]) -> float:
        """SegmentationVerifier's mask QC score (tiny / large / solidity / border; 0 without cells)."""
        if not metrics or "cell_count" not in metrics:
            return 0.0
        return float(self._linear.adjudicate(dict(metrics), "this_step", [], {}).scalar_score)

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        if int(context.get("trial_index") or 0) != 0:
            return VLRRResult("PASS", self.qc_score(metrics), dict(metrics), action="retune_here")
        self.begin(None)
        return VLRRResult("PASS", 0.0, dict(metrics), action="retune_here", notes="the input image")

    def call_error(self, index: int, state) -> Optional[str]:
        """A call that crashed, returned an error or failed the tool's own overlay QC."""
        result = state.trials[index][2]
        if isinstance(result, Mapping) and (result.get("error") or result.get("status") == "failed"):
            return str(result.get("error") or "overlay QC failed")
        return super().call_error(index, state)

    def guard(self, inp: VLRRResult, vres: VLRRResult) -> list[str]:
        """Cells and a valid overlay (no flow_error check: the flow_error note above `SCALE_FACTOR`)."""
        m, fails = vres.metrics or {}, []
        if m.get("cell_count", 0) <= 0:
            fails.append("no cells detected")
        if m.get("overlay_is_black") or m.get("coverage_implausible"):
            fails.append("visual output invalid (black overlay or implausible coverage)")
        return fails

    # ---- the admission: the call at the defaults ------------------------------------------------------
    def admit_defaults(self, inp, attempts, remaining: int) -> dict:
        """The linear loop's deterministic admission on round 1, the call at the defaults (`default`): RUN on a
        measured problem (the call failed, the guard, `admission_problems`), else SKIP -- the default call is delivered
        as it is. The agent is not asked: the default call is the default delivery when no reference call runs
        (`reference_default`), and no call has strictly fewer than no measured problems, so a tuning started without
        one could not deliver."""
        call = self.label(self.rounds[0][0])
        if call in attempts:
            problems = self.guard(inp, attempts[call][2]) + self.admission_problems(inp, attempts[call][2])
        else:
            error = next((f["error"] for f in self.failed if f["call"] == call), "")
            problems = [f"the call at the defaults failed: {error[-300:]}"]
        decision = "RUN" if problems else "SKIP"
        admission = {"call": call, "rule": decision, "problems": problems, "answer": None, "decision": decision}
        return self.set_admission(admission)

    def admission_problems(self, inp: VLRRResult, vres: VLRRResult) -> list[str]:
        """The measured problems of the call at the defaults beyond the guard (no cells and an invalid overlay
        are the guard's): `measured_problems` of that call, which used the defaults' diameter. A colour minority
        also draws the call's colour evidence (`_color_evidence`), attached after the input image from then on."""
        index = self.rounds[0][0]                                    # the call at the defaults (`admit_defaults`)
        populations = self._colors.get(index)
        if self._color_minority(populations):
            self._color_evidence(index)
        return self.measured_problems(vres, self.defaults()["diameter"], populations)

    def measured_problems(self, vres: VLRRResult, diameter, populations, color_filter: Optional[str] = None,
                          prefilter=None) -> list[str]:
        """The measured problems of one call, each naming the readout, its value and the threshold -- the
        admission's checks on that call: the linear loop's shape gates (`SegmentationVerifier.localize`, unchanged,
        on the readouts with flow_error set aside), arm A's scale check against the `diameter` the call used
        (`mask_diameter_px` off from it by `SCALE_FACTOR` or more, either way:
        `initial_params.diameter_correction_needed`) and arm A's colour gates on the call's color `populations`
        (`_color_minority`). Under-detection (`missed_foreground_fraction`, `count_miss_fraction`) is no problem:
        the readouts are shown, as information. They count every textured pixel no mask covers, the target's or
        not (unstained cells, background), so they neither start the tuning nor count in an override: in the
        NeurIPS22 CellSeg pilots the 20 searches started on under-detection alone found at most +0.002 AP50 over
        the default call, and in the dev30 run 4 of 5 such images were overridden for a worse AP50, the agent
        growing masks to cut the readout. A call with a `color_filter` (its color_class_filter; the admission's call
        at the defaults has none): no colour-minority check on its kept masks (the filter acted on that split; their
        populations stay information), and instead arm A's balance gate on the split it made (`_filter_balance`,
        on `prefilter`: the color populations of its masks before the filter)."""
        linear, m = self._linear, dict(vres.metrics or {})
        _, triggered, _ = linear.localize({**m, "flow_error": 0.0}, {"upstream": {}})    # its flow gate never fires
        problems = [f"{k} {m[k]:.3f} {'above' if op == '>' else 'below'} {thr:g}"
                    for k, thr, op in linear._SHAPE_GATES if k in triggered]
        # In balanced_v1 a scale mismatch is a candidate hypothesis, not a
        # stand-alone reason to launch a search or prefer the corrected scale.
        scale = None if self.balanced_vrr else self._scale_problem(m, diameter)
        colour = self._filter_balance(color_filter, prefilter) if color_filter else self._color_minority(populations)
        return problems + [p for p in (scale, colour) if p]

    def call_problems(self, attempts, call: str) -> list[str]:
        """The measured problems of a finished call: the guard's failures (a call without cells or with an invalid
        overlay), then `measured_problems` at the diameter of its configuration, on its color populations, with its
        colour filter and the color populations of its masks before the filter."""
        _, params, vres = attempts[call]
        index = self._labels.get(call)
        return self.guard(None, vres) + self.measured_problems(
            vres, params.get("diameter"), self._colors.get(index), params.get("color_class_filter"),
            self._prefilter_colors.get(index))

    @staticmethod
    def _filter_balance(color_filter: str, prefilter) -> Optional[str]:
        """Arm A's balance gate on a colour filter's split (the gates `color_class_filter.decide` applies before it
        filters), as a problem: the population the filter keeps (`COLOR_FILTERS`) above `_MINORITY_MAX_FRAC` or below
        `_MINORITY_MIN_COUNT` of the masks before the filter (`prefilter`, their color populations; the fraction is
        decide's: the kept masks / all of them); None when it holds, or without the split (the filter did not
        act)."""
        side = COLOR_FILTERS.get(color_filter)
        if side is None or not isinstance(prefilter, Mapping):
            return None
        kept, total = prefilter[side]["masks"], prefilter["objects"]
        fails = ([f"above {ccf._MINORITY_MAX_FRAC:.0%}"] if kept / total > ccf._MINORITY_MAX_FRAC else []) + (
            [f"below {ccf._MINORITY_MIN_COUNT} masks"] if kept < ccf._MINORITY_MIN_COUNT else [])
        if not fails:
            return None
        return f"colour filter keeps {kept} of {total} masks ({kept / total:.1%}), {' and '.join(fails)}"

    @staticmethod
    def _color_minority(populations) -> Optional[str]:
        """Arm A's colour gates on a call's color populations (`_color_populations`): the gates of
        `color_class_filter.decide` on the population it keeps -- at most `_MINORITY_MAX_FRAC` and at least
        `_MINORITY_MIN_COUNT` of the masks --, met by one population, as a problem; None otherwise (or without a
        split). The fraction is decide's: the population's masks / all masks."""
        if not isinstance(populations, Mapping):
            return None
        total = populations["objects"]
        for side, tint in (("high", "bluer"), ("low", "redder")):
            masks = populations[side]["masks"]
            if masks / total <= ccf._MINORITY_MAX_FRAC and masks >= ccf._MINORITY_MIN_COUNT:
                return (f"colour minority: {masks} of {total} masks ({masks / total:.1%}) in the {side} ({tint}) "
                        "population")
        return None

    def _color_evidence(self, index: int) -> None:
        """The colour evidence of call `index`'s masks: what arm A's vision model looks at (the masks on the raw
        image's own colours, `color_class_filter.render_evidence_png`), without its question, its rules and its
        decision. Two rows, the high (bluer / purpler) population on top and the low (redder / pinker) one below;
        in each, up to `COLOR_TILES` of the population's masks, evenly spaced in its (B - R) order (a row spans
        its range); each mask a square of the raw image (2 x the mask's longer box side, at least
        `COLOR_TILE_MIN_PX`, around the box's centre, clipped to the image; the boxes of one
        `ndimage.find_objects` pass), stretched as `render_evidence_png` stretches the image (its 1st and 99th
        percentile), resized to `COLOR_TILE_PX` and the mask outlined thinly on it. No text in the image: the
        attachment label describes the rows. Recorded: (label, PNG) as `_evidence`, the mask ids each row shows
        as `color_evidence`; an image that cannot be drawn is left out (logged)."""
        from PIL import Image
        from scipy import ndimage
        from skimage.segmentation import find_boundaries

        try:
            mask = np.load(self._masks[index])
            stats = ccf.color_cluster_stats(self._rgb, mask)
            raw = ccf._to_rgb(self._rgb)
            lo, hi = np.percentile(raw, (1, 99))           # render_evidence_png's two percentiles, in one pass
            boxes, (h, w), size = ndimage.find_objects(mask), mask.shape, (COLOR_TILE_PX, COLOR_TILE_PX)
            rows, shown = [], {}
            for side in ("high", "low"):
                ids = sorted((i for i, v in stats["br"].items() if (v > stats["cut"]) == (side == "high")),
                             key=lambda i: (stats["br"][i], i))
                picks = [ids[k] for k in np.unique(np.round(np.linspace(0, len(ids) - 1, min(COLOR_TILES, len(ids))))
                                                   .astype(int))]
                tiles = []
                for i in picks:
                    box = boxes[i - 1]
                    square = max(COLOR_TILE_MIN_PX, 2 * max(box[0].stop - box[0].start, box[1].stop - box[1].start))
                    y0 = math.floor((box[0].start + box[0].stop - square) / 2)
                    x0 = math.floor((box[1].start + box[1].stop - square) / 2)
                    window = (slice(max(0, y0), min(h, y0 + square)), slice(max(0, x0), min(w, x0 + square)))
                    pixels = (np.clip((raw[window] - lo) / (hi - lo + 1e-8), 0, 1) * 255).astype(np.uint8)
                    tile = np.array(Image.fromarray(pixels).resize(size, Image.BILINEAR))
                    inside = np.array(Image.fromarray((mask[window] == i).astype(np.uint8)).resize(size, Image.NEAREST))
                    tile[find_boundaries(inside > 0, mode="outer")] = (30, 30, 30)   # render_evidence_png's outline
                    tiles.append(tile)
                rows.append(tiles)
                shown[side] = [int(i) for i in picks]
            gap, cols = 2, max(len(tiles) for tiles in rows)                  # tiles 2 px apart, on black
            canvas = np.zeros((2 * COLOR_TILE_PX + gap, cols * COLOR_TILE_PX + (cols - 1) * gap, 3), np.uint8)
            for r, tiles in enumerate(rows):
                for c, tile in enumerate(tiles):
                    y, x = r * (COLOR_TILE_PX + gap), c * (COLOR_TILE_PX + gap)
                    canvas[y:y + COLOR_TILE_PX, x:x + COLOR_TILE_PX] = tile
            buf = io.BytesIO()
            Image.fromarray(canvas).save(buf, format="PNG")
        except Exception as exc:  # noqa: BLE001 -- evidence only: an image that cannot be drawn is left out
            self.log("color_evidence", text=f"not drawn: {type(exc).__name__}: {exc}")
            return
        self._evidence = (f"COLOUR POPULATIONS {self.label(index)} (top row: high, {stats['hi_n']} masks, "
                          f"{stats['hi_frac']:.1%}, mean B-R {stats['hi_mean']:+.1f}; bottom row: low, "
                          f"{stats['lo_n']} masks, {stats['lo_frac']:.1%}, mean B-R {stats['lo_mean']:+.1f}):",
                          buf.getvalue())
        self.color_evidence = shown
        self.log("color_evidence", text=json.dumps(shown))

    @staticmethod
    def _scale_problem(metrics: Mapping[str, float], used) -> Optional[str]:
        """Arm A's scale check on a call: its mask_diameter_px off from the diameter it used by `SCALE_FACTOR` or
        more, either way (`initial_params.diameter_correction_needed`), as a problem; None within the factor."""
        measured = metrics.get("mask_diameter_px")
        if not diameter_correction_needed(measured, used, SCALE_FACTOR):
            return None
        return (f"mask_diameter_px {measured:.2f} is {measured / used:.2f} x the diameter used ({used:g}), off by a "
                f"factor of {SCALE_FACTOR:g} or more")

    def reference_moves(self, inp, attempts) -> list[dict]:
        """The system's reference call after a RUN admission: ONE move from the call at the defaults holding every
        arm A correction that holds on it -- `diameter` set to its mask_diameter_px when the admission's scale
        check holds (`_scale_problem`; a number lever: the value as measured, within the lever's range), and
        `color_class_filter` set to arm A's colour rule when it applies (`_color_rule`); its diagnosis names each
        rule (as the system's rules: the agent never sees "arm A"). None of them -> no reference call."""
        index = self.rounds[0][0]
        call = self.label(index)
        if call not in attempts:
            return []
        metrics, changes, rules = attempts[call][2].metrics or {}, [], []
        if self._scale_problem(metrics, attempts[call][1].get("diameter")):
            changes.append({"lever": "diameter", "value": json.dumps(metrics["mask_diameter_px"])})
            rules.append(f"the system's diameter rule (mask_diameter_px {metrics['mask_diameter_px']:.1f})")
        try:
            applied = self._color_rule(index)
        except ExecutionError as exc:
            if is_fatal_llm_error(exc):
                raise
            rules.append(f"the system's colour rule failed ({exc})")
            return [{"parent": call, "changes": changes, "diagnosis": "; ".join(rules),
                     "undecided": {"color_class_filter": "undecided (the colour rule failed)"}, "error": str(exc)}]
        if applied:
            changes.append({"lever": "color_class_filter", "value": json.dumps(applied)})
            reason = self.color_decision.get("reason") or "no reason given"
            rules.append(f"the system's colour rule: {applied} ({reason})")
        if self.cohort_levers:                   # the cohort mode: the group's consensus on top of the corrections
            changed = {c["lever"] for c in changes}
            consensus = {k: v for k, v in self.cohort_levers.items() if k not in changed}
            changes += [{"lever": k, "value": json.dumps(v)} for k, v in consensus.items()]
            if consensus:
                rules.append(f"the group's calibrated parameters {json.dumps(consensus, default=str)}")
        if not changes:
            return []
        return [{"parent": call, "changes": changes, "diagnosis": "; ".join(rules)}]

    def _color_rule(self, index: int) -> Optional[str]:
        """Arm A's colour rule on call `index`'s masks, asked when they have a colour minority (`_color_minority`):
        `color_class_filter.decide` with this verifier's engine -- its own question, rules and gates (confidence,
        kept fraction, kept count) on the raw image --, tried as every reasoner question is (`REASONER_ATTEMPTS`
        times, `REASONER_RETRY_SECONDS` apart): decide catches the engine's errors, so a try without an answer
        (no confidence) is a failed one, the engine's error its reason (`_Recording`). Recorded as `color_decision`:
        its action, confidence, reason and applied; or "unavailable (<why>)" -- no reasoner (not asked), a mask that
        cannot be read, or every try failed (then counted in `reasoner_failures` and recorded as an execution
        error). Fatal errors propagate. Returns its `applied` when that is keep_high_br / keep_low_br, else None."""
        if not self._color_minority(self._colors.get(index)):
            return None
        if self.llm_engine is None:
            self.reasoner_failures += 1
            error = NoEngineError(f"{self.name}: no reasoner engine is configured for the colour rule")
            error.shape_question = "color_rule"
            raise error
        try:
            mask = np.load(self._masks[index]).astype(np.int32)
        except Exception as exc:  # noqa: BLE001 -- a decision that cannot be made sets no filter
            self.color_decision = f"unavailable ({type(exc).__name__}: {exc})"[:300]
            self.log("color_decision", text=self.color_decision)
            return None
        for attempt in range(1, vrr_base.REASONER_ATTEMPTS + 1):
            engine = _Recording(self.llm_engine, self, "color_rule", attempt)
            try:
                info = ccf.decide(engine, self._rgb, mask)
            except ExecutionError as exc:
                if is_fatal_llm_error(exc):
                    self.reasoner_failures += 1
                    exc.shape_question = "color_rule"
                    raise
                if engine.error is None and "malformed answer" not in str(exc):
                    engine.error = exc
                info = {"candidate": True, "confidence": None}
            if info.get("confidence") is not None or not info.get("candidate"):
                break
            why = f"{type(engine.error).__name__}: {engine.error}" if engine.error else "no usable answer"
            self.log("reasoner_error", label="color_rule", text=f"try {attempt}: {why}")
            if attempt < vrr_base.REASONER_ATTEMPTS:
                time.sleep(vrr_base.REASONER_RETRY_SECONDS)
                continue
            self.reasoner_failures += 1
            self.color_decision = f"unavailable ({why})"[:300]
            self.log("color_decision", text=self.color_decision)
            from shape.utils.schemas import QuestionFailed
            self.question_failed(QuestionFailed(why, label="color_rule", cause=why))
            raise ExecutionError(f"the colour rule failed {vrr_base.REASONER_ATTEMPTS} times; last error: {why}") from engine.error
        self.color_decision = {k: info.get(k) for k in ("action", "confidence", "reason", "applied")}
        self.log("color_decision", text=json.dumps(self.color_decision, default=str))
        applied = info.get("applied")
        return applied if applied in COLOR_FILTERS else None

    def reference_default(self, inp, attempts) -> Optional[str]:
        """The default delivery: the system's reference call when it finished and the guard admits it (the base
        rule); without one, the call at the defaults (the admission's call, `default`: round 1) when it finished and
        the guard admits it; None without either. Read from round 1, not the admission's record, so the default is
        already bound for the first round prompt (`recommend_wave` binds before the admission runs)."""
        if self.balanced_vrr and self.rounds:
            original = self.label(self.rounds[0][0])
            if original in attempts and not self.guard(inp, attempts[original][2]):
                return original
        default = super().reference_default(inp, attempts)
        if default is not None or not self.rounds:
            return default
        call = self.label(self.rounds[0][0])
        return call if call in attempts and not self.guard(inp, attempts[call][2]) else None

    # ---- the cohort mode: the group's consensus applied, no agent round -------------------------------
    def defaults(self) -> dict:
        """The CARD defaults; for an image the cohort routes to its own VRR, its group's consensus in their place
        (`start_params`: the call at the defaults is then the group's configuration)."""
        base = super().defaults()
        return {**base, **{k: v for k, v in (self.start_params or {}).items() if k in base}}

    def unbeatable_reference(self, inp, attempts):
        """No candidate can have strictly fewer than zero problems under our override rule."""
        if self.balanced_vrr or self.efficiency_policy["zero_problem_reference"] != "1":
            return None
        reference = self.reference_default(inp, attempts)
        if reference in attempts and not self.guard(inp, attempts[reference][2]) and not self.call_problems(attempts, reference):
            return reference
        return None

    def confirm_stop_required(self, plan, inp, attempts):
        if not self.confirm_stop or self.efficiency_policy["stop_confirmation"] == "always":
            return self.confirm_stop
        call = getattr(plan, "deliver", None)
        default = self.reference_default(inp, attempts)
        valid = call in self.admissible(inp, attempts)
        if valid and default is not None and call != default:
            valid, _ = self.override_allowed(inp, attempts, call, default)
        if valid:
            self.efficiency_events.append({"action": "skip_stop_confirmation", "call": call,
                                           "reason": "admissible_delivery_satisfies_final_rule"})
        return not valid

    def stop_confirmation_text(self):
        if self.efficiency_policy["stop_confirmation"] == "always":
            return super().stop_confirmation_text()
        return " A stop with a valid admissible deliver satisfying the final rule needs no second confirmation."

    def _table(self, inp, attempts):
        if self.efficiency_policy["compact_prompts"] != "1" or not attempts:
            return super()._table(inp, attempts)
        # Column names occur once. Keep every row and field; no history or decision evidence is dropped.
        rows = self._rows(inp, attempts)
        columns = list(dict.fromkeys(k for row in rows for k in row))
        return json.dumps({"columns": columns, "rows": [[row.get(k) for k in columns] for row in rows]},
                          separators=(",", ":"), default=str)

    def plan_round(self, inp, attempts, k_max: int, remaining: int) -> list[dict]:
        """The agent's round; in the cohort mode (`cohort_levers` set) none: the shared loop has run the call at
        the defaults, its admission and, after a RUN, the one reference call -- the image's own corrections (the
        diameter rule, the colour rule) with the group's consensus levers (`reference_moves`) --, so the search
        stops here ("cohort_applied") and `choose` delivers the reference default unasked."""
        reference = self.unbeatable_reference(inp, attempts)
        if self.cohort_levers is None and reference is not None:
            self.stop_reason = "zero_problem_reference"
            self.efficiency_events.append({"action": "stop_search", "call": reference,
                                           "reason": "strictly_fewer_than_zero_problems_is_impossible"})
            return []
        if self.cohort_levers is None:
            return super().plan_round(inp, attempts, k_max, remaining)
        self.stop_reason = "cohort_applied"
        return []

    def choose(self, inp, attempts, admissible: list[str]) -> tuple[Optional[str], str]:
        """The base's final question; in the cohort mode the reference default (the cohort's call when it ran and
        the guard admits it, else the call at the defaults), unasked."""
        if self.balanced_vrr:
            if len(admissible)==1 and self.stop_reason not in {'execution_error','invalid_search_plan'}:
                call=admissible[0]
                self.choice={'answers':[]}
                self.efficiency_events.append(dict(action='skip_final_selection',call=call,reason='only_admissible_call'))
                self.log('rule_selection',call=call,text='the only admissible finished call; no ranking decision')
                return call,'only_admissible_call'
            # Cohort and local candidates receive the same explicit comparison;
            # no unasked reference delivery bypasses this choice.
            return AgenticVerifier.choose(self, inp, attempts, admissible)
        default = self.reference_default(inp, attempts)
        reference = self.unbeatable_reference(inp, attempts)
        if self.cohort_levers is None and reference in admissible:
            self.efficiency_events.append({"action": "skip_final_selection", "call": reference,
                                           "reason": "zero_problem_reference"})
            return reference, "zero_problem_reference"
        if self.cohort_levers is None or default not in admissible:
            return super().choose(inp, attempts, admissible)
        return default, "cohort_applied"

    def _default_is_default_call(self) -> bool:
        return self.default_call is not None and self.default_call not in self.reference_calls

    def reference_default_text(self) -> str:
        """The base rule's text, the default named as the call at the defaults when it is the default call."""
        text = super().reference_default_text()
        if self._default_is_default_call():
            text = text.replace("(the system's reference call)", "(the call at the defaults)", 1)
        return text

    def finalize_search(self, state, selection):
        selection = super().finalize_search(state, selection)
        selection.details["efficiency_events"] = list(self.efficiency_events)
        return selection

    def select(self, state, stop_reason: str) -> WaveSelection:
        """The base selection; when the call at the defaults is the default, the details record it as
        `reference_default` as the base does for a reference call (the call, whether the delivery overrode it,
        why)."""
        selection = super().select(state, stop_reason)
        details = selection.details
        if self._default_is_default_call() and "final_answers" in details and "reference_default" not in details:
            call = details.get("call")
            details["reference_default"] = {"call": self.default_call,
                                            "overridden": call != self.default_call,
                                            "reason": (self.choice.get("override") or {}).get("reason")}
        return selection

    def override_allowed(self, inp, attempts, call: str, default: str) -> tuple[bool, str]:
        """An override of the reference default needs strictly fewer measured problems (`call_problems`) than the
        default AND the agent's assessment of the call better (a lower severity) than its assessment of the default
        (a call not assessed is not better); the reason lists both calls' problems and assessments. (The NeurIPS22
        CellSeg replay, `runners/replay_shape_delivery.py`: the problem count alone overrode 15 times, 12 for a worse
        AP50.)"""
        if self.balanced_vrr:
            from ..balanced_policy import balanced_override
            if call not in attempts or self.guard(inp, attempts[call][2]):
                return False,'candidate fails the native artifact guard'
            records = self.choice.get('answers') or []
            report = records[-1] if records and records[-1].get('call') == call else (self.plans[-1] if self.plans else {})
            return balanced_override(call,default,attempts[call][2].metrics or {},
                                     attempts[default][2].metrics or {},self.assessments,report.get('comparison'))
        mine, theirs = self.call_problems(attempts, call), self.call_problems(attempts, default)
        fewer = len(mine) < len(theirs)
        a, b = ((self.assessments.get(c) or {}).get("severity") for c in (call, default))
        better = a is not None and b is not None and _SEVERITY_RANK[a] < _SEVERITY_RANK[b]
        return fewer and better, (
            f"{call} has {len(mine)} measured problem(s) ({'; '.join(mine) or 'none'}) and {default} has "
            f"{len(theirs)} ({'; '.join(theirs) or 'none'}): {'' if fewer else 'not '}strictly fewer; your "
            f"assessment of {call} is {a or 'none'} and of {default} {b or 'none'}: {'' if better else 'not '}better")

    def override_text(self, default: str) -> str:
        if self.balanced_vrr:
            return (f'has a high-confidence explicit visual comparison against {default}, cites at least two observations, '
                    'has no worse assessed defect severity, and visually explains any >25% count/coverage loss. '
                    'Equal problem counts or equal severity alone do not disqualify it. Never maximize foreground or count blindly')
        return (f"has strictly fewer measured_problems than {default} (the calls table) and your assessment of it\n"
                f"is better (a lower severity) than your assessment of {default}")

    # ---- the finished calls, as last bound (the final question's assessments) ---------------------------
    def _bind(self, state):
        inp, attempts = super()._bind(state)
        self._attempts = attempts
        return inp, attempts

    # ---- the agent's assessments: the progress signal and the ranking ---------------------------------
    def _to_assess(self) -> list[str]:
        """The finished calls of the round just run -- after the system's reference round (round 2), with the
        call at the defaults before it, which no question has shown: their overlays are attached for the agent
        to assess."""
        rounds = self.rounds if self.reference_calls and len(self.rounds) == 2 else self.rounds[-1:]
        skip = {f["call"] for f in self.failed}
        return [c for c in (self.label(i) for r in rounds for i in r) if c not in skip]

    def _assess(self, call: str, report, attempts) -> None:
        """The agent's assessment of a finished call's overlay (problem, severity, confidence), also recorded
        as that trial's `judge_severity` / `judge_problem`."""
        if call in attempts:
            self.assessments[call] = {"problem": report.problem, "severity": report.severity,
                                      "confidence": report.confidence}
            vres = attempts[call][2]
            vres.judge_severity, vres.judge_problem = report.severity, report.problem

    def record_plan(self, plan, label: str, attempts) -> None:
        """Also records the agent's assessment of each finished call's overlay (`_assess`) and the calls whose
        overlays the question showed to assess (the base records the plan's focus_levers for the narrowing)."""
        for a in plan.assessments:
            self._assess(a.call, a, attempts)
        super().record_plan(plan, label, attempts)
        self.plans[-1]["shown"] = self._to_assess()
        if self.gain_policy == 'evidence_v1':
            evidence = getattr(plan, 'continuation', None)
            self._continuation_evidence = evidence.model_dump() if evidence is not None else None

    def _continuation_decision(self, inp, attempts):
        """Use real finished waves and native scores, never the proposed moves."""
        from ..segmentation_adaptive import cost_stop
        history = [[self.label(i) for i in wave if self.label(i) in attempts] for wave in self.rounds]
        recent = [c for wave in history[-2:] for c in wave]
        before = [c for wave in history[:-2] for c in wave]
        gain = None
        if before and recent:
            try:
                old = [float(attempts[c][2].scalar_score) for c in before]
                new = [float(attempts[c][2].scalar_score) for c in recent]
                if all(math.isfinite(x) for x in old + new):
                    gain = max(0., max(new) - max(old))
            except (TypeError, ValueError, OverflowError):
                pass  # Missing/invalid measured progress cannot authorize a stop.
        return cost_stop(self._continuation_evidence, admissible=self.admissible(inp, attempts),
            assessments=self.assessments, stalled_rounds=self._stall(inp, attempts),
            recent_native_gain=gain, native_noise=self.native_gain_noise,
            recent_calls=recent, search_ready=self._search is None or self._search.plateau_ready(self))

    def _plateau(self, inp, attempts):
        if self.gain_policy == 'legacy':
            return super()._plateau(inp, attempts)
        decision = self._continuation_decision(inp, attempts)
        self.continuation_events.append(dict(round=len(self.rounds)+1, **decision))
        if not decision['stop']:
            return False
        self.stop_reason = 'evidence_plateau'
        self.log('evidence_plateau', text=json.dumps(decision))
        # The usual final question, admissibility and baseline-override guards
        # still select the delivered call. This is not a selection override.
        return True

    def _rank(self, call: str) -> int:
        """The agent's severity of `call` (none 0, minor 1, major 2); 1 when not assessed yet."""
        return _SEVERITY_RANK.get((self.assessments.get(call) or {}).get("severity"), 1)

    def _stall(self, inp, attempts) -> int:
        """Rounds in a row without progress: the finished rounds (a round of failed calls included) after
        the last round that improved the best assessed severity (the agent's assessments, the admission's
        answer included)."""
        best, stall = None, 0
        for indices in self.rounds:
            ranks = [_SEVERITY_RANK[self.assessments[self.label(i)]["severity"]] for i in indices
                     if self.label(i) in self.assessments]
            top = min(ranks, default=None)
            if top is not None and (best is None or top < best):
                best, stall = top, 0
            else:
                stall += 1
        return stall

    def fallback_choice(self, inp, attempts, pool=None) -> Optional[str]:
        """The call in `pool` (default: the admissible calls) with the best (agent assessment, QC score);
        ties: the earlier call. None when the pool is empty. (The final choice's rule delivers the reference
        default instead when there is one: `base.AgenticVerifier.choose`.)"""
        pool = self.admissible(inp, attempts) if pool is None else list(pool)
        if not pool:
            return None
        return min(pool, key=lambda c: (self._rank(c), -attempts[c][2].scalar_score, self._labels.get(c, 0)))

    def least_bad(self, inp, attempts) -> Optional[str]:
        """No admissible call: the finished call with the best (assessment, QC score) is delivered."""
        return self.fallback_choice(inp, attempts, attempts)

    def fallback_moves(self, inp, attempts, add) -> None:
        """For the finished call with the best (assessment, QC score): SegmentationVerifier's diagnosis, with
        flow_error set aside as in the admission (the flow_error note above `SCALE_FACTOR`), its directions turned
        into values (`PARAM_BOUNDS` steps), then arm A's diameter correction -- the diameter set to the call's
        mask_diameter_px when the scale check holds (`_scale_problem`), as arm A re-segments once at the measured
        diameter."""
        best = self.fallback_choice(inp, attempts, attempts)
        if best is None:
            return
        metrics, config = dict(attempts[best][2].metrics or {}), self._params(attempts, best)
        unflowed = {**metrics, "flow_error": 0.0}
        _, triggered, _ = self._linear.localize(unflowed, {"upstream": {}})
        delta, _, why = self._linear._diagnose(unflowed, {"parameters": config}) if triggered else ({}, None, "")
        changes = []
        for lever, way in (delta or {}).items():
            if lever not in self.lever_fields:
                continue
            if way in ("up", "down") and lever in PARAM_BOUNDS:
                start = config.get(lever)
                if start is None:
                    start = EFFECTIVE_DEFAULTS["niter"]                    # niter null = 200 (the tool's default)
                value = float(f"{apply_bounded({lever: float(start)}, {lever: way})[lever]:.6g}")
            else:
                value = way
            changes.append({"lever": lever, "value": json.dumps(value)})
        if changes:
            add(best, changes, why)
        scale = self._scale_problem(metrics, config.get("diameter"))
        if scale:
            add(best, [{"lever": "diameter", "value": json.dumps(metrics["mask_diameter_px"])}],
                f"{scale} -> diameter at the measured mask_diameter_px")

    def _ask(self, content, schema, label: str):
        """The base question; a final choice's assessments are recorded as the rounds' are (`_assess`), only for the
        calls not assessed before: an assessment made when a call's overlay was shown in a round is never replaced
        at the final question, where the choice is made. (The dev30 run, da67a815: re-assessed at the final
        question, 2 of the 5 overrides came from a call the rounds had assessed as major, as the default.)"""
        answer = super()._ask(content, schema, label)
        if isinstance(answer, SegChoice):
            for a in answer.assessments:
                if a.call not in self.assessments:
                    self._assess(a.call, a, self._attempts)
        return answer

    def extra_details(self, inp, attempts) -> dict:
        return {"vrr_policy": 'balanced_v1' if self.balanced_vrr else 'legacy',
                "efficiency_policy": dict(self.efficiency_policy),
                "efficiency_events": list(self.efficiency_events),
                "gain_policy": self.gain_policy, "native_gain_noise": self.native_gain_noise,
                "continuation_events": list(self.continuation_events),
                "assessments": dict(self.assessments),
                "focus_history": list(self.focus_history),
                "input": {k: v for k, v in self.input.items() if k != "status"}, "preflight": self.preflight,
                "color_evidence": self.color_evidence, "color_decision": self.color_decision,
                "color_populations": {self.label(i): v for i, v in sorted(self._colors.items())},
                "measured_problems": {c: self.call_problems(attempts, c) for c in attempts},
                "target": dict(zip(("source", "text"), self.target()))}

    # ---- prompts ----------------------------------------------------------------------------------------
    def target(self) -> tuple[str, str]:
        """(label, text) of the target: the text of the request's section headed "## Target" (any case; up to its
        next H1 / H2 heading or a "---" rule; the request's CRLF / CR line ends read as LF), verbatim, "Target
        (from the request)"; without one (or an empty one), the CARD's `DEFAULT_TARGET_SECTION`, "Target (default
        convention)"."""
        request = (self.request or "").replace("\r\n", "\n").replace("\r", "\n")
        heading = TARGET_HEADING.search(request)
        if heading:
            body = request[heading.end():]
            end = TARGET_END.search(body)
            text = body[:end.start() if end else None].strip()
            if text:
                return "Target (from the request)", text
        section = self._card_section(DEFAULT_TARGET_SECTION)
        return "Target (default convention)", section.split("\n", 1)[1].strip() if "\n" in section else ""

    def task_text(self) -> str:
        label, text = self.target()
        return f"the instance segmentation this image's analysis needs, as accurate as possible.\n{label}: {text}"

    def resources_text(self) -> str:
        return (f"Resources: at most {self.cap} call(s) run at the same time on this node (the segmentation "
                "model's replicas).")

    def final_rule_text(self) -> str:
        """The guard's rule; with a reference default (which qualifies), what a choice still not qualifying delivers is
        that default, and its rule follows (`reference_default_text`)."""
        head = ("Final output: at the end you choose one finished call as the delivered segmentation. It must have\n"
                "cells and a valid overlay. A choice that does not qualify is\n"
                "answered with the reason and asked once more; if it still does not qualify, ")
        if self.default_call:
            return f"{head}the reference default\n{self.default_call} is delivered.\n{self.reference_default_text()}"
        return (f"{head}the qualifying call\n"
                "with your best assessment (a call you have not assessed counts as minor), then the best QC score,\n"
                "is delivered. If no call qualifies, you are not asked: the finished call ranked the same way is\n"
                "delivered.")

    def plateau_text(self) -> str:
        return f"{self.patience} rounds in a row have not improved your best assessed severity"

    def evidence_text(self) -> str:
        text = ("Assess: for each call listed here, whose overlay is attached, return one assessment (call id, the\n"
                "main problem or none, severity none | minor | major, confidence): "
                f"{self._to_assess() or 'none this round'}.")
        if self.gain_policy == 'evidence_v1':
            text += ('\nContinuation: after assessing the overlays, name the best finished call, actionable faults, '
                'expected visual gain (none/minor/major/uncertain), confidence and concrete reason. '
                'Use uncertain if evidence is missing. Scalar score or size alerts alone cannot rule out visual gain. '
                'Keep searching while a specific useful refinement remains plausible; late candidates may improve '
                'missed/merged/fragmented instances despite a flat score. Stop only when remaining benefit does '
                'not justify another native trial and model question. Final delivery guards remain binding.')
        return text

    def readouts_text(self) -> str:
        return ("Input image facts: size, channels, dtype, image QC, and preflight = a vision model's read of the\n"
                "raw image made before any segmentation (the specimen context, the object classes it sees, its\n"
                "provisional guess of the target class, its confidence), or unavailable when it could not be made.\n"
                "QC readouts of a call (GT-free, measured on its mask; not ground truth): cell_count;\n"
                "foreground_coverage (fraction of pixels inside masks); mask_area_median and mask_area_CV;\n"
                "mask_diameter_px (the diameter of a circle with the call's median mask area, in the diameter\n"
                "lever's units, clamped to the lever's range);\n"
                "tiny_mask_fraction (masks < 0.3 x the median area) and large_mask_fraction (> 3 x); "
                "border_mask_fraction;\n"
                "mean_solidity (area / bounding box); missed_foreground_fraction, count_miss_fraction and\n"
                "missed_blob_size_ratio (textured foreground no mask covers, of the target or not -- unstained\n"
                "cells and background texture count too, so they are no measured problem; on a call whose colour\n"
                "filter acted, all three are measured on its masks before the filter, the masks it removed counting\n"
                "as covered); qc_score (0-1 from tiny / large / solidity / border).\n"
                "color_populations of a call (measured on its masks over the raw image): each mask's mean (blue - red)\n"
                "intensity, split in two at this image's Otsu threshold (cut_b_minus_r) -- the objects, and for the\n"
                "high (bluer / purpler) and the low (redder / pinker) population its masks, fraction and mean;\n"
                "null = no split (fewer than 4 masks, or no spread of the values). The color_class_filter lever\n"
                "makes this split on its own call's masks before filtering: keep_high_br keeps only the high\n"
                "population's masks, keep_low_br only the low population's; without a split every mask is kept. A\n"
                "call with the filter reports the split of the masks it kept.\n"
                "COLOUR POPULATIONS default, an image attached after the input image when the default call has a\n"
                f"colour minority, shows up to {COLOR_TILES} of its masks per color population (top row high, bottom "
                "row low), evenly spaced in\ntheir (blue - red) order, each outlined in a square of the raw image "
                "(contrast-stretched).")

    def _input_facts(self) -> dict:
        return {k: self.input.get(k) for k in ("image_shape", "channels", "dtype")} | {
            "image_qc": {k: round(v, 4) for k, v in (self.input.get("image_qc") or {}).items()},
            "preflight": self.preflight}

    def input_text(self, inp: VLRRResult) -> str:
        return f"Input image: {json.dumps(self._input_facts(), default=str)}"

    def calls_legend(self) -> str:
        return ("change = relative to the parent; non_default = the full configuration's\n"
                "non-default values; vs_parent = QC readout deltas; prediction = each stated expectation with the\n"
                "observed change from the parent; color_populations = the color split of its masks;\n"
                "measured_problems = its guard failures and the admission's checks on the call (the shape gates,\n"
                "the scale against the diameter it used, a colour minority of its masks; on a\n"
                "call with color_class_filter, instead of the colour minority: the population the filter keeps\n"
                f"at most {ccf._MINORITY_MAX_FRAC:.0%} and at least {ccf._MINORITY_MIN_COUNT} of its masks before "
                "the filter); your_assessment\n"
                "= what you said about its overlay; guard = whether it is an admissible final choice")

    def _prediction(self, parent: VLRRResult, vres: VLRRResult) -> list[dict]:
        """The shared predictions (`base.AgenticVerifier._prediction`) without the readouts never shown
        (`HIDDEN_READOUTS`)."""
        return [p for p in super()._prediction(parent, vres) if p["metric"] not in HIDDEN_READOUTS]

    def _rows(self, inp, attempts) -> list[dict]:
        """One row per finished call: its configuration, QC readouts and score, its color populations and measured
        problems (`call_problems`), the QC deltas and each stated prediction against its parent (none against the
        input image), the agent's assessment and the guard."""
        defaults, rows = self.defaults(), []
        for call in attempts:                    # in run order
            parent, params, vres = attempts[call]
            from_input = parent == INPUT
            fails = self.guard(inp, vres)
            rows.append({
                "call": call, "parent": parent, "change": vres.param_delta or "none (the tool at its defaults)",
                "non_default": {k: v for k, v in params.items() if defaults.get(k) != v},
                "diagnosis": vres.notes,
                "qc": {k: round(vres.metrics[k], 4) for k in QC_READOUTS if k in (vres.metrics or {})},
                "qc_score": round(vres.scalar_score, 4),
                "color_populations": self._colors.get(self._labels.get(call)),
                "measured_problems": self.call_problems(attempts, call),
                "vs_parent": "parent is the input image" if from_input else self._effects(attempts[parent][2], vres),
                "prediction": [] if from_input else self._prediction(attempts[parent][2], vres),
                "your_assessment": self.assessments.get(call),
                "guard": "admissible" if not fails else "; ".join(fails)})
        return rows

    def _input_image(self) -> bytes | None:
        if self._input_png is None and self.input.get("image"):
            self._input_png = _png(self.input["image"], PROMPT_IMAGE_PX)
        return self._input_png

    def _overlay(self, call: str, max_px: int) -> bytes | None:
        path = self._overlays.get(self._labels.get(call, -1))
        return _png(path, max_px) if path and os.path.exists(path) else None

    @staticmethod
    def _attach(pairs) -> list:
        """(label, image) pairs as prompt content, without the images that could not be read."""
        return [item for label, data in pairs if data for item in (label, data)]

    def _input_pairs(self) -> list:
        """The input image and, right after it when the call at the defaults has a colour minority, that call's
        colour evidence (`_color_evidence`)."""
        return [("INPUT IMAGE:", self._input_image())] + ([self._evidence] if self._evidence else [])

    def round_attachments(self, inp, attempts) -> list:
        """The input image (`_input_pairs`), the overlays of the calls just finished (to assess) and, for
        comparison, of the best earlier call the guard admits -- else of the best earlier call, labelled as not
        admissible."""
        shown = self._to_assess()
        earlier = [c for c in attempts if c not in shown]
        admissible = [c for c in self.admissible(inp, attempts) if c not in shown]
        compare = self.fallback_choice(inp, attempts, admissible or earlier) if earlier else None
        pairs = self._input_pairs()
        pairs += [(f"OVERLAY {c} (to assess):", self._overlay(c, PROMPT_IMAGE_PX)) for c in shown]
        if compare:
            which = "your best admissible call so far" if admissible else "your best call so far, not admissible"
            pairs.append((f"OVERLAY {compare} ({which}, for comparison):", self._overlay(compare, PROMPT_IMAGE_PX)))
        return self._attach(pairs)

    def choice_text(self) -> str:
        return "Choose the delivered segmentation of this image: one finished call."

    def admissible_text(self) -> str:
        return "cells, a valid overlay"

    def final_evidence_text(self, inp, attempts, admissible: list[str]) -> str:
        """The admissible calls not assessed yet: assessed first (`SegChoice.assessments`), their overlays attached
        with the others'."""
        todo = [c for c in admissible if c not in self.assessments]
        return ("Assess first: for each call listed here, whose overlay is attached, return one assessment (call\n"
                f"id, the main problem or none, severity none | minor | major, confidence): {todo or 'none'}. Then\n"
                "choose.")

    def final_attachments(self, inp, attempts, admissible: list[str]) -> list:
        """The input image (`_input_pairs`) and the overlay of each admissible call."""
        return self._attach(self._input_pairs()
                            + [(f"OVERLAY {c}:", self._overlay(c, FINAL_IMAGE_PX)) for c in admissible])

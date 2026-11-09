"""Mask-measured cell diameter (the reliable half of Recommend).

A pre-segmentation guess of the cell diameter from raw-image gradients fragments on
real phase contrast (it underestimated large cells ~4x). The cells are measured far
more reliably AFTER the first segmentation, from the median mask area: equivalent
diameter = 2*sqrt(area/pi). The benchmark runs one default-diameter pass, measures the
true size here, and re-segments only if the default was off by a large factor.
"""

from __future__ import annotations

import math
from typing import Literal

from pydantic import BaseModel

_MIN_D = 5.0
_MAX_D = 200.0


def diameter_from_mask_area(mask_area_median, default: float = 30.0) -> float:
    """Equivalent cell diameter (px) from the median mask AREA (px^2), clamped to a
    sane range. Falls back to ``default`` when no median is available (e.g. zero
    cells, or a metric dict that predates area measurement)."""
    try:
        area = float(mask_area_median)
    except (TypeError, ValueError):
        return float(default)
    if not math.isfinite(area) or area <= 0:
        return float(default)
    diameter = 2.0 * math.sqrt(area / math.pi)
    return float(min(_MAX_D, max(_MIN_D, diameter)))


def diameter_correction_needed(measured: float, used: float, factor: float = 2.0) -> bool:
    """Whether to re-segment: the diameter actually used was off from the measured
    cell size by more than `factor`x (either direction). cpsam is scale-robust, so
    only a large mismatch is worth a rerun."""
    if not measured or not used:
        return False
    ratio = measured / used if used else 0.0
    return ratio >= factor or (ratio > 0 and ratio <= 1.0 / factor)


# --------------------------------------------------------------------------- #
# Pre-flight SSL training-param calibration (learning_rate + epoch_budget).
# The Recommend half: seed params from the crop profile + sample count, and derive the
# calibration subset from the data. Bounds MIRROR the Cell_State_Analyzer_Tool CARD.md
# (the recommend follows the card); the bounded few-round loop confirms against the
# label-free training_health_score. These are soft priors, not benchmark magic numbers.
# --------------------------------------------------------------------------- #
LR_DEFAULT = 3e-5
LR_BOUNDS = (1e-5, 1e-4)
EPOCH_BOUNDS = (3, 30)
_CALIB_MIN_SOURCES_PER_CLASS = 3   # >= this distinct source images per class for LOSO CV
_CALIB_MIN_CELLS_PER_CLASS = 8     # and >= this many cells for a stable health signal


def profile_crops(crops, sample: int = 64, seed: int = 0) -> dict:
    """Label-free crop-feature profile (the analyzer's analogue of diameter_from_mask_area):
    sample count and sampled dynamic-range / bad-crop fraction via the unified crop QC. Seeds
    the lr/epoch recommend (mainly via N) and is recorded for the audit."""
    import numpy as np
    from .verifiers.crop import bare_crop_qc, is_crop_valid

    paths = list(crops)
    n = len(paths)
    profile = {"n_crops": n, "mean_dynamic_range": None,
               "bad_crop_fraction": None, "low_contrast": None}
    if not n:
        return profile
    rng = np.random.default_rng(seed)
    idx = rng.choice(n, size=min(sample, n), replace=False)
    dyn, bad = [], 0
    for i in idx:
        q = bare_crop_qc(paths[int(i)])
        ok, _ = is_crop_valid(q)
        bad += 0 if ok else 1
        if isinstance(q.get("dynamic_range"), (int, float)):
            dyn.append(float(q["dynamic_range"]))
    if dyn:
        profile["mean_dynamic_range"] = float(np.mean(dyn))
        profile["low_contrast"] = bool(np.mean(dyn) < 0.20)
    profile["bad_crop_fraction"] = float(bad / len(idx))
    return profile


def recommend_lr_epoch(profile: dict) -> dict:
    """Seed learning_rate + epoch budget from the crop profile, clamped to the card's safe
    bounds. Honest: crop appearance does not predict the best lr, so lr starts at the default
    and the bounded calibration loop confirms it; the profile sets the epoch budget from the
    full sample count N (the same N rule the analyzer's resolve_training_config uses)."""
    n = max(1, int(profile.get("n_crops", 1)))
    epoch = round(8 + 6 * math.sqrt(128 / n))
    epoch = int(min(EPOCH_BOUNDS[1], max(EPOCH_BOUNDS[0], epoch)))
    lr = float(min(LR_BOUNDS[1], max(LR_BOUNDS[0], LR_DEFAULT)))
    return {"learning_rate": lr, "max_epochs": epoch,
            "lr_bounds": list(LR_BOUNDS), "epoch_bounds": list(EPOCH_BOUNDS)}


def recommend_calibration_subset(records) -> dict:
    """Derive the per-class calibration subset size FROM THE DATA, not a hardcoded constant:
    the smallest per-class size that meets the coverage priors (>= min cells, and enough for
    leave-one-source-out), capped at the smallest class's availability. `records` are dicts
    with 'cell_line' and 'source_image'. The agentic reasoning here IS the sample-size /
    coverage adequacy decision."""
    from collections import Counter, defaultdict

    cells = Counter(r["cell_line"] for r in records)
    sources = defaultdict(set)
    for r in records:
        sources[r["cell_line"]].add(r["source_image"])
    if not cells:
        return {"per_class": 0, "adequate": False, "reason": "no records"}
    min_class_cells = int(min(cells.values()))
    target = max(_CALIB_MIN_CELLS_PER_CLASS, _CALIB_MIN_SOURCES_PER_CLASS)
    per_class = int(min(min_class_cells, target))
    adequate = (per_class >= _CALIB_MIN_CELLS_PER_CLASS
                and all(len(s) >= _CALIB_MIN_SOURCES_PER_CLASS for s in sources.values()))
    return {
        "per_class": per_class,
        "min_class_cells": min_class_cells,
        "n_classes": len(cells),
        "adequate": bool(adequate),
        "priors": {"min_cells_per_class": _CALIB_MIN_CELLS_PER_CLASS,
                   "min_sources_per_class": _CALIB_MIN_SOURCES_PER_CLASS},
        "reason": "smallest per-class size meeting coverage priors, capped at availability",
    }


# Fallback centering cut when there are too few crops to estimate a distribution -- matches the
# crop verifier's _CENTROID_OFFSET_MAX. The agentic path prefers the data-derived bound below.
_OFF_CENTER_FALLBACK = 0.30
_OFF_CENTER_MIN_N = 20             # need at least this many offsets to trust a data-derived bound


def recommend_off_center_bound(offsets) -> dict:
    """Data-derived off-center cut: remove only the EXTREME crops relative to THIS dataset's
    offset distribution, instead of a hardcoded "centered" threshold. Bare-crop centroids are
    estimates (noisy for reticular/peripheral stains), so a fixed cut over-rejects; the robust
    Tukey upper fence (Q3 + 1.5*IQR) flags statistical outliers only. The agentic reasoning IS
    this bound. Falls back to the fixed cut when N is too small to estimate a spread.

    `offsets`: per-crop centroid_offset values (fraction of half the short side)."""
    import numpy as np

    vals = np.asarray([o for o in offsets if isinstance(o, (int, float))
                       and np.isfinite(o)], dtype=float)
    n = int(vals.size)
    if n < _OFF_CENTER_MIN_N:
        return {"bound": _OFF_CENTER_FALLBACK, "method": "fallback_fixed", "n": n,
                "reason": f"only {n} offsets (< {_OFF_CENTER_MIN_N}) -> fixed fallback cut"}
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = float(q3 - q1)
    bound = float(q3 + 1.5 * iqr)
    frac = float((vals > bound).mean())
    return {
        "bound": bound, "method": "tukey_upper_fence", "n": n,
        "q1": float(q1), "q3": float(q3), "iqr": iqr,
        "off_center_fraction_at_bound": round(frac, 4),
        "reason": "Q3 + 1.5*IQR of the observed centroid offsets -- removes extreme outliers "
                  "only, not a fixed centering cut (bare-crop centroid is an estimate)",
    }


_DOMINANCE_FALLBACK = 0.5   # small-N fallback only, mirroring _OFF_CENTER_FALLBACK's role


def recommend_dominance_bound(dominances) -> dict:
    """Data-derived LOWER bound for target_dominance (neighbour crowding), mirroring
    `recommend_off_center_bound`'s Tukey fence exactly but on the other tail: low dominance is
    the bad direction here (the target cell is crowded out by neighbours), so this is the LOWER
    fence (Q1 - 1.5*IQR) instead of the upper one. Falls back to the fixed cut when N is too small
    to estimate a spread.

    Comparing the dataset's MEAN dominance to its own lower fence is not circular: for any
    roughly-symmetric unimodal distribution the mean essentially never falls beyond its own
    1.5*IQR fence; if it does, that itself is evidence of a real low-dominance cluster dragging
    the mean down, not just a couple of stray crops -- the same interpretation Tukey fences carry
    everywhere else in this codebase.

    `dominances`: per-crop target_dominance values (fraction of foreground occupied by the
    largest connected component)."""
    import numpy as np

    vals = np.asarray([d for d in dominances if isinstance(d, (int, float))
                       and np.isfinite(d)], dtype=float)
    n = int(vals.size)
    if n < _OFF_CENTER_MIN_N:
        return {"bound": _DOMINANCE_FALLBACK, "method": "fallback_fixed", "n": n,
                "reason": f"only {n} dominances (< {_OFF_CENTER_MIN_N}) -> fixed fallback cut"}
    q1, q3 = np.percentile(vals, [25, 75])
    iqr = float(q3 - q1)
    bound = float(q1 - 1.5 * iqr)
    frac = float((vals < bound).mean())
    return {
        "bound": bound, "method": "tukey_lower_fence", "n": n,
        "q1": float(q1), "q3": float(q3), "iqr": iqr,
        "dominance_fraction_below_bound": round(frac, 4),
        "reason": "Q1 - 1.5*IQR of the observed target_dominance values -- the mean falling below "
                  "this fence indicates a real low-dominance cluster, not just outlier noise",
    }


# --------------------------------------------------------------------------- #
# Up-front (Recommend #1) initial analyzer config (task 008 P2). Reasons from the numeric
# `profile_dataset` report + the sampled-crop vision profile (`crop_vision.crop_vision_profile`)
# via the SAME CARD-driven mechanism the mid-loop analyzer retune and segmentation already use
# (`_recommend_from_card`, `llm_recommend.py`). Deterministic fallback = the current
# `diagnostic_planner._rule_plan`'s field rules copied VERBATIM (the byte-parity gate) --
# `_rule_plan` itself is left untouched (P3 retires it). Copied rather than imported because
# CORE must not import the benchmark.
# --------------------------------------------------------------------------- #

def _source_reliable(report: dict) -> bool:
    """Copied VERBATIM from the HPA benchmark's `diagnostic_planner.py::_source_reliable` (that benchmark,
    benchmarks/representation_HPA, was removed from the repository; core never imported it). field_aware needs a
    trustworthy per-crop SOURCE id. HPA's source
    is only a filename-regex guess and fields are mostly singletons -> unreliable. Reliable only
    with real field/well metadata AND multi-cell fields (median crops/source >= 2)."""
    scale = report.get("scale_and_balance", {})
    meta = scale.get("metadata_availability", {})
    has_meta = any(meta.get(k) not in (None, "unavailable") for k in ("well_id", "plate_id"))
    mcps = scale.get("median_crops_per_source")
    multi_cell = isinstance(mcps, (int, float)) and mcps >= 2
    return bool(has_meta and multi_cell)


def _rule_initial_params(report: dict) -> dict:
    """The field->config rules from `diagnostic_planner._rule_plan`, copied VERBATIM (the
    byte-parity gate), returning the 4 up-front categorical knobs this function's deterministic
    path decides. `learning_rate`/`max_epochs` come from `recommend_lr_epoch` instead (a
    label-free, data-derived prior, not part of `_rule_plan`'s rules) -- see
    `recommend_initial_analyzer_config`, which appends them."""
    mod = report.get("modality", {})
    scale = report.get("scale_and_balance", {})
    sep = report.get("separability_baseline", {})
    quality = report.get("crop_quality", {})
    leak = report.get("field_and_leakage", {})
    single = bool(mod.get("is_single_channel"))
    area_dom = (isinstance(sep.get("PC1_area_correlation"), (int, float))
                and abs(sep["PC1_area_correlation"]) > 0.7)
    dom_bound = (quality.get("dominance_bound") or {}).get("bound")   # data-derived Tukey fence
    dominance_low = (isinstance(quality.get("mean_target_dominance"), (int, float))
                     and isinstance(dom_bound, (int, float))
                     and quality["mean_target_dominance"] < dom_bound)   # neighbours crowd out the target
    src_ok = _source_reliable(report)
    source_clusters = (isinstance(leak.get("silhouette_by_source"), (int, float))
                       and leak["silhouette_by_source"] > 0)
    field_aware = bool(src_ok and source_clusters)
    return {
        "augment_color": (not single),               # single-channel -> off
        "field_aware": field_aware,
        # Adaptation depth: how much of the backbone adapts. Deliberately NOT thresholded here.
        # The obvious deterministic rule ("few crops per class -> freeze") thresholds the wrong
        # quantity: crops from one source image are not independent samples (measured on HPA:
        # 2421 crops over 1234 source images, ~2 per image), so a crops-per-class cutoff can be
        # an order of magnitude off in effective sample size -- and any cutoff is a constant
        # asserted about data it has never seen.
        # The evidence for this decision is already produced pre-flight and costs nothing extra:
        # `baseline_embedding` (a zero-shot forward pass -- itself one of the three candidate
        # actions) and `separability_baseline` measure how good the PRETRAINED features already
        # are on THIS input, which is what "domain gap" means operationally. The agent reads
        # that and chooses; this line is only the no-engine floor, and it takes the one value
        # that is always admissible (see `admissible_adaptation_depths`: >3 channels leaves the
        # patch embedding partly randomly initialised, so only `trained` can repair it).
        "adaptation_depth": "trained",
    }


class InitialConfigVerdict(BaseModel):
    """What the vision/text model REPORTS about the up-front crop-vision profile + numeric
    report + the ONE up-front knob it proposes overriding from the analyzer CARD. Field-for-field
    identical to `SegOverlayVerdict`/`RepresentationVerdict` (`llm_recommend.py`) -- required, not
    stylistic: `_recommend_from_card`'s notes formatting reads `rec.param/direction/image_usable/
    confidence/cell_size/cell_density/problem` unconditionally, and `llm_recommend.py` is not in
    P2's allowed files, so the verdict schema must supply every field that call site expects."""
    specimen: str
    cell_size: Literal["tiny", "small", "medium", "large"]
    cell_density: Literal["sparse", "moderate", "crowded", "confluent"]
    image_usable: Literal["yes", "marginal", "no"]
    problem: str
    severity: Literal["none", "minor", "major"]   # only "major" warrants overriding a knob
    param: Literal["augment_color", "field_aware", "adaptation_depth", "none"]
    direction: Literal["zero_shot", "frozen_head", "trained", "on", "off", "none"]
    confidence: Literal["high", "medium", "low"]
    reasoning: str


# Card physics for the up-front knobs (mirrors `_PARAM_EFFECT`/`_REP_PARAM_EFFECT` in
# `llm_recommend.py`): the deterministic metric a (param, direction) move should shift. Only
# `input_strategy` has a documented metric effect today (the others are label-free training-setup
# choices with no single deterministic metric to reconcile against); still valid moves, just with
# an empty (but non-None, i.e. in-vocabulary) expected_effect.
_INITIAL_PARAM_EFFECT: dict[tuple[str, str], dict[str, str]] = {
    ("augment_color", "on"): {}, ("augment_color", "off"): {},
    ("field_aware", "on"): {}, ("field_aware", "off"): {},
    ("adaptation_depth", "zero_shot"): {}, ("adaptation_depth", "frozen_head"): {},
    ("adaptation_depth", "trained"): {},
}
# (param, direction) -> the actual value assigned into initial_params (categorical params pass
# their direction through as-is; the on/off knobs map to booleans).
_INITIAL_PARAM_KEY = {"augment_color": "augment_color",
                      "field_aware": "field_aware", "adaptation_depth": "adaptation_depth"}
_INITIAL_PARAM_VALUE = {
    ("augment_color", "on"): True, ("augment_color", "off"): False,
    ("field_aware", "on"): True, ("field_aware", "off"): False,
    ("adaptation_depth", "zero_shot"): "zero_shot",
    ("adaptation_depth", "frozen_head"): "frozen_head",
    ("adaptation_depth", "trained"): "trained",
}


def _initial_config_to_recommendation(rec: InitialConfigVerdict):
    """Turn the LLM PROPOSAL into (action, param_delta, expected_effect); mirrors
    `_verdict_to_recommendation`'s accept-by-default shape (`llm_recommend.py`). Only a 'major'
    problem with an in-vocabulary knob overrides ONE up-front param; the other 3 (plus
    learning_rate/max_epochs) keep the deterministic value."""
    if rec.severity != "major" or rec.param == "none" or rec.direction == "none":
        return "accept", None, {}
    ee = _INITIAL_PARAM_EFFECT.get((rec.param, rec.direction))
    value = _INITIAL_PARAM_VALUE.get((rec.param, rec.direction))
    if ee is None or value is None:          # out-of-vocabulary / nonsensical direction
        return "accept", None, {}
    return "retune_here", {_INITIAL_PARAM_KEY[rec.param]: value}, dict(ee)


def _build_initial_config_prompt(report: dict, crop_vision: dict, card: str | None) -> str:
    import json
    card_section = (f"\nTOOL CARD -- how this analyzer's up-front config knobs work; choose "
                    f"your fix from here:\n{card.strip()}\n" if card else "")
    return (
        "You are recommending the INITIAL training config for a self-supervised cell-"
        "representation analyzer, before any training. A deterministic rule-based config has "
        "already been proposed from the numeric report; you may override ONE of its up-front "
        "knobs (augment_color, field_aware, adaptation_depth) if the evidence "
        "below clearly calls for it.\n\n"
        "STEP 1 -- read the sampled-crop vision profile and the numeric report.\n\n"
        "STEP 2 -- find the ONE main problem with the deterministic proposal, if any.\n\n"
        "STEP 3 -- rate severity, defaulting to ACCEPT (severity='none' or 'minor', param='none'"
        "). Only a CLEARLY major problem warrants overriding a knob.\n\n"
        "STEP 4 -- if severity is 'major', use the TOOL CARD to choose the SINGLE up-front knob "
        "and direction whose documented effect best fixes the problem; otherwise param='none', "
        "direction='none'.\n"
        f"{card_section}\n"
        "Report: specimen; cell_size; cell_density; image_usable; problem (one short phrase or "
        "'none'); severity; param; direction; confidence; and a one-sentence reasoning.\n\n"
        f"Crop-vision profile: {json.dumps(crop_vision, default=str)}\n\n"
        "Numeric report (scale_and_balance / modality / crop_quality / separability_baseline / "
        f"field_and_leakage): {json.dumps({k: report.get(k) for k in ('scale_and_balance', 'modality', 'crop_quality', 'separability_baseline', 'field_and_leakage')}, default=str)}"
    )


def recommend_initial_analyzer_config(report: dict, crop_vision: dict, *, engine=None,
                                      card: str | None = None) -> dict:
    """The up-front #1 recommend (see BRIEF's "two recommend moments"): `initial_params`
    {augment_color, field_aware, adaptation_depth, learning_rate, max_epochs} for the
    analyzer. Deterministic fallback (`engine is None`) = `_rule_initial_params`, i.e.
    `diagnostic_planner._rule_plan`'s field rules VERBATIM (the byte-parity gate) --
    `learning_rate`/`max_epochs` are ALWAYS from `recommend_lr_epoch` (a label-free, data-derived
    prior, not vision-judged), appended on top of whichever of the 4 categorical knobs the
    deterministic rules or the LLM decide. Reuses `_recommend_from_card`'s shared control flow
    (CARD + evidence -> structured verdict -> clamp; a failed vision call raises `ExecutionError`).
    """
    from .llm_recommend import _recommend_from_card
    from shape.utils.schemas import VLRRResult

    base = _rule_initial_params(report)
    n_crops = (report.get("scale_and_balance") or {}).get("total_crops", 0)
    lr_epoch = recommend_lr_epoch({"n_crops": n_crops})
    deterministic_cfg = {**base, "learning_rate": lr_epoch["learning_rate"],
                         "max_epochs": lr_epoch["max_epochs"]}
    if engine is None:
        return deterministic_cfg

    det = VLRRResult("PASS", 1.0, {}, action="accept")
    prompt = _build_initial_config_prompt(report, crop_vision, card)
    result = _recommend_from_card(
        engine, evidence_images=[], prompt=prompt, schema=InitialConfigVerdict,
        to_recommendation=_initial_config_to_recommendation, deterministic=det)
    if result.action != "retune_here" or not result.param_delta:
        return deterministic_cfg
    return {**deterministic_cfg, **result.param_delta}


def initial_config_provenance(report: dict, initial_params: dict) -> dict:
    """Task 021 Stage 1b: audit trail for the up-front recommend -- per-knob whether
    `initial_params` (`recommend_initial_analyzer_config`'s return) came from the
    deterministic base (`_rule_initial_params`) or an LLM override, plus `detect_modality`'s
    `n_channels`/`is_single_channel` (already sitting in `report["modality"]`, unchanged by
    this task). Recomputes `_rule_initial_params(report)` (pure, cheap) to diff against the
    actual `initial_params` rather than threading new state through that function's return
    contract -- every existing caller keeps getting a plain VLRR-vocabulary param dict.
    Settles, at the next run's audit, WHY e.g. `augment_color` started on/off (RGB-replicated
    crops mis-read as multi-channel vs a genuine LLM override) -- see THREAD.md's P4 analysis.
    """
    base = _rule_initial_params(report)
    mod = report.get("modality") or {}
    return {
        "n_channels": mod.get("n_channels"),
        "is_single_channel": mod.get("is_single_channel"),
        "knobs": {
            knob: {"value": initial_params.get(knob),
                  "source": ("deterministic_base" if initial_params.get(knob) == base_value
                             else "llm_override")}
            for knob, base_value in base.items()
        },
    }


_QC_SAMPLE_START = 48      # first stratified draw
_QC_SAMPLE_STEP = 48       # stratified increment
_QC_SAMPLE_REL_TOL = 0.10  # stop when the off-center-bound estimate moves < this, relative


def recommend_qc_sample_size(records, qc_batch_fn=None, *, seed: int = 0) -> dict:
    """Agentic (data-derived) QC sample size: instead of a hardcoded budget, grow a STRATIFIED
    sample (round-robin by cell line) in increments and re-estimate the off-center outlier bound
    each step; STOP as soon as the estimate stabilizes (relative change < tol) AND every class is
    covered -- the smallest sample that is still representative (fast). Returns the decided size,
    the QC dicts already computed (so the caller does not re-QC), and the reasoning.

    `records`: dicts with 'cell_line' and 'path'. `qc_batch_fn(paths)->list[dict]` QCs a batch
    (default: sequential bare_crop_qc); the caller may pass a parallel version."""
    import numpy as np

    from .verifiers.crop import bare_crop_qc

    if qc_batch_fn is None:
        def qc_batch_fn(paths):
            return [bare_crop_qc(p) for p in paths]

    n = len(records)
    if n == 0:
        return {"sample_size": 0, "stabilized": False, "all_classes_covered": True,
                "qc": [], "sample": [], "reason": "no records"}

    # stratified round-robin order over shuffled per-class queues
    rng = np.random.default_rng(seed)
    by_class = {}
    for r in records:
        by_class.setdefault(r["cell_line"], []).append(r)
    for items in by_class.values():
        rng.shuffle(items)
    classes = list(by_class)
    rng.shuffle(classes)
    order, ci = [], 0
    while len(order) < n:
        q = by_class[classes[ci % len(classes)]]
        if q:
            order.append(q.pop())
        ci += 1

    qc, prev_bound, size = [], None, 0
    reason = "grew to full N without stabilizing"
    while size < n:
        nxt = min(n, size + (_QC_SAMPLE_START if size == 0 else _QC_SAMPLE_STEP))
        qc.extend(qc_batch_fn([r["path"] for r in order[size:nxt]]))
        size = nxt
        offs = [q.get("centroid_offset") for q in qc]
        bound = recommend_off_center_bound(offs)["bound"]
        covered = {r["cell_line"] for r in order[:size]} == set(classes)
        if (prev_bound is not None and covered
                and abs(bound - prev_bound) <= _QC_SAMPLE_REL_TOL * max(prev_bound, 1e-6)):
            reason = (f"off-center bound stabilized (|Δ|<{_QC_SAMPLE_REL_TOL:.0%}) with all "
                      f"{len(classes)} classes covered")
            break
        prev_bound = bound
    return {
        "sample_size": size, "stabilized": size < n,
        "all_classes_covered": {r["cell_line"] for r in order[:size]} == set(classes),
        "n_total": n, "qc": qc, "sample": order[:size], "reason": reason,
    }

"""The segmentation cohort: a set of images segmented by sample type, calibrated per group.

`solve()` segments a set of images through `run`: every image is grouped by its preflight read
(`specimen_context`, `modality`); a group of at least `MIN_GROUP` images is calibrated -- a random sample
(`calibration_size`) runs the full per-image agentic VRR, and the agent-tuned levers its delivered calls agree on
(`consensus_params`, CPJUMP1's rule) become the group's consensus --; every other image of the group runs in the
cohort mode of `SegmentationAgenticVerifier` (`cohort_levers`: the call at the defaults, the admission and one
reference call with the image's own measured corrections plus the consensus, no agent round), and an image whose
cohort result still has a problem by the admission standard (the guard and `measured_problems`) gets the full
per-image VRR from the consensus (`start_params`); only the flagged images are routed, however many (a group's
other images keep their cohort result). The images WITHOUT a confident preflight read are calibrated as one
`catch_all` class (sample -> consensus -> cohort), not each on the full per-image VRR; a group of fewer images
and an unstable consensus (the catch-all's included) still take the per-image VRR. The groups run at the same
time, at most `workers` images at once.
Every decision is recorded (`run`'s record). The CPJUMP1 harness's SHAPE arm calls `run` (and `preflight_read`,
the grouping read solve() uses). `route_outliers` (more than `MAX_OUTLIER_FRACTION` of a set flagged routes all of
it) was the CPJUMP1 harness's earlier routing; `run` does not use it.

The per-image VRR itself is unchanged; this module decides which images run it and from which parameters.
"""

from __future__ import annotations

import hashlib
import json
import logging
import math
import random
import threading
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from typing import Any, Callable, Iterable, Optional

import numpy as np

logger = logging.getLogger(__name__)

TOOL = "Cell_Segmenter_Tool"
TOOLS = (TOOL, "Organoid_Segmenter_Tool")
MIN_GROUP = 10                    # a smaller group is segmented image by image (no calibration)
MIN_CALIBRATION = 3
CALIBRATION_RATES = ((100, 0.05), (1000, 0.025), (None, 0.01))   # progressive: (band upper edge, rate)
SUPPORT = 0.75                    # a lever transfers when this share of the calibration VRRs changed it
MAX_RELATIVE_MAD = 0.25           # ... a numeric lever's values agreeing within this relative MAD
MAX_OUTLIER_FRACTION = 0.20       # CPJUMP1's routing (`route_outliers`): more flagged than this routes them all
MEASURED_LEVERS = ("diameter", "color_class_filter")   # set per image by the measured corrections, never consensus
SOURCES = ("calibration", "cohort", "per_image_vrr")
_CATCH_ALL_KEY = ("__ungrouped__", "__catch_all__")   # the ungrouped (unreadable) images, calibrated as one class


def calibration_size(n: int) -> int:
    """Calibration images of a group of `n`: 0 below `MIN_GROUP`; else the progressive rates, each on the images
    within its band (5% of the first 100, 2.5% of the next 900, 1% beyond 1000), summed and rounded up, at
    least `MIN_CALIBRATION` (monotone in `n`; 1000 images -> 28)."""
    if n < MIN_GROUP:
        return 0
    total, lower = 0.0, 0
    for upper, rate in CALIBRATION_RATES:
        top = n if upper is None else min(n, upper)
        if top > lower:
            total += rate * (top - lower)
        if upper is None or n <= upper:
            break
        lower = upper
    return max(MIN_CALIBRATION, math.ceil(total - 1e-9))


def group_key(profile) -> Optional[tuple[str, str]]:
    """The image's group, (specimen_context, modality) of its preflight read; None (no group: the per-image VRR)
    for a missing or unconfident read or an unknown specimen."""
    if profile is None or getattr(profile, "confidence", "low") == "low":
        return None
    specimen = getattr(profile, "specimen_context", "unknown")
    if specimen == "unknown":
        return None
    return specimen, getattr(profile, "modality", "unknown")


def draw(ids: list[str], size: int, key) -> list[str]:
    """A random sample of `size` of `ids`, reproducible: seeded by the group key and the sorted ids."""
    seed = int(hashlib.sha256(json.dumps([list(key or ()), sorted(ids)]).encode()).hexdigest()[:16], 16)
    return sorted(random.Random(seed).sample(sorted(ids), min(size, len(ids))))


def _same(a: Any, b: Any) -> bool:
    if isinstance(a, (int, float)) and not isinstance(a, bool) and isinstance(b, (int, float)) and not isinstance(b, bool):
        return math.isclose(float(a), float(b), rel_tol=1e-9, abs_tol=1e-9)
    return a == b


def consensus_params(defaults: dict, records: Iterable[dict], allowed: Iterable[str], *, support: float = SUPPORT,
                     max_relative_mad: float = MAX_RELATIVE_MAD) -> tuple[dict, dict]:
    """The levers the calibration VRRs agree on (CPJUMP1's rule), over `allowed`: each record's `levers` are its
    delivered call's levers that differ from the defaults. A lever transfers when at least `support` of the records
    changed it -- numeric: their median, the relative MAD at most `max_relative_mad`; categorical: their mode, its
    agreement at least `support`. A lever changed often enough but dispersed (or without a modal value) makes the
    consensus unstable. Returns (the parameters: the defaults with the transferred levers, the report)."""
    records = list(records)
    params, decisions = dict(defaults), {}
    stable = True
    for lever in allowed:
        changed = [r.get("levers", {}).get(lever) for r in records if lever in (r.get("levers") or {})]
        fraction = len(changed) / max(1, len(records))
        decision = {"support": round(fraction, 4), "n_changed": len(changed), "n_calibration": len(records)}
        if fraction < support:
            decision["transfer"] = False
            decisions[lever] = decision
            continue
        if all(isinstance(v, (int, float)) and not isinstance(v, bool) for v in changed):
            values = np.asarray(changed, dtype=float)
            value = float(np.median(values))
            mad = float(np.median(np.abs(values - value)))
            relative = mad / max(abs(value), 1e-12)
            decision.update(value=value, relative_mad=round(relative, 6))
            if not np.isfinite(value) or relative > max_relative_mad:
                decision.update(transfer=False, reason="numeric calibration is dispersed")
                stable = False
            else:
                if isinstance(defaults.get(lever), int) and not isinstance(defaults.get(lever), bool):
                    value = int(round(value))
                params[lever] = value
                decision["transfer"] = True
        else:
            counts = Counter(repr(v) for v in changed)
            winner, count = counts.most_common(1)[0]
            value = next(v for v in changed if repr(v) == winner)
            agreement = count / len(changed)
            decision.update(value=value, agreement=round(agreement, 4))
            if agreement < support:
                decision.update(transfer=False, reason="categorical calibration has no consensus")
                stable = False
            else:
                params[lever] = value
                decision["transfer"] = True
        decisions[lever] = decision
    changed_params = {k: v for k, v in params.items() if not _same(v, defaults.get(k))}
    report = {"stable": stable, "support_threshold": support, "max_relative_mad": max_relative_mad,
              "selected_params": changed_params, "levers": decisions}
    return params, report


def route_outliers(problems: dict[str, list[str]], image_ids: list[str],
                   threshold: float = MAX_OUTLIER_FRACTION) -> tuple[set[str], dict]:
    """The images to run the per-image VRR: those with a problem; every image when more than `threshold` of them
    have one (a high failure rate is evidence of heterogeneity)."""
    flagged = {key for key, value in problems.items() if value}
    fraction = len(flagged) / max(1, len(image_ids))
    all_images = fraction > threshold
    return (set(image_ids) if all_images else flagged), {
        "initial_outliers": len(flagged), "initial_outlier_fraction": round(fraction, 6),
        "threshold": threshold, "route_all": all_images,
    }


def preflight_read(path: str, engine, image_id: Optional[str] = None):
    """The image's preflight read for its group (`image_context.read_image_context_profile`; its process cache
    makes the image's own VRR reuse it): None without an engine, or when the read fails (the image is then
    ungrouped); a fatal LLM error raises."""
    if engine is None:
        return None
    from shape.agent.verification.image_context import read_image_context_profile
    from shape.utils.image_processor import ImageProcessor
    try:
        return read_image_context_profile(ImageProcessor.load_image(path), engine=engine)
    except Exception as exc:  # noqa: BLE001 -- an image without a read is segmented on its own (ungrouped)
        from shape.engine.errors import is_fatal_llm_error
        if is_fatal_llm_error(exc):
            raise
        logger.warning(f"segmentation cohort: no preflight read of {image_id or path}: {exc}")
        return None


def _segmentation(resolutions) -> Any:
    return next((r for r in resolutions or [] if (getattr(r, "details", None) or {}).get("admission") is not None),
                (resolutions or [None])[-1])


def delivered_levers(resolutions, defaults: dict) -> dict:
    """The delivered call's levers that differ from the CARD `defaults` (a calibration record's `levers`)."""
    res = _segmentation(resolutions)
    params = dict(getattr(res, "params", None) or {})
    return {k: params[k] for k in defaults if k in params and not _same(params[k], defaults[k])}


def problems_of(resolutions) -> list[str]:
    """The admission standard on the delivered call: its guard failures and measured problems (the stage record's
    `measured_problems` of the delivered call); a stage that delivered nothing or conceded is a problem."""
    res = _segmentation(resolutions)
    if res is None or getattr(res, "result", None) is None:
        return ["no segmentation was delivered"]
    details = getattr(res, "details", None) or {}
    call = details.get("call")
    found = list((details.get("measured_problems") or {}).get(call) or [])
    if getattr(res, "status", "satisfied") != "satisfied":
        found.append(f"the stage {res.status} ({res.reason})")
    return found


def run(specs: list[dict], run_one: Callable[[dict, Optional[dict]], tuple], map_images: Callable,
        read_profile: Callable[[dict], Any], defaults: dict, levers: Iterable[str],
        workers: int = 1, *, tool_name: str = TOOL, calibration: Optional[str] = None,
        routing_policy: str = 'legacy', grouping_policy=None, read_features=None,
        triage=None, gain_policy: str = 'legacy', organoid_selective_review: bool = False) -> tuple[list[tuple], dict]:
    """Segment `specs` as a cohort. `run_one(spec, verifier_options)` is the per-image managed run (returns
    (resolutions, result, status, fault)); `map_images(specs, fn)` maps a function over images (the solver's
    `image_workers`); `read_profile(spec)` is the image's preflight read (or None); `defaults` the segmenter's
    CARD defaults and `levers` its CARD levers. The ungrouped images and every group run at the same time, at most
    `workers` images segmented at once across them (one after the other when 1). Returns the runs in the order of
    `specs` and the cohort record."""
    if organoid_selective_review and tool_name != "Organoid_Segmenter_Tool":
        raise ValueError("organoid selective review is only for the CO model")
    if tool_name not in TOOLS:
        raise ValueError(f"unsupported segmentation cohort tool: {tool_name}")
    if tool_name != TOOL and routing_policy != "legacy":
        raise ValueError("the stratified Cell routing policy is not a CO model policy")
    from .segmentation_policy import calibration_mode
    calibration = calibration_mode(calibration)
    if routing_policy != 'legacy':
        from . import segmentation_adaptive
        if routing_policy != segmentation_adaptive.POLICY or not callable(read_features):
            raise ValueError('unknown routing policy or missing native feature reader')
        return segmentation_adaptive.run(specs, run_one, map_images, read_features,
            defaults, levers, grouping=grouping_policy, triage=triage, workers=workers,
            calibration=calibration, gain_policy=gain_policy)
    if grouping_policy is not None or read_features is not None or triage is not None or gain_policy != 'legacy':
        raise ValueError('adaptive options require an explicit adaptive routing policy')
    ids = [s["image_id"] for s in specs]
    if len(set(ids)) != len(ids):
        raise ValueError("the segmentation cohort needs a unique image_id per image")
    by_id = dict(zip(ids, specs))
    profiles = dict(zip(ids, map_images(specs, read_profile)))
    keys = {i: group_key(profiles[i]) for i in ids}
    tuned = [lever for lever in levers if lever not in MEASURED_LEVERS]
    runs: dict[str, tuple] = {}
    source: dict[str, str] = {}
    record = {"tool": tool_name, "n_images": len(ids), "min_group": MIN_GROUP, "calibration_mode": calibration, "groups": [],
              "ungrouped": sorted(i for i in ids if keys[i] is None)}
    gate = threading.BoundedSemaphore(max(1, workers))

    def segment(group_ids: list[str], options: Optional[dict], label: str) -> None:
        def one(spec):
            with gate:                           # at most `workers` images at once, across the groups
                effective = options
                if organoid_selective_review:
                    effective = {**(options or {}), "organoid_selective_review": True,
                                 "organoid_representative": label == "calibration"}
                return run_one(spec, effective)
        for i, out in zip(group_ids, map_images([by_id[i] for i in group_ids], one)):
            runs[i], source[i] = out, label

    def calibrate(key, members: list[str], entry: dict) -> None:
        if len(members) < MIN_GROUP:
            entry["routing"] = "per_image_small_group"
            segment(members, None, "per_image_vrr")
            return
        sample = draw(members, entry["calibration_size"], key) if calibration == "full" else []
        entry["calibration_policy"] = calibration
        entry["calibration"] = sample
        if sample:
            segment(sample, None, "calibration")
        eligible = list(sample)
        if tool_name == "Organoid_Segmenter_Tool" and sample:
            def exclusions():
                excluded = {}
                for i in sample:
                    res = _segmentation(runs[i][0])
                    issues = problems_of(runs[i][0])
                    if organoid_selective_review:
                        assessment = (getattr(res, "details", None) or {}).get("selected_visual_assessment") or {}
                        if assessment.get("severity") not in ("none", "minor") or assessment.get("confidence") != "high":
                            issues += ["no confident whole-organoid visual calibration assessment"]
                    if (getattr(res,"details",None) or {}).get("negative_confirmed"):
                        issues = issues + ["confirmed negative has no positive target for calibration"]
                    if issues: excluded[i] = issues
                return excluded
            excluded = exclusions()
            if organoid_selective_review:
                entry["calibration_initial"] = list(sample)
                entry["calibration_replacements"] = []
                # Replace an unusable representative, at most three times. All sampled
                # results are reused below; neither GT nor extra runs of that image enter selection.
                for _ in range(MIN_CALIBRATION):
                    if len(sample) - len(excluded) >= MIN_CALIBRATION: break
                    pool = [i for i in members if i not in sample]
                    if not pool: break
                    replacement = draw(pool, 1, (*key, "replacement", str(len(sample))))
                    segment(replacement, None, "calibration")
                    sample.extend(replacement)
                    entry["calibration_replacements"].extend(replacement)
                    excluded = exclusions()
            eligible = [i for i in sample if i not in excluded]
            entry["calibration_excluded"] = excluded
            entry["calibration_eligible"] = eligible
            if len(eligible) < MIN_CALIBRATION:
                entry["routing"] = "per_image_insufficient_clean_calibration"
                entry["calibration_levers"] = []
                segment([i for i in members if i not in sample], None, "per_image_vrr")
                return
        calibrated = [{"image_id": i, "levers": delivered_levers(runs[i][0], defaults)} for i in eligible]
        entry["calibration_levers"] = calibrated
        _, consensus = consensus_params(defaults, calibrated, tuned)
        if calibration == "skip":
            consensus["reason"] = "calibration_disabled_by_policy"
        entry["consensus"] = consensus
        rest = [i for i in members if i not in sample]
        if not consensus["stable"]:
            entry["routing"] = "per_image_unstable_calibration"
            segment(rest, None, "per_image_vrr")
            return
        cohort_levers = dict(consensus["selected_params"])
        entry["cohort_levers"] = cohort_levers
        segment(rest, {"cohort_levers": cohort_levers}, "cohort")
        problems = {i: problems_of(runs[i][0]) for i in rest}
        entry["cohort_problems"] = {i: p for i, p in problems.items() if p}
        routed = sorted(entry["cohort_problems"])
        entry["routing"] = {"flagged": len(routed), "flagged_fraction": round(len(routed) / max(1, len(rest)), 6)}
        entry["routed"] = routed
        if routed:
            entry["fallback_from"] = {i: {
                "params": dict(getattr(_segmentation(runs[i][0]), "params", None) or {}),
                "problems": list(problems[i]),
                "mask_path": (runs[i][1] or {}).get("mask_path") or (runs[i][1] or {}).get("organoid_mask_path"),
                "details": dict(getattr(_segmentation(runs[i][0]), "details", None) or {}),
            } for i in routed}
            segment(routed, {"start_params": cohort_levers} if cohort_levers else None, "per_image_vrr")

    groups: dict[tuple, list[str]] = {}
    for i in ids:
        if keys[i] is not None:
            groups.setdefault(keys[i], []).append(i)
    # The ungrouped images (no confident preflight read) are calibrated as ONE catch-all class -- sample,
    # find the class consensus, cohort-apply the rest -- instead of each taking the full per-image VRR.
    # calibrate() still falls back to per-image when the catch-all consensus is unstable or the class is too
    # small to sample, so a heterogeneous outlier set is never forced onto a bad shared parameter.
    # (user 2026-10-27: avoid the per-image VLRR blow-up on unreadable images; keeps the cohort's token win.)
    units = []
    if record["ungrouped"]:
        catch = {"group": {"specimen_context": "ungrouped", "modality": "catch_all"}, "catch_all": True,
                 "images": record["ungrouped"], "calibration_size": calibration_size(len(record["ungrouped"]))}
        record["groups"].append(catch)
        units.append(lambda: calibrate(_CATCH_ALL_KEY, record["ungrouped"], catch))
    for key, members in sorted(groups.items()):
        entry = {"group": {"specimen_context": key[0], "modality": key[1]}, "images": members,
                 "calibration_size": calibration_size(len(members)) if calibration == "full" else 0}
        record["groups"].append(entry)
        units.append(lambda key=key, members=members, entry=entry: calibrate(key, members, entry))
    if workers > 1:
        with ThreadPoolExecutor(max_workers=min(max(1, workers), len(units))) as pool:
            for future in [pool.submit(unit) for unit in units]:
                future.result()                  # an error in any group propagates
    else:
        for unit in units:
            unit()
    record["parameter_source"] = {i: source[i] for i in ids}
    record["counts"] = dict(Counter(source[i] for i in ids))
    return [runs[i] for i in ids], record

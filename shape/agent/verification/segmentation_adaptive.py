"""Opt-in policies for stratified segmentation VRR; no dataset paths or targets.

Batch fields and native measurements are supplied by the caller. Statistical
size alerts are retained even when a high-confidence visual Verify accepts
heterogeneity. Missing evidence always falls back to individual VRR.
"""
from __future__ import annotations

from collections import defaultdict
from collections import Counter
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
import json
import math
import threading
from typing import Literal

import numpy as np
from pydantic import BaseModel, Field


POLICY = 'stratified_triage_v1'
SOFT_SIZE_ALERTS = frozenset(('tiny_mask_fraction', 'large_mask_fraction'))
REQUIRED_VIEWS = frozenset(('raw_overview', 'mask_overlay', 'size_extreme_zoom'))


@dataclass(frozen=True)
class GroupingPolicy:
    batch_fields: tuple[str, ...]
    min_group: int = 10
    max_depth: int = 2
    min_scale_ratio: float = math.sqrt(2)
    min_density_ratio: float = 2.

    def __post_init__(self):
        if self.min_group < 3 or not 0 <= self.max_depth <= 4:
            raise ValueError('invalid grouping support/depth')
        if any(not isinstance(f, str) or not f for f in self.batch_fields):
            raise ValueError('batch field names must be explicit nonempty strings')
        if len(set(self.batch_fields)) != len(self.batch_fields):
            raise ValueError('duplicate grouping fields')
        if not all(math.isfinite(v) and v > 1 for v in (self.min_scale_ratio, self.min_density_ratio)):
            raise ValueError('grouping separation ratios must exceed one')


def native_features(cell_count, median_area, image_shape) -> dict:
    """Scale/density from native masks, without another model or endpoint target."""
    if len(image_shape) < 2:
        raise ValueError('a two-dimensional image shape is required')
    count, area = float(cell_count), float(median_area)
    pixels = math.prod(image_shape[:2])
    if not all(math.isfinite(v) and v > 0 for v in (count, area, pixels)):
        raise ValueError('valid nonempty native mask measurements are required')
    return dict(scale=math.sqrt(4 * area / math.pi), density=count * 1e6 / pixels)


def stratify(specs: list[dict], features: dict[str, dict], policy: GroupingPolicy):
    """Deterministic supported splits inside caller-declared acquisition groups.

    A split is accepted only if both children retain minimum support and the
    10th-to-90th percentile scale/density separation is meaningful. Small leaves
    stay at their parent; they never fragment into many unsupported searches.
    Missing native measurements remain explicitly ungrouped.
    """
    ids = [s['image_id'] for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError('image IDs must be unique')
    batches = defaultdict(list)
    ungrouped = []
    for spec in sorted(specs, key=lambda s: s['image_id']):
        if any(spec.get(f) is None for f in policy.batch_fields):
            raise ValueError('missing declared batch metadata for ' + spec['image_id'])
        f = features.get(spec['image_id']) or {}
        if any(not isinstance(f.get(k), (int, float)) or not math.isfinite(f[k]) or f[k] <= 0
               for k in ('scale', 'density')):
            ungrouped.append(spec['image_id'])
            continue
        batches[tuple(str(spec[f]) for f in policy.batch_fields)].append(spec['image_id'])
    groups = []

    def split(members, batch, path, depth):
        choices = []
        if depth < policy.max_depth and len(members) >= 2 * policy.min_group:
            for axis, ratio in [('scale', policy.min_scale_ratio), ('density', policy.min_density_ratio)]:
                values = np.asarray([features[i][axis] for i in members], dtype=float)
                low, high = np.quantile(values, [.1, .9])
                if high / low < ratio:
                    continue
                threshold = float(np.median(values))
                left = [i for i in members if features[i][axis] <= threshold]
                right = [i for i in members if features[i][axis] > threshold]
                if min(len(left), len(right)) >= policy.min_group:
                    choices.append((math.log(high / low) / math.log(ratio), axis, threshold, left, right))
        if choices:
            _, axis, threshold, left, right = sorted(choices, key=lambda c: (-c[0], c[1]))[0]
            split(left, batch, path + [dict(axis=axis, threshold=threshold, side='low')], depth + 1)
            split(right, batch, path + [dict(axis=axis, threshold=threshold, side='high')], depth + 1)
        else:
            groups.append(dict(batch=dict(zip(policy.batch_fields, batch)), strata=path,
                               images=sorted(members), shared_calibration=len(members) >= policy.min_group))

    for batch, members in sorted(batches.items()):
        split(members, batch, [], 0)
    return dict(policy=POLICY, groups=groups, ungrouped=sorted(ungrouped))


class TriageVerdict(BaseModel):
    image_id: str
    action: Literal['full_vrr', 'accept_with_flags', 'abstain']
    confidence: Literal['high', 'medium', 'low']
    image_quality_sufficient: bool
    normal_heterogeneity: bool
    visible_faults: list[str] = Field(description='Actionable missed/merged/fragmented or spurious whole-cell errors')
    evidence_views: list[Literal['raw_overview', 'mask_overlay', 'size_extreme_zoom']]
    reason: str


class TriageReply(BaseModel):
    verdicts: list[TriageVerdict]


def only_size_alerts(problems):
    return bool(problems) and all(isinstance(p, str) and p.strip()
        and p.split()[0] in SOFT_SIZE_ALERTS for p in problems)


def native_guard_ok(resolution):
    result = getattr(resolution, 'result', None)
    if not isinstance(result, dict) or getattr(resolution, 'status', None) != 'satisfied':
        return False
    try:
        count = float(result.get('cell_count', 0))
        return (result.get('status') == 'success' and math.isfinite(count) and count > 0
                and bool(result.get('mask_path')))
    except (ValueError, TypeError):
        return False


def triage_decision(item: dict, verdict: dict | None) -> dict:
    """Keep native failures/unknown complaints out of the accept branch."""
    common = dict(image_id=item['image_id'], residual_problems=list(item.get('problems') or []),
                  rendered_evidence_views=list(item.get('evidence_views') or []))
    def full(reason):
        return dict(**common, route='full_vrr', reason=reason)
    if item.get('guard_ok') is not True:
        return full('native_delivery_guard_requires_individual_vrr')
    if not common['residual_problems']:
        return full('triage_received_no_recorded_problem')
    if not only_size_alerts(common['residual_problems']):
        return full('not_exclusively_statistical_size_alerts')
    if not REQUIRED_VIEWS <= set(item.get('evidence_views') or []):
        return full('incomplete_rendered_evidence')
    try:
        answer = TriageVerdict.model_validate(verdict)
    except (ValueError, TypeError):
        return full('missing_or_invalid_visual_verdict')
    if answer.image_id != item['image_id']:
        return full('visual_verdict_image_mismatch')
    if (answer.action != 'accept_with_flags' or answer.confidence != 'high'
            or not answer.image_quality_sufficient or not answer.normal_heterogeneity
            or answer.visible_faults or not REQUIRED_VIEWS <= set(answer.evidence_views)
            or not answer.reason.strip()):
        return full('actionable_or_uncertain_visual_evidence')
    return dict(**common, route='accept_with_flags', reason=answer.reason,
                verdict=answer.model_dump(), measured_problems_cleared=False)


def triage_batch_decisions(items: list[dict], reply: dict | None) -> list[dict]:
    """Partial replies abstain on missing IDs; duplicate/extra IDs reject the batch."""
    wanted = {item['image_id'] for item in items}
    if len(wanted) != len(items):
        raise ValueError('duplicate input image IDs')
    try:
        parsed = TriageReply.model_validate(reply)
        ids = [v.image_id for v in parsed.verdicts]
        if len(ids) != len(set(ids)) or set(ids) - wanted:
            raise ValueError('ambiguous or unknown visual verdict IDs')
        answers = {v.image_id: v.model_dump() for v in parsed.verdicts}
    except (ValueError, TypeError):
        answers = {}
    return [triage_decision(item, answers.get(item['image_id'])) for item in items]


class ContinuationEvidence(BaseModel):
    best_call: str
    actionable_faults: list[str]
    expected_gain: Literal['none', 'minor', 'major', 'uncertain']
    confidence: Literal['high', 'medium', 'low']
    reason: str


def cost_stop(evidence: dict | None, *, admissible: list[str], assessments: dict,
              stalled_rounds: int, recent_native_gain: float | None,
              native_noise: float, recent_calls: list[str], search_ready: bool) -> dict:
    """Stop only after both native and explicit visual evidence support no benefit.

    Positive/uncertain expected visual gain keeps exploration possible even if
    scalar proxies stall; late useful candidates are not cut by a shorter cap.
    Final delivery remains subject to the verifier's existing selection guards.
    """
    def keep(reason):
        return dict(stop=False, reason=reason)
    try:
        answer = ContinuationEvidence.model_validate(evidence)
    except (ValueError, TypeError):
        return keep('missing_or_invalid_continuation_evidence')
    if not search_ready or stalled_rounds < 2 or len(recent_calls) < 2:
        return keep('insufficient_search_history')
    if answer.best_call not in admissible:
        return keep('best_call_not_admissible')
    if answer.confidence != 'high' or answer.actionable_faults or answer.expected_gain != 'none' or not answer.reason.strip():
        return keep('actionable_or_uncertain_expected_gain')
    for call in set(recent_calls + [answer.best_call]):
        a = assessments.get(call) or {}
        if a.get('confidence') != 'high' or a.get('severity') not in ('none', 'minor'):
            return keep('unassessed_uncertain_or_major_visual_defect')
    if (recent_native_gain is None or not math.isfinite(recent_native_gain)
            or not math.isfinite(native_noise) or native_noise < 0):
        return keep('native_progress_not_measurable')
    if recent_native_gain > native_noise:
        return keep('native_gain_still_present')
    return dict(stop=True, call=answer.best_call, reason=answer.reason,
                residual_faults_accepted=True, evidence=answer.model_dump(),
                stalled_rounds=stalled_rounds, recent_native_gain=recent_native_gain,
                native_noise=native_noise)


def run(specs, run_one, map_images, read_features, defaults, levers, *, grouping,
        triage, workers=1, calibration='full', gain_policy='evidence_v1'):
    """The opt-in cohort VRR; default per-image verification/delivery stays intact.

    The feature reader supplies native first-pass measurements. The triage
    callable must return recorded core triage decisions, including their visual
    verdict and rendered-view declarations; naked ``accept`` routes are ignored.
    All unresolved/uncertain cases retain the normal individual VRR path.
    """
    from . import segmentation_cohort as legacy
    if not specs or workers < 1 or gain_policy != 'evidence_v1':
        raise ValueError('nonempty cohort, positive workers and evidence_v1 gain policy required')
    if not isinstance(grouping, GroupingPolicy) or not callable(triage):
        raise ValueError('explicit supported grouping and visual triage are required')
    ids = [s['image_id'] for s in specs]
    if len(ids) != len(set(ids)):
        raise ValueError('image IDs must be unique')
    by_id = dict(zip(ids, specs))
    features = dict(zip(ids, map_images(specs, read_features)))
    stratification = stratify(specs, features, grouping)
    runs, source = {}, {}
    gate = threading.BoundedSemaphore(workers)
    record = dict(tool=legacy.TOOL, policy=POLICY, n_images=len(ids), calibration_mode=calibration,
                  grouping=dict(batch_fields=list(grouping.batch_fields), min_group=grouping.min_group,
                                max_depth=grouping.max_depth, min_scale_ratio=grouping.min_scale_ratio,
                                min_density_ratio=grouping.min_density_ratio),
                  native_features=features, ungrouped=stratification['ungrouped'], groups=[])
    tuned = [lever for lever in levers if lever not in legacy.MEASURED_LEVERS]

    def segment(group_ids, options, label):
        options = {**(options or {}), 'gain_policy': gain_policy}
        def one(spec):
            with gate:
                return run_one(spec, options)
        for ident, result in zip(group_ids, map_images([by_id[i] for i in group_ids], one)):
            runs[ident], source[ident] = result, label

    def calibrate(group):
        members = group['images']
        entry = dict(group=dict(batch=group['batch'], strata=group['strata']), images=members)
        if not group['shared_calibration']:
            entry['routing'] = 'per_image_small_supported_group'
            segment(members, None, 'per_image_vrr')
            return entry
        key = (json.dumps(entry['group'], sort_keys=True),)
        size = legacy.calibration_size(len(members)) if calibration == 'full' else 0
        sample = legacy.draw(members, size, key) if size else []
        entry.update(calibration=sample, calibration_size=size)
        if sample:
            segment(sample, None, 'calibration')
        calibrated = [dict(image_id=i, levers=legacy.delivered_levers(runs[i][0], defaults)) for i in sample]
        _, consensus = legacy.consensus_params(defaults, calibrated, tuned)
        entry.update(calibration_levers=calibrated, consensus=consensus)
        rest = [i for i in members if i not in sample]
        if not consensus['stable']:
            entry['routing'] = 'per_image_unstable_calibration'
            segment(rest, None, 'per_image_vrr')
            return entry
        cohort_levers = dict(consensus['selected_params'])
        entry['cohort_levers'] = cohort_levers
        segment(rest, dict(cohort_levers=cohort_levers), 'cohort')
        problems = {i: p for i in rest if (p := legacy.problems_of(runs[i][0]))}
        entry['cohort_problems'] = problems
        requests = [dict(spec=by_id[i], resolutions=runs[i][0], record=runs[i][1],
                         problems=problems[i], group=entry['group']) for i in sorted(problems)
                    if only_size_alerts(problems[i]) and native_guard_ok(legacy._segmentation(runs[i][0]))]
        # A callback failure cannot make any image accepted. Fatal provider
        # errors still propagate rather than silently changing the protocol.
        try:
            decisions = triage(requests) if requests else []
        except Exception as exc:
            from shape.engine.errors import is_fatal_llm_error
            if is_fatal_llm_error(exc):
                raise
            entry['triage_error'] = dict(type=type(exc).__name__, message=str(exc))
            decisions = []
        if not isinstance(decisions, list) or any(not isinstance(d, dict) for d in decisions):
            entry['triage_error'] = dict(type='InvalidTriageReply', message='triage must return decision records')
            decisions = []
        mapped = {}
        for decision in decisions:
            ident = decision.get('image_id')
            if ident in mapped or ident not in problems:
                entry['triage_error'] = dict(type='InvalidTriageReply', message='duplicate or unknown image ID')
                mapped = {}
                break
            mapped[ident] = decision
        routed, outcomes = [], {}
        for ident in sorted(problems):
            decision = mapped.get(ident) or {}
            resolutions = runs[ident][0]
            resolution = legacy._segmentation(resolutions)
            guard_ok = native_guard_ok(resolution)
            checked = triage_decision(dict(image_id=ident, guard_ok=guard_ok, problems=problems[ident],
                evidence_views=decision.get('rendered_evidence_views') or []), decision.get('verdict'))
            if decision.get('receipt'):
                checked['receipt'] = decision['receipt']
            outcomes[ident] = checked
            if checked['route'] != 'accept_with_flags':
                routed.append(ident)
        entry.update(triage=outcomes, routed=routed, routing=dict(flagged=len(problems),
            flagged_fraction=len(problems)/max(1, len(rest)), individual_vrr=len(routed),
            accepted_with_recorded_alerts=len(problems)-len(routed)))
        if routed:
            segment(routed, dict(start_params=cohort_levers), 'per_image_vrr')
        return entry

    units = [lambda group=group: calibrate(group) for group in stratification['groups']]
    if stratification['ungrouped']:
        def ungrouped():
            segment(stratification['ungrouped'], None, 'per_image_vrr')
        units.append(ungrouped)
    from shape.utils.concurrency import ordered_map
    entries = ordered_map(lambda unit: unit(), units, min(workers, len(units) or 1))
    record['groups'] = [entry for entry in entries if entry is not None]
    record['parameter_source'] = {i: source[i] for i in ids}
    record['counts'] = dict(Counter(source[i] for i in ids))
    return [runs[i] for i in ids], record

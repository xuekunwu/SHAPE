"""Guard, grouping and evidence-stop invariants independent of any case study."""
import copy
import math
from types import SimpleNamespace

import pytest

from shape.agent.verification.segmentation_adaptive import (
    GroupingPolicy, REQUIRED_VIEWS, cost_stop, native_features, stratify,
    triage_batch_decisions, triage_decision,
)


def test_supported_strata_preserve_every_image_and_batch_boundaries():
    specs = [dict(image_id=f'{batch}_{i:02}', acquisition=batch) for batch in ('a', 'b') for i in range(40)]
    features = {s['image_id']: dict(scale=10 if int(s['image_id'][-2:]) < 20 else 40, density=100.) for s in specs}
    policy = GroupingPolicy(batch_fields=('acquisition',))
    result = stratify(specs, features, policy)
    assert result == stratify(list(reversed(specs)), features, policy)
    assert len(result['groups']) == 4
    delivered = [i for g in result['groups'] for i in g['images']]
    assert len(delivered) == len(set(delivered)) == len(specs)
    for group in result['groups']:
        assert len(group['images']) >= policy.min_group
        assert {i.split('_')[0] for i in group['images']} == {group['batch']['acquisition']}


def test_unsupported_split_stays_at_parent_instead_of_fragmenting():
    specs = [dict(image_id=f'i{i}', batch='a') for i in range(15)]
    features = {s['image_id']: dict(scale=10 if i < 7 else 100, density=100.) for i, s in enumerate(specs)}
    result = stratify(specs, features, GroupingPolicy(batch_fields=('batch',)))
    assert len(result['groups']) == 1
    assert result['groups'][0]['shared_calibration']
    assert result['groups'][0]['strata'] == []


def test_missing_native_features_are_explicitly_ungrouped_and_metadata_required():
    specs = [dict(image_id='i1', batch='a'), dict(image_id='i2', batch='a')]
    result = stratify(specs, {'i1': dict(scale=math.nan, density=100)}, GroupingPolicy(('batch',)))
    assert result['ungrouped'] == ['i1', 'i2']
    with pytest.raises(ValueError, match='missing declared batch'):
        stratify([dict(image_id='i1')], {}, GroupingPolicy(('batch',)))


def test_native_features_do_not_depend_on_labels():
    values = native_features(100, math.pi * 25, (1000, 1000))
    assert values == pytest.approx(dict(scale=10., density=100.))
    with pytest.raises(ValueError):
        native_features(0, 10, (1000, 1000))


@pytest.fixture
def triage():
    item = dict(image_id='i1', guard_ok=True, problems=['tiny_mask_fraction 0.2 above 0.15'],
                evidence_views=sorted(REQUIRED_VIEWS))
    verdict = dict(image_id='i1', action='accept_with_flags', confidence='high',
                   image_quality_sufficient=True, normal_heterogeneity=True, visible_faults=[],
                   evidence_views=sorted(REQUIRED_VIEWS), reason='Overview and size-extreme zooms show intact heterogeneous cells.')
    return item, verdict


def test_visual_acceptance_preserves_original_qc_alarms(triage):
    item, verdict = triage
    decision = triage_decision(item, verdict)
    assert decision['route'] == 'accept_with_flags'
    assert decision['residual_problems'] == item['problems']
    assert decision['measured_problems_cleared'] is False


@pytest.mark.parametrize('change', [dict(guard_ok=False), dict(problems=['empty mask']),
    dict(problems=['mean_solidity 0.1 below 0.35']), dict(problems=['']), dict(evidence_views=['mask_overlay'])])
def test_native_failures_or_incomplete_evidence_cannot_be_visually_waived(triage, change):
    item, verdict = triage
    assert triage_decision({**item, **change}, verdict)['route'] == 'full_vrr'


@pytest.mark.parametrize('change', [dict(confidence='medium'), dict(visible_faults=['merged cells']),
    dict(image_quality_sufficient=False), dict(normal_heterogeneity=False), dict(action='abstain'),
    dict(image_id='different-image'), dict(reason='')])
def test_uncertain_or_actionable_visual_verdict_retains_full_vrr(triage, change):
    item, verdict = triage
    assert triage_decision(item, {**verdict, **change})['route'] == 'full_vrr'


def test_partial_reply_abstains_on_missing_ids_and_duplicates_reject_batch(triage):
    item, verdict = triage
    second = {**item, 'image_id': 'i2'}
    decisions = triage_batch_decisions([item, second], dict(verdicts=[verdict]))
    assert [d['route'] for d in decisions] == ['accept_with_flags', 'full_vrr']
    duplicate = triage_batch_decisions([item], dict(verdicts=[verdict, verdict]))
    assert duplicate[0]['route'] == 'full_vrr'


@pytest.fixture
def stop_case():
    evidence = dict(best_call='agent_2', actionable_faults=[], expected_gain='none', confidence='high',
                    reason='No actionable defect remains; two probes did not improve native or visual evidence.')
    inputs = dict(admissible=['default', 'agent_1', 'agent_2'],
                  assessments={c: dict(severity='minor', confidence='high') for c in ['agent_1', 'agent_2']},
                  stalled_rounds=2, recent_native_gain=0., native_noise=.001,
                  recent_calls=['agent_1', 'agent_2'], search_ready=True)
    return evidence, inputs


def test_confirmed_low_benefit_can_stop_without_changing_final_delivery_guards(stop_case):
    evidence, inputs = stop_case
    decision = cost_stop(evidence, **inputs)
    assert decision['stop'] and decision['call'] == 'agent_2'


@pytest.mark.parametrize('gain', ['minor', 'major', 'uncertain'])
def test_possible_late_visual_gain_is_not_cut_by_scalar_plateau(stop_case, gain):
    evidence, inputs = stop_case
    assert not cost_stop({**evidence, 'expected_gain': gain}, **inputs)['stop']


@pytest.mark.parametrize('change', [dict(stalled_rounds=1), dict(recent_native_gain=.02),
    dict(recent_native_gain=None), dict(search_ready=False), dict(admissible=['default']),
    dict(recent_calls=['agent_2'])])
def test_insufficient_or_positive_native_evidence_prevents_forced_stop(stop_case, change):
    evidence, inputs = stop_case
    assert not cost_stop(evidence, **{**inputs, **change})['stop']


def test_major_or_unassessed_defects_keep_search_possible(stop_case):
    evidence, inputs = stop_case
    changed = copy.deepcopy(inputs)
    changed['assessments']['agent_2']['severity'] = 'major'
    assert not cost_stop(evidence, **changed)['stop']
    changed['assessments'].pop('agent_2')
    assert not cost_stop(evidence, **changed)['stop']


@pytest.mark.parametrize('workers', [1, 3])
def test_real_cohort_loop_retains_uncertain_images_and_rechecks_visual_acceptance(workers):
    from shape.agent.verification import segmentation_cohort
    specs = [dict(image_id=f'i{i:02}', batch='a') for i in range(20)]
    calls = []
    seen = []
    defaults = dict(cellprob_threshold=0.)
    def run_one(spec, options):
        calls.append((spec['image_id'], dict(options)))
        problems = ['tiny_mask_fraction 0.2 above 0.15']
        resolution = SimpleNamespace(status='satisfied', reason='finished',
            params=defaults, result=dict(status='success', cell_count=10, mask_path='native-mask.npy'),
            details=dict(admission={'decision': 'RUN'}, call='default', measured_problems={'default': problems}))
        return [resolution], dict(levers={}, cell_count=10), 'satisfied', None
    def triage(requests):
        seen.extend(r['spec']['image_id'] for r in requests)
        first = requests[0]['spec']['image_id']
        item = dict(image_id=first, guard_ok=True, problems=requests[0]['problems'], evidence_views=sorted(REQUIRED_VIEWS))
        verdict = dict(image_id=first, action='accept_with_flags', confidence='high',
            image_quality_sufficient=True, normal_heterogeneity=True, visible_faults=[],
            evidence_views=sorted(REQUIRED_VIEWS), reason='Overview, contours and zooms support harmless size heterogeneity.')
        # A naked accept route on the second image must not bypass verification.
        return [triage_decision(item, verdict), dict(image_id=requests[1]['spec']['image_id'], route='accept_with_flags')]
    results, record = segmentation_cohort.run(specs, run_one,
        lambda items, fn: [fn(item) for item in items], lambda spec: None,
        defaults, list(defaults), workers=workers, routing_policy='stratified_triage_v1',
        grouping_policy=GroupingPolicy(('batch',), max_depth=0),
        read_features=lambda s: dict(scale=30., density=100.), triage=triage, gain_policy='evidence_v1')
    assert len(results) == len(specs)
    assert len(seen) == 17  # three actual calibration searches, then 17 cohort images
    group = record['groups'][0]
    assert group['routing']['accepted_with_recorded_alerts'] == 1
    assert group['routing']['individual_vrr'] == 16
    assert all(options['gain_policy'] == 'evidence_v1' for _, options in calls)
    accepted = next(i for i, d in group['triage'].items() if d['route'] == 'accept_with_flags')
    assert record['parameter_source'][accepted] == 'cohort'
    assert group['triage'][accepted]['residual_problems']


def test_adaptive_options_cannot_silently_activate_in_legacy_cohort():
    from shape.agent.verification import segmentation_cohort
    with pytest.raises(ValueError, match='explicit adaptive routing'):
        segmentation_cohort.run([], None, None, None, {}, [], grouping_policy=GroupingPolicy(('batch',)))

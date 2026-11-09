"""Exercise the real verifier's opt-in schema, history and plateau guard."""
import pytest

from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier
from shape.utils.schemas import VLRRResult


@pytest.fixture
def search(monkeypatch):
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY', 'four_stage')
    monkeypatch.setenv('SHAPE_VRR_POLICY', 'balanced_v1')
    verifier = SegmentationAgenticVerifier(gain_policy='evidence_v1')
    names = ['default', 'agent_1', 'agent_2', 'agent_3']
    verifier.rounds = [[i] for i in range(1, 5)]
    verifier._names.update(dict(enumerate(names, 1)))
    attempts = {c: ({}, {}, VLRRResult('PASS', .5, {'cell_count': 10.})) for c in names}
    verifier.assessments = {c: dict(severity='minor', confidence='high') for c in names}
    payload = dict(analysis='Compared native contours and readouts.', assessments=[], moves=[],
        continuation=dict(best_call='agent_3', actionable_faults=[], expected_gain='none', confidence='high',
            reason='Finished controls show no further useful visual refinement.'),
        search=dict(phase='confirm', reason='No useful expected gain.'))
    plan = verifier.plan_schema.model_validate(payload)
    verifier.record_plan(plan, 'round', attempts)
    return verifier, attempts


def test_real_plan_schema_retains_four_stage_comparison_and_continuation(search):
    verifier, attempts = search
    fields = verifier.plan_schema.model_fields
    assert {'search', 'comparison', 'continuation'} <= set(fields)
    assert verifier._plateau(None, attempts)
    assert verifier.stop_reason == 'evidence_plateau'
    assert verifier.continuation_events[-1]['stop']
    assert verifier.plan_delivery is None  # no bypass of actual final selection


@pytest.mark.parametrize('cause', ['visual_gain', 'unknown', 'native_gain', 'unassessed', 'major', 'stage_not_ready'])
def test_real_plateau_keeps_plausible_late_candidates(search, cause):
    verifier, attempts = search
    if cause == 'visual_gain':
        verifier._continuation_evidence['expected_gain'] = 'major'
    elif cause == 'unknown':
        verifier._continuation_evidence = None
    elif cause == 'native_gain':
        attempts['agent_3'][2].scalar_score = .6
    elif cause == 'unassessed':
        verifier.assessments.pop('agent_2')
    elif cause == 'major':
        verifier.assessments['agent_2']['severity'] = 'major'
    else:
        verifier._search.phase_round = len(verifier.rounds)
    assert not verifier._plateau(None, attempts)


def test_original_schema_is_unchanged_without_opt_in(monkeypatch):
    monkeypatch.setenv('SHAPE_SEARCH_STRATEGY', 'four_stage')
    verifier = SegmentationAgenticVerifier()
    assert 'continuation' not in verifier.plan_schema.model_fields
    assert verifier.gain_policy == 'legacy'

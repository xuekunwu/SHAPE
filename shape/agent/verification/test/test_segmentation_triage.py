"""Evidence-backed callback tests; no GPU or paid model calls."""
import json
from types import SimpleNamespace

import numpy as np
import pytest
from PIL import Image

from shape.agent.verification.segmentation_adaptive import REQUIRED_VIEWS
from shape.agent.verification.segmentation_triage import VisualTriage, evidence_card


@pytest.fixture
def native(tmp_path):
    raw = np.indices((48, 64)).sum(axis=0).astype('uint8')
    image = tmp_path/'image.png'
    Image.fromarray(raw).save(image)
    mask = np.zeros(raw.shape, dtype='uint32')
    mask[3:7, 4:8] = 1
    mask[20:40, 30:55] = 4_000_000_000
    path = tmp_path/'mask.npy'
    np.save(path, mask)
    resolution = SimpleNamespace(verifier='segmentation', status='satisfied',
        result=dict(status='success', cell_count=2, mask_path=str(path)))
    return dict(spec=dict(image_id='image_a', path=str(image)), resolutions=[resolution],
                problems=['tiny_mask_fraction 0.2 above 0.15'])


def verdict(ident='image_a', **changes):
    return dict(image_id=ident, action='accept_with_flags', confidence='high',
        image_quality_sufficient=True, normal_heterogeneity=True, visible_faults=[],
        evidence_views=sorted(REQUIRED_VIEWS), reason='Raw, overlay and zooms support normal heterogeneity.', **changes)


class Engine:
    def __init__(self, answer=None):
        self.answer = answer or dict(verdicts=[verdict()])
        self.calls = []

    def generate(self, content, **kwargs):
        self.calls.append(content)
        return dict(content=self.answer, usage=dict(total_tokens=123))


def test_actual_callback_saves_native_cards_model_usage_and_checked_decisions(tmp_path, native):
    engine = Engine()
    triage = VisualTriage(engine, tmp_path/'triage', task='Whole-cell instance segmentation.',
                          model_identity=dict(model='requested', effort='high'))
    result = triage([native])
    assert result[0]['route'] == 'accept_with_flags'
    assert result[0]['residual_problems'] == native['problems']
    assert len(engine.calls) == 1 and len(engine.calls[0]) == 2
    receipt = json.loads(next((tmp_path/'triage').glob('*/RECEIPT.json')).read_text())
    assert receipt['usage']['total_tokens'] == 123 and not receipt['decision_reused']
    request = json.loads(next((tmp_path/'triage').glob('*/REQUEST.json')).read_text())
    assert request['evidence'][0]['zooms'][1]['label'] == 4_000_000_000
    assert set(request['evidence'][0]['views']) == REQUIRED_VIEWS


def test_same_inputs_keep_fingerprint_but_reasoning_is_not_silently_reused(tmp_path, native):
    engine = Engine()
    callback = VisualTriage(engine, tmp_path/'triage', task='Whole cells', model_identity=dict(model='test'))
    callback([native]); callback([native])
    records = [json.loads(p.read_text()) for p in (tmp_path/'triage').glob('*/REQUEST.json')]
    assert len(engine.calls) == 2
    assert records[0]['fingerprint'] == records[1]['fingerprint']
    np.save(native['resolutions'][0].result['mask_path'], np.ones((48, 64), dtype='uint32'))
    callback([native])
    assert len({json.loads(p.read_text())['fingerprint'] for p in (tmp_path/'triage').glob('*/REQUEST.json')}) == 2


@pytest.mark.parametrize('kind', ['empty', 'shape', 'float', 'negative'])
def test_invalid_native_evidence_never_calls_model_or_accepts(tmp_path, native, kind):
    invalid = dict(empty=np.zeros((48, 64), dtype='uint32'), shape=np.ones((8, 8), dtype='uint32'),
                   float=np.ones((48, 64)), negative=-np.ones((48, 64), dtype='int32'))[kind]
    np.save(native['resolutions'][0].result['mask_path'], invalid)
    engine = Engine()
    decisions = VisualTriage(engine, tmp_path/'triage', task='Whole cells', model_identity=dict(model='test'))([native])
    assert decisions[0]['route'] == 'full_vrr' and not engine.calls
    assert json.loads(next((tmp_path/'triage').glob('*/REQUEST.json')).read_text())['render_errors']


@pytest.mark.parametrize('answers', [dict(verdicts=[]), dict(verdicts=[verdict(), verdict()]),
    dict(verdicts=[verdict('unknown')]), dict(broken=True)])
def test_ambiguous_missing_and_malformed_answers_fall_back(tmp_path, native, answers):
    decisions = VisualTriage(Engine(answers), tmp_path/'triage', task='Whole cells',
        model_identity=dict(model='test'))([native])
    assert decisions[0]['route'] == 'full_vrr'


def test_each_batched_card_is_bound_to_its_own_image_id(tmp_path, native):
    other = {**native, 'spec': {**native['spec'], 'image_id': 'image_b'}}
    engine = Engine(dict(verdicts=[verdict(), verdict('image_b')]))
    decisions = VisualTriage(engine, tmp_path/'triage', task='Whole cells',
        model_identity=dict(model='test'))([native, other])
    assert len(engine.calls) == 1 and len(engine.calls[0]) == 3
    assert {d['image_id'] for d in decisions if d['route'] == 'accept_with_flags'} == {'image_a', 'image_b'}

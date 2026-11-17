"""Offline checks of the organoid segmentation cohort wiring (`organoid_cohort.segmentation`): SHAPE's real
`segmentation_cohort.run` routes the images (grouping, seeded calibration sample, consensus, cohort mode, routed images);
the segmenter, the reasoner and the VRR (`segment_resolution`) are stand-ins. No GPU, no model, no network. These tests
check the wiring and the outputs; they say nothing about segmentation quality or the cohort's real token use."""
from __future__ import annotations

import dataclasses
import importlib
import json
from pathlib import Path
import sys
import types

import numpy as np
import pytest
import tifffile

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

cohort = importlib.import_module('case studies.cardiac organoids.runners.cohort.organoid_cohort')
from benchmarks.full_CPJUMP1.harness import pipeline  # noqa: E402
from benchmarks._shared.harness import shape_arm  # noqa: E402
from shape.agent.verification import segmentation_cohort  # noqa: E402

SMALL = (24, 32)
DEFAULTS = {'min_size': 15, 'flow_threshold': 0.4}
FLAGGED = 'i0007'            # an image whose cohort-mode result the admission standard flags


def blob(value):
    mask = np.zeros(SMALL, dtype=np.int32)
    mask[4:14, 6:20] = value
    return mask


class FakeTool:
    device = types.SimpleNamespace(type='cuda')

    def __init__(self):
        self.calls = []

    def execute(self, image, image_id, query_cache_dir, **kwargs):
        self.calls.append((image_id, kwargs))
        folder = Path(query_cache_dir)
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f'cell_mask_{image_id}.npy'
        np.save(path, blob(1))
        return {'mask_path': str(path), 'overlay_path': None}


def resolution(params, problems=()):
    details = {'admission': {'decision': 'SKIP'}, 'call': 'S001', 'measured_problems': {'S001': list(problems)} if problems else {}}
    return types.SimpleNamespace(status='satisfied', reason='', fault=None, result={'cell_count': 1}, params=dict(params), details=details)


@pytest.fixture
def world(tmp_path, monkeypatch):
    prep = tmp_path / 'prep'
    (prep / 'public' / 'images').mkdir(parents=True)
    images = []
    for i in range(40):
        ident = f'i{i:04d}'
        tifffile.imwrite(prep / 'public' / 'images' / f'{ident}.tif', np.zeros(SMALL, dtype=np.uint16))
        images.append(dict(image_id=ident, segment_file=f'images/{ident}.tif'))
    real = pipeline.dataset({'dataset': 'co_fc_timecourse', 'smoke': True})
    spec = dataclasses.replace(real, image_shape=SMALL)
    tool = FakeTool()
    seen = []

    def fake_resolution(C, tool_, item, levers, defaults, engine, scratch, log_dir, lock=None, budget=None,
                        verifier_options=None, tool_name='', fixed=None):
        options = dict(verifier_options or {})
        seen.append((item['image_id'], options, tool_name))
        scratch.mkdir(parents=True, exist_ok=True)
        if 'cohort_levers' in options:                     # cohort mode: the consensus levers, one reference call
            changed = dict(options['cohort_levers'])
            problems = ['coverage far below the cohort'] if item['image_id'] == FLAGGED else []
        else:                                              # the image's own VRR (calibration, routed): flow_threshold 0.2
            changed = {'flow_threshold': 0.2}
            problems = []
        mask_path = scratch / f"mask_{item['image_id']}.npy"
        np.save(mask_path, blob(2))
        record = dict(status='searched', delivered='C1', defaults_call_reused=True, levers=changed,
                      mask_npy=str(mask_path) if changed else None, cell_count=1, n_calls=1 if options.get('cohort_levers') else 2,
                      fallback_rounds=0, reasoner_failures=0, execution_errors=[], stop_reason=None)    # the health counters of a real record
        return record, resolution({**defaults, **changed}, problems)

    monkeypatch.setattr(pipeline, 'dataset', lambda config: spec)
    monkeypatch.setattr(pipeline, 'engine_for', lambda config, stage: object())
    monkeypatch.setattr(pipeline, 'segmenter', lambda spec_, params=None: dict(
        tool=tool, name='Organoid_Segmenter_Tool', fixed={'adaptive_rescue': False}, registry={k: None for k in DEFAULTS}, defaults=dict(DEFAULTS)))
    monkeypatch.setattr(shape_arm, 'tools_common', lambda: object())
    monkeypatch.setattr(shape_arm, 'segment_resolution', fake_resolution)
    profile = types.SimpleNamespace(confidence='high', specimen_context='organoid', modality='phase contrast')
    monkeypatch.setattr(segmentation_cohort, 'preflight_read', lambda path, engine, image_id=None: profile)
    config = {'dataset': 'co_fc_timecourse', 'arm': 'shape', 'segmentation_regime': 'cohort', 'cores': 2, 'deadline': None}
    return prep, tmp_path / 'run', config, images, tool, seen


def test_cohort_routes_and_writes_outputs(world):
    prep, run, config, images, tool, seen = world
    (run).mkdir()
    cohort.segmentation(prep, run, config, images)
    outputs = json.loads((run / 'segmentation.json').read_text())
    assert [o['image_id'] for o in outputs] == [i['image_id'] for i in images]
    sources = {o['image_id']: o['parameter_source'] for o in outputs}
    n_calibration = segmentation_cohort.calibration_size(len(images))
    assert n_calibration == 3                                  # 40 images: 5 % rounds to 2, the floor is 3
    assert sum(s == 'calibration' for s in sources.values()) == n_calibration
    assert sources[FLAGGED] == 'per_image_vrr'                 # flagged in cohort mode, routed to its own VRR
    assert sum(s == 'cohort' for s in sources.values()) == len(images) - n_calibration - 1
    record = json.loads((run / 'segment_COHORT.json').read_text())
    assert record['implementation'] == cohort.IMPLEMENTATION and record['organoid_selective_review'] is False
    assert record['tool'] == 'Organoid_Segmenter_Tool' and record['counts'] == {'calibration': 3, 'cohort': 36, 'per_image_vrr': 1}
    assert record['groups'][0]['consensus']['selected_params'] == {'flow_threshold': 0.2}
    # the cohort pass of every non-calibration image carries the consensus levers; the routed image starts from them
    by_mode = {}
    for ident, options, tool_name in seen:
        assert tool_name == 'Organoid_Segmenter_Tool'
        by_mode.setdefault('cohort' if 'cohort_levers' in options else 'routed' if 'start_params' in options else 'search', []).append(ident)
    assert len(by_mode['search']) == 3 and len(by_mode['cohort']) == 37 and by_mode['routed'] == [FLAGGED]
    # outputs: the selected mask where levers changed (blob 2), each image's own S001 call made once
    mask = tifffile.imread(outputs[0]['mask'])
    assert mask.shape == SMALL and int(mask.max()) == 2 and outputs[0]['min_area'] == 15
    assert len(tool.calls) == len(images) and all(kwargs['adaptive_rescue'] is False for _, kwargs in tool.calls)
    cohort_image = next(i for i, s in sources.items() if s == 'cohort')
    assert (run / 'segment' / FLAGGED / 'SEARCH.json').is_file()
    assert (run / 'segment' / cohort_image / 'COHORT_SEARCH.json').is_file()


def test_arm_without_reasoner_or_regime_takes_the_per_image_path(world, monkeypatch):
    prep, run, config, images, _, _ = world
    called = []
    monkeypatch.setattr(pipeline, 'segmentation', lambda *args: called.append('per-image'))
    monkeypatch.setattr(pipeline, 'engine_for', lambda config, stage: None)
    cohort.segmentation(prep, run, config, images)                          # control: no engine
    monkeypatch.setattr(pipeline, 'engine_for', lambda config, stage: object())
    cohort.segmentation(prep, run, {**config, 'segmentation_regime': 'per-image'}, images)
    assert called == ['per-image', 'per-image']


def test_wrong_tool_is_refused(world, monkeypatch):
    prep, run, config, images, tool, _ = world
    monkeypatch.setattr(pipeline, 'segmenter', lambda spec_, params=None: dict(
        tool=tool, name='Cell_Segmenter_Tool', fixed={}, registry={k: None for k in DEFAULTS}, defaults=dict(DEFAULTS)))
    with pytest.raises(ValueError, match='organoid segmenter'):
        cohort.segmentation(prep, run, config, images)


def test_install_replaces_only_the_segment_stage(monkeypatch):
    original = dict(pipeline.STAGES)
    monkeypatch.setattr(pipeline, 'STAGES', dict(original))
    cohort.install()
    assert pipeline.STAGES['segment'] is cohort.segmentation
    assert {k: v for k, v in pipeline.STAGES.items() if k != 'segment'} == {k: v for k, v in original.items() if k != 'segment'}

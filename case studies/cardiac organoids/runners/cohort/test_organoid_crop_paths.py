"""Offline checks of what the harness's crop stage does for the organoid datasets: the crops are cut from the row's raw
stack (two channels in the `ssl_two_channel` variant), an image without an object is not an error, and the crop VRR's
mask-error check can be switched off (`crop_mask_check`). The cropper and the reasoner are stand-ins; the harness's
`_crop_materialize`, `retained_crops` and `crop_one` run for real. These tests check wiring, not crop quality."""
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

from benchmarks.full_CPJUMP1.harness import pipeline  # noqa: E402
from benchmarks.full_CPJUMP1.harness.data import digest  # noqa: E402

SMALL = (24, 32)


def two_channel_spec():
    real = importlib.import_module('case studies.cardiac organoids.fc_timecourse.runners.spec').for_config(
        {'smoke': True, 'variant': 'ssl_two_channel'})
    return dataclasses.replace(real, image_shape=SMALL)


@pytest.fixture
def world(tmp_path, monkeypatch):
    prep, run = tmp_path / 'prep', tmp_path / 'run'
    (prep / 'public' / 'images').mkdir(parents=True)
    (prep / 'public' / 'raw').mkdir()
    spec = two_channel_spec()
    monkeypatch.setattr(pipeline, 'dataset', lambda config: spec)
    rows, masks = {}, {}
    for ident, objects in (('i0000', 1), ('i0001', 0)):
        tifffile.imwrite(prep / 'public' / 'images' / f'{ident}.tif', np.full(SMALL, 1000, dtype=np.uint16))
        tifffile.imwrite(prep / 'public' / 'raw' / f'{ident}.tif',
                         np.stack([np.full(SMALL, 1000, np.uint16), np.full(SMALL, 7000, np.uint16)]))
        mask = np.zeros(SMALL, dtype=np.uint32)
        if objects:
            mask[4:14, 6:20] = 1                                   # bbox (4, 6, 14, 20), area 140
        path = tmp_path / f'cell_mask_{ident}.tif'
        tifffile.imwrite(path, mask)
        masks[ident] = dict(image_id=ident, mask=str(path), sha256=digest(path), min_area=15, n_cells=objects)
        rows[ident] = dict(image_id=ident, file=f'images/{ident}.tif', raw_file=f'raw/{ident}.tif', plate='p000',
                           well=f'w{ident[-2:]}', usage='development_train')

    class Cropper:
        def execute(self, *, original_image, nuclei_mask, source_image_id, query_cache_dir, **params):
            folder = Path(query_cache_dir)
            folder.mkdir(parents=True, exist_ok=True)
            display = folder / f'{source_image_id}.png'
            display.write_bytes(b'png')
            cells = []
            if np.any(tifffile.imread(nuclei_mask)):
                cells.append(dict(cell_id=f'default_{source_image_id}_cell_0', crop_path=str(display),
                                  crop_bbox=[3, 6, 17, 20], original_bbox=[4, 6, 14, 20], area=140))
            meta = folder / 'cohort_crops_metadata.json'
            meta.write_text(json.dumps({'cell_metadata': cells}))
            return {'cell_crops_metadata_path': str(meta), 'cell_count': len(cells)}

    monkeypatch.setattr('shape.tools.single_cell_cropper.tool.Single_Cell_Cropper_Tool', Cropper)
    config = {'dataset': 'co_fc_timecourse', 'arm': 'shape', 'cores': 2, 'deadline': None}
    return prep, run, config, rows, masks


def materialize(world, ident):
    prep, run, config, rows, masks = world
    delivered = {'params': {}, 'cell_overrides': {}, 'rejected': set(), 'record': None}
    return pipeline._crop_materialize((rows[ident], prep, run, config, masks[ident], delivered))


def test_crops_are_cut_from_the_raw_stack_of_the_row(world):
    kept, excluded, record, _ = materialize(world, 'i0000')
    (cell,) = kept
    raw = tifffile.imread(cell['crop_path'])
    assert raw.shape == (2, 14, 14) and int(raw[0, 0, 0]) == 1000 and int(raw[1, 0, 0]) == 7000
    assert cell['mask_label'] == 1 and excluded == []


def test_an_image_without_an_object_has_no_crops_and_is_not_an_error(world):
    kept, excluded, _, _ = materialize(world, 'i0001')
    assert kept == [] and excluded == []


def test_an_empty_mask_is_an_error_where_the_dataset_does_not_allow_it(world, monkeypatch):
    spec = dataclasses.replace(two_channel_spec(), allow_empty_masks=False)
    monkeypatch.setattr(pipeline, 'dataset', lambda config: spec)
    with pytest.raises(RuntimeError, match='no retained crops for i0001'):
        materialize(world, 'i0001')


@pytest.mark.parametrize('mask_check', [True, False])
def test_crop_one_hands_the_mask_check_switch_to_the_crop_task(world, monkeypatch, mask_check):
    from benchmarks.full_HCMA.harness import shape_arm
    from shape.agent.verification import step_verifier
    prep, run, config, rows, masks = world
    seen = {}

    class Stepper:
        def __init__(self, **kwargs):
            pass

        def resolve_stage(self, verifier, adapter, runner, params, input_record, upstream_ctx=None):
            seen['task'] = upstream_ctx['crop_task']
            return types.SimpleNamespace(status='satisfied', reason='', fault=None, trials=[None], params={},
                                         result=None, details={})

    monkeypatch.setattr(step_verifier, 'StepVerifier', Stepper)
    item = {'image_id': 'i0000', 'path': str(prep / 'public' / 'images' / 'i0000.tif'), 'mask': masks['i0000']['mask'],
            'min_area': 15, 'defaults': {}, 'brief': 'crop'}
    shape_arm.crop_one(shape_arm.tools_common(), lambda: object(), item, None, None, 1, run / 'scratch', run / 'logs',
                       budget=1, mask_check=mask_check)
    assert seen['task'].enable_mask_check is mask_check

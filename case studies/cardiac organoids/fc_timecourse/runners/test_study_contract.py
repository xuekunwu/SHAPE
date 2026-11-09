"""Offline checks of the FC time-course study: preparation and split contract, the organoid segmenter registry
and prompt contract of the agent arms, and the trusted predict / score stages on synthetic data. No GPU, no model,
no network."""
from __future__ import annotations

import importlib
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
import tifffile

REPO = Path(__file__).resolve().parents[4]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

data = importlib.import_module('case studies.cardiac organoids.fc_timecourse.runners.data')
spec_module = importlib.import_module('case studies.cardiac organoids.fc_timecourse.runners.spec')
score = importlib.import_module('case studies.cardiac organoids.fc_timecourse.runners.score')

SMALL = [24, 32]
WELLS = [f'{row}{col}' for row in 'BCDEFG' for col in (3, 4, 5, 6) if f'{row}{col}' != 'F6']


@pytest.fixture
def small_images(monkeypatch):
    monkeypatch.setitem(data.SPEC, 'image_shape', SMALL)


def synthetic_source(root: Path, seed=0) -> Path:
    """The source layout: per-time-point folders with phase and GFP TIFFs; F6 only at day 4."""
    rng = np.random.default_rng(seed)
    for folder, day in data.DAYS.items():
        d = root / folder
        d.mkdir(parents=True)
        wells = WELLS + (['F6'] if day == 4 else [])
        seq = '01' if day in (0, 2) else '02'
        for well in wells:
            phase = rng.integers(1000, 60000, size=SMALL, dtype=np.uint16)
            gfp = rng.integers(5000, 20000, size=SMALL, dtype=np.uint16)
            if well[0] in 'EFG':
                gfp[4:12, 6:20] = 30000          # a bright focus in the FC wells
            tifffile.imwrite(d / f'{well}_{seq}_1_1_Phase Contrast_001.tif', phase)
            tifffile.imwrite(d / f'{well}_{seq}_2_1_GFP_001.tif', gfp)
    return root


def test_prepare_contract(tmp_path, small_images):
    source = synthetic_source(tmp_path / 'source')
    record = data.prepare(source, tmp_path / 'prep')
    assert record['images'] == 184 and record['wells'] == 23
    assert record['wells_by_condition'] == {'Control': 12, 'FC': 11}
    assert record['excluded_wells'] == {'F6': {'days': [4], 'reason': 'incomplete time course'}}
    rows = data.load_prepared(tmp_path / 'prep', verify_pixels=True)
    private = data.private_images(tmp_path / 'prep')
    splits = json.loads((tmp_path / 'prep' / 'private' / 'splits.json').read_text())
    # the public table carries no condition, day or source name
    for r in rows:
        assert set(r) & {'condition', 'day', 'source_well', 'timepoint'} == set()
        assert r['image_id'].startswith('i') and r['well'].startswith('w')
    # the split is by well, stratified by condition, at the contract's counts
    by_usage = {}
    for well, info in splits['wells'].items():
        by_usage.setdefault((info['condition'], info['usage']), []).append(well)
    assert {k: len(v) for k, v in by_usage.items()} == {
        ('Control', 'development_train'): 6, ('Control', 'development_val'): 2, ('Control', 'evaluation_only'): 4,
        ('FC', 'development_train'): 5, ('FC', 'development_val'): 2, ('FC', 'evaluation_only'): 4}
    for r in rows:
        assert private[r['image_id']]['usage'] == r['usage'] == splits['wells'][r['well']]['usage']
    # every well has eight time points and the pseudonyms do not follow the plate order
    wells = {}
    for r in rows:
        wells.setdefault(r['well'], set()).add(private[r['image_id']]['day'])
    assert all(days == set(data.DAYS.values()) for days in wells.values())
    control_ids = sorted(w for w, i in splits['wells'].items() if i['condition'] == 'Control')
    assert control_ids != [f'w{i:02d}' for i in range(12)]
    # the GFP threshold is the 99th percentile of baseline GFP
    assert data.gfp_threshold(tmp_path / 'prep') > 5000
    smoke = data.smoke_images(rows, private)
    assert len(smoke) == 6
    train = [private[r['image_id']] for r in smoke if r['usage'] == 'development_train']
    assert {t['condition'] for t in train} == {'Control', 'FC'} and {t['day'] for t in train} == {0, 16}
    assert [r['usage'] for r in smoke].count('evaluation_only') == 1
    # a second prepare into the same folder is refused; a changed table is detected
    with pytest.raises(FileExistsError):
        data.prepare(source, tmp_path / 'prep')
    table = tmp_path / 'prep' / 'public' / 'images.json'
    table.write_text(table.read_text().replace('"usage": "development_val"', '"usage": "development_train"', 1))
    with pytest.raises(ValueError):
        data.load_prepared(tmp_path / 'prep')


def test_dataset_declaration_and_registry():
    from benchmarks.full_CPJUMP1.harness import pipeline
    spec = spec_module.CO_FC
    assert spec.segment_tool == 'organoid_segmenter' and spec.channels == ('PhaseContrast',)
    assert pipeline.dataset({'dataset': 'co_fc_timecourse'}).allowed_sites == (8,)
    assert pipeline.dataset({'dataset': 'co_fc_timecourse', 'smoke': True}).allowed_sites == tuple(range(1, 9))
    assert pipeline.CPJUMP1.segment_tool == 'cell_segmenter'
    registry = pipeline.segment_registry(spec)
    assert list(registry) == ['diameter', 'flow_threshold', 'cellprob_threshold', 'min_size', 'niter',
                              'tile_overlap', 'augment', 'resample']
    assert registry['flow_threshold'].default == 0.4 and registry['min_size'].default == 15


def _fake_run(prep: Path, run: Path, rows, private, seed=0):
    """A finished embed stage on synthetic data: one object per image whose embedding separates the conditions."""
    rng = np.random.default_rng(seed)
    (run / 'segment').mkdir(parents=True)
    masks, cells, x = [], [], []
    for r in rows:
        ident = r['image_id']
        mask = np.zeros(SMALL, dtype=np.uint32)
        mask[6:18, 8:24] = 1
        if private[ident]['day'] >= 8:
            mask[20:23, 1:5] = 2                         # a second small object late in the course
        path = run / 'segment' / f'cell_mask_{ident}.tif'
        tifffile.imwrite(path, mask)
        masks.append(dict(image_id=ident, mask=str(path), sha256=data.digest(path), min_area=15,
                          n_cells=int(mask.max())))
        for label in range(1, int(mask.max()) + 1):
            area = int((mask == label).sum())
            cells.append(dict(cell_id=f'{ident}_{label}', image_id=ident, plate=r['plate'], well=r['well'],
                              usage=r['usage'], area=area, mask_label=label, crop_path='', sha256=''))
            fc = private[ident]['condition'] == 'FC'
            x.append(np.concatenate([[3.0 if fc else -3.0, private[ident]['day'] / 4.0], rng.normal(size=6)]))
    data.write_json(run / 'segmentation.json', masks)
    data.write_json(run / 'cells.json', cells)
    np.save(run / 'embedding.npy', np.asarray(x, dtype=np.float32))
    data.write_json(run / 'CONFIG.json', dict(arm='control', model='gpt-5.6-luna', seed=seed, smoke=True,
                                              images=[r['image_id'] for r in rows]))


def test_predict_and_score(tmp_path, small_images):
    source = synthetic_source(tmp_path / 'source')
    data.prepare(source, tmp_path / 'prep')
    prep = tmp_path / 'prep'
    rows = data.load_prepared(prep)
    private = data.private_images(prep)
    run = tmp_path / 'run'
    _fake_run(prep, run, rows, private)
    record = score.predict(prep, run)
    assert record['primary'] == 'day_classifier'
    day = record['day_classifier']
    assert day['fit_split'] == 'development_train' and day['C'] in score.C_GRID and day['classes'] == score.DAYS
    assert record['condition_probe']['selection_split'] == 'development_val'
    predictions = data.read_json(run / 'predictions.json')
    assert len(predictions) == len(data.read_json(run / 'cells.json'))
    assert all(len(p['day_class_probabilities']) == 8 and p['day_class_prediction'] in score.DAYS for p in predictions)
    metrics = score.score(prep, run)
    p = metrics['prediction']
    assert p['heldout_wells'] == 8 and p['heldout_images'] == 64
    d = p['day_image_level']
    assert d['n'] == 64 and d['accuracy'] > 0.3 and d['within_one_timepoint'] > d['accuracy'] and d['chance_accuracy'] == 0.125
    assert np.asarray(d['confusion']['matrix']).sum() == 64 and len(d['confusion']['matrix']) == 8
    assert set(p['day_image_level_by_condition']) == {'Control', 'FC'}
    assert p['day_regression']['spearman_rho'] > 0.5
    assert p['condition_image_level']['accuracy'] > 0.9 and p['condition_crop_level']['n'] > 64
    assert p['bootstrap']['unit'] == 'held-out well' and 'day_accuracy' in p['bootstrap']['image_level']
    gfp = metrics['gfp']
    assert len(gfp['by_condition_day']) == 16 and gfp['target'] == score.GFP_TARGET
    assert gfp['by_condition_day']['FC|0']['gfp_positive_fraction'] > gfp['by_condition_day']['Control|0']['gfp_positive_fraction']
    assert gfp['by_condition_day']['FC|0']['gfp_integrated_bgsub'] > gfp['by_condition_day']['Control|0']['gfp_integrated_bgsub']
    # the GFP probe and its label-only reference are scored on held-out images and bootstrapped over wells
    assert record['gfp_probe']['target'] == score.GFP_TARGET and record['gfp_probe']['fit_split'] == 'development_train'
    assert record['gfp_label_baseline']['n_parameters'] == 9
    g = p['gfp_image_level']
    assert g['n'] == 64 and {'spearman_rho', 'mae', 'baseline_spearman_rho', 'baseline_mae'} <= set(g)
    assert 'gfp_spearman_rho' in p['bootstrap']['image_level']
    assert all('gfp_prediction' in q and 'gfp_label_baseline' in q for q in predictions)
    table = pd.read_csv(run / 'per_object_gfp.csv')
    assert {'gfp_integrated_bgsub', 'gfp_background_mean', 'gfp_prediction', 'gfp_label_baseline'} <= set(table.columns)
    assert np.allclose(table.gfp_integrated_bgsub, table.gfp_sum - table.area * table.gfp_background_mean)
    assert metrics['segmentation']['images_with_objects'] == 184
    assert (run / 'per_image.csv').is_file() and (run / 'per_object_gfp.csv').is_file()
    # held-out labels never enter fitting: the classifier records name development splits only
    assert day['selection_split'] == 'development_val' and day['refit_on_validation'] is False
    # a changed prediction table is refused by the scorer
    (run / 'predictions.json').write_text('[]')
    with pytest.raises(ValueError):
        score.score(prep, run)


def test_defaults_call_keeps_a_qc_failed_organoid_mask(tmp_path, monkeypatch):
    """The organoid segmenter reports a failed visual QC as `error` with its mask still delivered: the call at the
    defaults keeps the mask and records the complaint; Cellpose-SAM's contract (error = no delivery) is unchanged."""
    from benchmarks.full_CPJUMP1.harness import pipeline

    class FakeTool:
        def __init__(self, error):
            self.error = error

        def execute(self, image, image_id, query_cache_dir, **params):
            folder = Path(query_cache_dir)
            folder.mkdir(parents=True, exist_ok=True)
            mask = np.zeros(SMALL, dtype=np.int32)
            mask[2:10, 3:12] = 1
            path = folder / f'organoid_mask_{image_id}.npy'
            np.save(path, mask)
            out = {'mask_path': str(path), 'overlay_path': None, 'organoid_count': 1, 'overlay_qc': {'passed': not self.error}}
            if self.error:
                out['error'] = 'Segmentation output failed visual QC: organoids=1, foreground_coverage=0.000100'
            return out

    def fake_segmenter(spec, tool_params=None):
        return {'tool': FakeTool(error=True), 'name': 'X', 'fixed': {}, 'registry': {'min_size': None}, 'defaults': {'min_size': 15}}

    monkeypatch.setattr(pipeline, 'segmenter', fake_segmenter)
    prep = tmp_path / 'prep'
    (prep / 'public' / 'images').mkdir(parents=True)
    tifffile.imwrite(prep / 'public' / 'images' / 'i0.tif', np.zeros(SMALL, dtype=np.uint16))
    rows = [dict(image_id='i0', segment_file='images/i0.tif', file='images/i0.tif')]
    organoid = dict(dataset='co_fc_timecourse', arm='control', smoke=True, cores=1, deadline=None, images=['i0'])
    from dataclasses import replace
    small_spec = replace(spec_module.CO_FC, image_shape=tuple(SMALL))
    monkeypatch.setattr(pipeline, 'dataset', lambda config: small_spec if config.get('dataset') == 'co_fc_timecourse'
                        else replace(pipeline.CPJUMP1, image_shape=tuple(SMALL)))
    run = tmp_path / 'run_organoid'
    pipeline._segmentation_per_image(prep, run, organoid, rows, engine=None)
    record = json.loads((run / 'segmentation.json').read_text())[0]
    assert record['n_cells'] == 1 and 'failed visual QC' in record['defaults_qc_error']
    cell = dict(organoid, dataset='CPJUMP1')
    with pytest.raises(RuntimeError):
        pipeline._segmentation_per_image(prep, tmp_path / 'run_cell', cell, rows, engine=None)


def test_two_channel_ssl_variant(tmp_path, small_images):
    """The ssl_two_channel variant: segmentation and crop decisions still read the phase-contrast image; the public
    row carries a two-channel (phase, GFP) stack that the crop stage cuts the SSL crops from."""
    from benchmarks.full_CPJUMP1.harness import pipeline
    from . import spec as spec_module
    source = synthetic_source(tmp_path / 'source')
    record = data.prepare(source, tmp_path / 'prep', ssl_channels=data.TWO_CHANNEL)
    assert record['variant'] == 'ssl_two_channel' and record['ssl_channels'] == ['PhaseContrast', 'GFP']
    assert record['stage_channels'] == dict(segment=['PhaseContrast'], crop=['PhaseContrast'], embed=['PhaseContrast', 'GFP'])
    rows = data.load_prepared(tmp_path / 'prep', verify_pixels=True)
    private = data.private_images(tmp_path / 'prep')
    assert all(r['file'] == r['segment_file'] and r['raw_file'] == f"raw/{r['image_id']}.tif" and r['shape'][0] == 2 for r in rows)
    assert data.variant(tmp_path / 'prep')['variant'] == 'ssl_two_channel'
    config = dict(dataset='co_fc_timecourse', variant='ssl_two_channel')
    spec = pipeline.dataset(config)
    assert spec.channels == ('PhaseContrast', 'GFP') and 'two-channel' in spec.tasks['embed'] and 'GFP pixels' not in spec.withheld.split('.')[0]
    assert spec_module.for_config(dict(dataset='co_fc_timecourse')).channels == ('PhaseContrast',)
    # the decision image is the phase channel alone; the raw stack is phase + GFP, pixel-identical to the sources
    row = rows[0]
    phase = tifffile.imread(tmp_path / 'prep' / 'public' / row['file'])
    raw = pipeline.raw_stack(tmp_path / 'prep' / 'public' / row['raw_file'], spec)
    gfp = tifffile.imread(tmp_path / 'prep' / 'private' / private[row['image_id']]['gfp_file'])
    assert phase.ndim == 2 and raw.shape == (2, *SMALL) and np.array_equal(raw[0], phase) and np.array_equal(raw[1], gfp)
    # a crop box chosen on the phase image cuts both channels at identical coordinates
    mask = np.zeros(SMALL, dtype=np.int32)
    mask[2:14, 4:16] = 1
    crops = [dict(cell_id='c0', crop_bbox=[2, 4, 14, 16], original_bbox=[2, 4, 14, 16], crop_path='display.png', area=int(mask.sum()))]
    kept, excluded = pipeline.retained_crops(raw, row, crops, set(), tmp_path / 'crop', spec, mask)
    assert len(kept) == 1 and not excluded and kept[0]['mask_label'] == 1
    crop = tifffile.imread(kept[0]['crop_path'])
    assert crop.shape[0] == 2 and crop.dtype == np.uint16 and np.array_equal(crop[0], phase[2:14, 4:16]) and np.array_equal(crop[1], gfp[2:14, 4:16])
    # a phase-only preparation has no raw stack and reads as the phase_only variant
    phase_only = data.prepare(source, tmp_path / 'prep_phase')
    assert phase_only['variant'] == 'phase_only' and 'raw_file' not in data.load_prepared(tmp_path / 'prep_phase')[0]


def test_prepare_contract_cv(tmp_path, small_images):
    source = synthetic_source(tmp_path / 'source')
    record = data.prepare(source, tmp_path / 'prep', protocol=data.CV_PROTOCOL)
    assert record['images'] == 184 and record['wells'] == 23
    assert record['wells_by_condition'] == {'Control': 12, 'FC': 11}
    assert record['excluded_wells'] == {'F6': {'days': [4], 'reason': 'incomplete time course'}}
    rows = data.load_prepared(tmp_path / 'prep', verify_pixels=True)
    private = data.private_images(tmp_path / 'prep')
    splits = json.loads((tmp_path / 'prep' / 'private' / 'splits.json').read_text())
    # the public table carries no condition, day or source name
    for r in rows:
        assert set(r) & {'condition', 'day', 'source_well', 'timepoint'} == set()
        assert r['image_id'].startswith('i') and r['well'].startswith('w')
    # the SSL split is by well: the four named wells validate, every other well trains, nothing is held out
    by_usage = {}
    for well, info in splits['wells'].items():
        by_usage.setdefault((info['condition'], info['usage']), []).append(well)
    assert {k: len(v) for k, v in by_usage.items()} == {
        ('Control', 'development_train'): 10, ('Control', 'development_val'): 2,
        ('FC', 'development_train'): 9, ('FC', 'development_val'): 2}
    assert {r['usage'] for r in rows} == {'development_train', 'development_val'}
    assert {private[r['image_id']]['source_well'] for r in rows if r['usage'] == 'development_val'} == {'C6', 'D6', 'F4', 'G3'}
    for r in rows:
        assert private[r['image_id']]['usage'] == r['usage'] == splits['wells'][r['well']]['usage']
    # the cross-validation folds are the plate columns, private, and every well is in exactly one
    assert splits['cv_fold_sizes'] == {'1': 6, '2': 6, '3': 6, '4': 5}
    for r in rows:
        info = private[r['image_id']]
        assert info['cv_fold'] == int(info['source_well'][1:]) - 2 == splits['wells'][r['well']]['cv_fold']
        assert 'cv_fold' not in r
    # every well has eight time points and the pseudonyms do not follow the plate order
    wells = {}
    for r in rows:
        wells.setdefault(r['well'], set()).add(private[r['image_id']]['day'])
    assert all(days == set(data.DAYS.values()) for days in wells.values())
    control_ids = sorted(w for w, i in splits['wells'].items() if i['condition'] == 'Control')
    assert control_ids != [f'w{i:02d}' for i in range(12)]
    # the GFP threshold is the 99th percentile of baseline GFP
    assert data.gfp_threshold(tmp_path / 'prep') > 5000
    smoke = data.smoke_images(rows, private)
    assert len(smoke) == 32
    assert {private[r['image_id']]['cv_fold'] for r in smoke} == {1, 2, 3, 4} and len({r['well'] for r in smoke}) == 4
    assert {private[r['image_id']]['condition'] for r in smoke} == {'Control', 'FC'}
    assert {private[r['image_id']]['day'] for r in smoke} == set(data.DAYS.values())
    assert len({r['well'] for r in smoke if r['usage'] == 'development_val'}) == 1
    # a second prepare into the same folder is refused; a changed table is detected
    with pytest.raises(FileExistsError):
        data.prepare(source, tmp_path / 'prep', protocol=data.CV_PROTOCOL)
    table = tmp_path / 'prep' / 'public' / 'images.json'
    table.write_text(table.read_text().replace('"usage": "development_val"', '"usage": "development_train"', 1))
    with pytest.raises(ValueError):
        data.load_prepared(tmp_path / 'prep')


def test_predict_and_score_cv(tmp_path, small_images):
    source = synthetic_source(tmp_path / 'source')
    data.prepare(source, tmp_path / 'prep', protocol=data.CV_PROTOCOL)
    prep = tmp_path / 'prep'
    rows = data.load_prepared(prep)
    private = data.private_images(prep)
    run = tmp_path / 'run'
    _fake_run(prep, run, rows, private)
    record = score.predict(prep, run)
    assert record['primary'] == 'day_classifier' and [f['fold'] for f in record['folds']] == [1, 2, 3, 4]
    day = record['folds'][0]['day_classifier']
    assert day['fit_split'] == 'development_train' and day['C'] in score.C_GRID and day['classes'] == score.DAYS
    assert record['folds'][0]['condition_probe']['selection_split'] == 'development_val'
    # per fold: test, validation and training wells are disjoint, the validation fold is the next one (cyclic), and
    # the test wells of the four folds are the 23 wells, once each
    for f in record['folds']:
        parts = [set(f['test_wells']), set(f['validation_wells']), set(f['training_wells'])]
        assert not (parts[0] & parts[1] or parts[0] & parts[2] or parts[1] & parts[2])
        assert f['day_classifier']['training_wells'] == f['training_wells']
    assert sorted(w for f in record['folds'] for w in f['test_wells']) == sorted({r['well'] for r in rows})
    assert {r['fold']: len(r['test_wells']) for r in record['folds']} == {1: 6, 2: 6, 3: 6, 4: 5}
    fold_of_well = {r['well']: private[r['image_id']]['cv_fold'] for r in rows}
    for f in record['folds']:
        assert {fold_of_well[w] for w in f['validation_wells']} == {f['fold'] % 4 + 1}
    predictions = data.read_json(run / 'predictions.json')
    assert len(predictions) == len(data.read_json(run / 'cells.json'))
    assert {p['usage'] for p in predictions} == {'evaluation_only'} and {p['cv_fold'] for p in predictions} == {1, 2, 3, 4}
    assert all(len(p['day_class_probabilities']) == 8 and p['day_class_prediction'] in score.DAYS for p in predictions)
    metrics = score.score(prep, run)
    p = metrics['prediction']
    assert p['heldout_wells'] == 23 and p['heldout_images'] == 184
    d = p['day_image_level']
    assert d['n'] == 184 and d['accuracy'] > 0.3 and d['within_one_timepoint'] > d['accuracy'] and d['chance_accuracy'] == 0.125
    assert np.asarray(d['confusion']['matrix']).sum() == 184 and len(d['confusion']['matrix']) == 8
    assert set(p['day_image_level_by_condition']) == {'Control', 'FC'}
    assert p['day_regression']['spearman_rho'] > 0.5
    assert p['condition_image_level']['accuracy'] > 0.9 and p['condition_crop_level']['n'] > 184
    assert p['bootstrap']['unit'] == 'well (out of fold)' and 'day_accuracy' in p['bootstrap']['image_level']
    gfp = metrics['gfp']
    assert len(gfp['by_condition_day']) == 16 and gfp['target'] == score.GFP_TARGET
    assert gfp['by_condition_day']['FC|0']['gfp_positive_fraction'] > gfp['by_condition_day']['Control|0']['gfp_positive_fraction']
    assert gfp['by_condition_day']['FC|0']['gfp_integrated_bgsub'] > gfp['by_condition_day']['Control|0']['gfp_integrated_bgsub']
    # the GFP probe and its label-only reference are scored on held-out images and bootstrapped over wells
    assert record['folds'][0]['gfp_probe']['target'] == score.GFP_TARGET
    assert record['folds'][0]['gfp_probe']['fit_split'] == 'development_train'
    assert record['folds'][0]['gfp_label_baseline']['n_parameters'] == 9
    g = p['gfp_image_level']
    assert g['n'] == 184 and {'spearman_rho', 'mae', 'baseline_spearman_rho', 'baseline_mae'} <= set(g)
    assert 'gfp_spearman_rho' in p['bootstrap']['image_level']
    assert all('gfp_prediction' in q and 'gfp_label_baseline' in q for q in predictions)
    table = pd.read_csv(run / 'per_object_gfp.csv')
    assert {'gfp_integrated_bgsub', 'gfp_background_mean', 'gfp_prediction', 'gfp_label_baseline'} <= set(table.columns)
    assert np.allclose(table.gfp_integrated_bgsub, table.gfp_sum - table.area * table.gfp_background_mean)
    assert metrics['segmentation']['images_with_objects'] == 184
    assert (run / 'per_image.csv').is_file() and (run / 'per_object_gfp.csv').is_file()
    # a fold's own labels never enter fitting: the classifier records name development splits only
    assert day['selection_split'] == 'development_val' and day['refit_on_validation'] is False
    assert data.read_json(run / 'PREDICTIONS_COMPLETE.json')['labels_of_a_fold_used_to_score_it'] is False
    # a changed prediction table is refused by the scorer
    (run / 'predictions.json').write_text('[]')
    with pytest.raises(ValueError):
        score.score(prep, run)


def test_cv_fold_predictions_do_not_depend_on_their_own_labels(tmp_path, small_images, monkeypatch):
    """Scrambling the labels of fold 1 changes the other folds' fits but must leave fold 1's own predictions identical."""
    source = synthetic_source(tmp_path / 'source')
    data.prepare(source, tmp_path / 'prep', protocol=data.CV_PROTOCOL)
    prep = tmp_path / 'prep'
    rows = data.load_prepared(prep)
    private = data.private_images(prep)
    run_a, run_b = tmp_path / 'run_a', tmp_path / 'run_b'
    _fake_run(prep, run_a, rows, private)
    _fake_run(prep, run_b, rows, private)
    score.predict(prep, run_a)
    scrambled = {k: dict(v) for k, v in private.items()}
    for v in scrambled.values():
        if v['cv_fold'] == 1:
            v['day'] = 16 - v['day'] if v['day'] in (0, 16) else 0
            v['condition'] = 'Control' if v['condition'] == 'FC' else 'FC'
    monkeypatch.setattr(data, 'private_images', lambda _prep: scrambled)
    score.predict(prep, run_b)
    a = {p['cell_id']: p for p in data.read_json(run_a / 'predictions.json')}
    b = {p['cell_id']: p for p in data.read_json(run_b / 'predictions.json')}
    own = [k for k in a if a[k]['cv_fold'] == 1]
    other = [k for k in a if a[k]['cv_fold'] != 1]
    assert own and all(a[k]['day_class_probabilities'] == b[k]['day_class_probabilities']
                       and a[k]['fc_probability'] == b[k]['fc_probability']
                       and a[k]['day_regression'] == b[k]['day_regression'] for k in own)
    assert any(a[k]['day_class_probabilities'] != b[k]['day_class_probabilities'] for k in other)


def test_cv_smoke_subset_scores_end_to_end(tmp_path, small_images):
    """The thirty-two smoke images run through the cross-validated predict and score without error: each outer fold
    has a test, a validation and training wells, and both conditions in training."""
    source = synthetic_source(tmp_path / 'source')
    data.prepare(source, tmp_path / 'prep', protocol=data.CV_PROTOCOL)
    prep = tmp_path / 'prep'
    rows = data.smoke_images(data.load_prepared(prep), data.private_images(prep))
    private = data.private_images(prep)
    run = tmp_path / 'run'
    _fake_run(prep, run, rows, private)
    record = score.predict(prep, run)
    assert [len(f['test_wells']) for f in record['folds']] == [1, 1, 1, 1]
    metrics = score.score(prep, run)
    assert metrics['images'] == 32 and metrics['prediction']['heldout_wells'] == 4
    assert metrics['prediction']['day_image_level']['n'] == 32


def test_organoid_segmentation_cohort_regime(tmp_path, monkeypatch):
    """segmentation_regime cohort with the organoid segmenter: SHAPE's segmentation cohort runs with the organoid tool
    name and its fixed kwargs, a calibration sample has its own VRR, the rest the cohort mode; the failed visual QC of
    the call at the defaults keeps its mask and is recorded. Control (no reasoner) stays per image."""
    from dataclasses import replace
    from types import SimpleNamespace
    import threading
    import benchmarks.full_HCMA.harness.shape_arm as hcma
    from benchmarks.full_CPJUMP1.harness import pipeline
    from shape.agent.verification import segmentation_cohort

    spec = replace(spec_module.CO_FC, image_shape=tuple(SMALL))
    monkeypatch.setattr(pipeline, 'dataset', lambda config: spec)
    prep, run = tmp_path / 'prep', tmp_path / 'run'
    (prep / 'public').mkdir(parents=True)
    (run / 'logs').mkdir(parents=True)
    rows = []
    for i in range(12):
        name = f'img_{i:02d}.tif'
        tifffile.imwrite(prep / 'public' / name, np.zeros(SMALL, np.uint16))
        rows.append({'image_id': f'img_{i:02d}', 'segment_file': name})
    ids = [r['image_id'] for r in rows]
    key = ('cardiac organoid', 'phase contrast')
    sample = segmentation_cohort.draw(ids, segmentation_cohort.calibration_size(12), key)

    class FakeOrganoid:
        device = type('Device', (), {'type': 'cuda'})()
        fixed_seen = []

        def execute(self, **kwargs):
            self.fixed_seen.append(kwargs.get('adaptive_rescue'))
            mask = Path(kwargs['query_cache_dir']) / 'mask.npy'
            mask.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask, np.ones(SMALL, np.int32))
            out = {'mask_path': str(mask), 'overlay_path': None, 'organoid_count': 1}
            if kwargs['image_id'] == 'img_00':
                out['error'] = 'Segmentation output failed visual QC: organoids=1'
            return out

    tool = FakeOrganoid()
    defaults = {'flow_threshold': 0.4, 'min_size': 15}
    monkeypatch.setattr(pipeline, 'segmenter', lambda spec_, tool_params=None: {
        'tool': tool, 'name': 'Organoid_Segmenter_Tool', 'fixed': {'adaptive_rescue': False},
        'registry': {'flow_threshold': None, 'min_size': None}, 'defaults': dict(defaults)})
    calls, lock = [], threading.Lock()

    def segment_resolution(C, tool_, item, levers, defaults_, engine, scratch, log_dir, model_lock,
                           verifier_options=None, tool_name=None, fixed=None):
        options = verifier_options or {}
        mode = 'cohort' if 'cohort_levers' in options else 'routed' if 'start_params' in options else 'search'
        with lock:
            calls.append((item['image_id'], mode, tool_name, dict(fixed)))
        changed = {'flow_threshold': 0.5} if mode != 'cohort' else dict(options['cohort_levers'])
        mask = Path(scratch) / 'selected.npy'
        mask.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask, np.full(SMALL, 2, np.int32))
        res = SimpleNamespace(details={'admission': {'decision': 'RUN'}, 'call': 'c1', 'measured_problems': {'c1': []}},
                              params={**defaults_, **changed}, result={'mask_path': str(mask)}, status='satisfied',
                              reason='satisfied', fault=None)
        record = {'status': 'searched', 'levers': changed, 'mask_npy': str(mask), 'n_calls': 1,
                  'reasoner_failures': 0, 'fallback_rounds': 0, 'execution_errors': [], 'stop_reason': 'satisfied'}
        return record, res

    monkeypatch.setattr(hcma, 'segment_resolution', segment_resolution)
    monkeypatch.setattr(segmentation_cohort, 'preflight_read', lambda path, engine, image_id=None: SimpleNamespace(
        specimen_context=key[0], modality=key[1], confidence='high'))
    monkeypatch.setattr(pipeline, 'engine_for', lambda config, stage: object())
    config = {'arm': 'shape', 'dataset': 'co_fc_timecourse', 'deadline': None, 'cores': 4, 'segmentation_regime': 'cohort',
              'smoke_policy': {'shape_search_images': 12, 'shape_search_rounds': 2}}      # a smoke does not force per image
    pipeline.segmentation(prep, run, config, rows)
    cohort = data.read_json(run / 'segment_COHORT.json')
    assert cohort['tool'] == 'Organoid_Segmenter_Tool' and cohort['groups'][0]['calibration'] == sample
    assert {c[2] for c in calls} == {'Organoid_Segmenter_Tool'} and {tuple(c[3].items()) for c in calls} == {(('adaptive_rescue', False),)}
    assert {c[0] for c in calls if c[1] == 'cohort'} == set(ids) - set(sample)
    assert {c[0] for c in calls if c[1] == 'search'} == set(sample)
    assert set(tool.fixed_seen) == {False}
    outputs = {r['image_id']: r for r in data.read_json(run / 'segmentation.json')}
    assert set(outputs) == set(ids) and all(Path(r['mask']).is_file() for r in outputs.values())
    assert 'failed visual QC' in outputs['img_00']['defaults_qc_error']
    assert [i for i, r in outputs.items() if 'defaults_qc_error' in r] == ['img_00']
    assert {r['parameter_source'] for r in outputs.values()} == {'calibration', 'cohort'}
    # control and codex (no SHAPE reasoner) never take the cohort path
    taken = []
    monkeypatch.setattr(pipeline, 'engine_for', lambda config, stage: None)
    monkeypatch.setattr(pipeline, '_segmentation_cohort_organoid', lambda *a: taken.append('cohort'))
    monkeypatch.setattr(pipeline, '_segmentation_per_image', lambda *a: taken.append('per_image'))
    pipeline.segmentation(prep, run, dict(config, arm='control'), rows)
    assert taken == ['per_image']


def test_protocols_are_selected_by_the_prepared_data(tmp_path, small_images):
    """The default protocol is the well-stratified held-out split; the cross-validation protocol is opt-in, has no
    two-channel variant, and its agent wording differs (no held-out wells in the SSL stage)."""
    source = synthetic_source(tmp_path / 'source')
    assert data.prepare(source, tmp_path / 'prep')['protocol'] == data.PROTOCOL != data.CV_PROTOCOL
    assert data.protocol_of(tmp_path / 'prep') == data.PROTOCOL
    assert {r['usage'] for r in data.load_prepared(tmp_path / 'prep')} == set(data.USAGES)
    with pytest.raises(ValueError):
        data.prepare(source, tmp_path / 'prep_cv_two', ssl_channels=data.TWO_CHANNEL, protocol=data.CV_PROTOCOL)
    with pytest.raises(ValueError):
        data.prepare(source, tmp_path / 'prep_unknown', protocol='no-such-protocol')
    assert data.prepare(source, tmp_path / 'prep_cv', protocol=data.CV_PROTOCOL)['protocol'] == data.CV_PROTOCOL
    assert data.protocol_of(tmp_path / 'prep_cv') == data.CV_PROTOCOL
    held = spec_module.for_config(dict(dataset='co_fc_timecourse'))
    cv = spec_module.for_config(dict(dataset='co_fc_timecourse', protocol=data.CV_PROTOCOL))
    assert held.tasks['embed'] != cv.tasks['embed'] and 'held-out' in held.tasks['embed'] and 'held-out' not in cv.tasks['embed']
    assert 'cross-validation folds' in cv.withheld and cv.usages == held.usages

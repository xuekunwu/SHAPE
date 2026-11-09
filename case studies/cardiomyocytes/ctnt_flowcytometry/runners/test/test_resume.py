"""Recovery integrity and genuinely unlimited arm execution."""
import importlib
reasoning=importlib.import_module('case studies.cardiomyocytes.ctnt_flowcytometry.runners.runtime.reasoning')
from pathlib import Path
import sys

import numpy as np
import pytest
import tifffile

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
PKG = "case studies.cardiomyocytes.ctnt_flowcytometry.runners"
recovery = importlib.import_module(PKG+".workflows.recover_shape_crop")
data = recovery.data


@pytest.fixture
def crop(tmp_path):
    prep, run = tmp_path/'prep', tmp_path/'run'
    (prep/'public').mkdir(parents=True)
    folder = run/'crop'/'i1'
    (folder/'raw_crops').mkdir(parents=True)
    raw = np.arange(16, dtype=np.uint16).reshape(4, 4)
    tifffile.imwrite(prep/'public'/'i1.tif', raw)
    mask = np.zeros((4, 4), dtype=np.uint16)
    mask[:2, :2] = 1
    tifffile.imwrite(run/'mask.tif', mask)
    raw_path = folder/'raw_crops'/'cell_000000.ome.tif'
    tifffile.imwrite(raw_path, raw[:2, :2])
    tifffile.imwrite(folder/'display.tif', np.zeros((2, 2), dtype=np.uint8))
    cells = [{"cell_id":"c1", "crop_bbox":[0,0,2,2], "original_bbox":[0,0,2,2], "area":4,
              "crop_path":str(folder/'display.tif')}]
    data.write_json(folder/'metadata.json', {"cell_metadata":cells})
    data.write_json(folder/'result.json', {"cell_crops_metadata_path":str(folder/'metadata.json')})
    data.write_json(folder/'SEARCH.json', {"status":"searched", "boxes":{"c1":[0,0,2,2]},
                    "rejected_tool_cells":[], "fallback_rounds":0, "reasoner_failures":0, "execution_errors":[]})
    row = dict(image_id="i1", file="i1.tif", plate="p1", well="w1", usage="development_train")
    return prep, run, {"dataset":"ctnt_flowcytometry"}, row, {
        "mask":str(run/'mask.tif'), "sha256":data.digest(run/'mask.tif')}


def test_reuse_is_read_only_and_rejects_changed_pixels(crop):
    prep, run, *_ = crop
    before = {str(p):data.digest(p) for p in run.rglob('*') if p.is_file()}
    cells, excluded = recovery.recover_image(*crop)
    assert len(cells) == 1 and cells[0]['mask_label'] == 1 and not excluded
    assert before == {str(p):data.digest(p) for p in run.rglob('*') if p.is_file()}
    tifffile.imwrite(Path(cells[0]['crop_path']), np.zeros((2, 2), dtype=np.uint16))
    with pytest.raises(ValueError, match="raw pixels differ"):
        recovery.recover_image(*crop)


def test_reuse_rejects_changed_selection_or_mask(crop):
    prep, run, *_ = crop
    path = run/'crop'/'i1'/'SEARCH.json'
    record = data.read_json(path)
    record['boxes']['c1'] = [1,1,3,3]
    data.write_json(path, record)
    with pytest.raises(ValueError, match="boxes differ"):
        recovery.recover_image(*crop)
    record['boxes']['c1'] = [0,0,2,2]
    data.write_json(path, record)
    tifffile.imwrite(run/'mask.tif', np.ones((4,4), dtype=np.uint16))
    with pytest.raises(ValueError, match="changed segmentation"):
        recovery.recover_image(*crop)


def test_unlimited_cli_is_explicit():
    args = recovery.e2e.parser().parse_args(['run','--prep','p','--out','r','--assets-root','a',
                                           '--expected-source','s','--no-time-limit'])
    assert args.no_time_limit


def test_resumed_comparison_requires_lineage_and_preserves_input_checks(tmp_path):
    from copy import deepcopy
    compare = importlib.import_module(PKG+'.evaluation.arm_comparison')
    from benchmarks.full_CPJUMP1.harness.pipeline import reasoning_record
    protocol = reasoning.reasoning_protocol()
    configs = {a:dict(arm=a, images=['i1'], pilot_sha256='pilot', source_sha256='new',
                      budget_hours=None, deadline=None, reasoning=reasoning_record(a, protocol))
               for a in ('control','shape','codex')}
    configs['control'].update(source_sha256='old', budget_hours=4, deadline=100)
    original = deepcopy(configs['shape'])
    original.update(source_sha256='old', budget_hours=4, deadline=100)
    backup = tmp_path/'attempts'/'old'
    data.write_json(backup/'CONFIG.json', original)
    lineage = dict(policy='resume-crop-no-walltime-v1', source_sha256='new',
                   previous_source_sha256='old', original_budget_hours=4, backup=str(backup),
                   original_artifact_sha256={'CONFIG.json':data.digest(backup/'CONFIG.json')})
    data.write_json(tmp_path/'RESUME.json', lineage)
    assert compare.resumed_configs(configs, tmp_path, 'new') == lineage
    with pytest.raises(ValueError, match='differ'):
        compare.comparable(configs)  # The ordinary matched-budget comparison stays strict.
    configs['codex']['images'] = ['other']
    with pytest.raises(ValueError, match='differ'):
        compare.resumed_configs(configs, tmp_path, 'new')
    configs['codex']['images'] = ['i1']
    data.write_json(backup/'CONFIG.json', {**original, 'images':['changed']})
    with pytest.raises(ValueError, match='evidence changed'):
        compare.resumed_configs(configs, tmp_path, 'new')


def test_codex_restart_lineage_preserves_protocol_and_evidence(tmp_path):
    from copy import deepcopy
    compare = importlib.import_module(PKG+'.evaluation.arm_comparison')
    from benchmarks.full_CPJUMP1.harness.pipeline import reasoning_record
    protocol = reasoning.reasoning_protocol()
    common = dict(images=['i1'], pilot_sha256='pilot', seed=1, model='model', cores=4,
                  train_cells_per_site=10, ssl_epoch_budget=2, protocol='pilot', target='ctnt',
                  source_sha256='shape-source', budget_hours=None, deadline=None)
    configs = {a:dict(common, arm=a, reasoning=reasoning_record(a, protocol))
               for a in ('control','shape','codex')}
    configs['control'].update(source_sha256='original', budget_hours=4, deadline=100)
    original = dict(configs['shape'], source_sha256='original', budget_hours=4, deadline=100)
    shape_run, codex_run = tmp_path/'shape', tmp_path/'codex'
    backup = shape_run/'attempts'/'original'
    data.write_json(backup/'CONFIG.json', original)
    lineage = dict(policy='resume-crop-no-walltime-v1', source_sha256='shape-source',
                   previous_source_sha256='original', original_budget_hours=4, backup=str(backup),
                   original_artifact_sha256={'CONFIG.json':data.digest(backup/'CONFIG.json')})
    data.write_json(shape_run/'RESUME.json', lineage)
    old_codex = deepcopy(configs['codex'])
    prior = codex_run/'attempts'/'failed'
    data.write_json(prior/'CONFIG.json', old_codex)
    restart = dict(policy='codex-crop-input-bound-v1', source_sha256='fixed-source',
                   previous_source_sha256='shape-source', backup=str(prior),
                   original_artifact_sha256={'CONFIG.json':data.digest(prior/'CONFIG.json')})
    data.write_json(codex_run/'RESUME.json', restart)
    configs['codex']['source_sha256'] = 'fixed-source'
    assert compare.resumed_configs(configs, shape_run, 'fixed-source', codex_run)['codex_restart'] == restart
    configs['codex']['seed'] = 2
    with pytest.raises(ValueError, match='protocol changed'):
        compare.resumed_configs(configs, shape_run, 'fixed-source', codex_run)
    configs['codex']['seed'] = 1
    data.write_json(prior/'CONFIG.json', dict(old_codex, seed=2))
    with pytest.raises(ValueError, match='evidence changed'):
        compare.resumed_configs(configs, shape_run, 'fixed-source', codex_run)

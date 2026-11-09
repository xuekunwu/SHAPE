"""Shared process execution, failure propagation, scope and scientific boundaries."""
import ast
import importlib
import json
from pathlib import Path
import sys
import pytest

PKG='case studies.cardiomyocytes.ctnt_flowcytometry.runners'
runner=importlib.import_module(PKG+'.runtime.stage_runner')
sequence=importlib.import_module(PKG+'.workflows.stage_sequence')
progress=importlib.import_module(PKG+'.runtime.suite_progress')
data=importlib.import_module(PKG+'.dataset.preparation')


def test_real_worker_requires_a_fresh_completion_proof(tmp_path):
    run=tmp_path/'run';(run/'logs').mkdir(parents=True)
    marker=run/'predict_COMPLETE.json'
    payload=json.dumps(dict(status='complete',stage='predict'))
    code=f'from pathlib import Path; Path({str(marker)!r}).write_text({payload!r})'
    result=runner.invoke('unused','predict',tmp_path,run,command=[sys.executable,'-c',code])
    assert result['status']=='complete'
    before=marker.read_bytes()
    with pytest.raises(FileExistsError):runner.invoke('unused','predict',tmp_path,run,command=[sys.executable,'-c','raise RuntimeError()'])
    assert marker.read_bytes()==before


def test_successful_shell_without_scientific_marker_is_not_success(tmp_path):
    (tmp_path/'logs').mkdir()
    with pytest.raises(FileNotFoundError):
        runner.invoke('unused','crop',tmp_path,tmp_path,command=[sys.executable,'-c','print("no delivery")'])


def test_stage_failure_preserves_completed_stages_and_stops_following_work(tmp_path,monkeypatch):
    cli=importlib.import_module('benchmarks.full_CPJUMP1.harness.cli');calls=[]
    def child(command,log,timeout,**kwargs):
        stage=command[4];calls.append(stage)
        if stage=='crop':raise RuntimeError('controlled worker failure')
        data.write_json(tmp_path/(stage+'_COMPLETE.json'),dict(status='complete',stage=stage))
    monkeypatch.setattr(cli,'child',child)
    state={};config=dict(arm='control',deadline=None,reasoning=dict(protocol={}))
    with pytest.raises(RuntimeError,match='controlled worker failure'):
        sequence.run_stages(('segment','crop','embed'),'worker',tmp_path,tmp_path,config,state)
    assert calls==['segment','crop']
    assert list(state['stages'])==['segment'] and state['current_stage']=='crop'
    assert not (tmp_path/'embed_COMPLETE.json').exists()


def test_progress_never_confuses_complete_initial_masks_with_finished_segmentation(tmp_path):
    data.write_json(tmp_path/'CONFIG.json',dict(images=['i1','i2']))
    data.write_json(tmp_path/'RUN_META.json',dict(status='running',current_stage='segment'))
    for ident in ('i1','i2'):data.write_json(tmp_path/'segment'/ident/'defaults.json',{})
    value=progress.arm_status(tmp_path,'segment')
    assert '2/2' in value and 'VERIFY' in value and 'DONE' not in value
    data.write_json(tmp_path/'segment_COMPLETE.json',dict(status='complete'))
    assert 'DONE' in progress.arm_status(tmp_path,'segment')


def test_predictor_has_no_heldout_evaluation_dependency_or_private_label_reader():
    module=importlib.import_module(PKG+'.prediction.well_predictor')
    tree=ast.parse(Path(module.__file__).read_text(encoding='utf-8'))
    imports=[ast.unparse(n) for n in ast.walk(tree) if isinstance(n,(ast.Import,ast.ImportFrom))]
    assert all('heldout_evaluation' not in value and 'verify_private' not in value for value in imports)
    assert not hasattr(module,'score')
    endpoint=importlib.import_module(PKG+'.evaluation.heldout_evaluation')
    assert not hasattr(endpoint,'fit_predict') and not hasattr(endpoint,'predict')


def test_five_image_smoke_is_label_independent_and_preserves_original_splits():
    smoke=importlib.import_module(PKG+'.workflows.image_stage_smoke')
    rows=[dict(image_id=f'{usage}-{i}',usage=usage,well=f'{usage}-well') for usage in data.USAGES for i in range(8)]
    selected=smoke.select_images(rows,5)
    assert len(selected)==5 and {r['usage'] for r in selected}==set(data.USAGES)
    changed=[dict(r,ctnt_positive_percent=99,quality=1234) for r in reversed(rows)]
    assert [r['image_id'] for r in selected]==[r['image_id'] for r in smoke.select_images(changed,5)]
    assert selected!=smoke.select_images(rows,5,seed=7)

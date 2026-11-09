import importlib
reasoning=importlib.import_module('case studies.cardiomyocytes.ctnt_flowcytometry.runners.runtime.reasoning')
from types import SimpleNamespace
import pytest

module='case studies.cardiomyocytes.ctnt_flowcytometry.runners.'
full=importlib.import_module(module+'workflows.hybrid_upstream')
pilot=importlib.import_module(module+'dataset.sampling')


def test_fresh_config_uses_new_source_and_never_inherits_pilot_or_ssl_choice():
    original=dict(images=['i1'],smoke_policy=dict(shape_search_images=1),pilot_sha256='old',
                  ssl_selection=dict(checkpoint='old_checkpoint'),shape_policy=dict(crop_vrr_workers=1))
    config=full.fresh_config(original,'new_source',12)
    assert config['source_sha256']=='new_source' and config['shape_policy']['crop_vrr_workers']==12
    assert config['ssl_selection']['checkpoint'] is None
    assert config['ssl_search_policy']=='evidence_cost_v1'
    assert '2 SE' in config['ssl_optimization']['stop']
    assert config['search_strategies']==dict(segment='four_stage',crop='legacy',embed='four_stage',batch='four_stage')
    assert config['vrr_environment']['SHAPE_SEGMENTATION_COMPACT_PROMPTS']=='1'
    assert 'smoke_policy' not in config and 'pilot_sha256' not in config
    assert original['ssl_selection']['checkpoint']=='old_checkpoint'


def test_crop_delivery_reuse_requires_explicit_new_config_opt_in():
    reference=dict(images=['i1'],seed=20261015,reuse_verified_crop_delivery=True)
    assert 'reuse_verified_crop_delivery' not in full.fresh_config(reference,'new',2)
    config=full.fresh_config(reference,'new',2,reuse_crop_delivery=True)
    assert config['reuse_verified_crop_delivery'] is True
    assert reference['reuse_verified_crop_delivery'] is True


def test_full_orchestration_runs_fresh_stages_without_scoring_internal_expert(tmp_path,monkeypatch):
    for key,value in full.vrr_environment().items():monkeypatch.setenv(key,value)
    data=full.data;events=[];source='new';reference=tmp_path/'old_shape';reference.mkdir()
    out=tmp_path/'shape_hybrid';out.mkdir();(out/'logs').mkdir()
    config=dict(images=['train','heldout'],model='gpt-6-sol',assets_root='/assets',
                reasoning=dict(protocol=reasoning.reasoning_protocol('all-high')),cores=12)
    data.write_json(reference/'CONFIG.json',config)
    old_files={p.name:p.read_bytes() for p in reference.iterdir()}
    data.write_json(out/'RUN_META.json',dict(status='waiting'))
    prep=tmp_path/'prep';data.write_json(prep/'PREPARE.json',dict(test_fixture=True))
    rows=[dict(image_id='train'),dict(image_id='heldout')]
    monkeypatch.setattr(data,'load_prepared',lambda *a,**k:rows)
    monkeypatch.setattr(full,'verify_source',lambda source:None)
    def preflight(args):events.append('preflight');return dict(source_sha256=source)
    monkeypatch.setattr(full.e2e,'preflight',preflight)
    def canary(prep,parent,folder,cfg):
        events.append('canary');data.write_json(folder/'REPORT.json',dict(status='complete'))
    monkeypatch.setattr(full.vrr_canary,'run',canary)
    def child(command,**kwargs):
        stage=command[4] if command[2]==full.e2e.MODULE else command[command.index('--name')+1]
        events.append(stage)
        assert command[command.index('--run')+1]==str(out/'upstream')
        data.write_json(out/'upstream'/(stage+'_COMPLETE.json'),dict(status='complete',seconds=1,usage={}))
    launcher=importlib.import_module('benchmarks.full_CPJUMP1.harness.cli')
    monkeypatch.setattr(launcher,'child',lambda command,log,timeout,**kwargs: child(command,**kwargs))
    def freeze(args,record):
        events.append('freeze')
        assert args.out==out/'upstream' and not args.smoke
        assert not (args.out/'RESULT.json').exists()
    monkeypatch.setattr(full.e2e,'freeze',freeze)
    readouts=importlib.import_module(module+'evaluation.stage_quality')
    monkeypatch.setattr(readouts,'collect',lambda *args:events.append('readouts'))
    def execute(args):
        events.append('eligible_ssl_entry')
        assert args.parent==out/'upstream' and args.full_run
        assert (args.parent/'UPSTREAM_FROZEN').read_text().strip()==source
        assert not (args.parent/'CTNT_FLOW_COMPLETE').exists()
        assert not (args.parent/'RESULT.json').exists()
    monkeypatch.setattr(full.hybrid,'execute',execute)
    args=SimpleNamespace(parent=reference,out=out,prep=prep,expected_source=source,cores=12)
    full.execute_full(args)
    assert events==['preflight','canary','segment','crop','embed','batch','predict','freeze','readouts','eligible_ssl_entry']
    assert args.parent==reference and old_files=={p.name:p.read_bytes() for p in reference.iterdir()}


def test_unscored_parent_exception_rejects_a_regular_external_run(tmp_path):
    full.data.write_json(tmp_path/'DELIVERY.json',{})
    full.data.write_json(tmp_path/'CONFIG.json',dict(internal_unscored_zero_expert=False))
    (tmp_path/'UPSTREAM_FROZEN').write_text('source')
    with pytest.raises(ValueError,match='restricted'):
        full.hybrid.verify_parent(tmp_path,tmp_path,unscored_internal=True)


def test_adaptive_configuration_is_case_owned_explicit_and_preserves_split_seed():
    reference=dict(seed=20261015,images=['train','val','heldout'],
                   split_sha256='frozen',segmentation_adaptive={'stale':True})
    legacy=full.fresh_config(reference,'snapshot',12)
    adaptive=full.fresh_config(reference,'snapshot',12,adaptive_segmentation=True)
    assert 'segmentation_adaptive' not in legacy
    settings=adaptive['segmentation_adaptive']
    assert settings['native_seed']==reference['seed']
    assert settings['grouping']['batch_fields']==['experiment_date','plate_id']
    assert settings['gain_policy']=='evidence_v1'
    assert adaptive['images']==reference['images'] and adaptive['split_sha256']=='frozen'
    assert reference['segmentation_adaptive']=={'stale':True}

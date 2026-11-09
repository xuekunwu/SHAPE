import importlib

import pytest

module='case studies.cardiomyocytes.ctnt_flowcytometry.runners.'
canary=importlib.import_module(module+'workflows.adaptive_qa')


def test_selection_excludes_earlier_canary_wells_and_all_validation_heldout():
    rows=[dict(image_id=f'{p}_{w}_{i}_{u}',well=f'{p}_{w}',plate_id=p,usage=u)
          for p in ('a','b') for w in range(4) for i in range(5)
          for u in ('development_train','development_val','evaluation_only')]
    chosen,record=canary.select_images(rows)
    reverse,_=canary.select_images(list(reversed(rows)))
    assert chosen==reverse and len(chosen)==20
    assert all(r['usage']=='development_train' and r['well'] not in record['excluded_previous_canary_wells'] for r in chosen)
    assert all(s['included'] for s in record['plate_support'].values())
    changed=[{**r,'ctnt_positive_percent':999,'score':-1} for r in rows]
    assert [r['image_id'] for r in canary.select_images(changed)[0]]==[r['image_id'] for r in chosen]


def test_unsupported_plate_stays_explicitly_excluded():
    rows=[dict(image_id=f'{p}_{w}_{i}',well=f'{p}_{w}',plate_id=p,usage='development_train')
          for p,wells in [('a',4),('b',1)] for w in range(wells) for i in range(5)]
    chosen,record=canary.select_images(rows)
    assert {r['plate_id'] for r in chosen}=={'a'}
    assert record['plate_support']['b']==dict(eligible_images=0,eligible_wells=0,included=False)
    with pytest.raises(ValueError,match='minimum calibration support'):
        canary.select_images([r for r in rows if r['plate_id']=='b'])


def test_failed_or_missing_cohort_and_visual_evidence_cannot_pass_canary(tmp_path):
    assert len(canary.failures(tmp_path,{'a'}))==2
    data=canary.data
    for stage,name in [('segment','COHORT_SEARCH.json'),('crop','SEARCH.json')]:
        data.write_json(tmp_path/stage/'a'/name,dict(reasoner_failures=0,execution_errors=[]))
    assert canary.failures(tmp_path,{'a'})==[]
    data.write_json(tmp_path/'segment'/'triage'/'batch_a'/'RECEIPT.json',dict(error=dict(type='InvalidReply')))
    assert len(canary.failures(tmp_path,{'a'}))==1


def test_real_stage_commands_use_identical_training_manifest_and_separate_outputs(tmp_path,monkeypatch):
    rows=[dict(image_id=f'i{w}_{i}',well=f'w{w}',plate_id='plate',usage='development_train')
          for w in range(4) for i in range(5)]
    monkeypatch.setattr(canary.data,'load_prepared',lambda prep:rows)
    prep=tmp_path/'prep';canary.data.write_json(prep/'public'/'images.json',rows)
    original=tmp_path/'parent';original.mkdir();(original/'KEEP').write_text('immutable')
    calls=[]
    def child(command,**kwargs):
        folder=__import__('pathlib').Path(command[command.index('--run')+1])
        stage=command[4]
        config=canary.data.read_json(folder/'CONFIG.json')
        calls.append((folder.name,stage,config))
        for ident in config['images']:
            canary.data.write_json(folder/stage/ident/'SEARCH.json',dict(reasoner_failures=0))
        canary.data.write_json(folder/(stage+'_COMPLETE.json'),dict(status='complete',seconds=1,usage={'total_tokens':2}))
        canary.data.write_json(folder/'segmentation.json',[dict(n_cells=2)])
        canary.data.write_json(folder/'cells.json',[{}])
    launcher=importlib.import_module('benchmarks.full_CPJUMP1.harness.cli')
    monkeypatch.setattr(launcher,'child',lambda command,log,timeout,**kwargs: child(command,**kwargs))
    protocol=importlib.import_module(module+'runtime.reasoning').reasoning_protocol('all-high')
    config=dict(images=[r['image_id'] for r in rows],source_sha256='source',
                segmentation_adaptive=dict(routing_policy='stratified_triage_v1'),reasoning=dict(protocol=protocol),
                reuse_verified_crop_delivery=True)
    report=canary.run(prep,original,tmp_path/'qa',config)
    assert report['status']=='complete' and not report['targets_used']
    assert [(v,s) for v,s,c in calls]==[('legacy','segment'),('legacy','crop'),('adaptive','segment'),('adaptive','crop')]
    assert len({tuple(c['images']) for v,s,c in calls})==1
    assert all(('segmentation_adaptive' in c)==(v=='adaptive') for v,s,c in calls)
    assert all(('reuse_verified_crop_delivery' in c)==(v=='adaptive') for v,s,c in calls)
    assert list(original.iterdir())==[original/'KEEP']

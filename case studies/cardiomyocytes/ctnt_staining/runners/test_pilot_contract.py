"""Pilot sampling, exploratory scope and eligible-plate branch data isolation."""
import importlib
from collections import Counter
import numpy as np
import pytest

BASE='case studies.cardiomyocytes.ctnt_staining.runners.'
pilot=importlib.import_module(BASE+'pilot')
hybrid=importlib.import_module(BASE+'hybrid')
data=importlib.import_module(BASE+'data')


def test_unreviewed_references_need_explicit_limited_pilot():
    reference={'registration_verified':False,'reference_kind':'fixed_algorithmic_pseudo_reference'}
    with pytest.raises(ValueError):pilot.reference_scope(reference,False,False,{'fraction':.2})
    with pytest.raises(ValueError):pilot.reference_scope(reference,False,True,None)
    with pytest.raises(ValueError):pilot.reference_scope(reference,False,True,{'fraction':1.})
    assert pilot.reference_scope(reference,False,True,{'fraction':.2})=='exploratory20percentpilot'


def test_full_pseudo_reference_scope_needs_its_own_flag_and_no_pilot():
    reference={'registration_verified':False,'reference_kind':'fixed_algorithmic_pseudo_reference'}
    with pytest.raises(ValueError):pilot.reference_scope(reference,False,True,None,False,1.)
    with pytest.raises(ValueError):pilot.reference_scope(reference,False,False,{'fraction':.2},True,1.)
    assert pilot.reference_scope(reference,False,False,None,True,1.)=='exploratory_full_pseudo_reference'
    assert pilot.reference_scope(reference,False,False,None,True,.05)=='exploratory_full_subset_pseudo_reference'


def test_prepare_full_command_prepares_all_wells(tmp_path,monkeypatch):
    e2e=importlib.import_module(BASE+'e2e');seen=[]
    monkeypatch.setattr(pilot,'prepare',lambda raw,out,split,workers,all_wells=False:seen.append(all_wells) or {})
    for command in ('prepare-pilot','prepare-full'):
        e2e.main([command,'--raw',str(tmp_path),'--out',str(tmp_path/command),'--frozen-split',str(tmp_path/'split.json')])
    assert seen==[False,True]


def test_complete_well_sampling_ignores_labels_and_preserves_parent_split():
    records=[{'plate_id':f'260717_Plate_{w%3}','well_id':str(w),'site_id':str(site)}
             for w in range(60) for site in range(1,5)]
    full,split=pilot.metadata_rows(records)
    changed,other=pilot.metadata_rows([{**r,'ctnt_positive':i%2,'donor_id':str(i)} for i,r in enumerate(records)])
    chosen=pilot.select(full,pilot.PILOT_SEED)
    assert chosen==pilot.select(changed,pilot.PILOT_SEED) and split==other
    assert set(Counter(r['well'] for r in chosen).values())=={4}
    assert all(r['usage']==split[r['well']] for r in chosen)


def test_eligible_ssl_never_fits_heldout_or_ineligible_cells_and_keeps_shared_budget(tmp_path,monkeypatch):
    from benchmarks.full_CPJUMP1.harness import pipeline
    rows=[]
    for plate,usage,well in [('p0','development_train','t0'),('p0','development_val','v0'),
            ('p0','development_val','v1'),('p0','evaluation_only','h0'),
            ('p1','development_train','t1'),('p1','development_val','v2')]:
        rows.append(dict(plate_id=plate,plate=plate,usage=usage,well=well,image_id=well))
    cells=[dict(cell_id=r['image_id'],plate=r['plate'],well=r['well'],usage=r['usage']) for r in rows]
    data.write_json(tmp_path/'cells.json',cells)
    budget=pipeline.ssl_epoch_budget(rows,128);calls=[]
    checkpoint=tmp_path/'test_checkpoint';checkpoint.write_bytes(b'test checkpoint')
    def fake(prep,run,config,images):
        calls.append((run,config,images))
        if config.get('fit_only'):
            actual=data.read_json(run/'cells.json')
            assert actual and all(c['plate']=='p0' and c['usage']!='evaluation_only' for c in actual)
            assert config['ssl_epoch_budget']==budget and images==rows
            selection={'mode':'frozen_checkpoint','checkpoint':str(checkpoint),'checkpoint_sha256':data.digest(checkpoint)}
        else:
            selection=config['ssl_selection']
            np.save(run/'embedding.npy',np.zeros((len(cells),3),dtype=np.float32))
        data.write_json(run/'SSL_SELECTION.json',selection)
    monkeypatch.setattr(pipeline,'representation',fake)
    hybrid.representation(tmp_path,tmp_path,{'train_cells_per_site':128,'ssl_epoch_budget':budget},rows)
    assert len(calls)==3
    record=data.read_json(tmp_path/'HYBRID_SSL.json')
    assert record['eligible_plates']==['p0'] and record['frozen_finetuned_embedding_available']
    assert not record['heldout_labels_used']

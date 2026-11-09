"""Scientific isolation and plate-hybrid regression checks before the real five-image smoke."""
import importlib
import numpy as np
import pytest

ROOT='case studies.cardiomyocytes.ctnt_staining.runners.'
hybrid=importlib.import_module(ROOT+'hybrid')
score=importlib.import_module(ROOT+'score')
data=importlib.import_module(ROOT+'data')
fluorescence=importlib.import_module(ROOT+'fluorescence')


def fixture():
    cells=[];labels={}
    for plate,nval in [('p0',2),('p1',1)]:
        for i in range(nval):
            ident=f'{plate}:{i}'
            cells.append(dict(cell_id=ident,plate=plate,well=ident,usage='development_val'))
            labels[ident]=dict(ctnt_positive=i%2,usage='development_val')
    cells.append(dict(cell_id='test',plate='p0',well='heldout',usage='evaluation_only'))
    eligibility={'p0':{'eligible':True},'p1':{'eligible':False}}
    return cells,labels,eligibility


def test_plate_routes_only_eligible_improving_validation():
    cells,labels,e=fixture()
    routes=hybrid.choose_routes(cells,labels,np.array([.4,.6,.4,.5]),np.array([.1,.9,.1,1.]),e)
    assert routes['p0']['branch']=='finetuned'
    assert routes['p1']['branch']=='zero_shot'
    # Changes to held-out predictions cannot affect routing.
    changed=hybrid.choose_routes(cells,labels,np.array([.4,.6,.4,1.]),np.array([.1,.9,.1,0.]),e)
    assert routes==changed


@pytest.mark.parametrize('fine',[None,np.array([.4,.6,.4,.9]),np.array([.49,.51,.4,.1])])
def test_missing_tied_or_worse_fine_keeps_zero(fine):
    cells,labels,e=fixture()
    assert hybrid.choose_routes(cells,labels,np.array([.4,.6,.4,.5]),fine,e)['p0']['branch']=='zero_shot'


def test_heldout_labels_rejected():
    cells,labels,e=fixture();labels['test']={'ctnt_positive':1,'usage':'evaluation_only'}
    with pytest.raises(ValueError,match='held-out'):
        hybrid.choose_routes(cells,labels,np.array([.4,.6,.4,.5]),None,e)


def test_per_expert_hard_predictions_are_used():
    value=score.metrics([0,1],[.6,.7],hard_predictions=[0,1])
    assert value['accuracy']==1 and value['sensitivity']==1 and value['specificity']==1


def test_smoke_profiles_allow_one_view_without_changing_normal_contract():
    spec=importlib.import_module(ROOT+'spec')
    assert spec.for_config({'technical_smoke_images':5}).allowed_sites==(1,)
    assert spec.for_config({}).allowed_sites==(2,4)
    assert spec.STAINING.channels==('Brightfield',)


def test_threshold_fits_training_only():
    rng=np.random.default_rng(1);protocol=fluorescence.MeasurementProtocol().identity();records=[]
    for well in ('train1','train2'):
        for i,value in enumerate(np.r_[rng.normal(-4,.3,120),rng.normal(4,.3,120)]):
            records.append(dict(usage='development_train',well=well,qc_status='evaluable',
                measurement_protocol_sha256=protocol,reference_cell_id=f'{well}:{i}',signal=float(value)))
    first=fluorescence.fit_threshold(records,protocol)
    second=fluorescence.fit_threshold(records+[dict(usage='evaluation_only',well='test',qc_status='evaluable',
        measurement_protocol_sha256=protocol,reference_cell_id='test:0',signal=1e12)],protocol)
    assert first==second and first['calibration_wells']==['train1','train2']


def test_fixed_split_ignores_fluorescence_labels():
    records=[{'metadata':{'plate_id':f'260717_Plate_{i%3}','well_id':str(i)}} for i in range(30)]
    a=data.split_wells(records,20261017)
    b=data.split_wells([{**r,'ctnt_positive':i%2} for i,r in enumerate(reversed(records))],20261017)
    assert a==b


def test_hybrid_no_eligible_plate_is_explicit(tmp_path,monkeypatch):
    from benchmarks.full_CPJUMP1.harness import pipeline
    cells=[dict(cell_id='c',plate='p0',well='w0',usage='development_train')]
    data.write_json(tmp_path/'cells.json',cells)
    calls=[]
    def fake(prep,run,config,rows):
        calls.append(config)
        data.write_json(run/'SSL_SELECTION.json',config['ssl_selection'])
    monkeypatch.setattr(pipeline,'representation',fake)
    rows=[dict(plate_id='p0',well='w0',usage='development_train'),dict(plate_id='p0',well='w1',usage='development_val')]
    hybrid.representation(tmp_path,tmp_path,{'train_cells_per_site':128},rows)
    assert len(calls)==1 and calls[0]['ssl_selection']['mode']=='zero_shot'
    result=data.read_json(tmp_path/'HYBRID_SSL.json')
    assert result['eligible_plates']==[] and not result['heldout_labels_used']


def test_object_matching_compatible_bool_assignment_and_fragmentation():
    match=importlib.import_module(ROOT+'matching').match_objects
    reference=np.zeros((12,12),dtype=np.int32);reference[1:5,1:5]=1;reference[7:11,7:11]=2
    assert len(match(reference,reference)['matches'])==2
    fragmented=reference.copy();fragmented[1:3,1:5]=3
    value=match(fragmented,reference,.4)
    assert len(value['matches'])==2 and len(value['unmatched_predicted'])==1
    assert len({r['reference_label'] for r in value['matches']})==2

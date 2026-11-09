import importlib
import numpy as np
import pytest

ROOT='case studies.cardiomyocytes.ctnt_staining.runners.'
hybrid=importlib.import_module(ROOT+'hybrid')
score=importlib.import_module(ROOT+'score')
data=importlib.import_module(ROOT+'data')


def test_mixed_threshold_metrics_use_explicit_decisions():
    m=score.metrics([0,1],[.4,.55],None,hard_predictions=[0,1])
    assert m['accuracy']==1 and m['classifier_probability_threshold'] is None
    with pytest.raises(ValueError,match='explicit'):score.metrics([0,1],[.4,.55],None)


def test_training_execution_is_reported_even_if_its_checkpoint_was_not_adopted(tmp_path):
    data.write_json(tmp_path/'ssl/call_000.params.json',{'adaptation_depth':'zero_shot'})
    data.write_json(tmp_path/'ssl/call_000.result.json',{'epochs_trained':0})
    data.write_json(tmp_path/'ssl/call_001.params.json',{'trainable_pct':25})
    data.write_json(tmp_path/'ssl/call_001.result.json',{'epochs_trained':4,'steps_trained':500,'resume_best':'trained.pt'})
    summary=hybrid.training_summary(tmp_path)
    assert summary['training_attempted'] and summary['training_executed']
    assert summary['calls'][0]['training_requested'] is False


def test_missing_training_telemetry_is_unknown_rather_than_no_training(tmp_path):
    data.write_json(tmp_path/'ssl/call_001.params.json',{'trainable_pct':25})
    assert hybrid.training_summary(tmp_path)['training_executed'] is None


def test_hybrid_summary_reports_actual_validation_and_both_experts(tmp_path,monkeypatch):
    cells=[]
    for i,(usage,well) in enumerate([('development_train','t'),('development_val','v0'),
                                    ('development_val','v1'),('evaluation_only','test')]):
        cells.append(dict(cell_id=str(i),image_id='i'+str(i),plate='p',well=well,usage=usage,mask_label=1))
    data.write_json(tmp_path/'cells.json',cells)
    data.write_json(tmp_path/'segmentation.json',[])
    data.write_json(tmp_path/'CONFIG.json',dict(seed=1))
    data.write_json(tmp_path/'ELIGIBILITY.json',{'p':{'eligible':True}})
    data.write_json(tmp_path/'SSL_SELECTION.json',{'mode':'plate_hybrid'})
    data.write_json(tmp_path/'REFERENCE_COMPLETE.json',{'status':'complete'})
    np.save(tmp_path/'embedding.npy',np.zeros((4,2)))
    fine=tmp_path/'hybrid/finetuned';fine.mkdir(parents=True)
    np.save(fine/'embedding.npy',np.ones((4,2)))
    labels={str(i):dict(ctnt_positive=i%2,usage=c['usage']) for i,c in enumerate(cells) if c['usage']!='evaluation_only'}
    monkeypatch.setattr(score,'matched_development',lambda *args:labels)
    def fit(x,c,labels,seed):
        is_fine=bool(x[0,0]);prob=np.array([.2,.9,.1,.7]) if is_fine else np.array([.2,.4,.8,.7])
        return prob,dict(coef=np.ones((1,2))),dict(C=.01,classifier_probability_threshold=.6 if is_fine else .3,
                                                   validation={'accuracy':.123})
    monkeypatch.setattr(score,'fit_classifier',fit)
    result=hybrid.predict(tmp_path,tmp_path,tmp_path,.5)
    assert result['mode']=='plate_hybrid' and result['validation']['accuracy']==1
    assert result['validation']!=result['zero_shot']['validation']
    assert result['classifier_probability_threshold'] is None
    assert result['thresholds_by_expert']=={'zero_shot':.3,'finetuned':.6}
    selection=data.read_json(tmp_path/'SSL_SELECTION.json')
    assert selection['routing_status']=='frozen' and selection['adopted_finetuned_plates']==['p']
    predictions=data.read_json(tmp_path/'predictions.json')
    assert all(p['prediction_branch']=='finetuned' and p['classifier_probability_threshold']==.6 for p in predictions)

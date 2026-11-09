import importlib
import json
from pathlib import Path

import pytest

PKG='case studies.cardiomyocytes.ctnt_flowcytometry.runners'
report=importlib.import_module(PKG+'.evaluation.historical.six_arm_report')
exporter=importlib.import_module(PKG+'.evaluation.historical.six_arm_export')


def test_export_refuses_incomplete_suite_before_creating_output(tmp_path):
    with pytest.raises(ValueError,match='all six'):
        exporter.export(tmp_path,{'runs':{'control':{}}},tmp_path/'export')
    assert not (tmp_path/'export').exists()


def test_portable_report_pairing_and_tamper_detection(tmp_path):
    arms=exporter.ARMS
    for index,arm in enumerate(arms):
        p=tmp_path/arm
        p.mkdir()
        config={'model':exporter.MODELS[arm],'source_sha256':str(index)*64}
        metric={'mae_percentage_points':1.+index,'rmse_percentage_points':1.+index,
                'bias_percentage_points':1.+index,'r2':None}
        stages={'process':{s:{'seconds':60,'usage':{'total_tokens':100},'reasoning':{}}
                           for s in report.STAGES},
                'segment':{'well_macro':{'cells':1.},'per_image':[{'image_id':'i1','well':'w1','cells':1}]},
                'crop':{'well_macro_quality':{'quality':.8},'per_image':[{'image_id':'i1','well':'w1','retained':1}]},
                'embed':{},'batch':{},'unavailable':{}}
        for name,value in [('CONFIG',config),('RESULT',{'primary':metric,'batch_metrics':{}}),('STAGE_READOUTS',stages)]:
            (p/(name+'.json')).write_text(json.dumps(value))
        (p/'heldout_predictions.csv').write_text('well,ctnt_positive_percent,predicted_ctnt_positive_percent\nw1,50,'+str(51+index)+'\n')
    files={p.relative_to(tmp_path).as_posix():report.digest(p) for p in tmp_path.rglob('*') if p.is_file()}
    (tmp_path/'EXPORT.json').write_text(json.dumps({'arms':arms,'files':files}))
    result=report.reproduce(tmp_path)
    assert result['arms']['biomni']['endpoint']['mae_percentage_points']==5
    assert (tmp_path/'compare'/'paired_per_well.csv').read_text().count('w1')==6
    (tmp_path/'control'/'CONFIG.json').write_text('{}')
    with pytest.raises(ValueError,match='evidence changed'):
        report.reproduce(tmp_path)

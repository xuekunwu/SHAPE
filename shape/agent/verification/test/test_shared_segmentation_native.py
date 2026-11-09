import json

import numpy as np
import pytest
import tifffile
from PIL import Image

from shape.agent.verification.segmentation_shared_native import run_native,recorded_generate


def test_native_adapter_measures_freezes_and_resumes_without_model_or_native_repeat(tmp_path,monkeypatch):
    monkeypatch.setenv('SHAPE_VRR_POLICY','balanced_v1')
    source=tmp_path/'bf.tif';tifffile.imwrite(source,np.random.default_rng(42).integers(0,65535,(128,128),dtype=np.uint16))
    class Tool:
        calls=0
        def execute(self,**p):
            self.calls+=1;mask=np.zeros((128,128),np.int32)
            for i,(r,c) in enumerate([(10,10),(10,50),(50,10),(50,50)],1):mask[r:r+10,c:c+10]=i
            if p['min_size']==15:mask[90,90]=5
            folder=__import__('pathlib').Path(p['query_cache_dir']);folder.mkdir(parents=True,exist_ok=True)
            path=folder/'mask.npy';np.save(path,mask);overlay=folder/'overlay.png'
            Image.fromarray(np.full((128,128,3),120,np.uint8)).save(overlay)
            return dict(status='success',mask_path=str(path),overlay_path=str(overlay),cell_count=int(mask.max()),
                        parameters={'diameter':30},overlay_qc={},flow_error=0.)
    class Engine:
        calls=0
        def generate(self,content,**kwargs):
            self.calls+=1
            assert len(content)>1 and 'Tool contract' in content[0]
            return {'content':dict(reason='remove measured tiny fragments',reviews=[],stop=False,rules=[dict(
                issues=['tiny_mask_fraction'],params_json='{"min_size":30}',
                expected_effect_json='{"tiny_mask_fraction":"down"}',reason='test size prior natively')])}
    tool=Tool();engine=Engine();items=[dict(image_id='a',path=str(source))]
    model_identity={'model':'scripted-test','effort':'high'}
    selected,r=run_native(items,tool,{'min_size':15,'diameter':30},engine,tmp_path/'run',task='whole cells',model_identity=model_identity,max_rounds=2)
    assert tool.calls==2 and engine.calls==1 and r['delivery_summary']['qc_pass']==1
    assert selected['a']['params']['min_size']==30
    selected,resumed=run_native(items,tool,{'min_size':15,'diameter':30},engine,tmp_path/'run',task='whole cells',model_identity=model_identity,max_rounds=2)
    assert tool.calls==2 and engine.calls==1
    with pytest.raises(ValueError,match='inputs or implementation changed'):
        run_native(items,tool,{'min_size':20,'diameter':30},engine,tmp_path/'run',task='whole cells',model_identity=model_identity,max_rounds=2)


def test_smoke_routes_through_the_new_cohort_controller(monkeypatch,tmp_path):
    from benchmarks.full_CPJUMP1.harness import pipeline
    called=[]
    monkeypatch.setattr(pipeline,'engine_for',lambda *args:object())
    monkeypatch.setattr(pipeline,'_segmentation_shared',lambda *args:called.append(args))
    pipeline.segmentation(tmp_path,tmp_path,dict(smoke_policy={'shape_search_rounds':2},
        segmentation_adaptive={'routing_policy':'conditional_cohort_v2'}),[])
    assert len(called)==1


def test_invalid_model_content_is_preserved_before_schema_validation(tmp_path):
    class Engine:
        def generate(self,*a,**kw):return {'content':'not valid JSON','usage':{'total_tokens':7}}
    recorded_generate(Engine(),'public BF facts',[],tmp_path,1,{'model':'scripted'},[])
    receipt=json.loads((tmp_path/'RECEIPT_01.json').read_text())
    assert receipt['raw_answer']=='not valid JSON' and receipt['usage']['total_tokens']==7
    assert receipt['status']=='response_received' and not receipt['decision_reused']


def test_transport_error_gets_a_receipt_and_is_reraised(tmp_path):
    class Engine:
        def generate(self,*a,**kw):raise ConnectionError('test connection failed')
    with pytest.raises(ConnectionError):
        recorded_generate(Engine(),'public BF facts',[],tmp_path,1,{'model':'scripted'},[])
    receipt=json.loads((tmp_path/'RECEIPT_01.json').read_text())
    assert receipt['status']=='request_error' and receipt['error']['type']=='ConnectionError'

"""Adapter -> real cohort -> visual receipt -> real tool-cache integration.

Only native inference and model answers are scripted; no GPU or API is required.
"""
from functools import wraps
import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from . import pipeline


def test_adapter_enables_core_policies_without_dataset_specific_rules(tmp_path, monkeypatch):
    from benchmarks.full_HCMA.harness import shape_arm
    from shape.tools.cell_segmenter import cache_runtime, tool as tool_module
    from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
    original=Cell_Segmenter_Tool._execute_native
    native_calls=[]
    @wraps(original)
    def native(self,**kwargs):
        native_calls.append(kwargs['image_id'])
        folder=__import__('pathlib').Path(kwargs['query_cache_dir'])
        mask=np.zeros((32,32),dtype='uint32');mask[2:6,2:6]=1;mask[14:28,14:28]=2
        np.save(folder/'mask.npy',mask)
        Image.fromarray((mask>0).astype('uint8')*255).save(folder/'overlay.png')
        return dict(status='success',cell_count=2,mask_path=str(folder/'mask.npy'),
                    overlay_path=str(folder/'overlay.png'),seconds=.01)
    monkeypatch.setattr(Cell_Segmenter_Tool,'_execute_native',native)
    checkpoint=tmp_path/'loaded.weights';checkpoint.write_bytes(b'preexisting model weights')
    tool=Cell_Segmenter_Tool.__new__(Cell_Segmenter_Tool)
    tool.device=SimpleNamespace(type='cuda')  # adapter's hardware admission; no CUDA call occurs
    tool.model=SimpleNamespace(pretrained_model=str(checkpoint),net=torch.nn.Linear(1,1))
    monkeypatch.setattr(tool_module,'Cell_Segmenter_Tool',lambda:tool)
    real_configure=cache_runtime.configure
    def configure(tool,root,**kwargs):
        tool.device=torch.device('cpu')
        return real_configure(tool,root,**kwargs)
    monkeypatch.setattr(cache_runtime,'configure',configure)
    monkeypatch.setattr(shape_arm,'tools_common',lambda:SimpleNamespace(
        tool_levers=lambda *a:dict(flow_threshold=.4,min_size=15),normal_default=lambda v:v))
    monkeypatch.setattr(pipeline,'dataset',lambda cfg:SimpleNamespace(
        image_shape=(32,32),allow_empty_masks=False,tasks=dict(segment='Whole-cell instance segmentation.')))
    monkeypatch.setattr(pipeline,'record_search',lambda *a:None)
    options_seen=[]
    def resolution(C,tool,item,levers,defaults,engine,scratch,logdir,lock,verifier_options):
        options_seen.append(verifier_options)
        with lock:
            delivered=tool.execute(image=item['path'],image_id=item['image_id'],query_cache_dir=str(scratch))
        details=dict(call='default',measured_problems=dict(default=['tiny_mask_fraction 0.2 above 0.15']))
        res=SimpleNamespace(status='satisfied',fault=None,params=defaults,result=delivered,details=details)
        return dict(levers={},n_calls=2,mask_npy=None,cell_count=2),res
    monkeypatch.setattr(shape_arm,'segment_resolution',resolution)
    prep=tmp_path/'prep';(prep/'public').mkdir(parents=True)
    images=[]
    for i in range(20):
        name=f'i{i:02}'
        Image.fromarray(np.arange(1024,dtype='uint8').reshape(32,32)).save(prep/'public'/(name+'.png'))
        images.append(dict(image_id=name,segment_file=name+'.png',acquisition='batch_a'))
    class Engine:
        def generate(self,content,**kwargs):
            items=json.loads(content[0].split('Images and recorded alerts:\n')[1])
            return dict(usage=dict(total_tokens=50),content=dict(verdicts=[dict(
                image_id=i['image_id'],action='accept_with_flags',confidence='high',
                image_quality_sufficient=True,normal_heterogeneity=True,visible_faults=[],
                evidence_views=['raw_overview','mask_overlay','size_extreme_zoom'],
                reason='Paired raw, contours and zooms support ordinary size variation.') for i in items]))
    config=dict(model='scripted',cores=3,deadline=None,segmentation_adaptive=dict(
        routing_policy='stratified_triage_v1',gain_policy='evidence_v1',native_seed=20261015,
        grouping=dict(batch_fields=['acquisition'],min_group=10,max_depth=0),triage_batch_size=4))
    out=tmp_path/'run'
    pipeline._segmentation_cohort(prep,out,config,images,Engine())
    cohort=pipeline.read_json(out/'segment_COHORT.json')
    assert cohort['groups'][0]['routing']['accepted_with_recorded_alerts']==17
    assert cohort['counts']==dict(calibration=3,cohort=17)
    assert len(native_calls)==20  # actual second-pass tool calls all reused saved trials
    events=[pipeline.read_json(p) for p in (out/'segment'/'native_trials'/'events').glob('*.json')]
    assert sum(e['hit'] for e in events)==20
    assert all(o['gain_policy']=='evidence_v1' for o in options_seen)
    assert len(pipeline.read_json(out/'segmentation.json'))==20
    assert len(list((out/'segment'/'triage').glob('*/RECEIPT.json')))==5

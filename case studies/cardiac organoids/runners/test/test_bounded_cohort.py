"""Real coordinator/scorer with synthetic inference: resume must not rerun completed predictions."""
from dataclasses import asdict
import json
from pathlib import Path
import sys
from types import SimpleNamespace
import numpy as np
import pytest
sys.path.insert(0,str(Path(__file__).resolve().parents[1]/'segmentation'))
import run_organoid_bounded as runner
import common
from cohort_runtime import digest,atomic_json,file_sha,completed_response
from shape.utils.schemas import StageResolution,VLRRResult

class FakeWorker:
    calls=[];starts=0;interrupt_at=None;timeout_once=False;interrupt_after_complete=False
    def __init__(self,target,config):self.process=None;self.config=config
    def start(self):
        self.process=True;type(self).starts+=1
        return dict(hardware=dict(gpu='synthetic',cuda='synthetic'))
    def stop(self):self.process=None
    def request(self,request,seconds):
        if self.process is None:self.start()
        cls=type(self)
        if cls.interrupt_at==len(cls.calls):
            cls.interrupt_at=None
            raise KeyboardInterrupt('synthetic interruption')
        assert 'label' not in request and 'ground_truth' not in request
        cls.calls.append(request['image_id'])
        folder=Path(request['folder'])
        mask=np.zeros((32,32),dtype=np.int32);mask[8:24,8:24]=1
        path=folder/'prediction.npy';np.save(path,mask)
        result=dict(mask_path=str(path),cell_count=1,organoid_count=1,status='success')
        if cls.timeout_once:
            cls.timeout_once=False
            atomic_json(folder/'candidate.json',dict(result=result,selection_basis='synthetic best QC'))
            return dict(status='TIMEOUT',seconds=.1,error='synthetic timeout')
        res=StageResolution(result=result,status='satisfied',params={},verdict=VLRRResult('PASS',1,{}),
            details=dict(call='default',measured_problems={'default':[]},negative_confirmed=False,qc_pass=True))
        response=dict(status='DONE',vrr_status='satisfied',result=result,resolutions=[asdict(res)],seconds=.1)
        if cls.interrupt_after_complete:
            cls.interrupt_after_complete=False
            completed_response(folder,self.config['output'],response)
            raise KeyboardInterrupt('completed child response before parent acknowledgement')
        return response

@pytest.fixture
def inputs(tmp_path,monkeypatch):
    from PIL import Image
    pairs=[]
    for folder in ('images','labels','weights'):(tmp_path/folder).mkdir()
    for i in range(3):
        image=tmp_path/'images'/f'image{i}.png';Image.fromarray(np.arange(1024,dtype=np.uint8).reshape(32,32)).save(image)
        label=tmp_path/'labels'/f'gt{i}.npy';m=np.zeros((32,32),dtype=np.int32);m[8:24,8:24]=1;np.save(label,m)
        pairs.append(dict(image_id=f'image{i}',image=str(image),label=str(label),shape=[32,32],image_sha256=file_sha(image),label_sha256=file_sha(label)))
    manifest=dict(schema=1,pairs=pairs);manifest['fingerprint']=digest(manifest)
    path=tmp_path/'manifest.json';atomic_json(path,manifest)
    weights=tmp_path/'weights'/'co';weights.write_bytes(b'fixture')
    monkeypatch.setenv('SLURM_JOB_ID','synthetic')
    monkeypatch.setattr(common,'configure',lambda *a: {})
    monkeypatch.setattr(common,'versions',lambda: {})
    monkeypatch.setattr(common,'source_identity',lambda: {'commit':'pin','source_sha256':'synthetic'})
    monkeypatch.setattr(runner.subprocess,'check_output',lambda cmd,**kw:'pin\n' if cmd[1]=='rev-parse' else '')
    from shape.agent.verification import vision_engine
    monkeypatch.setattr(vision_engine,'build_vision_engine',lambda model: object())
    from shape.agent.verification import segmentation_cohort
    monkeypatch.setattr(segmentation_cohort,'preflight_read',lambda *a: SimpleNamespace(confidence='low'))
    monkeypatch.setattr(runner,'IsolatedWorker',FakeWorker)
    FakeWorker.calls=[];FakeWorker.starts=0;FakeWorker.interrupt_at=None;FakeWorker.timeout_once=False;FakeWorker.interrupt_after_complete=False
    argv=['runner','--manifest',str(path),'--model-path',str(weights),'--run-dir',str(tmp_path/'run'),'--expected-commit','pin']
    monkeypatch.setattr(sys,'argv',argv)
    return tmp_path/'run',argv


def test_completed_resume_makes_no_new_worker_or_model_calls(inputs,monkeypatch):
    root,argv=inputs
    assert runner.main()==0
    assert len(FakeWorker.calls)==3
    before=FakeWorker.starts
    exported={name:(root/name).read_bytes() for name in ('shape_cohort_bounded.csv','records.jsonl')}
    monkeypatch.setattr(sys,'argv',argv+['--resume'])
    assert runner.main()==0
    assert len(FakeWorker.calls)==3 and FakeWorker.starts==before
    assert {name:(root/name).read_bytes() for name in exported}==exported
    summary=json.loads((root/'summary.json').read_text())
    assert summary['n_recorded']==3 and summary['n_delivered']==3


def test_interruption_preserves_completed_image_and_reserves_unmeasured_request_budget(inputs,monkeypatch):
    root,argv=inputs;FakeWorker.interrupt_at=1
    with pytest.raises(KeyboardInterrupt):runner.main()
    assert (root/'images/image0/record.provisional.json').exists()
    monkeypatch.setattr(sys,'argv',argv+['--resume'])
    assert runner.main()==2
    assert FakeWorker.calls==['image0','image2']
    records=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    assert len(records)==3 and records[0]['delivered'] and records[2]['delivered']
    assert not records[1]['delivered'] and records[1]['needs_review']
    assert json.loads((root/'summary.json').read_text())['unmeasured_interrupted_passes']==1
    assert records[1]['managed_usage']['tokens']==0
    assert records[1]['budget_consumed_or_reserved']['tokens']==24000


def test_timeout_retains_candidate_with_review_flag_and_is_not_retried_on_resume(inputs,monkeypatch):
    root,argv=inputs;FakeWorker.timeout_once=True
    assert runner.main()==2
    records=[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()]
    assert records[0]['delivered'] and records[0]['needs_review']
    assert records[0]['vrr_status']=='conceded' and not records[0]['operational_success']
    monkeypatch.setattr(sys,'argv',argv+['--resume'])
    assert runner.main()==2
    assert FakeWorker.calls==['image0','image1','image2']


def test_partial_final_audit_is_preserved_and_marked_unknown(tmp_path):
    p=tmp_path/'LLM_CALLS.jsonl'
    p.write_text('{"tokens":17}\n{"tokens":')
    totals=runner.audit_totals(tmp_path)
    assert totals['tokens']==17 and totals['unknown_usage']==1 and totals['incomplete_audit_records']==1
    assert p.read_text().endswith('{"tokens":')


def test_malformed_interior_audit_is_rejected(tmp_path):
    (tmp_path/'LLM_CALLS.jsonl').write_text('broken\n{"tokens":17}\n')
    with pytest.raises(ValueError,match='interior'):runner.audit_totals(tmp_path)


def test_array_task_does_not_read_or_validate_other_shards(inputs,monkeypatch):
    root,argv=inputs
    manifest=json.loads(Path(argv[2]).read_text())
    Path(manifest['pairs'][1]['image']).write_bytes(b'changed unselected image')
    monkeypatch.setattr(sys,'argv',argv+['--shard-size','1','--shard-index','0'])
    assert runner.main()==0 and FakeWorker.calls==['image0']


def test_changed_selected_image_is_rejected_before_gpu_start(inputs):
    root,argv=inputs
    manifest=json.loads(Path(argv[2]).read_text())
    Path(manifest['pairs'][0]['image']).write_bytes(b'changed selected image')
    with pytest.raises(ValueError,match='input changed'):runner.main()
    assert FakeWorker.starts==0


def test_bounded_dispatch_uses_the_same_resume_capable_entrypoint(monkeypatch):
    import run_organoid_cohort
    monkeypatch.setattr(sys,'argv',['legacy','--bounded','--manifest','unused'])
    monkeypatch.setattr(runner,'main',lambda:17)
    assert run_organoid_cohort.main()==17 and '--bounded' not in sys.argv


def test_completed_child_response_is_recovered_without_rerunning_prediction(inputs,monkeypatch):
    root,argv=inputs;FakeWorker.interrupt_after_complete=True
    with pytest.raises(KeyboardInterrupt):runner.main()
    monkeypatch.setattr(sys,'argv',argv+['--resume'])
    assert runner.main()==0
    assert FakeWorker.calls==['image0','image1','image2']
    assert json.loads((root/'summary.json').read_text())['unmeasured_interrupted_passes']==0


def test_parallel_profiles_start_only_one_gpu_worker(inputs,monkeypatch):
    root,argv=inputs
    monkeypatch.setenv('SLURM_CPUS_PER_TASK','4')
    assert runner.main()==0
    assert FakeWorker.starts==1 and FakeWorker.calls==['image0','image1','image2']


def test_export_handles_missing_observed_usage_without_fabricating_cost(tmp_path):
    import csv
    rows=[dict(image_id='first',managed_usage={},budget_consumed_or_reserved={'tokens':24000}),
          dict(image_id='second',managed_usage={'tokens':17},budget_consumed_or_reserved={'tokens':17})]
    runner.export_records(tmp_path,rows)
    with (tmp_path/'shape_cohort_bounded.csv').open() as f:values=list(csv.DictReader(f))
    assert values[0]['managed_tokens']=='' and values[0]['budget_consumed_or_reserved_tokens']=='24000'
    assert values[1]['managed_tokens']=='17'


def test_export_recovery_validates_completed_evidence_without_restarting_inference(inputs,monkeypatch):
    from verify_bounded_exports import verify_run
    root,argv=inputs
    assert runner.main()==0
    before=json.loads((root/'summary.json').read_text())
    receipt=root.parent/'prior-validation.json'
    atomic_json(receipt,dict(source_commit='pin',job_id='synthetic',error='Replay changed exported records',completed_resume_exit_code=0,before_resume_summary=before))
    monkeypatch.setattr(sys,'argv',argv+['--resume']);assert runner.main()==0
    calls,starts=list(FakeWorker.calls),FakeWorker.starts
    report=verify_run(root,Path(argv[2]),receipt,root.parent/'export-recovery',inference_commit='pin',export_commit='fixed-export')
    assert report['technical_gate_passed'] and not report['quality_gate_passed']
    assert report['completed_resume_no_new_model_calls'] and report['new_inference_calls']==0
    assert FakeWorker.calls==calls and FakeWorker.starts==starts
    # Changed legacy CSV values cannot be dismissed as an ordering difference.
    path=root/'shape_cohort_bounded.csv';text=path.read_text();path.write_text(text.replace('image0','altered',1))
    with pytest.raises(ValueError,match='CSV values differ'):
        verify_run(root,Path(argv[2]),receipt,root.parent/'reject-recovery',inference_commit='pin',export_commit='fixed-export')

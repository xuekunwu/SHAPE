#!/usr/bin/env python3
"""Paired opt-in selective-review trial on one fixed manifest and GPU allocation."""
import argparse,csv,hashlib,json,os,subprocess,sys,time
from pathlib import Path
from validate_bounded_stage import check_replay


def sha(path): return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def read(path): return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--run-root',type=Path,required=True)
    p.add_argument('--model-path',type=Path,required=True);p.add_argument('--expected-commit',required=True)
    p.add_argument('--condition',choices=['baseline','selective'],help='Run one condition when an immutable paired reference already exists')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run in a Slurm GPU allocation')
    manifest=read(a.manifest);expected=[x['image_id'] for x in manifest['pairs']]
    receipt=dict(schema=1,job_id=os.environ['SLURM_JOB_ID'],source_commit=a.expected_commit,
        manifest_fingerprint=manifest['fingerprint'],manifest_sha256=sha(a.manifest),n=len(expected),
        scope='Outcome-informed development comparison; not representative whole-dataset validation',
        quality_certified=False,conditions={})
    a.run_root.mkdir(parents=True,exist_ok=True)
    if (a.run_root/'trial_receipt.json').exists():raise ValueError('Use a fresh trial root')
    try:
        for condition in ([a.condition] if a.condition else ['baseline','selective']):
            root=a.run_root/condition
            if root.exists():raise ValueError('Use fresh condition paths')
            cmd=[sys.executable,str(Path(__file__).with_name('run_organoid_bounded.py')),
                '--manifest',str(a.manifest),'--run-dir',str(root),'--model-path',str(a.model_path),
                '--expected-commit',a.expected_commit]
            if condition=='selective':cmd+=['--selective-review']
            started=time.monotonic()
            with (a.run_root/(condition+'.log')).open('x') as f:
                rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
            if rc not in (0,2):raise RuntimeError(condition+' runtime failure; inspect log')
            before=read(root/'summary.json');identity=read(root/'identity.json')
            if before['n_recorded']!=len(expected):raise ValueError('Incomplete fixed cohort')
            for k in ['unknown_usage','incomplete_audit_records','incomplete_tool_requests','unmeasured_interrupted_passes']:
                if before[k]:raise ValueError('Unconfirmed completion or costs: '+k)
            csv_before=(root/'shape_cohort_bounded.csv').read_bytes()
            with (a.run_root/(condition+'.resume.log')).open('x') as f:
                rc_resume=subprocess.run(cmd+['--resume'],stdout=f,stderr=subprocess.STDOUT).returncode
            if rc_resume not in (0,2):raise RuntimeError(condition+' completed replay failure')
            check_replay(before,read(root/'summary.json'),csv_before,(root/'shape_cohort_bounded.csv').read_bytes())
            receipt['conditions'][condition]=dict(inference_exit_code=rc,summary=before,
                csv_sha256=sha(root/'shape_cohort_bounded.csv'),completed_resume_no_new_model_calls=True,
                total_stage_seconds=time.monotonic()-started,hardware=read(root/'sessions.json')[0]['hardware'],
                identity=identity)
        if a.condition:
            receipt['technical_condition_complete']=True
            return
        observations={c:[json.loads(s) for s in (a.run_root/c/'records.jsonl').read_text().splitlines()]
                      for c in receipt['conditions']}
        pairs=[]
        for i,image_id in enumerate(expected):
            left,right=(observations[c][i] for c in ['baseline','selective'])
            for k in ['image_id','image_sha256','ground_truth_sha256','image_height_px','image_width_px','checkpoint_sha256']:
                if left[k]!=right[k]:raise ValueError('Paired image identity differs: '+k)
            if left['image_id']!=image_id:raise ValueError('Condition membership/order differs')
            row=dict(image_id=image_id,baseline_map=left['map_50_95'],selective_map=right['map_50_95'],
                delta_map=right['map_50_95']-left['map_50_95'])
            for c,r in [('baseline',left),('selective',right)]:
                for k in ['delivered','needs_review','vrr_status','predicted_cell_count','ground_truth_cell_count','parameter_source']:
                    row[c+'_'+k]=r[k]
                for k in ['tokens','llm_calls','native_calls']:row[c+'_'+k]=r['managed_usage'].get(k,0)
            pairs.append(row)
        with (a.run_root/'paired.csv').open('x',newline='') as f:
            w=csv.DictWriter(f,fieldnames=list(pairs[0]),lineterminator='\n');w.writeheader();w.writerows(pairs)
        receipt.update(technical_comparison_complete=True,
            wins=sum(r['delta_map']>1e-9 for r in pairs),ties=sum(abs(r['delta_map'])<=1e-9 for r in pairs),
            losses=sum(r['delta_map']< -1e-9 for r in pairs))
    except Exception as exc:
        receipt['error']=str(exc);raise
    finally:
        (a.run_root/'trial_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(json.dumps({c:v['summary'] for c,v in receipt['conditions'].items()}))


if __name__=='__main__':main()

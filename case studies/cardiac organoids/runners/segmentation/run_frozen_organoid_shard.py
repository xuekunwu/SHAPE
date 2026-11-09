#!/usr/bin/env python3
"""Finite shard harness around the unchanged, previously tested organoid source."""
import argparse,csv,hashlib,json,os,shutil,subprocess,sys,time
from pathlib import Path

SOURCE='826227bef3e34f4c6f3276f0f6c95a9c7b55da15'

def read(path):return json.loads(Path(path).read_text())
def write(path,value):Path(path).write_text(json.dumps(value,indent=2,allow_nan=False)+'\n')

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--arm',choices=['shape','react','llm'],required=True)
    p.add_argument('--study-root',type=Path,required=True)
    p.add_argument('--source-root',type=Path,required=True)
    p.add_argument('--shard',type=int,required=True)
    p.add_argument('--resume',action='store_true')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run in Slurm')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=a.source_root,text=True).strip()!=SOURCE:
        raise ValueError('Inference checkout differs from frozen source')
    if subprocess.check_output(['git','status','--porcelain'],cwd=a.source_root,text=True).strip():
        raise ValueError('Dirty inference checkout')
    plan=read(a.study_root/'plan.json')
    if not 0<=a.shard<plan['tasks'][a.arm]:raise ValueError('Invalid shard')
    manifest=a.study_root/'manifests'/('full.json' if a.arm=='shape' else f'pilot_{a.shard:03d}.json')
    if a.arm!='shape' and plan.get('agent_manifest_pattern'):
        manifest=a.study_root/'manifests'/plan['agent_manifest_pattern'].format(shard=a.shard)
    effective_shard=a.shard
    continuation=a.study_root/'continuation.json'
    continuation_plan=read(continuation) if a.arm=='shape' and continuation.exists() else None
    if continuation_plan and a.shard>=continuation_plan['first_continuation_slot']:
        manifest=a.study_root/continuation_plan['manifest']
        if hashlib.sha256(manifest.read_bytes()).hexdigest()!=continuation_plan['manifest_sha256']:
            raise ValueError('Continuation manifest changed')
        effective_shard=a.shard-continuation_plan['first_continuation_slot']
    pairs=read(manifest)['pairs']
    selected=pairs[effective_shard*64:(effective_shard+1)*64] if a.arm=='shape' else pairs
    root=a.study_root/a.arm/f'shard_{a.shard:03d}'
    root.mkdir(parents=True,exist_ok=True)
    receipt_path=root/'stage_receipt.json'
    if receipt_path.exists():
        previous=read(receipt_path)
        if previous.get('technical_complete'):
            if (previous.get('n_expected')!=len(selected) or previous.get('source_commit')!=SOURCE or
                previous.get('manifest_sha256')!=hashlib.sha256(manifest.read_bytes()).hexdigest()):
                raise ValueError('Completed receipt does not match selected stage')
            print(json.dumps(dict(status='SKIPPED_ALREADY_COMPLETE',arm=a.arm,shard=a.shard,
                n_recorded=previous['n_recorded'],new_model_calls=0)),flush=True)
            return
        if not a.resume:raise ValueError('Incomplete stage requires explicit resume')
    receipt=dict(arm=a.arm,shard=a.shard,n_expected=len(selected),source_commit=SOURCE,
        harness_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        job_id=os.environ['SLURM_JOB_ID'],quality_certified=False,technical_complete=False,
        manifest_sha256=hashlib.sha256(manifest.read_bytes()).hexdigest())
    if continuation_plan and a.shard>=continuation_plan['first_continuation_slot']:
        receipt.update(continuation=True,effective_shard=effective_shard,
            comparison_manifest_fingerprint=plan['full_manifest'],
            continuation_plan_sha256=hashlib.sha256(continuation.read_bytes()).hexdigest())
    runner=a.source_root/'case studies/cardiac organoids/runners/segmentation'
    os.environ['PYTHONPATH']=str(a.source_root)
    weights=Path(os.environ['GROUP_SCRATCH'])/'your-user/shape_benchmark/CO_cpsam_finetune/shape_seg_co'
    output=root/'run'
    if a.arm=='shape':
        cmd=[sys.executable,str(runner/'run_organoid_bounded.py'),'--manifest',str(manifest),
            '--run-dir',str(output),'--model-path',str(weights),'--expected-commit',SOURCE,
            '--shard-size','64','--shard-index',str(effective_shard)]
    else:
        cmd=[sys.executable,str(runner/'run_organoid_segmentation.py'),'run','--manifest',str(manifest),
            '--run-dir',str(output),'--arm',a.arm,'--model-path',str(weights)]
        if a.arm=='react':cmd+=['--require-gpu']
    started=time.monotonic()
    try:
        if not selected:
            receipt.update(technical_complete=True,n_recorded=0,n_delivered=0,
                empty_continuation_slot=True,new_model_calls=0)
            return
        if a.resume:cmd+=['--resume']
        with (root/'inference.log').open('a' if a.resume else 'x') as f:
            rc=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        receipt.update(inference_exit_code=rc,subprocess_wall_seconds=time.monotonic()-started)
        if rc not in (0,2):raise RuntimeError('Inference failed; inspect inference.log')
        data=output if a.arm=='shape' else output/a.arm
        rows=[json.loads(s) for s in (data/'records.jsonl').read_text().splitlines()]
        if [r['image_id'] for r in rows]!=[r['image_id'] for r in selected]:raise ValueError('Incomplete or reordered stage')
        if a.arm=='shape':
            before=read(data/'summary.json');csv_before=(data/'shape_cohort_bounded.csv').read_bytes()
            for k in ['unknown_usage','incomplete_audit_records','incomplete_tool_requests','unmeasured_interrupted_passes']:
                if before[k]:raise ValueError('Incomplete cost evidence: '+k)
            with (root/'completed_resume.log').open('w') as f:
                replay=subprocess.run(cmd if a.resume else cmd+['--resume'],stdout=f,stderr=subprocess.STDOUT).returncode
            if replay not in (0,2):raise RuntimeError('Completed resume failed')
            sys.path.insert(0,str(runner))
            from validate_bounded_stage import check_replay
            check_replay(before,read(data/'summary.json'),csv_before,(data/'shape_cohort_bounded.csv').read_bytes())
            receipt.update(summary=before,completed_resume_no_new_model_calls=True)
        else:
            if read(data/'run_meta.json')['status'] not in ['COMPLETE','COMPLETED_WITH_FAILURES']:raise ValueError('Incomplete arm')
            calls=[json.loads(s) for path in data.rglob('LLM_CALLS.jsonl') for s in path.read_text().splitlines() if s]
            unknown=sum(not isinstance(x.get('tokens'),int) or x['tokens']<=0 for x in calls)
            receipt.update(total_tokens=sum(x.get('tokens') or 0 for x in calls),llm_calls=len(calls),unknown_usage=unknown)
            if unknown:raise ValueError('Unmeasured model usage')
            flat=[]
            for row,pair in zip(rows,selected):
                value={k:json.dumps(v) if isinstance(v,(dict,list)) else v for k,v in row.items()}
                value.update(image_sha256=pair['image_sha256'],ground_truth_sha256=pair['label_sha256'],
                    source_commit=SOURCE,manifest_fingerprint=read(manifest)['fingerprint'])
                flat.append(value)
            with (data/'per_image.csv').open('w',newline='') as f:
                w=csv.DictWriter(f,fieldnames=sorted({k for r in flat for k in r}),lineterminator='\n');w.writeheader();w.writerows(flat)
        receipt.update(technical_complete=True,n_recorded=len(rows),n_delivered=sum(r['delivered'] for r in rows))
    except Exception as exc:
        receipt['error']=str(exc);raise
    finally:
        write(receipt_path,receipt)
        durable=Path(os.environ['GROUP_HOME'])/'your-user/shape_benchmark_evidence'/a.study_root.name/a.arm/f'shard_{a.shard:03d}'
        durable.mkdir(parents=True,exist_ok=True);shutil.copy2(receipt_path,durable/receipt_path.name)
        data=output if a.arm=='shape' else output/a.arm
        for name in ['summary.json','identity.json','sessions.json','records.jsonl','shape_cohort_bounded.csv','run_meta.json','per_image.csv']:
            if (data/name).exists():shutil.copy2(data/name,durable/name)
    print(json.dumps(receipt),flush=True)

if __name__=='__main__':main()

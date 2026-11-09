#!/usr/bin/env python3
"""Recover an export-only validation failure from attested completed inference, with no new model calls."""
import argparse,csv,json,os,subprocess
from pathlib import Path
from cohort_runtime import RunStore,atomic_json,file_sha
from run_organoid_bounded import export_records,audit_totals


def require(value,message):
    if not value:raise ValueError(message)


def verify_run(root,manifest_path,receipt_path,out,*,inference_commit,export_commit):
    root,out=Path(root).resolve(),Path(out).resolve()
    require(root!=out and root not in out.parents and out not in root.parents,'Export recovery must use a separate output directory')
    identity=json.loads((root/'identity.json').read_text())
    manifest=json.loads(Path(manifest_path).read_text())
    prior=json.loads(Path(receipt_path).read_text())
    require(identity['source']['commit']==inference_commit==prior['source_commit'],'Unexpected inference source')
    require(manifest['fingerprint']==identity['manifest'],'Challenge manifest differs')
    require(prior.get('error')=='Replay changed exported records' and prior.get('completed_resume_exit_code') in (0,2),'Only completed export-order failures are recoverable here')
    before=prior['before_resume_summary'];after=json.loads((root/'summary.json').read_text())
    stable=('n_planned','n_recorded','n_delivered','n_operational_success','n_needs_review','llm_calls','tokens','native_calls','tool_calls','tool_requests_started','unknown_usage','incomplete_audit_records','incomplete_tool_requests','unmeasured_interrupted_passes','ap50','ap75','map_50_95')
    require(all(before[k]==after[k] for k in stable),'Completed replay changed metrics or usage')
    require(after['n_recorded']==after['n_delivered']==len(manifest['pairs']),'Incomplete delivery')
    require(not any(after[k] for k in ('unknown_usage','incomplete_audit_records','incomplete_tool_requests','unmeasured_interrupted_passes')),'Unmeasured attempt requires separate investigation')
    with RunStore(root,identity,resume=True) as store:
        by_id={}
        for p in (root/'shards').glob('*/receipt.json'):
            payload=store.load(str(p.parent.relative_to(root)))
            for r in payload['records']:
                require(r['image_id'] not in by_id,'Duplicate attested image');by_id[r['image_id']]=r
        records=[by_id[p['image_id']] for p in manifest['pairs']]
        require(len(records)==len(by_id),'Foreign attested images')
        require(records==[json.loads(s) for s in (root/'records.jsonl').read_text().splitlines()],'JSONL differs from attested inference')
        for r in records:require(json.loads((root/'images'/r['image_id']/'record.json').read_text())==r,'Per-image record changed')
        out.mkdir(parents=True,exist_ok=False)
        a,b=out/'canonical',out/'reordered';a.mkdir();b.mkdir()
        export_records(a,records)
        reordered=json.loads(json.dumps(records),object_pairs_hook=lambda pairs:dict(reversed(pairs)))
        export_records(b,reordered)
        for name in ('records.jsonl','shape_cohort_bounded.csv'):
            require((a/name).read_bytes()==(b/name).read_bytes(),'Export bytes depend on object key order')
        def csv_rows(p):
            with p.open() as f:return list(csv.DictReader(f))
        require(csv_rows(root/'shape_cohort_bounded.csv')==csv_rows(a/'shape_cohort_bounded.csv'),'Legacy CSV values differ from attested records')
        totals=audit_totals(root)
        require(all(totals[k]==after[k] for k in totals),'Audited usage differs from completed-run summary')
        report=dict(schema=1,validation='completed_challenge_export_recovery',technical_gate_passed=True,quality_gate_passed=False,
            inference_source_commit=inference_commit,export_source_commit=export_commit,inference_job_id=prior['job_id'],
            n_recorded=len(records),n_delivered=after['n_delivered'],n_needs_review=after['n_needs_review'],
            manifest_fingerprint=manifest['fingerprint'],completed_resume_values_unchanged=True,completed_resume_no_new_model_calls=True,
            deterministic_export_bytes_verified=True,new_inference_calls=0,observed_usage=totals,
            csv_sha256=file_sha(a/'shape_cohort_bounded.csv'),jsonl_sha256=file_sha(a/'records.jsonl'),
            scope='Retains original inference provenance. Verifies attested masks/records, completed replay usage/values, and fixed export bytes under reordered dictionaries. Does not certify biological quality or a new GPU inference run.')
        atomic_json(out/'recovery-receipt.json',report)
        return report


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('run-dir','manifest','receipt','output-dir'):p.add_argument('--'+name,required=True,type=Path)
    p.add_argument('--expected-inference-commit',required=True);p.add_argument('--expected-export-commit',required=True)
    a=p.parse_args()
    require(os.environ.get('SLURM_JOB_ID'),'Run artifact validation inside Slurm')
    from common import REPO
    git=lambda *args:subprocess.check_output(['git',*args],cwd=REPO,text=True).strip()
    require(git('rev-parse','HEAD')==a.expected_export_commit and not git('status','--porcelain'),'Unexpected or dirty export source')
    base='case studies/cardiac organoids/runners/'
    allowed={base+'segmentation/'+n for n in ('run_organoid_bounded.py','verify_bounded_exports.py','README.md')}|{base+'test/test_bounded_cohort.py'}
    require(set(git('diff','--name-only',a.expected_inference_commit,a.expected_export_commit).splitlines())<=allowed,'Inference implementation changed; a new challenge inference is required')
    require(REPO not in a.output_dir.resolve().parents and REPO!=a.output_dir.resolve(),'Output must be outside source')
    print(json.dumps(verify_run(a.run_dir,a.manifest,a.receipt,a.output_dir,inference_commit=a.expected_inference_commit,export_commit=a.expected_export_commit)),flush=True)

if __name__=='__main__':main()

#!/usr/bin/env python3
"""Exclude completed same-source images while preserving already started full shards."""
import argparse,csv,hashlib,io,json,os,shutil
from pathlib import Path

SOURCE='826227bef3e34f4c6f3276f0f6c95a9c7b55da15'
WEIGHTS='f125b9754521efa042ef1ee614d8d900843b2c933517f393daf2e37268d20c48'
def read(p):return json.loads(p.read_text())
def sha(p):return hashlib.sha256(p.read_bytes()).hexdigest()
def digest(d):return hashlib.sha256(json.dumps(d,sort_keys=True).encode()).hexdigest()
def write(p,d):
    temp=p.with_suffix(p.suffix+'.tmp');temp.write_text(json.dumps(d,indent=2)+'\n');temp.replace(p)

def prepare(study,prior,first_slot):
    if (study/'continuation.json').exists():raise ValueError('Continuation already prepared')
    plan=read(study/'plan.json');full=read(study/'manifests/full.json')
    assert plan['source_commit']==SOURCE and not plan['selective_review']
    assert digest({k:v for k,v in full.items() if k!='fingerprint'})==full['fingerprint']==plan['full_manifest']
    pairs=full['pairs'];byid={p['image_id']:p for p in pairs}
    assert 0<first_slot<plan['tasks']['shape'] and len(byid)==len(pairs)==7582
    if any((study/'shape'/f'shard_{i:03d}'/'run').exists() for i in range(first_slot,plan['tasks']['shape'])):
        raise ValueError('A reserved continuation slot has already started; choose a later boundary')
    reserved=pairs[:first_slot*64];reserved_ids={p['image_id'] for p in reserved}
    records={};csv_rows={};sources=[]
    for split,n in [('challenge',24),('pilot',64)]:
        folder=prior/split;summary=read(folder/'summary.json')
        assert summary['n_recorded']==summary['n_delivered']==n and summary['n_failed']==0
        for key in ['unknown_usage','incomplete_audit_records','incomplete_tool_requests','unmeasured_interrupted_passes']:
            assert summary[key]==0
        current=[json.loads(x) for x in (folder/'records.jsonl').read_text().splitlines()]
        rows=list(csv.DictReader((folder/'shape_cohort_bounded.csv').open()))
        assert len(current)==len(rows)==n and [x['image_id'] for x in current]==[x['image_id'] for x in rows]
        for rec,row in zip(current,rows):
            p=byid[rec['image_id']]
            assert rec['image_id'] not in records and rec['source_commit']==SOURCE and rec['checkpoint_sha256']==WEIGHTS
            assert rec['delivered'] and rec['image_sha256']==p['image_sha256'] and rec['ground_truth_sha256']==p['label_sha256']
            assert [rec['image_height_px'],rec['image_width_px']]==p['shape'][:2]
            records[rec['image_id']]=rec;csv_rows[rec['image_id']]=row
        sources.append(dict(split=split,run=str(folder),records_sha256=sha(folder/'records.jsonl'),csv_sha256=sha(folder/'shape_cohort_bounded.csv'),n=n,tokens=summary['tokens'],llm_calls=summary['llm_calls']))
    reusable=[p for p in pairs if p['image_id'] in records and p['image_id'] not in reserved_ids]
    reuse_ids={p['image_id'] for p in reusable}
    pending=[p for p in pairs if p['image_id'] not in reserved_ids and p['image_id'] not in reuse_ids]
    assert len(reserved)+len(reusable)+len(pending)==len(pairs)
    assert len(reserved_ids|reuse_ids|{p['image_id'] for p in pending})==len(pairs)
    manifest=dict(schema=1,split='full_remaining',pairs=pending,selection=dict(policy='Exclude completed same-source images and preserve started full shards; no score-based selection',full_manifest_fingerprint=full['fingerprint'],first_continuation_slot=first_slot,prior_sources=sources))
    manifest['fingerprint']=digest(manifest)
    path=study/'manifests/full_remaining.json';write(path,manifest)
    reuse_path=study/'reused_completed_records.jsonl'
    reuse_path.write_text(''.join(json.dumps(records[p['image_id']],sort_keys=True)+'\n' for p in reusable))
    fields=sorted({k for p in reusable for k in csv_rows[p['image_id']]})
    with (study/'reused_completed.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');w.writeheader();w.writerows(csv_rows[p['image_id']] for p in reusable)
    result=dict(schema=1,source_commit=SOURCE,checkpoint_sha256=WEIGHTS,first_continuation_slot=first_slot,
        manifest='manifests/full_remaining.json',manifest_sha256=sha(path),remaining_manifest_fingerprint=manifest['fingerprint'],comparison_manifest_fingerprint=full['fingerprint'],
        n_full=len(pairs),n_reserved=len(reserved),n_reused=len(reusable),n_pending=len(pending),n_prior_total=len(records),n_prior_in_reserved=len(records.keys()&reserved_ids),
        source_records=sources,reused_records_sha256=sha(reuse_path),reused_csv_sha256=sha(study/'reused_completed.csv'),
        cost_scope='New stage receipts count incremental full-run costs. Prior source pipeline costs retain their original88-image scopes; report once per source, separately from new costs. No invented per-image allocation of shared costs.',
        grouping='Remaining images form new64-image cohorts under the unchanged scientific source; source cohorts and remaining cohorts keep distinct manifests.',
        quality_certified=False)
    durable=Path(os.environ['GROUP_HOME'])/'your-user/shape_benchmark_evidence'/study.name/'continuation'
    durable.mkdir(parents=True,exist_ok=False)
    for f in [path,reuse_path,study/'reused_completed.csv']:shutil.copy2(f,durable/f.name)
    write(study/'continuation.json',result);shutil.copy2(study/'continuation.json',durable/'continuation.json')
    return result

def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--study-root',type=Path,required=True);p.add_argument('--prior-root',type=Path,required=True);p.add_argument('--first-continuation-slot',type=int,required=True)
    a=p.parse_args();print(json.dumps(prepare(a.study_root,a.prior_root,a.first_continuation_slot)))
if __name__=='__main__':main()

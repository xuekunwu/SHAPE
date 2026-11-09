#!/usr/bin/env python3
"""Compare immutable selective-review evidence with frozen per-arm observations."""
import argparse, json, shutil
from pathlib import Path
from compare_bounded_validation import (ARMS, FROZEN_MANIFEST_SHA, HF_REVISION,
    compare, load, sha, write_csv)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--evidence-dir',type=Path,required=True)
    p.add_argument('--baseline-dir',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    a=p.parse_args()
    if a.output_dir.exists():raise ValueError('Use a fresh export directory')
    inventory=json.loads((a.evidence_dir/'bundle_manifest.json').read_text())
    for relative,item in inventory.items():
        path=(a.evidence_dir/relative).resolve()
        if not path.is_relative_to(a.evidence_dir.resolve()) or sha(path)!=item['sha256']:
            raise ValueError('Changed or unsafe evidence: '+relative)
    frozen=a.baseline_dir/'manifest.json'
    if sha(frozen)!=FROZEN_MANIFEST_SHA:raise ValueError('Unexpected baseline manifest')
    manifest=json.loads(frozen.read_text());baselines={}
    for arm in ARMS:
        path=a.baseline_dir/(arm+'.csv')
        if sha(path)!=manifest['files'][path.name]['sha256']:raise ValueError('Changed baseline')
        baselines[arm]=load(path)
    studies=[('prototype14','ab14','pairs14.json','ab14/baseline'),
             ('corrected14','ab14_v2','pairs14.json','ab14/baseline'),
             ('normal_guard1','guard1_v2','guard1.json','guard1_v2/baseline')]
    result=dict(schema=1,baseline_hf_revision=HF_REVISION,
        scope='Outcome-informed development cases; no independent holdout or whole-dataset certification',
        baseline_manifest_sha256=FROZEN_MANIFEST_SHA,quality_certified=False,
        metrics='AP(t)=TP/(TP+FP+FN), t=.50:.05:.95; not confidence-ranked COCO AP',
        missing='Unobserved stays missing; recorded failures retain original score',
        costs='Provider-reported total includes raw-image profiles, worker preflights, calibration and repair. Zero managed mask-review calls is not zero whole-pipeline VLM calls.',
        caveats=['Prototype representative schema rejected valid call IDs; retained as a failed development experiment.',
                 'Corrected body threshold 0.55 was selected using prior development outcomes.',
                 'Corrected14 reuses immutable control14 from the prototype source; default inference behavior is unchanged.',
                 'Single realizations, not repeated-run estimates; raw-image profiles and reasoner outputs can vary.',
                 'Fixed64 and full dataset were not launched because corrected repair gain was insufficient.'],studies={})
    a.output_dir.mkdir(parents=True)
    for name,trial,mf,control in studies:
        pairs=json.loads((a.evidence_dir/mf).read_text())
        receipt=json.loads((a.evidence_dir/trial/'trial_receipt.json').read_text())
        if not (receipt.get('technical_condition_complete') or receipt.get('technical_comparison_complete')):
            raise ValueError('Incomplete trial')
        rows=load(a.evidence_dir/trial/'selective/shape_cohort_bounded.csv')
        reference=load(a.evidence_dir/control/'shape_cohort_bounded.csv')
        expected=[x['image_id'] for x in pairs['pairs']]
        if [r['image_id'] for r in rows]!=expected or [r['image_id'] for r in reference]!=expected:
            raise ValueError('Changed paired membership/order')
        identity=json.loads((a.evidence_dir/trial/'selective/identity.json').read_text())
        if identity['manifest']!=pairs['fingerprint'] or any(r['source_commit']!=identity['source']['commit'] for r in rows):
            raise ValueError('Unexpected inference provenance')
        outputs,scores,paired=compare(rows,dict(baselines,control=reference))
        folder=a.output_dir/name;folder.mkdir()
        for arm,data in outputs.items():write_csv(folder/(arm+'.csv'),data)
        write_csv(folder/'paired.csv',paired)
        selected=receipt['conditions']['selective']
        if control.startswith(trial+'/'):
            old=receipt['conditions']['baseline']
        else:
            old=json.loads((a.evidence_dir/'ab14/trial_receipt.json').read_text())['conditions']['baseline']
        if sha(a.evidence_dir/trial/'selective/shape_cohort_bounded.csv')!=selected['csv_sha256'] or sha(a.evidence_dir/control/'shape_cohort_bounded.csv')!=old['csv_sha256']:
            raise ValueError('CSV differs from inference receipt')
        if selected['summary']['n_recorded']!=len(rows) or old['summary']['n_recorded']!=len(reference):
            raise ValueError('Incomplete result count')
        for value in [old,selected]:
            if not value['completed_resume_no_new_model_calls']:raise ValueError('Replay not verified')
            if any(value['summary'][k] for k in ['unknown_usage','incomplete_audit_records','incomplete_tool_requests','unmeasured_interrupted_passes']):
                raise ValueError('Incomplete cost evidence')
        result['studies'][name]=dict(n=len(rows),source_commit=identity['source']['commit'],
            control_source_commit=reference[0]['source_commit'],manifest_fingerprint=pairs['fingerprint'],
            selective_summary=selected['summary'],control_summary=old['summary'],arms=scores,
            hardware=selected['hardware'],control_hardware=old['hardware'])
        plots=a.evidence_dir/trial/'plots'
        if plots.exists():shutil.copytree(plots,folder/'plots')
    post=json.loads((a.evidence_dir/'posthoc_candidates.json').read_text())
    diagnostics=[]
    for name,trial,_,_ in studies[:2]:
        rows=load(a.output_dir/name/'shape_cohort_bounded.csv')
        for row in rows:
            candidates=[x for x in post['observations'] if x['trial']==trial and x['condition']=='selective' and x['image_id']==row['image_id']]
            best=max((x['map_50_95'] for x in candidates),default=None)
            diagnostics.append(dict(study=name,image_id=row['image_id'],selected_map=float(row['map_50_95']),
                posthoc_best_native_map=best,posthoc_best_minus_selected=None if best is None else best-float(row['map_50_95']),
                scope='GT only after inference; not a delivered prediction or deployable score'))
    write_csv(a.output_dir/'posthoc_candidate_comparison.csv',diagnostics)
    (a.output_dir/'comparison.json').write_text(json.dumps(result,indent=2,allow_nan=False)+'\n')
    print(json.dumps({k:dict(map=v['selective_summary']['map_50_95'],tokens=v['selective_summary']['tokens'],
        paired=v['arms']['control']) for k,v in result['studies'].items()},indent=2))


if __name__=='__main__':main()

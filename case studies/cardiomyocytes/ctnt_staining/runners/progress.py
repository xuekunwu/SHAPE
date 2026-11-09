"""Three-arm progress, including private three-channel crop and cell scoring stages."""
from pathlib import Path
import json
import sys


def main():
    root=Path(sys.argv[1]);meta=json.loads((root/'SUITE_META.json').read_text())
    print(f"cTnT STAINING SMOKE | {meta['status']} | {meta.get('current_stage','-')}")
    print(f"{'stage':18} | {'CONTROL':24} | {'SHAPE':24} | {'CODEX':24}")
    print('-'*100)
    for stage in ['preflight','segment','crop','embed','batch','predict','evaluation_crops','score']:
        states=[]
        for arm in ['control','shape','codex']:
            p=root/arm;run=json.loads((p/'RUN_META.json').read_text()) if (p/'RUN_META.json').exists() else {}
            done=(p/('PREFLIGHT.json' if stage=='preflight' else stage+'_COMPLETE.json')).exists()
            if done:s='[########] 100% DONE'
            elif run.get('status')=='failed' and run.get('current_stage')==stage:s='[!!!!!!!!] FAILED'
            elif run.get('current_stage')==stage:
                count=None;total=None
                if (p/'CONFIG.json').exists():total=len(json.loads((p/'CONFIG.json').read_text())['images'])
                if stage=='segment' and (p/'segmentation.json').exists():count=len(json.loads((p/'segmentation.json').read_text()))
                if stage=='crop':count=len(list((p/'crop').glob('*/result.json')))
                if count is not None and total:
                    n=min(8,int(8*count/total));s='['+'#'*n+'-'*(8-n)+f'] {round(100*count/total):3}% RUN {count}/{total}'
                else:s='[....>...]     RUNNING'
            else:s='[--------]   0% WAIT'
            states.append(s)
        print(f'{stage:18} | '+' | '.join(f'{s:24}' for s in states))
    for arm in ('shape','codex','control'):
        run=root/arm
        checkpoint=run/'segment/shared_vrr/CHECKPOINT.json'
        if checkpoint.exists():
            s=json.loads(checkpoint.read_text());rounds=s['rounds'];n=len(rounds);cap=s['max_rounds']
            passing=len(set(s['frozen'])-set(s['accepted_with_flags']));total=len(s['history'])
            if not s.get('baseline_complete',True):
                done=len({i for i,h in s['history'].items() if h}|{e['image_id'] for e in s['errors'] if e['round']==0})
                blocks=int(12*done/max(1,total))
                print(f'{arm.upper()} SEG BASELINE ['+'#'*blocks+'.'*(12-blocks)+f'] {done}/{total}')
            print(f'{arm.upper()} SEG VRR ['+'#'*n+'.'*(cap-n)+f'] {n}/{cap} shared rounds | QC {passing}/{total} | accepted flags {len(s["accepted_with_flags"])}')
            for r in rounds:
                print(f'  r{r["round"]}: native {r["native_calls"]}/{r.get("planned_native_calls","pending")} | problems {r["problems_before"]} -> {r.get("problems_after","pending")} | improved {r["improved"]} | new pass {r["newly_passing"]} | reverted {r["reverted"]} | {r.get("phase","complete")}')
        crop=run/'CROP_VRR_PROGRESS.json'
        if crop.exists():
            c=json.loads(crop.read_text());print(f'{arm.upper()} CROP VRR | phase={c["phase"]} | model calls={c["model_calls"]}')
            if c.get('total_materialize_images'):
                done=c.get('materialized_images',0);total=c['total_materialize_images'];blocks=int(12*done/total)
                print('  Delivery ['+'#'*blocks+'.'*(12-blocks)+f'] {done}/{total} images')
            for r in c['round_progress']:
                print(f'  r{r["round"]}: problems {r["problems_before"]} -> {r["problems_after"]} | improved {r["improved"]} | new pass {r["newly_passing"]} | passing {r["passing"]} | reverted {r["reverted"]}')
    if meta.get('error'):print('ERROR:',meta['error'])


if __name__=='__main__':main()

#!/usr/bin/env python3
"""Post-inference review pack. GT is used only here for inspection, never supplied to VRR."""
import argparse,hashlib,json,os,shutil,sys,zipfile
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from common import labels,sha256,protect_output


def edge(mask):
    result=np.zeros(mask.shape,dtype=bool)
    result[1:]|=mask[1:]!=mask[:-1];result[:,1:]|=mask[:,1:]!=mask[:,:-1]
    return result&(mask>0)


def raw_rgb(path):
    with Image.open(path) as im:a=np.asarray(im).copy()
    if a.ndim==3:a=a[...,0]
    lo,hi=np.percentile(a[np.isfinite(a)],[1,99])
    if hi<=lo:lo,hi=float(np.min(a)),float(np.max(a))
    display=np.clip((a.astype(float)-lo)/max(hi-lo,1e-12),0,1)
    return np.repeat((display*255).astype('uint8')[...,None],3,axis=2)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--challenge-run',type=Path,required=True);p.add_argument('--challenge-manifest',type=Path,required=True)
    p.add_argument('--pilot-run',type=Path,required=True);p.add_argument('--pilot-manifest',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--include-image',action='append',default=[],help='Additional post-inference audit image ID; does not change needs_review or inference')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run image review preparation inside Slurm')
    all_pairs=[];entries=[];runs={};outputs=[]
    for name,root,mf in [('challenge',a.challenge_run,a.challenge_manifest),('pilot',a.pilot_run,a.pilot_manifest)]:
        manifest=json.loads(mf.read_text());identity=json.loads((root/'identity.json').read_text())
        if identity['manifest']!=manifest['fingerprint']:raise ValueError('Manifest/run identity differs')
        pairs={r['image_id']:r for r in manifest['pairs']};all_pairs+=list(pairs.values())
        records=[json.loads(x) for x in (root/'records.jsonl').read_text().splitlines()]
        if [r['image_id'] for r in records]!=list(pairs):raise ValueError('Records do not cover the fixed manifest in order')
        runs[name]=dict(root=str(root),manifest=str(mf),source_commit=identity['source']['commit'],manifest_fingerprint=manifest['fingerprint'],summary=json.loads((root/'summary.json').read_text()))
        pass_details={}
        for marker in root.glob('passes/*/*/attempt.json'):
            request=json.loads(marker.read_text());stage=marker.parent/'stage.json'
            if not stage.exists():continue
            payload=json.loads(stage.read_text())
            for rr in payload.get('resolutions') or []:
                dd=rr.get('details') or {}
                brief=dict(reason=rr.get('reason'),status=rr.get('status'),parameters=rr.get('params'),
                    budget_stop=dd.get('organoid_budget_stop'),negative_reviews=dd.get('negative_reviews'),
                    phases=dd.get('llm_calls'),selected=dd.get('selected'),call=dd.get('call'),
                    selection_basis=dd.get('retained_candidate_basis'),failed=dd.get('failed'),assessments=dd.get('assessments'))
                pass_details.setdefault(request['image_id'],[]).append(brief)
        for row in records:
            if row['needs_review'] or row['image_id'] in a.include_image:
                pair=pairs[row['image_id']]
                for field in ('image','label'):
                    if sha256(pair[field])!=pair[field+'_sha256']:raise ValueError('Review input changed')
                pred=root/'images'/row['image_id']/'mask.tif'
                if sha256(pred)!=row['mask_sha256']:raise ValueError('Prediction changed')
                entries.append(dict(split=name,selection='native_flag' if row['needs_review'] else 'posthoc_audit',record={k:v for k,v in row.items() if k!='_thresholds'},pair=pair,
                    prediction=str(pred),passes=pass_details.get(row['image_id'],[])))
    if set(a.include_image)-{p['image_id'] for p in all_pairs}:raise ValueError('Requested audit image is outside the fixed cohorts')
    protect_output(a.output_dir,all_pairs)
    for root in (a.challenge_run.resolve(),a.pilot_run.resolve()):
        if a.output_dir.resolve()==root or root in a.output_dir.resolve().parents:raise ValueError('Review outputs must be separate from inference')
    a.output_dir.mkdir(parents=True,exist_ok=False)
    for name,run in runs.items():
        target=a.output_dir/name;target.mkdir()
        root=Path(run['root'])
        for file in ('shape_cohort_bounded.csv','summary.json','identity.json','sessions.json'):shutil.copy2(root/file,target/file)
    cards=[];priority=None
    for i,entry in enumerate(entries):
        raw=raw_rgb(entry['pair']['image']);gt=labels(entry['pair']['label']);pred=labels(entry['prediction'])
        combined=raw.copy();combined[edge(gt)]=[0,255,80];combined[edge(pred)]=[255,40,40]
        title=f"{i:02d} {entry['split']} GT={int(np.count_nonzero(np.unique(gt)))} pred={int(np.count_nonzero(np.unique(pred)))} AP50={entry['record']['ap50']:.2f}"
        tile=Image.new('RGB',(256,224),'white');tile.paste(Image.fromarray(combined).resize((256,189)),(0,30));ImageDraw.Draw(tile).text((3,3),title,fill='black')
        cards.append(tile);entry['review_index']=i
        if entry['record']['ground_truth_cell_count']==0 and entry['record']['predicted_cell_count']>0:
            sheet=Image.new('RGB',(1536,430),'white');draw=ImageDraw.Draw(sheet)
            for j,(label,mask,color) in enumerate([('RAW',None,None),('PREDICTION',pred,[255,40,40]),('GROUND TRUTH',gt,[0,255,80])]):
                pixels=raw.copy()
                if mask is not None:pixels[edge(mask)]=color
                sheet.paste(Image.fromarray(pixels).resize((512,378)),(j*512,42));draw.text((j*512+4,5),label,fill='black')
            priority=a.output_dir/f'priority_{i:02d}.jpg';sheet.save(priority,quality=72,optimize=True)
    overview=Image.new('RGB',(4*256,((len(cards)+3)//4)*224),'white')
    for i,tile in enumerate(cards):overview.paste(tile,((i%4)*256,(i//4)*224))
    overview.save(a.output_dir/'overview.jpg',quality=68,optimize=True)
    report=dict(schema=1,job_id=os.environ['SLURM_JOB_ID'],script_sha256=sha256(__file__),runs=runs,
        n_review=len(entries),n_native_flagged=sum(bool(e['record']['needs_review']) for e in entries),n_posthoc_audit=sum(e['selection']=='posthoc_audit' for e in entries),legend='Green=GT outline, red=prediction outline. Post-inference only; no GT supplied to the segmenter.',entries=entries)
    (a.output_dir/'review_index.json').write_text(json.dumps(report,indent=2)+'\n')
    with zipfile.ZipFile(a.output_dir/'metadata.zip','w',compression=zipfile.ZIP_DEFLATED) as archive:
        archive.write(a.output_dir/'review_index.json','review_index.json')
        for name in runs:
            for file in ('shape_cohort_bounded.csv','summary.json','identity.json','sessions.json'):archive.write(a.output_dir/name/file,name+'/'+file)
    important=Path(os.environ['GROUP_HOME'])/'your-user/shape_benchmark_evidence'/a.output_dir.name
    important.mkdir(parents=True,exist_ok=True)
    for file in ('metadata.zip','review_index.json','overview.jpg'):shutil.copy2(a.output_dir/file,important/file)
    if priority:shutil.copy2(priority,important/priority.name)
    print(json.dumps(dict(n_review=len(entries),output=str(a.output_dir),files=[dict(name=p.name,bytes=p.stat().st_size,sha256=sha256(p)) for p in a.output_dir.iterdir() if p.is_file()])))

if __name__=='__main__':main()

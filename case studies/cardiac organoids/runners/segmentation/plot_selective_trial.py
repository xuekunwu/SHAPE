#!/usr/bin/env python3
"""Post-inference paired mask plots; GT never enters the segmentation worker."""
import argparse,json,os
from pathlib import Path
import numpy as np
from PIL import Image,ImageDraw
from scipy import ndimage as ndi
from common import labels,sha256,protect_output


def edge(mask):
    out=np.zeros(mask.shape,dtype=bool)
    out[1:]|=mask[1:]!=mask[:-1];out[:,1:]|=mask[:,1:]!=mask[:,:-1]
    return out&(mask>0)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--trial-root',type=Path,required=True)
    p.add_argument('--output-dir',type=Path,required=True)
    p.add_argument('--baseline-run',type=Path,help='Immutable baseline condition directory when plotting a selective-only rerun')
    a=p.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Prepare microscopy plots in Slurm')
    manifest=json.loads(a.manifest.read_text());pairs=manifest['pairs'];protect_output(a.output_dir,pairs)
    roots={c:a.baseline_run if c=='baseline' and a.baseline_run else a.trial_root/c for c in ['baseline','selective']}
    runs={c:[json.loads(t) for t in (roots[c]/'records.jsonl').read_text().splitlines()]
          for c in ['baseline','selective']}
    for c,rows in runs.items():
        if [r['image_id'] for r in rows]!=[r['image_id'] for r in pairs]:raise ValueError('Run membership/order differs')
    a.output_dir.mkdir(parents=True,exist_ok=False);cards=[]
    for i,pair in enumerate(pairs):
        for k in ['image','label']:
            if sha256(pair[k])!=pair[k+'_sha256']:raise ValueError('Plot input changed')
        with Image.open(pair['image']) as im:raw=np.asarray(im).copy()
        if raw.ndim==3:raw=raw[...,0]
        lo,hi=np.percentile(raw,[1,99]);raw=np.clip((raw.astype(float)-lo)/max(hi-lo,1e-12),0,1)
        size=(256,189);rgb=np.asarray(Image.fromarray((raw*255).astype('uint8')).resize(size).convert('RGB')).copy()
        gt=labels(pair['label']);gt_small=np.asarray(Image.fromarray(gt.astype(np.int32)).resize(size,Image.Resampling.NEAREST))
        card=Image.new('RGB',(768,229),'white');draw=ImageDraw.Draw(card);draw.text((3,2),f"{i:02d} {pair['image_id']}",fill='black')
        for j,c in enumerate(['raw','baseline','selective']):
            pixels=rgb.copy();title='RAW'
            if c!='raw':
                r=runs[c][i]
                if r['image_sha256']!=pair['image_sha256'] or r['ground_truth_sha256']!=pair['label_sha256']:raise ValueError('Prediction input identity differs')
                pixels[ndi.binary_dilation(edge(gt_small),iterations=1)]=[0,220,40]
                title=f"{c} mAP={r['map_50_95']:.2f}"
                if r['delivered']:
                    path=roots[c]/'images'/pair['image_id']/'mask.tif'
                    if sha256(path)!=r['mask_sha256']:raise ValueError('Prediction artifact changed')
                    pred=labels(path)
                    if pred.shape!=gt.shape:raise ValueError('Prediction geometry differs')
                    small=np.asarray(Image.fromarray(pred.astype(np.int32)).resize(size,Image.Resampling.NEAREST))
                    pixels[edge(small)]=[255,30,30];title+=f" n={r['predicted_cell_count']}"
                else:title+=' NO MASK'
            card.paste(Image.fromarray(pixels),(j*256,40));draw.text((j*256+3,23),title,fill='black')
        cards.append(card)
    for page in range((len(cards)+6)//7):
        subset=cards[page*7:(page+1)*7];sheet=Image.new('RGB',(768,229*len(subset)),'white')
        for i,card in enumerate(subset):sheet.paste(card,(0,i*229))
        sheet.save(a.output_dir/f'comparison_{page:02d}.jpg',quality=72,optimize=True)
    (a.output_dir/'legend.json').write_text(json.dumps(dict(legend='Green=GT, red=prediction; outlines thickened for display only',
        source_manifest=manifest['fingerprint'],plot_script_sha256=sha256(__file__),n=len(pairs),ground_truth_post_inference_only=True),indent=2)+'\n')
    print(json.dumps(dict(output=str(a.output_dir),n=len(pairs))))


if __name__=='__main__':main()

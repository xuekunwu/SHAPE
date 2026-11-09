"""Small real CONTROL segmentation/crop smoke; does not estimate prediction accuracy."""
import argparse
import hashlib
import importlib.metadata
import json
from pathlib import Path
import subprocess
import time
from types import SimpleNamespace

from ..dataset import preparation as data
from . import experiment_runner as experiment
from ..runtime.reasoning import reasoning_protocol
from ..runtime.source_snapshot import verify_source
from .stage_sequence import run_stages


def select_images(rows,count=5,seed=20261015):
    if count<3:raise ValueError('image-stage smoke needs at least one image per split')
    ordered=sorted(rows,key=lambda r:hashlib.sha256(f'image-stage-smoke-v1|{seed}|{r["image_id"]}'.encode()).hexdigest())
    selected=[]
    for split in data.USAGES:
        group=[r for r in ordered if r['usage']==split]
        if not group:raise ValueError('missing original split in prepared data: '+split)
        selected.append(group[0])
    used={r['image_id'] for r in selected}
    selected.extend(r for r in ordered if r['image_id'] not in used)
    if len(selected)<count:raise ValueError('not enough distinct images')
    return selected[:count]


def main(argv=None):
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('prep','assets-root','out'):parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--expected-source',required=True)
    parser.add_argument('--images',type=int,default=5)
    parser.add_argument('--seed',type=int,default=20261015)
    parser.add_argument('--cores',type=int,default=2)
    args=parser.parse_args(argv)
    args.prep=args.prep.resolve();args.out=args.out.resolve();args.assets_root=args.assets_root.resolve()
    if args.cores<1:parser.error('cores must be positive')
    verify_source(args.expected_source)
    data.fresh_output(args.out,args.prep)
    rows=select_images(data.load_prepared(args.prep),args.images,args.seed)
    for row in rows:
        if row['file']!=row['segment_file'] or row['sha256']!=row['segment_sha256']:
            raise ValueError('raw/segmentation channel identity mismatch')
        if data.digest(data.within(args.prep/'public',row['file']))!=row['sha256']:
            raise ValueError('selected source image hash mismatch: '+row['image_id'])
    args.out.mkdir();(args.out/'logs').mkdir()
    state=dict(status='preflight',stages={},scope='technical_image_stages_only',
               prepare_sha256=data.digest(args.prep/'PREPARE.json'),source_sha256=args.expected_source)
    data.write_json(args.out/'RUN_META.json',state)
    started=time.time()
    try:
        preflight=experiment.preflight(SimpleNamespace(arm='control',model=experiment.MODEL,
            expected_source=args.expected_source,assets_root=args.assets_root,
            reasoning_profile='all-high',out=args.out/'PREFLIGHT.json'))
        data.write_json(args.out/'PREFLIGHT.json',preflight)
        config=dict(dataset='ctnt_flowcytometry',arm='control',model=experiment.MODEL,
            source_sha256=args.expected_source,cores=args.cores,seed=args.seed,
            deadline=None,budget_hours=None,assets_root=str(args.assets_root),
            images=[r['image_id'] for r in rows],smoke=True,reasoning_profile='all-high',
            reasoning=dict(protocol=reasoning_protocol()),segmentation_regime='per-image',
            technical_smoke_scope='CONTROL segmentation and original-pixel crop integrity only')
        data.write_json(args.out/'CONFIG.json',config)
        data.write_json(args.out/'SMOKE_INPUTS.json',dict(
            prepare_sha256=state['prepare_sha256'],selection='label-free SHA256 order; one image per original split then fill',
            seed=args.seed,images=[{k:r[k] for k in ('image_id','well','usage','sha256')} for r in rows],
            original_split_assignments_preserved=True,complete_wells=False))
        state.update(status='running',config=config)
        run_stages(('segment','crop'),experiment.MODULE,args.prep,args.out,config,state)
        masks=data.read_json(args.out/'segmentation.json');cells=data.read_json(args.out/'cells.json')
        if {m['image_id'] for m in masks}!=set(config['images']):raise ValueError('segmentation missed smoke images')
        import numpy as np
        import tifffile
        from benchmarks.full_CPJUMP1.harness.pipeline import raw_stack
        from ..dataset.contract import FLOW
        by_image={r['image_id']:r for r in rows};cached={}
        for mask in masks:
            if data.digest(Path(mask['mask']))!=mask['sha256']:raise ValueError('changed delivered mask')
        for cell in cells:
            row=by_image[cell['image_id']]
            if row['image_id'] not in cached:cached[row['image_id']]=raw_stack(data.within(args.prep/'public',row['file']),FLOW)
            raw=cached[row['image_id']];y0,x0,y1,x1=map(int,cell['crop_bbox'])
            path=Path(cell['crop_path']);actual=tifffile.imread(path);expected=raw[:,y0:y1,x0:x1]
            if (data.digest(path)!=cell['sha256'] or actual.dtype!=raw.dtype
                    or actual.size!=expected.size or not np.array_equal(actual.reshape(expected.shape),expected)):
                raise ValueError('crop is not an exact original-pixel slice')
        verify_source(args.expected_source)
        result=dict(status='complete',scope='technical_image_stages_only',source_sha256=args.expected_source,
            images=len(rows),masks=len(masks),retained_cells=len(cells),raw_crop_integrity='all bit-exact',
            seconds=time.time()-started,llm_calls=0,token_usage=0,
            packages={d.metadata['Name']:d.version for d in importlib.metadata.distributions() if d.metadata.get('Name')},
            gpu=subprocess.check_output(['nvidia-smi','--query-gpu=name,driver_version,memory.total','--format=csv,noheader'],text=True).strip(),
            not_tested=['LLM/VLM decisions','SSL search/training','batch selection','full-cohort prediction accuracy'])
        data.write_json(args.out/'SMOKE_RESULT.json',result)
        state.update(status='complete',current_stage='complete');data.write_json(args.out/'RUN_META.json',state)
        (args.out/'CTNT_RUNNER_IMAGE_SMOKE_COMPLETE').write_text(args.expected_source+'\n')
        print('CTNT_RUNNER_IMAGE_SMOKE_COMPLETE',json.dumps({k:result[k] for k in ('images','retained_cells','seconds','gpu')}),flush=True)
    except BaseException as exc:
        state.update(status='failed',error=type(exc).__name__+': '+str(exc));data.write_json(args.out/'RUN_META.json',state)
        raise


if __name__=='__main__':main()

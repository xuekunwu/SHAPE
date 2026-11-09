"""Queued SHAPE crop replay: gated SSL expert plus zero-shot prediction expert.

Prediction reuses SHAPE crops. A separate development-only upstream VRR canary
does not supply prediction inputs. Eligibility and expert routing use development
data only. Feature spaces have separate heads; only percentages are merged.
"""

# Internal worker CLI; the supported public entrypoint is run.py.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    __package__ = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows'

import argparse
from collections import defaultdict
import importlib
import os
from pathlib import Path
import shutil
import subprocess
import sys
import time
from types import SimpleNamespace
import numpy as np

from ..dataset import preparation as data
from ..evaluation import heldout_evaluation as score
from ..prediction import well_predictor
from ..runtime.stage_runner import invoke
from ..prediction.plate_ssl_routing import eligible_plates
from ..prediction.plate_ssl_routing import choose_routes
from ..prediction.plate_ssl_routing import merge_predictions
from ..runtime.source_snapshot import verify_source

MODULE = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows.hybrid_regression'
STAGES = ('preflight','segment','crop','vrr_canary','ssl_search','embed','batch','predict','readouts','score')


def persist(path, value):
    temporary = path.with_suffix('.tmp')
    data.write_json(temporary,value)
    os.replace(temporary,path)


def reuse_file(source, destination):
    destination.parent.mkdir(parents=True,exist_ok=True)
    if destination.exists():raise FileExistsError(destination)
    # Large arrays are immutable inputs; small records are independent copies.
    if source.suffix == '.npy':os.link(source,destination)
    else:shutil.copy2(source,destination)


def verify_parent(parent, prep, *, unscored_internal=False):
    delivery = data.read_json(parent/'DELIVERY.json')
    marker=parent/('UPSTREAM_FROZEN' if unscored_internal else 'CTNT_FLOW_COMPLETE')
    if not marker.exists():raise ValueError('SHAPE parent not complete/frozen')
    if unscored_internal and not data.read_json(parent/'CONFIG.json').get('internal_unscored_zero_expert'):
        raise ValueError('unscored reuse is restricted to the fresh internal zero-shot expert')
    if marker.read_text().strip() != delivery['source_sha256']:
        raise ValueError('parent completion source mismatch')
    if data.digest(prep/'PREPARE.json') != delivery['prepare_sha256']:
        raise ValueError('different preparation')
    for name, sha in delivery['artifacts'].items():
        if data.digest(parent/name) != sha:raise ValueError('parent artifact changed: '+name)
    selection=data.read_json(parent/'SSL_SELECTION.json')
    if selection['mode'] != 'zero_shot' or selection.get('checkpoint'):
        raise ValueError('zero-shot reuse requires the completed zero-shot SHAPE parent')
    if not np.array_equal(np.load(parent/'profiles.npy'),np.load(parent/'corrected.npy')):
        raise ValueError('parent zero-shot profiles were corrected')
    return delivery


def mark(run, stage, seconds, **extra):
    data.write_json(run/(stage+'_COMPLETE.json'),dict(status='complete',stage=stage,seconds=seconds,**extra))


def stage(args):
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from benchmarks.full_CPJUMP1.harness import pipeline
    from shape.engine import usage
    config=data.read_json(args.run/'CONFIG.json')
    verify_source(config['source_sha256'])
    asset_environment(Path(config['assets_root']))
    wanted=set(config['images'])
    rows=[r for r in data.load_prepared(args.prep) if r['image_id'] in wanted]
    started=time.time()
    if args.name in ('ssl_search','embed'):
        if args.name=='ssl_search':
            expected=set(data.read_json(args.run/'ELIGIBLE_PLATES.json'))
            cells=data.read_json(args.run/'cells.json')
            if not cells or any(c['usage']=='evaluation_only' or c['plate'] not in expected for c in cells):
                raise ValueError('SSL search received held-out or ineligible cells')
            os.environ.update(pipeline.stage_env('shape','embed',config['reasoning']['protocol']))
        pipeline.representation(args.prep,args.run,config,rows)
        if args.name=='ssl_search' and usage.total_tokens()<=0:
            raise RuntimeError('SSL search has no observed model usage')
    elif args.name=='batch':
        pipeline.batch_correction(args.prep,args.run,config,rows)
        if data.read_json(args.run/'batch_SEARCH.json')['admission']['decision'] not in ('ABSTAIN','SKIP'):
            raise ValueError('new batch correction requires independently validated references')
        if not np.array_equal(np.load(args.run/'profiles.npy'),np.load(args.run/'corrected.npy')):
            raise ValueError('abstention changed profiles')
    elif args.name=='predict':well_predictor.predict(args.prep,args.run)
    mark(args.run,args.name,time.time()-started,usage=usage.snapshot())


def execute(args):
    from benchmarks.full_CPJUMP1.harness import pipeline
    from . import experiment_runner as e2e
    from ..dataset import sampling as pilot
    from shape.agent.verification.balanced_policy import selected_policy
    verify_source(args.expected_source)
    full_run=getattr(args,'full_run',False)
    parent=args.parent
    delivery=verify_parent(parent,args.prep,unscored_internal=full_run)
    rows=data.load_prepared(args.prep)
    parent_config=data.read_json(parent/'CONFIG.json')
    wanted=set(parent_config['images']);rows=[r for r in rows if r['image_id'] in wanted]
    cells=data.read_json(parent/'cells.json')
    eligibility=eligible_plates(rows)
    eligible={p for p,r in eligibility.items() if r['eligible']}
    config={**parent_config,'arm':'shape_hybrid','source_sha256':args.expected_source,
            'cores':args.cores,'deadline':None,'budget_hours':None,
            'hybrid_protocol':('shape-full-updated-evidence-ssl-validation-routing-v3' if parent_config.get('ssl_search_policy')=='evidence_cost_v1'
                else 'shape-full-updated-two-expert-validation-routing-v2') if full_run else 'shape-crop-replay-two-expert-validation-routing-v1',
            'execution_mode':'full_updated_pipeline' if full_run else 'historical_crop_replay',
            'vrr_policy':selected_policy(),
            'eligibility':eligibility,'routing_minimum_validation_wells':2,
            'routing_rule':'adopted SSL checkpoint AND lower plate-specific validation MAE; ties retain zero-shot',
            'parent_run':str(parent),'parent_delivery_sha256':data.digest(parent/'DELIVERY.json'),
            'source_reuse':'fresh internal upstream only; no historical masks/crops/features/head' if full_run else 'SHAPE segment/crop and zero-shot features/head; no upstream rerun',
            'fit_scope':'eligible-plate development_train cells only; eligible validation for SSL selection',
            'regression_scope':'each expert has its own scaler and Ridge fitted on all development_train wells',
            'merge_scope':'percentage predictions only; feature spaces never concatenated or mixed'}
    if full_run:
        config.pop('ssl_selection',None)
        config['internal_unscored_zero_expert']=False
    data.write_json(args.out/'CONFIG.json',config)
    data.write_json(args.out/'ELIGIBILITY.json',eligibility)
    state=data.read_json(args.out/'RUN_META.json')
    def current(name):
        state.update(status='running',current_stage=name);persist(args.out/'RUN_META.json',state)
        print('HYBRID_STAGE_START',name,flush=True)
    def child(name, folder):
        current(name);started=time.time()
        invoke(MODULE,name,args.prep,folder,log=args.out/'logs'/(name+'.log'),style='named')
        if folder!=args.out:reuse_file(folder/(name+'_COMPLETE.json'),args.out/(name+'_COMPLETE.json'))
        print('HYBRID_STAGE_COMPLETE',name,time.time()-started,flush=True)
    current('preflight')
    preflight_args=SimpleNamespace(arm='shape',model=config['model'],assets_root=Path(config['assets_root']),
                  expected_source=args.expected_source,reasoning_profile='all-high',out=args.out/'PREFLIGHT.json')
    if not full_run:data.write_json(args.out/'PREFLIGHT.json',e2e.preflight(preflight_args))
    for name in ['cells.json','excluded_cells.json','segmentation.json']:
        reuse_file(parent/name,args.out/name)
    for name in ['segment','crop']:
        if full_run:
            existing=args.out/(name+'_COMPLETE.json')
            if existing.exists():
                if data.digest(existing)!=data.digest(parent/(name+'_COMPLETE.json')):
                    raise ValueError('fresh upstream stage record changed')
            else:reuse_file(parent/(name+'_COMPLETE.json'),existing)
        else:
            mark(args.out,name,0.,reused=True,parent=str(parent),parent_stage=data.read_json(parent/(name+'_COMPLETE.json')))
    if not full_run:current('vrr_canary')
    from . import upstream_qa as vrr_canary
    if full_run:
        canary_report=data.read_json(args.out/'vrr_canary/REPORT.json')
    else:
        canary_started=time.time()
        canary_report=vrr_canary.run(args.prep,parent,args.out/'vrr_canary',config)
        mark(args.out,'vrr_canary',time.time()-canary_started,report='vrr_canary/REPORT.json',prediction_crops_replaced=False)
    zero=args.out/'zero_shot';zero.mkdir()
    for name in ['cells.json','embedding.npy','profiles.npy','profiles.csv','corrected.npy','predictions.json',
                 'REGRESSOR.json','regressor.npz','SSL_SELECTION.json','batch_SEARCH.json']:
        reuse_file(parent/name,zero/name)
    data.write_json(zero/'REUSE.json',dict(parent=str(parent),parent_delivery_sha256=config['parent_delivery_sha256'],
                     artifacts={p.name:data.digest(p) for p in zero.iterdir() if p.is_file()}))
    common={**parent_config,'source_sha256':args.expected_source,'cores':args.cores,'deadline':None,'budget_hours':None}
    common.pop('ssl_selection',None)  # Fresh zero-expert inference must not disable the subsequent SSL search.
    search=args.out/'ssl_search';search.mkdir()
    search_rows=[r for r in rows if r['plate_id'] in eligible and r['usage']!='evaluation_only']
    search_cells=[c for c in cells if c['plate'] in eligible and c['usage']!='evaluation_only']
    if not eligible or not search_cells:raise ValueError('no eligible SSL training/validation plate')
    search_config={**common,'arm':'shape','images':[r['image_id'] for r in search_rows],'fit_only':True,
                   'ssl_epoch_budget':pipeline.ssl_epoch_budget(search_rows,common['train_cells_per_site'])}
    data.write_json(search/'CONFIG.json',search_config);data.write_json(search/'cells.json',search_cells)
    data.write_json(search/'ELIGIBLE_PLATES.json',sorted(eligible))
    child('ssl_search',search)
    selection=data.read_json(search/'SSL_SELECTION.json')
    if selection.get('gate',{}).get('decision')!='RUN':
        raise RuntimeError('eligible-only SSL search did not pass the original measurability gate')
    fine_predictions=None;fine=None
    if selection.get('checkpoint'):
        fine=args.out/'finetuned';fine.mkdir()
        reuse_file(parent/'cells.json',fine/'cells.json')
        inference_config={**common,'arm':'control','ssl_selection':selection}
        data.write_json(fine/'CONFIG.json',inference_config)
        child('embed',fine);child('batch',fine);child('predict',fine)
        fine_predictions=data.read_json(fine/'predictions.json')
    else:
        for name in ['embed','batch','predict']:
            mark(args.out,name,0.,reason='SSL candidate not adopted; zero-shot expert reused')
    # Only development targets are opened for routing. Held-out targets enter score() after DELIVERY is frozen.
    current('predict')
    development_path=args.prep/'private/development_labels.json'
    if data.digest(development_path)!=data.read_json(args.prep/'PREPARE.json')['development_labels_sha256']:
        raise ValueError('development labels changed')
    development=data.read_json(development_path)
    zero_predictions=data.read_json(zero/'predictions.json')
    labels={p['well']:development[p['well']][data.TARGET] for p in zero_predictions if p['usage']!='evaluation_only'}
    routes=choose_routes(zero_predictions,fine_predictions,labels,eligibility)
    data.write_json(args.out/'ROUTING.json',dict(routes=routes,selection_split='development_val',
                     heldout_targets_used=False,validation_size_warning='Per-plate routing has only 2/2/4 validation wells; exploratory model selection'))
    merged=merge_predictions(zero_predictions,fine_predictions,routes)
    data.write_json(args.out/'predictions.json',merged)
    data.write_json(args.out/'SSL_SELECTION.json',dict(protocol=config['hybrid_protocol'],
                     candidate=selection,routes=routes,feature_merge=False))
    current('readouts');started=time.time()
    original=data.read_json(parent/'STAGE_READOUTS.json')
    readouts=dict(source_sha256=args.expected_source,parent_delivery_sha256=config['parent_delivery_sha256'],
                  upstream_reused=dict(segment=original['segment'],crop=original['crop'],
                     source=str(parent/'STAGE_READOUTS.json'),sha256=data.digest(parent/'STAGE_READOUTS.json')),
                  zero_shot_embed=original['embed'],finetuned_embed=None,routes=routes,
                  vrr_canary=canary_report,
                  interpretation='Shared SHAPE objects; separate expert features and regression heads; merged percentages only')
    if full_run:
        readouts['fresh_upstream']=dict(source=str(parent),historical_artifacts_used=False,
            zero_expert_process={s:data.read_json(parent/(s+'_COMPLETE.json')) for s in ('embed','batch','predict')})
    if fine is not None:
        from shape.tools.cell_state_analyzer import quant
        x=np.load(fine/'embedding.npy',mmap_mode='r');by_split={}
        for usage in data.USAGES:
            indices=np.array(sorted([i for i,c in enumerate(cells) if c['usage']==usage],key=lambda i:cells[i]['cell_id']))
            indices=indices[np.random.default_rng(0).permutation(len(indices))[:quant.HEALTH_ROWS]]
            meta=[dict(site=cells[i]['image_id'],well=cells[i]['well'],plate=cells[i]['plate']) for i in indices]
            values=np.asarray(x[indices])
            by_split[usage]=dict(sampled_cells=len(indices),health=quant.health(values),
                     image_leakage=quant.image_leakage(values,meta),
                     sibling_site_map=quant.sibling_site_map(values,meta,seed=quant.FRESH_SEED))
        readouts['finetuned_embed']=dict(selection=selection,by_split=by_split,
                         batch=data.read_json(fine/'batch_SEARCH.json'))
    readouts['process']={s:data.read_json(args.out/(s+'_COMPLETE.json')) for s in ['segment','crop','ssl_search','embed','batch','predict']}
    data.write_json(args.out/'STAGE_READOUTS.json',readouts);mark(args.out,'readouts',time.time()-started)
    # Verify reused parent files again before finalizing the new delivery.
    verify_parent(parent,args.prep,unscored_internal=full_run)
    assert data.digest(parent/'DELIVERY.json')==config['parent_delivery_sha256']
    verify_source(args.expected_source)
    excluded={'RUN_META.json','DELIVERY.json'}
    artifacts={p.relative_to(args.out).as_posix():data.digest(p) for p in args.out.rglob('*')
               if p.is_file() and p.suffix in {'.json','.npy','.npz','.csv','.pth','.pt','.ckpt'}
               and 'logs' not in p.relative_to(args.out).parts and p.name not in excluded}
    checkpoint=selection.get('checkpoint')
    if checkpoint:
        cp=Path(checkpoint).resolve()
        if not cp.is_relative_to(args.out.resolve()):raise ValueError('checkpoint outside hybrid run')
        artifacts[cp.relative_to(args.out.resolve()).as_posix()]=data.digest(cp)
    data.write_json(args.out/'DELIVERY.json',dict(status='complete',arm='shape_hybrid',source_sha256=args.expected_source,
                     prepare_sha256=delivery['prepare_sha256'],smoke=False,pilot=False,artifacts=artifacts,
                     parent_delivery_sha256=config['parent_delivery_sha256'],
                     boundary='frozen expert models, development-only routes and final percentages; original SHAPE crops by hash reference'))
    current('score');started=time.time()
    result=score.score(args.prep,args.out)
    result.update(arm='shape_hybrid',hybrid_protocol=config['hybrid_protocol'],routes=routes,
                  parent_run=str(parent),parent_prediction_sha256=data.digest(parent/'predictions.json'),
                  evaluation_note='Post-hoc exploratory fourth arm; original held-out scores were previously inspected')
    data.write_json(args.out/'RESULT.json',result);mark(args.out,'score',time.time()-started)
    state.update(status='complete',current_stage='score',result=result);persist(args.out/'RUN_META.json',state)
    (args.out/'CTNT_FLOW_COMPLETE').write_text(args.expected_source+'\n')
    print('HYBRID_COMPLETE',flush=True)


def queue(args):
    if args.cores<1 or args.cores>(os.cpu_count() or 1):raise ValueError('cores must fit the VM')
    verify_source(args.expected_source)
    previous=None
    if getattr(args,'resume_waiting',False):
        previous=data.read_json(args.out/'RUN_META.json')
        if (previous.get('status')!='waiting' or previous.get('parent')!=str(args.parent)
                or previous.get('after')!=str(args.after) or (args.out/'CONFIG.json').exists()
                or any(p.name not in {'RUN_META.json','logs'} for p in args.out.iterdir())):
            raise ValueError('only an untouched waiting queue may be upgraded')
    else:
        args.out.mkdir(parents=True,exist_ok=False);(args.out/'logs').mkdir()
    state=dict(status='waiting',current_stage='waiting_predecessor',parent=str(args.parent),
               after=str(args.after),source_sha256=args.expected_source,arm='shape_hybrid')
    if getattr(args,'full_pipeline',False):
        from .hybrid_upstream import vrr_environment
        state['vrr_environment']=vrr_environment()
    if previous:state['previous_waiting_source_sha256']=previous['source_sha256']
    persist(args.out/'RUN_META.json',state)
    try:
        from .suite_runner import wait_for_predecessor
        from .suite_runner import configure_private_environment
        wait_for_predecessor(args.after,require_suite=True)
        configure_private_environment(args.api_env,expected_base='https://api.openai.com/v1')
        os.environ.update(OMP_NUM_THREADS=str(args.cores),OPENBLAS_NUM_THREADS=str(args.cores),
                          MKL_NUM_THREADS=str(args.cores),PYTHONUNBUFFERED='1',SHAPE_VRR_POLICY='balanced_v1')
        if getattr(args,'full_pipeline',False):
            from .hybrid_upstream import execute_full
            execute_full(args)
        else:execute(args)
    except BaseException as error:
        state=data.read_json(args.out/'RUN_META.json');state.update(status='failed',error_type=type(error).__name__)
        persist(args.out/'RUN_META.json',state)
        raise


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    sub=parser.add_subparsers(dest='command',required=True)
    run=sub.add_parser('queue')
    for name in ['prep','parent','after','out','api-env']:run.add_argument('--'+name,type=Path,required=True)
    run.add_argument('--expected-source',required=True);run.add_argument('--cores',type=int,default=12)
    run.add_argument('--resume-waiting',action='store_true',help='Upgrade only an untouched waiting queue')
    run.add_argument('--full-pipeline',action='store_true',help='Run updated segmentation/crop from raw images before expert routing')
    run.add_argument('--adaptive-segmentation',action='store_true',
        help='Opt in to core stratified visual triage, evidence continuation and persisted native trials (requires full pipeline)')
    run.add_argument('--reuse-crop-delivery',action='store_true',
        help='Reuse the verified selected native crop result without a duplicate delivery call (requires full pipeline)')
    worker=sub.add_parser('stage');worker.add_argument('--name',choices=['ssl_search','embed','batch','predict'],required=True)
    worker.add_argument('--prep',type=Path,required=True);worker.add_argument('--run',type=Path,required=True)
    args=parser.parse_args()
    if getattr(args,'adaptive_segmentation',False) and not getattr(args,'full_pipeline',False):
        parser.error('--adaptive-segmentation requires --full-pipeline')
    if getattr(args,'reuse_crop_delivery',False) and not getattr(args,'full_pipeline',False):
        parser.error('--reuse-crop-delivery requires --full-pipeline')
    if args.command=='stage':stage(args)
    else:queue(args)


if __name__=='__main__':main()

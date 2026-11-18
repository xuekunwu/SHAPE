"""Native three-arm cell E2E stages; private fluorescence is used only by trusted evaluation."""
import argparse
import importlib
import os
from pathlib import Path
import sys
import time
from types import SimpleNamespace
import numpy as np

from . import data
from ..ctnt_flowcytometry.source_snapshot import verify_source, pack
from ..ctnt_flowcytometry.reasoning import reasoning_protocol

MODULE='case studies.cardiomyocytes.ctnt_staining.e2e'
STAGES=('segment','crop','embed','batch','predict','evaluation_crops','score')


def agent_environment(model):
    os.environ['CTNT_CODEX_MODEL']=model
    os.environ['CTNT_CODEX_BASE_URL']=os.environ.get('LLM_BASE_URL','').rstrip('/')


def preflight(args):
    verify_source(args.expected_source)
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from benchmarks.full_CPJUMP1.harness import pipeline
    import torch
    import shape
    if not Path(shape.__file__).resolve().is_relative_to(data.REPO):raise RuntimeError('SHAPE loaded from another checkout')
    assets=asset_environment(args.assets_root)
    if not torch.cuda.is_available():raise RuntimeError('CUDA is required')
    model_preflight={}
    if args.arm in ('shape','codex'):
        from benchmarks.full_CPJUMP1.harness.gateway_route import validate
        model_preflight['gateway_route']=validate(os.environ.get('LLM_BASE_URL',''))
        if not os.environ.get('LLM_API_KEY'):raise RuntimeError('missing SHAPE LLM_API_KEY')
        os.environ.update(pipeline.stage_env(args.arm,'segment',reasoning_protocol('all-high')))
        if args.arm=='shape':
            from shape.engine import usage
            before=usage.total_tokens()
            pipeline.engine_for(dict(arm='shape',model=args.model,reasoning={'protocol':reasoning_protocol('all-high')}),'segment')
            model_preflight['total_tokens']=usage.total_tokens()-before
            if model_preflight['total_tokens']<=0:raise RuntimeError('SHAPE model preflight has no observed token usage')
        else:
            agent_environment(args.model)
            from benchmarks.full_CPJUMP1.harness.codex_arm import backend
            model_preflight.update(backend('codex').preflight(args.out.parent/'codex_preflight',args.model))
    from shape.utils.runtime_environment import capture
    lock=os.environ.get('SHAPE_BENCHMARK_LOCK')   # optional: an exact package lock the installed set is checked against
    environment=capture(data.REPO/lock if lock else None)
    data.write_json(args.out.parent/'ENVIRONMENT.json',environment)
    return dict(status='complete',source_sha256=args.expected_source,arm=args.arm,model=args.model,
        gpu=torch.cuda.get_device_name(),assets=assets,reasoning_effort='high',
        prediction_channels=['Brightfield'],evaluation_channels=['Brightfield','DAPI','TroponinT'],model_preflight=model_preflight)


def run_stage(args):
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from shape.engine import usage
    config=data.read_json(args.run/'CONFIG.json');verify_source(config['source_sha256'])
    agent_environment(config['model'])
    selected=set(config['images']);rows=[r for r in data.load_prepared(args.prep) if r['image_id'] in selected]
    protocol=config['reasoning']['protocol'];started=time.time();usage.reset()
    stage=args.stage
    if stage in ('segment','crop','embed','batch'):
        asset_environment(Path(config['assets_root']));pipeline.check_stage_env(config['arm'],stage,protocol)
        try:
            if config['arm']=='codex':
                from benchmarks.full_CPJUMP1.harness.codex_arm import run_stage as native
                native(stage,args.prep,args.run,config)
            else:pipeline.STAGES[stage](args.prep,args.run,config,rows)   # SHAPE-v2.0: native crop/embed, no cs-cm hybrid/cohort workflows
        finally:data.write_json(args.run/f'{stage}_USAGE.json',usage.snapshot())
        tokens=usage.snapshot()
        if config['arm']=='codex':tokens=data.read_json(args.run/'agent'/stage/'SELECTED.json')
        if config['arm']!='control' and tokens.get('total_tokens',0)<=0:raise RuntimeError('reasoner stage has no observed token usage')
        data.write_json(args.run/f'{stage}_USAGE.json',tokens)
        if stage=='batch':
            if not np.array_equal(np.load(args.run/'profiles.npy'),np.load(args.run/'corrected.npy')):
                raise RuntimeError('well-level batch correction is not validated for the cell classifier')
    elif stage=='predict':
        from .score import predict
        predict(args.prep,args.run,Path(config['references']),config['minimum_matching_iou']);tokens=None
    elif stage=='evaluation_crops':
        from .references import export_crops
        export_crops(args.prep,args.run,Path(config['private_evaluation_root']),workers=config['cores']);tokens=None
    elif stage=='score':
        from .score import score
        score(args.prep,args.run,Path(config['references']),config['minimum_reference_coverage'],config['bootstrap_draws'])
        from .readouts import collect
        collect(args.prep,args.run,Path(config['private_evaluation_root']));tokens=None
    else:raise ValueError(stage)
    data.write_json(args.run/f'{stage}_COMPLETE.json',dict(status='complete',stage=stage,seconds=time.time()-started,usage=tokens))


def run(args):
    from benchmarks.full_CPJUMP1.harness.cli import child
    from benchmarks.full_CPJUMP1.harness import pipeline
    from .references import verify
    verify_source(args.expected_source);rows=data.sample_images(data.load_prepared(args.prep,verify_pixels=True),args.fraction,args.seed)
    from . import pilot
    pilot_record=pilot.verify(args.prep,args.pilot_manifest) if args.pilot_manifest else None
    if pilot_record and args.fraction!=1:raise ValueError('prepared pilot already sampled; do not sample again')
    reference_record,refs=verify(args.prep,args.references)
    if not {r['image_id'] for r in rows}<=set(refs):raise ValueError('missing independent reference masks')
    scope=pilot.reference_scope(reference_record,args.smoke,args.allow_exploratory,pilot_record,args.allow_exploratory_full,args.fraction)
    data.fresh_output(args.out,args.prep);args.out.mkdir(parents=True);(args.out/'logs').mkdir()
    protocol=reasoning_protocol('all-high')
    config=dict(dataset='ctnt_staining',arm=args.arm,model=args.model,seed=args.seed,source_sha256=args.expected_source,
        # the v2.0 Codex arm needs a finite deadline; --budget-hours gives every arm the same one
        deadline=None if args.budget_hours is None else time.time()+args.budget_hours*3600,
        budget_hours=args.budget_hours,cores=args.cores,assets_root=str(args.assets_root.resolve()),
        images=[r['image_id'] for r in rows],smoke=args.smoke,train_cells_per_site=128,inference_chunk=2048,
        reasoning={'protocol':protocol},shape_policy={'crop_vrr_workers':args.cores},
        ssl_epoch_budget=pipeline.ssl_epoch_budget(rows,128),references=str(args.references.resolve()),
        reference_complete_sha256=data.digest(args.references/'REFERENCE_COMPLETE.json'),
        private_evaluation_root=str(args.out.parent/'private_evaluation'/args.out.name),minimum_matching_iou=.5,
        minimum_reference_coverage=.8,bootstrap_draws=200 if args.smoke else 1000,
        prediction_channels=['Brightfield'],evaluation_channels=['Brightfield','DAPI','TroponinT'],
        segmentation_regime='per-image' if args.smoke else 'cohort',fraction=args.fraction,
        agent_context_policy='tool_api_v1',agent_context_stages=['segment','crop','embed','batch'],
        agent_decision_rounds=8,hybrid_protocol=None,
        technical_smoke_images=5 if args.smoke and len(rows)==5 else None)
    # SHAPE-v2.0: no segmentation_adaptive (it needs cs-cm's Cell_Segmenter_Tool trial cache); v2.0 default segmentation.
    if args.smoke:
        # All selected images are searched by both adaptive arms; smoke is not an asymmetric one-image SHAPE search.
        config['smoke_policy']=dict(shape_search_images=len(rows),shape_search_rounds=2,codex_rounds=2)
    if not pilot_record:config['evaluation_scope']=scope
    data.write_json(args.out/'CONFIG.json',config)
    if pilot_record:
        config.update(pilot_fraction=pilot_record['fraction'],pilot_seed=pilot_record['seed'],
            pilot_sha256=data.digest(args.pilot_manifest),evaluation_scope=scope,
            source_metadata_sha256=data.read_json(args.prep/'PREPARE.json')['metadata_sha256'])
        data.write_json(args.out/'PILOT.json',pilot_record)
        data.write_json(args.out/'CONFIG.json',config)
    meta=dict(status='preflight',config=config,stages={})
    try:
        data.write_json(args.out/'PREFLIGHT.json',preflight(args))
        for stage in STAGES:
            meta.update(status='running',current_stage=stage);data.write_json(args.out/'RUN_META.json',meta)
            print('STAGE_START',stage,flush=True)
            env=dict(os.environ)
            if stage in ('segment','crop','embed','batch'):
                env.update(pipeline.stage_env(args.arm,stage,protocol))
                if stage=='segment' and args.arm=='shape':
                    from ..ctnt_flowcytometry.hybrid_upstream import vrr_environment
                    env.update(vrr_environment())
            child([sys.executable,'-m',MODULE,'stage',stage,'--prep',str(args.prep),'--run',str(args.out)],args.out/'logs'/f'{stage}.log',None,env=env)
            meta['stages'][stage]=data.read_json(args.out/f'{stage}_COMPLETE.json')
            print('STAGE_COMPLETE',stage,flush=True)
        meta.update(status='complete');data.write_json(args.out/'RUN_META.json',meta)
        marker='CTNT_STAINING_SMOKE_COMPLETE' if args.smoke else 'CTNT_STAINING_PILOT_COMPLETE' if pilot_record else 'CTNT_STAINING_COMPLETE'
        (args.out/marker).write_text(args.expected_source+'\n')
    except BaseException as exc:
        meta.update(status='failed',error=type(exc).__name__+': '+str(exc));data.write_json(args.out/'RUN_META.json',meta);raise


def reference_segment(args):
    """A fixed BF-only pseudo-reference pass, independent of the compared agent searches."""
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    verify_source(args.expected_source);asset_environment(args.assets_root)
    data.fresh_output(args.out,args.prep);args.out.mkdir(parents=True);(args.out/'logs').mkdir()
    rows=data.sample_images(data.load_prepared(args.prep),args.fraction,args.seed)
    config=dict(dataset='ctnt_staining',arm='control',model='gpt-5.6-luna',seed=args.seed,deadline=None,cores=args.cores,
        technical_smoke_images=5 if len(rows)==5 else None,reasoning={'protocol':reasoning_protocol('all-high')})
    pipeline.segmentation(args.prep,args.out,config,rows)
    masks=data.read_json(args.out/'segmentation.json')
    data.write_json(args.out/'MASKS.json',{m['image_id']:{'mask':m['mask'],'sha256':m['sha256']} for m in masks})
    data.write_json(args.out/'REFERENCE_MASK_PROVENANCE.json',dict(kind='fixed_algorithmic_pseudo_reference',
        algorithm='Cellpose-SAM frozen tool defaults on brightfield only',source_sha256=args.expected_source,
        note='Shares its algorithm with CONTROL. Detection agreement is not an independent accuracy test.'))


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__);sub=p.add_subparsers(dest='command',required=True)
    c=sub.add_parser('prepare');c.add_argument('--channels-root',type=Path,required=True);c.add_argument('--out',type=Path,required=True);c.add_argument('--seed',type=int,default=20261017)
    c=sub.add_parser('freeze-references');c.add_argument('--prep',type=Path,required=True);c.add_argument('--masks',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
    c.add_argument('--kind',choices=['curated_independent','fixed_algorithmic_pseudo_reference'],required=True)
    c.add_argument('--registration-verified',action='store_true');c.add_argument('--allow-exploratory',action='store_true');c.add_argument('--seed',type=int,default=20261017)
    c=sub.add_parser('pack');c.add_argument('--out',type=Path,required=True)
    for command in ['prepare-pilot','prepare-full']:
        c=sub.add_parser(command);c.add_argument('--raw',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
        c.add_argument('--frozen-split',type=Path,help='private/splits.json of an earlier preparation, to check against (default: none)');c.add_argument('--workers',type=int,default=16)
    for command in ['run','reference-segment']:
        c=sub.add_parser(command);c.add_argument('--prep',type=Path,required=True);c.add_argument('--out',type=Path,required=True)
        c.add_argument('--assets-root',type=Path,required=True);c.add_argument('--expected-source',required=True)
        c.add_argument('--cores',type=int,default=8);c.add_argument('--fraction',type=float,default=.05);c.add_argument('--seed',type=int,default=20261017)
        if command=='run':
            c.add_argument('--references',type=Path,required=True);c.add_argument('--arm',choices=['control','shape','codex'],required=True)
            c.add_argument('--model',default='gpt-5.6-luna');c.add_argument('--smoke',action='store_true')
            c.add_argument('--pilot-manifest',type=Path);c.add_argument('--allow-exploratory',action='store_true')
            c.add_argument('--allow-exploratory-full',action='store_true')
            c.add_argument('--budget-hours',type=float)
    c=sub.add_parser('stage');c.add_argument('stage',choices=STAGES);c.add_argument('--prep',type=Path,required=True);c.add_argument('--run',type=Path,required=True)
    c=sub.add_parser('native-trial');c.add_argument('stage',choices=['segment','crop','embed','batch']);c.add_argument('--prep',type=Path,required=True);c.add_argument('--run',type=Path,required=True);c.add_argument('--config',type=Path)
    args=p.parse_args(argv)
    if args.command=='prepare':print(data.prepare(args.channels_root,args.out,args.seed))
    elif args.command in ('prepare-pilot','prepare-full'):
        from .pilot import prepare
        print(prepare(args.raw,args.out,args.frozen_split,args.workers,all_wells=args.command=='prepare-full'))
    elif args.command=='pack':print(pack(args.out))
    elif args.command=='freeze-references':
        from .references import freeze
        print(freeze(args.prep,args.masks,args.out,kind=args.kind,registration_verified=args.registration_verified,allow_exploratory=args.allow_exploratory,seed=args.seed))
    elif args.command=='reference-segment':reference_segment(args)
    elif args.command=='run':run(args)
    elif args.command=='stage':run_stage(args)
    else:
        config=data.read_json(args.config or args.run/'CONFIG.json');verify_source(config['source_sha256']);agent_environment(config['model'])
        from benchmarks.full_CPJUMP1.harness.cli import native_trial
        native_trial(args)


if __name__=='__main__':main()

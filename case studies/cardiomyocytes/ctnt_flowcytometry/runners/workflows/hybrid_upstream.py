"""Fresh updated upstream plus eligible-plate SSL; no historical crop reuse."""

from pathlib import Path
import os
import subprocess
import sys
import time
from types import SimpleNamespace

from ..dataset import preparation as data
from . import experiment_runner as e2e
from . import hybrid_regression as hybrid
from . import upstream_qa as vrr_canary
from ..runtime.source_snapshot import verify_source
from ..runtime.stage_runner import invoke


def vrr_environment():
    """Pin scheduling explicitly; legacy crop still uses native Verify/Reason/Refine."""
    return dict(SHAPE_VRR_POLICY='balanced_v1',SHAPE_SEARCH_STRATEGY='four_stage',
        SHAPE_SEARCH_STRATEGY_SEGMENTATION='four_stage',SHAPE_SEARCH_STRATEGY_CROP='legacy',
        SHAPE_SEARCH_STRATEGY_REPRESENTATION='four_stage',SHAPE_SEARCH_STRATEGY_BATCH_CORRECTION='four_stage',
        SHAPE_SEGMENTATION_EARLY_STOP='1',SHAPE_SEGMENTATION_CONFIRM_STOP='valid_choice',
        SHAPE_SEGMENTATION_COMPACT_PROMPTS='1')


def adaptive_segmentation_config(seed):
    """cTnT acquisition metadata adapter; the actual policies live in SHAPE."""
    return dict(routing_policy='stratified_triage_v1', gain_policy='evidence_v1',
        native_seed=seed, triage_batch_size=4,
        grouping=dict(batch_fields=['experiment_date','plate_id'], min_group=10,
                      max_depth=2, min_scale_ratio=2**.5, min_density_ratio=2.))


def fresh_config(reference,source,cores,*,adaptive_segmentation=False,reuse_crop_delivery=False):
    config={**reference,'arm':'shape','source_sha256':source,'cores':cores,
            'deadline':None,'budget_hours':None,'smoke':False,
            'vrr_policy':'balanced_v1','internal_unscored_zero_expert':True,
            'ssl_search_policy':'evidence_cost_v1',
            'ssl_optimization':dict(parameter_inheritance='complete parent-relative effective kwargs with native audit',
                staged_unfreezing='50% at 3e-5; optional next search round resumes at 100% and 1e-5',
                stop='paired well-level delta must exceed 2 SE to force continuation; reasoner confirms uncertain stop',
                adoption='unchanged frozen-input guard plus plate-specific validation percentage MAE routing'),
            'ssl_selection':dict(mode='zero_shot',checkpoint=None,
                reason='fresh zero-shot expert; eligible-plate SSL is searched separately'),
            'historical_artifacts_used':False}
    config['vrr_environment']=vrr_environment()
    config['search_strategies']=dict(segment='four_stage',crop='legacy',embed='four_stage',batch='four_stage')
    config.pop('smoke_policy',None)
    config.pop('pilot_sha256',None)
    config.pop('segmentation_adaptive',None)
    config.pop('reuse_verified_crop_delivery',None)
    if reuse_crop_delivery:
        config['reuse_verified_crop_delivery']=True
    if adaptive_segmentation:
        config['segmentation_adaptive']=adaptive_segmentation_config(reference['seed'])
    config['shape_policy']={**config.get('shape_policy',{}),'crop_vrr_workers':cores}
    return config


def execute_full(args):
    from benchmarks.full_CPJUMP1.harness import pipeline
    from ..evaluation.stage_quality import collect
    verify_source(args.expected_source)
    os.environ.update(vrr_environment())
    # The old SHAPE CONFIG supplies the matched cohort/assets/model settings only.
    # No old crop, embedding or fitted regressor is passed into the fresh pipeline.
    reference=data.read_json(args.parent/'CONFIG.json')
    config=fresh_config(reference,args.expected_source,args.cores,
                        adaptive_segmentation=getattr(args,'adaptive_segmentation',False),
                        reuse_crop_delivery=getattr(args,'reuse_crop_delivery',False))
    if set(config['images'])!={r['image_id'] for r in data.load_prepared(args.prep)}:
        raise ValueError('full hybrid requires the exact complete prepared cohort')
    data.write_json(args.out/'FULL_PIPELINE_PLAN.json',dict(config=config,
        reference_config_sha256=data.digest(args.parent/'CONFIG.json'),
        stages=['development_canary','fresh_segment','fresh_crop','zero_expert','eligible_SSL','routing','heldout_score'],
        heldout_targets_opened_before_routes=False))
    state=data.read_json(args.out/'RUN_META.json')
    def current(stage):
        state.update(status='running',current_stage=stage,execution_mode='full_updated_pipeline')
        hybrid.persist(args.out/'RUN_META.json',state)
        print('FULL_HYBRID_STAGE_START',stage,flush=True)
    current('preflight')
    preflight=e2e.preflight(SimpleNamespace(arm='shape',model=config['model'],
        assets_root=Path(config['assets_root']),expected_source=args.expected_source,
        reasoning_profile='all-high',out=args.out/'PREFLIGHT.json'))
    data.write_json(args.out/'PREFLIGHT.json',preflight)
    current('vrr_canary');started=time.time()
    vrr_canary.run(args.prep,args.parent,args.out/'vrr_canary',config)
    hybrid.mark(args.out,'vrr_canary',time.time()-started,report='vrr_canary/REPORT.json',prediction_crops_replaced=False)
    if config.get('segmentation_adaptive'):
        from . import adaptive_qa as adaptive_canary
        current('independent_canary');started=time.time()
        adaptive_canary.run(args.prep,args.parent,args.out/'independent_canary',config)
        hybrid.mark(args.out,'independent_canary',time.time()-started,
                    report='independent_canary/REPORT.json',prediction_crops_replaced=False)
    upstream=args.out/'upstream';upstream.mkdir();(upstream/'logs').mkdir()
    data.write_json(upstream/'CONFIG.json',config);data.write_json(upstream/'PREFLIGHT.json',preflight)
    for stage in ('segment','crop','embed','batch','predict'):
        current(stage)
        data.write_json(upstream/'RUN_META.json',dict(status='running',current_stage=stage))
        env={**os.environ,**pipeline.stage_env('shape',stage,config['reasoning']['protocol'])}
        invoke(e2e.MODULE if stage in ('segment','crop') else hybrid.MODULE,
               stage,args.prep,upstream,env=env,
               style='stage' if stage in ('segment','crop') else 'named')
        if stage in ('segment','crop'):
            hybrid.reuse_file(upstream/(stage+'_COMPLETE.json'),args.out/(stage+'_COMPLETE.json'))
        # freeze() expects usage files for every native stage even when no model
        # call is needed by the deliberately frozen zero-shot inference branch.
        if not (upstream/(stage+'_USAGE.json')).exists() and stage!='predict':
            data.write_json(upstream/(stage+'_USAGE.json'),data.read_json(upstream/(stage+'_COMPLETE.json')).get('usage',{}))
    freeze_args=SimpleNamespace(expected_source=args.expected_source,prep=args.prep,out=upstream,
                                arm='shape',smoke=False,pilot_manifest=None)
    e2e.freeze(freeze_args,dict(prepare_sha256=data.digest(args.prep/'PREPARE.json')))
    current('upstream_readouts');collect(args.prep,upstream)
    data.write_json(upstream/'RUN_META.json',dict(status='frozen_unscored',current_stage='ready_for_ssl',
        heldout_targets_opened=False))
    (upstream/'UPSTREAM_FROZEN').write_text(args.expected_source+'\n')
    forwarded=SimpleNamespace(**vars(args));forwarded.parent=upstream;forwarded.full_run=True
    hybrid.execute(forwarded)

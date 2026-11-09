"""Wait for the current flow suite, prepare staining, and run the same three arms."""
import argparse
from pathlib import Path
import os
import subprocess
import sys
import time
from . import data
from .e2e import MODULE
from ...ctnt_flowcytometry.runners.runtime.source_snapshot import verify_source


def main(argv=None):
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--channels-root',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    p.add_argument('--assets-root',type=Path,required=True);p.add_argument('--api-env',type=Path,required=True)
    p.add_argument('--expected-source',required=True);p.add_argument('--after-flow',type=Path)
    p.add_argument('--cores',type=int,default=12);p.add_argument('--fraction',type=float,default=.05)
    p.add_argument('--seed',type=int,default=20261017)
    p.add_argument('--model',default='gpt-5.6-luna')
    args=p.parse_args(argv);verify_source(args.expected_source)
    data.fresh_output(args.out,args.channels_root);args.out.mkdir(parents=True)
    meta=dict(status='waiting_for_flow' if args.after_flow else 'preparing',current_stage='wait',model=args.model,
        arms=['shape','codex','control'],reasoning_effort='high',source_sha256=args.expected_source,
        source_commit=None,seed=args.seed,fraction=args.fraction,cores=args.cores,completed=[])
    def save():data.write_json(args.out/'SUITE_META.json',meta)
    save()
    try:
        if args.after_flow:
            while True:
                f=args.after_flow/'SUITE_META.json'
                if f.exists():
                    previous=data.read_json(f)
                    if previous.get('status') in ('complete','failed'):
                        meta['preceding_flow_status']=previous['status'];break
                time.sleep(30)
        from dotenv import dotenv_values
        env=dict(os.environ)
        env.update({k:v for k,v in dotenv_values(args.api_env).items() if v is not None})
        env.update(SHAPE_SEARCH_STRATEGY='four_stage',OMP_NUM_THREADS=str(args.cores),
            OPENBLAS_NUM_THREADS=str(args.cores),MKL_NUM_THREADS=str(args.cores),PYTHONUNBUFFERED='1')
        env['PATH']=str(Path.home()/'.local/bin')+os.pathsep+env.get('PATH','')
        if not env.get('LLM_API_KEY'):raise RuntimeError('LLM_API_KEY is not configured')
        from benchmarks.full_CPJUMP1.harness.gateway_route import validate
        before=dict(os.environ)
        try:
            os.environ.update(env)
            meta['gateway_route']=validate(env.get('LLM_BASE_URL',''))
        finally:
            os.environ.clear();os.environ.update(before)
        def call(stage,arguments):
            meta.update(status='running',current_stage=stage);save();print('SUITE_STAGE',stage,flush=True)
            with (args.out/(stage+'.log')).open('x') as log:
                subprocess.run([sys.executable,'-u','-m',MODULE,*arguments],cwd=data.REPO,env=env,
                    stdin=subprocess.DEVNULL,stdout=log,stderr=subprocess.STDOUT,check=True)
        prep=args.out/'prepared';masks=args.out/'reference_masks';references=args.out/'references'
        call('prepare',['prepare','--channels-root',str(args.channels_root),'--out',str(prep),'--seed',str(args.seed)])
        common=['--prep',str(prep),'--assets-root',str(args.assets_root),'--expected-source',args.expected_source,
            '--cores',str(args.cores),'--fraction',str(args.fraction),'--seed',str(args.seed)]
        call('reference_segment',['reference-segment',*common,'--out',str(masks)])
        call('reference_labels',['freeze-references','--prep',str(prep),'--masks',str(masks),'--out',str(references),
            '--kind','fixed_algorithmic_pseudo_reference','--allow-exploratory','--seed',str(args.seed)])
        for arm in meta['arms']:
            meta['current_arm']=arm;save()
            call(arm,['run',*common,'--references',str(references),'--arm',arm,'--model',args.model,'--smoke','--out',str(args.out/arm)])
            if not (args.out/arm/'CTNT_STAINING_SMOKE_COMPLETE').exists():raise RuntimeError('arm completion marker missing')
            meta['completed'].append(arm);save()
        meta.update(status='complete',current_stage='complete');meta.pop('current_arm',None);save()
        (args.out/'STAINING_SUITE_COMPLETE').write_text(args.expected_source+'\n')
    except BaseException as exc:
        meta.update(status='failed',error=type(exc).__name__+': '+str(exc));save();raise


if __name__=='__main__':main()

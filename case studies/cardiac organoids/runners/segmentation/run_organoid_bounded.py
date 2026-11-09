#!/usr/bin/env python3
"""Persistent organoid cohort shards with independent negative review and isolated bounded VRR.

This is a new schema-2 protocol; historical cohort exports/runs remain immutable.
"""
from __future__ import annotations
import argparse
from concurrent.futures import ThreadPoolExecutor
import threading
import csv
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import time
from types import SimpleNamespace
from uuid import uuid4
from cohort_runtime import RunStore,IsolatedWorker,atomic_json,digest,file_sha,completed_response


def audit_totals(root):
    counts=dict(llm_calls=0,tokens=0,unknown_usage=0,native_calls=0,tool_calls=0,incomplete_audit_records=0,tool_requests_started=0)
    for filename,key in [('LLM_CALLS.jsonl','llm_calls'),('NATIVE_CALLS.jsonl','native_calls'),('TOOL_CALLS.jsonl','tool_calls'),('TOOL_REQUESTS_STARTED.jsonl','tool_requests_started')]:
        for p in root.rglob(filename):
            for line in p.read_text().splitlines():
                if not line:continue
                try:r=json.loads(line)
                except json.JSONDecodeError:
                    if line!=p.read_text().splitlines()[-1]:raise ValueError('Invalid audit interior: '+str(p))
                    counts['incomplete_audit_records']+=1
                    if key=='llm_calls':counts['unknown_usage']+=1
                    continue  # a killed request's final partial line is preserved; its cost remains unknown
                counts[key]+=1
                if key=='llm_calls':
                    value=r.get('tokens')
                    if isinstance(value,int) and value>0:counts['tokens']+=value
                    else:counts['unknown_usage']+=1
    counts['incomplete_tool_requests']=max(0,counts['tool_requests_started']-counts['tool_calls'])
    return counts


def stage_tuple(payload):
    from shape.utils.schemas import StageResolution,VLRRResult
    resolutions=[]
    for raw in payload.get('resolutions',[]):
        raw=dict(raw);raw['verdict']=VLRRResult(**raw['verdict'])
        raw['trials']=[(p,VLRRResult(**v),r) for p,v,r in raw.get('trials',[])]
        resolutions.append(StageResolution(**raw))
    return resolutions,payload.get('result'),payload.get('vrr_status','conceded'),None


def persist_candidate_paths(value,root):
    """Attest every referenced file below the durable run (never an image/GT input)."""
    found=set()
    def visit(v):
        if isinstance(v,dict):
            for x in v.values():visit(x)
        elif isinstance(v,list):
            for x in v:visit(x)
        elif isinstance(v,str):
            try:
                p=Path(v)
                if p.is_absolute() and p.is_relative_to(root) and p.is_file():found.add(p)
            except (ValueError,OSError):pass
    visit(value)
    return found


def export_records(root,records):
    """Stable artifact bytes regardless of dictionary order after journal replay."""
    root=Path(root)
    with (root/'records.jsonl').open('w',encoding='utf-8',newline='\n') as f:
        for record in records:f.write(json.dumps(record,sort_keys=True,allow_nan=False)+'\n')
    export_rows=[]
    for record in records:
        row={k:v for k,v in record.items() if k not in ('_thresholds','managed_usage','budget_consumed_or_reserved')}
        row.update({'managed_'+k:v for k,v in record['managed_usage'].items()})
        row.update({'budget_consumed_or_reserved_'+k:v for k,v in record['budget_consumed_or_reserved'].items()})
        export_rows.append(row)
    fields=sorted({k for row in export_rows for k in row})
    with (root/'shape_cohort_bounded.csv').open('w',encoding='utf-8',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=fields,lineterminator='\n');writer.writeheader();writer.writerows(export_rows)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument('--manifest',required=True,type=Path)
    cli.add_argument('--run-dir',required=True,type=Path)
    cli.add_argument('--model-path',required=True,type=Path)
    cli.add_argument('--expected-commit',required=True)
    cli.add_argument('--resume',action='store_true')
    cli.add_argument('--image-workers',type=int,choices=[1],default=1,help='One isolated GPU model worker; parallelize fixed shards across GPUs')
    cli.add_argument('--cohort-min-group',type=int,default=10)
    cli.add_argument('--selective-review',action='store_true',help='Organoid representative visual confirmation and GT-free dark-body anomaly routing')
    cli.add_argument('--disable-native-cache',action='store_true')
    cli.add_argument('--shard-size',type=int,default=64)
    cli.add_argument('--shard-index',type=int,help='Run one fixed shard; use a distinct run-dir per array task')
    cli.add_argument('--max-image-seconds',type=float,default=300)
    cli.add_argument('--max-native-calls',type=int,default=6)
    cli.add_argument('--max-reported-tokens',type=int,default=24000)
    cli.add_argument('--max-rescaled-pixels',type=int,default=16000000)
    cli.add_argument('--cache-seed',type=int,default=0)
    args=cli.parse_args()
    if not os.environ.get('SLURM_JOB_ID'):raise RuntimeError('Run inside a Slurm GPU allocation')
    if args.cohort_min_group<3:raise ValueError('At least three cohort supports required')
    if min(args.shard_size,args.max_image_seconds,args.max_native_calls,args.max_reported_tokens,args.max_rescaled_pixels)<=0:
        raise ValueError('Positive shard and per-image limits required')
    if not 0<=args.cache_seed<2**32:raise ValueError('cache seed must be uint32')
    from common import REPO,MODEL,configure,read_manifest,source_identity,versions,protect_output,cellseg,sha256,labels
    from audit import llm_audit,safe
    from shape.agent.verification.vision_engine import build_vision_engine
    from run_organoid_segmentation import evaluate_pair
    from shape.agent.verification import segmentation_cohort as cohort
    from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
    from cohort_worker import worker_main
    settings=configure('legacy')
    if subprocess.check_output(['git','rev-parse','HEAD'],cwd=REPO,text=True).strip()!=args.expected_commit:
        raise ValueError('Unexpected inference commit')
    if subprocess.check_output(['git','status','--porcelain'],cwd=REPO,text=True).strip():raise ValueError('Dirty inference source')
    manifest=read_manifest(args.manifest,verify=False)
    pairs=manifest['pairs'];root=args.run_dir.resolve();weights=args.model_path.resolve()
    protect_output(root,pairs,weights)
    limits=dict(max_image_seconds=args.max_image_seconds,max_native_calls=args.max_native_calls,
                max_reported_tokens=args.max_reported_tokens,max_rescaled_pixels=args.max_rescaled_pixels)
    all_shards=[pairs[n:n+args.shard_size] for n in range(0,len(pairs),args.shard_size)]
    if args.shard_index is not None and not 0<=args.shard_index<len(all_shards):raise ValueError('Invalid shard index')
    selected=list(enumerate(all_shards)) if args.shard_index is None else [(args.shard_index,all_shards[args.shard_index])]
    selected_pairs=[p for _,ps in selected for p in ps]
    # An array task validates only its fixed shard, rather than rereading all 7582 inputs on every GPU.
    for pair in selected_pairs:
        for name in ('image','label'):
            if sha256(pair[name])!=pair[name+'_sha256']:raise ValueError('Selected input changed since preparation')
        if list(labels(pair['label']).shape)!=pair['shape']:raise ValueError('Selected GT geometry changed')
    cohort.MIN_GROUP=args.cohort_min_group
    seed=None if args.disable_native_cache else args.cache_seed
    profile_workers=min(4,max(1,int(os.environ.get('SLURM_CPUS_PER_TASK','1'))))
    identity=dict(schema=2,protocol='organoid_bounded_v1',manifest=manifest['fingerprint'],source=source_identity(),
        weights_sha=file_sha(weights),model=MODEL,reasoning_effort='high',settings=settings,llm_timeout=os.environ.get('SHAPE_LLM_TIMEOUT','120'),packages=versions(),python=sys.version,
        limits=limits,selective_review=args.selective_review,profile_workers=profile_workers,cohort_min_group=args.cohort_min_group,shard_size=args.shard_size,shard_index=args.shard_index,cache_seed=seed)
    config=dict(weights=str(weights),weights_sha=identity['weights_sha'],output=str(root),limits=limits,cache_seed=seed)
    with RunStore(root,identity,resume=args.resume) as store:
        worker=IsolatedWorker(worker_main,config)
        engine=build_vision_engine(MODEL)
        if engine is None:raise RuntimeError('Pinned gateway engine required')
        hardware=None
        sessions_path=root/'sessions.json'
        sessions=json.loads(sessions_path.read_text()) if sessions_path.exists() else []
        interrupted=any(s['status']=='RUNNING' for s in sessions)
        session=dict(status='RUNNING',job_id=os.environ['SLURM_JOB_ID'],host=platform.node(),hardware=hardware,observed_wall_seconds=0)
        sessions.append(session);atomic_json(sessions_path,sessions)
        started=time.monotonic();records=[]
        worker_start_lock=threading.Lock()
        def ensure_worker():
            with worker_start_lock:
                if worker.process is not None:return
                hardware=worker.start()['hardware']
                old=store.load('hardware/runtime')
                if old is not None and old!=hardware:
                    worker.stop();raise ValueError('GPU hardware differs from resumed run')
                if old is None:store.save('hardware/runtime',hardware)
                session['hardware']=hardware
        def heartbeat():
            session['observed_wall_seconds']=time.monotonic()-started
            atomic_json(sessions_path,sessions)
        try:
            card=OrganoidSegmentationAgenticVerifier()
            for shard_index,shard_pairs in selected:
                shard_scope=dict(index=shard_index,ids=[p['image_id'] for p in shard_pairs])
                shard_key=store.key('shards',shard_scope)
                complete=store.load(shard_key)
                if complete is not None:
                    records.extend(complete['records']);continue
                specs=[dict(path=p['image'],image_id=p['image_id']) for p in shard_pairs]
                profile_values={};usage_by_image={};observed_by_image={};last_by_image={}
                by_id={p['image_id']:p for p in shard_pairs}
                def read_profile(spec):
                    key=store.key('profiles',spec)
                    cached=store.load(key)
                    if cached is None:
                        ensure_worker()
                        profile=cohort.preflight_read(spec['path'],engine,spec['image_id'])
                        value=safe(profile.model_dump() if hasattr(profile,'model_dump') else vars(profile)) if profile is not None else None
                        cached=dict(profile=value)
                        store.save(key,cached)
                    value=cached['profile'];profile_values[spec['image_id']]=value
                    return SimpleNamespace(**value) if value is not None else None
                def run_one(spec,options):
                    key=store.key('passes',dict(image=spec,options=options))
                    cached=store.load(key)
                    if cached is None:
                        usage=usage_by_image.setdefault(spec['image_id'],dict(seconds=0,tokens=0,native_calls=0))
                        remaining=dict(seconds=max(0,args.max_image_seconds-usage['seconds']),tokens=max(0,args.max_reported_tokens-usage['tokens']),native_calls=max(0,args.max_native_calls-usage['native_calls']))
                        previous=sorted((root/key).glob('*/attempt.json'),key=lambda p:p.stat().st_mtime_ns)
                        folder=None
                        for marker in reversed(previous):
                            done=completed_response(marker.parent,root)
                            if done is not None and done.get('fatal'):
                                continue  # known system/auth error was not a scientific prediction
                            folder=marker.parent
                            if done is not None:
                                cached=done
                            else:
                                # The interrupted request's usage cannot safely be reset. Reserve the
                                # remainder of its image budget and expose it for review, never as fake tokens.
                                cached=dict(status='INTERRUPTED',error='Unconfirmed interrupted pass; remaining budget reserved',
                                    seconds=0,budget_charge=json.loads(marker.read_text())['remaining'],unmeasured_attempt=True)
                            break
                        if cached is None:
                            if min(remaining.values())>0:ensure_worker()
                            folder=root/key/uuid4().hex;folder.mkdir(parents=True)
                            atomic_json(folder/'attempt.json',dict(status='STARTED',image_id=spec['image_id'],remaining=remaining,options=options))
                            if min(remaining.values())<=0:
                                cached=dict(status='BUDGET',error='Cumulative image budget exhausted',seconds=0)
                            else:
                                profile=profile_values[spec['image_id']]
                                override=profile if profile is not None else 'unavailable (no confident raw-image profile)'
                                cached=worker.request(dict(action='run',folder=str(folder),image=spec['path'],image_id=spec['image_id'],options=options,profile=override,remaining=remaining),seconds=remaining['seconds'])
                        if cached.get('fatal'):raise RuntimeError(cached.get('error','Fatal worker error'))
                        observed=audit_totals(folder)
                        cached.update({k:observed[k] for k in ('tokens','native_calls','tool_calls','llm_calls','unknown_usage')})
                        if cached.get('unknown_usage') and cached['status']!='DONE':
                            cached.setdefault('budget_charge',remaining)
                            cached['unmeasured_attempt']=True
                        if cached['status']!='DONE':
                            candidate=folder/'candidate.json'
                            saved=json.loads(candidate.read_text()) if candidate.exists() else {}
                            cached.update(result=saved.get('result') or last_by_image.get(spec['image_id']),vrr_status='conceded',resolutions=[],needs_review=True,retained_candidate_basis=saved.get('selection_basis'))
                        store.save(key,cached,persist_candidate_paths(cached,root))
                        heartbeat()
                    if cached.get('result'):
                        last_by_image[spec['image_id']]=cached['result']
                    # Persist an evaluable per-pass prediction before the rest of this shard finishes.
                    # Scoring is never returned to the worker, grouping or VRR.
                    result=cached.get('result') or {}
                    mask=result.get('mask_path') or result.get('organoid_mask_path')
                    image_folder=root/'images'/spec['image_id'];image_folder.mkdir(parents=True,exist_ok=True)
                    provisional=evaluate_pair(by_id[spec['image_id']],cellseg.load_mask(mask) if mask else None,image_folder)
                    provisional.update(provisional=True,vrr_status=cached.get('vrr_status','conceded'))
                    atomic_json(image_folder/'record.provisional.json',provisional)
                    usage=usage_by_image.setdefault(spec['image_id'],dict(seconds=0,tokens=0,native_calls=0))
                    charge=cached.get('budget_charge') or dict(seconds=cached['seconds'],tokens=cached.get('tokens',0),native_calls=cached.get('tool_calls',0))
                    usage['seconds']+=charge['seconds'];usage['tokens']+=charge['tokens']
                    usage['native_calls']+=charge['native_calls']
                    measured=observed_by_image.setdefault(spec['image_id'],dict(seconds=0,tokens=0,native_calls=0,tool_calls=0,llm_calls=0,unmeasured_attempts=0))
                    for field in ('seconds','tokens','native_calls','tool_calls','llm_calls'):
                        measured[field]+=cached.get(field,0)
                    measured['unmeasured_attempts']+=int(bool(cached.get('unmeasured_attempt')))
                    return stage_tuple(cached)
                def map_images(items,fn):
                    if fn is read_profile and profile_workers>1:
                        with ThreadPoolExecutor(max_workers=min(profile_workers,max(1,len(items)))) as pool:
                            return list(pool.map(fn,items))
                    return [fn(s) for s in items]
                profile_audit=root/'profile_audits'/uuid4().hex;profile_audit.mkdir(parents=True)
                # One process-wide audit context, never nested patches from profile threads.
                with llm_audit(profile_audit):
                    runs,routing=cohort.run(specs,run_one,map_images,read_profile,
                        card.defaults(),card.levers,workers=1,tool_name='Organoid_Segmenter_Tool',organoid_selective_review=args.selective_review)
                shard_records=[]
                for pair,(resolutions,result,status,_) in zip(shard_pairs,runs):
                    result=result or {};details=resolutions[-1].details if resolutions else {}
                    path=result.get('mask_path') or result.get('organoid_mask_path')
                    folder=root/'images'/pair['image_id'];folder.mkdir(parents=True,exist_ok=True)
                    prediction=cellseg.load_mask(path) if path else None
                    record=evaluate_pair(pair,prediction,folder)
                    record.update(vrr_status=status,operational_success=status=='satisfied',
                        needs_review=status!='satisfied' or bool(details.get('needs_review') or details.get('accepted_with_flags')),
                        qc_pass=details.get('qc_pass'),quality_certified=False,negative_confirmed=bool(details.get('negative_confirmed')),
                        parameter_source=routing['parameter_source'][pair['image_id']],shard_index=shard_index,
                        managed_usage=observed_by_image.get(pair['image_id'],{}),
                        budget_consumed_or_reserved=usage_by_image.get(pair['image_id'],{}),arm='shape_cohort_bounded',
                        source_commit=identity['source']['commit'],source_sha256=identity['source']['source_sha256'],
                        manifest_fingerprint=manifest['fingerprint'],checkpoint_sha256=identity['weights_sha'],
                        image_sha256=pair['image_sha256'],ground_truth_sha256=pair['label_sha256'],
                        image_height_px=pair['shape'][0],image_width_px=pair['shape'][1])
                    atomic_json(folder/'record.json',record);shard_records.append(record)
                complete=dict(records=shard_records,routing=safe(routing))
                store.save(shard_key,complete,[root/'images'/r['image_id']/'mask.tif' for r in shard_records if r['delivered']])
                records.extend(shard_records);heartbeat()
                atomic_json(root/'progress.json',dict(n_planned=len(selected_pairs),n_recorded=len(records),completed_shard=shard_index))
            heartbeat()
            totals=audit_totals(root)
            summary=dict(schema=2,n_planned=len(selected_pairs),n_recorded=len(records),n_delivered=sum(r['delivered'] for r in records),
                n_operational_success=sum(r['operational_success'] for r in records),n_needs_review=sum(r['needs_review'] for r in records),
                n_failed=sum(not r['delivered'] for r in records),unmeasured_interrupted_passes=sum(r['managed_usage'].get('unmeasured_attempts',0) for r in records),parameter_sources={s:sum(r['parameter_source']==s for r in records) for s in cohort.SOURCES},
                **totals,wall_seconds_observed=sum(s['observed_wall_seconds'] for s in sessions),unmeasured_interrupted_tail=interrupted,
                cost_scope='Includes worker gateway preflights, grouping and all pass attempts, including interrupted attempts with recorded usage. Reported-token stop may overshoot by one in-flight engine call, including internal structured fallback/SDK retries; unknown usage remains unknown.')
            for k in ('ap50','ap75','map_50_95'):summary[k]=sum(r[k] for r in records)/len(records)
            empty=[r for r in records if r['ground_truth_cell_count']==0]
            summary['empty_label_diagnostics']=dict(n=len(empty),
                empty_prediction=sum(r['delivered'] and r['predicted_cell_count']==0 for r in empty),
                nonempty_prediction=sum(r['delivered'] and r['predicted_cell_count']>0 for r in empty),
                negative_confirmed=sum(r['negative_confirmed'] for r in empty),review_required=sum(r['needs_review'] for r in empty))
            atomic_json(root/'summary.json',summary)
            export_records(root,records)
            session['status']='COMPLETE_WITH_FLAGS' if summary['n_needs_review'] else 'COMPLETE'
            atomic_json(root/'run_meta.json',dict(identity=identity,sessions=sessions,status=session['status'],summary=summary))
            print(json.dumps(summary),flush=True)
            return 2 if summary['n_needs_review'] else 0
        except BaseException as exc:
            session['status']='INTERRUPTED';session['error']=safe(f'{type(exc).__name__}: {exc}')
            raise
        finally:
            heartbeat();worker.stop()

if __name__=='__main__':raise SystemExit(main())

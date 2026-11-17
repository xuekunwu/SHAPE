"""A persistent single-GPU child; no GT is sent to this inference process."""
from __future__ import annotations
from dataclasses import asdict
import json
import os
from pathlib import Path
import time
from uuid import uuid4
from cohort_runtime import atomic_json,completed_response


def worker_main(conn,config):
    # Imports and CUDA initialization happen in the spawned child only.
    from common import configure,MODEL,sha256,append
    from audit import llm_audit,tool_audit,safe
    from shape.engine.errors import is_fatal_llm_error
    try:
        configure('legacy')
        os.environ['SHAPE_ORGANOID_MODEL_PATH']=config['weights']
        os.environ['ORGANOID_EXPECTED_SHA256']=config['weights_sha']
        os.environ['ORGANOID_REQUIRE_GPU']='1'
        os.environ['SHAPE_ORGANOID_ADAPTIVE_RESCUE']='0'
        os.environ['SHAPE_ORGANOID_MAX_RESCALED_PIXELS']=str(config['limits']['max_rescaled_pixels'])
        import torch
        from shape.agent.executor import Executor
        from shape.agent.verification.integration import verify_managed_tool
        from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
        from shape.solver import _gpu_slots
        from arms import gateway_preflight,probe_dependencies
        probe_dependencies('shape')
        if not torch.cuda.is_available():raise RuntimeError('CUDA required')
        if sha256(config['weights'])!=config['weights_sha']:raise RuntimeError('CO checkpoint changed')
        tool=Organoid_Segmenter_Tool(model_path=config['weights']);tool._load_model()
        if not str(tool.device).startswith('cuda'):raise RuntimeError('GPU model required')
        executor=Executor(MODEL,query_cache_dir=config['output'],enable_signal=False)
        executor._managed_tools['Organoid_Segmenter_Tool']=tool
        executor.gpu_slots=_gpu_slots()
        probe=Path(config['output'])/'worker_preflights'/uuid4().hex;probe.mkdir(parents=True)
        with llm_audit(probe):engine=gateway_preflight(probe)
        executor._verifier_llm_mm,executor._verifier_llm_tried=engine,True
        conn.send(dict(status='READY',hardware=dict(gpu=torch.cuda.get_device_name(0),cuda=torch.version.cuda,
                       gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory)))
    except Exception as exc:
        conn.send(dict(status='ERROR',error=safe(f'{type(exc).__name__}: {exc}')));return
    while True:
        try:request=conn.recv()
        except EOFError:return
        if request.get('action')=='close':return
        folder=Path(request['folder']);folder.mkdir(parents=True,exist_ok=True)
        started=time.monotonic()
        calls=0
        original=executor.run_managed_tool
        max_native=request['remaining']['native_calls']
        def bounded(name,base,params):
            nonlocal calls
            if calls>=max_native or time.monotonic()-started>=request['remaining']['seconds']:
                return {'error':'organoid native/time budget exhausted','status':'failed'}
            calls+=1
            append(folder/'TOOL_REQUESTS_STARTED.jsonl',dict(image_id=request['image_id'],request_index=calls,started_unix=time.time()))
            return original(name,base,params)
        def candidate(value):atomic_json(folder/'candidate.json',safe(value))
        executor.run_managed_tool=bounded
        executor.set_query_cache_dir(str(folder/'managed'))
        executor._managed_tools['Organoid_Segmenter_Tool']=tool
        if config['cache_seed'] is not None:
            executor.organoid_trial_cache={'root':str(Path(config['output'])/'native_trials'),'seed':config['cache_seed']}
        options={**(request.get('options') or {}),
            'preflight_override':request['profile'],'candidate_checkpoint':candidate,
            'max_native_calls':max_native,'max_reported_tokens':request['remaining']['tokens'],
            'max_image_seconds':request['remaining']['seconds']}
        llm,native,tools=[],[],[]
        try:
            with llm_audit(folder) as llm,tool_audit(folder,Organoid_Segmenter_Tool) as (tools,native):
                resolutions,result,status=verify_managed_tool(executor,'Organoid_Segmenter_Tool',
                    {'image':request['image'],'image_id':request['image_id']},
                    request='Segment each whole cardiac organoid or spheroid as one instance, preserving border targets. A true negative may have no organoids.',
                    verifier_options=options)
            if any(not str(n['device']).startswith('cuda') or n['weights_sha256']!=config['weights_sha'] for n in native):
                raise RuntimeError('Native GPU/checkpoint attestation failed')
            response=dict(status='DONE',vrr_status=status,result=result,resolutions=[asdict(r) for r in resolutions],
                seconds=time.monotonic()-started,native_calls=len(native),tool_calls=len(tools),llm_calls=len(llm),
                tokens=sum(r.get('tokens') or 0 for r in llm),unknown_usage=sum(not isinstance(r.get('tokens'),int) or r['tokens']<=0 for r in llm))
            response=safe(response)
            atomic_json(folder/'stage.json',response)
            completed_response(folder,config['output'],response)
            try:conn.send(response)
            except (OSError,EOFError):return
        except Exception as exc:
            response=dict(status='ERROR',error=safe(f'{type(exc).__name__}: {exc}'),fatal=is_fatal_llm_error(exc),seconds=time.monotonic()-started)
            atomic_json(folder/'stage.json',response)
            completed_response(folder,config['output'],response)
            try:conn.send(response)
            except (OSError,EOFError):return
            if response['fatal']:return
        finally:
            executor.run_managed_tool=original

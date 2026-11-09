"""Native Cell_Segmenter adapter for the shared conditional VRR controller."""
import json
from pathlib import Path
import threading
import time

from pydantic import BaseModel, Field


class Condition(BaseModel):
    metric: str
    min: float | None = None
    max: float | None = None


class Rule(BaseModel):
    issues: list[str] = Field(default_factory=list)
    items: list[str] = Field(default_factory=list)
    when: list[Condition] = Field(default_factory=list)
    params_json: str
    expected_effect_json: str
    reason: str


class Review(BaseModel):
    image_id: str
    candidate: str
    baseline: str
    confidence: str
    observations: list[str]
    reason: str
    loss_explanation: str = ''
    accept_with_flags: bool = False
    normal_heterogeneity: bool = False
    image_quality_sufficient: bool = False
    visible_faults: list[str] = Field(default_factory=list)
    evidence_views: list[str] = Field(default_factory=list)


class Decision(BaseModel):
    reviews: list[Review] = Field(default_factory=list)
    rules: list[Rule] = Field(default_factory=list)
    stop: bool = False
    reason: str


def recorded_generate(engine,prompt,cards,root,number,model_identity,evidence):
    """Preserve every invocation, even when later decision validation fails."""
    import hashlib
    from shape.tools.cell_segmenter.trial_cache import atomic_json
    root=Path(root)
    atomic_json(root/f'REQUEST_{number:02}.json',dict(prompt=prompt,
        prompt_sha256=hashlib.sha256(prompt.encode()).hexdigest(),schema=Decision.model_json_schema(),
        requested_model=model_identity,evidence=evidence))
    response=None;error=None;started=time.monotonic()
    try:
        response=engine.generate([prompt,*cards],response_format=Decision)
        return response
    except Exception as exc:
        error={'type':type(exc).__name__};raise
    finally:
        raw=response.get('content') if isinstance(response,dict) else response
        raw=raw.model_dump() if hasattr(raw,'model_dump') else raw
        atomic_json(root/f'RECEIPT_{number:02}.json',dict(
            status='request_error' if error else 'response_received',error=error,
            seconds=time.monotonic()-started,raw_answer=raw,
            usage=response.get('usage') if isinstance(response,dict) else None,
            requested_model=model_identity,decision_reused=False))


def run_native(items, tool, defaults, engine, root, *, task, model_identity, workers=1, max_rounds=8):
    """Only caller-supplied segmentation images enter native or visual evidence.

    The tool's own cache binds native pixels and kwargs. Decisions are never
    cached. Input/output fingerprints and all candidate histories are recorded.
    """
    from .segmentation_shared import run, identity
    from .segmentation_triage import evidence_card
    from .verifiers.segmentation import SegmentationAgenticVerifier
    from shape.engine.errors import is_fatal_llm_error
    from shape.tools.cell_segmenter.trial_cache import atomic_json, digest
    from shape.utils.schemas import VLRRResult
    from shape.utils.image_processor import ImageProcessor
    import numpy as np
    import tifffile

    root = Path(root); root.mkdir(parents=True,exist_ok=True)
    if not model_identity:raise ValueError('explicit requested model identity required')
    by_id = {i['image_id']:i for i in items};lock=threading.Lock()
    binding=dict(images={i['image_id']:digest(Path(i['path'])) for i in items},defaults=defaults,
        task=task,model_identity=model_identity,max_rounds=max_rounds,controller_sha256=digest(Path(__file__).with_name('segmentation_shared.py')),
        adapter_sha256=digest(Path(__file__)))
    binding_file=root/'INPUTS.json'
    if binding_file.exists() and json.loads(binding_file.read_text())!=binding:
        raise ValueError('shared VRR inputs or implementation changed')
    atomic_json(binding_file,binding)
    resume=json.loads((root/'CHECKPOINT.json').read_text()) if (root/'CHECKPOINT.json').exists() else None
    if resume:
        for ident,values in resume['history'].items():
            for observation in values.values():
                if observation['image_sha256']!=binding['images'][ident]:
                    raise ValueError('checkpoint image identity changed')
                if digest(Path(observation['result']['mask_path']))!=observation['mask_sha256']:
                    raise ValueError('checkpoint native mask changed')
    validator=SegmentationAgenticVerifier()
    def coerce(key,value):
        if key not in defaults or key not in validator.levers:
            raise ValueError('unknown segmentation lever: '+key)
        return validator.coerce(key,value)
    def measure(item,params):
        if digest(Path(item['path']))!=binding['images'][item['image_id']]:
            raise ValueError('segmentation input pixels changed')
        folder=root/item['image_id']/identity(params);folder.mkdir(parents=True,exist_ok=True)
        with lock:
            result=tool.execute(image=item['path'],image_id=item['image_id'],model_type='cpsam',
                bsize=256,query_cache_dir=str(folder),**params)
        if result.get('error') or not result.get('mask_path'):
            raise RuntimeError(result.get('error') or 'native call has no mask')
        v=SegmentationAgenticVerifier();v._input_path=item['path']
        metrics=v.quantify(result,dict(trial_index=1,image_path=item['path']))
        verdict=VLRRResult('PASS',v.qc_score(metrics),metrics,action='retune_here')
        problems=v.guard(None,verdict)+v.measured_problems(verdict,params.get('diameter'),v._colors.get(1),
            params.get('color_class_filter'),v._prefilter_colors.get(1))
        evidence=dict(result=result,metrics=metrics,score=verdict.scalar_score,
            problems=problems,guard_ok=not v.guard(None,verdict),
            mask_sha256=digest(Path(result['mask_path'])),image_sha256=digest(Path(item['path'])))
        if evidence['image_sha256']!=binding['images'][item['image_id']]:
            raise ValueError('segmentation input changed during execution')
        atomic_json(folder/'MEASUREMENT.json',evidence)
        return evidence

    def propose(packet):
        # Rotate evidence coverage. Tables still include every unresolved image;
        # only visually shown IDs may receive an explicit pairwise review.
        ids=packet['unresolved'];start=((packet['round']-1)*8)%len(ids)
        shown=(ids[start:]+ids[:start])[:8];cards=[];bindings={};evidence=[]
        for item in packet['items']:
            ident=item['image_id']
            if ident not in shown:continue
            current=item['current']; candidates=[current]
            other=[o for o in item['history'] if o['candidate']!=current['candidate']]
            if other:candidates.append(other[-1])
            bindings[ident]={o['candidate'] for o in candidates}
            source=Path(by_id[ident]['path'])
            if digest(source)!=binding['images'][ident]:raise ValueError('visual input pixels changed')
            raw=ImageProcessor.load_image(str(source)).data
            if digest(source)!=binding['images'][ident]:raise ValueError('visual input changed while reading')
            for o in candidates:
                if digest(Path(o['result']['mask_path']))!=o['mask_sha256']:
                    raise ValueError('candidate mask changed before visual review')
                mask=np.load(o['result']['mask_path'],allow_pickle=False)
                if np.any(mask):
                    card,metadata=evidence_card(raw,mask,ident+' '+o['candidate'][:12])
                else:
                    # A zero baseline still needs visible raw evidence before a
                    # later candidate may claim to have recovered missed cells.
                    import io
                    from PIL import Image,ImageDraw
                    from .segmentation_triage import _rgb
                    try:rgb=_rgb(raw)
                    except ValueError:rgb=np.zeros((*mask.shape,3),np.uint8)
                    canvas=Image.new('RGB',(768,430),'white')
                    picture=Image.fromarray(rgb);picture.thumbnail((384,384))
                    canvas.paste(picture,(0,30));canvas.paste(picture,(384,30))
                    ImageDraw.Draw(canvas).text((8,8),ident+' '+o['candidate'][:12]+' | RAW / zero-mask overlay; no size-extreme objects',fill='black')
                    stream=io.BytesIO();canvas.save(stream,format='PNG');card=stream.getvalue()
                    metadata={'views':['raw_overview','mask_overlay']}
                file=root/f"round_{packet['round']:02}_{ident}_{o['candidate'][:12]}.png"
                file.write_bytes(card);cards.append(card)
                evidence.append(dict(image_id=ident,candidate=o['candidate'],path=str(file),
                    sha256=digest(file),views=metadata['views']))
        columns=['image_id','candidate','score','problems','cell_count','foreground_coverage',
                 'tiny_mask_fraction','large_mask_fraction','mean_solidity','mask_diameter_px']
        metric_columns=columns[4:];columns.append('hypothesis_checks')
        rows=[]
        for i in packet['items']:
            for o in i['history']:
                rows.append([i['image_id'],o['candidate'],o['score'],o['problems'],
                    *[o['metrics'].get(k) for k in metric_columns],o.get('hypothesis_checks')])
        prompt=('One shared segmentation VRR over all unresolved images, at most eight replies total. '
            'Return conditional rules by problem and measured morphology; params_json is a JSON object '
            'of lever values; expected_effect_json maps native readouts to up/down/preserve. '
            'A failed expected effect refutes that hypothesis even when another measured benefit is retained. '
            'of documented tool levers. Parameters inherit from each image incumbent. Passing images '
            'are frozen. Do not optimize count/coverage alone or infer biological labels. Review only '
            'the exact candidate and baseline overlays attached. At least two distinct visual observations '
            'are required. Accept size alerts with flags only for visibly normal heterogeneity with '
            'no actionable faults. Explain any >25% count/coverage loss. Unshown images cannot receive '
            'a visual review, but rules may be tested natively on them. stop=true requires no rules. '
            'Task: '+task+'\nTool contract:\n'+validator._card_section()+
            '\n'+json.dumps(dict(round=packet['round'],remaining=packet['rounds_remaining'],
                columns=columns,rows=rows,incumbents={i['image_id']:i['current']['candidate'] for i in packet['items']},
                native_errors=packet['errors'],recent_decisions=packet['recent_decisions'],shown=evidence),default=str))
        response=recorded_generate(engine,prompt,cards,root,packet['round'],model_identity,evidence)
        content=response.get('content') if isinstance(response,dict) else response
        if isinstance(content,str):content=json.loads(content)
        decision=Decision.model_validate(content.model_dump() if hasattr(content,'model_dump') else content).model_dump()
        # A claimed image ID is insufficient: both actual cards must exist.
        actual={(r['image_id'],r['candidate']) for r in evidence}
        for review in decision['reviews']:
            if any((review['image_id'],review[k]) not in actual for k in ('candidate','baseline')):
                raise ValueError('visual comparison references evidence not shown')
        for rule in decision['rules']:
            rule['params']=json.loads(rule.pop('params_json'))
            rule['expected_effect']=json.loads(rule.pop('expected_effect_json'))
            if not isinstance(rule['params'],dict):raise ValueError('params_json must be an object')
            rule['when']=[{k:v for k,v in c.items() if v is not None} for c in rule['when']]
        atomic_json(root/f"ROUND_{packet['round']:02}.json",dict(decision=decision,evidence=evidence,
            usage=response.get('usage') if isinstance(response,dict) else None))
        return decision

    return run(items,defaults,measure,propose,coerce,workers=workers,max_rounds=max_rounds,
        checkpoint=lambda record:atomic_json(root/'CHECKPOINT.json',record),resume=resume,fatal=is_fatal_llm_error)

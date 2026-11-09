"""Development-only upstream QA; never supplies the hybrid prediction crops."""

import hashlib
import os
from pathlib import Path
import subprocess
import sys
import time

from ..dataset import preparation as data
from ..runtime.stage_runner import invoke


def select_images(rows):
    """One deterministic training image per plate, without labels or scores."""
    selected = {}
    for row in sorted(rows, key=lambda r: hashlib.sha256(
            ('balanced-v1-canary|' + r['image_id']).encode()).hexdigest()):
        if row['usage'] == 'development_train':
            selected.setdefault(row['plate_id'], row)
    return [selected[p] for p in sorted(selected)]


def decision_failures(out, ids):
    failures=[]
    for stage in ('segment','crop'):
        for ident in sorted(ids):
            path=out/stage/ident/'SEARCH.json'
            if not path.exists():
                failures.append(dict(stage=stage,image_id=ident,reason='missing VRR record'));continue
            record=data.read_json(path)
            if record.get('reasoner_failures') or record.get('fallback_rounds') or record.get('execution_errors'):
                failures.append(dict(stage=stage,image_id=ident,reason='recorded VRR execution failure'))
    return failures


def run(prep, parent, out, config):
    from benchmarks.full_CPJUMP1.harness import pipeline
    rows = data.load_prepared(prep)
    wanted = set(config['images'])
    selected = select_images([r for r in rows if r['image_id'] in wanted])
    if not selected:
        raise ValueError('no development_train images for VRR canary')
    out.mkdir(exist_ok=False)
    (out/'logs').mkdir()
    canary = {**config, 'arm': 'shape', 'images': [r['image_id'] for r in selected],
              'smoke': False, 'segmentation_regime': 'development-only per-image VRR canary',
              'smoke_policy': {'shape_search_images': len(selected), 'shape_search_rounds': 8},
              'scientific_role': 'upstream QA only; crops never enter hybrid prediction'}
    data.write_json(out/'CONFIG.json', canary)
    data.write_json(out/'MANIFEST.json', {
        'images': [{k:r[k] for k in ('image_id','well','plate_id','usage')} for r in selected],
        'selection': 'one SHA256-ordered training image per plate; metadata only',
        'heldout_images_used': False, 'prediction_crops_replaced': False,
        'comparison_limit': 'Original parent used cohort calibration; this is per-image QA, not a matched efficacy benchmark'})
    started = time.time()
    for stage in ('segment','crop'):
        data.write_json(out/'RUN_META.json', {'status':'running','current_stage':stage})
        env = {**os.environ, **pipeline.stage_env('shape',stage,config['reasoning']['protocol'])}
        invoke('case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows.experiment_runner',stage,prep,out,env=env)
    ids = set(canary['images'])
    original = [r for r in data.read_json(parent/'segmentation.json') if r['image_id'] in ids]
    new = data.read_json(out/'segmentation.json')
    original_cells = [r for r in data.read_json(parent/'cells.json') if r['image_id'] in ids]
    cells = data.read_json(out/'cells.json')
    failures=decision_failures(out,ids)
    report = {'status':'complete','source_sha256':config['source_sha256'],
              'vrr_policy':config['vrr_policy'],'seconds':time.time()-started,
              'prediction_crops_replaced':False,
              'segmentation_objects':{'parent':sum(r['n_cells'] for r in original),
                                      'canary':sum(r['n_cells'] for r in new)},
              'retained_crops':{'parent':len(original_cells),'canary':len(cells)},
              'vrr_execution_failures':failures,
              'stages':{s:data.read_json(out/(s+'_COMPLETE.json')) for s in ('segment','crop')},
              'interpretation':'Counts are diagnostics, not ground-truth quality. No claim of superior prediction from this canary.'}
    data.write_json(out/'REPORT.json',report)
    if failures:
        report['status']='failed';data.write_json(out/'REPORT.json',report)
        data.write_json(out/'RUN_META.json',dict(status='failed',current_stage='vrr_validation'))
        raise RuntimeError('VRR canary execution failed; full pipeline not started')
    data.write_json(out/'RUN_META.json',{'status':'complete','current_stage':'crop'})
    return report

"""Independent training-well QA for the opt-in core segmentation policies.

No cTnT targets are loaded, no crops enter prediction, and counts are not accuracy.
The same frozen image manifest is run through legacy and adaptive segmentation
and then the unchanged crop stage, serially, with separate native trial stores.
"""

import hashlib
import os
import subprocess
import sys
import time

from . import preparation as data
from .stage_runner import invoke
from . import upstream_qa as vrr_canary


def select_images(rows, *, per_plate=10):
    old = vrr_canary.select_images(rows)
    excluded = {r['well'] for r in old}
    plates = sorted({r['plate_id'] for r in rows if r['usage']=='development_train'})
    selected, support = [], {}
    for plate in plates:
        eligible = [r for r in rows if r['usage']=='development_train'
                    and r['plate_id']==plate and r['well'] not in excluded]
        # Round-robin wells before selecting another image from the same well.
        by_well = {}
        for row in sorted(eligible, key=lambda r: hashlib.sha256(
                ('adaptive-independent-canary-v1|'+r['image_id']).encode()).hexdigest()):
            by_well.setdefault(row['well'], []).append(row)
        ordered = [rows[index] for index in range(max(map(len,by_well.values()),default=0))
                   for _,rows in sorted(by_well.items()) if index<len(rows)]
        support[plate] = dict(eligible_images=len(eligible), eligible_wells=len(by_well),
                             included=len(eligible)>=per_plate)
        if len(eligible)>=per_plate:
            selected.extend(ordered[:per_plate])
    if not selected:
        raise ValueError('no independent training-well cohort has minimum calibration support')
    return selected, dict(excluded_previous_canary_wells=sorted(excluded), plate_support=support,
                         per_plate=per_plate, selection='fixed SHA256 order, round-robin training wells')


def failures(folder, ids):
    found = []
    for ident in sorted(ids):
        for stage in ('segment','crop'):
            records = [folder/stage/ident/'SEARCH.json']
            if stage=='segment':
                records.append(folder/stage/ident/'COHORT_SEARCH.json')
            existing = [p for p in records if p.exists()]
            if not existing:
                found.append(dict(image_id=ident,stage=stage,reason='missing VRR evidence'))
            for path in existing:
                record = data.read_json(path)
                if record.get('reasoner_failures') or record.get('fallback_rounds') or record.get('execution_errors'):
                    found.append(dict(image_id=ident,stage=stage,reason='recorded VRR execution failure'))
    cohort_file=folder/'segment_COHORT.json'
    if cohort_file.exists():
        for group in data.read_json(cohort_file).get('groups',[]):
            if group.get('triage_error'):
                found.append(dict(stage='segment',reason='visual triage failure',error=group['triage_error']))
    for path in (folder/'segment'/'triage').glob('*/REQUEST.json'):
        if data.read_json(path).get('render_errors'):
            found.append(dict(stage='segment',reason='triage rendering failure',record=str(path)))
    for path in (folder/'segment'/'triage').glob('*/RECEIPT.json'):
        if data.read_json(path).get('error'):
            found.append(dict(stage='segment',reason='triage model failure',record=str(path)))
    return found


def run(prep, parent, out, config):
    from benchmarks.full_CPJUMP1.harness import pipeline
    if not config.get('segmentation_adaptive'):
        raise ValueError('independent adaptive canary requires explicit adaptive config')
    rows = [r for r in data.load_prepared(prep) if r['image_id'] in set(config['images'])]
    selected, selection = select_images(rows)
    ids = {r['image_id'] for r in selected}
    out.mkdir(exist_ok=False)
    manifest=dict(**selection, images=[{k:r[k] for k in ('image_id','well','plate_id','usage')} for r in selected],
        source_sha256=config['source_sha256'], public_metadata_sha256=data.digest(prep/'public'/'images.json'),
        targets_used=False, prediction_crops_replaced=False,
        order=['legacy','adaptive'],
        interpretation='Independent of earlier canary wells; supported plates only. Development QA, not held-out efficacy.')
    data.write_json(out/'MANIFEST.json',manifest)
    results={}
    for variant in manifest['order']:
        folder=out/variant;folder.mkdir();(folder/'logs').mkdir()
        candidate={**config,'images':sorted(ids),'scientific_role':'independent development QA only'}
        candidate.pop('smoke_policy',None)
        if variant=='legacy':
            candidate.pop('segmentation_adaptive',None)
            candidate.pop('reuse_verified_crop_delivery',None)
        data.write_json(folder/'CONFIG.json',candidate)
        started=time.time()
        for stage in ('segment','crop'):
            env={**os.environ,**pipeline.stage_env('shape',stage,config['reasoning']['protocol'])}
            invoke('case studies.cardiomyocytes.ctnt_flowcytometry.experiment_runner',stage,prep,folder,env=env)
        issues=failures(folder,ids)
        results[variant]=dict(seconds=time.time()-started,failures=issues,
            stages={stage:data.read_json(folder/(stage+'_COMPLETE.json')) for stage in ('segment','crop')},
            segmented_objects=sum(r['n_cells'] for r in data.read_json(folder/'segmentation.json')),
            retained_crops=len(data.read_json(folder/'cells.json')))
        data.write_json(out/'REPORT.json',dict(status='running',variants=results,manifest_sha256=data.digest(out/'MANIFEST.json')))
        if issues:
            data.write_json(out/'REPORT.json',dict(status='failed',variants=results,
                manifest_sha256=data.digest(out/'MANIFEST.json')))
            raise RuntimeError('independent canary recorded VRR failures; do not start full adaptive pipeline')
    report=dict(status='complete',variants=results,manifest_sha256=data.digest(out/'MANIFEST.json'),
        prediction_crops_replaced=False,targets_used=False,
        interpretation='Counts and development timings do not prove segmentation accuracy or superior held-out prediction.')
    data.write_json(out/'REPORT.json',report)
    return report

"""Trusted reference freezing and exact, private three-channel crop export."""
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import multiprocessing
from pathlib import Path
import numpy as np
import tifffile

from . import data
from .fluorescence import MeasurementProtocol, measure, fit_threshold, apply_threshold, nuclear_qc, nuclei_from_dapi


def freeze(prep,masks,out,*,kind,registration_verified,allow_exploratory=False,seed=20261017):
    """Reference masks are fixed before compared arms; no per-arm label redefinition."""
    data.fresh_output(out,prep)
    if kind not in ('curated_independent','fixed_algorithmic_pseudo_reference'):raise ValueError('unknown reference provenance')
    if (kind!='curated_independent' or not registration_verified) and not allow_exploratory:
        raise ValueError('unverified/pseudo references require explicitly exploratory evaluation')
    rows=data.load_prepared(prep);private=data.private_images(prep);selected=data.read_json(masks/'MASKS.json')
    known={r['image_id']:r for r in rows}
    if not selected or not set(selected)<=set(known):raise ValueError('unknown reference images')
    protocol=MeasurementProtocol();out.mkdir(parents=True)
    records=[];manifest={};dapi_qc={}
    for ident,entry in sorted(selected.items()):
        row=known[ident];path=Path(entry['mask'])
        if data.digest(path)!=entry['sha256']:raise ValueError('reference mask changed')
        mask=tifffile.imread(path)
        channels={}
        for name in ('DAPI','TroponinT'):
            source=private[ident]['channels'][name];p=data.within(prep/'private',source['file'])
            if data.digest(p)!=source['sha256']:raise ValueError('reference fluorescence changed')
            channels[name]=tifffile.imread(p)
        measured,nuclei,qc=measure(mask,channels['TroponinT'],channels['DAPI'],protocol)
        destination=out/'masks'/(ident+'.tif');data.link_verified(path,destination,entry['sha256'])
        dapi_qc[ident]=qc
        manifest[ident]=dict(mask=str(destination.resolve()),sha256=data.digest(destination),kind=kind,
            registration_verified=registration_verified,well=row['well'],usage=row['usage'],
            no_reference_detections=not bool(np.any(mask)))
        for r in measured:
            r.update(image_id=ident,reference_cell_id=f'{ident}:r{r["mask_label"]}',well=row['well'],plate=row['plate'],usage=row['usage'])
            records.append(r)
        print(f'REFERENCE {len(manifest)}/{len(selected)} {ident}',flush=True)
    # Calibration is frozen before held-out labels are assigned. fit_threshold filters train only.
    threshold=fit_threshold(records,protocol.identity(),seed=seed)
    data.write_json(out/'THRESHOLD.json',threshold)
    labels=apply_threshold(records,threshold)
    data.write_json(out/'MEASUREMENTS.json',records)
    data.write_json(out/'development_labels.json',[r for r in labels if r['usage']!='evaluation_only'])
    data.write_json(out/'heldout_labels.json',[r for r in labels if r['usage']=='evaluation_only'])
    data.write_json(out/'REFERENCES.json',manifest)
    data.write_json(out/'DAPI_REFERENCE_QC.json',dapi_qc)
    record=dict(status='complete',reference_kind=kind,registration_verified=registration_verified,
        scope='exploratory' if allow_exploratory else 'curated_reference_evaluation',
        protocol=asdict(protocol),protocol_sha256=protocol.identity(),prepare_sha256=data.digest(prep/'PREPARE.json'),
        artifacts={f.name:data.digest(f) for f in out.glob('*.json')},
        cells=len(labels),no_reference_detection_images=[i for i,m in manifest.items() if m['no_reference_detections']],
        qc_status_counts=dict(Counter(r['qc_status'] for r in labels)),
        confident_classes=dict(Counter(str(r['ctnt_positive']) for r in labels)),
        limitations=['Fluorescence proxy labels are not an independently validated cell-identity assay',
            'Zero-reference-object images remain scheduled; zero algorithm detections do not establish biological absence']+
            (['Pseudo-reference detection overlap favors its source algorithm; no segmentation ground-truth AP claim'] if kind!='curated_independent' else [])+
            (['Composite channel registration has not been independently validated'] if not registration_verified else []))
    data.write_json(out/'REFERENCE_COMPLETE.json',record)
    return record


def verify(prep,references):
    record=data.read_json(references/'REFERENCE_COMPLETE.json')
    if record['status']!='complete' or record['prepare_sha256']!=data.digest(prep/'PREPARE.json'):
        raise ValueError('references belong to another preparation')
    for name,sha in record['artifacts'].items():
        if data.digest(references/name)!=sha:raise ValueError('frozen fluorescence reference changed')
    manifest=data.read_json(references/'REFERENCES.json')
    for r in manifest.values():
        if data.digest(Path(r['mask']))!=r['sha256']:raise ValueError('frozen reference mask changed')
    return record,manifest


def _export_image(task):
    """One image's private crops and DAPI QC; top level, so a spawned worker process can run it."""
    prep,out,row,channel_sources,mask_record,items,position=task
    ident=row['image_id'];channels=[tifffile.imread(data.within(prep/'public',row['file']))]
    for name in ('DAPI','TroponinT'):
        source=channel_sources[name];path=data.within(prep/'private',source['file'])
        if data.digest(path)!=source['sha256']:raise ValueError('evaluation channel changed')
        channels.append(tifffile.imread(path))
    stack=np.stack(channels);mask=tifffile.imread(mask_record['mask'])
    if data.digest(Path(mask_record['mask']))!=mask_record['sha256']:raise ValueError('delivered mask changed')
    nuclei=nuclei_from_dapi(channels[1],MeasurementProtocol())
    retained=np.where(np.isin(mask,[c['mask_label'] for c in items]),mask,0)
    qc={'segmented':nuclear_qc(mask,nuclei),'retained':nuclear_qc(retained,nuclei)};records=[]
    for index,c in enumerate(items):
        y0,x0,y1,x1=map(int,c['crop_bbox'])
        if not (0<=y0<y1<=mask.shape[0] and 0<=x0<x1<=mask.shape[1] and y1-y0==x1-x0):raise ValueError('invalid crop geometry')
        crop=stack[:,y0:y1,x0:x1]
        bf=tifffile.imread(c['crop_path'])
        if data.digest(Path(c['crop_path']))!=c['sha256'] or not np.array_equal(bf.squeeze(),crop[0]):
            raise ValueError('brightfield crop does not match private evaluation geometry')
        path=out/ident/f'cell_{index:06d}.tif';path.parent.mkdir(exist_ok=True)
        tifffile.imwrite(path,crop,metadata={'axes':'CYX','ChannelNames':['Brightfield','DAPI','TroponinT']},photometric='minisblack')
        if not np.array_equal(tifffile.imread(path),crop):raise ValueError('three-channel crop roundtrip failed')
        records.append(dict(cell_id=c['cell_id'],image_id=ident,mask_label=c['mask_label'],crop_bbox=c['crop_bbox'],
            private_evaluation_crop=str(path.resolve()),sha256=data.digest(path),channels=['Brightfield','DAPI','TroponinT']))
    print(f'EVAL_CROP {position} {ident}',flush=True)
    return ident,qc,records


def export_crops(prep,run,out,workers=1):
    """Same delivered boxes, raw uint16 BF/DAPI/cTnT; never returned to the predictor. `workers` images at once, each
    in a spawned process; records and QC in image order, the same as one at a time."""
    data.fresh_output(out,prep);out.mkdir(parents=True)
    rows={r['image_id']:r for r in data.load_prepared(prep)};private=data.private_images(prep)
    cells=data.read_json(run/'cells.json')
    grouped={m['image_id']:[] for m in data.read_json(run/'segmentation.json')}
    for c in cells:grouped[c['image_id']].append(c)
    masks={m['image_id']:m for m in data.read_json(run/'segmentation.json')}
    tasks=[(prep,out,rows[ident],private[ident]['channels'],masks[ident],items,f'{n}/{len(grouped)}')
           for n,(ident,items) in enumerate(sorted(grouped.items()),1)]
    if workers>1:
        pool=ProcessPoolExecutor(max_workers=workers,mp_context=multiprocessing.get_context('spawn'))
        try:results=list(pool.map(_export_image,tasks))
        finally:pool.shutdown(wait=True,cancel_futures=True)
    else:results=[_export_image(t) for t in tasks]
    records=[r for _,_,image_records in results for r in image_records];dapi_qc={ident:qc for ident,qc,_ in results}
    data.write_json(out/'CROPS.json',records)
    data.write_json(out/'DAPI_QC.json',dapi_qc)
    data.write_json(out/'EVALUATION_CROPS_COMPLETE.json',dict(status='complete',cells=len(records),prediction_channels=['Brightfield'],
        evaluation_channels=['Brightfield','DAPI','TroponinT'],crops_sha256=data.digest(out/'CROPS.json'),dapi_qc_sha256=data.digest(out/'DAPI_QC.json')))
    return out

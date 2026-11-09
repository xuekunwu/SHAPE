"""Prepare the verified three-channel export without exposing fluorescence to agents."""
from collections import Counter, defaultdict
from pathlib import Path
import hashlib
import math
import os
import re
import shutil

from ...ctnt_flowcytometry.runners.runtime.artifacts import digest, read_json, write_json, within, fresh_output
from ...ctnt_flowcytometry.runners.dataset.preparation import apportion

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SPEC = read_json(HERE.parent/'harness/dataset.json')
USAGES = ('development_train','development_val','evaluation_only')
PROTOCOL = 'staining-well-balanced-40-10-50-v1'


def split_wells(records, seed):
    """Exact well totals; acquisition prefix then plate, independent of fluorescence."""
    from ...ctnt_flowcytometry.runners.dataset.preparation import well_split
    rows=[]
    for r in records:
        md=r['metadata']
        prefix=md['plate_id'].split('_Plate_')[0]
        acquisition=prefix[:6] if re.fullmatch(r'\d{6}',prefix[:6]) else prefix
        rows.append(dict(plate_id=md['plate_id'],well_id=md['well_id'],experiment_date=acquisition))
    return well_split(rows,seed)


def link_verified(source, target, sha):
    if digest(source)!=sha:
        raise ValueError('source channel hash changed')
    target.parent.mkdir(parents=True,exist_ok=True)
    try: os.link(source,target)
    except OSError: shutil.copyfile(source,target)
    if digest(target)!=sha:
        raise ValueError('channel transfer changed pixels')


def prepare(channels_root, out, seed=20261017):
    """Consume the previously verified export; all metadata categories are retained."""
    fresh_output(out,channels_root)
    original=read_json(channels_root/'PREPARE.json')
    source_manifest=channels_root/'private/IMAGE_MANIFEST.json'
    if (original.get('revision')!=SPEC['revision'] or original.get('metadata_sha256')!=SPEC['metadata_sha256']
        or not original.get('lossless_channel_roundtrip_verified') or digest(source_manifest)!=original['private_manifest_sha256']):
        raise ValueError('wrong or unverified channel export')
    records=read_json(source_manifest)
    if len(records)!=SPEC['expected_images']:
        raise ValueError('incomplete image export')
    assignments=split_wells(records,seed)
    plates={p:f'p{i:03d}' for i,p in enumerate(sorted({k[0] for k in assignments}))}
    wells={w:f'w{i:04d}' for i,w in enumerate(sorted(assignments))}
    acquisitions=sorted({r['metadata']['plate_id'].split('_Plate_')[0][:6] for r in records})
    dates={d:f'a{i:03d}' for i,d in enumerate(acquisitions)}
    out.mkdir(parents=True)
    public=[];private={}
    for index,r in enumerate(records):
        md=r['metadata'];key=(md['plate_id'],md['well_id']);ident=r['image_id'];usage=assignments[key]
        image=f'images/{ident}.tif';channel=r['channels']['Brightfield']
        link_verified(within(channels_root,channel['path']),out/'public'/image,channel['sha256'])
        refs={}
        for name in ('DAPI','TroponinT'):
            source=r['channels'][name];relative=f'channels/{name}/{ident}.tif'
            link_verified(within(channels_root,source['path']),out/'private'/relative,source['sha256'])
            refs[name]={'file':relative,'sha256':source['sha256']}
        acquisition=dates[md['plate_id'].split('_Plate_')[0][:6]]
        public.append(dict(image_id=ident,file=image,segment_file=image,sha256=channel['sha256'],
            segment_sha256=channel['sha256'],plate=plates[key[0]],plate_id=plates[key[0]],well=wells[key],site=md['site_id'],
            acquisition_id=acquisition,experiment_date=acquisition,usage=usage,
            partition='held_out' if usage==USAGES[2] else 'seen',shape=[1,*r['shape'][1:]],dtype='uint16',
            calibration_group='',negative_control=False))
        private[ident]=dict(channels=refs,source_metadata=md,well=wells[key],usage=usage,
            registration={'status':'not_verified','note':'Composite grid identity does not establish optical alignment.'})
        if (index+1)%100==0:print(f'PREPARE {index+1}/{len(records)}',flush=True)
    write_json(out/'public/images.json',public)
    write_json(out/'private/images.json',private)
    write_json(out/'private/splits.json',dict(protocol=PROTOCOL,seed=seed,unit='well',ratios=[.4,.1,.5],
        counts=dict(Counter(assignments.values())),wells={wells[k]:v for k,v in assignments.items()},
        balance_fields=['acquisition_prefix_from_plate_id','plate_id']))
    record=dict(status='complete',dataset='ctnt_staining',protocol=PROTOCOL,revision=SPEC['revision'],
        metadata_sha256=SPEC['metadata_sha256'],images=len(public),wells=len(wells),seed=seed,
        images_sha256=digest(out/'public/images.json'),private_images_sha256=digest(out/'private/images.json'),
        splits_sha256=digest(out/'private/splits.json'),prediction_channels=['Brightfield'],
        evaluation_channels=['Brightfield','DAPI','TroponinT'],metadata_flags_policy='retain_all; no silent source-row deletion',
        metadata_status_counts=dict(Counter(r['metadata']['metadata_status'] for r in records)),
        source_export_sha256=digest(channels_root/'PREPARE.json'),source_export=str(channels_root.resolve()),
        references_ready=False,acquisition_note='Six-digit plate-name prefix is a stratification proxy, not verified acquisition metadata.')
    write_json(out/'PREPARE.json',record)
    return record


def validate_rows(rows):
    if not rows or len({r['image_id'] for r in rows})!=len(rows):raise ValueError('empty/duplicate image IDs')
    wells=defaultdict(list)
    for r in rows:
        if r['usage'] not in USAGES or r['file']!=r['segment_file'] or r['sha256']!=r['segment_sha256']:
            raise ValueError('invalid split or non-brightfield segmentation input')
        wells[r['well']].append(r)
    for group in wells.values():
        if len({r['usage'] for r in group})!=1 or len({r['plate'] for r in group})!=1:
            raise ValueError('well crosses split/plate')
    if {r['usage'] for r in rows}!=set(USAGES):raise ValueError('missing split')


def load_prepared(prep, *, verify_pixels=False):
    record=read_json(prep/'PREPARE.json');path=prep/'public/images.json'
    if record.get('protocol')!=PROTOCOL or record.get('status')!='complete' or digest(path)!=record['images_sha256']:
        raise ValueError('wrong/changed staining preparation')
    rows=read_json(path);validate_rows(rows)
    if verify_pixels:
        for r in rows:
            if digest(within(prep/'public',r['file']))!=r['sha256']:raise ValueError('brightfield pixels changed')
    return rows


def private_images(prep):
    record=read_json(prep/'PREPARE.json');path=prep/'private/images.json'
    if digest(path)!=record['private_images_sha256']:raise ValueError('private fluorescence manifest changed')
    return read_json(path)


def sample_images(rows, fraction, seed):
    """Same complete-well subset for every arm; no target-based sampling."""
    if not 0<fraction<=1:raise ValueError('fraction must be in (0,1]')
    chosen=set()
    for usage in USAGES:
        wells=sorted({r['well'] for r in rows if r['usage']==usage},
            key=lambda w:hashlib.sha256(f'{seed}|{usage}|{w}'.encode()).hexdigest())
        chosen.update(wells[:max(1,round(len(wells)*fraction))])
    return [r for r in rows if r['well'] in chosen]

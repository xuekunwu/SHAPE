"""Fixed metadata-only 20% complete-well sample, preserving the original full-data split."""
from collections import Counter
from concurrent.futures import ThreadPoolExecutor,as_completed
from pathlib import Path
import csv
import hashlib
import json
import shutil
import numpy as np
import tifffile
from . import data
from ..ctnt_flowcytometry.sampling import select
from ..ctnt_flowcytometry.plate_ssl_routing import eligible_plates

SPLIT_SEED=20261017
PILOT_SEED=20261016
METADATA_SHA=data.SPEC['metadata_sha256']   # the public archive's metadata CSV (dataset.json); the pre-release local backup was c1b2a9ea...
PROTOCOL='staining-pilot20-fixed-wells-local-backup-v1'
FULL_PROTOCOL='staining-full-fixed-wells-local-backup-v1'


def reference_scope(reference,smoke,allow_exploratory,pilot_record,allow_exploratory_full=False,fraction=1.):
    if reference['registration_verified'] and reference['reference_kind']=='curated_independent':return 'curated_reference_evaluation'
    if smoke:return 'exploratory_technical_smoke'
    if allow_exploratory and pilot_record and pilot_record['fraction']<=.2:return 'exploratory20percentpilot'
    # User decision 2026-10-27: the full dataset may be scored against the fixed pseudo-reference, explicitly exploratory.
    if allow_exploratory_full and not pilot_record:
        return 'exploratory_full_pseudo_reference' if fraction==1 else 'exploratory_full_subset_pseudo_reference'
    raise ValueError('Unreviewed reference masks/registration require an explicit exploratory pilot of at most20%; full scientific evaluation still requires curated references')


def metadata_rows(rows):
    assignments=data.split_wells([{'metadata':r} for r in rows],SPLIT_SEED)
    plates={p:f'p{i:03d}' for i,p in enumerate(sorted({k[0] for k in assignments}))}
    wells={w:f'w{i:04d}' for i,w in enumerate(sorted(assignments))}
    dates={d:f'a{i:03d}' for i,d in enumerate(sorted({r['plate_id'].split('_Plate_')[0][:6] for r in rows}))}
    public=[]
    for i,r in enumerate(rows):
        key=(r['plate_id'],r['well_id']);usage=assignments[key]
        public.append(dict(image_id=f'i{i:05d}',plate=plates[key[0]],plate_id=plates[key[0]],well=wells[key],
            usage=usage,site=r['site_id'],experiment_date=dates[key[0].split('_Plate_')[0][:6]],
            acquisition_id=dates[key[0].split('_Plate_')[0][:6]],
            partition='held_out' if usage=='evaluation_only' else 'seen'))
    return public,{wells[k]:v for k,v in assignments.items()}


def verify(prep,path):
    record=data.read_json(path);preparation=data.read_json(prep/'PREPARE.json')
    full=prep/'sampling/FULL_METADATA.json'
    if data.digest(full)!=preparation['sampling_metadata_sha256']:raise ValueError('sampling metadata changed')
    if record['protocol']!=PROTOCOL or record['fraction']!=.2 or record['seed']!=PILOT_SEED:
        raise ValueError('wrong pilot protocol')
    chosen=select(data.read_json(full),PILOT_SEED)
    current=data.load_prepared(prep)
    if record['images']!=[r['image_id'] for r in chosen] or record['images']!=[r['image_id'] for r in current]:
        raise ValueError('pilot images differ from frozen full-split metadata selection')
    for r,c in zip(chosen,current):
        if any(r[k]!=c[k] for k in ('usage','well','plate_id','experiment_date')):raise ValueError('pilot split reassigned')
    if record['prepare_sha256']!=data.digest(prep/'PREPARE.json'):raise ValueError('pilot preparation identity changed')
    return record


def prepare(raw,out,frozen_split,workers=16,all_wells=False):
    """The fixed 20% pilot, or with all_wells every well of the same frozen split (same image/well IDs as the pilot)."""
    data.fresh_output(out,raw)
    metadata=raw/'cTnT_composite_metadata.csv'
    if data.digest(metadata)!=METADATA_SHA:raise ValueError('metadata differs from the dataset.json pin')
    with metadata.open(encoding='utf-8-sig',newline='') as stream:original=list(csv.DictReader(stream))
    if len(original)!=2294:raise ValueError('source row count changed')
    full,splits=metadata_rows(original)
    if frozen_split is not None:   # optional check against an earlier preparation's private/splits.json
        old=data.read_json(frozen_split)
        if old['seed']!=SPLIT_SEED or old['wells']!=splits:raise ValueError('full well split differs from completed smoke')
    if all_wells:
        chosen=full
        if len({r['well'] for r in chosen})!=574:raise ValueError('unexpected full well count')
        if shutil.disk_usage(out.parent).free<150*1024**3:raise ValueError('need at least150GiB free for prepared full data and immediate outputs')
    else:
        chosen=select(full,PILOT_SEED)
        if len({r['well'] for r in chosen})!=114 or len(chosen)!=456:raise ValueError('unexpected fixed pilot size')
        view_counts=Counter(Counter(r['well'] for r in chosen).values())
        if dict(view_counts)!={4:114}:raise ValueError('pilot must retain all4views per selected well')
        if shutil.disk_usage(out.parent).free<30*1024**3:raise ValueError('need at least30GiB free for prepared pilot and immediate outputs')
    out.mkdir(parents=True)
    data.write_json(out/'PREPARATION_PROGRESS.json',{'status':'running','done':0,'total':len(chosen)})
    def one(row):
        source_row=original[int(row['image_id'][1:])]
        source=(raw/source_row['relative_path']).resolve()
        if not source.is_relative_to(raw.resolve()) or not source.is_file():raise ValueError('source image outside verified backup')
        with tifffile.TiffFile(source) as tif:
            series=tif.series[0];description=json.loads(tif.pages[0].description);names=description['ChannelNames']
            if series.axes!='CYX' or series.shape!=(3,2048,2048) or str(series.dtype)!='uint16':raise ValueError('wrong composite axes/shape/dtype')
            if set(names)!={'Brightfield','DAPI','TroponinT'}:raise ValueError('ambiguous channels')
            x=series.asarray(maxworkers=1)
        channels={}
        for name in ('Brightfield','DAPI','TroponinT'):
            relative=('public/images/' if name=='Brightfield' else 'private/channels/'+name+'/')+row['image_id']+'.tif'
            target=out/relative;target.parent.mkdir(parents=True,exist_ok=True)
            pixels=x[names.index(name)]
            tifffile.imwrite(target,pixels,metadata={'axes':'YX','ChannelName':name},photometric='minisblack')
            if not np.array_equal(tifffile.imread(target),pixels):raise ValueError('channel roundtrip changed pixels')
            channels[name]={'file':relative.removeprefix('private/'),'sha256':data.digest(target)}
        bf=channels['Brightfield'];file='images/'+row['image_id']+'.tif'
        public={**row,'file':file,'segment_file':file,'sha256':bf['sha256'],'segment_sha256':bf['sha256'],
            'shape':[1,2048,2048],'dtype':'uint16','calibration_group':'','negative_control':False}
        private={'channels':{k:v for k,v in channels.items() if k!='Brightfield'},'source_metadata':source_row,
            'source_sha256':data.digest(source),'well':row['well'],'usage':row['usage'],
            'registration':{'status':'not_verified','note':'Composite grid only; optical registration not independently reviewed'}}
        return public,private
    result={}
    with ThreadPoolExecutor(max_workers=workers) as pool:
        futures=[pool.submit(one,r) for r in chosen]
        for future in as_completed(futures):
            public,private=future.result();result[public['image_id']]=(public,private)
            if len(result)%16==0 or len(result)==len(chosen):
                data.write_json(out/'PREPARATION_PROGRESS.json',{'status':'running','done':len(result),'total':len(chosen)})
                print('PREPARE_CHANNELS',len(result),'/',len(chosen),flush=True)
    public=[result[r['image_id']][0] for r in chosen];private={r['image_id']:result[r['image_id']][1] for r in chosen}
    data.write_json(out/'public/images.json',public);data.write_json(out/'private/images.json',private)
    data.write_json(out/'sampling/FULL_METADATA.json',full)
    if frozen_split is not None:shutil.copy2(frozen_split,out/'private/splits.json')
    else:data.write_json(out/'private/splits.json',dict(protocol=data.PROTOCOL,seed=SPLIT_SEED,unit='well',ratios=[.4,.1,.5],   # data.prepare's layout
        counts=dict(Counter(splits.values())),wells=splits,balance_fields=['acquisition_prefix_from_plate_id','plate_id']))
    wells={r['well']:r for r in public}
    sampled={'seed':SPLIT_SEED} if all_wells else {'seed':SPLIT_SEED,'pilot_seed':PILOT_SEED,'pilot_fraction':.2}
    record={'status':'complete','dataset':'ctnt_staining','protocol':data.PROTOCOL,'revision':data.SPEC['revision'],
        'metadata_sha256':METADATA_SHA,'images':len(public),'wells':len(wells),'full_dataset_images':2294,'full_dataset_wells':574,
        **sampled,
        'images_sha256':data.digest(out/'public/images.json'),'private_images_sha256':data.digest(out/'private/images.json'),
        'sampling_metadata_sha256':data.digest(out/'sampling/FULL_METADATA.json'),'splits_sha256':data.digest(out/'private/splits.json'),
        'prediction_channels':['Brightfield'],'evaluation_channels':['Brightfield','DAPI','TroponinT'],
        'lossless_channel_roundtrip_verified':True,'references_ready':False,'scope':'full_dataset' if all_wells else 'exploratory20percentpilot'}
    data.write_json(out/'PREPARE.json',record)
    summary={'prepare_sha256':data.digest(out/'PREPARE.json'),'images':[r['image_id'] for r in public],'wells':sorted(wells),
        'split_wells':dict(Counter(r['usage'] for r in wells.values())),'split_images':dict(Counter(r['usage'] for r in public)),
        'eligibility':eligible_plates(public)}
    if all_wells:
        manifest={'protocol':FULL_PROTOCOL,'split_seed':SPLIT_SEED,'unit':'well',**summary}
        data.write_json(out/'FULL.json',manifest)
    else:
        manifest={'protocol':PROTOCOL,'fraction':.2,'seed':PILOT_SEED,'split_seed':SPLIT_SEED,'unit':'well',
            'sampling':'date/plate-stratified random sample within each frozen split; all views retained',**summary}
        data.write_json(out/'PILOT.json',manifest);verify(out,out/'PILOT.json')
    data.write_json(out/'PREPARATION_PROGRESS.json',{'status':'complete','done':len(public),'total':len(public)})
    return manifest

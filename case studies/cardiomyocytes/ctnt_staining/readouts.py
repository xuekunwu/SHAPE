"""Stage QC, DAPI diagnostics, and cost accounting after predictions are frozen."""
from collections import defaultdict
from pathlib import Path
import numpy as np
import tifffile
from . import data


def collect(prep,run,evaluation_root):
    from shape.tools.cell_segmenter.quant import quant_score
    from shape.tools.single_cell_cropper.quant import crop_quality_score
    from shape.tools.cell_state_analyzer import quant as representation
    rows={r['image_id']:r for r in data.load_prepared(prep)};masks=data.read_json(run/'segmentation.json')
    cells=data.read_json(run/'cells.json');grouped=defaultdict(list)
    for c in cells:grouped[c['image_id']].append(c)
    segmentation=[];cropping=[]
    for m in masks:
        ident=m['image_id'];row=rows[ident];mask=tifffile.imread(m['mask'])
        image=tifffile.imread(data.within(prep/'public',row['file']))
        segmentation.append(dict(image_id=ident,well=row['well'],cells=m['n_cells'],**quant_score(image,mask)))
        quality=[]
        for c in grouped[ident]:
            y0,x0,y1,x1=c['crop_bbox'];quality.append(crop_quality_score(tifffile.imread(c['crop_path']).squeeze(),mask[y0:y1,x0:x1]==c['mask_label']))
        cropping.append(dict(image_id=ident,well=row['well'],retained_cells=len(quality),
            quality={k:float(np.mean([q[k] for q in quality])) for k in quality[0]} if quality else {},
            status='measured' if quality else 'no_objects_not_scored'))
    def macro(records,keys):
        wells=sorted({r['well'] for r in records})
        return {k:float(np.mean([np.mean([r[k] for r in records if r['well']==w]) for w in wells])) for k in keys}
    x=np.load(run/'embedding.npy',mmap_mode='r');health={}
    for usage in data.USAGES:
        indices=[i for i,c in enumerate(cells) if c['usage']==usage]
        indices=np.random.default_rng(0).permutation(indices)[:representation.HEALTH_ROWS]
        values=np.asarray(x[indices]);meta=[dict(site=cells[i]['image_id'],well=cells[i]['well'],plate=cells[i]['plate']) for i in indices]
        health[usage]=dict(sampled_cells=len(indices),health=representation.health(values),
            image_leakage=representation.image_leakage(values,meta),sibling_site_map=representation.sibling_site_map(values,meta))
    process={}
    for stage in ['segment','crop','embed','batch','predict','evaluation_crops']:
        process[stage]=data.read_json(run/f'{stage}_COMPLETE.json')
    result=dict(segment=dict(well_macro=macro(segmentation,[k for k in segmentation[0] if k not in ['image_id','well']]),per_image=segmentation),
        crop=dict(well_macro_quality=macro([dict(well=r['well'],**r['quality']) for r in cropping if r['quality']],
            list(next((r['quality'] for r in cropping if r['quality']),{}))),
            per_image=cropping,retained_cells=len(cells),excluded_cells=len(data.read_json(run/'excluded_cells.json'))),
        dapi=data.read_json(evaluation_root/'DAPI_QC.json'),embed=dict(selection=data.read_json(run/'SSL_SELECTION.json'),by_split=health),
        batch=dict(admission=data.read_json(run/'batch_SEARCH.json')['admission'],cell_embeddings_corrected=False),process=process,
        interpretation='GT-free morphology QC and DAPI nuclear diagnostics; not segmentation AP or proof of whole-cell accuracy')
    data.write_json(run/'STAGE_READOUTS.json',result)
    return result

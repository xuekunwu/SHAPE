import csv
import importlib
from pathlib import Path
import sys
import numpy as np
import pytest
import tifffile

sys.path.insert(0,str(Path(__file__).resolve().parents[5]))
PKG="case studies.cardiomyocytes.ctnt_flowcytometry.runners"
data=importlib.import_module(PKG+".dataset.preparation")
readouts=importlib.import_module(PKG+".evaluation.stage_quality")


def test_stage_readouts_use_shared_quant_and_keep_missing_truth_explicit(tmp_path,monkeypatch):
    from shape.tools.cell_segmenter.quant import quant_score
    from shape.tools.single_cell_cropper.quant import crop_quality_score
    prep,run=tmp_path/"prep",tmp_path/"run"
    (prep/"public").mkdir(parents=True)
    run.mkdir()
    rows,masks,cells=[],[],[]
    image=np.zeros((32,32),dtype=np.uint16)
    image[8:16,8:16]=100
    mask=np.zeros((32,32),dtype=np.uint32)
    mask[8:16,8:16]=1
    for w,usage in enumerate(data.USAGES):
        for site in range(2):
            ident=f"i{w}{site}"
            tifffile.imwrite(prep/"public"/f"{ident}.tif",image)
            path=run/f"{ident}_mask.tif"
            tifffile.imwrite(path,mask)
            rows.append(dict(image_id=ident,well=f"w{w}",plate="p",usage=usage,file=f"{ident}.tif"))
            masks.append(dict(image_id=ident,mask=str(path),sha256=data.digest(path),n_cells=1))
            crop=run/f"{ident}_crop.tif"
            tifffile.imwrite(crop,image[6:18,6:18])
            cells.append(dict(image_id=ident,cell_id=ident,well=f"w{w}",plate="p",usage=usage,
                              crop_path=str(crop),sha256=data.digest(crop),crop_bbox=[6,6,18,18],mask_label=1))
    data.write_json(run/"CONFIG.json",{"images":[r['image_id'] for r in rows]})
    data.write_json(run/"segmentation.json",masks)
    data.write_json(run/"cells.json",cells)
    data.write_json(run/"excluded_cells.json",[])
    data.write_json(run/"SSL_SELECTION.json",{"mode":"zero_shot","gate":{"decision":"SKIP"}})
    data.write_json(run/"batch_SEARCH.json",{"admission":{"decision":"SKIP"}})
    np.save(run/"embedding.npy",np.random.default_rng(0).normal(size=(6,4)))
    profiles=np.array([[1.,0],[1,1],[0,1]])
    for name in ('profiles','corrected'):
        np.save(run/f"{name}.npy",profiles)
    with (run/"profiles.csv").open('w',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=['well','experiment_date','plate_id'])
        writer.writeheader()
        writer.writerows(dict(well=f"w{w}",experiment_date=f"d{w%2}",plate_id=f"p{w%2}") for w in range(3))
    for stage in ('segment','crop','embed','batch','predict'):
        data.write_json(run/f"{stage}_COMPLETE.json",{'seconds':1,'usage':{'total_tokens':0}})
    data.write_json(run/"DELIVERY.json",{'artifacts':{p.name:data.digest(p) for p in run.iterdir()},
                                         'source_sha256':'source','prepare_sha256':'prep'})
    monkeypatch.setattr(readouts,'load_prepared',lambda p:rows)
    result=readouts.collect(prep,run)
    assert result['segment']['well_macro']['quant_score']==pytest.approx(quant_score(image,mask)['quant_score'])
    assert result['crop']['well_macro_quality']['crop_quality_score']==pytest.approx(
        crop_quality_score(image[6:18,6:18],mask[6:18,6:18]==1)['crop_quality_score'])
    assert result['batch']['profiles_unchanged'] is True
    assert result['batch']['biological_preservation']['value'] is None
    assert 'no annotated' in result['unavailable']['segmentation_AP_F1']
    assert set(result['embed']['by_split'])==set(data.USAGES)
    Path(cells[0]['crop_path']).write_bytes(b'changed')
    with pytest.raises(ValueError,match='delivery changed'):
        readouts.collect(prep,run)


def test_batch_diagnostic_marks_single_batch_unavailable():
    out=readouts.batch_structure(np.ones((3,2)),[{'plate_id':'p'}]*3,'plate_id')
    assert out['value'] is None and out['reason']

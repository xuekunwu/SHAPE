"""No fabricated detections, no disappearing missed cells, and no silent lost views."""
import importlib
from unittest.mock import patch
import numpy as np
import pytest
import tifffile

BASE='case studies.cardiomyocytes.ctnt_staining.runners.'
data=importlib.import_module(BASE+'data')


def test_zero_detection_is_opt_in_and_keeps_failed_visual_qc(tmp_path):
    from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
    class EmptyModel:
        def eval(self,images,**kwargs):
            return [np.zeros(images[0].shape[:2],dtype=np.int32)],None,None
    source=tmp_path/'bf.tif'
    tifffile.imwrite(source,np.random.default_rng(1).integers(1000,2000,(64,64),dtype=np.uint16))
    with patch('shape.tools.cell_segmenter.tool.models.CellposeModel',return_value=EmptyModel()):
        tool=Cell_Segmenter_Tool()
        rejected=tool.execute(str(source),query_cache_dir=str(tmp_path/'strict'))
        assert rejected.get('error') and rejected['cell_count']==0
        tool.allow_empty_masks=True
        empty=tool.execute(str(source),query_cache_dir=str(tmp_path/'record'))
        assert not empty.get('error') and empty['status']=='no_detection'
        assert empty['overlay_qc']['passed'] is False and empty['qc_warning']
        assert not np.load(empty['mask_path']).any()


def test_low_coverage_detection_is_opt_in_and_keeps_failed_visual_qc(tmp_path):
    from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
    class FixedModel:
        def __init__(self,mask):self.mask=mask
        def eval(self,images,**kwargs):
            return [self.mask.copy()],None,None
    source=tmp_path/'bf.tif'
    tifffile.imwrite(source,np.random.default_rng(1).integers(1000,2000,(640,640),dtype=np.uint16))
    debris=np.zeros((640,640),np.int32);debris[10:16,10:16]=1   # coverage 8.8e-5: below the visual QC's floor
    y,x=np.mgrid[:640,:640];tiled=(y//80*8+x//80+1).astype(np.int32)   # coverage 1: implausible the other way, an error
    for name,mask,delivered in (('debris',debris,True),('tiled',tiled,False)):
        with patch('shape.tools.cell_segmenter.tool.models.CellposeModel',return_value=FixedModel(mask)):
            tool=Cell_Segmenter_Tool()
            assert tool.execute(str(source),query_cache_dir=str(tmp_path/name/'strict')).get('error')
            tool.allow_empty_masks=True
            result=tool.execute(str(source),query_cache_dir=str(tmp_path/name/'record'))
        assert bool(result.get('error'))!=delivered and result['overlay_qc']['passed'] is False
        if delivered:
            assert result['status']=='low_coverage_detection' and result['qc_warning'] and result['cell_count']==1
            np.testing.assert_array_equal(np.load(result['mask_path']),debris)


def test_zero_crops_only_permitted_on_a_dataset_that_allows_empty_masks(tmp_path):
    # An image with no retained crop contributes no cells when the dataset allows empty masks, whether its
    # mask was empty or the crop loop rejected every cell; elsewhere it stops the stage.
    from benchmarks.full_CPJUMP1.harness.pipeline import retained_crops,CPJUMP1
    spec=importlib.import_module(BASE+'spec').STAINING
    row={'image_id':'i0'};mask=np.zeros((4,4),np.uint32)
    assert retained_crops(np.zeros((1,4,4)),row,[],set(),tmp_path,spec,mask)==([],[])
    mask[1,1]=1
    assert retained_crops(np.zeros((1,4,4)),row,[],set(),tmp_path,spec,mask)==([],[])
    with pytest.raises(RuntimeError):retained_crops(np.zeros((5,4,4)),row,[],set(),tmp_path,CPJUMP1,None)


def test_missing_view_is_reported_and_not_zero_imputed():
    from benchmarks.full_CPJUMP1.harness.data import well_profiles
    rows=[dict(image_id=f'i{i}',well='w',plate='p',usage='evaluation_only',calibration_group='',negative_control=False) for i in range(2)]
    cells=[dict(cell_id='c',image_id='i0')]
    with pytest.raises(ValueError):well_profiles(np.array([[2.,4.]]),cells,rows)
    profiles,table=well_profiles(np.array([[2.,4.]]),cells,rows,allow_empty_sites=True)
    np.testing.assert_equal(profiles,[[2.,4.]])
    assert table.iloc[0].scheduled_sites==2 and table.iloc[0].nonempty_sites==1
    assert table.iloc[0].no_detection_sites==['i1']
    with pytest.raises(ValueError):well_profiles(np.array([[2.,4.]]),cells,[{**rows[0],'image_id':'other'}],allowed_sites=(1,),allow_empty_sites=True)


def test_codex_zero_delivery_needs_dataset_policy_and_zero_crop_input(tmp_path):
    from benchmarks.full_CPJUMP1.harness.codex_arm import StageRun
    from benchmarks.full_CPJUMP1.harness.pipeline import CPJUMP1
    spec=importlib.import_module(BASE+'spec').STAINING
    run=StageRun.__new__(StageRun);run.spec=spec;run.run=tmp_path;run.stage='segment'
    run.calls={'i':{'c0':dict(status='complete',delivery=dict(n_cells=0))}}
    assert run.deliverable('i','c0') is None
    run.spec=CPJUMP1
    assert run.deliverable('i','c0') is not None
    run.spec=spec;run.stage='crop';run.calls={'i':{'c0':dict(status='complete',delivery=dict(n_crops=0))}}
    data.write_json(tmp_path/'segmentation.json',[dict(image_id='i',n_cells=0)])
    run._segmented=None
    assert run.deliverable('i','c0') is None
    data.write_json(tmp_path/'segmentation.json',[dict(image_id='i',n_cells=1)])
    run._segmented=None
    assert run.deliverable('i','c0') is not None


def test_scoring_keeps_missed_reference_cells_when_no_prediction(tmp_path,monkeypatch):
    score=importlib.import_module(BASE+'score')
    run=tmp_path/'run';refs=tmp_path/'refs';run.mkdir();refs.mkdir()
    reference=refs/'mask.tif';tifffile.imwrite(reference,np.array([[1,1],[0,0]],np.uint32))
    predicted=run/'mask.tif';tifffile.imwrite(predicted,np.zeros((2,2),np.uint32))
    data.write_json(refs/'REFERENCE_COMPLETE.json',{})
    data.write_json(refs/'THRESHOLD.json',{})
    data.write_json(refs/'heldout_labels.json',[dict(image_id='i',mask_label=1,ctnt_positive=1,reference_cell_id='r',well='w')])
    data.write_json(run/'segmentation.json',[dict(image_id='i',mask=str(predicted),sha256=data.digest(predicted),n_cells=0)])
    data.write_json(run/'predictions.json',[]);data.write_json(run/'cells.json',[])
    data.write_json(run/'CONFIG.json',dict(images=['i'],arm='control',source_sha256='test',seed=1))
    data.write_json(run/'CLASSIFIER.json',dict(classifier_probability_threshold=.5,minimum_matching_iou=.5))
    np.save(run/'embedding.npy',np.empty((0,2)));np.savez(run/'classifier.npz',weights=np.array([1.]))
    keys=[('predictions.json','predictions_sha256'),('CLASSIFIER.json','classifier_sha256'),
          ('classifier.npz','weights_sha256'),('embedding.npy','embedding_sha256'),
          ('cells.json','cells_sha256'),('segmentation.json','segmentation_sha256')]
    data.write_json(run/'PREDICTIONS_COMPLETE.json',dict({k:data.digest(run/f) for f,k in keys},reference_complete_sha256=data.digest(refs/'REFERENCE_COMPLETE.json')))
    monkeypatch.setattr(data,'load_prepared',lambda prep:[dict(image_id='i',usage='evaluation_only')])
    monkeypatch.setattr(score,'verify_references',lambda *args:({'reference_kind':'fixed_algorithmic_pseudo_reference','limitations':[]},{'i':dict(mask=str(reference),well='w')}))
    result=score.score(tmp_path,run,refs,bootstrap_draws=0)
    assert result['detection']['reference_cells']==1
    assert result['detection']['unmatched_reference']==1
    assert result['detection']['unmatched_positive_references']==1
    assert result['detection']['reference_matching_coverage']==0
    assert result['detection']['assessed_heldout_images']==1
    assert result['detection']['no_prediction_images']==['i']

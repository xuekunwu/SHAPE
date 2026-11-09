"""Shared SHAPE tools receive only brightfield, including at crop and SSL stages."""
from benchmarks.full_CPJUMP1.harness.pipeline import Dataset

MODULE='case studies.cardiomyocytes.ctnt_staining.runners'
STAINING=Dataset(name='ctnt_staining',cli=MODULE+'.e2e',data=MODULE+'.data',score=MODULE+'.score',
    n_images=2294,marker='CTNT_STAINING_COMPLETE',image_shape=(2048,2048),channels=('Brightfield',),
    dtype='uint16',plate_type='fixed iPSC-CM',mask_label=True,allowed_sites=(2,4),allow_empty_masks=True,
    tasks={
        'segment':'Segment whole fixed iPSC-derived cells from brightfield alone, one instance per cell. Do not use fluorescence.',
        'crop':'Choose complete square single-cell crop boxes from brightfield and its predicted mask. Trusted evaluation later copies identical boxes from private DAPI/cTnT; these channels are withheld.',
        'embed':'Fit SSL on brightfield crops from seen training wells only; seen validation selects, held-out wells are inference-only. No fluorescence or cTnT labels are visible.',
        'batch':'Diagnose acquisition/plate structure. No validated cross-batch technical reference labels or cell-level correction transform are supplied. Preserve original cell embeddings.'},
    inputs={
        'segment':'Inputs: {n} {dtype} brightfield images, one mask instance per whole cell. A valid all-zero mask records no detections, not verified absence; do not fabricate cells to satisfy a count gate.',
        'crop':'Inputs: {n} {dtype} brightfield images and this arm\'s masks. Return crop geometry from brightfield alone. An all-zero input mask requires an explicit zero-object crop record; do not invent crops.',
        'embed':'Inputs: brightfield crops, at most {cells} development cells per view. Train/val wells only during SSL selection.',
        'batch':'Inputs: diagnostic well profiles; acquisition/plate batches. Shared admission: {rule}. No validated technical controls.'},
    input_channels={'segment':'Brightfield','crop':'Brightfield'},rules={},
    withheld='DAPI, cTnT, fluorescence intensities, reference masks, cell labels, donor identity and source filenames are withheld from agent/tool decisions.')


def for_config(config):
    from dataclasses import replace
    from benchmarks.full_CPJUMP1.harness.pipeline import CPJUMP1
    return replace(STAINING,allowed_sites=(1,) if config.get('technical_smoke_images')==5 else STAINING.allowed_sites,
                   rules=CPJUMP1.rules)

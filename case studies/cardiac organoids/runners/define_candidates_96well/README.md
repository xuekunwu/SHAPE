# cardiac organoids/runners/define_candidates_96well

SHAPE on a 96-well cardiac-organoid drug screen: 1,141 phase-contrast + GFP images in 12 plate / time-point
folders. Segmentation and cropping read the phase-contrast image; the self-supervised embedding reads two-channel
(phase contrast, GFP) crops cut at the boxes chosen on the phase image. The split is train / validation 8:2 by
plate-well (every time point of a plate-well on one side, no held-out set); the final embedding covers every crop.

## Dataset

[SHAPE-agent/CO_drugscreen](https://huggingface.co/datasets/SHAPE-agent/CO_drugscreen): one folder per plate and
time point (`<date>_<time>_Plate N_<baseline|dN>`). `prepare_full.py` writes the public phase-contrast images, the
two-channel stacks, the private GFP files and the split; `check_prepare.py` checks the split contract:

```
hf download SHAPE-agent/CO_drugscreen --repo-type dataset --local-dir data/CO_drugscreen
python "case studies/cardiac organoids/runners/define_candidates_96well/prepare_full.py" data/CO_drugscreen PREP
python "case studies/cardiac organoids/runners/define_candidates_96well/check_prepare.py" PREP
```

## Running

`full_driver.py` runs segment, crop and embed through the shared harness, each stage in its own process. The
segment stage is the organoid segmentation cohort of `../cohort/organoid_cohort.py` (calibration sample plus a
per-group fast pass); the crop stage is the shared harness's crop cohort.
`SMOKE_ROOT` is the run folder, `CO_SRC` the source tree, `CO_ORGANOID_MODEL` the organoid checkpoint and
`CO_ASSETS` the pinned asset root:

```
export SMOKE_ROOT=RUNS CO_SRC=/path/to/SHAPE CO_ORGANOID_MODEL=CO_CKPT CO_ASSETS=ASSETS
python "case studies/cardiac organoids/runners/define_candidates_96well/full_driver.py" run NAME shape validate   # 10 plate-wells
python "case studies/cardiac organoids/runners/define_candidates_96well/full_driver.py" run NAME shape full       # all images
python "case studies/cardiac organoids/runners/define_candidates_96well/post_full.py" NAME                         # QA summary
```

# cardiac organoids/drug_screen

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
python "case studies/cardiac organoids/drug_screen/prepare_full.py" data/CO_drugscreen RUNS/prepared
python "case studies/cardiac organoids/drug_screen/check_prepare.py" RUNS/prepared
```

## Running

`full_driver.py` runs segment, crop and embed through the shared harness, each stage in its own process. The
segment stage is the organoid segmentation cohort of `../cohort/organoid_cohort.py` (calibration sample plus a
per-group fast pass); the crop stage is the shared harness's crop cohort.
`SMOKE_ROOT` is the run folder (it reads the preparation from `$SMOKE_ROOT/prepared`, so put `RUNS/prepared` there),
`CO_SRC` a pinned source snapshot (the time course's `pack` archive, extracted; the driver checks every file against
its `SOURCE_SNAPSHOT.json`), `CO_ORGANOID_MODEL` the organoid checkpoint
(`hf download SHAPE-agent/organoid-segmenter shape_seg_co --local-dir models`) and `CO_ASSETS` the pinned asset root:

```
python -m "case studies.cardiac organoids.timecourse.e2e" pack --out SNAPSHOT.tar.gz
mkdir SRC && tar -xzf SNAPSHOT.tar.gz -C SRC
export SMOKE_ROOT=$PWD/RUNS CO_SRC=$PWD/SRC CO_ORGANOID_MODEL=$PWD/models/shape_seg_co CO_ASSETS=$PWD/ASSETS   # absolute: the driver changes into CO_SRC
DRIVER="SRC/case studies/cardiac organoids/drug_screen"
python "$DRIVER/full_driver.py" run NAME shape validate   # 10 plate-wells
python "$DRIVER/full_driver.py" run NAME shape full       # all images
python "$DRIVER/post_full.py" NAME                         # QA summary
```

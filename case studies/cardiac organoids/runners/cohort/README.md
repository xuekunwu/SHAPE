# Organoid segmentation cohort (SHAPE arm)

A study-layer module that runs SHAPE's segment stage of the organoid studies as a **cohort** instead of one verify-reason-refine
loop (VRR) per image. No shared harness file is edited: `install()` swaps the stage function into `pipeline.STAGES` of the stage process.

| Module | What it does | Switch in `CONFIG.json` |
|---|---|---|
| `organoid_cohort.py` | Segmentation cohort for the organoid segmenter (`Organoid_Segmenter_Tool`): `segmentation_cohort.run` at its defaults groups the images by their preflight read, runs the full VRR on a seeded calibration sample (`calibration_size`: 29 images of 1141), applies the consensus of the levers the calibration agrees on to the other images in the verifier's cohort mode and gives flagged images their own VRR. Control (no reasoner) takes the harness's per-image path | `segmentation_regime = "cohort"` |
Crops are not a module here: the SHAPE arm of a full run crops through the harness's own crop cohort (`pipeline._cropping_cohort`:
grouping, calibration sample and consensus, driven by `segmentation_cohort.run`). For the organoid datasets it (a) cuts the crops from the
row's raw stack (`raw_file`: two channels in the `ssl_two_channel` variant), (b) does not run a crop VRR for an image whose mask is empty
when the dataset allows empty masks, and (c) honours `crop_mask_check = false` in `CONFIG.json`, which switches off the crop VRR's mask-error
check (`CropTask.enable_mask_check`; it rejected a clean single organoid in the first validation run). Default: on.

Tests (`test_organoid_cohort.py`) use stand-ins for the model, the reasoner and the VRR; SHAPE's real
`segmentation_cohort.run` does the routing. They check wiring, outputs and failure
paths, not segmentation or crop quality and not token use. Run them from the repository root with
`python -m pytest "case studies/cardiac organoids/runners/cohort" -q` (CPU only).

# SHAPE run on the 96-well CO DEFINE candidates (250429_96well_CO_DEFINE candidates)

Run scripts for 1141 phase-contrast + GFP images (12 plate/time-point folders). Segment and crop read the phase-contrast image; the SSL
(embed) stage reads two-channel (PhaseContrast, GFP) crops cut at the boxes chosen on the phase image (`ssl_two_channel`).
Train / validation only (8:2 by plate-well, all time points of a plate-well in one split, no evaluation set); the final embedding covers every
crop. Image, well and plate identifiers are pseudonyms.

| File | Purpose |
|---|---|
| `prepare_full.py`, `check_prepare.py` | prepare the images (public phase images, two-channel stacks, private GFP, split) and check the split contract |
| `full_driver.py`, `post_full.py` | run segment, crop, embed through the shared harness (`validate`: 10 plate-wells at the full-run search settings; `full`: all images) and write the QA summary |

The full set runs on a single-GPU VM through `run_vm.sh`-style wrappers of `full_driver.py`. `full_driver.py` takes its run folder, source tree, organoid model and asset root from `SMOKE_ROOT`, `CO_SRC`, `CO_ORGANOID_MODEL` and `CO_ASSETS`.
The segment stage uses the organoid cohort module in `../cohort`; the crop stage is the harness's crop cohort (mask-error check off).

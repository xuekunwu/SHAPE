# Cardiac organoid FC time course: CONTROL / SHAPE / CODEX end-to-end comparison

Everything below describes the protocol the three arms follow.

## Question

One 96-well plate of cardiac organoids, 12 Control wells (rows B to D, columns 3 to 6) and 11 FC
wells (rows E to G, columns 3 to 6; F6 is absent except at day 4), imaged in phase contrast and GFP
at eight time points (baseline, day 2, 4, 6, 8, 10, 12 and 16). From the **phase-contrast image
alone**, can an end-to-end image pipeline (segment the organoid, crop it, learn a representation,
read it out) recover the experimental condition (Control vs FC) and the culture day on wells it
never saw, and how do three ways of operating the same tools compare?

| Arm | What decides the tool parameters |
| --- | --- |
| `control` | Nothing: every stage runs the SHAPE tool at its CARD defaults, the representation is the frozen pretrained backbone (zero-shot) |
| `shape` | SHAPE's own verify-reason-refine loops per stage (organoid segmentation VRR, crop VRR, representation VRR) with `gpt-5.6-luna` at high effort |
| `codex` | The Codex CLI (`gpt-5.6-luna`, high effort) through the shared agent-arm stage protocol of `benchmarks/full_CPJUMP1/harness/codex_arm.py`: it sees the tool CARD, the tool results and the images it asks to inspect, and requests tool calls; the harness runs them |

The three arms get identical prepared inputs, the same fixed split, the same supervised read-out
and the same per-stage tool-round budgets (8 tool-call rounds per image for segment and crop, 8
rounds for the representation stage, a fixed SSL epoch budget for every training call).

The comparison is **descriptive**: 23 wells give 8 held-out wells, too few for a powered test
between arms. Intervals are bootstrap over held-out wells and are reported as such.

## Data contract

- Source: `/scratch/users/your-user/shape_benchmark/CO-FC_timecourse` on the cluster (private;
  not in Git). Per time-point folders `bs, d2, d4, d6, d8, d10, d12, d16` hold one
  `<well>_<seq>_1_1_Phase Contrast_001.tif` (904 x 1224 uint16) and one
  `<well>_<seq>_2_1_GFP_001.tif` per well. The top-level and `Control/` two-page `.tiff` files
  are the same two channels stacked (page 0 = phase contrast, page 1 = GFP, verified pixel-identical);
  they are not used.
- Prepared by `runners/e2e.py prepare`: `public/images/<image_id>.tif` is the phase-contrast
  channel copied byte-for-byte (sha256 recorded); `private/gfp/<image_id>.tif` the GFP channel;
  `private/images.json` the mapping image_id -> well, condition, day, source paths and hashes;
  `private/splits.json` the fixed well split. Image, well and plate identifiers are pseudonymised
  (`i0000`, `w00`, `p000`); condition, day, GFP, well position and source file names are not
  available to any arm's decisions or tools.
- 184 images = 23 wells x 8 time points. Well F6 (one image, day 4 only) is excluded from the
  study because it has no trajectory; the exclusion is recorded in `PREPARE.json`.
- Split (unit = well, every time point of a well in one split, seed 20261024, stratified by
  condition): development_train 11 wells (6 Control + 5 FC, 88 images), development_val 4 wells
  (2 + 2, 32 images), evaluation_only 8 wells (4 + 4, 64 images). Held-out wells enter frozen
  inference and the scorer only.

## Pipeline (every arm)

1. **segment**: `Organoid_Segmenter_Tool` with the fixed custom CO Cellpose checkpoint
   (`shape_seg_co`, sha256 recorded), adaptive rescue disabled, phase-contrast input only. The
   control arm delivers the CARD defaults; the SHAPE arm runs the per-image organoid VRR
   (`OrganoidSegmentationAgenticVerifier`, the standard SHAPE path, no cohort calibration); the
   Codex arm requests calls through the stage protocol. An image with zero detections is a
   recorded empty mask, never a fabricated organoid.
2. **crop**: `Single_Cell_Cropper_Tool` on the arm's own mask (every object is one crop; crops
   are cut from the raw uint16 phase image at the delivered boxes). SHAPE: crop VRR per image;
   Codex: stage protocol; control: defaults.
3. **embed**: `Cell_State_Analyzer_Tool` (DINOv3 ViT-B/16). Development crops only (at most 128
   per image) enter SSL fitting and selection; validation wells select; the selected checkpoint
   (or the pretrained backbone) embeds every retained crop. The epoch budget is the harness's
   fixed rule on the development_train image count and is not a lever.
4. **predict** (shared, trusted): probes per arm on its own embeddings (rows L2-normalised,
   StandardScaler fitted on development_train crops, equal total weight per training well, no
   refit on validation; held-out labels are read only by the scorer).
   **Primary: the time point.** A multinomial logistic regression over the eight days
   (`C` from {0.01, 0.1, 1, 10} by development_val balanced accuracy, ties: smaller C); an image's
   prediction is the argmax of the mean class probabilities of its objects. A ridge regression of
   the day (alpha from {0.1, 1, 10, 100} by development_val MAE) is reported alongside.
   **Secondary: the condition** (FC = 1), logistic regression with the same protocol, threshold 0.5.
   (User decision 2026-10-24: the day, not Control vs FC, is the end point; the condition is a
   strong, size-driven and plate-row-confounded signal.)
5. **score** (shared, trusted): held-out image-level (and crop-level) exact-day accuracy
   (chance 12.5 percent), adjacent-time-point accuracy, balanced accuracy, the 8 x 8 confusion
   matrix, per condition and per day; day-regression MAE and Spearman; condition accuracy and
   AUROC; bootstrap over the 8 held-out wells (1000 draws, seed 20261024). Read-outs from the withheld GFP
   channel inside each delivered object: the **integrated background-subtracted GFP** (sum of GFP inside the
   object minus its area times the mean GFP of the pixels outside every object of the image; user definition
   2026-10-25), the mean GFP and the GFP-positive pixel fraction (threshold fixed as the 99th percentile of
   the baseline images' GFP, computed once in `prepare`), summarised per condition and day. **GFP probe:** a
   ridge regression of the integrated GFP on the embedding (same fitting protocol as the day regression,
   target standardised on development_train), reported on held-out images as Spearman rho and MAE next to a
   label-only reference (least squares on one-hot day + condition, fitted on development_train) that bounds
   what the withheld labels alone explain; both are bootstrapped over wells. Also the Spearman correlation
   between the arm's FC probability and object GFP on held-out images. Segmentation descriptives (objects per
   image, largest-object area over time), stage durations, tokens and tool-call counts.

No batch-correction stage: one plate, no technical batch key.

### Variant `ssl_two_channel` (user request 2026-10-25)

Prepared with `prepare --ssl-channels PhaseContrast,GFP`. Segmentation and crop decisions are unchanged: every
arm's segmenter, crop tool, VRR and Codex inspection read the phase-contrast image alone. The public row adds
`raw_file`, a two-channel (PhaseContrast, GFP) uint16 stack pixel-identical to the two source channels, and the
crop stage cuts the SSL crops from that stack at the boxes chosen on the phase image. The embed stage therefore
trains and embeds on two-channel crops (the analyzer's channel adapter maps channel 1 to red and channel 2 to
green; by default each crop channel is scaled by its own 1st to 99th percentile, so absolute GFP brightness is
only preserved when an arm selects `intensity_norm=per_plate`). The read-out is the same scorer; its GFP probe
and condition read-out are then no longer independent of GFP, which `metrics.json` records as a caveat. The
variant is recorded in `PREPARE.json` (`variant`, `ssl_channels`, `stage_channels`), copied into `CONFIG.json`,
and selects the Dataset declaration (`spec.for_config`). Runs of the two variants are kept under separate
prepared folders and run names (`prepared_2ch`, `<arm>_2ch`).

### Protocol `co-fc-timecourse-ssl19-cvcol4-v1` (cross-validation, user request 2026-10-26)

`prepare --protocol co-fc-timecourse-ssl19-cvcol4-v1` (default: `co-fc-timecourse-well-stratified-11-4-8-v1`, the split above)
prepares the same 184 images with no held-out wells. SSL stage (the public `usage`): `development_train` 19 wells (10 Control
+ 9 FC, 152 images) and `development_val` 4 wells (C6, D6, F4, G3; 32 images); the SSL stage is label-free, so the pixels of
every well may train it. Read-out (the private `cv_fold`): the plate column, 4 folds (columns 3 to 6; 6, 6, 6 and 5 wells).
The scorer scores each outer fold once, out of fold (the next fold, cyclic, validates, the other two fit), so all 184 images
are scored by four classifiers and the bootstrap is over the 23 wells, out of fold (`heldout_*` metric keys count the cells
scored out of fold). The protocol was run with phase contrast only: it cannot be combined with `ssl_two_channel`.
`run` records the protocol in CONFIG.json (`protocol`); with this protocol the shape arm's segmentation is one organoid cohort
(`segmentation_regime: cohort`, `pipeline._segmentation_cohort_organoid`: SHAPE's `segmentation_cohort.run` with the organoid
segmenter), and its crop is the harness's own crop cohort (`pipeline._cropping_cohort`, as for every full SHAPE run). Control and
Codex segment every image by themselves. The agent wording of the embed stage and the withheld list differ (no held-out wells,
the cross-validation folds are withheld). The technical smoke (`--smoke`) is 32 images, one well per fold.

## What is withheld from the arms

Condition, day, well position, GFP pixels, GFP-derived values, source file names and the split
labels of held-out wells. The SSL stage receives `split` (train / val) and the pseudonymous
`well` / `plate` for its own view pairing, as in the other end-to-end studies.

## Budgets and models

`gpt-5.6-luna` at high reasoning effort for every reasoning stage of the SHAPE and Codex arms
(the `all-high` protocol of the cardiomyocyte studies), through `LLM_API_KEY` / `LLM_BASE_URL`
(Stanford gateway). Codex CLI version pinned by the harness (`codex-cli 0.162.0`). Wall-clock
deadline per arm: none (the finite tool-round and epoch budgets bound the work); every LLM
request, tool call, parameter, decision and usage is recorded by the existing harness.

## Layout

- `harness/dataset.json`: the pinned contract (plate layout, time points, image geometry, split).
- `runners/`: `data.py` (prepare, load, private tables), `spec.py` (the `Dataset` declaration
  used by the shared harness), `e2e.py` (preflight, run, stage workers, native trials, pack),
  `score.py` (predict and score), `compare.py` (three-arm table), `snapshot.py` (source identity
  for cluster transfer), offline tests `test_*.py`.
Raw images, masks, crops, embeddings, weights and credentials stay outside Git.

Run (from the repository root, inside a GPU allocation, `shape_bench` environment):

```bash
E2E="case studies.cardiac organoids.fc_timecourse.runners.e2e"
python -m "$E2E" pack --out $SNAPSHOT.tar.gz            # prints the source sha256 the runs are bound to
python -m "$E2E" prepare --source $SOURCE --out $PREP
python -m "$E2E" run --prep $PREP --out $RUNS/control --arm control --assets-root $ASSETS \
  --organoid-model $CO_CKPT --expected-source $SOURCE_SHA --cores 8
# replace control with shape or codex for the other arms; --smoke runs a 6-image technical check first
python -m "case studies.cardiac organoids.fc_timecourse.runners.compare" --control $RUNS/control \
  --shape $RUNS/shape --codex $RUNS/codex --out $RUNS/comparison
```

`$ASSETS` is the pinned asset root of the shared harness (`benchmarks/full_CPJUMP1/harness/cli.py assets`:
DINOv3 source at the pinned commit, its ViT-B/16 weights, the Cellpose-SAM weights); `$CO_CKPT` the CO
organoid checkpoint (`shape_seg_co`). `LLM_API_KEY` / `LLM_BASE_URL` come from the environment for the
`shape` and `codex` arms; the `codex` arm also needs the Codex CLI and Node on `PATH`.

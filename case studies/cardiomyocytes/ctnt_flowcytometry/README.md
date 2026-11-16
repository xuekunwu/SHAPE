# cTnT Flow Cytometry: Well-Level Proportion Prediction

Status: authenticated metadata checks and an image-to-well E2E runner are implemented.
The runner verifies all pixels, freezes well-disjoint train/val/held-out splits
(35/9/43 wells, targeting 40/10/50%) balanced by experiment_date then plate_id,
executes the shared SHAPE image stages and predicts `ctnt_positive_percent` (0–100). See
[runner instructions](runners/) for preparation, VM preflight and completion criteria.
Successful model performance must be established by completed run artifacts.

- [Harness](harness/): pinned data contract and metadata entry point.
- [Runners](runners/): categorized preparation, E2E execution, monitoring and recovery.

In the pinned standalone CSV `ctnt_positive_percent` is complete and consistent
within each `(plate_id, well_id)` group, and each group has one `flow_sample_id`.
Background metadata are partly incomplete, which does not imply a missing flow
label. The archive's older CSV is excluded by the pinned hash check.

Source: [SHAPE-agent/iPSC-CM-cTnT_flowcytometry](https://huggingface.co/datasets/SHAPE-agent/iPSC-CM-cTnT_flowcytometry).
Live iPSC-derived cardiomyocytes are imaged in brightfield, followed by cTnT
staining and flow cytometry. Predict the measured **cTnT-positive proportion per
culture well** as a numeric value from that well's brightfield images.

## Confirmed Label Unit

**One culture well has one measured proportion; multiple images share it.** This
was confirmed by the dataset owner on 2026-10-14. An image can therefore carry
the well's label in an exported table, but it has no independent flow measurement.
Keep all views from a well in the same split, aggregate their information and
evaluate once per well. Also group related wells by biological sample or
differentiation batch where needed to prevent leakage.

Use globally unique well IDs (for example, experiment + plate + well), not an
ambiguous coordinate such as A01. Preserve `assay_id` for the actual flow record.
Record imaging and harvest times, sampled area, missing views and gating population.
The gated flow fraction and the imaged population must be comparable; it need not
equal the fraction of every object detected in an image. Do not assign the well's
fraction or a thresholded version of it as an observed label for each cell.

## Proposed Data Contract

These are the proposed prepared-record fields. The source already supplies
`image_id`, `plate_id`, `well_id`, `site_id`, `relative_path`, `donor_id`,
`flow_sample_id` and `ctnt_positive_percent`; source `well_id` is a local
coordinate and must be qualified by `plate_id`. Other prepared fields remain
subject to image/protocol verification:

- Image manifest: `image_id`, `sample_id`, `assay_id`, globally unique `well_id`,
  `plate_id`, `differentiation_batch_id`, `acquisition_id`, `split`,
  `brightfield_path`, image hash, axes and imaging time.
- Assay table: `assay_id`, `well_id`, measured `ctnt_positive_percent` in [0, 100],
  explicit source unit (`percent`),
  staining/flow protocol, gating population, measurement time and QC status.
  Include positive/total gated event counts and gate/control records when
  available; do not fabricate missing counts. Resolve repeat assays by a
  development-defined policy before assigning one target per well.
- Pairing table: all image-to-well-to-assay relationships and any predefined
  view weights. Repeated copies of a well label must agree.
- Predictions: one row per `well_id`, contributing image IDs, retained/excluded
  cell counts, aggregation policy and predicted `ctnt_positive_percent` in [0, 100].

Reject missing mappings, nonfinite labels, values outside the declared unit range
and conflicting measurements. A label of `1` cannot mean 1% or 100% without
explicit units. Freeze grouped splits before image-processing calibration,
representation learning or supervised fitting. A well is the scoring unit;
the independent unit for inference may be a larger biological sample/batch.

## End-to-End Evaluation

Brightfield views -> segment -> crop -> SSL embedding -> admitted batch handling
-> well representation -> supervised proportion predictor -> frozen well-level
predictions -> independent flow measurements. Use training labels for the
regression head and validation labels for selection; keep test flow labels sealed.

A simple planned baseline is a training-only regressor over pooled embeddings,
with the same fitting policy for every arm. Alternatively, a multiple-instance
model can learn from all cells/views in a well using its shared proportion label.
Averaging cell-level cTnT probabilities requires a separately validated live-cell
probability model; flow proportions alone do not provide observed cell labels.
Fix weighting/aggregation on development data: equal-view, cell-count and
sampled-area weighting estimate different quantities.

Proposed primary metric: **well-macro MAE in percentage points**
(`abs(predicted_ctnt_positive_percent - ctnt_positive_percent)`). Also report RMSE, signed
bias, calibration and R-squared where defined, with uncertainty resampled over
independent biological samples/batches. Correlation alone is insufficient.
Compare against a training-set mean-proportion predictor on identical splits.

Report all scheduled wells, absent/failed views and cell-retention coverage.
Duplicating a well's label over images must not increase its weight in scoring or
confidence intervals. Any image-level outputs are exploratory predictions,
not individually validated cTnT fractions.

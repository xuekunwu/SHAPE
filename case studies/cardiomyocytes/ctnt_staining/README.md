# cTnT Staining: Cell-Level Classification

Status: authenticated dataset access and metadata admission are implemented.
Image preparation, cell-label derivation and full pipeline execution are pending.

- [Harness](harness/): pinned data contract and metadata entry point.
- [Runners](runners/): executable inspect/fetch/audit commands.

The embedded CSV carries image-level experimental metadata with no per-cell
intensity or label columns.

Source: [SHAPE-Morph/iPSC-CM-cTnT_staining](https://huggingface.co/datasets/SHAPE-Morph/iPSC-CM-cTnT_staining).
Fixed iPSC-derived cardiomyocytes are imaged in brightfield, DAPI and cTnT/Cy5.
The target is a **per-cell cTnT-positive label (0/1)** derived from fluorescence,
predicted using the corresponding brightfield morphology alone.

## Reference Labels

Define reference cell IDs and spatial masks independently of the compared arms,
register the channels and quantify background-corrected cTnT intensity per cell.
Freeze the intensity statistic, compartment, illumination/background correction
and threshold protocol using assay controls or development data before evaluation.
Do not assume a universal numeric threshold or retune it on held-out predictions.

DAPI can support reference annotations, but a nucleus is not automatically a
whole-cell mask. Resolve multinucleated cells, merged objects and crowded borders
explicitly. Retain raw measurements, threshold provenance, label and QC status;
ambiguous or saturated measurements are flagged, never silently labeled negative.
If thresholds are calibrated per acquisition batch, require predefined independent
controls rather than fitting a threshold to each test field.

Every evaluable reference cell has one label. Each arm's predicted objects must
be matched to these references with the same frozen matching rule. Reference
fluorescence images, masks, intensities and held-out labels remain outside the
agent's and predictor's inputs. Arm-specific masks must not redefine ground truth.

## Proposed Data Contract

These are proposed prepared-record fields; the pinned source schema is audited
separately and does not yet supply the private cell-reference table:

- Image manifest: `image_id`, `sample_id`, globally unique `well_id`,
  `differentiation_batch_id`, `acquisition_id`, `split`, `brightfield_path`,
  image hash, axes and pixel size.
- Private reference manifest: `image_id`, DAPI/cTnT paths and hashes, verified
  channel identifiers and registration transform/QC.
- Cell labels: `image_id`, `reference_cell_id`, reference mask ID, corrected cTnT
  intensity, intensity statistic, threshold, threshold protocol ID, `ctnt_positive`
  (0/1 for evaluable cells), QC status and exclusion reason.
- Predictions: `image_id`, predicted cell ID, mask/crop provenance,
  `ctnt_positive_probability` in [0, 1] and binary prediction using a frozen
  **classifier** threshold. This threshold differs from the fluorescence
  threshold that defines the reference label.

Training labels may be exposed to the supervised head after splits are frozen;
held-out references are available only to independent evaluation.

## End-to-End Evaluation

Brightfield -> segment -> crop -> SSL embedding -> admitted batch handling ->
supervised cell classifier -> frozen predictions -> independent fluorescence labels.

Predeclare a matched-cell classification endpoint (proposed primary: AUROC;
secondary: AUPRC, sensitivity, specificity, F1, balanced accuracy and probability
calibration). Pair it with mandatory reference-cell detection recall, precision,
matching coverage and excluded-cell counts by sample and label. Report undefined
metrics explicitly for single-class groups. A classifier score on detected cells
alone is not a complete end-to-end success criterion: missed positive cells and
unmatched false detections must remain visible.

Aggregate to per-image/sample positive fractions as a secondary endpoint with
explicit denominators. Require sample-level confidence intervals and a predefined
coverage criterion before comparing arms; do not improve apparent classification
performance by dropping difficult cells. Segmentation accuracy needs independent
mask annotations; cTnT intensity alone does not establish mask accuracy.

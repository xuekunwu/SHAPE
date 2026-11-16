# Cardiomyocytes

Two end-to-end applications for iPSC-derived cardiomyocytes. Both predict cTnT
positivity from **brightfield only**, with different reference measurements.

| Study | Acquired images | Reference target | Prediction |
| --- | --- | --- | --- |
| [cTnT staining](ctnt_staining/) | Fixed cells: brightfield, DAPI, cTnT/Cy5 | Per-cell binary label derived from cTnT fluorescence intensity | Probability and yes/no call for each cell |
| [cTnT flow cytometry](ctnt_flowcytometry/) | Live cells: multiple brightfield views per well | One cTnT-positive proportion per culture well from subsequent staining and flow cytometry | Numeric proportion per well |

- Staining: [SHAPE-agent/iPSC-CM-cTnT_staining](https://huggingface.co/datasets/SHAPE-agent/iPSC-CM-cTnT_staining).
- Flow cytometry: [SHAPE-agent/iPSC-CM-cTnT_flowcytometry](https://huggingface.co/datasets/SHAPE-agent/iPSC-CM-cTnT_flowcytometry).

Both datasets are public on Hugging Face (CC BY 4.0); donor demographics and
diagnosis are withheld. Each sub-study has `harness/` and `runners/`, a pinned
dataset contract and executable metadata admission commands. The flow archive
carries an older CSV beside the pinned standalone one; use the pinned CSV and
never silently substitute the embedded one.
For staining, the CSV is inside `cTnT_composite/`; a bounded archive-prefix read
retrieves it without downloading all images. Channel order, image integrity,
license conditions and staining cell-label generation still require verification.

**Confirmed pairing:** each flow-cytometry label belongs to one culture well;
multiple images from that well share the label. All its images stay in one split,
and the primary prediction and scoring unit is the well.

## Shared Protocol

1. Match images to experimental samples and assay records; retain immutable raw
   files and checksums. Resolve donor/cell line, differentiation batch, well,
   acquisition session and technical-replicate identities where available.
2. Split by the independent biological unit before fitting or tuning. Keep every
   cell, field, technical replicate and repeated measurement from a unit together.
   Hold out differentiation batches where feasible; donor-generalization claims
   require donor-disjoint evaluation. Do not infer IDs from unverified filenames.
3. Segment and crop brightfield with the shared SHAPE tools. Fluorescence may
   support independent reference annotations, but DAPI/cTnT-derived masks must
   not enter the primary brightfield-only prediction pipeline.
4. Fit SSL on training images; use development validation for permitted selection,
   then freeze before held-out inference. Fit the supervised head with training
   labels; validation selects its hyperparameters and decision threshold. Held-out
   labels are available only to the scorer.
5. Retain all four tools' admission rules and shared VRR behavior. SSL keeps its
   multi-epoch/checkpoint semantics. Batch correction requires a measurable
   technical effect and a suitable study design; do not force Harmony or remove
   differences in cardiomyocyte purity as if they were technical noise. Corrected
   held-out inference needs a train-fitted out-of-sample transform; otherwise use
   an uncorrected primary pipeline and report joint unlabeled correction separately
   as transductive analysis.
6. Deliver predictions before opening held-out references. Record exclusions,
   failures and denominator changes. Bootstrap uncertainty at the independent
   sample/batch level, not over correlated individual cells.

Stage adapters, manifests, audit records and delivery/scoring separation follow
the [parent design](../README.md). Classification and regression heads are
study-specific additions; UMAP or clustering is descriptive and does not itself
produce calibrated cTnT predictions.

## Decisions Needed Before Execution

- Complete image inventory, channel/axis layout, acquisition metadata and license.
- Staining cell masks/intensity-derived labels; flow labels are present in the pinned CSV.
- Biological replicate identities and feasible train/validation/test grouping.
- Staining: reference cell masks, registration, controls and intensity thresholds.
- Flow: image existence for the verified well/assay mapping, gated event population
  and the flow/imaging sampling relationship (the source label unit is percent).
- Supported comparison arms and equal training/search budgets on pinned code.

Fixed and live acquisition are separate domains. Cross-dataset transfer is an
additional experiment requiring matched metadata and independent validation;
a fixed-cell classifier is not automatically a live-cell classifier.

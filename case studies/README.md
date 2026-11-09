# SHAPE Case Studies

End-to-end studies on experimental datasets, from microscopy images to a
biological prediction and independent evaluation. Public-dataset evaluations
belong in `benchmarks/`; experimental applications belong here, including when
their data are distributed through Hugging Face.

## Study Areas

- [Cardiomyocytes](cardiomyocytes/): brightfield prediction of cTnT positivity,
  with separate cell-level staining and well-level flow-cytometry studies.
- [Cardiac organoids](cardiac%20organoids/): experimental study scope pending.
- [Tissues](tissues/): Xenium Prime human lung cancer crops, three-arm SSL
  embeddings scored by a cell-type linear probe on held-out spatial tiles.

## Shared End-to-End Design

Use [`benchmarks/full_CPJUMP1`](../benchmarks/full_CPJUMP1/README.md)
as the architectural reference: prepare and verify inputs, segment, crop, learn
representations, apply batch correction when admitted, deliver predictions,
and score against independent reference measurements. Study-specific adapters
must define prediction units, labels, splits and biological endpoints;
CPJUMP1's five-channel input and gene-retrieval endpoints do not apply here.

Cardiomyocyte metadata admission is
executable; the full image pipelines remain study designs, not completed runs. Pin the actual
implementation commit, dataset revision and model assets when preparing a run.
Reuse shared SHAPE tools and VRR/admission rules rather than copying a second
implementation into each study. Quality-gate skips remain valid outcomes.

Compare arms on identical inputs, splits, supervised-head policy and budgets.
Record every observable internal operation: tool calls, parameters, evidence,
LLM requests/responses, repairs, decisions, stage gates, selected artifacts,
timestamps, usage, errors and checkpoint/resume provenance. This is a required
audit contract for the future harness; this documentation does not implement it.
Keep secret credentials out of exported records.

Raw images, private labels, model weights and large outputs remain outside Git.
Each implemented study should provide its data contract, preparation adapter,
runner and scorer. Missing mappings or labels must block
scoring, not silently become negative examples or zero proportions.

## Directory Responsibilities

Each cardiomyocyte sub-study has `harness/` (contracts and pipeline logic),
and `runners/` (entry points and instructions).
Cardiac organoids and tissues have the same structure. [Shared metadata admission](_shared/README.md)
handles HF authentication and verification for both cardiomyocyte datasets.
No private metadata tables or raw experimental images are committed.

## Retained Legacy Files

`basic_usage.py`, `iPSC-cardiomyocyte.tif` and `baseball.png` were retained when
`examples/` was renamed. They are generic usage/smoke fixtures, not the datasets,
labels or end-to-end runners for these studies. The TIFF has not been verified
as a member of either linked cardiomyocyte dataset.


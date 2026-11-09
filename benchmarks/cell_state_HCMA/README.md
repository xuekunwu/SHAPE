# cell_state_HCMA — end-to-end label-free cell-state inference

The first **fully end-to-end** SHAPE benchmark on the **Human Cell Morphology Atlas
(HCMA, ~2,546 multi-source images)**. The agent receives raw fields and a single
instruction — *"do label-free cell-state inference on these images and output an AnnData
with an embedding and a cluster assignment"* — and runs the whole pipeline:

```
raw image  ->  segment  ->  crop single cells  ->  label-free embedding + Leiden cluster  ->  AnnData
```

Cell-type labels are **never** shown to the agent. A **fixed evaluation script**
([`evaluate.py`](evaluate.py)) owns all scoring. This split is the whole point: the agent
only *produces*; the benchmark *judges*.

## Why this benchmark exists

`representation_HPA` starts from pre-segmented crops and asks a narrow question
(cell-line predictability per organelle). HCMA closes the loop end-to-end **and** stresses
the one thing that makes multi-source morphology atlases treacherous: **technical
variation**. HCMA is stitched from many labs; a large fraction of cell types appear in
only one or two source images. Any embedding — even a useless one — can "recognize" such a
cell type by recognizing its imaging condition. A benchmark that ignores this rewards
batch memorization and calls it biology.

## What the agent outputs (its entire deliverable)

One AnnData (`cell_state_analyzed.h5ad`) with:

| slot | meaning |
|---|---|
| `adata.X` (or `obsm[...]`) | the label-free morphology **embedding** |
| `obs['leiden_<res>']` | the unsupervised **cluster** assignment |
| `obs['cell_id']` | stable per-cell id (used only to rejoin held-out labels) |
| `obs['image_name']` | source image id — the CV group **and** the multi-source unit |

The agent must not put cell-type labels in the AnnData; it never had them.

## What the fixed script computes ([`evaluate.py`](evaluate.py))

Two metric families against the held-out cell-type labels:

1. **Clustering vs cell type** — `ARI`, `NMI` between `obs['leiden_<res>']` and GT cell
   type. (Clustering quality, independent of any classifier.)
2. **Representation label probes** on the embedding — `kNN accuracy`, `linear-probe
   accuracy`, `linear NLL`. (Representation quality, independent of the cluster count.)

Two confound guards, both **enforced by the script** (an agent cannot bypass them):

- **(A) Multi-source subset** — only cell types present in **≥ `min_sources` (default 2)
  independent source images** are scored. Single-image cell types are dropped and counted
  (`n_cell_types_dropped_single_source`), so "recognizing the imaging condition" cannot
  pose as biology.
- **(B) Image-grouped GroupKFold** — every probe is cross-validated with folds **grouped
  by `image_name`**: a source image is never split across train and test. An embedding that
  encodes per-image technical signature therefore scores *poorly*, exactly as it should.

Test-fold classes unseen in their train fold are dropped from that fold's scoring so
accuracy/NLL stay well-defined; per-fold means are reported with `n_folds`.

### Labels stay with the evaluator, never the agent

Ground-truth is supplied to `evaluate.py` only, two ways:

```bash
# join an external GT table onto the agent's AnnData by a stable id, then score:
python -m benchmarks.cell_state_HCMA.evaluate \
    --adata cell_state_analyzed.h5ad \
    --image-key image_name --cluster-key leiden_0.5 --embedding-key X \
    --labels data/HCMA/labels.parquet --join-key cell_id --cell-type-key cell_type \
    --min-sources 2 --out scores.json
```

`--labels` is a csv/parquet with `[join-key, cell_type]`; it is joined onto `obs` on
`--join-key` (default: the image key). If a trusted harness already injected
`obs['cell_type']`, `--labels` may be omitted.

## The four end-to-end stages (each runs its own VLRR loop)

| stage | what it does | tool / module | GPU |
|---|---|---|---|
| **1 — segment** | raw image → per-cell masks, VLRR-verified | `Cell_Segmenter_Tool` (SAM) | yes |
| **2 — crop** | tight per-cell bbox crops, VLRR-verified | cropper (mask-check triage VLRR) | yes |
| **3 — embed + probe** | DINOv3 embedding (zero-shot OK, no training needed), visualize classes, train a linear probe | `Cell_State_Analyzer_Tool` + [`evaluate.py`](evaluate.py) | yes |
| **4 — screen** | *is batch correction needed, given how few images each cell type has?* — the dataset-construction gate | [`stage4_screening.py`](stage4_screening.py) | **no** |

## Stage 4 — dataset-composition screening (built & validated)

Stage 4 is the *screening at the start of the dataset*: before trusting any embedding,
decide from metadata alone (no GPU) **(a)** the multi-source subset — which cell types have
≥ `min_sources` independent source images — and **(b)** whether **batch correction is
required**, i.e. whether cell type is so confounded with lab/source that a classifier could
cheat on the imaging condition. It runs as a real VLRR loop (mirrors
`shape.agent.verification.BaseVerifier`: `quantify → localize → adjudicate → VLRRResult`)
with one lever, `min_sources`, raised until the surviving subset clears the source-confound
gate (`NMI(cell_type, source)`), or — if it never clears — the verdict is
`BATCH_CORRECTION_REQUIRED`.

Three terminal verdicts:

| verdict | meaning | example |
|---|---|---|
| `PASS` | multi-source subset adequate, cell type ⟂ source | a balanced replicate design |
| `BATCH_CORRECTION_REQUIRED` | populated but confounded; raising `min_sources` only empties it → systematic lab/source-specific cell types | a confounded design |
| `INSUFFICIENT_DATA` | too few multi-source labels; batch correction can't manufacture sources — acquire more images | a single-source cache |

## Files

| file | role |
|---|---|
| [`Prompt.txt`](Prompt.txt) | the agent-facing task (label-free; produce AnnData) |
| [`config.py`](config.py) | canonical hyperparameters + eval settings (`BenchmarkConfig`) |
| [`stage4_screening.py`](stage4_screening.py) | **stage 4** — VLRR dataset-composition / batch-correction screening |
| [`evaluate.py`](evaluate.py) | **the fixed evaluator** (stage 3 readout) — ARI/NMI + image-grouped kNN/linear probe |
| [`run_shape_hcma.py`](run_shape_hcma.py) | SHAPE reference runner: crops → embed+cluster → evaluate |
| [`run_hcma.sh`](run_hcma.sh) | **launch script** (referenced per the benchmark convention) |

## How to run (SHAPE reference)

```bash
# GPU-bound; launch via `!` so the sandbox does not SIGKILL background CUDA.
bash benchmarks/cell_state_HCMA/run_hcma.sh \
    solver_cache/HCMA/crops.parquet \
    data/HCMA/labels.parquet \
    solver_cache/HCMA/run1
# add --fast for a screening pass, --zero-shot for the pretrained-backbone baseline.
```

`crops.parquet` is a manifest with `[path, cell_id, image_name]`; `run_shape_hcma.py` can
alternatively discover a directory of `"<image_id>__<cell_id>.png"` crops via `--crops-dir`.
The runner feeds the crops to `Cell_State_Analyzer_Tool` (label-free contrastive embedding
+ Leiden), writes the AnnData, and calls the fixed `evaluate.evaluate()` with the held-out
labels.

## Status / prerequisites

- **Stage 4 (screening): built and validated on real JUMP metadata and synthetic
  regimes.** No GPU. Snaps onto HCMA labels unchanged.
- **Stage 3 evaluator: implemented and validated** on synthetic AnnData for both layouts
  (embedding in `obsm` and in `adata.X`; in-obs labels and external-join labels).
  Multi-source subset, drop-count, and image-grouped GroupKFold all verified.
- **Stages 1–2 + stage-3 embedding: implemented (runner + launch script), NOT yet run.**
  Blocked on two things:
  1. **GPU** — stages 1 (SAM), 2 (cropper) and 3 (DINOv3, even zero-shot) need a GPU.
  2. **HCMA data** — the 2,546 images + `data/HCMA/labels.parquet` are NOT in-repo
     (only `data/JUMP` and `data/NeurIPS22-CellSeg` exist). Downloaded data is gitignored.

  Once the GPU is attached and HCMA data is provisioned, `run_hcma.sh` runs stages 1→3.

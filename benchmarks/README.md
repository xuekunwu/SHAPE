# SHAPE Benchmarks

Datasets and eval drivers used to evaluate the SHAPE agent end to end, one directory per
tool per dataset: `benchmarks/<eval-task>_<dataset>/`. Each directory's `README.md` gives the
dataset, the GPU it needs and the commands that run every arm.

## Index (tool -> eval task -> dataset)

| Directory | Tool under test | Dataset | Key driver(s) |
|---|---|---|---|
| `segmentation_NeurIPS22-CellSeg/` | `Cell_Segmenter_Tool` | NeurIPS22-CellSeg | `run_cellseg_benchmark.py` + `runners/cellseg_arms.py` (Cellpose-SAM, Cellpose3, micro-sam, StarDist, an LLM alone, ReAct, SHAPE and the SHAPE cohort arm; hard/medium/easy Cellpose-SAM AP50 bands); `runners/run_all_arms.sh` runs them all |
| `cropper_NeurIPS22-CellSeg/` | `Single_Cell_Cropper_Tool` | NeurIPS22-CellSeg | `run_benchmark.py` + `runners/cropper_arms.py` (five arms on the ground-truth masks: MaskCrop, CellProfiler as a control, an LLM alone, ReAct, SHAPE) |
| `batch_correction_JUMP/` | `Batch_Corrector_Tool` (Harmony) | JUMP (Arevalo protocol, scenarios 1-5) | `run_batch_benchmark.py` (`prepare` / `control` / `suite` / `report`); the arms in `runners/` (the SHAPE arm is `runners/shape_arm.py`); vendored upstream preprocessing and its provenance in its `README.md` |
| `representation_RxRx1/` | `Cell_State_Analyzer_Tool` (SSL fine-tune of DINOv3) | RxRx1 (HEPG2 by default) | `run_rxrx1.py` (`prepare` / `assets` / `control` / `external` / `shape` / `llm` / `react` / `score` / `finish`; the SHAPE arm is `runners/shape_arm.py`, scored by well mAP in `runners/score.py`) |
| `full_CPJUMP1/` | the whole chain: segment -> crop -> embed -> batch-correct -> cluster | CPJUMP1 (A549 Cas9 CRISPR) and ModalityProbe3 | `harness/cli.py`; `runners/run_all.sh` picks the dataset, the phase and the arms (control, SHAPE, Codex, Claude Code, Biomni) |
| `full_HCMA/` | -- | -- | not a benchmark in this release: the shared harness (`harness/`, `workspace_template/`) that `full_CPJUMP1` and `representation_RxRx1` import |
| `baselines/` | -- | NeurIPS22-CellSeg | `run_baseline.py`: external coding agents (Claude Code, Codex) on the segmentation and cropper tasks, head to head against SHAPE |
| `_shared/` | -- | -- | `segmentation_metrics.py`, `full_metrics.py` (the metric contract of the `full_*` benchmarks), the shared ReAct runner, `live_agent_check.py` |
| `gpu_services.sh` | -- | -- | start / stop / status the shared Cellpose-SAM and DINOv3 services on a GPU node, and print the URLs to export |
| `benchmark tasks.xlsx` | -- | -- | the task table: datasets, the arms and the readouts each task is scored on |

## Data

The benchmark inputs are on Hugging Face,
[`SHAPE-agent/benchmark-data`](https://huggingface.co/datasets/SHAPE-agent/benchmark-data). From
the repository root, download a folder into `data/` (git-ignored) with the `hf` CLI of
`huggingface_hub`:

```bash
hf download SHAPE-agent/benchmark-data --repo-type dataset --include "NeurIPS22-CellSeg/*" --local-dir data
```

| Folder | Size | Used by |
|---|---|---|
| `NeurIPS22-CellSeg/` | 3.4 GB | `segmentation_NeurIPS22-CellSeg`, `cropper_NeurIPS22-CellSeg`, `baselines` |
| `CPJUMP1/` | 12.8 GB | `full_CPJUMP1` (CPJUMP1) |
| `JUMP/` | 15.4 GB | `batch_correction_JUMP` |

RxRx1 is not in that dataset; `representation_RxRx1` builds its inputs from a local RxRx1 copy
(`run_rxrx1.py prepare --data-root`). ModalityProbe3 is fetched from the Cell Painting Gallery by
`full_CPJUMP1/runners/download_modality_probe3.py`.

Drivers that take a dataset root also read `$SHAPE_BENCH_DATA_ROOT` and append their own
dataset name to it, so after the download

```bash
export SHAPE_BENCH_DATA_ROOT=$PWD/data
```

makes `$SHAPE_BENCH_DATA_ROOT/NeurIPS22-CellSeg` the default of the segmentation drivers. With the
variable unset the `--data-root` flag is required, so a driver never silently reads the wrong tree.
Drivers that name their input explicitly (`--data-dir`, `--base`, `--data`) take the `data/...` path
as shown in their README.

# SHAPE Benchmarks

Datasets and eval drivers used to evaluate the SHAPE agent end to end, one directory per
tool per dataset: `benchmarks/<eval-task>_<dataset>/`. Each holds its key agentic driver
script, its query `Prompt.txt` (where the tool takes an LLM-facing task), and any
tool-specific metrics/helpers. Superseded process/probe scripts from earlier task
investigations are not kept here (git history preserves them) -- only the live,
re-runnable evaluation surface.

## Index (tool -> eval task -> dataset)

| Directory | Tool under test | Dataset | Key driver(s) |
|---|---|---|---|
| `segmentation_NeurIPS22-CellSeg/` | `Cell_Segmenter_Tool` | NeurIPS22-CellSeg | `run_cellseg_benchmark.py` + `runners/cellseg_arms.py` (seven arms: Cellpose-SAM, Cellpose3, micro-sam, StarDist, gpt-5.6-luna alone, ReAct, SHAPE; hard/medium/easy Cellpose-AP50 bands); standalone SHAPE runner `runners/run_shape_neurips_cellseg.py` |
| `segmentation_Inhouse_OrganoidSeg/` | `Organoid_Segmenter_Tool` | In-house organoid segmentation (`CO_cpsam_finetune/v8/test`) | `run_organoid_inhouse.py` (direct `shape_seg_co` AP50 baseline, then SHAPE organoid arm) |
| `cropper_NeurIPS22-CellSeg/` | `Single_Cell_Cropper_Tool` | NeurIPS22-CellSeg | `run_benchmark.py` + `runners/cropper_arms.py` (five arms on the ground-truth masks: MaskCrop, CellProfiler as a control, gpt-5.6-luna alone, ReAct, SHAPE); the arm functions in `runners/run_shape_crop.py`. See its own `README.md`. |
| `batch_correction_JUMP/` | `Batch_Corrector_Tool` (Harmony) | JUMP (Arevalo protocol, scenarios 1-5) | `run_batch_benchmark.py` (`prepare` / `control` / `suite` / `report`); the six arms in `runners/` (the SHAPE arm is `runners/shape_arm.py`); vendored upstream preprocessing and its provenance in its `README.md`. |
| `representation_RxRx1/` | `Cell_State_Analyzer_Tool` (SSL fine-tune of DINOv3) | RxRx1 (task-116 v3 preps; HEPG2 by default) | `run_rxrx1.py` / `runners/run_rxrx1.sh` (`prepare` / `control` / `shape` / `score` / `finish`; the SHAPE arm is `runners/shape_arm.py`, scored by well mAP in `runners/score.py`). |
| `full_HCMA/` | the whole chain: `Cell_Segmenter_Tool` -> `Single_Cell_Cropper_Tool` -> `Cell_State_Analyzer_Tool` -> `Batch_Corrector_Tool` -> Leiden | HCMA (1,205 raw images, 4 sources) | `harness/cli.py` (`prepare` / `smoke` / `control` / `agent claude` / `finish`); the Claude Code arm gets only the tool CLIs in `workspace_template/tools/` + `TASK.md`. See its `README.md`. |
| `skills/` | -- | -- | Portable per-tool skill packs (`segmentation_cell/`, `single_cell_cropper/`) -- prompt + contract, not runnable drivers |
| `_shared/` | -- | -- | `segmentation_metrics.py` (efficiency/cost metrics shared by >1 segmentation-family eval-task dir); `live_agent_check.py` |

Each directory's own driver docstring (`--help`, or the module docstring at the top of
the file) is the authoritative usage reference; this table is a map, not a manual.

## Dataset location

Benchmark drivers take their dataset root from `$SHAPE_BENCH_DATA_ROOT`, and
append their own dataset name to it. Set it once:

```bash
export SHAPE_BENCH_DATA_ROOT=/path/to/shape_benchmark
```

`run_shape_neurips_cellseg.py` then defaults to
`$SHAPE_BENCH_DATA_ROOT/NeurIPS22-CellSeg`, the JUMP batch-correction benchmark to
`$SHAPE_BENCH_DATA_ROOT/JUMP`, and so on. With the variable unset the
`--data-root` flag is required, so a driver never silently reads the wrong tree.

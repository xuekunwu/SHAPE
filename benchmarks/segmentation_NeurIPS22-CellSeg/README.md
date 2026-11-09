# segmentation_NeurIPS22-CellSeg

Instance cell segmentation on the NeurIPS 2022 Cell Segmentation Challenge test set (1000
heterogeneous images: brightfield, phase contrast, DIC, fluorescence). Seven arms segment the same
images; each delivers one integer instance mask per image, scored against the ground-truth mask.

## Arms

| # | arm | what runs | tool calls |
|---|---|---|---|
| 1 | `cellpose_sam` | `Cell_Segmenter_Tool` (Cellpose-SAM) once, at the tool's default parameters | 1 |
| 2 | `cellpose3` | Cellpose 3 `cyto3`, the diameter estimated by its size model, on the highest-variance channel (`channels=[0, 0]`) | 1 |
| 3 | `microsam` | micro-sam automatic instance segmentation, `vit_b_lm`, defaults (tiled above 2048 px) | 1 |
| 4 | `stardist` | StarDist2D: `2D_versatile_he` for an RGB image with a bright background, else `2D_versatile_fluo` on the highest-variance channel (inverted when the background is bright) | 1 |
| 5 | `gpt` | gpt-5.6-luna alone: sees the image (long side <= 1024 px), returns one polygon per cell as JSON; the polygons are rasterized | 0 |
| 6 | `react` | gpt-5.6-luna + `Cell_Segmenter_Tool` on the shared ReAct runner (`benchmarks/_shared/react_arm.py`, `REACT_ARM_CONTRACT.md`; below) | <= 8 (`BUDGET`) |
| 7 | `shape` | the SHAPE agent: `shape.solver.solve` with `results/Prompt.txt`, `Cell_Segmenter_Tool` only, gpt-5.6-luna, `max_steps` 10, the segmenter's agentic VRR (below) | the segmenter calls it made (<= 8 per VRR stage) |

A set arm, `shape_cohort`, gives SHAPE the whole set instead: one `solve()` over every listed image (all bands
together) with the segmentation cohort on (`segmentation_cohort=True`, `image_workers` 8; CARD "Cohort mode"):
the images are grouped by their preflight sample type, a group of at least 10 is calibrated on a sample, its
consensus applied to the rest, and an image with a measured problem routed to its own VRR. The `shape` arm, one
`solve()` per image, never forms a cohort. Run by `run_cellseg_benchmark.py run-set --arm shape_cohort`; each
record adds the image's `parameter_source` (calibration / cohort / per_image_vrr) and the admission-standard
`problems` of its delivered call; its seconds and tokens are the solve()'s divided by its images;
`<run>/shape_cohort/set/set_meta.json` holds the cohort record. `compare --reference shape_cohort` takes the
paired deltas from it. A partly recorded cohort run is refused (the cohort spans the whole set).

The three LLM arms get the same target definition (the `## Target` section of `results/Prompt.txt`) and
use the same gateway engine (`build_vision_engine`, the one SHAPE's judge uses). Code:
`runners/cellseg_arms.py` (the `cellseg_` prefix keeps it apart from the cropper benchmark's `arms.py`).
The `gpt` arm keeps the complete polygons of a reply cut at its token cap and skips malformed ones
(counted in its `trace.json`).

### The react arm (an LLM plus the tool)

`react` is the shared ReAct design of every benchmark (`benchmarks/_shared/REACT_ARM_CONTRACT.md`): the
shared runner `react_arm.run_react` with this benchmark's adapter `cellseg_arms.SegmenterAdapter`, run by
`run_cellseg_benchmark.run_react_arm`. One session per image, `--workers` (default 8) sessions at a time; the
in-process Cellpose-SAM model takes one call at a time (one lock). Each round's prompt holds SHAPE's task text
in this benchmark (the whole `results/Prompt.txt`, the shape arm's question), the image's file facts (name,
size, channels, dtype), the tool's parameter table and the budget. The parameters are SHAPE's segmentation
lever set (`SegmentationAgenticVerifier`) without SHAPE's own levers `resample` and `color_class_filter`, which the
tool runs at its defaults: `diameter`, `cellprob_threshold`, `flow_threshold`, `min_size`, `niter`, `tile_overlap`,
`augment`, with name, type and default only; no CARD text, tool description, rule or SHAPE readout. An observation
is the tool's returned dict and the overlay PNG the tool wrote for that call (<= 1024 px); the session's first
question also carries the raw input image (the file the segmenter gets, as SHAPE's vision reads render it).
Between rounds the agent may also run its own Python (the shared runner's `python` action, not a round) on the
raw image and each call's mask and overlay (`SegmenterAdapter.python_inputs`; never `labels/`), so it designs
its own label-free validation signal. The
budget is
SHAPE's: `BUDGET` (8) rounds, the default call is round 1 (`admission_call`), at most the verifier's
`concurrency()` calls per round (1 in-process). The agent's `finish` names the call to deliver; a session
without a valid finish delivers its default call (`delivery` = `default_session_unfinished`). Records per image:
`TRACE.jsonl`, `CALLS.json`, `SELECTED.json`, `trace.json`, `mask.tif`; the record adds `rounds`,
`delivered_call`, `delivery`, `stop` and `wall_seconds_per_image` (`seconds` is the image's own session, which
shares the GPU with the other sessions). An earlier ReAct design (up to 10 turns, a hand-picked parameter
subset, the tool's description) is superseded by the arm described here.

### SHAPE's segmentation loop (`shape/tools/cell_segmenter/CARD.md`, "Agentic VRR")

Calls are named by role: `default` (the tool at its defaults), `reference` (the system's reference
call), `agent_1`, `agent_2`, ... (the agent's), `fallback_<n>` (the deterministic fallback, only when the
reasoner fails).

1. A vision-LLM preflight read of the raw image (specimen, visible object classes; information for the agent).
2. `default`: Cellpose-SAM at its CARD defaults, measured (QC readouts, colour populations of the masks).
3. Admission (rules, no LLM): a measured problem of `default` -- a shape gate (tiny / large masks, low
   solidity), the scale check (`mask_diameter_px` 2x or more off the diameter used), a colour minority
   (one colour population <= 10% of the masks), the guard (no cells, invalid overlay) -- starts the
   tuning; none delivers `default` as it is. Under-detection (`missed_foreground_fraction`,
   `count_miss_fraction`) is shown as information and is no problem: it counts every textured pixel no
   mask covers, the target's or not.
4. `reference`: `default` at the measured diameter (scale problem) and with the colour filter the vision
   LLM chooses for the target population (colour minority, `color_class_filter.decide`), when they apply.
5. The agent's rounds (one call per round: the in-process model runs one call at a time): it assesses
   each overlay, explores (several levers in one call, from any finished call) and narrows to
   `focus_levers` of its choosing; once a round has run under a focus, the focus is never widened (the
   one rule the code checks). It stops when satisfied; backstops: a plateau of 3 calls without a better
   assessed severity, 8 rounds in all (`default` included).
6. Delivery: the reference default (`reference`, else `default`), unless the agent's final choice has
   strictly fewer measured problems AND a better assessment (made in a round) than it.

## Images: three difficulty bands

Bands are the Cellpose-SAM-alone per-image AP50 over all 1000 images: **hard** AP50 < 0.30,
**medium** 0.30-0.60, **easy** > 0.60.
A run draws `n` images per band once, into `<run>/lists/<band>.txt`; every arm runs exactly those images.
`lists --exclude <list files>` leaves images out before the draw, so an evaluation run can be drawn
from the images no earlier development run (pilots, dev30, seed checks, stress test, smoke) listed
and no SHAPE rule was set on its images.

## Readouts

Per arm x band and pooled (`all`), over the images every compared arm attempted:

| readout | definition |
|---|---|
| AP50 | mean per-image `TP / (TP + FP + FN)` at IoU 0.50 (maximum-cardinality matching; no confidence scores, so not COCO AP) |
| mAP@0.50:0.95 | mean per-image AP over IoU 0.50, 0.55, ..., 0.95 |
| Runtime per image | mean wall-clock seconds of the arm on the image (`time.perf_counter` around the arm's call, inside its process; the model loaded once per arm, before the first image, is not counted; the LLM arms include the gateway's latency) |
| Tool calls per image | mean segmentation-tool calls (0 for `gpt`) |
| AP50 gain per tool call | `sum(AP50 - Cellpose-SAM AP50) / sum(tool calls)` over the images, the reference being this run's `cellpose_sam` arm (the definition of `benchmarks/_shared/segmentation_metrics.py`); n/a with no tool calls |
| Tokens per image | mean LLM tokens the gateway reported for the image (0 for the domain tools) |

An attempted image without a delivered mask (an error, an unparsable LLM reply) scores AP 0 and is
counted in `errors`. `compare` also reports SHAPE minus each arm, AP50 paired per image: the mean, a
bootstrap 95% CI (10 000 resamples of the images, seed 0) and a two-sided exact sign test (ties,
|delta| <= 1e-9, dropped). Scoring: `runners/run_cellpose_sam_only.evaluate_instance_segmentation`
(arrays, no `shape` import; the same definition as `runners/run_shape_neurips_cellseg.py`, tested).

## Run rules

- One image set per run, drawn before any arm runs; every arm runs every image once, from an empty
  per-image directory. A resumed run (`RUN_ROOT`) only completes the images without a record; no image
  is rerun or replaced after it is scored.
- The ground truth is read only to score a delivered mask; no arm, prompt or agent sees it.
- Fixed settings: gpt-5.6-luna for `gpt`, `react` and `shape`; `gpt` <= 32 000 output tokens, the image
  at <= 1024 px; `react` SHAPE's budget (above), images at <= 1024 px; `shape` `max_steps` 10,
  its VRR at most 8 calls; the domain tools at the settings in the table. No arm is tuned per run.
- Order on one GPU node: the four domain tools one after another, alone on the node (their runtime is
  their own); then `react` and `shape` one after the other (the GPUs are exclusive-process: two
  Cellpose-SAM processes cannot share one), with `gpt` (no GPU) alongside them. Every arm segments
  in-process (`SHAPE_SAM_SERVER_URL` unset). Thread pools are capped at the job's CPUs.
- Checks before the run: the checkout at the expected commit with no tracked
  change, `import shape` from it, the data, the tool envs, the GPU, the gateway, and a one-image smoke of
  each domain tool. Each arm's `run_meta.json` records the commit, host, env, packages and times.

## Records and traces

```
<run>/lists/<band>.txt               the drawn images
<run>/<arm>/run_meta.json            commit, host, env, package versions, model, start / end
<run>/<arm>/<band>/records.jsonl     one scored record per image (the resume point)
<run>/<arm>/<band>/<image>/          trace.json (+ the arm's own files: SHAPE's verification_summary.json
                                     = every VRR trial, decision and prompt record; gpt's gpt_reply.json =
                                     its prompt and full reply; react's TRACE.jsonl / CALLS.json /
                                     SELECTED.json = every round, call and the delivery); masks and overlays
                                     stay on scratch
<run>/<arm>/<band>/run_summary.json  the band's readouts for that arm
<run>/compare/                       readouts.csv, per_image.csv, paired_vs_shape.csv, compare.json,
                                     summary.md
<run>/audit/<arm>/<band>/<image>.json  the per-image audit record (export_run_records.py), + index.json
<run>/logs/                          per-arm logs, the lists log, the job's output
```

**Audit records** (`export_run_records.py`, schema `cellseg-run-record/1`): one JSON per arm x image,
built only from the run's own artifacts. Each holds the input (image id, band), the provenance
(commit, host, env, packages, model, max_steps, times), the arm's configuration, every step in order
(the tool call and its arguments, its result, and the rationale the agent stated for it as the run
recorded it: react's per-round thought, calls and observations; SHAPE's preflight read, admission, every call by name with its
parent, QC readouts, measured problems and the agent's assessment, each round's analysis, hypotheses,
focus_levers and moves, the final choice and the override check), the final output, the score and
cost, the status (delivered, errors, stop reasons, reasoner failures, fallback rounds, focus history,
the warnings / errors its log printed) and the source files used or missing.

The run writes the audit records after `compare`, then packs the run's non-image files
(`runners/pack_noimg.sh <run> [job ids]` -> `<run>.noimg.tgz` beside the run dir: json, jsonl, csv, md,
log, txt; no masks, overlays or figures). That archive, unpacked, is what `results/<run>/` holds. For an
earlier run: `python export_run_records.py --run-dir <run>`.

## Environments

`shape_bench` runs `cellpose_sam`, `gpt`, `react` and `shape`. `cellpose3` runs in its own env
(cellpose 3.x; cellpose 3 and 4 cannot share an env; the arm refuses another major version). StarDist
(TensorFlow 2.15 / Keras 2, numpy < 2) and micro-sam (conda-forge, its own pytorch) do not fit in
shape_bench either, so `seg_stardist` and `seg_microsam` are built next to it once. The arm code
imports `shape` only inside the arms that need it.

```bash
python benchmarks/segmentation_NeurIPS22-CellSeg/run_cellseg_benchmark.py lists --run-dir <run>     --n-per-band 20 --seed <s> [--exclude <earlier runs' lists/*.txt>]
python benchmarks/segmentation_NeurIPS22-CellSeg/run_cellseg_benchmark.py run --run-dir <run> --arms <arms>
```

`compare` leaves out an arm without records and says so.

## Data layout

```
<data-root>/images/<stem>.{png,bmp,tif,tiff,jpg}
<data-root>/labels/<stem>_label.tiff
```
`--data-root` defaults to `$SHAPE_BENCH_DATA_ROOT/NeurIPS22-CellSeg`.

## Layout

The top level holds only the benchmark's own entry points; everything else is in two subfolders.

```
run_cellseg_benchmark.py          the seven-arm benchmark: lists / run / compare / smoke
export_run_records.py             the per-image audit records of a run dir (above)
runners/
  cellseg_arms.py                 the seven arms
  cellseg_config.py               arms, bands, n_per_band, seed, model, agent_max_steps, + the runners'
                                  defaults (named apart from the other benchmarks' config.py)
  pack_noimg.sh                   a run's non-image files -> <run>.noimg.tgz
  replay_shape_delivery.py        offline replay of SHAPE's delivery rules on a run's own calls (no LLM)
  run_shape_neurips_cellseg.py    the standalone SHAPE-vs-Cellpose-SAM runner (run_batch, its scoring, the
                                  opt-in --octotools-plus controller of task 083); imported by the shape arm,
                                  by segmentation_Inhouse_OrganoidSeg and by _shared/segmentation_metrics.py
  run_cellpose_sam_only.py        Cellpose-SAM over the whole set -> the per-image CSV; its scoring is shared
                                  with run_cellseg_benchmark.py and the cropper benchmark
  run_concurrent_benchmark.py     the many-process SHAPE runner (cell_state_HCMA/run_pipeline.sh:
                                  python -m benchmarks.segmentation_NeurIPS22-CellSeg.runners.run_concurrent_benchmark)
  test_benchmark.py               the tests of this folder (offline: fake arms, engines, tool and solve())
  test_cellseg_react.py           the react arm's adapter and runner (a fake engine, a stub segmenter)
results/
  Prompt.txt                      the SHAPE agent's task prompt (its ## Target section is every LLM arm's target)
  NeurIPS22-CellSeg-cell_image_metrics.csv
                                  Cellpose-SAM-alone metrics of all 1000 images (+ modality, staining, tissue),
                                  the band source
  <run>/                          the reported run's non-image records (above)
```

The scripts put `runners/` on `sys.path`, so its modules are imported by name (`import cellseg_arms`,
`from cellseg_config import DEFAULT`, `import run_cellpose_sam_only`).

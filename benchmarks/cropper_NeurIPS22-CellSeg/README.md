# cropper_NeurIPS22-CellSeg

Formal single-cell cropper benchmark on NeurIPS22-CellSeg with five arms and five
registered readouts. The question: given the same mask and targets, how well does each
arm frame every target cell in one square crop, and at what cost?

## Layout

The top level holds only the benchmark's own entry points; everything else is in subfolders.

```
run_benchmark.py                  the five-arm benchmark: prepare / run / compare
export_run_records.py             the per-image audit records of a run dir (below)
runners/
  cropper_arms.py                 the five arms (named apart from the other benchmarks' arm modules)
  cropper_config.py               the legacy runners' defaults (named apart from the other benchmarks' config.py)
  cellprofiler_native.py          the CellProfiler SaveCroppedObjects helper (runs in the CellProfiler env)
  run_shape_crop.py               the MaskCrop and SHAPE arm functions the arms reuse; its legacy standalone runner
  pairing.py                      the image / label pairing and cropper defaults the drivers reuse
  mask_cache.py                   the cached Cellpose-SAM mask of one image
  run_concurrent_benchmark.py     the many-process SHAPE runner (cell_state_HCMA/run_pipeline.sh:
                                  python -m benchmarks.cropper_NeurIPS22-CellSeg.runners.run_concurrent_benchmark)
  conftest.py                     puts runners/ and the top level on sys.path for the tests
  test_cropper_arms.py            the five-arm runner and its arms (offline: stub engines, the real tool)
  test_run_shape_crop.py          run_shape_crop.py
  test_export_run_records.py      export_run_records.py, the seed and the export wiring of the cluster scripts
results/                          a run's own records, as the CellSeg benchmark writes them (no pixel files)
```

The scripts put `runners/` on `sys.path`, so its modules are imported by name (`import cropper_arms`,
`import run_shape_crop`, `from cropper_config import DEFAULT`).

Run outputs follow the SHAPE results layout (`projects/SHAPE/runtime/docs/paths.md`): everything under
`${SHAPE_RESULTS_ROOT:-/scratch/groups/your-group/your-user/shape_benchmark/results}/cropper_NeurIPS22-CellSeg/`:
`logs/`, `scripts/`, `cache/`, and one directory per
run, `pilot_n<N>_s<SEED>_<commit8>_<UTC>_<jobid>/` (RUN_META.json, the outputs, `logs/<arm>.log`, COMPLETE when the
validation passed). Only the key / final evaluation runs, chosen by the user, are
committed: the run's non-image archive (records, compare, audit, SHAPE agent logs, RUN_META, the job and arm
logs), unpacked into this folder's `results/<run_id>/`; development runs stay on scratch and in their local
mirrors.

With `CROPPER_WORK_ROOT` set (node-local scratch), `run` runs each image's arm in
`<CROPPER_WORK_ROOT>/<arm>/<image>`, then copies its non-pixel files (`export_run_records.PIXEL_SUFFIXES` excluded) to
`<arm>/images/<image>/` at the same relative paths, points the record's paths there and removes the work dir, for
failed images too. The run dir keeps every JSON / log of every call but no crop pixels (crop TIFFs, masks, zips,
montages); they are reproducible from commit + seed. Unset: the arm writes into `<arm>/images/<image>/` directly.

## Input

Every arm crops the image's **ground-truth** instance mask (`labels/<image>_label.tiff`): the arms
differ only in how they crop, never in segmentation. The targets are the mask's objects with
`area >= min_area` (the tool's default, the cell segmenter's `min_size`), the same `cell_label` set
for every arm. `neighbor_contamination` and `target_retention` are measured on that same ground-truth
mask. Ground truth never reaches an arm beyond the mask itself (no scores, no labels).

## Arms

1. `maskcrop` -- MaskCrop (`Single_Cell_Cropper_Tool`) at its fraction-mode defaults, one call.
2. `cellprofiler` -- a control, the standard-pipeline reference: native CellProfiler 4.2.8
   `SaveCroppedObjects`, then the benchmark's shared MaskCrop square-geometry adapter. Its scored
   crops share MaskCrop's geometry, so its quality readouts equal MaskCrop's by construction; it
   reports implementation parity (the same targets and boxes, checked per crop) and runtime. Its crop
   image files are raw source pixels (MaskCrop writes display-normalized crops); the mask readouts do
   not depend on pixels.
3. `gpt` -- `gpt-5.6-luna` alone, one shot, producing the crop boxes directly.
4. `react` -- an LLM plus the tool: `gpt-5.6-luna` with MaskCrop as its only tool on the shared ReAct
   runner (`benchmarks/_shared/react_arm.py`, `REACT_ARM_CONTRACT.md`), SHAPE's task, parameters and budget,
   none of SHAPE's verifier layer; between rounds it may run its own Python (the runner's `python` action,
   not a round) on the image, the given mask and each call's crops and metadata (`CropToolAdapter.python_inputs`;
   no scoring ground truth beyond the given mask). Its question: with the same tool, task and budget, can a generic agent
   loop match SHAPE?
5. `shape` -- SHAPE's agentic crop loop (`CropAgenticVerifier`) with `gpt-5.6-luna` and MaskCrop.

What each LLM arm sees, what it can change and its budget (`cap` = SHAPE's per-round concurrency cap,
`cores // THREADS_PER_CALL` = `cores // 2`; the pilot's 8 cores give cap 4):

| arm | sees | can change | budget |
|---|---|---|---|
| `gpt` | the task "Preserve the entire target while minimizing neighboring cells", square in-bounds boxes; the input image with the mask's object boundaries (SHAPE's input overlay); every object of the mask (`cell_label`, bbox, area); the image shape; the targets of its request | the square box of every target, directly | one shot, no feedback: one request per 64 targets (reply length), each prompt carrying all objects; an invalid reply is asked again with the error, a gateway error after a 30 s wait, at most 3 tries |
| `react` | SHAPE's task text for the image (`CropAgenticVerifier.task_text` on SHAPE's input record: one crop per object, the number of objects at `min_area`); the input (the image and mask file names, the image size, the fixed `min_area`); the tool's parameter table, name, type and default only (SHAPE's CARD levers `margin_mode`, `margin_frac`, `margin`, `recenter`, `tight_buffer_px`, `crop_size` with the defaults SHAPE resolves for the mask, and `cell_overrides`: cell id -> the per-cell keys `margin_mode`, `recenter`, `auto_tighten`, `min_containment` SHAPE's cell rules set); the line "Tuning the tool's parameters can improve the tool's performance."; the budget and the time left; every call's own tool output (its returned dict without the crop objects) and the crop tiles the tool wrote (at most 16 images per round, longest side 1024 px). No CARD text, readouts, guard, overlay or panel | any parameter of the table per call, the rest at its default; `finish` names the delivered call | SHAPE's: at most 8 rounds (`BUDGET`), round 1 the tool at its defaults (`default`; SHAPE's `admission_call`), at most `cap` calls per later round (`CropAgenticVerifier.concurrency()`; at most `1 + 7 x cap` = 29 calls at 8 cores); one reply per round, an invalid reply or request is an error observation that uses its round |
| `shape` | the input overlay and the crop panels of the previous round's calls; the input facts with per-cell quantiles, and per-cell readouts through its cell rules; every call's readouts and guard, relative to its parent and to the defaults call (when that is not its parent), with the noise and the Pareto front; the objectives and the final rule; the time left. Its mask check (a vision judge of screened crops) runs once, after the choice, on the delivered call only; its counts reach no prompt | the same CARD levers and per-cell keys through the same cell rules, each call's parent, and the final choice among the admissible calls | at most 8 rounds, round 1 the defaults call alone, at most `cap` calls per later round (at most `1 + 7 x cap` tool calls); 3 tries per reasoner question |

ReAct runs one session per image, the images one after another as every arm's (`run_benchmark.py run`),
with `workers=1`; its calls of a round run at once. Its records per image: `react/images/<image>/<image>/`
`TRACE.jsonl` (every round: prompt sha256, image refs, the reply, the action, the observations), `CALLS.json`,
`SELECTED.json`, and `react/images/<image>/REACT_SUMMARY.json`; the record's `trace` carries them. ReAct and
SHAPE run on the same cores (`--cores`, recorded in `run_meta.json`); the comparability check requires
ReAct's cap to equal SHAPE's and its `react_tool_budget` to equal `1 + 7 x` that cap. What SHAPE has beyond
ReAct is its verifier layer and loop: its readouts and guard, the input overlay and crop panels, the admission
on the defaults call, the noise and Pareto bookkeeping, the hypothesis list and the mask check. No arm gets an
example value or a hint.

Delivery: SHAPE's scored crops are its delivered call (`resolution.result`, which the loop points at the
delivered copy in `tool_cache/`). ReAct's are the call its `finish` names (`delivery = "agent_finish"`); a session that ends without a
valid finish delivers its `default` call (`delivery = "default_session_unfinished"`; SHAPE's is its `choice`).
The harness never chooses among ReAct's calls; a delivered call that does not crop every target once, square,
makes the image an error record. The mask check of SHAPE, run on the delivered call only, rejects crops
whose mask it judges wrong; with ground-truth masks such rejects are judge errors or real ambiguities. Rule: every target is still
delivered once and scored, rejected crops included; each crop carries `triage_rejected`, and
`n_triage_rejected` per image (null for the other arms) is reported as a separate readout. Every arm must
return exactly one in-bounds square crop per target for an image to count as successful; one failing image
is an error record and does not stop the arm.

Note on the readouts: with the ground-truth mask as the input, the per-crop readouts both agentic arms see
and optimise (`neighbor_contamination`, `target_retention`, `centroid_offset`, ...) are computed on the
same mask the benchmark scores, so the agents optimise the evaluation metric directly; the comparison
measures how well each arm searches the crop geometry, not how well GT-free proxies track quality. SHAPE's
own prompt text (in `shape/`, unchanged here) still calls these readouts "GT-free"; on this input they are
not.

## Metrics

The formal comparison reports exactly:

| metric | direction | note |
|---|---:|---|
| `neighbor_contamination` | lower | other-cell foreground (ground-truth mask) inside the delivered crop |
| `target_retention` | higher | target-cell pixels (ground-truth mask) retained inside the delivered crop |
| `runtime_per_image` | lower | wall-clock seconds per image |
| `tool_calls_per_image` | lower for equal quality | domain-tool calls per image |
| `tokens_per_image` | lower for equal quality | LLM/VLM token delta per image; deterministic tools report `0` |

AP50 is not a readout: the input is the ground-truth mask (AP50 = 1 by definition) and no arm changes
the mask.

The output also records usage-reported `model_calls_per_image`,
`input_tokens_per_image`, and `output_tokens_per_image` as diagnostic fields.
Input/output counts are `null`
when the provider did not report them on every model call. `tool_calls_per_image`
counts actual crop-tool invocations, including SHAPE's parallel candidates;
model calls are a separate quantity.
SHAPE also reports `reasoner_calls_per_image`, `vision_calls_per_image`, and
their separate token counts, metered per image. A category's token count is `null` if any of
that image's calls lacks provider usage. The original `tokens_per_image` remains the total.

Every summary mean is over the comparable set: the selected images every present arm finished,
the CellProfiler control aside (`summary.json["comparable_image_ids"]`; the control's own means are over
the comparable images it also finished, and its gaps are reported by the domain-tool check). An absent arm
does not blank the other means: it is listed in `missing_arms` and makes the five-arm `comparable` flag
false with its reason. `summary.csv` carries `n_comparable_images` and the `comparable` flag on every row.

All five arms are checked against the same selected images, ground-truth masks,
`min_area`, target `cell_label` values, and square-output contract. The two
domain-tool arms are additionally paired on exact geometry. CellProfiler's
native tight rectangular crops are retained under each image output for provenance;
the crops used for scoring are re-cut from the source image with Maskcrop's exact
`margin_mode="fraction"`, `margin`, `margin_frac`, bbox-center, and border-shift
rules. A run is valid only when all five arms return exactly one successful record
for every selected image and the same `(image_id, cell_label)` targets. The two
domain tools must also return identical final bboxes. This makes crop quality directly paired; expected
differences between the deterministic domain tools are runtime and implementation
overhead, not crop geometry.

## Pilot

The pre-experiment uses a seeded random subset of `N_IMAGES` images (default 30), run into one run dir:
prepare (code, environment, data and CellProfiler checks, the selection), then the five arms (each checks what
it uses: the gateway for the LLM arms, native CellProfiler for its arm), then compare, which runs after every
arm whether it finished or failed and validates the run.

`SEED` (default 0) seeds the image selection (`prepare --seed`, recorded in `selection.json`,
`provenance.json` and each arm's `run_meta.json`); seed 0 is the original pilot selection, the seed of every
run before 2026-10-06. `results/seeds.json` records every seed used, when and how it was drawn, and its runs.

Logs: `.../results/cropper_NeurIPS22-CellSeg/logs/crop<N_IMAGES>-all_<job>.out` and one `<run dir>/logs/<arm>.log`
per arm; run dir: `.../results/cropper_NeurIPS22-CellSeg/pilot_n<N_IMAGES>_s<SEED>_<commit8>_<UTC>_<jobid>` (named by
the run; `RUN_DIR` overrides it and a resume must name it). Time limits for 30 images (about 1.5 x 30 x the slowest image measured in the
n=10 pilots): maskcrop and cellprofiler 1 h, gpt 3 h, react 3 h (not measured yet on
the shared runner), shape 4 h; another
`N_IMAGES` scales each arm's hours to ceil(hours x N_IMAGES / 30), at least 1 h. The arms run at once, so the job
asks for the longest arm's hours plus 1 h (at most 48 h; 5 h for 30 images), `CORES` cpus (8; the arms mostly wait
for the model, and react and shape keep `--cores CORES`, the protocol's concurrency cap) and 16G. Each arm writes one record per image as it goes,
so a job that hits its limit loses only the images in progress; `resume` finishes them.
The first run pulls the CellProfiler image (`apptainer pull`), which needs network access.

The CellProfiler arm is intentionally isolated from the SHAPE environment. Set
`CELLPROFILER_PYTHON` to a Python executable or wrapper that has CellProfiler
4.2.8 installed. On the cluster, use an Apptainer wrapper around the official
`cellprofiler/cellprofiler:4.2.8` image.

Outputs:

- `selection.json` -- the shared image set (`mask_source: ground_truth`).
- `<arm>/run_meta.json` -- the arm's settings (for ReAct and SHAPE: cores, cap, budget, deadline;
  `resumed_utc` after a resume).
- `<arm>/records.jsonl` -- one record per image for each arm, including per-crop QC.
- `compare/summary.json` and `compare/summary.csv` -- the registered readouts over the comparable set.
- `compare/per_image.jsonl` and `compare/per_image.csv` -- status and metrics for
  every selected image and arm.
- `compare/per_crop.jsonl` and `compare/per_crop.csv` -- one row per delivered
  crop with its final box, source, contamination, retention, occupancy, dominance and
  `triage_rejected`; CellProfiler rows also retain the native rectangle and file path.
- `compare/domain_tool_pairs.jsonl` and `compare/domain_tool_pairs.csv` -- wide,
  paired MaskCrop/CellProfiler results for each `(image_id, cell_label)` target.

`compare/summary.json["comparability"]["domain_tools"]` records the selection
hash, image coverage, crop-key coverage, square compliance, bbox equality, and the
final comparability decision.
`compare/summary.json["comparability"]["all_arms"]` records whether all five arms
have complete image and target coverage and matching selection, model and budget settings.
The `COMPLETE` marker is written only after both checks pass. The audit records (`audit/<arm>/<image>.json`,
`audit/index.json`, below) are written after compare and its validation, whether the validation passed or not.

## Audit records

`export_run_records.py` (schema `cropper-run-record/1`): one JSON per arm x image, built only from the run's
own artifacts (`selection.json`, `provenance.json`, `<arm>/run_meta.json`, `<arm>/records.jsonl`,
`shape/images/<image>/agent_log.jsonl` + `agent_details.json`, and the job logs: the `<job>.out` / `.err`
whose header names the run dir, and `<job>-<arm>.log`). Each holds the input (image id, image and mask path,
seed, selection index), the provenance (commit, model, host, python, packages, the `llm` block: provider,
gateway host, reasoning_effort, temperature / top_p, max_tokens and response schema per question kind,
attempts, retry wait, response cache; seed, times, the job's status), the arm's configuration, every step in
order, the final output (the delivered call, crops count, delivery), the score (`neighbor_contamination`,
`target_retention`, `crop_coverage`, `n_crops`, `n_triage_rejected`), the cost (runtime, tool calls, tokens,
model calls, input / output / cached / reasoning tokens, SHAPE's reasoner / vision split), the status (status,
error, delivery, fallbacks, retries, the warnings / errors its log printed while that image ran) and the
source files and fields used or missing. Steps: maskcrop and cellprofiler their one tool call; gpt every request
(batch, attempt, targets, prompt, attached images, parameters, reply as returned, usage, error); react every
round (the prompt's sha256, images, reply, usage, the `thought` it stated, its action) and every tool call with
its parameters, the tool's output, images and error (records of the earlier design: every model step and tool
call with its change, cell rules, readouts and guard); SHAPE the defaults call, the admission, every question (prompt,
images, usage, the reply as returned: analysis, hypotheses, moves, final reasoning; errors and retries), the
rejected / ignored moves and other system rules, every call with its round, parent, change, configuration and
readouts, the plan's delivery or the final choice, and the mask check (counts, per-montage prompt, image and
reply, per-crop verdicts). Images are recorded as `{label, sha256, bytes}`, never their bytes. The rationale is
the one the agent stated in its reply; hidden chain-of-thought is never requested or stored (provider
reasoning is a token count only). Runs from before 2026-10-06 lack some of this (gpt's per-request record,
react's prompts and images, SHAPE's prompt images and mask-check montages, run_meta's host / python /
packages / llm / seed): the record lists each in `sources.missing` as "not recorded in this run".

The run writes them after compare; for an earlier run:

```bash
python benchmarks/cropper_NeurIPS22-CellSeg/export_run_records.py --run-dir <run> [--logs-dir <job logs>]
```

A run's records are archived as the CellSeg benchmark's are: its non-image archive (the `*_light.tgz` the run
scripts pack: records, compare, audit, SHAPE agent logs, RUN_META, the job and arm logs), unpacked into
`results/<run_id>/`, with the job's `.out` / `.err` in its `logs/`, and its seed added to `results/seeds.json`.

`--logs-dir` (where the job's `.out` / `.err` are) defaults to `<run>/logs` and the run dir's parent (where the
cluster job writes its logs); per-arm stage logs `<run>/logs/<arm>.log` are read too when present.
Optional, for a cross-run collection outside `results/`: `--into-data DATA` copies the run's non-pixel artifacts to
`DATA/runs/<run_id>/` (`run_id` = the run dir's
name): `selection.json`, `provenance.json`, `<arm>/run_meta.json`, `<arm>/records.jsonl`, `compare/`,
`shape/images/<image>/agent_log.jsonl` + `agent_details.json`, the job logs (in `logs/`) and `audit/`; never
crop TIFFs, zips, montage PNGs or other images (gigabytes, reproducible from commit + seed; `.gitignore`
excludes them under `results/`). It replaces the run's rows in `DATA/cropper-run_image_metrics.csv` (one row
per arm x image: run_id, commit, seed, n_images, arm, image_id, status, the five registered readouts,
`crop_coverage`, `n_crops`, the model-usage fields, SHAPE's reasoner / vision calls and tokens, `delivery`,
`n_triage_rejected`) and updates `DATA/runs/index.json` and `DATA/seeds.json`; a re-export replaces that run's
copy and rows.

The legacy MaskCrop-vs-SHAPE benchmark (predicted-mask regime; `runners/benchmark.py`, its README and
`run_agentic_ab.sh`) was removed; git history keeps it (last at commit 8d7d885b).

## Tests

```bash
python -m pytest benchmarks/cropper_NeurIPS22-CellSeg -q
```

`runners/test_cropper_arms.py` covers the five-arm runner and its arms, `runners/test_export_run_records.py`
the audit records, the `--into-data` copy, the seed and the cluster scripts' export wiring; the other test files
cover the legacy runners in `runners/`, which the five-arm runner partly reuses.

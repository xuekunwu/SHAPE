# Cardiac organoid segmentation evaluation

Seven arms on one frozen evaluation image/instance-GT manifest. This study reuses
the CellSeg model adapters and scorer, the shared ReAct runner, and the live
SHAPE solver/Organoid VRR. It does not maintain a second segmentation controller.

| Arm | Inference / agent |
| --- | --- |
| `cellpose_sam` | Generic CPSAM through `Cell_Segmenter_Tool`, once at its defaults (diameter 30) |
| `shape_seg_co` | Trained CO checkpoint through `Organoid_Segmenter_Tool`, once at its native-scale defaults (diameter=None), adaptive rescue disabled |
| `microsam` | Shared microSAM automatic instance segmentation, `vit_b_lm`; tiles above 2048 px |
| `stardist` | Shared StarDist2D deterministic choice of `2D_versatile_he` / `2D_versatile_fluo`, normalization and tiling |
| `llm` | GPT directly returns polygons for whole organoids; rasterized at original resolution, no segmentation tool |
| `react` | Shared ReAct + `Organoid_Segmenter_Tool` with the fixed CO checkpoint; no SHAPE QC/rules in its prompt |
| `shape` | Full `solve()` with `Organoid_Segmenter_Tool` only, per-image verification and VRR enabled |

All three LLM arms use **`gpt-5.6-luna`, reasoning effort `high`**, through
`LLM_API_KEY` / `LLM_BASE_URL=https://llm-gateway.example.edu/v1`.
The runner pins these settings rather than inheriting another experiment's model.
There is no alternate model/provider or deterministic no-LLM fallback. Before
an LLM arm, a small text + image structured preflight must pass with positive
reported usage; plain-completion fallback is disabled during this preflight.
Regular SHAPE engine schema repair/retries remain observable through its existing
engine behavior. The run fails on missing usage, including otherwise plausible answers.

## Inputs

User-specified defaults:

- Images: `/scratch/groups/your-group/your-user/shape_benchmark/Inhouse_Organoid_Seg`
- Instance GT: `/scratch/groups/your-group/your-user/shape_benchmark/Inhouse_Organoid_Seg_GT`
- Trained checkpoint: `/scratch/groups/your-group/your-user/shape_benchmark/CO_cpsam_finetune/shape_seg_co`

`--model-path` must name the actual checkpoint file. No directory scanning or
substitution with generic CPSAM is allowed. Native CO inference records the
loaded checkpoint's SHA-256 and checks it against this file for every image.

`prepare` recursively pairs images with GT at the same relative location in
the GT directory, either the same stem or `<stem>_seg` (override
`--label-suffix` if necessary). Flat `<stem>_seg.npy` Cellpose dictionaries
are also supported. Labels must be nonnegative 2D integer instance IDs with
background 0 and the same H x W as the image; RGB-colored annotations are rejected.
Ambiguous/missing GT blocks preparation. No files are deleted or rewritten.
NPY dictionaries are loaded only from the trusted local dataset.

The manifest records image/GT paths, dimensions and SHA-256, with a manifest
fingerprint. Optional `--sample-size` and `--seed` (default 112) select once
during preparation; all arms then use exactly those pairs. There is no GT-based
image difficulty selection. Inputs are rehashed before running and comparing.
GT is used only by the harness after prediction, never supplied to an agent.
The manifest labels these as user-provided evaluation data; training/test
non-overlap has not been verified from the remote data and is not asserted.

## CLI interface

Entry point: `run_organoid_segmentation.py`; its `--help` and each subcommand's
`--help` expose the full interface. These are runner interfaces, not a Slurm
submission packet; no remote job has been launched or synchronized by this change.

| Command | Required options | Other options |
| --- | --- | --- |
| `prepare` | `--manifest` | `--data-root`, `--image-root`, `--label-root`, `--label-suffix`, `--sample-size`, `--seed` |
| `run` | `--manifest`, `--run-dir`, `--arm` | `--model-path`, `--max-steps` (10), `--search-strategy` (`legacy` / `four_stage`), `--require-gpu`, `--preflight-only`, `--resume` |
| `compare` | `--manifest`, `--run-dir` | `--arms` (default all seven; a subset is explicitly labelled by its arm names) |

Use an allocated compute node for inference. Put the manifest, model download
caches, logs and results in a fresh scratch results directory outside input/model
directories and the source checkout. Set `XDG_CACHE_HOME`, `HF_HOME`,
`TORCH_HOME`, `MPLCONFIGDIR` and `TMPDIR` to scratch for the job; microSAM/StarDist
may download their pretrained assets on first use. Use the same checkout, frozen
manifest and run directory for every arm; pin/push the intended implementation
commit before a scientific remote run. The runtime records source hashes as well
as Git identity so uncommitted local tests cannot be mistaken for the base commit.

Run **one arm per process, all seven in the existing `shape_bench` environment**.
The four SHAPE/Cellpose arms require Cellpose 4; `llm` needs SHAPE's engine dependencies.
microSAM needs `micro-sam` and its automatic-segmentation dependencies; StarDist
needs `stardist`, `csbdeep`, TensorFlow. All must be available in `shape_bench`,
along with NumPy/SciPy/scikit-image/Pillow/tifffile. Each process imports only its
selected backend; separate sequential processes release GPU resources between
arms. Record the interpreter, package versions and import origins for every arm.
Selected-backend imports/API and SHAPE import origins are checked before paid
calls or image loops, including `--preflight-only`; this does not download/load
all model weights or establish their accuracy.
`--require-gpu` checks the applicable backend and rejects Cellpose CPU fallback.
Each image/session runs sequentially; model sharing and lazy loading follow the
existing adapters. Per-image seconds include lazy loading and SHAPE internal
work, audit I/O and final-mask scoring; adapter construction time is recorded
separately in `run_meta.json`. Cellpose native inference and LLM request seconds
are also recorded separately from the harness wall time.

`--resume` requires identical manifest, source hashes, model hash, settings,
package versions and Python executable. It verifies delivered mask hashes,
skips recorded images (including failures), and keeps interrupted attempts in
separate directories. To retry recorded failures, use a fresh run directory.
Corrupt/truncated JSONL is rejected rather than silently repaired.

## VRR and accounting

SHAPE retains the updated deterministic rules: early stop only for an admissible
zero-problem reference, confirmation skipped only for a valid choice, compact
prompts off. The default search remains `legacy`; `four_stage` is an explicit
alternative recorded in provenance. Neither setting disables Verify/Reconcile.
The organoid verifier retains native measurements, expected-versus-observed
effects, failed-call exclusion and reconciliation. Cohort calibration is disabled
in this per-image comparison. ReAct imports the same `BUDGET` and organoid
concurrency cap (1); the mandatory default consumes its first tuning round.
The ReAct parameter table follows the shared CellSeg contract (no SHAPE-only
`resample` lever). Nested adaptive rescue is disabled for all CO arms: additional
CO predictions must be visible agent/VRR calls.

Every attempted tool execution and outermost Cellpose `model.eval` inference request is recorded,
including failed attempts. Hidden extra inference within one call invalidates
a delivery; a failed call rejected before inference is recorded with zero
native requests and can be recovered by a later valid agent call. Candidate
masks are snapshotted before later calls can overwrite them. Successful native
QC warnings do not remove direct baseline predictions from scoring; a failed
SHAPE/ReAct delivery cannot be credited as a valid agent selection.
Cellpose's internal recursion over a one-image list is part of that one request,
not an additional tool call. Repeated outermost requests remain separately counted.

Per-image directories contain `record.json`, `trace.json`, `mask.tif`,
`TOOL_CALLS.jsonl`, `NATIVE_CALLS.jsonl`, candidate mask snapshots, and applicable
`LLM_CALLS.jsonl` / exact attached image bytes. SHAPE also writes
`verification_summary.json` and `actions.json`; ReAct writes its shared
`TRACE.jsonl`, `CALLS.json`, `SELECTED.json` and Python workspace audit.
LLM records include prompts, schemas, replies, usage, requested model/effort and
timestamps, with credentials redacted. These are generate-level observations;
individual HTTP retries internal to the SDK are not separate records.

## Scoring and completion

The shared CellSeg scorer matches instance masks independently at IoU thresholds
0.50 through 0.95 using maximum-cardinality assignment. AP is TP/(TP+FP+FN),
not confidence-ranked detection AP. Report macro AP50, AP75, mAP, precision,
recall and F1. Prediction/inference failures remain in the denominator as empty
predictions (all FN); missing GT instead invalidates the dataset.

`compare` requires the complete manifest for every requested arm and matching
code/settings/CO checkpoint; it writes `comparison.json` and `comparison.csv`.
It reports native calls, LLM calls, tokens and seconds per image, retune rates,
and delivered AP50 gain relative to `shape_seg_co`. Retune rates compare
consecutive actual, valid calls offline, not a fabricated final call or the
best GT-scoring candidate. Preflight and interrupted-attempt costs are reported
separately and included in total tokens, including after resume. GT scores never
feed back into either agent.

`COMPLETE` means all listed images finished without errors.
`COMPLETED_WITH_FAILURES` retains those failures, exits 2 and remains comparable.
`PREFLIGHT_ONLY`, interrupted runs and provenance mismatches cannot be compared
as completed evaluations. No biological accuracy conclusion follows from local
synthetic contract tests; real weights and evaluation images require a GPU run.

## Launch

The launch controller
requires `--expected-commit` (full SHA), clean `codex/organoid-agentic-vrr`, and
matching HEAD/origin branch refs before any GPU or paid model work. It never
fetches, resets, cleans, cancels or obtains an allocation. Source/API setup
occurs inside the child, preserving the existing allocated outer shell.

The initial `--phase smoke` defaults to one fixed image across all seven arms,
planning for 60 minutes plus a two-minute safety margin; this is an estimate,
not a measured runtime. `--check-only` performs imports/GPU probes without
inference or paid calls. Each arm is a separate sequential subprocess, and
all model/download caches, logs and outputs stay under the study's scratch
results directory. The controller checks the current host belongs to the
running Slurm job and preserves its `CUDA_VISIBLE_DEVICES`. The shape and
microSAM probes must identify the requested UUID as their sole visible GPU:
`GPU-084bed5f-96ca-460c-1cc8-4504d86f57ca` by default. The probe records memory,
architecture, CUDA versions and Cellpose precision policy; it does not force
BF16 on older GPUs. A tiny tensor operation catches unusable CUDA builds.
If PyTorch omits its device UUID attribute, the probe queries the device of
the actual current CUDA context through the [CUDA driver API](https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__CTX.html),
then reads its UUID. The observed UUID and query source are recorded; driver
query errors and real mismatches still fail the probe. It does not equate
`CUDA_VISIBLE_DEVICES=0` with physical nvidia-smi index 0.

The single default interpreter for **all seven arms**, preparation, probes and
comparison is `/home/groups/your-group/your-user/miniconda3/shape_bench/bin/python`,
as requested by the user. `--shape-python` can specify its actual location;
there are no separate microSAM/StarDist interpreter options. Its current remote
dependencies have not yet been inspected. Missing imports are reported in the
per-arm preflight logs before paid calls. StarDist must expose a TensorFlow GPU
in `shape_bench` for this GPU comparison. No existing environment is modified.
TensorFlow growth is enabled to avoid reserving the
whole GPU up front. Model OOMs remain recorded failures, not silent parameter
or model changes. Full runtime must be estimated from the smoke; full mode
requires a scheduler-owned `orgseg-batch*` job rather than assuming that the
current interactive allocation can hold the entire experiment.

The controller writes `RUN_META.json`, environment reports and child logs,
and writes `COMPLETE` only after all seven arms and comparison pass with
unchanged code. Failure leaves records and the allocation intact.
`--resume-root` resumes only a compatible unfinished run and preserves old
logs/session metadata. Small decisive JSON/CSV/provenance evidence must be
copied back to durable project records after completion; model weights,
raw images and mask collections stay outside Git.

Cache contracts are defined by [Cellpose](https://github.com/MouseLand/cellpose/blob/main/cellpose/models.py),
[microSAM](https://github.com/computational-cell-analytics/micro-sam/blob/master/micro_sam/util.py),
and [Keras](https://github.com/keras-team/keras/blob/v2.15.0/keras/utils/data_utils.py).
Allocation and storage requirements follow the GPU guide of the cluster in use.


## CO cohort evaluation and GPU smoke

run_organoid_cohort.py evaluates the real managed Organoid agentic VRR and cohort
without the outer solve planner. It freezes one manifest, requires a clean expected
commit and a Slurm CUDA allocation, pins the CO checkpoint and the existing
high-reasoning model, and forbids CPU fallback. It records global LLM/native/tool
audits, every image's fast/fallback pass, cohort decisions, failure-inclusive mask
scores and actual GPU/package provenance. The scorer opens GT only after prediction;
GT never enters grouping, calibration, tuning or final selection.

Example (inside an allocated GPU job):

    python "case studies/cardiac organoids/runners/segmentation/run_organoid_cohort.py" --manifest /path/pairs150.json --model-path /path/shape_seg_co --expected-commit FULL_SHA --run-dir /scratch/users/your-user/shape_benchmark/results/cardiac_organoids_segmentation/cohort150_RUN --image-workers 4

The default cohort minimum is 10. For a five-image technical smoke, use the frozen
five-image manifest and --cohort-min-group 5 to exercise three calibration images
and two fast-pass candidates when preflight confidently places them in one group.
The 150-image evaluation retains the production minimum 10. Preflight uncertainty
still sends images to individual VRR under either setting. A technical smoke passes
only when all frozen images deliver valid scored masks with the pinned GPU/model;
its metric values are reported, not a claim of superiority.

--cache-seed defaults to 0; --disable-native-cache gives the uncached control.
The cache pins and restores native Python/NumPy/Torch RNG streams. This is a new
recorded runtime policy, not a byte-exact replay of a historical unseeded run.
Native-call counts exclude verified hits; global token/time accounting retains
all preflight, calibration, fast-pass and fallback calls. Per-image seconds sum
its managed passes and exclude shared preflight; concurrent sums are not wall time.
This entrypoint has no implicit resume: interrupted pass records remain evidence,
and an existing output directory is refused. Use a fresh run directory for retries.

## Bounded organoid cohort and resume (schema 2)

`run_organoid_bounded.py` is the safety protocol for larger CO runs. It keeps the
historical runner and exports unchanged. It partitions the frozen manifest into
contiguous fixed shards (64 images by default), with one persistent isolated GPU
worker. Grouping reads are journaled and each managed pass is attested before the
next image runs. A per-pass evaluable prediction is saved immediately; final
routing and records are finalized per shard. One process serves one GPU model;
this safety runner currently serializes image VRR. Parallelize independent fixed
shards across separately allocated GPUs rather than starting multiple model
copies on one device. It is not a demonstrated throughput improvement over the
old four-image concurrent runner.

```bash
python "case studies/cardiac organoids/runners/segmentation/run_organoid_bounded.py" --manifest /path/to/frozen-pairs.json --model-path /path/to/shape_seg_co --expected-commit FULL_COMMIT --run-dir /scratch/path/to/new-run
# After an interrupted allocation, use the identical command plus:
# --resume
# A Slurm array may select --shard-index N with a distinct run-dir for each task.
```

Run only inside an allocated GPU job. Output must be persistent scratch outside
the source/input/checkpoint directories, never only ephemeral node-local storage.
Source tree must be clean at the expected commit. Resume checks complete ordered
manifest identity, actual input hashes/GT geometry, code, checkpoint, packages,
Python, budgets, shard selection, seed and GPU hardware. An OS lock rejects a
second writer and is released if the process dies. Artifact corruption raises;
it is not silently recomputed. Terminal conceded/time-budget/worker-exit passes
are replayed, including failures. A child commits its response and artifact hashes before sending it, so a completed
but unacknowledged pass is recovered without inference. An unconfirmed interrupted
request reserves the remainder of its image budget, keeps any candidate for
review, and is not blindly retried. Known fatal configuration/auth failures stop
execution and may be retried by an explicitly resumed session after correction.
Gateway authentication/model/endpoint errors stop the run; their attempted audit
files remain, and they are not reported as scientific prediction failures.

Default managed VRR budgets are 300 seconds per image across cohort and fallback
passes, six tool requests, and 24,000 reported tokens. CLI flags are
`--max-image-seconds`, `--max-native-calls`, `--max-reported-tokens`, and
`--max-rescaled-pixels` (16,000,000). A reported-token threshold is checked before
each new logical request; it can overshoot by one engine call and is not a
provider-side billing cap. Grouping reads/worker startup are outside the managed
image budget, but all recorded usage appears in global totals. HTTP timeouts
remain enforced by the existing engine. The parent stops a timed-out GPU child
before launching another. A completed best native-QC candidate may be retained
on timeout with `conceded`/`needs_review`; it is never relabeled quality-certified.

`shape_cohort_bounded.csv`, `records.jsonl`, `summary.json`, `sessions.json`,
`identity.json` and per-pass receipts form schema 2. `delivered` indicates that a
mask is available for evaluation; `operational_success`, `qc_pass`,
`negative_confirmed`, `needs_review` and `quality_certified` are distinct. This is
not the old exporter's mutually exclusive success/error CSV schema. GT is read
only by the coordinator's scorer, after inference, and is never sent to a worker,
negative review, calibration or VRR. Empty-label diagnostics remain separately
reported, pending annotation review. A small curated challenge run and an
independent representative pilot are required before any full-dataset launch.

Costs include all audited attempts across resumes. Incomplete final audit lines
from a killed worker remain in place and are counted as unknown; malformed
interior lines stop replay. Started requests without completed audits are exposed
as incomplete tool requests, not invented native calls. Wall time is the sum of
persisted session observations; a killed parent's unmeasured tail is flagged.
Tests cover interruption/replay, corruption, identity changes, one writer,
subprocess timeout/restart, independent negative admission and positive repair.

The original cohort entrypoint accepts `--bounded` to dispatch to this protocol.
For that mode, pass `--image-workers 1`; per-image concurrent GPU models are not
supported. `--cohort-min-group` and `--disable-native-cache` are attested protocol
options. `prepare_bounded_validation.py` materializes explicitly frozen challenge
and random-pilot selections without decoding image data. Its selection records
which source manifest and sampling policy were used. Do not call a curated
challenge representative, or infer training/test independence from the pilot.

Per-image `managed_*` CSV columns contain observed pass usage only. The separate
`budget_consumed_or_reserved_*` columns are scheduling ledger entries, including
conservative reservations after unconfirmed interruption; they must not be
summed as observed tokens or runtime. `unmeasured_interrupted_passes` exposes this
case globally. An attested child completion is recoverable without a new model
call even if the parent died before acknowledging it.

Grouping preflight reads may overlap on up to four allocated CPU workers
(`min(4, SLURM_CPUS_PER_TASK)`), with one process-wide audit context. GPU-model
startup is locked across those threads. Managed image VRR and native inference
remain serialized in the single GPU child. This avoids duplicate GPU models
while retaining concurrent grouping requests. The profile-worker count is part
of resume identity.

Schema-2 CSV columns and JSONL object keys use a fixed sorted order, so a completed
resume preserves export bytes as well as values. CSV fields use the union across
records; absent observed usage stays blank rather than becoming a budget reserve.
The `export_records` helper can validate output serialization against attested
completed records without loading weights or making model calls.

`verify_bounded_exports.py` recovers this specific legacy export-order validation
failure without rerunning completed GPU inference. Run it inside Slurm with
`--run-dir`, the original `--manifest` and validation `--receipt`, a separate
`--output-dir`, `--expected-inference-commit` and `--expected-export-commit`.
It verifies shard payload/artifact hashes, per-image records, CSV values, unchanged
completed-resume usage, and deterministic exports under reordered dictionaries.
It permits only export/validation source changes from the inference commit;
changes to the inference implementation require a new challenge inference.
The recovery receipt preserves separate inference/export commits and never grants
biological-quality approval. Historical failed-job receipts remain evidence.

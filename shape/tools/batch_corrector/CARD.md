---
tool: Batch_Corrector_Tool
category: representation
stage: embedding_postprocess
depends_on: []
tunable: true
metadata_consumes: [batch, group, control]
metadata_none: not routed
metadata_partial: routed; no calibration view (batch labels only, or fewer than 2 replicated non-control groups); the frozen admission decides SKIP or ABSTAIN
metadata_full: routed; calibration view on the declared groups and the control images (2 or more replicated non-control groups); the agent tunes after a RUN admission; delivered only past the guard
---
# Batch Corrector Tool

**Goal:** reduce technical batch variation while preserving biological differences.
This card defines the correction tool, its tuning loop (the budgeted VRR of
`BatchCorrectionVerifier`, run by `StepVerifier`), and the label-dependent metrics in
`quant.py` that the loop scores its calibration rows with and the benchmarks evaluate with.
The loop and the evaluation read disjoint labels: sharing the metric does not give the loop
access to evaluation labels.

| Evidence channel | Inputs | Decision it supports |
|---|---|---|
| Tuning loop (Verify / Recommend) | Embeddings, technical batch metadata, the tool's GT-free readouts and report, and calibration readouts on a caller-declared calibration view (calibration reference groups and control rows) | Admission (RUN / SKIP / ABSTAIN), each round of tool calls, when to stop, and the final choice |
| Benchmark evaluation | The chosen output plus evaluation-only labels (held-out reference groups) | Whether the chosen correction improves on the tool defaults on rows the loop never scored |

## Correction interface

**When to use:** post-process a DINOv3 CLS embedding matrix (or any profile matrix) when a
batch column is available. The tool first measures GT-free batch predictability. It returns
the input unchanged when the batch signal is not detected, and calls Harmony only when the
pre-registered detector crosses its effect margin.

**Correction input:** `cls_output`, a finite `(n_samples, n_features)` matrix; `metadata`,
a table with one row per sample; `batch_key`, the metadata column used for batch detection
and correction; optional unique `row_ids`; optional `harmony_params` (every key in
`DEFAULT_CONFIG`, the levers below). Biological labels are not accepted or consumed by this
tool.

**Correction output:** the corrected matrix (same rows; `input_pca` changes the columns),
status, detector evidence, Harmony parameters when applied, input processing (`preprocess`),
Harmony telemetry (objective traces, number of clusters and their occupancy), `seconds`, the
GT-free readouts (`gt_free_metrics`) and provenance; optionally an output directory with
`corrected_cls_output.npy` and `BATCH_CORRECTOR_RESULT.json`.

## Behavior

1. Validate finite numeric input and, when supplied, row-ID cardinality and uniqueness.
2. Apply the input processing of `harmony_params` (`input_scale`, `input_pca`), then estimate
   balanced-accuracy batch predictability with deterministic stratified folds.
3. Compare predictability against the multi-class chance level. Apply Harmony only when
   the frozen margin is exactly `detect_margin=0.10`; callers cannot override it.
4. Without `harmony_params`, use the sample-aware Harmony library defaults (`theta=2.0`,
   `lambda_value=None`, `sigma=0.1`, `nclust` resolved to a bounded sample-aware cluster
   count, `random_state=0`). With `harmony_params` (the loop passes `DEFAULT_CONFIG` and its
   moves), null keys are not passed to harmonypy, so harmonypy applies its own defaults.
5. Preserve the input unchanged for `SKIPPED_NO_BATCH_EFFECT` and
   `SKIPPED_INSUFFICIENT_BATCHES`.
6. Return the GT-free readouts of every call (definitions in `tool.py`): `batch_mixing_{input,output}`,
   `centroid_ratio_{input,output}`, `within_batch_neighbour_overlap`, `effective_rank_ratio`
   and `displacement`, relative to the input before input processing.

## Budgeted VRR (`BatchCorrectionVerifier`, StepVerifier wave mode)

Rules measure and guard; the agent decides. The tool owns correction, `StepVerifier` owns the
loop, and `BatchCorrectionVerifier`
(`shape/agent/verification/verifiers/batch_correction_unified.py`) measures, guards and asks the
reasoner (the agent) for every decision: which tool calls to run, how many per round, from which
parents, which levers, when it is satisfied, how to read its failed predictions, and the final
call. It subclasses `AgenticVerifier` (`shape/agent/verification/base.py`), the agentic VRR every
agentically tuned tool shares: the shared layer asks the reasoner, validates its moves, runs the
rounds, the plateau stop, failure accounting and the final choice, and writes the prompt
frame; the batch corrector adds its measurements (trial 0, the admission, the noise), harmonypy's
value rules (`coerce`, `check_change`), the guard and the
prompt text. The final choice may keep the uncorrected input (`keep_input`). The admission -- whether
the tool needs tuning at all -- is on trial 0, the input itself (step 1), so the loop makes no call
at the tool's defaults first (the shared layer's `admission_call`, for tools whose input cannot show
whether they need tuning). The verifier sets
`uses_waves`, so `StepVerifier.resolve_stage` runs its rounds. The settings named below are listed
with their values under "Loop settings".

The caller passes a `BatchTask` in the stage context (`upstream_ctx["batch_task"]`): the
per-row batch labels, the tool input, the `CalibrationView` (the rows the loop may score --
calibration reference groups and control rows -- and their metadata; never the evaluation
rows' labels), the task's stated noise, a one-line brief and the cores for the tool calls.
In `solve()` the system-routed batch step builds it from the metadata profile
(`shape/agent/verification/batch_stage.py::batch_task_for`): with a calibration view in the full
tier, without one in the partial tier. Without a calibration view separability cannot be verified:
the gate decides `SKIP`, otherwise `ABSTAIN`.

0. **Trial 0 is the uncorrected input** (`uncorrected_result`), a measurement and not a tool
   call: the tool's gate on all rows and its GT-free readouts of the input against itself. Every
   trial is a `VLRRResult`: `scalar_score` = the calibration `overall`, `metrics` = the
   calibration readouts (`quant.composite_all` on the calibration view, at float32) and the
   GT-free readouts, `param_delta` / `expected_effect` / `notes` = the move that produced it.
1. **Admission** (on trial 0). Evidence: the gate on all rows and on the negative-control rows
   only (a batch signal on controls is technical by construction); how many calibration
   reference groups (e.g. compounds) occur in two or more batch groups and in all of them;
   controls per batch group. Rule: no detected batch effect -> `SKIP`; no reference group
   shared across batch groups -> `ABSTAIN`; otherwise `RUN`. The reasoner may turn `RUN` into
   `ABSTAIN` only by citing a soft criterion that holds (`admission_concerns`); a veto that
   cites none is rejected. Soft criteria: fewer than max(2, min(`ADMISSION_MIN_SHARED_GROUPS`,
   the number of distinct non-control reference groups)) reference groups occur in every batch
   group (`admission_min_shared_groups`, adaptive to the design size: a JUMP-scale design keeps
   `ADMISSION_MIN_SHARED_GROUPS`, a 3-condition design needs its 3; an absolute count, so a
   design whose other groups are batch-specific stays verifiable through the shared ones); a
   batch group has no negative controls; the gate finds no batch effect on the negative
   controls alone. Better cross-batch replicate retrieval after correction is evidence for
   separability, not against it. A caller may pass a precomputed or forced admission. `SKIP`
   and `ABSTAIN` end the stage without a tool call.
2. **Noise** (measured on trial 0 when admission is `RUN`). The calibration view of the input is
   scored `NOISE_REPEATS` times; the measured noise of a readout is 2SE of the difference of two
   scorings of the same representation (2 * sqrt(2) * sd; the kNN graph and the Leiden clustering
   of the metric vary from run to run). The noise used for the facts and the guard is the larger
   of the measured and the task's stated noise (`NOISE_BAND` when none is stated). Both are shown
   to the reasoner.
3. **Rounds** (one VRR cycle each). Recommend: the reasoner returns the calls of the round -- at
   least one, at most the concurrency cap -- each a parent (`input` = the
   tool's `DEFAULT_CONFIG`, or any finished call), the levers it changes relative to the parent's
   configuration (one or several; none from `input` = the tool at its defaults), a diagnosis
   naming its evidence and the readouts expected to move up, down or stay (`preserve`); with every
   plan it returns its hypothesis list (id, statement, status open / supported / refuted /
   abandoned, evidence), shown back to it next round. The narrowing (`narrowing`, the shared layer's,
   as the segmenter's): broad, then narrow, at the reasoner's pace -- it explores first, several levers in
   one call from any finished call, and narrows by naming `focus_levers` in a plan, whose moves then change
   only those levers. Two rules (`_schedule_problem`): a first focus needs a lever of every CARD Family
   (harmony, input, solver; `seed` exempt, `coverage_exempt`) changed by a call first; once a round has
   run under a focus, every later plan names focus_levers again, the same levers or fewer
   (`focus_history`).
   StepVerifier runs the round concurrently and
   verifies every result. Reconcile (facts shown in the next prompt): per call, the key readouts
   relative to its parent and to the uncorrected input, each marked beyond or within the noise;
   each stated expectation next to the observed change; the tool report (Harmony clusters and
   their occupancy, Harmony rounds, PCA components, seconds); whether the call is an admissible
   final choice now (step 5); Pareto membership on `OBJECTIVES`; the calls that failed; the levers
   no call has changed yet; the effective values of the null / off levers at the defaults; the
   rounds done and left, the cores and the cap, per-call seconds, the
   wall time used and, when the caller knows it (`BatchTask.deadline`, e.g. the Slurm job's
   limit), the time left. Whether a
   prediction failed, and what follows from it, is the reasoner's judgement. The linear VRR of
   the other tools is the case of one call per round.
4. **Stops**, whichever comes first: the reasoner returns no moves (satisfied), at any round -- with
   rounds left, it is asked once more (`confirm_stop`, `stop_check_text`: the rounds that remain -- the
   same result in fewer rounds costs less --, the lever groups no call has changed, the best admissible
   call so far) and may continue or confirm the stop (`stop_checks`); the
   backstops: `PATIENCE` rounds in a row -- the rounds after the last one that raised the best
   calibration `overall` (input and finished calls) beyond the noise -- without a
   rise (plateau: early stopping, as SSL training's); the budget -- at most `BUDGET`
   rounds (StepVerifier `max_iters` = 1 + budget, the linear loop's unit: an iteration is a round,
   trial 0 is not one; the managed stage uses the verifier's `budget`), so a run makes at most
   `BUDGET` x cap calls; no valid move remains. A failed call counts in its round and is never
   re-run, never a parent and never chosen.
5. **Final choice** by the reasoner among the admissible calls. The guard: a finished call whose
   calibration `overall` beats the uncorrected input's by more than the noise and whose `PRESERVE`
   readouts are not below the input's by more than the noise. An inadmissible choice is answered
   with the conditions it fails and asked once more; still inadmissible, a choice of none, or no
   admissible call -> no correction (`result` None, `conceded`, fault `ceiling`: keep the
   uncorrected input). A stopping plan (no moves) may name the call to deliver (`deliver`; after a
   stop check, the confirming answer's): an admissible one is delivered without the final question
   (`agent_choice`); otherwise, or with none named, the final question runs as above. The outcome is
   a `StageResolution`: `satisfied` with the chosen trial; `conceded` with no output when admission
   abstains or no correction is chosen; `satisfied` with trial 0 (the input) when the gate skips.
   `StageResolution.parents` / `details` record the tree of trials, the ledger entries, the
   admission, the noise, the rounds, the reasoner's plans and hypotheses, the failed calls, the
   reasoner failures and fallback rounds, and the choice.

Validation (deterministic): CARD lever names and values within `PARAM_BOUNDS` and the tool's
limits (`nclust` below the rows, `input_pca` below the features), no repeated configuration
(failed ones included; a value that acts as the parent's -- e.g. `nclust` 100 where null resolves
to 100 -- is the same configuration), the round cap. A value is a JSON literal, a plain number
(`.07`) or null; anything else is rejected with the text received. Rejected moves get one repair
question with the reasons, naming the accepted moves that already run, while the round has room.
A reasoner question that fails (a gateway error or a malformed answer) is tried
`REASONER_ATTEMPTS` times, `REASONER_RETRY_SECONDS` apart, before it counts as failed. Concurrency: at most `cores // threads` calls at the same
time, each with `threads` = ceil(rows / `ROWS_PER_THREAD`) BLAS threads within [1, `THREADS_PER_CALL`] (at most the
cores; the caller sets this thread limit, `blas_limits`).
Scoring stays on the loop's thread: each result is scored as soon as its call finishes, while the
round's other calls still run. Scoring several results at once in threads of one process was tried
and deadlocked intermittently on the cluster (scenario_1, 2026-09-29, cause not isolated; the process
holds libgomp and four OpenBLAS copies, among them harmonypy's bundled 0.3.3), so it is not used.

Failed reasoner questions never generate deterministic tuning calls. Exhausted nonfatal failures
are recorded in `execution_errors`; a failed admission leaves the input unsearched, a failed round
ends the search, and a failed final question retains the input. Missing required engines and fatal
authentication, permission, model or connection errors propagate. `fallback_rounds` stays zero.

Information level: the loop reads calibration readouts on the caller's calibration view, the
GT-free readouts and the tool report. The evaluation rows' labels are never passed to it:
development uses the calibration reference; the untouched evaluation split judges the result.
Unlike a free coding agent, the loop cannot run its own analyses of the data beyond these
readouts.

## Levers

Every parameter the tool accepts (`Batch_Corrector_Tool.execute(harmony_params=...)`) is a
lever below. Each has one line `Family | Range | Default`:

- Family: `harmony` (the Harmony objective), `input` (processing before the detector and
  Harmony), `solver` (harmonypy's convergence settings), `seed` (initialisation only).
  Seed-only variation is not broad parameter evidence.
- Ranges are checked against `PARAM_BOUNDS` (`shape/agent/verification/base.py`);
  the reasoner sets values within the range, one or several levers per call.
- `null` means the harmonypy default (the argument is not passed); the loop shows the
  reasoner the effective value of every null / off lever at the defaults.

Not exposed by the tool: harmonypy's `alpha`, `batch_prop_cutoff`, `ncores`, `verbose`.

### theta

Family: harmony | Range: [0, 100] | Default: 2

Diversity penalty of the Harmony objective. Higher values push each cluster towards
batch-balanced membership: stronger mixing, larger displacement, and a growing risk of
removing biological variation that is unevenly distributed over batches.

### sigma

Family: harmony | Range: [0.0001, 10] | Default: 0.1

Width of the soft k-means kernel. Lower values give harder cluster assignments (each row is
corrected mostly by one cluster); higher values give softer, more global assignments.

### lambda_value

Family: harmony | Range: [1e-06, 1000] | Default: null

Ridge penalty of the per-cluster batch regression. `null` (the default; `-1` is the same in
harmonypy 2.0) = harmonypy's dynamic per-cluster estimate (its `alpha`, 0.2, is not exposed), so
the defaults have no single effective lambda; a positive value
fixes it. Higher fixed values shrink the batch offsets (more conservative correction); lower
values allow larger offsets.

### nclust

Family: harmony | Range: [2, 10000] | Default: null

Number of Harmony clusters (also below the number of rows). `null` = harmonypy's rule
`min(round(n/30), 100)`, which does not grow beyond 100 however many rows there are. More
clusters make the correction more local (each cluster gets its own batch regression); too
many leave clusters dominated by one batch, fragment small batches, and cost roughly linear
time.

### input_pca

Family: input | Range: [2, 4096] | Default: 0

Principal components computed before the tool (`0` = off: the raw features). The detector
and Harmony then work on the component scores. Fewer components remove low-variance
directions (noise, but possibly biology); more keep more of the input. From `0` the first
step starts at the number of input features.

### input_scale

Family: input | Range: none, zscore, l2 | Default: none

Per-feature z-score (`zscore`) or per-row L2 normalisation (`l2`) before the tool. It
changes which features dominate the distances.

### max_iter_harmony

Family: solver | Range: [1, 200] | Default: null

Outer Harmony iterations. Harmony stops earlier when the objective changes by less than
`epsilon_harmony`, so a larger cap only matters when convergence is not reached.

### epsilon_harmony

Family: solver | Range: [1e-10, 1] | Default: null

Convergence tolerance of the outer loop. Smaller values run more iterations (a more fully
converged, usually stronger correction).

### max_iter_kmeans

Family: solver | Range: [1, 200] | Default: null

k-means iterations per Harmony round.

### epsilon_cluster

Family: solver | Range: [1e-10, 1] | Default: null

Convergence tolerance of the per-round k-means.

### block_size

Family: solver | Range: [0.001, 1] | Default: null

Fraction of rows updated per k-means block.

### tau

Family: solver | Range: [0, 10000] | Default: null

Protection of small batches in the diversity penalty (discounts the penalty for batches
with few rows in a cluster).

### random_state

Family: seed | Range: [0, 2147483647] | Default: 0

k-means initialisation seed. A different seed changes the result only by run-to-run
variation, so a seed change is a measurement of the tool's own run-to-run noise, not an
improvement.

## Loop settings

Every setting of the batch-correction loop, with its value and where it lives: the shared agentic
VRR's in `base.py` (`AgenticVerifier`; the same for every agentically tuned tool), the batch
corrector's own in `verifiers/batch_correction_unified.py`, the tool's in `tool.py`. The
consistency test (`test_batch_budgeted_loop.py`) checks each value and its place against the code.

| setting | value | where | meaning |
|---|---|---|---|
| `BUDGET` | 8 | `base.py` | rounds at most, a limit and not a target (each round: the calls the reasoner runs at once, at most the concurrency cap); trial 0 (the uncorrected input) is not a round (callers may pass another budget; the managed stage uses the verifier's `budget`, i.e. this one) |
| `PATIENCE` | 3 | same | rounds in a row without a rise of the best `overall` beyond the noise -> plateau stop, a backstop (the verifier's `patience`; `adaptive_training.early_stopping_patience(BUDGET)`) |
| `REASONER_ATTEMPTS` | 3 | same | tries of one reasoner question before it counts as failed |
| `REASONER_RETRY_SECONDS` | 30 | same | wait between two tries |
| `_LEDGER_REL_FLOOR` | 0.02 | same | `classify_lever` relative floor of StepVerifier's ledger, kept for the GT-free readouts |
| `NOISE_BAND` | 0.002 | `verifiers/batch_correction_unified.py` | the stated noise when the caller states none (the caller passes the task's) |
| `NOISE_REPEATS` | 3 | same | scorings of the input's calibration view that measure the metric's noise |
| `THREADS_PER_CALL` | 8 | same | BLAS threads per tool call at most (the largest inputs); concurrency cap = cores // the resolved threads (the verifier's `threads_per_call`), at least 1 |
| `ROWS_PER_THREAD` | 25000 | same | one BLAS thread per this many input rows: threads = ceil(rows / this) within [1, `THREADS_PER_CALL`], at most the cores |
| `ADMISSION_MIN_SHARED_GROUPS` | 10 | same | soft admission criterion's cap: reference groups present in every batch group, at least max(2, min(10, the distinct non-control reference groups)) |
| `SCORE` | overall | same | the calibration score of an attempt (`scalar_score`; the verifier's `score_key`) |
| `OBJECTIVES` | overall, nonrep_mean_map | same | Pareto membership shown to the reasoner (the verifier's `objectives`) |
| `PRESERVE` | bio_mean, nonrep_mean_map | same | readouts the final choice may not lower below the input's by more than the noise |
| `READOUT_ROWS` | 3000 | `tools/batch_corrector/tool.py` | rows subsampled (seed 0) for the GT-free readouts |
| `READOUT_NEIGHBOURS` | 15 | same | k of the GT-free readouts |
| `DETECT_MARGIN` | 0.10 | same | batch-effect gate over chance (frozen) |

## Calibration and evaluation metrics (`quant.py`)

`quant.py` is the one label-dependent metric module of the batch corrector: the loop scores its
calibration view with it and the JUMP benchmark (`benchmarks/batch_correction_JUMP/`) evaluates
with it. It is not a GT-free readout, and the correction tool never reads labels.

```python
from shape.tools.batch_corrector import quant

scores = quant.composite_all(metadata, profiles, label_key="Metadata_JCP2022",
                             batch_key="Metadata_Source", plate_types=["TARGET2"])
print(scores["batch_mean"], scores["bio_mean"], scores["overall"], scores["skipped"])

# views of a prepared table with a `split` column (val | test | negcon | poscon)
test = quant.score_split(metadata, profiles, "test", batch_key="Metadata_Source", plate_types=["TARGET2"])
halves = quant.plate_halves(metadata.loc[quant.split_mask(metadata, "test")], 20, batch_key="Metadata_Source")
```

Install the metric backends with `pip install -e '.[batch-metrics]'` (`scib-metrics`,
`copairs`). The definitions follow Arevalo et al. (2024) and the authors' `metrics/scib.py` and
`metrics/map.py`, with their choices.

| Output | Group | Definition (higher is better) |
|---|---|---|
| silhouette_batch | batch | scib-metrics batch silhouette within each label (1 - absolute cosine silhouette of the batches), label mean |
| lisi_batch | batch | scaled iLISI (batch diversity) on a k = 25 neighbour graph |
| graph_conn | batch | graph connectivity: per label, the largest connected component of the label's k = 25 neighbour subgraph / label size, label mean |
| lisi_label | bio | scaled cLISI (label purity) on the same graph |
| ari | bio | adjusted Rand index between the labels and one Leiden clustering at resolution 1.0 |
| nmi | bio | normalised mutual information between the same two partitions |
| asw | bio | (cosine label silhouette + 1) / 2 |
| negcon_mean_map | bio | copairs mean average precision of each compound's replicates (other wells) against same-plate DMSO wells (every DMSO well its own pseudo-label), threshold 0.05, null size 10000; mean over compounds |
| nonrep_mean_map | bio | the same retrieval against the other compounds on the plate (DMSO ignored) |

Rows: DMSO is dropped before the scib metrics and kept as the mAP-negcon reference; the
silhouettes subsample 20000 rows (seed 0, the same rows for every representation of the same
size). mAP rows: the plate types of the scenario, positive controls and singleton compounds
dropped, and the over-replicated compound JCP2022_033954 dropped (the authors' filter).

Composite: `batch_mean` = mean of the available batch metrics, `bio_mean` = mean of the
available bio metrics, `overall = 0.4 * batch_mean + 0.6 * bio_mean`, the weights renormalised
over the groups that have a metric. A metric whose precondition fails on the scored rows (fewer
than two batches; fewer than two labels; no DMSO; fewer than two replicated compounds; no plate
or perturbation-type columns) is left out and listed in `skipped` with the reason;
nothing is imputed. kBET is omitted (no maintained Python implementation).

Views and uncertainty: a split's rows are its reference groups plus every control row
(`split_mask`); the uncertainty of a paired comparison comes from fixed random halves of the
plates drawn within each batch group (`plate_halves`), with the 2SE band of a full-view delta
approximated as `2 * sd(half deltas) / sqrt(2)` (`paired_band`).

Evaluation protocol: declare the calibration and evaluation reference groups, the rows of each
view and the metric settings before comparing, and keep them identical for the uncorrected
input, the defaults and the chosen output. Freeze the loop's choice, then score it on the
evaluation view. Report `overall` for the prespecified comparison, but show batch removal and
biology separately: better mixing with worse biology is a tradeoff, not an unqualified success.
A disappointing evaluation result can motivate a new development hypothesis, not another move
of the same loop; once evaluation labels are used for development, that split is no longer
untouched evidence.

## LLM gateway configuration

Agentic tuning requires these three environment variables:

LLM_MODEL="gpt-4o"
LLM_API_KEY="<your-llm-api-key>"
LLM_BASE_URL="https://llm-gateway.example.edu/v1"

SHAPE does not read STANFORD_AI_API_ALIAS or STANFORD_AI_USAGE_URL; they are not part of the runtime contract.

## Metric references

- Arevalo et al. (2024), *Evaluating batch correction methods for image-based
  cell profiling*, Nature Communications 15:6516, Methods p. 9,
  [DOI](https://doi.org/10.1038/s41467-024-50613-5).
- [Authors' implementation](https://github.com/carpenter-singh-lab/2023_Arevalo_BatchCorrection)
  at commit cd9bcf99, `metrics/scib.py` and `metrics/map.py` (its preprocessing is vendored in
  `benchmarks/batch_correction_JUMP/runners/vendor/`).
- [scib-metrics](https://github.com/YosefLab/scib-metrics) and
  [copairs](https://github.com/cytomining/copairs).

The PDF prints the recall increment in reverse order in its AP equation.
Positive AP and the authors' implementation use increasing recall:
`AP = sum(precision@k for positive ranks k) / number_of_positives`.

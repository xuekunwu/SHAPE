# batch_correction_JUMP

Batch correction of JUMP Cell Painting well profiles on the scenarios of Arevalo et al. (2024,
*Nature Communications* 15:6516). Six representations of the same scenario -- the uncorrected input and five
comparison arms -- are scored on held-out compounds with the authors' metrics.

## Layout

```
run_batch_benchmark.py     the entry point: prepare, control, suite, report
export_run_records.py      the record set of a finished run, for results/<run_id>/
runners/                   the implementation (contract, config, prepare, control, the arms, score, suite),
                           the tests and the vendored upstream preprocessing
```

## Arms

`raw` is the uncorrected reference. The five comparison arms (`runners/benchmark_contract.py`):

1. `harmony`: Harmony at harmonypy's library defaults (adaptive `nclust = min(round(n/30), 100)`, dynamic lambda),
   one call. This is not the configuration of the authors' pipeline (`nclust = 300`, `lamb = 1`).
2. `scanorama`: Scanorama without PCA, one call.
3. `gpt_luna`: gpt-5.6-luna without a domain tool. From a matrix summary (profile and feature counts, profiles
   per batch, the batch-centroid share of the variance) it chooses one generic linear operation: `identity`, or
   `batch_center` (subtract `strength` times each batch's centroid offset). Its prompt and the summary are kept in
   `ARM_OUTPUT.json`.
4. `react_luna_harmony`: the shared ReAct arm (`benchmarks/_shared/react_arm.py`, `REACT_ARM_CONTRACT.md`):
   gpt-5.6-luna plus `Batch_Corrector_Tool`, called as the SHAPE arm calls it (every well, the scenario's batch
   key). One session for the dataset. The prompt holds the SHAPE arm's task brief, the task's stated noise, the
   tool's parameters as SHAPE's verifier sets them (the CARD's 13 levers, `input_scale` and `input_pca` among them:
   name, type -- a choice with its choices -- and default only), the runner's one factual line, the budget and the
   time left. Each observation is the tool's own returned dict (arrays dropped) plus the calibration scores
   SHAPE's verifier gets for the call. Between rounds the agent may run its own Python (the runner's `python`
   action, not a round) on the uncorrected features, the label-free well metadata (source, batch, plate, well,
   plate type) and each call's corrected matrix (`BatchToolAdapter.python_inputs`; no compound or split column).
   Trial 0 (`default`) is the uncorrected input, not a tool call and not a round (as SHAPE's `admission_on_input`);
   then at most `BUDGET` (8) rounds of up to SHAPE's per-round cap calls: compared with an existing run
   (`react --compare-run`), the `concurrency_cap` that run's SHAPE arm recorded (refused when absent); otherwise
   `concurrency_cap(cores, features)` (`round_cap_source` in `ARM_OUTPUT.json` records which).
   The agent delivers a call with `finish`; without a valid finish the uncorrected input is delivered
   (`default_session_unfinished`). The harness never chooses by score (`runners/llm_arms.py`, `BatchToolAdapter`).
5. `shape_luna_harmony`: SHAPE's `BatchCorrectionVerifier` through `StepVerifier` wave mode on
   `Batch_Corrector_Tool` (`shape/tools/batch_corrector/CARD.md`, "Budgeted VRR"): the admission on the input,
   at most 8 rounds of up to `cores // 8` concurrent calls, broad then narrow at the agent's pace (it explores,
   several levers in one call if it likes, and narrows by naming `focus_levers` once a lever of every group --
   harmony, input, solver -- has been changed; a focus is never widened), a stop with rounds left asked once more
   with the facts, the plateau backstop counted in rounds (3 without progress), the guard on the final choice.

Every Luna call has the same output-token limit (`LUNA_MAX_TOKENS`, reasoning tokens included).

## Information boundary

Treatment compounds are split deterministically into calibration (`val`) and evaluation (`test`) halves
(`sha256(f"{seed}:{id}")` order). SHAPE scores its candidates on the calibration compounds and the controls; ReAct
gets the same wells with their compound labels as a python input (`inputs/calibration_labels.csv`) and computes its
own signal, never SHAPE's scores (2026-10-12). Neither sees an evaluation-compound label. GPT-only sees no biological label and no metric. The final
comparison uses the evaluation compounds and the controls, with a 2SE band from 20 fixed plate halves.

## Metrics

- `batch_mean`: silhouette batch, batch LISI, graph connectivity. `bio_mean`: label LISI, ARI, NMI, ASW,
  negative-control mAP, non-replicate mAP. `overall = 0.4 * batch_mean + 0.6 * bio_mean`
  (`shape/tools/batch_corrector/quant.py`, the authors' `metrics/scib.py` and `metrics/map.py`, kBET omitted).
- Runtime per profile: the arm's own wall time (LLM calls, tool calls, its calibration scoring; the harness's final
  scoring excluded) over the well profiles; for Harmony and Scanorama the one correction call.
- Tool calls and tokens per profile: domain-tool calls and recorded Luna tokens over the well profiles.
- `ap50_gain_per_tool_call` is `null` (no AP50 for embeddings); `nonrep_map_gain_per_tool_call` replaces it.

## Run

```bash
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py prepare \
  --base JUMP --output PREP --scenario scenario_4 \
  --microscope-config microscope_config.csv --arevalo-cache CACHE
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py control --prep PREP
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py suite \
  --prep PREP --output RUN --model gpt-5.6-luna --max-rounds 8
```

The ReAct arm alone on a finished run's prep (control included), scored with the same scorer and plate halves
into `<run>/react_<id>/REACT_REPORT.json` (one report row in the suite's form; whether the control's tool and
metric code matches the current code is recorded, not refused), at the per-round cap of that run's SHAPE arm:

```bash
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py react \
  --prep <run>/prep --output <run>/react_<id> --compare-run <run> --cores 16
```

`suite` and `react` need `LLM_API_KEY` and `LLM_BASE_URL` and a fresh output directory; `suite` writes `BENCHMARK_REPORT.json /
.csv / .md`, each arm's output and the SHAPE ledger. CPU only; install with `pip install -e '.[batch-metrics]'`
(Scanorama, scib-metrics, copairs).

A run writes under `$SHAPE_RESULTS_ROOT/batch_correction_JUMP/`: the preprocessing cache in `cache/arevalo/`
and one run directory `scenario<N>_sixarm_<commit8>_<UTC>/` (`RUN_META.json`, `prep/`, `suite/`, `COMPLETE`
after the validation in `runners/validate_suite.py`, then the record set `records/`). `prepare` stops with
`MISSING_INPUTS` when a source of the scenario has no profiles under `inputs/<source>/workspace/profiles`.

### Reasoning-effort sweep

The SHAPE arm alone at each reasoning-effort level the model accepts through the gateway (`none`, `low`,
`medium`, `high`, `xhigh`; `runners/effort_sweep.py`): one run per level, scored by the same scorer against the
same control stage, nothing else changed between the runs. The level reaches the engine through
`SHAPE_REASONING_EFFORT` (read on every request); every level gets the same per-reply token cap (`--max-tokens`,
default 32000, reasoning included: the engine's own default of 4000 cuts high-effort replies) and request timeout
(`--timeout`, default 900 s). Each level's run dir `shape_<effort>/` adds `EFFORT.json` and
`logs/reasoner_calls.jsonl` (each reply's usage, reasoning tokens included, and any error reply); a level with a
`RESULT.json` is kept on a rerun. `SWEEP_REPORT.json / .csv / .md`: per level the test-view scores, the deltas
against Harmony and raw (+- 2SE, plate halves), rounds, calls, stop reason, tokens, reasoning tokens per reply and
wall time.

```bash
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py effort-sweep \
  --prep PREP --output OUT [--efforts none low medium high xhigh] [--max-tokens 32000] [--timeout 900] --cores 16
```

## Records and traces

```
<run>/RUN_META.json, COMPLETE        branch, commit, job, host, start, scenario, model; the validation's marker
<run>/prep/PREPARE.json              the scenario's wells, plates, sources, features and the val / test split
<run>/control/CONTROL.json           raw, harmony, scanorama: configuration, tool record, scores per view, halves
<run>/gpt_luna/ARM_OUTPUT.json       its prompt, the matrix summary, its reply and the operation
<run>/react_luna_harmony/            ARM_OUTPUT.json (every call's round, reasoning, configuration, calibration
                                     scores; the delivered call and its source), logs/tool_calls.jsonl,
                                     react/<scenario>/TRACE.jsonl, CALLS.json, SELECTED.json, react/REACT_SUMMARY.json
<run>/shape_luna_harmony/            RESULT.json / .md (scores, deltas, claims), NOTES.md, AGENT_RUN.json,
                                     RUN_META.json, submission/decision.json, logs/tool_calls.jsonl (every call
                                     with its parent, rationale and scores), traces/round_00.jsonl (every prompt
                                     and reply); the corrected matrices stay on scratch
<run>/compare/                       compare.json, readouts.csv (the suite's report), paired_vs_shape.csv,
                                     summary.md
<run>/audit/<arm>.json               the arm's audit record (export_run_records.py), + index.json
<run>/logs/                          the job's output
<run>/MANIFEST.json                  every file with its size and sha256, the expected files missing
```

`export_run_records.py --run-dir <run> --log <log> --out <dir>` writes this record set from a finished run
directory (into `<run>/records/`): it
copies the files above byte for byte (the suite's `BENCHMARK_REPORT.json` / `.csv` as `compare/compare.json` /
`readouts.csv`; `BENCHMARK_REPORT.md` is replaced by `compare/summary.md`), and builds `paired_vs_shape.csv`
(SHAPE minus each control arm per metric: the test-view delta, its 2SE band from the plate halves, beyond or
within it, the all-wells delta; from SHAPE's `RESULT.json`), `summary.md` and the audit records. The reported
runs' record sets are `results/<run_id>/`, kept byte for byte (`.gitattributes`). Nothing is copied to the
control plane: the tarball stays with the run on scratch.

**Audit records** (schema `batch-jump-run-record/1`): one JSON per arm, built only from the run's own files. Each
holds the input (scenario, sources, wells, plates, the split; the evaluation compounds never shown), the
provenance (the run's meta, package and tool versions; SHAPE's own run meta), the arm's configuration, every step
in order (the tool call and its configuration, its result and the rationale the arm stated for it: GPT-only's one
reply; ReAct's per-step reasoning and calibration scores; SHAPE's admission, each round's analysis, hypotheses,
`focus_levers` and moves, every call with its parent, rationale and calibration scores, the stop check, the
plateau and the final choice), the final output, the score per view and the cost, the status (stop reasons,
reasoner failures, fallback rounds, stop checks, focus history) and the source files used or missing.

## Data preparation

`prepare` runs the authors' `mad_int_featselect` chain (vendored, below) and writes `features.npy`,
`meta.parquet` and `PREPARE.json`. A cached stage is reused only when its marker holds the current fingerprint
(vendored code, scenario conf, microscope table, numpy / pandas / pyarrow / scipy / scikit-learn versions, raw
inputs path); otherwise it and every later stage are recomputed. The pinned microscope table is
`jump-cellpainting/datasets@181fa0dc96b0d68511b437cf75a712ec782576aa` (SHA-256
`f67c22d0bb891a3976722c801fd93b7dc07c6fddda5a4c64bd0aef20769235d4`); `prepare` serves this staged copy instead of
the download the vendored `preprocessing.metadata` performs at import time.

## Code provenance

`runners/vendor/2023_Arevalo_NatComm_BatchCorrection/` holds the parts of the authors' pipeline this benchmark runs,
unchanged: repository https://github.com/carpenter-singh-lab/2023_Arevalo_NatComm_BatchCorrection, commit
`cd9bcf99240880a5c9f9858debf70e94f5b4c0f7`, BSD 3-Clause (`LICENSE` there); paper doi:10.1038/s41467-024-50613-5.
Kept: `preprocessing/` (the `mad_int_featselect` chain of `rules/processing.smk`), `pycytominer/operations/` (the
feature-selection operations it imports) and `inputs/conf/scenario_{1..5}.json`; the rest of the upstream tree is
not used. The metric definitions of the authors' `metrics/scib.py` and `metrics/map.py` are re-hosted in
`shape/tools/batch_corrector/quant.py` (scib-metrics and copairs backends; kBET omitted). Earlier experiments in
this folder (tasks 079, 092, 113; the trace / efficiency exporters and single-arm CLI commands) were removed; git
history keeps them.

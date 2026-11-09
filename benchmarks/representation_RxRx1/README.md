# representation_RxRx1

Single-cell representation learning on RxRx1 HEPG2 (Cell Painting-style 6-channel crops, 30 siRNA positive controls).
Seven arms each deliver one embedding of every cell: SHAPE tuning the self-supervised fine-tune of
`Cell_State_Analyzer_Tool` (DINOv3 ViT-B/16 with a learned 6 -> 3 channel adapter, 224 px, CLS) through its agentic
VRR, the same tool at its defaults or frozen, an LLM with the tool (ReAct), an LLM writing its own features, and two
external feature sets. The embeddings are scored by well mAP (same-siRNA retrieval of well profiles) and a siRNA
probe across experiments; no label enters any arm's decision. The contract and its amendments (A1-A23):
`projects/SHAPE/tasks/116_ssl-vlrr/SHAPE_SSL_VRR_CONTRACT.md`.

## Data

The task-116 v3 preps of one cell type (`config.py` `PREPS`; HEPG2 by default, a cell type task 116 never tuned on),
made with the task-116 launcher:

- FULL `prep_hepg2_e11_v3`: 11 experiments, the positive-control wells only, 30 siRNA, 61,804 cells; a label-free
  split of the wells into train, val (a share of every plate's wells) and held-out (2 whole experiments, HEPG2-01 and
  HEPG2-03); `images.npy` (uint8, N x 6 x 64 x 64, one crop per cell centred on a nucleus, per-site 0.1 / 99.9
  percentile stretch);
- run prep `prep_hepg2_e11_v3_t10`: all cells of 10 % of every plate's train wells plus all val cells (train 5,074 /
  val 5,070 cells, 36 plates). The tool sees the run prep only.

`prepare.py` writes no crop file: `prep/cells.csv` (`cell_id`, `crop_id`, `image_name` = site, `group`, `plate`,
`experiment`, plate-qualified `well`, `site`, `split` in train | val | unused | heldout, and `array_row`, the cell's
row of the FULL `images.npy`) and `prep/PREPARE.json` (with `crop_array`: the path, sha256, size, dtype and shape of
`images.npy`) under `<output root>/<cell type>/`. The siRNA labels go to `private/labels.csv`, read only by `score.py`
(and the offline `summarize_results.py`). Every analyzer call gets `images.npy` as its packed-array input
(`crop_array`, amendment A1) with an integer `array_row` per metadata row; the analyzer reads the rows it needs into
memory once and fingerprints an embedding by the content of those rows, so every stage gets the same fingerprints on
any machine. Every stage refuses an `images.npy` whose size or sha256 differs from PREPARE.json (`prepare.load`):
the full sha256 runs once per file state (`prepare.verify_crop_array`; `prep/CROP_ARRAY_VERIFIED.json` holds the path,
the sha256, size, mtime, ctime and inode after a match; any other state is hashed again).

Scoring groups: `seen` = the train, val and unused wells of the 9 seen experiments (50,558 cells; the primary
endpoint), `heldout` = the 2 held-out experiments (11,246 cells), `run_prep` = train + val (the cells the tool saw);
audit groups `trained_wells`, `unused_wells`, `val_wells`.

### Full-set preps

The full-set v4 preps of HEPG2, HUVEC, RPE and U2OS (`config.py` `FULL_V4_PREPS`, `BenchmarkConfig(prep_set="full_v4")`;
the default stays the v3 preps; the outputs of a full_v4 configuration go to `<output root>/<cell type>_full_v4/`) are
built from this repository by `build_prep.py`, a port of the task-116 prep builder that writes
the v3 format (`images.npy` uint8 N x 6 x 64 x 64, `cells.csv`, `PREPARE.json`); with the v3 options it gives the v3
crops and split. Per cell type, under `/scratch/groups/your-group/your-user/shape_benchmark/RxRx/agentic_ssl`:

- FULL `prep_<cell type>_full_v4`: every site of every experiment of the train and test datasets, every well type
  (`--well-types all`); `min(n - 1, round(0.8 n))` of the n experiments held out (seeded sha256 order); val = per plate
  of the seen experiments a seeded share of the wells, the share chosen for about 200 val wells per cell type (HEPG2
  and RPE 0.08, U2OS 0.16, HUVEC 0.035); train = the rest;
- run prep `prep_<cell type>_full_v4_t5k`: all cells of the smallest seeded share of every plate's train wells that
  holds at least 5,000 training cells (`subsample --train-cells 5000`), plus every val cell.

The steps are a timing probe on the first 200 sites of the first cell type, then `prepare --workers` and
`subsample` per cell type; an existing output is never overwritten:

```bash
python -m benchmarks.representation_RxRx1.runners.build_prep prepare|subsample|resplit --help
```

#### Full-set v5: a well-level split

The full-set v5 preps (`config.py` `FULL_V5_PREPS`, `BenchmarkConfig(prep_set="full_v5")`, run label
`<cell type>_full_v5`) hold the cells and crops of the v4 preps with a WELL-level split instead of v4's
experiment-level one (user decision). They are made from the v4 preps by `resplit --split wells`, which never re-reads
an image: `images.npy` is hard-linked to the v4 file (a copy where linking fails; PREPARE.json `resplit.images` and
the log say which). The link changes the shared inode's ctime, so the v4 cache's `CROP_ARRAY_VERIFIED.json` no longer
matches and its next load hashes `images.npy` once more.

- FULL `prep_<cell type>_full_v5`, the split (`build_prep.WELL_SPLIT_RULE`; label-free, every experiment and every
  well type, the EMPTY negative controls included): per plate, its n wells in a seeded sha256 order;
  `n_seen = floor(0.10 n + 1/2)` and `n_val = max(2, floor(0.20 n_seen + 1/2))` (round half up, exact on the decimal
  option values); the first `n_val` wells of the order are val, the next `n_seen - n_val` train, the other
  `n - n_seen` (90 %) heldout. A plate of 308 wells: 31 seen (6 val, 25 train), 277 held out. No experiment is held
  out. PREPARE.json records `split_mode` wells, `seen_fraction`, `val_of_seen`, `min_val_wells`, the seed, the rule,
  `cells_per_split`, `wells_per_split` and `wells_per_plate` (min, max per split).
- run prep `prep_<cell type>_full_v5_run`: every train cell and every val cell (`subsample --train-well-fraction 1.0`):
  no train well is left out, so the prepare stage's `unused` split is empty (the `unused_wells` audit group has no
  well and is reported as n/a); `seen` = train + val, `heldout` = the held-out wells.
- Held-out wells now come from every experiment. Held-out treatment retrieval still runs across experiments: each
  treatment siRNA has one well per experiment, so a held-out treatment well's positives are in other experiments.
- Expected training cells, which the run times scale with: about 8 % of the cells (10 % seen x 80 % train),
  ~23k (U2OS), ~51k (HEPG2), ~49k (RPE) and ~111k (HUVEC); the epoch cap is
  `clamp(round(8 + 6 sqrt(128 / N_train)), 8, 20)`.

A v5 prep is built from an existing v4 prep of the same cell type; an existing v5 output is never overwritten.
If U2OS's seen group cannot support the seen well mAP, it may be dropped from full_v5.

## Arms

| # | arm | what runs | tool calls |
| --- | --- | --- | --- |
| 1 | `frozen` | the analyzer at `adaptation_depth="zero_shot"` on the run prep (the pretrained backbone; SHAPE's trial 0) | 1 |
| 2 | `dinov3` | the plain DINOv3 fine-tune: the analyzer at its defaults, seed 0, on the train wells only (`control.py`) | 1 |
| 3 | `shape` | SHAPE's agentic VRR on the analyzer (`shape_arm.py`, below), gpt-5.6-luna | trial 0 + at most 8 |
| 4 | `react` | gpt-5.6-luna + the analyzer on the shared ReAct runner (`react_arm.py`, below) | trial 0 + at most 8 |
| 5 | `llm` | gpt-5.6-luna writes a hand-crafted feature module, run on every cell (`llm_arms.py`, below) | 0 |
| 6 | `openphenom` | OpenPhenom-S/16 zero-shot (`external.py`, below) | 0 |
| 7 | `cellprofiler` | CellProfiler-style features with cp_measure (`external.py`, below) | 0 |

Analysis only: `shape_top`, the finished SHAPE call with the highest val proxy, the guard ignored. `default_s0` (A10-A22: the defaults on the train and val crops, best point by the val
proxy) is retired; a cached CONTROL.json holding it is stale (`CONTROL_STALE`) and is kept aside, never overwritten.

**The epoch cap** (A6, `epoch_budget.py`): `E = clamp(round(8 + 6 * sqrt(128 / N_train)), 8, 20)` from the actual
training split (HEPG2: 5,074 cells, E = 9), published once as `cache/<cell type>/TRAINING_BUDGET.json` (a different
existing manifest is refused). Every SHAPE and ReAct call trains at most E epochs (`max_epochs` is not a lever they may
set), early stopping unchanged; a resumed call trains at most E more. `dinov3` gets `max_epochs` = E x 8 rounds = 72 as
an explicit cap and the tool's training-loss plateau with no epoch floor (`min_epochs=1`, an analyzer kwarg that is not
a CARD lever); without val crops the tool keeps the lowest-training-loss epoch as its `resume_best`.

### SHAPE (`shape_arm.py`)

`RepresentationAgenticVerifier` through `StepVerifier`'s wave mode (`shape/tools/cell_state_analyzer/CARD.md`,
"Agentic VRR"); the benchmark calls the verifier directly, not through `solve()`. The tool sees the run prep's crops
with their plate, well, site, experiment and split, never a label. Trial 0 is the frozen call (`ssl_calls/input`), not
a round. Each round the agent returns one move (concurrency 1): a parent (`input` or a finished call) and lever changes
from the CARD's `## Levers`, each with a basis; a move from a finished call resumes its `resume_best` unless it sets
`init_from`. Round 1 is the expert start (A21, SHAPE only, `shape_arm.EXPERT_CONFIG`): view_pairing same_well,
contrastive_mask same_well, batch_sampling per_plate, intensity_norm per_plate, augmentation light, and no other lever;
a move that differs is rejected through the repair question. From round 2 the search is the agent's: it may continue,
go back to an earlier call, change direction or stop (no forced exploration at a plateau, A20; no list of untried
levers, A21); the round budget (8) is the backstop. Every call is measured on the val wells (the label-free proxy
`sibling_site_map` and its alarms); delivery is the agent's final choice among the calls the guard admits against
frozen, else frozen. `submission/decision.json` is written after trial 0 and every round as a provisional record and
overwritten by the delivery. The reasoner runs at `reasoning_effort` high with a 16,000-token completion cap (A20;
`run_rxrx1.REASONING_EFFORT`, `MAX_COMPLETION_TOKENS`) and `SHAPE_REQUIRE_AGENTIC=1` (a reasoner failure is an error;
`--no-engine` runs the deterministic fallback, a smoke only). `RXRX_SEED` (default 0) is the analyzer seed of every
call; a run of another seed is another run.

### ReAct (`react_arm.py`)

The shared ReAct design of every benchmark (`benchmarks/_shared/react_arm.py`, `REACT_ARM_CONTRACT.md`), one session,
the unit being the dataset. The prompt holds SHAPE's task text verbatim (`shape_arm.task_brief`), the input facts
(the crops, their label-free metadata, that the default call is the frozen trial 0, the fixed epoch line), SHAPE's 15
levers (without `max_epochs`) as name, type and default, the budget (8 rounds after trial 0, one call per round) and
the runner's python action. RxRx1-only deviation (A16): a PARAMETERS reference with each lever's CARD Type, Range,
Default and mechanics text, with the sentences about SHAPE's readouts cut (`react_arm.SHAPE_CONTEXT`). No SHAPE
readout, alarm, guard, search rule, expert start or label. Each observation is the tool's own returned dict (with
`val_history`) and its loss-curve PNG; python steps (not rounds) read each finished call's h5ad and the label-free
metadata, never `private/`. A call is a SHAPE call (the same tool input, managed invocation and kwargs); values are not
checked by SHAPE's move checks. Delivery is the agent's `finish` (default = frozen); a session without one delivers
frozen (`default_session_unfinished`). A fatal engine reply stops the arm (`REACT_LLM_FATAL`).

### llm (`llm_arms.py`)

LLM-written hand-crafted features: no analyzer, no training, no pretrained weights. One request returns a Python module
defining `features(crops)`; the task text states facts only (the cell count, the 6 channels, 64 x 64 px uint8 crops,
the stretch, the goal, the interface `features(crops) -> (n, d)` with 1 <= d <= 4096, the packages, 16 CPUs + 1 GPU,
the time left and the attempts). A static check (`static_check`) rejects imports and names with file, network or process
access; the module runs in a separate Python process (a fresh temporary cwd outside the run tree, no credential in its
environment, an audit hook blocking network, subprocesses, native code it wrote and an open under the benchmark's
roots) that receives only the crops: first 256 (smoke), then every crop. A failed check, a crash or an invalid output
goes back to the model with its module; no result or metric is ever fed back. Budget (contract A5 as amended
2026-10-13, user decision: the SHAPE and ReAct arms' budget in kind and size; before, 3 attempts within a fixed 5 h):
at most `max_rounds` attempts (SHAPE's rounds, 8; one module per attempt, as one SHAPE round holds one call; `score`
refuses an llm run whose `max_attempts` is not the SHAPE run's `max_rounds`) and SHAPE's wall-clock rule -- the
job's deadline (the Slurm limit; on the VM `RXRX_DEADLINE_UTC`, else none), a module stopped 15 min before it (SHAPE's
per-call cap `max_minutes`, the only per-module timeout) and no attempt started after that; without a deadline no
time limit. It refuses a job that ends within those 15 min, fewer than 16 CPUs or no GPU. A run of the fixed 5 h
budget is not current (`llm_design_problems`). Scored as an external feature arm. A
delivered run of an earlier commit is reused when its design is unchanged (`llm_arms.llm_design_problems`,
`RXRX_REUSE_LLM_RUN`).

### External arms (`external.py`)

Both embed every FULL cell in the `external` stage into `<output root>/<cell type>/external/<arm>/` (`embeddings.npy`,
float32, raw; `EMBED.json`: the settings, package versions, commit and the sha256 of `external.py`, the embeddings, the
prep table and `images.npy`). Neither reads a label. An arm is kept when its settings, code, prep table and
`images.npy` match, and refused otherwise; score refuses one whose embeddings changed (`INTEGRITY_FAILED`).

- `openphenom`: OpenPhenom-S/16 zero-shot,
  [recursionpharma/OpenPhenom](https://huggingface.co/recursionpharma/OpenPhenom) at revision
  `0f92333685f6e9f031b804c70fe246f9b05ae90d` (Recursion's channel-agnostic MAE, ViT-S/16, trained on RxRx3 and JUMP
  cpg0016; no RxRx1). The `assets` stage (a login node) stages the snapshot under
  `<output root>/assets/openphenom/` with every file's sha256; the `external` stage loads it offline
  (`HF_HUB_OFFLINE=1`) with the snapshot's own code and `torch.load(weights_only=True)`. The 6 channels in their stored
  order enter at 256 px only (fixed positional grid): a 64 px crop is resized (`OPENPHENOM_FIT=resize`) or centred on a
  zero canvas (`pad`, the basic profile's setting). Readout: `MAEModel.predict` with
  `return_channelwise_embeddings=False` (384-d). The card's PCA-CenterScale is not applied (no PCA for any arm).
  License: Recursion's Non-Commercial End User License Agreement. Attribution (its section 4; also printed in
  RESULT.md): "We used the Phenom-Beta AI model, available from Recursion Pharmaceuticals, with software documentation
  at www.rxrx.ai, and from NVIDIA Corporation at https://developer.nvidia.com/docs/bionemo-service/phenom-beta.html,
  pursuant to Recursion Pharmaceuticals' licensing terms at
  https://huggingface.co/recursionpharma/OpenPhenom/blob/0f92333685f6e9f031b804c70fe246f9b05ae90d/LICENSE. Under
  this license, Recursion Pharmaceuticals disclaims all representations and warranties with respect to such AI
  model." The model is used unmodified for a non-commercial benchmark comparison.
- `cellprofiler`: CellProfiler-style features per crop with [cp_measure](https://github.com/afermg/cp_measure) 0.2.0
  (`bulk.get_core_measurements`), not the full CellProfiler pipeline. Nuclei = Otsu on the smoothed DNA channel (the
  prep's nucleus rule), the central one kept; cells = a seeded watershed of the cytoplasm foreground, the central
  nucleus's region kept (close to IdentifySecondaryObjects / Propagation); fallback the whole crop (counted in
  `EMBED.json` `masks`). Per compartment: shape, and per channel intensity, Haralick texture, granularity and radial
  distribution: 1,368 raw features; positional, constant and mostly non-finite features dropped. CPU, `CP_WORKERS`
  processes; about 2.9 s per cell and process.

Their packages go into a venv layered on shape_bench (`EXTERNAL_VENV`, default
`/home/groups/your-group/your-user/venvs/shape_external`), never into shape_bench itself (it carries transformers 5.x and
pillow 12). One-time setup:

```bash
source /home/groups/your-group/your-user/miniconda3/etc/profile.d/conda.sh
conda activate /home/groups/your-group/your-user/miniconda3/shape_bench
python -m venv --system-site-packages /home/groups/your-group/your-user/venvs/shape_external
/home/groups/your-group/your-user/venvs/shape_external/bin/python -m pip install \
    timm "transformers>=4.46,<5" mahotas "pillow<12" cp-measure==0.2.0 centrosome==1.3.4
```

## Metrics

`score.py` embeds every FULL cell with each analyzer arm's delivered state (`embed_only` with its `resume_best`; the
analyzer's UMAP / Leiden step skipped) and reads the external and llm arms' vectors. Every arm is scored on two tables
(A3, `score.post_process`, label-free, one code path):

- native: the analyzer arms and openphenom raw; cellprofiler and llm with one global robust centre-scale. The
  pre-registered reading (shape vs the default control, `dinov3`) and the analyzer comparisons are read here.
- matched: the same per-plate robust centre-scale (median, 1.4826 x MAD, clipped at +-10) on every arm. The external
  comparisons are read here. No PCA in either table.

| readout | definition |
| --- | --- |
| well mAP | well profile = the mean of the L2-normalised cells, L2-normalised; each well ranks the other wells of its group by cosine, positives = its siRNA's wells; AP per query, mean per siRNA, mean over siRNA (a port of the task-116 harness) |
| paired band | delta of two arms' per-well APs, cluster bootstrap over siRNA (2,000 resamples, seed 0), +- 2 SE; significant when abs(delta) > 2 SE |
| chance | the exact expected AP of a random ranking per query, averaged like the well mAP |
| probe (A9, `secondary.py`) | siRNA classification across experiments: train on the seen experiments, test on the held-out ones; well profiles and cells; kNN (cosine, k = 10) and logistic regression; balanced accuracy and macro F1 with 95 % intervals from 1,000 bootstrap draws of the held-out wells (the same draws for every arm); secondary, never the reading |
| diagnostic | per SHAPE / ReAct call, from its own embedding of the run-prep cells: the val proxy and the run-prep well mAP vs frozen, and the Spearman rho between them (analysis only, never fed back) |
| efficiency | tool calls, wall hours, reasoner tokens of shape, react and llm |

Pre-registered reading: shape - dinov3 > 2 SE on `seen` = SHAPE's loop beats its defaults; `heldout` read the same
way. A frozen delivery is reported with its case.

Per-arm score cache (`score.ArmScores`): everything one arm contributes (per table its well-mAP evaluation with the
per-well APs, the probe's runs under the fixed bootstrap draws, llm's zero-norm well count) is computed once and kept
in `<cell type>/score_cache/<sha256 of its key>.json`. The key binds the arm's content (an analyzer arm: its kwargs,
the sha256 of its checkpoint and of the analyzer's code; a feature arm: its embeddings' sha256), the prep, the scoring
code (score.py, secondary.py, numpy, scikit-learn), the table and its post-processing; an entry under another key is
recomputed. `score-arm <arm> [<run dir>]` scores one arm as soon as it exists and prints `SCORE_ARM <arm>: well mAP
(native) seen ... heldout ...; siRNA probe ...`; `score` / `finish` read the cache and assemble RESULT.json / RESULT.md;
`summarize_results.py` assembles the comparison offline from the same cache.

A run is refused (`ARM_STALE`) when its code hashes or prep differ from the current ones, its design differs (llm:
`llm_design_problems`; react: another `react_arm.py` or shared runner, another CARD.md, round budget or epoch budget),
or its model is not the SHAPE run's.

## How to run

Every stage refuses a checkout that is not at `RXRX_EXPECTED_COMMIT` (40-hex), not on the branch
`RXRX_EXPECTED_BRANCH` (default `main`, read by `runners/run_rxrx1.sh`) or has uncommitted files under
`benchmarks/representation_RxRx1` or `shape`. The
reasoner authenticates only through the Stanford AI Gateway (`LLM_API_KEY`, `LLM_BASE_URL` must be
`https://llm-gateway.example.edu/v1`); the key is never printed.

### Running on the full set

The full-set v4 preps (Data: Full-set preps; HEPG2, RPE, U2OS, HUVEC, every well, 80 % of the experiments held
out) run with the same driver and launcher; the v5 preps (90 % of every plate's wells held out) alike with
`RXRX_PREP_SET=full_v5` / `--prep-set full_v5` and the run label `<cell type>_full_v5`:

```bash
export RXRX_EXPECTED_COMMIT=<40-hex commit of the checkout> RXRX_EXPECTED_BRANCH=<its branch>
RXRX_PREP_SET=full_v4 CELL_TYPE=U2OS bash benchmarks/representation_RxRx1/runners/run_rxrx1.sh <stage>     # one stage
python benchmarks/representation_RxRx1/run_rxrx1.py <stage> --prep-set full_v4 --cell-type U2OS ...    # locally
```

- Everything of a full_v4 prep is named by its run label `<cell type>_full_v4`: the cache
  `cache/<cell type>_full_v4/` (prep, private, control, external, score cache), the runs
  `<stage>_<cell type>_full_v4_<commit8>_...`, the smoke, the stage logs and the driver's state file; v3 keeps
  `<cell type>`. U2OS is a full_v4 cell type only.
- The driver runs the prepare stage when `cache/<label>/prep` is missing (from `RXRX_DATA_ROOT/prep_<cell
  type>_full_v4[_t5k]`; it needs the pinned DINOv3 source and weights, as the launcher's prepare stage does) and the
  external stage (openphenom pad, cellprofiler on every CPU) when the external caches are missing; with v3 it only
  checks them.
- Reasoning effort: the three LLM arms (shape, react, llm) get one reasoner (`run_rxrx1.reasoner`): reasoning effort
  high and a 16,000-token completion cap, recorded in each RUN_META.json (`reasoning_effort`,
  `max_completion_tokens`; a resume refuses other values). Before this, the llm arm ran at the engine's default
  (medium) with its default cap: such a run is not an effort-matched llm arm.
- Negative controls: every negative_control well has siRNA `EMPTY`; those wells are not scored (neither query,
  positive nor candidate in the well mAP; not a class of the siRNA probe) but stay in every label-free step (the
  embeddings, the per-plate centre-scale). RESULT.json `negative_controls` records what was left out; the v3 preps have
  none and score exactly as before.
- Size: up to 1.42 M cells (HUVEC, a 35 GB `images.npy`) and about 1,138 siRNA classes. Each analyzer arm embeds every
  cell at score time (about 30 min for HUVEC). The siRNA probe keeps its well unit only (kNN and logistic regression
  on well profiles; user decision 2026-10-13): the cell unit (every held-out cell) would take days per arm and table
  at this size; RESULT.json `secondary.settings.skipped_units` and the probe layout say so (`score.PROBE_SKIPPED`).
- The llm arm has SHAPE's budget (A5 as amended, llm above): set `RXRX_DEADLINE_UTC` on the VM, else no limit
  stops a module.

## Records and traces

Everything a stage writes is under `<results root>/representation_RxRx1/` (`SHAPE_RESULTS_ROOT`):

```
logs/                                   the stage logs <stage>_<cell type>_<run tag>_<UTC>.log
cache/assets/                           the pinned DINOv3 (ASSETS.json), assets/openphenom/
cache/<cell type>/prep/, private/       the tool table and PREPARE.json; labels.csv (score.py only)
cache/<cell type>/control/              CONTROL.json (each arm's kwargs, input, training facts, provenance) + its calls
cache/<cell type>/TRAINING_BUDGET.json  the shared epoch cap E
cache/<cell type>/external/<arm>/       embeddings.npy, EMBED.json
cache/<cell type>/score_cache/          <sha256>.json per arm and table; embed/<arm>-<digest>/ the scoring embeddings
shape_<cell type>_<commit8>_<UTC>_<job or run tag>/
    RUN_META.json                       commit, host, model, reasoning effort, seed, expert_config, training budget,
                                        CARD sha256, the analyzer's code sha256
    AGENT_RUN.json                      timing, stop reason, reasoner tokens, the agentic record (rounds, calls, choice)
    logs/input_trial.json, tool_calls.jsonl   trial 0; one start / end record per call (parent, init, levers set,
                                        diagnosis, kwargs, the verifier's measurements, the tool's training facts)
    logs/agent_log.jsonl                every prompt, answer, rejection and stop of the verifier
    logs/reasoner.jsonl                 (VM) every reasoner request and its reply as returned: the replay source
    cache/cell_state_analysis/ssl_calls/<call>/   the tool's outputs; (VM) CALL_RECORD.json
    submission/decision.json            the delivered call (or frozen), its checkpoint, the case
    RESULT.json, RESULT.md, EXPORT_<run>.tar.gz, COMPLETE   written by finish
react_<...>/                            RUN_META.json (+ react_arm.py and shared-runner sha256, round cap, parameters),
                                        AGENT_RUN.json, react/<cell type>/{TRACE.jsonl, CALLS.json, SELECTED.json},
                                        react/REACT_SUMMARY.json, the shape-format logs/, cache/, submission/
llm_<...>/                              RUN_META.json, AGENT_RUN.json, modules/attempt_<k>.py + worker.py,
                                        logs/{requests,attempts,usage}.jsonl, attempt_<k>.stderr.log,
                                        submission/{embeddings.npy, EMBED.json, decision.json}
```

A results folder in this repository (`results/<run>/`) holds a run's non-image records: `compare/` (the offline
summary), `control/`, `shape/`, `react/`, `llm/` and `logs/` (no embeddings, checkpoints or h5ad), and its README.md
with the provenance.

## Code layout

The top level holds this README, the entry point `run_rxrx1.py` and `results/`; every other file is in `runners/`.
Runs made before this layout recorded the earlier paths
(`benchmarks/representation_RxRx1/<file>`) and the sha256 of the files before the move, whose imports changed with
it: a rerun recomputes the score cache entries, refuses a react run, a training budget or an external cache recorded
with the old hashes (`ARM_STALE`, `TRAINING_BUDGET_STALE`, `EXTERNAL_STALE`) and needs a fresh GPU smoke.

| file | what it does |
| --- | --- |
| `runners/config.py` | the benchmark's settings: cell type, the preps of each cell type, the output root, control seeds, rounds, model |
| `runners/prepare.py` | prepare stage: the task-116 preps as tool inputs (`cells.csv`, PREPARE.json, labels), the pinned DINOv3 assets |
| `runners/build_prep.py` | the prep builder (prepare, subsample, resplit): FULL and run preps in the task-116 v3 format |
| `runners/control.py` | control stage: the `frozen` and `dinov3` calls (CONTROL.json); shared helpers (`fs_retry`, provenance) |
| `runners/epoch_budget.py` | the dataset-derived per-call epoch cap E shared by SHAPE and ReAct (TRAINING_BUDGET.json) |
| `runners/shape_arm.py` | the SHAPE arm: the verifier loop, its ledgers and the provisional / final decision |
| `runners/react_arm.py` | the ReAct arm's tool adapter on the shared ReAct runner, and its records |
| `runners/llm_arms.py` | the llm arm: the request, the static check, the sandboxed module run and the delivery |
| `runners/external.py` | the openphenom and cellprofiler arms (assets stage, external stage, pilots) |
| `runners/score.py` | score stage: embeddings, both tables, well mAP, paired bands, the diagnostic, the score cache, RESULT |
| `runners/secondary.py` | the siRNA probe across experiments (balanced accuracy, macro F1, bootstrap intervals) |
| `runners/summarize_results.py` | the offline comparison of a scored run from the score cache (`compare/`) |
| `runners/resume.py` | the VM platform, resume by replay, and the driver's helpers (`find-run`, `status`, `add-tag`) |
| `runners/basic.py` | the driver's read-only gates: verified control and external caches, a complete RESULT |
| `runners/preflight.py` | the pre-stage check: commit, imports, GPU, epoch budget, a structured Gateway request |
| `runners/smoke_gpu.py` | the GPU smoke: the real DINOv3 trains and resumes at E on a small label-free subset |
| `run_rxrx1.py` | the stage CLI (prepare, assets, control, external, shape, llm, react, score-arm, score, finish) |
| `runners/run_rxrx1.sh` | the launcher of one stage on an allocated GPU machine (checks, environment, logs) |
| `runners/test_benchmark.py` | CPU tests on a synthetic prep: leaks, the scorer's parity with task 116, prepare / control / shape / score end to end, the external and llm arms, the score cache |
| `runners/test_react.py` | the ReAct adapter: prompt, observations, budget, delivery, the python allow-list, resume by replay |
| `runners/test_resume.py` | the VM platform and resume by replay of shape and llm |
| `runners/test_basic.py` | the stage gates (mocked shell) |
| `runners/test_epoch_budget.py` | the epoch cap |
| `runners/test_build_prep.py` | the prep builder on a fake RxRx1 zip (workers, well types, holdout guard, --train-cells, the prepare stage on a built prep) and the full-set launchers |
| `runners/test_secondary.py` | the probe against sklearn and on planted structure |
| `runners/test_summarize_results.py` | the offline summary on a stub score cache |
| `results/` | the reported run's records |

Tests (CPU): `python -m pytest benchmarks/representation_RxRx1 benchmarks/_shared -q`.

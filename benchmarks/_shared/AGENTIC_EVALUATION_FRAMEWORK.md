# Evaluation framework for the end-to-end agentic benchmarks (full_*)

Status: proposal and inventory (2026-10-14, five layers). It organises what the full_* benchmarks (full_CPJUMP1,
full_LIVECell, later ones) measure, what they do not yet measure, and the order in which the gaps are closed. The
verdict metrics stay as fixed by `FULL_METRICS_CONTRACT.md`. Reference design: BIABench (Pan, Panzeri et al., preprint,
2026-09-29): end-to-end bioimage-analysis tasks scored by an outcome score against ground truth and, separately, a
judged process score.

## 1. What is taken from BIABench

1. **Rank on the outcome; report the process beside it.** BIABench's judged process score did not predict the outcome
   (Pearson r = 0.29 over 282 runs, 0.09 without near-zero outcomes; an expert's own scores r = -0.03). Runs below 0.1 still
   got process scores of 0.74-0.82. So the process is reported, never used to rank. Here the process evidence is measured from the logged
   trajectories against ground truth (L3, L4), not judged by a model.
2. **Run end states are classified** (delivered / no deliverable / failed / refused); a stage delivered by a fallback is
   not a stage delivered by the agent (L4).
3. **Time and a plausible process do not show correctness**: within a task the rank correlation of wall-clock with outcome
   was about 0; runs below 0.1 took twice as long. Efficiency (L5) is reported as a cost, not as a quality signal.
4. **Efficiency is reported with its provenance**: tokens in / out / cached / reasoning, wall-clock, tool calls, cost with a
   stated price table; a signal an arm does not expose is "unavailable" (null), never zero; medians for heavy tails.
5. **Harness and model are separated, effort is fixed and logged** (done: the reasoning table is recorded per stage).
6. **Failure policy is declared**: infrastructure failures (VM stop, gateway outage) are retried once from the same inputs
   and logged; agent-side failures are scored as observed.

Not adopted (by decision, 2026-10-14): a judged process score and its rubric, a separate reliability layer, a separate
honesty layer, a separate delivery layer. Two cautions stay as reporting rules (section 5).

## 2. The five layers

| layer | question | scored against | role |
| --- | --- | --- | --- |
| L1 End-to-end outcome | is the delivered result scientifically right? | ground-truth labels (scorer only) | ranks the arms |
| L2 Stage outcome | is each stage's own product right? | ground-truth annotations, **only where the dataset has them** | reported; explains L1 |
| L3 Decision quality | were the agent's tuning decisions good? | the measured effect of every call, **only where it can be measured** | reported |
| L4 Protocol friction | how much did rules, formats and errors cost? what did the arm really deliver? | logged records | reported |
| L5 Efficiency and cost | what did the result cost? | logged usage and time | reported, with a quality-cost view |

Status below: EXISTS (computed in code today), PARTIAL, TODO.

## 3. Metrics

### L1 End-to-end outcome

| id | metric | status |
| --- | --- | --- |
| O1 | mAP, macro F1, balanced accuracy, each main and guard, with the verdict against the control (the contract) | EXISTS |
| O2 | retrieval endpoints (mAP, MRR, top-1), uncorrected and delivered | EXISTS |
| O3 | per-class precision / recall / F1 and the confusion matrix | post hoc script for LIVECell (synthetic-tested, not yet run on real arms); CPJUMP1 TODO |
| O4 | one-vs-rest AUROC and AUPRC, with chance | same |
| O5 | batch integration (batch ASW, iLISI, graph connectivity, cLISI, label ASW, NMI, ARI) and stage / batch leakage | same; CPJUMP1 only inside the batch search |
| O6 | single-cell morphology regression from GT instances (R^2, Pearson r, MSE) | LIVECell post hoc script |
| O7 | representation diagnostics (effective rank, uniformity, alignment, dead dimensions) | PARTIAL (logged per SSL call; alignment missing) |

LIVECell's unit-level mAP sits near its ceiling, so the cell-level macro F1 and balanced accuracy and O3-O6
are where the arms can differ.

### L2 Stage outcome against ground truth (depends on the dataset)

| dataset | which stages have ground truth | metrics | status |
| --- | --- | --- | --- |
| LIVECell | segmentation (expert instance masks), crop (derived from them) | S1 segmentation: AP50, mean AP@0.5:0.95, F1@0.5, predicted/GT count ratio; S2 crop: target retention, neighbour contamination, full-cover rate, unmatched-crop rate, GT recall; per type, stage, usage, official test wells | EXISTS |
| CPJUMP1 | none for segmentation or crop (labels exist only for the final gene identity) | none: stage effects are read only through L1 | n/a |
| a future dataset | follow the same rule: a stage gets an L2 row only if its product has independent annotations | | |

Rule: no L2 row is built from a proxy the agent itself sees (the tool's own quality readouts are not ground truth).

### L3 Decision quality: can it be judged?

Yes where the measured outcome of **every** call (not only the delivered one) is available and independent of what the
agent saw; no otherwise. The held-out and ground-truth labels are used post hoc by the scorer and are never visible to an
agent, so this measures the agent's judgment, it is not an optimisation target.

| stage | dataset | per-call outcome | cost to measure | judgeable? |
| --- | --- | --- | --- | --- |
| segmentation | LIVECell | GT AP50 / F1 of each call's mask (every attempt's mask is kept) | low (matching only) | **yes, fully** |
| crop | LIVECell | GT retention / contamination of each call's boxes (per-call crop metadata is kept) | low; needs one scalar crop quality, e.g. contamination subject to a retention floor | **yes, fully** |
| segmentation, crop | CPJUMP1 | none independent of the agent (no GT) | scoring each call downstream would mean one full downstream run per call | **no** |
| SSL (embed) | both | the downstream L1 probe on held-out cells, per trained call | medium: each call's checkpoint is kept; embedding the capped probe sample (<= 100 cells per held-out unit, about 40k cells) is minutes per call | **yes** (post hoc, held-out labels) |
| batch | both | the downstream L1 score of each call's corrected profiles | low (profiles are kept per call; transductive correction on all units) | **yes** |

Metrics (kept to five):

| id | metric | definition | status |
| --- | --- | --- | --- |
| Q1 | adaptation gain | delivered quality minus the default call's quality, per image (or per stage), with a paired bootstrap | TODO per call; the delivered-vs-control difference exists |
| Q2 | decision efficacy | share of the agent's retunes (calls after the first) that improved, did not change, or worsened the quality (threshold eps) | EXISTS for segmentation in the CellSeg benchmarks (`_shared/segmentation_metrics.py`); TODO for the full_* runs |
| Q3 | selection regret | quality of the best call minus the quality of the delivered call: the quality of the arm's own selection | TODO |
| Q4 | per-input adaptation | per-image delivered quality minus the best single parameter set applied to all images (tests that per-input tuning beats one global setting; reference point: the best global call among the arm's own calls) | TODO (LIVECell seg and crop) |
| Q5 | proxy-target alignment | rank correlation, over the arm's own calls, between the signal the agent saw (validation proxy, calibration score) and the held-out metric; whether the proxy-chosen call is the held-out best | TODO (SSL and batch) |

Not judgeable, so not scored: whether the stated reasoning was right (only outcomes are measured).

### L4 Protocol friction and what was really delivered

| id | metric | source | status |
| --- | --- | --- | --- |
| P1 | end state per run and stage: delivered by the agent, or by a fallback (`default_unnamed`, `default_session_unfinished`, `failed_round`, `no_valid_move`, `invalid_search_plan`), failed, timed out | `agent/<stage>/SELECTED.json`, `*_SEARCH.json`, markers, `RUN_META.json` | EXISTS (`_shared/agentic_process_readouts.py`) |
| P2 | rejected requests, malformed replies, repairs and repair success, execution errors, reasoner failures, per stage | `ROUNDS.json` outcomes, SHAPE `execution_errors`, `reasoner_failures`, `fallback_rounds` | EXISTS (`_shared/agentic_process_readouts.py`) |
| P3 | rounds used, rounds wasted on format / rule errors, stop reason (satisfied, plateau, max iterations, no valid move, conceded) | same | EXISTS (`_shared/agentic_process_readouts.py`) |
| P4 | rule activations: admission SKIP / ABSTAIN, guard rejections, four-stage contrast rejections | SHAPE records | EXISTS (`_shared/agentic_process_readouts.py`) |
| P5 | a delivered stage that is no better than the default: the arm spent its budget and delivered a default-equivalent result | L2 / L3 with P1 | TODO |

### L5 Efficiency and cost

| id | metric | status |
| --- | --- | --- |
| E1 | tokens in / out / reasoning / cached, per stage and total | EXISTS (`*_USAGE.json`; reasoning tokens only where the provider reports them; `_shared/agentic_process_readouts.py`) |
| E2 | wall-clock per stage and total | EXISTS (`_shared/agentic_process_readouts.py`) |
| E3 | tool calls and LLM calls per stage | EXISTS (`_shared/agentic_process_readouts.py`) |
| E4 | cost per run from a pinned price table; the token counts are always reported | TODO |
| E5 | gain per cost: quality gain (L3 Q1 or the L1 delta) per 1k tokens, per tool call, per minute; a quality-cost view across arms | PARTIAL (segmentation suite only) |
| E6 | a signal an arm does not expose is null with a reason | EXISTS (`_shared/agentic_process_readouts.py`) |

## 4. Reference rows

| id | row | purpose | status |
| --- | --- | --- | --- |
| B1 | zero-agent control (default parameters, one default batch correction) | the floor | EXISTS |
| B2 | best single global call among an arm's own calls | the reference for Q4 | TODO |

## 5. Reporting rules

1. Rank on L1; say which L2 results explain it; report L3-L5 beside it; never fold the layers into one number.
2. Every table states n, the seed count, the commit and the reasoning settings of each arm; arms on different commits or
   effort levels are marked "not strict".
3. **Single-seed caution.** Every full_* result so far is one seed, and the standard errors are bootstraps over test units,
   not over seeds. BIABench found run-to-run variance about six times the variance explained by the choice of agent, and in
   40% of agent-task pairs identical runs differed by more than 0.2. A difference between arms is therefore labelled a
   single-seed observation until repeats exist.
4. A missing signal is null with a reason; a stage an arm did not run is not a zero.
5. Stratify: by stage, cell type and density band (LIVECell), by difficulty where it exists.
6. L3 uses held-out or ground-truth labels only post hoc, never in a run; the scorer's use is reported as a diagnostic.

## 6. Order of work

| priority | work | needs |
| --- | --- | --- |
| P0 | `agentic_process_readouts.py`: P1-P4 and E1-E3, E6 from the existing run directories (CPJUMP1 five arms and reruns; LIVECell when done) | logs only |
| P1 | LIVECell L3 for segmentation and crop (Q1-Q4, with per-call GT scoring of the kept masks and boxes), after the SHAPE and Codex runs end | GT, per-call masks |
| P1 | L3 for SSL and batch on both datasets (Q3, Q5): re-embed the capped held-out sample per kept checkpoint; score each call's kept corrected profiles | VM time |
| P2 | O3-O6 post hoc for CPJUMP1; E4 price table | prices |

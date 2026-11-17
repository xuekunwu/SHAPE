# Full benchmark metric contract (user decision 2026-10-11)

Applies to every end-to-end `full_*` benchmark (today full_CPJUMP1). Scoring runs after delivery;
labels are read only by the scorer. One shared implementation: `benchmarks/_shared/full_metrics.py`. The
representation_RxRx1 SSL benchmark keeps its own reading. Its probe core (`scores`, `chance`, `knn_predict`,
`logreg_predict`) moves into the shared module unchanged, and `representation_RxRx1/runners/secondary.py` imports it,
so the probe is the same code everywhere.

## Verdict metrics (all three, each with its own verdict)

| metric | definition |
|---|---|
| mAP | same-label retrieval of unit profiles (unit profile = mean of L2-normalised cells, L2-normalised; cosine); per-query AP over the split's candidates; macro over labels |
| macro F1 | the probe's per-class F1, averaged over the classes with a true or a predicted sample (sklearn's definition, unchanged from representation_RxRx1; user decision 2026-10-11) |
| balanced accuracy | the probe's mean per-class recall over the classes with a true sample (sklearn) |

Probe settings:
- train on the split's train side and test on its test side, then swap the sides and average; a split's folds
  may hold several parts whose test predictions are pooled into the fold's confusion matrix;
- classifiers: kNN (cosine, k = 10, majority vote, the representation_RxRx1 tie rules) and logistic regression
  (StandardScaler, lbfgs, max_iter 2000, class_weight balanced, random_state 0);
- units: cells and unit profiles; the cell-level probes use at most 100 cells per unit (the harness cap), sampled
  deterministically from that arm's own delivered and agent-excluded cells (per unit: its cells sorted by cell id,
  permuted by `numpy.random.default_rng(0)`, the first 100; fewer cells: all of them); a sampled excluded cell
  (CPJUMP1 `excluded_cells.json`, HCMA the chain's excluded cells) is a test cell without a prediction, a miss, as
  HCMA's knn_transfer counts it, and never a train cell (user decision 2026-10-11);
- a test unit with neither delivered nor excluded cells counts as misses: at the cell level 100 test cells without a
  prediction (the HCMA rule for an expected image without cells); a unit without delivered cells is at the unit
  level one wrong prediction; for mAP a query unit
  without a profile has AP 0 and is never a candidate; `n_units_without_cells` is reported (user decision
  2026-10-11);
- verdicts use the **cell-level kNN** probe; the other three classifier / unit combinations are reported.

Splits and guards:

| benchmark | label (scorer only) | unit | main split | guard split |
|---|---|---|---|---|
| full_CPJUMP1 | gene (110 evaluation genes; development genes excluded) | well | cross-plate (train plate A, test plate B, and the reverse) | cross-plate AND cross-guide (the test well's guide never seen for that gene in training) |

full_CPJUMP1 unit profiles: batch correction runs on well profiles only, so the mAP verdict and the unit-level
probes use the delivered corrected well profile (batch correction is part of what is evaluated) and the
cell-level probes the uncorrected cell embedding; the mAP of the contract's profile (the mean of the
L2-normalised uncorrected cells) is reported, not a verdict (user decision 2026-10-11).

For each metric:
- Uncertainty: a 95% interval from 1000 bootstrap resamples of the test units, seed 0. A drawn unit brings all its
  cells, and every arm gets the same draws.
- Chance: a stratified random predictor (the representation_RxRx1 formulas), plus mAP chance from the candidate
  counts.
- Oracle: where the benchmark defines one; HCMA keeps its one-hot oracle.
- Verdict vs the control: delta = arm minus control on the same draws, SE = the bootstrap s.d. of the delta.

| verdict | condition |
|---|---|
| GAIN | delta > 2 SE and the guard delta >= 0 |
| SOURCE_SHORTCUT_SUSPECTED | delta > 2 SE and the guard delta < 0 |
| LOSS | delta < -2 SE |
| NO_RESOLVABLE_DIFFERENCE | otherwise |

The three verdicts are reported side by side; they are never collapsed into one.

## Kept as reported, not verdicts

- Benchmark-specific readings:
  - CPJUMP1: cross-guide cross-plate retrieval mAP/MRR/top-1 and same-guide retrieval;
  - HCMA: knn_transfer_macro_f1, state metrics, strata, scIB, NSC/NSCB.
- Attribution, candidates, process metrics and the summary table: HCMA's, applied to every full_* benchmark.

## Outputs

`RESULT.json["full_metrics"]` holds, per metric:
- each split's value and interval;
- chance;
- the delta vs the control, its SE and the verdict.

There is also a markdown table in RESULT.md, and `compare` adds the three verdicts per arm.

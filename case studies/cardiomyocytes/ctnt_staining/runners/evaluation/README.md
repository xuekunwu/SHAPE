# Frozen staining evaluation

This subfolder owns **post-freeze evaluation and publication**, not the native
segmentation/crop/SSL algorithms. The production E2E entry remains
[`../run.py`](../run.py). Analysis does not import SHAPE or require a GPU/API key.

It reads a completed fixed-pilot run of CONTROL / SHAPE / **CODEX-FULL**.
CODEX-FULL must not be renamed CODEX-API. Native run identities differ from this
analysis commit; the source manifest records the exact archive and per-file hashes.

## Modules and commands

| File | Responsibility | Inputs and products |
|---|---|---|
| `run.py` | Portable CLI, usable from any working directory | `prepare`, `tables`, `figures`, `umap`, `manifest` |
| `prepare.py` | Curate reviewed public evidence from immutable private exports | Native checkpoints/compact call ledgers/aggregate readouts → anonymous scalar evidence and input hashes |
| `report.py` | Recompute primary, split, donor and failure tables | Public `evidence/` → `main_metrics.csv`, `metrics.json`, `split_metrics.csv`, `donor_metrics.csv`, F3 JSON/CSV |
| `process.py` | Professional failure/efficiency accounting with explicit denominators | Separates native attempts, input records, request rejection, evidence admission, abstention and missing telemetry |
| `figures.py` | Deterministic data-native charts and full-cell UMAP coloring | Public aggregates or hash-verified private coordinates → PNG/PDF plus provenance and pending visual-QA receipt |
| `evaluation_common.py` | UTF-8 JSON, canonical LF CSV, SHA256 and missing-safe rates | Standard library only |
| `test_evaluation.py` | Accounting, abstention, donor-weighting and integrity regressions | Small synthetic ledgers; no API, data downloads or GPU |

`prepare`, `tables` and `manifest` use Python standard library only. Figures need
numpy, pandas and matplotlib. A pinned native runtime already lives at
[`environments/benchmark/linux-py312-cu128`](../../../../../environments/benchmark/linux-py312-cu128/PROFILE.json).
The figure receipt records the actual plotting environment separately.

## Replay published tables and figures

From the repository root, quote the path containing `case studies`:

```bash
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/run.py" --help
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/run.py" tables --result "<a completed run directory>" --out /tmp/staining-replay
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/run.py" figures --result /tmp/staining-replay --out /tmp/staining-replay/figures
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/test_evaluation.py"
```

The public bundle replays aggregate metrics without private data. It does not
pretend that aggregate replay reruns image measurement, fluorescence threshold
fitting, encoder training, logistic-regression fitting or UMAP fitting.

The report presents quality/prediction first, immediately followed by tool and
system efficiency. Its optimization hypotheses are supported by the aggregate
`evidence/shape_optimization_evidence.json`: training-well label mixture, SSL
candidate measurements/validation scope and batch-admission evidence. These
post-freeze diagnostics are not inputs to native agent or SSL decisions.

## Recreate the curated export or full UMAP panels

```bash
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/run.py" prepare --export-root /private/ctnt_staining_independent_pilot_20261021 --codex-trajectory /private/ctnt_staining_shape_repair_20261020/codex_full/trajectory --out /tmp/new-public-bundle
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/run.py" tables --result /tmp/new-public-bundle --out /tmp/new-public-bundle
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/run.py" umap --result /tmp/new-public-bundle --coordinates /private/ctnt_staining_independent_pilot_20261021/analysis_all_cells_umap_v2 --out /tmp/new-public-bundle/figures/umap
```

The input directory keeps the original frozen-export layout. `prepare` strips
artifact paths, credentials and cell labels; exports reviewed per-image logical
call records, anonymous donor aggregates, well-level quality and hashes. It
checks SHAPE history+errors against baseline+candidate attempts, and binds each
CODEX delivery to an actual recorded call. This is a case adapter for these
native schemas, not a universal log parser.

UMAP inputs are private CSVs with `cell_id`, `usage`, `ctnt_label`, `umap_1`,
`umap_2`. The command checks the exact frozen coordinate SHA256, unique cell IDs,
finite coordinates, valid labels, total counts and split counts before plotting.
All own-arm objects are shown, including ambiguous labels. Every arm has an
independent map; split and cTnT panels reuse identical limits/order/coordinates.
The original fit is train-only PCA50 + UMAP30-neighbors/cosine/min_dist0.3,
300 epochs, seed20261017; val/held-out transform only. This command **renders**
those coordinates; it does not refit them or use held-out labels to tune UMAP.

## Metric definitions

### Quality

Public stage QC preserves the common existing formulas, with no post-hoc
weight selection. Quality aggregates **objects → images → equally weighted
wells**. All three arms use the same 456 images / 114 wells.

- Seg `Q=(B+H+O+2F)/5`. `B=r/(1+r)` for the existing boundary-gradient ratio;
  `H=clip(1-mean_object_std/(0.25*(image_range+epsilon)),0,1)`;
  `O=0.5*tiny_score+0.5*coverage_gap_score`, where tiny area is below 0.3×median
  area, `tiny_score=clip(1-tiny_fraction/0.5,0,1)` and
  `coverage_gap_score=clip(1-abs(mask_fg_fraction-Otsu_fg_fraction)/0.5,0,1)`;
  `F` is the existing Otsu minority-foreground coverage by masks. This is a
  **proxy**, not validated segmentation accuracy. Validity gates and independent
  DAPI diagnostics remain essential; trivial invalid masks can get higher Q.
- Crop `R-C`: native target-mask retention minus neighboring-mask contamination.
  It inherits upstream mask errors. Also retain native
  `crop_quality=(0.25*not_truncated+target_dominance+0.25*centered)/1.5`
  and all component readouts to avoid metric selection bias.
- DAPI target nuclear retention: target-assigned nuclear pixels in crop / full
  assigned-nucleus area, undefined without an assigned nucleus. Other nuclear
  fraction: nuclear pixels not assigned to target / all nuclear pixels in crop.
  No assigned target but nuclear pixels present means 100% other; no nuclear
  pixels means undefined, never zero. Assignment uses common Otsu+watershed
  nuclei with ≥50% nucleus overlap in the own-arm mask. Neither registration nor
  complete cell boundaries have independent ground truth.

### Prediction

The primary classification here is an **explicitly separate supervised
linear-probe** on frozen embeddings. Train labels fit logistic regression,
val balanced accuracy chooses C (0.01/0.1/1/10, smaller C on ties), held-out only
scores. Row L2 normalization → train-fit StandardScaler → logistic regression,
all eligible train cells, threshold0.5, no encoder training or val refit.
No shared-cell intersection; ambiguous labels excluded only from scoring.

Pooled accuracy weights cells; donor-equal balanced accuracy weights donors,
averaging sensitivity and specificity within donors with both classes. Report
eligible counts and confusion matrices; a high pooled score can be dominated
by one large, mostly negative donor. Accuracy is conditional on detected and
confidently labeled cells, not detection recall.

### F3, tools and systems

- Native completion = complete logical attempts / all recorded native attempts.
  A QC exception is an execution error with its own class. Residual QC flags on
  delivered results are **not** execution errors.
- Request rejection = rejected controller decisions / recorded decisions;
  run admission and evidence admission use their own request denominators.
  Rejected pre-execution calls do not increase native tool-error counts.
- Work amplification = attempts / delivered units within the same stage;
  image units for seg/crop, dataset units for embed/batch. Do not pool them.
- Selected-attempt share = directly delivered attempts / attempts. Unselected
  probes may inform later decisions, so this is not a causal waste fraction.
- Uncorrected `input` is abstention, **not** a successful correction call.
- Missing ledgers, zero denominators and unmeasured utilization stay `null`.
  Four-stage elapsed sums are not wall time or GPU busy time. Historical
  concurrency, CPU budgets, decision limits and populations differ. There is
  no validated overall A/B grade without predeclared SLOs and common telemetry.

## Integrity and visual QA

```bash
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/run.py" manifest --result /tmp/new-public-bundle
python "case studies/cardiomyocytes/ctnt_staining/runners/evaluation/run.py" manifest --result /tmp/new-public-bundle --verify
```

Figure generation records input/output/script hashes, parameters and software,
leaving QA pending. Inspect each PNG, then explicitly record the reviewer’s
checks/disposition in `FIGURE_EVIDENCE.json` before sealing the final manifest.
Recreating PDFs may change metadata bytes; values, counts and PNG content are
the relevant semantic checks. Do not publish private coordinate/label tables.

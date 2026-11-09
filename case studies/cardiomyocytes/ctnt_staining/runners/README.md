# cTnT staining: preparation, three-arm E2E and repair validation

Post-freeze analysis has a separate, organized entry point:
[`evaluation/README.md`](evaluation/README.md). It rebuilds a fixed-pilot report
from a completed run: own-arm linear-probe tables, full-cell split/cTnT UMAPs
and F3/tool efficiency. It does not alter native run logic or equate this checkout with the
historical scientific source snapshot. The production commands below retain
their original protocol; the report explicitly distinguishes historical
matched-reference scores from the separate own-population readout.

The production runners are in this checkout, with explicit segmentation and
crop policies. `run.py` preserves the metadata
commands and dispatches the E2E commands to `e2e.py`.

## Channels and fixed data split

Segmentation, model evidence, crop geometry, embeddings and SSL receive only
Brightfield. The trusted `evaluation_crops` stage applies the delivered boxes to
Brightfield/DAPI/TroponinT and verifies raw uint16 pixels. DAPI is independent QC;
cTnT provides exploratory train-calibrated cell labels. Neither fluorescence nor
donor identity enters agent decisions. The 40/10/50 well split stays fixed
(seed 20261017); the pilot samples 20% within those splits (seed 20261016), with
114 wells / 456 images. Do not resample the prepared pilot.
`prepare-full` prepares all 574 wells / 2294 images from the same local backup and
frozen split, so pilot images keep their image, well and split IDs. Scoring the full
data against the fixed pseudo-reference needs `run --allow-exploratory-full`
(user decision 2026-10-27); it stays exploratory, not a curated-reference evaluation.

## Entry points

Run from the repository root; quote paths containing `case studies`:

```bash
python "case studies/cardiomyocytes/ctnt_staining/runners/run.py" --help
python "case studies/cardiomyocytes/ctnt_staining/runners/run.py" inspect
python "case studies/cardiomyocytes/ctnt_staining/runners/run.py" prepare-pilot --help
python "case studies/cardiomyocytes/ctnt_staining/runners/run.py" prepare-full --help
python "case studies/cardiomyocytes/ctnt_staining/runners/run.py" pack --out /workspace/source_snapshot.tar.gz
```

`pack` prints the exact source SHA256. A new source snapshot must use a fresh archive path and a new run identity. The snapshot includes the package lock under
`environments/benchmark/linux-py312-cu128` (`SHAPE_BENCHMARK_LOCK` selects another pinned lock, e.g.
`environments/benchmark/linux-py312-cu130-h200/requirements.lock.txt` on the H200 VM); preflight records actual packages,
GPU, CPU, driver and source identity. Existing prepared pixels and fixed splits
can be reused after verification; old completion markers cannot be reused as
completion of this new code.

Example on an already prepared **fixed pilot**, with previously frozen matching
references and configured local assets:

```bash
python "case studies/cardiomyocytes/ctnt_staining/runners/run.py" run   --prep /workspace/prepared_pilot20   --pilot-manifest /workspace/prepared_pilot20/PILOT.json   --references /workspace/references   --assets-root /workspace/assets   --expected-source SOURCE_SHA256   --out /workspace/new_pilot/shape   --arm shape --model gpt-5.6-luna --cores 8 --fraction 1 --allow-exploratory
```

Use a fresh output directory per arm; replace `shape` with `codex` or `control`
for the comparison arms. All reasoning stages use high effort. `--smoke` is a
technical check on an explicitly prepared small set, not an instruction to use
held-out results to tune. It also exercises the new shared segmentation path.
For a crop-only comparison, reuse the verified V13 masks in a new run through
`ctnt_flowcytometry/runners/run.py crop-from-masks --crop-policy rule-first-balanced`
workflow; do not rerun segmentation or change classifiers
at the same time when attributing crop effects.

Gateway configuration remains external: `LLM_API_KEY` and `LLM_BASE_URL` must be
set for the SHAPE code path. Stanford direct routing and the verified SSH
loopback transport are supported. No credential is embedded in this repository.
The runner performs the model/provider preflight before expensive stages.

## Implementation map

| Module | Responsibility |
|---|---|
| `data.py`, `pilot.py` | Lossless channel preparation, original well splits and fixed pilot identity |
| `references.py`, `fluorescence.py`, `matching.py` | Trusted references, train-only threshold, one-to-one matching and three-channel evaluation crops |
| `e2e.py` | Native stage execution and source-bound completion markers |
| `hybrid.py` | Eligible-plate SSL, separate classifier experts and validation-only routing |
| `score.py`, `readouts.py` | Frozen held-out single-cell scores, coverage and stage QC |
| `progress.py` | Three-arm table plus shared rounds, problem/pass/improvement/rollback counts |
| `run_suite.py` | Sequential technical suite with configured gateway/model; does not stop a VM |
| `export_suite.py` | Explicit result export; does not publish or stop a VM |

SHAPE core owns the algorithms: `segmentation_shared.py` and
`segmentation_shared_native.py` implement conditional cohort VRR;
`crop_cohort.py` and `verifiers/crop.py` implement rule-first crop search and
mixed historical-best delivery. The case adapter `workflows/cohort_cropping.py`
owns the scientific raw-crop boundary, not the VRR algorithm.

## New segmentation policy: conditional_cohort_v2

- Same fixed tool baseline; scan native QC independently for every image.
- Native passing images freeze. One shared session consumes at most eight
  decision slots over the entire cohort, including malformed/interrupted slots.
- Conditional rules may set different parameters by problem, measured morphology
  or explicit exception IDs; only unresolved images run again.
- Native measurements check expected effects. Failed hypotheses stay visible;
  measured gains below pass thresholds are retained in the best-so-far history.
- Large count/coverage losses need a high-confidence comparison of actual shown
  candidates. Accepting normal size heterogeneity requires sufficient raw,
  overlay and extreme-object evidence with no visible actionable faults.
- Native failures are recorded per image. Other images finish; missing outputs
  block a success marker instead of fabricating masks. Fatal provider errors
  propagate promptly; running I/O must still finish its bounded timeout.
- Resume binds exact image bytes, parameters, implementation, mask hashes and
  consumed decision slots. It cannot silently start a new eight-round budget.

`segment/shared_vrr/CHECKPOINT.json` holds all histories and chosen candidates;
`ROUND_*.json` binds visual evidence and proposals; `MEASUREMENT.json` binds
native pixels. `segment_COHORT.json` and `segmentation.json` describe final
selection. `delivery_complete` is not equivalent to zero QC complaints.

## New crop policy: rule first + balanced_v2

- Keep the existing fraction baseline, then measure the existing rule-generated
  geometry candidate before requesting agent tuning.
- Freeze accepted passing crops. Conditional cell rules act on unresolved crops.
- Delivery utility is explicitly `0.5*target_retention +
  0.3*(1-neighbor_contamination) + 0.2*(1-min(centroid_offset,1))`.
  The existing 0.90 native retention floor stays binding. These weights are a
  declared development policy, not fitted to this pilot's held-out labels.
- Below-threshold gains survive through per-crop historical-best delivery.
  Crossing a pass threshold cannot by itself replace a higher-utility history.
- Unchanged images reuse hash-checked metadata; overrides are indexed once per
  cohort rather than rescanned for every image. Unchanged object geometry uses
  a bounded run-local mask/parameter cache. Model decisions are not cached.
- Search measures geometry and writes evidence panels; final delivery writes the
  selected crops and checks each box. Three-channel evaluation stays independent.
- Crop retains its native maximum of eight cohort execution waves (baseline and
  the deterministic rule wave included), rather than eight per image. The small
  smoke uses at least three waves to cover baseline/rule/decision. Model replies
  and retries are recorded separately, so wave count is not a token-call count.

`CROP_VRR_PROGRESS.json` contains each round's problems before/after, improved,
newly passing, passing and reverted counts. `crop_COHORT.json` contains final
readouts and a `CROP_TRAJECTORY.jsonl` pointer with each crop's parameters,
geometry, native metrics and selected source. The optional semantic mask-error
screen remains recorded and affects actual exclusion decisions.

## Hybrid reporting

`SSL_SELECTION.json` now explicitly reports `plate_hybrid`, both experts and
actual adopted plates. `CLASSIFIER.json.validation` is recomputed from the
routed validation predictions. There is no fictitious shared threshold: the
aggregate threshold is null and `thresholds_by_expert` records each expert's
value. Held-out accuracy still uses each cell's frozen actual decision. Training
eligibility, routing criteria, seeds and fluorescence thresholds are unchanged.

## Validation scope

Offline regressions exercise real CPU crop geometry/materialization and mocked
native segmentation/model execution, including failure isolation, budget and
resume guards, evidence binding, mixed-threshold scoring, fixed splits and
channel isolation. These checks do not establish improved GPU throughput or
biological prediction accuracy. Next GPU acceptance should first use a small
smoke; compare crop variants on the same V13 masks and frozen downstream setup.

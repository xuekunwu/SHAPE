# cTnT flow-cytometry experiments

`run.py` is the single public Python entrypoint. Run it from any directory using
a quoted path. Internal modules have one maintained location; flat compatibility
files and the old `operations/` and `pipeline/` directories have been removed.

All directories use the same criterion: what the code **does**. A new experiment
is a workflow configuration, not a new top-level category.

| Directory | Owns | Must not own |
| --- | --- | --- |
| `dataset/` | Raw import, metadata contract, well splits and subset sampling | Model fitting, process launch, endpoint scoring |
| `prediction/` | Fitting well predictors and selecting/merging independent experts from validation | Run scheduling, native SSL training, held-out label access |
| `workflows/` | Every scientific run: ordinary, HYBRID, recovery, fixed-mask crop, development QA and smoke | Copies of native SHAPE algorithms or generic launchers |
| `runtime/` | Artifact IO, source snapshots, process launch/proofs, progress and implementation inventory | Dataset preparation, scientific stage order, model fitting, scoring |
| `evaluation/` | Metrics, frozen scoring, stage readouts, arm comparison and result export/replay | Producing or changing model predictions |
| `test/`, `docs/` | Regression checks and execution/reference documentation | Alternate runtime implementations |

There is no `execution/`, `experiments/` or `reports/` namespace. Ordinary and
experimental workflows have the same owner. Current evaluation and historical
analysis share one owner; `evaluation/historical/` limits version-specific schemas.

```bash
python "case studies/cardiomyocytes/ctnt_flowcytometry/runners/run.py" --help
python "case studies/cardiomyocytes/ctnt_flowcytometry/runners/run.py" scripts
python "case studies/cardiomyocytes/ctnt_flowcytometry/runners/run.py" results
```

The hybrid regression and the crop trajectory selection are the scientific variants.

## Fixed scientific protocol

The full dataset has 600 brightfield images / 87 wells. Seed `20261015` fixes
35 training / 9 validation / 43 held-out wells (40/10/50, balanced by date/plate).
Pilot seed `20261016` samples within those frozen assignments: 18 wells / 114
images (7 train / 2 validation / 9 held-out wells). Sampling never changes splits.
The frozen historical pilot manifest retains its original descriptive model field;
the executed model is bound and checked in each arm's CONFIG and preflight.

The endpoint is well-level `ctnt_positive_percent`. Fit scaling/ridge on train,
select on validation, freeze predictions, then evaluate held-out wells. Existing
MAE/RMSE/bias/R2 and published Acc@5pp remain unchanged. Stage QC is a proxy,
not annotated segmentation accuracy. HYBRID eligibility/routing and separate
expert heads remain unchanged. Algorithmic VRR remains in upstream `shape/`.

Tracked SHAPE/CODEX bindings are `gpt-6-sol` with `all-high` reasoning. Published
Luna runs and the active staining VM use their own recorded source patches;
this cleanup does not silently import those scientific changes. The full-well
`run --smoke` remains about 37 images. `smoke-image-stages` is a different,
five-image CONTROL segmentation/crop integrity check with no prediction claims.

Every new experiment must record source/data/split identities, requested lock
and actual packages, GPU chip/count/VRAM, CPU/RAM, CUDA/driver/Python, model/effort
and per-stage time/token accounting. Keep actual hardware in its result README.
Use the same source snapshot across arms, and never publish credentials.

# full_LIVECell

The full SHAPE pipeline (segment, crop, embed, batch, cluster) on LIVECell, a label-free phase-contrast dataset
with instance ground truth, eight cell lines. The harness is full_CPJUMP1's, run with the LIVECell dataset spec
(`harness/spec.py`). This file is the summary and the run guide.

## Dataset contract

- Source: the LIVECell S3 release, `images.zip` plus the 24 single-cell COCO files. CC BY-NC 4.0.
- Pins:
  - `images.zip`: 1,241,715,651 bytes, SHA256 `548a2c28...`, checked in `prepare` (`data.IMAGES_ZIP_SHA256`).
  - The annotation files: by SHA256 in `manifest.json`.
  - The manifest itself: `data.MANIFEST_SHA256`.
- Selection: all 655 units, a unit being (type, well, time point). Both locations of each unit are kept, one seeded
  crop each, giving 1310 images of 520 x 704 single-channel uint8.
- Prepared tree:
  - `public/images` (CYX, C = 1, channel `Phase`) and `public/segmentation` (YX, the same pixels).
  - `public/images.json`, which holds only anonymous fields.
  - `public/smoke.json`.
  - `private/labels.json` and `private/gt/*.npz`, read by the scorer only.
  - `PREPARE.json`, which records the hashes and the tifffile / numpy versions.

## Splits, batch and metrics

- Each type's own time course is cut into six stages. The `usage` of a unit is `train`, `val` or `heldout`.
  Stages 0-1 are `train` and `val`: one well per type is `val`, the rest `train`. Stages 2-5 are `heldout`.
- The scorer alone derives two pools of the heldout units from `stage` (no field carries them): `heldout_fit`
  (stages 2-3) and `heldout_test` (stages 4-5).
- Segment and crop use every image; they have no train / val / heldout split.
- SSL trains on `train`, validates and picks the best checkpoint on `val`, at most 128 cells per train or val site,
  then freezes and embeds `heldout` with no further tuning.
- Unit profiles: the per-site median cell, then the mean of the unit's two sites.
- Batch key = stage (`plate`). The correction is transductive (all profiles, `heldout` included). The calibration
  view that steers the search holds the `train` and `val` units only, as opaque per-type ids (`calibration_group`
  is empty on `heldout` units); `heldout` identities and labels never enter it.
  - There are no negative controls, and all three soft admission concerns hold by construction (DESIGN section 4).
  - Each arm's batch decision is reported next to the verdicts.
- Verdicts: the shared `full_metrics` (mAP, macro F1, balanced accuracy) on cell type.
  - Main split: one single-shot fold, cross-time: the probe fits on `heldout_fit` and is scored once on
    `heldout_test`; the reverse direction is not scored.
  - Guard split: cross-time and cross-well: per well index k one part (test = the `heldout_test` units of well
    index k, train = the `heldout_fit` units of the other well indices); the parts' test predictions are pooled
    into the fold's confusion matrix.
- Reported, not verdicts:
  - retrieval secondaries (cross-time and cross-time cross-well: queries `heldout_test`, candidates `heldout_fit`);
  - segmentation AP50 / mAP / F1 / count ratio (all images, per usage `train` / `val` / `heldout`, per type, per
    stage, official test wells);
  - crop retention / contamination / full cover / unmatched / GT recall, each with a paired bootstrap against the
    control arm.

## Arms

- control, SHAPE and Codex, exactly as in full_CPJUMP1: at most 8 tuning rounds per image per stage, model
  `gpt-5.6-luna`, the same rules.
- SHAPE and Codex get the same stage texts, which name no per-image cell type.
- `cells.json` records each cell's delivered mask instance (`mask_label`) for the crop GT metrics.

## Information boundary

- Public fields, file names and id order carry no cell type, real well, absolute time or official split. Only the
  relative stage is public.
- The runtime reads `public/` only: the smoke comes from `public/smoke.json`.
- Private labels are read by the scorer after delivery and by the VM admission hash checks.
- This is a programmatic boundary, not an OS sandbox.

## Run

One GPU. The environment and the model assets are full_CPJUMP1's (its `assets` command builds them).

```bash
C="python -m benchmarks.full_LIVECell.harness.cli"          # --help on any subcommand
COMMIT=$(git rev-parse HEAD)
$C select --raw RAW --manifest RAW/../manifest.json         # the manifest of the downloaded files
$C prepare --raw RAW --out PREP                             # the prepared tree + SOURCE.json
$C verify-prepared --prep PREP --out RECEIPT                # a transferred prepared tree, without rebuilding it
$C preflight --arm shape --expected-commit $COMMIT --assets-root ASSETS --out EXPERIMENT
$C run --arm shape --prep PREP --assets-root ASSETS --out EXPERIMENT \
   --expected-commit $COMMIT [--smoke] [--budget-hours 11] [--cores N]
$C score --prep PREP --run EXPERIMENT/shape
$C compare --prep PREP --control EXPERIMENT/control --shape EXPERIMENT/shape \
   --codex EXPERIMENT/codex --out EXPERIMENT/comparison --expected-commit $COMMIT
```

`stage` is the internal per-stage worker a launcher calls. Run `--help` on any subcommand for its exact flags.

- `prepare` reads a mirrored LIVECell S3 tree and writes `raw/`, `manifest.json` and `prepared/` beside it.
- A full run requires a complete three-arm smoke of the same commit.
- Runs go to `<runs root>/full_LIVECell/<mode>_<commit8>_<utc>_<rand>/`; each experiment also gets
  `<experiment>.noimg.tgz` (json/csv/md/log/txt/out/err).
- Arms run one after another, each with its own full budget. Cores default to every core of the machine and are
  recorded in CONFIG.json.
- `cli rerun --arm codex --from-stage crop` continues an arm that failed after its segment stage (hash-verified
  reuse, `RERUN.json`); `cli compare-rerun` is the post-hoc comparison. The source run is only read.
- The Claude Code and Biomni arms share one prompt and its size protections with Codex (`harness/codex_arm.py`);
  only the whole-prompt guard differs per arm (900k characters; Biomni 800k).

A full run prints and writes `LIVECELL_COMPLETE`; a smoke prints `SMOKE_COMPLETE` only.

## Tests

```bash
python -m pytest benchmarks/full_CPJUMP1/runners/test benchmarks/full_LIVECell/runners/test benchmarks/_shared -q
```

CPU tests do not prove real-data quality, Harmony / scib-metrics behaviour (stand-ins locally) or GPU throughput;
the GPU smoke comes before the full run.

## Post-hoc metrics and how to call them

Every post-hoc call below reads a delivered, scored run (never writes into it); none runs an arm and none changes a
verdict. Layers and metric ids are those of
[`../_shared/AGENTIC_EVALUATION_FRAMEWORK.md`](../_shared/AGENTIC_EVALUATION_FRAMEWORK.md). Put the repo on `PYTHONPATH`
and use the benchmark environment (`~/shape/cache/envs/shape_bench` on the VM).

| tool | layer / metric ids | what it produces |
| --- | --- | --- |
| `runners/extra_readouts.py` (one arm; needs `--prep`, the private labels and GT) | L1 O3 (per-class precision / recall / F1, confusion), O4 (AUROC / AUPRC), O5 (batch integration, stage leakage), O6 (cell morphology regression) | `extra_readouts_<arm>.json` and `.md` |
| `runners/process_readouts.py` (entry over `../_shared/agentic_process_readouts.py`) | L4 P1-P4, L5 E1-E3, E6 | `analysis/process_readouts.json`, `.csv`, `.md`: per arm run and stage who delivered, rejected / malformed / failed rounds, rounds used, stop reasons, SHAPE rule activations, tokens, seconds, calls, null + reason for every unavailable signal |

```bash
python benchmarks/full_LIVECell/runners/extra_readouts.py --prep ~/shape/data/LIVECell/prepared --run <experiment dir>/<arm> --out <out>/extra_readouts_<arm>.json --md [--workers N]
python benchmarks/full_LIVECell/runners/process_readouts.py --run <experiment dir> [--out OUT.json] [--csv OUT.csv] [--md]
bash benchmarks/full_LIVECell/runners/run_post_hoc_metrics.sh --run <experiment dir> [--out DIR] [--prep DIR] [--arms "control shape codex"]
```

`run_post_hoc_metrics.sh` runs `extra_readouts.py` for each of control, shape and codex (an arm whose run dir is absent
or lacks `LIVECELL_COMPLETE` / `RESULT.json` is skipped and the skip is printed and logged), then
`process_readouts.py`, and appends each step's exact command line, the commit it ran at and the UTC time to
`<out>/analysis/COMMANDS.txt` (default `<experiment dir>/analysis/`). The segmentation and crop stage outcomes against
the expert annotations (L2) are computed by the scorer at delivery, not post hoc.

# full_CPJUMP1

End-to-end benchmark for the fixed CPJUMP1 A549 Cas9 CRISPR subset on **e2e**.
Uses the native SHAPE tools and shared `AgenticVerifier` / `StepVerifier` loops.
HCMA's segmentation/crop adapters, model asset pinning and fail-closed run discipline
are reused; its culture-class scoring, metadata assumptions and SSL CLI are not.

## Dataset Contract

- Source: `cpg0000-jump-pilot/source_4`, acquisition `2020_11_04_CPJUMP1`.
- A549 Cas9, CRISPR, 144 h; plates BR00117000 and BR00117004.
- 130 genes, 242 distinct treatment guides; 112 dual-guide and 18 single-guide genes.
- 242 treatment wells + 8 NO_SITE negative-control wells per plate; two sites per well.
- **1000 multichannel views = 5000 channel TIFFs**, not 1000 individual TIFFs.
- Channels in order: **Mito, AGP, RNA, ER, DNA**; brightfield channels are not included.
- Manifest SHA256: `7cf7a28b12f703a152da4f2445e3a72aa62ed9078012e227e398ba4f92cadd31`.

`prepare` consumes `DATA/manifest.json` and `DATA/raw/<relative_path>` from the
existing downloader. It requires all 5000 files, decodes them, checks uint16 and
1080 x 1080 dimensions, and verifies lossless stack round-trips. It writes new
CYX OME-TIFFs, never modifies raw data, and records hashes of every source file.
The manifest is supplied with the downloaded data, not fetched from a moving URL.

### Reuse Existing Five-Channel Stacks

`prepare --data RAW_ROOT --merged-dir MERGED_ROOT --out PREP` also accepts the
existing `_5ch` export: `fields.csv` with columns `output,shape,dtype,ch1,...,ch5`,
and `COMPLETE` declaring `fields=1000`, `axes=CYX`, `channels=1,2,3,4,5`,
`pixels=verified`. The original pinned `manifest.json` and raw channels are still
required for identity and pixel verification. `--data` remains the original root,
not the `_5ch` directory.

The five source paths identify each manifest site; CSV row order and output
filenames do not assign labels. The importer rejects missing/duplicate/mixed
sites, paths outside the declared roots, wrong axes/dtype, and altered pixels or
channel order. It copies each already-merged TIFF byte-for-byte to an anonymous
path (independent files, no mutable hardlinks/symlinks) and creates only the
DNA+AGP segmentation view. It does not merge the channels again or edit either
source tree. Allow about 15 GiB for prepared pixels, plus model assets and runs.
Source paths stay in private provenance; `PREPARE.json` records the import mode
and index hashes.

Public image/well/plate IDs are anonymized. Gene symbols and guide sequences live
only in `prepared/private/labels.json`, read by the scorer after delivery. This
is a programmatic information boundary, not an OS security sandbox.

## Pipeline

1. **Segment:** a separate uint16 DNA + AGP view feeds Cellpose-SAM, so the model
   cannot silently truncate the five-channel input before DNA. Masks retain the
   raw image coordinates. A full SHAPE run segments through SHAPE's shared
   segmenter cohort (`shape.agent.verification.segmentation_cohort.run`, the one
   `solve()` uses; `shape/tools/cell_segmenter/CARD.md`, "Cohort mode"): images
   are grouped by their preflight read; per group a progressive calibration
   sample (~28 of 1000) runs the unchanged per-image VRR; the consensus of the
   agent-tuned levers (never `diameter` or `color_class_filter`, set per image)
   is applied to the rest in the verifier's cohort mode; only images the
   admission standard flags get their own VRR from the consensus. Groups run at
   the same time, `--cores` images at once. Records: `segment_COHORT.json`,
   `segment/<id>/SEARCH.json` (own VRR), `segment/<id>/COHORT_SEARCH.json`
   (cohort mode), `parameter_source` in `segmentation.json`.
2. **Crop:** no cohort: every image runs its own crop VRR (with its delivered-crop
   mask check), `--cores` images at a time with the cores divided among them.
   The native cropper makes display-normalized uint8 crops. This harness applies
   its final, verified crop boxes to the original five-channel uint16 stack and
   writes lossless scientific crops for SSL. Rejected cell IDs never reach SSL.
   `cell_mask_*` naming preserves the cropper's cell-specific default margin.
3. **Representation:** shared `RepresentationAgenticVerifier`, including multi-epoch
   trials and parent-checkpoint continuation. Only development wells enter trial 0,
   fitting, validation or parameter selection. A deterministic maximum of **128
   cells per development site** bounds SSL fitting cost; evaluation is not sampled.
   Every arm's training calls run a fixed epoch count (`pipeline.ssl_epoch_budget`, 2026-10-12, as
   representation_RxRx1): the analyzer's automatic rule on 128 x the development_train sites. `max_epochs` is not
   a lever: SHAPE uses representation_RxRx1's `FixedEpochVerifier`, and Codex's parameter table leaves it out.
   The selected `resume_best` checkpoint is frozen, then every retained cell is
   embedded in balanced chunks of at most `--inference-chunk` (job script: FULL_CPJUMP1_INFERENCE_CHUNK, default
   2048; the same for every arm, since compare matches it). A legitimate gate SKIP uses the
   frozen pretrained baseline; a missing trained checkpoint is an error.
4. **Aggregate:** median cell embeddings within each site, then equal-weight mean
   across its two sites. Batch correction operates on **500 well profiles**, not
   hundreds of thousands of correlated single cells.
5. **Batch:** shared `BatchCorrectionVerifier` with opaque calibration guide IDs
   and NO_SITE controls. The batch key is **plate**, not the constant acquisition
   batch. Joint unlabeled correction of all wells is explicitly **transductive**.
   The metric API's `DMSO` sentinel represents NO_SITE controls here; no DMSO
   treatment is asserted. RUN / SKIP / ABSTAIN gates retain shared semantics.
6. **Cluster:** the analyzer's existing UMAP + Leiden implementation on delivered
   well profiles, fixed resolution 0.5 / 15 neighbors. These are descriptive
   outputs, not ground-truth cell-state classifications or tuning objectives.
7. **Score:** after delivery hashes are frozen, load private gene labels.

The SHAPE arm uses all four shared admission rules. `SHAPE_SEARCH_STRATEGY` and
the existing per-tool overrides select `legacy` or `four_stage`; the sbatch
default is `four_stage`. The shared VRR state machine and tool rules are unchanged:
the benchmark adds only bounded image-level concurrency; segmentation's cohort
calibration and routing are SHAPE's shared ones. Smoke keeps the original direct path.

## Splits And Endpoints

- Development: 20 dual-guide genes, gene-disjoint from 110 evaluation genes.
- SSL: `development_train` -> train, `development_val` -> val;
  `evaluation_only` is **excluded from all search calls**, not relabeled as train.
- Batch objective: 40 development guide identities plus negative controls only.
- Primary: gene-macro mAP, MRR and top-1 for **different guide, different plate**
  retrieval on the **92 held-out dual-guide genes / 368 queries**. Other held-out
  genes remain distractors. Same-guide candidates are excluded.
- Secondary: same-guide cross-plate retrieval, 110 genes / 404 queries. This can
  benefit from repeated well-position or guide-specific effects and is not an
  independent-guide biological result.
- Both uncorrected and delivered profiles are scored using raw feature cosine.
  Ties are pessimistically ranked; collapsed vectors cannot win by row order.
- `gene_similarity_cosine.csv`: 130 x 130 signed cosine, with equal guide/plate
  weighting. `genes.csv` flags calibration and single-guide genes. This matrix is
  descriptive; validation of *different-gene* similarity needs external biological
  annotations. No claim of reproducing the paper's Fig. 2 UMAP is made.

Every selected site must retain cells and every selected well must be present.
Missing views, empty masks, rejected-all-cell sites, changed hashes, fallback
rounds or failed searches invalidate the run; basic LLM failures (engine, auth,
model, connection) stop it. Other failed in-run calls (a rejected search plan, a
failed repair or vision question) are recorded per image in `SEARCH_WARNINGS.jsonl`,
not fatal. Nothing is silently dropped to improve a metric. Quality gate SKIP is not an error.

## Arms And Outputs

Implemented arms: **control**, **SHAPE**, and **Codex**. Control uses default segmentation and
crop settings, frozen DINOv3 embeddings, and one default batch correction when
the shared rule admits it. All three arms use identical data, channels and scoring.
Each arm has its own allocation and full budget, not leftover time in a sequence.

SHAPE and Codex both request the exact model ID **`gpt-5.6-luna`**. The CLI and
comparison reject other IDs, including aliases. Positive token usage establishes
usage, not the identity of the served model. Missing model telemetry must remain
explicitly unverified; the requested ID alone is not independent attestation.

Codex uses the real headless CLI as a control arm under the same-task rule
SHAPE's stage task wording, the same inputs, the lever
set SHAPE's verifiers set (each CARD's `## Levers`, crop `cell_overrides`
included) with the CARD section SHAPE's agent sees, and SHAPE's budget of at most
**8 tool-call rounds per image per stage** (segment/crop: the default call `c0`
is round 1; embed/batch: trial 0 `input` is not a round), each round holding at
most the calls SHAPE's verifier allows per round (its `concurrency()`). Embed
training calls use SHAPE's search deadline (0.65 of the remaining time) and call
time cap. The prompt states facts only,
plus "Tuning the tool's parameters can improve the tool's performance."

Each decision names its scope: all images or a subset, one parameter set or
per-image parameters. Only named images re-run, in one trusted native worker,
in parallel over `--cores`. A request that would exceed either limit for any image
is rejected as a whole, recorded and reported to the agent. Codex chooses group
mode (one session, one compact row of the tool's own scalars per image) or hands
images to per-image mode (one ephemeral session per image, at most 8 rounds,
`--cores` sessions concurrently). `inspect` attaches raw inputs or tool-written
files to the next round; it costs a round, not a call. Codex sees only what the
tools return and write: no SHAPE readouts, admission problems, thresholds or
harness panels. `finish` names a completed call per image; unnamed images get
`c0`, recorded as `default_unnamed`; a reply that is not a valid decision is
handled as a rejected request (recorded as `malformed`, its error shown in the next
round, the session goes on); a session that ends without a valid finish delivers its
images' default call, recorded as `default_session_unfinished`. Only basic LLM failures stop the arm. Each stage is delivered before the next
starts; there is no cross-stage backtracking. Clustering is fixed in all arms.

The harness validates parameters, image/call identities and hashes before the
trusted worker runs or delivers anything. Private labels are reserved for scoring
after delivery. These protocol boundaries do not establish OS isolation; report
the actual CLI sandbox and tool policy with results.

SSL starts with development-only zero-shot `input`; no measurability gate is
applied to Codex. Calls can resume a finished call's best checkpoint
(`init_from` parent). Only after explicit final selection does a fresh worker
embed all cells with the frozen checkpoint. Batch keeps the shared batch rule as
a hard rule (SKIP/ABSTAIN reject correction requests); Codex sees the rule's
decision and the corrector's own outputs. SHAPE's four verifier loops are
unchanged. No minimum search count is forced.

The three arms (control, SHAPE, Codex) run against the same prepared tree, and the
comparison runs after all three succeed. `cli compare` rejects
partial/smoke/missing arms and mismatched commit, data, assets, seed, caps, model
or budget. It verifies deliveries and recomputes scores, then writes
`comparison.csv`, `COMPARISON.json`, and `COMPARISON_COMPLETE`. Different GPU
models are prominently flagged: they confound fixed-time search and timing.
The historical `claude_arm.py` module and its tests remain as legacy code.

Two more agent arms, **claude** (Claude Code, `--model claude-sonnet-5`) and **biomni**
(Biomni A1, `gpt-5.6-luna`, decision-only), share the Codex arm's stage protocol in
`codex_arm.py`; only the backend answering each round differs (`claude_backend.py`,
`biomni_backend.py` with `biomni_driver.py`). Their native workers
take the same agent-arm path as Codex. `cli compare` takes them as optional `--claude` /
`--biomni` runs, accepts a run at another commit only if every shared code path has the
same git object id at both commits, and reports every arm against control and SHAPE
against each agent arm.

Per-run records: `PREFLIGHT.json`, `CONFIG.json`, `RUN_META.json`, stage logs,
VRR search records, `cells.json`, `excluded_cells.json`, `SSL_SELECTION.json`,
`embedding.npy`, `profiles.npy`, `profiles.csv`, `corrected.npy`, `profiles.h5ad`,
`states.csv`, `umap.csv`, `DELIVERY.json`, `RESULT.json`, and `evaluation/`.
Heavy stages use separate processes to release GPU memory between stages.

Successful full runs print `CPJUMP1_COMPLETE` and create the same-named marker.
The 20-view whole-well smoke prints **SMOKE_COMPLETE only**, never a full result.
It includes two validation wells per plate so SSL's measurability gate
can be exercised; an admitted search still is not forced if quality already suffices.
Partial runs retain diagnostics but cannot resume or overwrite an existing run;
use a new output root. SSL search continuation within a run is supported.

A post-hoc partial rerun (`cli rerun --source <complete SHAPE run> --from-stage embed
--arm shape ...`, `harness/rerun.py`) reruns embed, batch and cluster with the current
code on a completed run's segment and crop outputs. It refuses a source that is not a
complete full run of the arm on the same preparation, and never writes into it: the
reused JSON records are copied, the masks and crops hard-linked (copied where a link is
impossible, never symlinked), every file hash-checked, and the paths in
`segmentation.json` and `cells.json` rewritten to the copies (`RERUN.json` records the
mapping and every hash). `CONFIG.json` / `RUN_META.json` carry `rerun` (source run,
commit and config hash, reused and rerun stages, reason); the run is delivered and
scored as a full run (`--control`: its verdicts) and marked `RERUN_COMPLETE`, never
`CPJUMP1_COMPLETE`, so `cli compare` refuses it. `cli compare-rerun --rerun ... --control
... [--arm codex=DIR ...]` verifies every delivery, rescores each run into its own
output against the same control draws, and writes `POSTHOC_COMPARISON.json` / `.md`:
per run the three verdicts, the retrieval endpoints, per-stage commit, reasoning,
seconds and tokens, and the rerun against its source arm. It is not the strict
comparison: the runs differ in code and reasoning settings.

## Running And Tests

The raw download may run independently; prepare only after it finishes.

```bash
python -m benchmarks.full_CPJUMP1.harness.cli --help
python -m pytest benchmarks/full_CPJUMP1/runners/test -q
```

CPU tests include actual five-channel OME loading and native cropper execution,
split isolation, rejection propagation, equal-site aggregation, checkpoint
inference wiring, fixed-design scoring and artifact tamper rejection. They do
not establish real-data segmentation quality, GPU throughput or LLM reliability;
an actual GPU smoke run is required before a full run.

## Post-hoc metrics and how to call them

Every post-hoc call below reads the records of an existing experiment directory (or its curated results folder); none
runs an arm, none reads ground truth, none changes a verdict. Layers and metric ids are those of
[`../_shared/AGENTIC_EVALUATION_FRAMEWORK.md`](../_shared/AGENTIC_EVALUATION_FRAMEWORK.md). Put the repo on
`PYTHONPATH` (`export PYTHONPATH=$PWD`) and use the benchmark environment (`~/shape/cache/envs/shape_bench` on the VM).

| tool | layer / metric ids | what it produces |
| --- | --- | --- |
| `export_run_records.py` (this folder) | the record set the others read | `results/<experiment_id>/`: curated text-only copy, `summary.csv`, `MANIFEST.json` |
| `runners/pairwise_deltas.py` | L1 O1 (paired differences) | `analysis/pairwise_deltas.json` and `.md`: paired delta, SE, 95% interval, guard delta and verdict of run A against run B, on the saved bootstrap draws |
| `runners/process_readouts.py` (entry over `../_shared/agentic_process_readouts.py`) | L4 P1-P4, L5 E1-E3, E6 | `analysis/process_readouts.json`, `.csv`, `.md`: per arm run and stage who delivered, rejected / malformed / failed rounds, rounds used, stop reasons, SHAPE rule activations, tokens, seconds, calls, with null + reason for every unavailable signal |

Per-class retrieval and classification readouts (O3-O5) do not exist for CPJUMP1 yet (framework: TODO).

```bash
python benchmarks/full_CPJUMP1/export_run_records.py --run <experiment dir> --out benchmarks/full_CPJUMP1/results/<experiment_id> [--extra-log <driver log>]...
python benchmarks/full_CPJUMP1/export_run_records.py --check benchmarks/full_CPJUMP1/results/<experiment_id>
python benchmarks/full_CPJUMP1/runners/pairwise_deltas.py --run <experiment dir> --pair A:B [--pair C:D]... --out <results>/analysis/pairwise_deltas.json
python benchmarks/full_CPJUMP1/runners/process_readouts.py --run <experiment dir or results folder> [--out OUT.json] [--csv OUT.csv] [--md]
bash benchmarks/full_CPJUMP1/runners/run_post_hoc_metrics.sh --run <experiment dir> [--out <results>] [--pair A:B]... [--extra-log F]... [--skip-export]
```

`run_post_hoc_metrics.sh` runs the first four calls in this order (the pairwise step only with `--pair`, or `PAIRS`
in the environment; `--skip-export` when the results folder already exists) and appends each step's exact command
line, the commit it ran at and the UTC time to `<results>/analysis/COMMANDS.txt`, so the results folder carries the
calls that produced it. `process_readouts.py` writes `<run>/analysis/process_readouts.json` by default; the exporter's
`--check` ignores `analysis/`. It reads per-image SHAPE records when the experiment dir has them and
`<arm>/search_summary.json` otherwise (a curated folder), so the same call works on both.

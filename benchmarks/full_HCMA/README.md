# full_HCMA -- end-to-end single-cell morphological states on HCMA

An end-to-end benchmark on the Human Cell Morphology Atlas (HCMA: 1,205 raw images from 4
sources). Each arm starts from the raw images and must deliver, for every cell, a single-cell
embedding and a morphological cell state. Every arm uses the same pipeline and the same tool
code; the arms differ in who decides the tools' parameters and in what evidence the decider sees
(a SHAPE arm adds its verifiers' measurements; see Arms).

## The pipeline (SHAPE's logic)

| step | tool (SHAPE class) | what it does | what SHAPE's VRR adds |
| --- | --- | --- | --- |
| 1 segment | Cellpose-SAM (`Cell_Segmenter_Tool`) | one instance mask per image | per-image admission at the defaults, then agentic retuning of the CARD levers (scale, detection thresholds, colour-class filter) |
| 2 crop | MaskCrop (`Single_Cell_Cropper_Tool`) | one square single-cell crop per mask object, cut from real pixels | per-image geometry levers and per-cell rules (`cell_overrides`) |
| 3 embed | DINOv3 ViT-B/16 (`Cell_State_Analyzer_Tool`) | one embedding per crop (zero-shot, head-only or fine-tuned) | `RepresentationAgenticVerifier` (a benchmark runs it; not in `stage_map`): trial 0 zero-shot, a measurability gate (the replicate tier of `quant.tier_of`) before any search; the linear `RepresentationVerifier` stays in solve()'s path |
| 4 batch_correct | Harmony (`Batch_Corrector_Tool`) | SHAPE's calibration rule: it corrects only when images.csv gives label-free reference groups (`group` / `treatment`, `control`) replicated across >= 2 batches of the batch key, then makes one tool call at the tool's defaults (or the set levers; the tool's own gate decides); otherwise it returns the input unchanged (HCMA has no such groups) | none in the CLI: SHAPE's agentic batch loop needs a reasoner |
| 5 cluster | Leiden (the analyzer's own clustering code) | morphological cell states | today a linear check (`MorphologicalStateVerifier`: a deterministic stability rule over resolutions) |

Agentic VRR (`AgenticVerifier`) exists today for steps 1-4; step 5 has a linear verifier. SHAPE's claim is not that it runs this chain; it is that the VRR loop raises every step's
performance by adapting to each input, with whatever metadata exists (full, partial or none).
HCMA's own metadata is uneven across sources (`magnification` exists only for some sources). With
`METADATA=full` (default) every arm gets **all known technical metadata** and has to cope with its
gaps; `METADATA=none` gives no metadata at all.

## Arms

| arm | who decides | status |
| --- | --- | --- |
| `control` | nobody, the reference arm: the default-parameter workflow (DINOv3 zero-shot), i.e. the five tools at their default parameters, except that embed is zero-shot (`adaptation_depth=zero_shot`; user decision 2026-10-03), in a fixed chain (`batch_correct` with batch key `source`). The earlier fine-tune control (the analyzer's own default) is kept on the cluster as a reference; the comparison gate only accepts a zero-shot control | built |
| `claude` | Claude Code (headless, Stanford AI Gateway): the tools, `TASK.md` and run advice, nothing else of SHAPE | built |
| `shape` | SHAPE's own VRR loops (`harness/shape_arm.py`, reasoner `SHAPE_LLM_MODEL`) on the same prep and tools: `segment` defaults, then per image `SegmentationAgenticVerifier` (its call at the defaults = that S001 result) and one `segment --per-image` call for the changed images; per image `CropAgenticVerifier` (the CLI's defaults, the margin by the tool's rule; arm C's vision triage) and one `crop --masks-map --per-image --cell-overrides` call, triage rejects as `embed --exclude-cells`; `embed` zero-shot as `RepresentationAgenticVerifier`'s trial 0 (site = image_id, well = parent_id, plate = source; on HCMA the gate SKIPs: the batch tier, no micrograph-replicated sites on `pairs`, 289 / 916 micrographs on the full atlas); `batch_correct --batch-key source` (the calibration rule); `MorphologicalStateVerifier`'s clustering as `cluster --set resolution n_neighbors`. Every delivered call goes through the tool CLIs (`logs/control_commands.jsonl`), so `score --run`, candidates, process metrics and `attribute` read it as the control; `SHAPE_RUN.json` records each stage's decisions, the CLI-vs-in-process reproduction checks and the reasoner usage. The reasoner is checked by two smoke calls (a plain one and one structured question); claude-* and gpt-* both go through the OpenAI-compatible gateway when `LLM_BASE_URL` is set. Only a fatal LLM error (no engine, authentication / permission, the model not found, the endpoint unreachable; classified by exception type) stops the run with `SHAPE_FAILED`: the images not yet started are not started, and `SHAPE_RUN.json` (status `failed`, `stopped`: the stage, the image, the question label, the cause) and the log name it. Every other failure is recorded and the run goes on: a failed round question ends that image's search, a failed final question leaves the image at the stage's input (the defaults), a failed admission question leaves it unsearched (each in the outcome's `execution_errors`), a failed vision call is a failed call the agent sees, a search that ends in another error is the image's outcome `failed`; `process` counts them (`F3_failures.shape_llm`) | built |

What differs between the arms besides the decider (to keep in mind when a SHAPE arm is added):

- The Claude Code arm reads the score definition, the control's chain, the image-set design
  (`{{DESIGN}}`: the number of cell types, 2 images per type) and each CARD's lever text verbatim;
  it can view images and figures and write its own analysis. It is bound by the deadline, the
  lever-coverage rule and the literal-command rule.
- SHAPE's verifiers see measurements the tool CLIs do not return (segmentation QC readouts and
  reference calls, per-cell crop input measurements) and run per image without a wall clock.
- For a fair comparison both arms should use the same gateway model, and the SHAPE arm should
  either run through these tool CLIs (so its calls and excluded cells are in the ledger) or report
  its excluded cells per image: `score --h5ad` cannot see cells a SHAPE cropper rejected.

## Endpoint (pinned 2026-10-05)

- **Primary**: `knn_transfer_macro_f1`, the cross-micrograph DINO-weighted kNN label transfer (k 20, T 0.07) over the 7 morph classes, on all expected cells, misses counted.
- **Hard guard**: `knn_transfer_cross_source_macro_f1`, the same vote with neighbours from another source only; a primary gain beyond 2 SE counts as `GAIN` only when the guard's delta is >= 0, else `SOURCE_SHORTCUT_SUSPECTED`.
- Aligned to the task goal: it asks whether a cell's embedding predicts its culture's lineage class, which is what the pipeline is for.
- Final labels only: it reads the HCMA morph classes of the expected images, never the agent's own proxies.
- Shortcut-resistant: neighbours come from other micrographs, and the cross-source guard removes the source shortcut.
- Insensitive to clustering choices: it reads the delivered embedding, not the states or the resolution.
- Misses count: excluded cells and images without cells are recall misses, so delivering less cannot raise it.
- Literature precedent, fixed before further arms: DINO's weighted kNN (Caron et al. 2021) with NSCB-style partner rules (Ando et al. 2017), pinned before the SHAPE arm and further runs.

## Protocol

| item | choice |
| --- | --- |
| data | HCMA (`$HCMA_ROOT/images`, `HCMA_metadata.csv`, 1,205 images). Runs use a prep: `pairs` (default, `PREP=pairs`): for every `cell_state` type with >= 2 independent micrographs, 2 images from 2 different micrographs, and from 2 different sources where the type has more than one (seeded; `harness subset`); `full` (`PREP=full`): all 1,205 images |
| anonymization | `prepare` renames every image to a random id (`img_0001` ...): the original file names contain the vendor and the cell line. Images are copied (no symlinks) with their metadata segments removed losslessly; the decoded pixels are checked identical |
| agent sees | the images and `images.csv`: `image_id, file, parent_id` (anonymized micrograph id), `source`, `image_modality`, `magnification`, image size. `METADATA=full` (default) gives every known technical column; `METADATA=none` empties `parent_id, source, image_modality, magnification` in the workspace copy (no metadata at all), to test whether a decider adapts without metadata |
| withheld | `cell_name`, `cell_type`, `organ_category`, `cell_system`, `disease`, `morphology`, `growth_properties`, `source_catalog_id`, the original ids and names (`$ROOT/private`) |
| tools | `workspace_template/tools/`: thin CLIs around the SHAPE tool classes; the lever surface of each is the one SHAPE's loop tunes (the CARD's `## Levers`, read at run time; embed and cluster from their registries); `--describe` prints the CARD lever text verbatim. A tool reports what its SHAPE tool returns, nothing more |
| shared environment | the `shape_bench` conda env; pinned model files for every arm (`harness assets`): the DINOv3 code at commit 6876159a in the torch hub cache, the ViT-B/16 weights (sha256 73cec8be...) in an offline Hugging Face cache, the `cpsam` weights |
| agent freedom | tool parameters (global, per image, per cell), which calls / images / cells feed the next tool, its own analysis scripts; no other processing code, no edits to `tools/` (hashed) |
| deliverable | `submission/choice.json` = a cluster call and its resolution; the harness walks the call chain back (embedding, crops, masks) and checks every file against the hashes the tools recorded |
| lever coverage | the shared rule of every agentic tuning run: FINAL is accepted once every `<tool>.<lever>` was changed by a completed call or given a written reason |
| primary endpoint | `knn_transfer_macro_f1` (user decision 2026-10-02, after the pairs smoke): cell-level cross-micrograph kNN label transfer, the label-transfer evaluation used for single-cell references -- up to 100 cells per image are drawn (agent-excluded cells included, as misses); an expected image without any delivered cell counts as 100 cells without a prediction; each drawn cell's k = 20 nearest drawn cells (cosine, in the delivered embedding) from OTHER micrographs (`parent_id`) vote with weights exp(cos / 0.07), DINO's reference weighted kNN (Caron et al. 2021; user decision 2026-10-03 after a literature review; precedents: Ljosa et al. 2013 and Ando et al. 2017 (NSC / NSCB), Abdelaal et al. 2019, CHAMMI 2023, scIB 2022); tie: the nearest tied neighbour; macro F1 primary, median F1 reported; a class with fewer than two micrographs in the image set is not scored (fixed by the image set, not by the delivery); macro F1 over the prep's classes (a class without delivered cells: F1 0); SE by bootstrap over classes. Classes = `morph_class` from the versioned `morph_classes.csv`: 7 lineage classes (epithelial, endothelial, mesenchymal, muscle, immune, neural, melanocytic), approved by the user, covering all 80 known cell types; `unscored` (rounded, polygonal, mixed) leaves the coarse level only; a missing or incomplete map stops scoring with `NO_CLASS_MAP`. Secondary at the same level: the fine `cell_state` version, and the cluster-level cross-micrograph transfer (`transfer_macro_f1`: the majority class of the cell's state among other micrographs' cells; strict -- states that split by image get no prediction). A choice whose chain does not verify (`NO_VALID_CHOICE`) is scored as an empty delivery |
| secondary | listed right after the primary, the cross-source guard `knn_transfer_cross_source_macro_f1`: the same kNN vote with every cell of the query's own source excluded as well (a claimed gain must not be negative on it; classes whose images span fewer than two sources are removed and listed, as in Ando et al. 2017); state level at the chosen resolution: state replication (share of cells in states with >= 2 micrographs each >= 5 % of the state, median effective number of micrographs) and state-composition replicate consistency (AUC of the 1 - Jensen-Shannon similarity of images' state compositions, same type from another micrograph vs a different type; morph, fine and cross-source versions); the near-miss share of fine-level kNN errors (predicted type of the true lineage class); a per-run reference on the same kNN sample (BIABench's ground-truth replay): `oracle` (one-hot morph_class embedding: the best score this delivery allows, misses still count) and `chance` (random embedding), with `normalized_primary` = (primary - chance) / (oracle - chance); strata (`strata.csv`): the primary's predictions with the queries restricted to one `image_modality`, `source` or `magnification` value (macro F1 over the classes present, no SE); scIB isolated-label F1 and per-class best-cluster F1 (credits a class that forms a clean cluster without a cross-micrograph partner); image-level NSC / NSCB-style 1-NN accuracy on mean-cell profiles (other micrograph; other micrograph and other source; classes without a candidate removed from the denominator, as in Ando et al. 2017); transfer accuracy, median F1 and unlabeled share; plain per-state majority-vote accuracy; NMI / ARI of the states; scIB-style NMI / ARI (Luecken et al. 2022, Nature Methods): the sampled cells of the delivered embedding (the kNN sample) re-clustered at resolutions 0.1-2.0 (one cosine neighbour graph, k 15, Leiden only), the best NMI and its ARI; source predictability from the embedding; the earlier class-balanced linear-probe endpoint where its fixed folds apply; coverage; cost (wall time, tool calls, tokens); `umap_scored.png` (true class, states, kNN transfer correct / wrong / no prediction, source) |
| comparison | each run vs the control of the same prep, metadata level and GPU model the run used (`RUN_META.json`), only when both scored the same image set and the same class map (sha256 of each): difference of `knn_transfer_macro_f1`, paired bootstrap over classes; verdict `GAIN` (primary delta > 2 SE and cross-source guard delta >= 0), `SOURCE_SHORTCUT_SUSPECTED` (primary gain, guard delta < 0), `LOSS` (< -2 SE) or `NO_RESOLVABLE_DIFFERENCE`; a control without the guard: guard `unavailable`, verdict from the primary with a note; a paired bootstrap over the fine cell types (`knn_type_counts.csv`) is reported next to it. This verdict is RESULT.json `knn_transfer_verdict`; `verdict` holds the three full_metrics verdicts (mAP, macro F1, balanced accuracy; `benchmarks/_shared/FULL_METRICS_CONTRACT.md`, RESULT.json `full_metrics`, the reading in `full_metrics_draws.npz`; a control without it gives no full_metrics verdicts) |
| attribution | `harness attribute --run RUN --control CONTROL --out OUT`: a 2 x 2 factorial cut at the crops, objects O (segment + crop + embed-time exclusions) x representation R (embed + batch_correct + cluster recipe); two replay chains (the run's crops with the control's recipe, the control's crops with the run's recipe; the run's exclusions stay with O) on the GPU, then phi_O, phi_R, the interaction and the total for the primary and the guard with type-bootstrap CIs (`ATTRIBUTION.json`, `ATTRIBUTION.md`, `attribution.png`) |
| candidates | `score --all-candidates` (the launcher's default, `ALL_CANDIDATES=1`): every completed embed / batch_correct call (primary, fine, cross-source) and every cluster call at each resolution (cluster-level transfer, state replication, state composition) on the same image set, never shown to the agent; `candidates.csv` and RESULT.json `candidates`: selection regret (best candidate minus delivered), batch-decision regret (on the cross-source guard), resolution regret, the anytime curve of the best eligible primary, and proxy validity (Spearman of every numeric field of the candidates' tool summaries, i.e. the label-free numbers the agent could read, with the hidden score) |
| process metrics | `harness process` (run by `finish` after `score`), descriptive only, no labels: per-image adaptivity of segment / crop (share of images with their own setting, lever dispersion, rescued images), substantive vs probe-only lever coverage, note traceability, cost (wall / GPU / tool-busy hours, turns, tokens, USD marked as Claude Code's list-price estimate; an absent signal is null with a reason, never 0), failure taxonomy, the end state (delivered / no deliverable / crashed, timed out, refused) and the claim of completion (FINAL, closing text); `PROCESS.json`, `PROCESS.md` |
| summary table | `harness table --runs DIR ... [--control-root ROOT] --out OUT` (after BIABench Table 1): one row per configuration (arm, model, prep, metadata, GPU): n runs, primary / cross-source / normalized mean +- s.d. and range, D/N/C/T, median wall time, tokens, USD, GPU-hours and tool calls, the difference to the control, the between- vs within-configuration variance, completion claims among low-outcome runs; `TABLE.md`, `table.csv`, `runs.csv`, `pareto.png`. Repeat each agent configuration (3 runs) before comparing: BIABench found run-to-run variance larger than the between-agent one |
| channels | the DINOv3 analyzer refuses mixed channel counts in one call; when HCMA's images mix them, prepare brings every image to 3 channels (a single channel replicated, a constant 4th channel dropped; values and dtype unchanged) and records it in `PREPARE.json: channel_conversion` |

## Isolation and audit (as in the SSL agentic harness)

- The agent works in `<run>/workspace`; labels live in `$ROOT/private` (same Unix user: a rule plus
  an audit, not a wall). `score` lists every agent tool-call input that names the HCMA raw data,
  `$ROOT/private`, the control or other entries of `$BENCH` (other runs), the SHAPE repo (its VRR code, and
  `cell_state_HCMA`, which holds a labels file), and checks `tools/` against its setup hashes.
- Stanford AI Gateway only, key from `api.sh` through `apiKeyHelper`, never in the agent's env or
  on disk; Claude runs with an allow-list (`Bash Read Write Edit Glob Grep TodoWrite`), web tools
  off; a per-run venv; a watchdog kills the agent's process group at the deadline; rounds resume
  the agent until the deadline or an accepted FINAL; the trace is re-rendered live.

## Layout

The shared verifier search defaults to `legacy`. `SHAPE_SEARCH_STRATEGY=four_stage`
opts into cover, contrast, narrow, confirm; per-tool overrides append `_SEGMENTATION`,
`_CROP`, `_BATCH_CORRECTION`, or `_REPRESENTATION`. No strategy bypasses admission.
Confirmation currently means existing quality and delivery checks, not held-out
validation. SSL candidates remain multi-epoch runs with checkpoint/step context.
Narrowing requires controls matching its anchor's full background and context, or an explicit
waiver that records why the combination is being searched without parameter attribution.
The HCMA batch stage still uses its calibration-rule CLI, not BatchCorrectionVerifier;
its standalone verifier supports the strategy, but that override does not alter this
benchmark's batch recipe. Under-detection remains informational in segmentation;
this merge repair does not restore it as an admission trigger.

```
full_HCMA/
  harness/                prepare, assets, workspace, hardware, agents, trace_md, export,
                          ledger (call chain), coverage, score, process, table, control, shape_arm, cli
  workspace_template/     TASK.md (the agent's instructions) and tools/ (the five tool CLIs)
  test/                   CPU tests with the SHAPE tools stubbed
```

Input: the raw HCMA download, `$HCMA_ROOT = /scratch/groups/your-group/your-user/shape_benchmark/Human_Cell_Morphology_Atlas_HCMA`
(read-only). Everything the launcher writes is under
`$BENCH = ${SHAPE_RESULTS_ROOT:-/scratch/groups/your-group/your-user/shape_benchmark/results}/full_HCMA`
(the SHAPE results layout); the reusable intermediates are in `$ROOT = $BENCH/cache`:

| path | content |
| --- | --- |
| `$ROOT/prep/` | anonymized images, `images.csv`, `PREPARE.json` (agent-visible) |
| `$ROOT/private/` | `labels.csv`, `folds.csv`, `scorable.csv`, `PREPARE_PRIVATE.json` |
| `$ROOT/assets/` | pinned model caches, `ASSETS.json` |
| `$ROOT/prep_pairs/` | the `pairs` subset of `prep/` (same anonymized ids) |
| `$ROOT/control/<prep>/<metadata>/<gpu>/` | the default-parameter chain and its `RESULT.json` (one per prep, metadata level and GPU model) |
| `$BENCH/shape-<prep>-<metadata>_<commit8>_<utc>_<job>/` | one SHAPE-arm run (`shape-smoke<N>-...` for a smoke): `RUN_META.json` (arm `shape`), `SHAPE_RUN.json`, `workspace/`, `logs/tool_calls.jsonl`, `logs/control_commands.jsonl`, `logs/shape/` (each loop's log and selection details), `cache/shape_arm/` (the loops' own files, not exported), then the same outputs as an agent run |
| `$BENCH/claude-<prep>-<metadata>_<commit8>_<utc>_<job>/` | one agent run: `RUN_META.json`, `workspace/`, `logs/tool_calls.jsonl` (tool ledger), `traces/`, `trace.md`, `RESULT.json`, `RESULT.md`, `per_image.csv`, `candidates.csv`, `strata.csv`, `PROCESS.json`, `PROCESS.md`, `EXPORT_<run>.tar.gz`, `COMPLETE` (written last when scoring succeeded) |

Local evidence mirror: `projects/SHAPE/tasks/benchmark/full_HCMA/<run_id>/`.

## Run

On a machine with one GPU, through the harness CLI (`--help` for every option):

```bash
C="python -m benchmarks.full_HCMA.harness.cli"      # --metadata full (default) | none
$C prepare                                          # once (CPU): anonymized prep, the pairs subset, private labels
$C assets                                           # the pinned model caches
$C smoke                                            # the real tools on 8 images, end to end
$C control                                          # once per prep, metadata level and GPU: the default-parameter chain
$C agent --model claude-opus-5                      # one Claude Code run
$C shape --model <model>                            # the SHAPE arm; --images 8 for a smoke
$C score --run <run-dir> && $C trace --run <run-dir> && $C export --run <run-dir>
$C attribute --run <run-dir> --control $ROOT/control/<prep>/<metadata>/<gpu> --out <run-dir>/attribution
```

## Caveats

- The score is end-to-end: it does not attribute a difference between arms to a single step
  (segmentation, cropping, embedding, correction, clustering). The scIB-style re-clustering
  secondary separates the embedding from the agent's clustering choice; nothing scores masks or
  crops directly (HCMA has no segmentation ground truth).
- On HCMA `source` is nearly the imaging modality (ATCC: all bright-field), and most `pairs` types
  come from one source, so a correction on `source` could also remove type signal (earlier HCMA work
  measured lower cell_state NMI after Harmony on `source`, `cell_state_HCMA/results/hcma_scib_cell_state_source.json`);
  under the calibration rule (user decision 2026-10-06) step 4 returns its input unchanged on HCMA, which has no
  label-free reference groups replicated across sources.
- The `pairs` draw depends on the anonymized micrograph ids; `private/SUBSET_pairs.json` records
  the actual selection, and its source / modality / magnification composition differs from the
  full atlas.

- The labels are per culture, not per cell: the score asks whether a cell's embedding predicts its
  culture's lineage class, which is a proxy for single-cell morphological states.
- `pairs` has 2 images per type: per-type F1 rests on few images, so the bootstrap over types is the
  honest uncertainty; the cross-image transfer is what keeps image identity from scoring.
- At the vendor level HCMA is nearly single-source for every fine label
  (`cell_state_HCMA/README.md`, stage-4 screening); `source` and class are confounded, so the source
  predictability is reported next to the endpoint, not as one.
- Vendor logos are burned into the pixels (ATCC bottom-left with a scale bar, ScienCell top-right); prepare
  removes file metadata, not pixels, so the source stays readable from the image even under
  `METADATA=none` (user decision 2026-10-03: not masked; a known confound).
- Runs are not bit-reproducible: cuBLAS runs non-deterministically (`CUBLAS_WORKSPACE_CONFIG` is not set;
  user decision 2026-10-03), so a rerun with the same seed differs slightly.
- The control runs one seed (user decision 2026-10-03; `CONTROL_SEEDS=0,1,2` would estimate the
  run-to-run spread of the default fine-tune, which is unknown until then). Runs use `PREP=pairs`.
- Scoring time (16-core laptop, synthetic data): `pairs` about 30 s; the full prep (about 120k sampled cells) is
  estimated at roughly 10-25 min (exact kNN about 4.5 min, the scIB graph about 4 min, Leiden at 20 resolutions
  data-dependent); RESULT.json records per-component `timings`. The agent stage reserves 90 min of the allocation for
  scoring and export, and `finish <run-dir>` re-scores a run from another allocation.
- `crop.py` calls the cropper as SHAPE's own chain does (`shape/agent/executor.py`): no margin by
  default (a lever; the tool's rule: fraction mode, margin_frac 0.20, a 1 px floor for a cell_mask) and
  each image's segmentation `min_size` as `min_area` (not a lever), so the control and SHAPE start from the
  same crops.

# Task: single-cell morphological cell states from raw microscopy images, end to end

You are an autonomous research agent on a GPU node. You are given {{N_IMAGES}} raw
microscopy images of cell cultures from the Human Cell Morphology Atlas (HCMA) and five fixed
**tools**. {{DESIGN}} Your job: for every cell in every image, produce a single-cell morphology
embedding and a morphological cell state (a cluster of the UMAP / Leiden clustering), so that the
embedding and the states predict each cell's cell type. The tools form one chain:

    segment (Cellpose-SAM) -> crop (MaskCrop, square single-cell crops) -> embed (DINOv3 ViT-B/16)
    -> batch_correct (Harmony; optional) -> cluster (Leiden)

You choose the tools' parameters (for all images, per image or per cell), call the tools, read
what they return, reason, and call them again. Everything else (what to run, in which order, on
which images or cells, how to read the outputs, when to stop) is your judgment.

## Score (computed by the harness after the run; never shown to you)

Each image shows one culture whose cell type the harness knows and you do not. The harness groups
the cell types into 7 broad lineage classes -- epithelial, endothelial, mesenchymal, muscle,
immune, neural, melanocytic -- called the "type" below (cell types of mixed lineage are not
scored at this level); the same score over the finer cell types is also reported.
The score asks whether your cells' embedding predicts the type of single cells across
micrographs (cross-micrograph kNN label transfer):

- Up to 100 of your cells per image are drawn (a fixed random sample of the image's cells,
  including any you left out of the embed call).
- Each drawn cell's 20 nearest drawn cells (cosine similarity in the embedding your chosen cluster
  call clustered) that come from **other** micrographs are found (cells of its own micrograph,
  `parent_id`, do not count, also when this run does not give you `parent_id`; fewer than 20 such
  cells: all of them; none: no prediction). Each neighbour votes for its type with the weight
  exp(similarity / 0.07), and the cell is predicted as the type with the largest summed weight (a
  tie goes to the type of the nearest tied neighbour). Cells of the mixed-lineage cell types (those
  not scored at this level) are neither predicted nor used as neighbours. A drawn cell you left out
  of the embed call gets no prediction, and an image for which you deliver no cell at all counts as
  100 cells without a prediction.
- A type found in fewer than two micrographs of the image set is not scored (it cannot be predicted
  across micrographs); this is fixed by the image set, not by what you deliver.
- For every type the harness computes precision, recall and F1 over the drawn cells (a cell without
  a prediction lowers its type's recall and predicts no type); a type for which you deliver no cell
  has F1 = 0. Images whose type is unknown to the harness are not scored.
- **Score = the mean F1 over types (macro F1);** the median F1 over types is also reported.

Also reported: the same score with the cells of the drawn cell's own source excluded from its
neighbours as well (`source`, cross-source); the same score over the finer cell types and over
embedded cells only; whether each of your states is made of cells of several micrographs, and
whether images of the same type from different micrographs have similar shares of cells in your
states; for every
type, the F1 of the single cluster that best matches it when the harness re-clusters your
embedding at resolutions 0.1-2.0 (and its mean over the types found in the fewest sources); an
image-level nearest-neighbour accuracy (each image's mean cell embedding matched to the images of
other micrographs, and of other micrographs and other sources); the share of
cells predicted right; a cluster-level version of the transfer, in which each cell gets the most
frequent type, in its state at your chosen resolution, among the cells of other micrographs; the
plain majority-vote accuracy of your states; NMI and ARI of your states against the types; the best
NMI the harness reaches by re-clustering your embedding at resolutions 0.1-2.0 (and the ARI at that
resolution); a linear classifier's accuracy on your embedding across fixed micrograph folds; how well
the image source can be predicted from the embedding; the share of images with cells; the cost
(time, tool calls, tokens); and a UMAP figure of your states. Your other embed, batch_correct
and cluster calls are scored the same way for the analysis of the run; your score is that of the
chain you choose. You are compared with the same five
tools run once at their default parameters, except that the embedding is zero-shot, in a
fixed chain (segment -> crop -> embed with adaptation_depth zero_shot -> batch_correct with batch
key `source` -> cluster).

## Data (in `data/`)

| path | content |
| --- | --- |
| `data/images/<image_id>.<ext>` | the raw images; ids are random (`img_0001` ...), unrelated to their content |
| `data/images.csv` | one row per image: `image_id, file, parent_id, source, image_modality, magnification, height, width, channels` |
| `data/PREPARE.json` | counts |

Sources: {{SOURCES}}

`parent_id` names the micrograph an image was cut from (some micrographs were split into panels).
`source` is the vendor or dataset the image comes from. These columns hold every technical fact
this run gives about the images:

{{METADATA_NOTE}}

## The tools (in `tools/`)

    python tools/<tool>.py --describe        # every lever: type, range, default and meaning
    python tools/<tool>.py --name NAME [inputs] [--set lever=value ...] [--config FILE.json] \
        [--note "why"] [--dry-run]

| tool | input | output (in `calls/<ID>_<NAME>/`) | GPU |
| --- | --- | --- | --- |
| `segment.py` | images (`--images all`, a list, or `--images-file`) | `masks/`, `overlays/`, `per_image.csv` | yes |
| `crop.py` | masks of segment calls (`--masks S00k`, or `--masks-map` per image) | `crops/`, `crops.csv`, `per_image.csv` | no |
| `embed.py` | crops of crop calls (`--crops C00k`, or `--crops-map` per image; `--exclude-cells`; `--meta`) | `embedding.npy`, `cells.csv`, `cell_state_analyzed.h5ad` | yes |
| `batch_correct.py` | an embed call (`--embedding E00k`; `--batch-key COLUMN` or `--batch-file`); corrects only with label-free reference groups replicated across batches | `embedding.npy`, `cells.csv`, `tool_result.json` | no |
| `cluster.py` | an embed or batch_correct call (`--embedding E00k` or `B00k`) | `states.h5ad`, `states.csv`, `umap.png` | no |

- Every call gets an id (`S001`, `C001`, `E001`, `B001`, `K001`, ...) and its own directory with
  `meta.json` (what you asked for) and `result.json` (status, timing, the tool's summary, file
  hashes). Every call is logged by the harness.
- Per-input parameters: `segment.py --per-image FILE.json` (`{image_id: {lever: value}}`),
  `crop.py --per-image FILE.json` and `--cell-overrides FILE.json`
  (`{image_id: {tool_cell_id: {key: value}}}`), see `--describe`.
- A later call can combine the outputs of several earlier calls per image (`--masks-map`,
  `--crops-map`), so different images may come from different calls.
- `--workers N` runs segment (GPU processes) or crop (CPU processes) in parallel.
- `batch_correct.py` corrects only when images.csv gives label-free reference groups (columns `group` or
  `treatment`, and `control`) replicated across at least two batches of the batch key, SHAPE's
  calibration rule; otherwise it returns the input unchanged (`tool_result.json`: `corrected`, `reason`).
- Declare why you make each call (`--note`); it is kept with the call.
- Start every tool call as its own command in your Bash tool, with its name written out
  (`python tools/<tool>.py --name NAME ...`, optionally inside `nohup ... &`). The harness matches
  each call of your delivered chain to the commands it saw you run; a call started from a script, a
  loop, or with a variable in place of its name cannot be matched and breaks the chain.

## Rules

1. Images, masks, crops, embeddings, corrections and states are produced **only by the five
   tools**. Your own scripts may read the images and every tool output (analysis, statistics,
   figures) and decide which calls, images or cells feed the next tool, but they must not produce
   masks, crops, embeddings, corrections or states themselves, and must not run other GPU jobs. Do
   not modify anything in `tools/`.
2. No labels and no outside information about these images: do not read anything under the paths
   below, and do not try to recover the original image names or the cell types. Do not download
   anything.
{{FORBIDDEN}}
3. Stay inside this directory. Hard wall-clock deadline: **{{BUDGET_HOURS}} h** from the start
   (`python tools/time_left.py`); a tool refuses to start a call less than 10 minutes before it.
4. Lever coverage (a rule of every agentic tuning run): `submission/FINAL` is accepted only once
   every lever group -- each lever of each tool, named `<tool>.<lever>` (e.g. `segment.diameter`,
   `crop.cell_overrides`) -- has been changed by at least one completed call (set explicitly to a
   value other than its default), or `submission/choice.json` gives a reason for it under
   `"skipped"` (a tool name, e.g. `"batch_correct"`, gives one reason for all its levers).
   Otherwise the harness removes FINAL and resumes you; the deadline ends the run in any case.
   `python tools/lever_coverage.py` shows the state.

## Deliverable (keep it valid at all times)

`submission/choice.json`:

    {"cluster_call": "K00k", "resolution": 1.0, "reason": "...", "skipped": {"your-group": "why"}}

The harness follows that cluster call back through the calls it used (its embedding, the crops,
the masks), checks every file against the hashes the tools recorded and every call against the
commands you ran, and scores its states at
`resolution` (one of the resolutions of that call) and the embedding it clustered. Without a valid
choice the run scores nothing; FINAL with an invalid choice is not accepted.

## Running

- This node: {{GPU}}; up to {{GPU_SLOTS}} GPU call(s) at the same time (a lock enforces it);
  {{CPUS}} CPU cores; {{MEM_GB}} GB RAM.
- Measured on this GPU type by the default-parameter run:
{{TIMINGS}}
- Run long calls in the background and poll them:
  `nohup python tools/embed.py --name NAME ... > logs/NAME.log 2>&1 &`, then read the log and
  `calls/<ID>_<NAME>/result.json`. You may end your turn while a call runs: the harness resumes you
  (no earlier than 5 minutes after the previous round started) until the deadline or an accepted
  FINAL.
- You may be resumed in a fresh context. Keep `NOTES.md` current (what you ran, what the outputs
  showed, what you concluded, what next); it is your memory.
- You can look at any image and any tool figure (PNG / JPG) with the Read tool.
- When you are done and the choice is final, create the empty file `submission/FINAL`.

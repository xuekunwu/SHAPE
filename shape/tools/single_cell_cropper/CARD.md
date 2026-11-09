---
tool: Single_Cell_Cropper_Tool
category: bioimage_core
stage: cell_level
depends_on: [Cell_Segmenter_Tool, Organoid_Segmenter_Tool]
tunable: true
metadata_consumes: [group, plate, well, site, batch, experiment, technical_group]
metadata_none: geometry from the mask
metadata_partial: identity copied to every crop; geometry unchanged
metadata_full: same as partial
---
# Tuning card: Single_Cell_Cropper_Tool

**When to use:** after segmentation, to cut one crop per detected object for single-cell
analysis. **Requires** a mask from a segmenter first. Input: image + segmentation mask.
Output: per-cell crops + metadata.

**Engine:** MaskCrop -- our deterministic, mask-driven single-cell cropping engine
(per-cell crops, always square -- "Levers"). **Shipped default (task 038 follow-up,
part B redesign, human directive): the NAIVE, general-purpose baseline geometry**
(`margin_mode="fraction", margin_frac=0.20, recenter=False`) -- the tool's own default
is now IDENTICAL to the benchmark's own MaskCrop control arm, so "MaskCrop" and "this
tool's shipped default" are literally the same config. `margin_frac` raised 0.10 ->
0.20 (task 038 v9 restructure, human-locked): a v8 cluster run confirmed the
square-tight geometry lever's own quality gain stays ~0 (SS2's "Evidence / provenance"
explains why), and the 10% baseline was already close to its own ceiling -- 20% gives
the agent (and the naive baseline it starts from) genuinely more context/headroom. The
tight-margin + centroid-recenter geometry ("Levers") briefly WAS the shipped default (task
038 quality-geometry round through the part A/B recalibration rounds) -- it is now the
AGENT's own per-cell optimization TARGET, applied via `cell_overrides` ("Levers", SS5), not
what a caller gets by default. The cropper's analog of Cellpose-SAM for the segmenter: a
self-defined execution engine, not a third-party model.

## Purpose

**This tool's job is to PRODUCE high-quality crops from the segmentor's mask -- not to
triage them away.** A high-quality crop is, in this order:
1. **Square, always** -- the analyzer resizes every crop to 224 x 224 without keeping the aspect
   ratio (`cell_state_analyzer/tool.py` `_get_eval_transform`, and the adaptive path's
   `adaptive/augment.py` `clean_transform`), so a non-square crop's cell would be stretched. Every
   crop the tool writes is square (height = width), in every `margin_mode` and at the image border
   too, cut from real image pixels: no constant padding, no masking of neighbours. At the border the
   same-size square shifts inside the image (the target off-center), and in a dense field a square
   holds more neighbour pixels than a rectangle tight to the target would -- an accepted cost.
2. **Contains ONE cell** -- the target, whole (not cut off).
3. **Cell CENTERED** -- the target sits at the crop's own center.
4. **Cell occupies the MAIN AREA** -- the target dominates the frame, not a minority sliver.
5. **Minimal / no other cells** -- neighbours minimised, never chased at the cost of 1-4.

These are the 5 signals `quant.py::crop_quality_score` reports: sharpness (a blurred crop is
not USABLE even though focus is not one of the 5 geometric axes by name) · single_object (axis
2: ONE cell, intact) · centered (axis 3) · target_dominance (axis 4) · not_truncated (axis 2:
contained). Its COMBINED score weights ONLY the 3 axes this tool's geometry levers ADJUST
(not_truncated/target_dominance/centered, `quant.py::_WEIGHTS`); sharpness/single_object are
computed and reported (diagnostic). The tool's job is to MAXIMIZE these via deterministic
GEOMETRY (the crop margin, mode and placement) -- reject (a separate check) is a narrow,
strictly-bounded LAST RESORT for a crop too bad to fix by geometry, not the tool's normal
output.

Neighbour cells in the crop are TOLERATED per axis 5's own wording ("minimal", not "zero") --
in a dense field a tight crop still admits neighbours; we do NOT mask them out (that destroys
real context for downstream cell-state analysis). The real levers for all 5 axes are crop
GEOMETRY; a separate, narrow keep/reject judgment is layered on top.

---

## Agentic VRR (`CropAgenticVerifier`, the shared `AgenticVerifier`, StepVerifier wave mode)

**Call names** (`AgenticVerifier.label`, shared by every agentically tuned tool): each call is named by
who proposed it, once it has run -- `input` (trial 0, the input; not a call), `default` (the tool at its
defaults), `reference` (the system's reference call), `agent_1`, `agent_2`, ... (the agent's calls, in
order); a second default or reference call is
`default_2` / `reference_2`. The prompts, the parents of moves, the final choice and the records use these
names.

Rules measure and guard; the agent decides. The tool cuts the crops, `StepVerifier` owns the loop, and
`CropAgenticVerifier` (`shape/agent/verification/verifiers/crop.py`) measures, guards and asks the
reasoner (the agent, a multimodal model) for every decision the measurements leave open: whether the
tool's call at its defaults needs tuning, which tool calls to run, how many per round, from which
parents, which image-level levers and cell rules, when it is satisfied, how to read its failed
predictions, and the final call. It subclasses the loop every agentically tuned tool shares,
`AgenticVerifier` (`shape/agent/verification/base.py`: the admission on the call at the defaults, the
rounds, the reasoner's questions and their retries, the validation of the CARD levers, the repair
question, the hypothesis list, the budget and the plateau, the final choice and its retry, the
execution-error accounting); this section lists what the cropper adds. It is the crop stage's verifier
for `Single_Cell_Cropper_Tool` (`stage_map.py`, `verifier_by_tool`); the linear `CropVerifier` (section
5) stays the stage's own verifier, used when the opt-in OctoTools++ controller is enabled. The settings
named below are listed with their values under "Loop settings".

A caller may pass a `CropTask` in the stage context (`upstream_ctx["crop_task"]`): the cores for the
tool calls, a deadline (e.g. the job's time limit) and a one-line brief. Without it (the `solve()`
path) the cores are those this process may use and there is no deadline.

0. **Trial 0 is the input** (`input_record(base_kwargs)`), a measurement and not a tool call: the image
   (size, channels, dtype, per-channel 1st/99th percentile) and the mask (objects, objects at
   `min_area`, objects no square crop can contain at `RETENTION_FLOOR` -- `n_uncroppable`, listed by
   `cell_id` as `uncroppable_cells` --, foreground fraction) and, for every object at
   `min_area`, its per-cell input measurements keyed by the tool's own `cell_id` (the same regionprops
   order `_generate_cell_crops` enumerates): `area`, `eq_diameter`, `bbox_h`, `bbox_w`, `aspect_ratio`,
   `solidity`, `bbox_other_fraction` (the share of the mask foreground inside the object's own bbox
   that belongs to other objects), `nn_gap_px` (the pixel gap to the nearest other object, measured up
   to `NN_GAP_CAP_PX`) and `touches_image_border`. An object is uncroppable when its bbox is longer
   than the image's shorter side and even the tool's best placement of the largest square that fits
   (`compute_square_crop_containing_bbox`, centred on the bbox midpoint or the centroid) keeps less than
   `RETENTION_FLOOR` of it: no call can give it a crop. `min_area` is the delivering segmentation's
   `min_size`, passed by the caller (the pipeline's cropper command and, on the `solve()` path,
   `integration.build_base_kwargs` pass it for a cell or nuclei mask); without a known producer it is the
   tool's default by mask file name: the cell segmenter's default `min_size`, 15
   (`tool.py::SEGMENTER_DEFAULT_MIN_SIZE`), and 1000 for an organoid mask (both callers pass 50 for an
   organoid mask). It is fixed for the loop, not a lever; `n_croppable`,
   `n_uncroppable` and the guard's `n_missing_crops` count over the objects at `min_area`. The default
   `margin` follows the tool's rule by mask file name. The input also fixes the crop
   panel: `PANEL_CELLS` cells spread over `bbox_other_fraction`, shown for every call. A mask without an
   object at `min_area` ends the stage (`conceded`, `no_objects`).
1. **Admission** (`admission_call`): whether the image needs tuning, decided on what the tool delivers
   untuned. Round 1 is the tool at its defaults -- the configuration of a move from `input`, with the
   margin resolved for the mask -- run before the reasoner is asked; it is `default`, measured like every call
   (step 2), and counts as round 1 of `BUDGET`. A measured problem of `default` starts the tuning: the call
   failed or produced no crops; the guard fails (step 5); or any of its crops -- all of them, before the
   mask-error check, as the readouts -- is one the linear loop's round-1 geometry rule (section 5) would
   re-run: `neighbor_contamination` above `CropVerifier._MASK_BBOX_CONTAMINATION_MIN` (re-run in
   `mask_bbox`) or `centroid_offset` above `CropVerifier._OFF_CENTER_MAX` (recentred) -- except a crop
   whose offset the image edge forces (`edge_limited`: no square of its side inside the image places the
   target's centroid closer to the crop centre, the box within `EDGE_TOLERANCE_PX` of the image edge on
   the side the centroid lies toward; no placement fixes it, so it is not counted as off-centre). Each
   readout is counted over all the crops (a crop above both counts in both) and stated with its count and
   threshold, e.g. `2 of 7 crops with centroid_offset above 0.2; 1 more are edge-limited` (the second part
   when edge-limited crops were left out). Without a measured problem the reasoner is asked
   (the checks named, the call's table row, the input image and `default`'s crop panel) for the main problem
   it sees in `default`, rated `none`, `minor` or `major`: `major` starts the tuning; `minor`, `none` or no
   answer (the reasoner unavailable or failed) delivers `default` as it is (`satisfied`, reason and choice
   `no_tuning_needed`).
   When the tuning starts, every round prompt states why: the measured problems, or the reasoner's major
   problem. `details["admission"]` records the call, the rule (`RUN` on a measured problem, else `SKIP`),
   the problems, the reasoner's answer and the decision.
2. **Rounds** (one VRR cycle each). Recommend: the reasoner returns the calls of the round -- at least
   one, at most the concurrency cap -- each a parent (`input` = the tool at its defaults, or any
   finished call), the image-level levers it changes relative to the parent's configuration (one or
   several; none from `input` = the tool at its defaults), optional cell rules, a diagnosis naming its
   evidence and the readouts expected to move up, down or stay (`preserve`). With every plan it returns
   its hypothesis list (id, statement, status open / supported / refuted / abandoned, evidence), shown
   back to it next round.
   - A **cell rule** selects the cells for which all its conditions (`readout op value`, op one of `>`,
     `>=`, `<`, `<=`, `==`) hold -- a rule without a condition selects every cell -- and sets per-cell
     keys for them ("Levers", "Per-cell keys").
     Conditions read the per-cell input measurements (always) and, when the parent is a finished call,
     that call's per-crop readouts of the cell: `neighbor_contamination`, `target_dominance`,
     `target_retention`, `centroid_offset`, `centered`, `occupancy`, `border_touch`, `n_complete_cells`,
     `contamination_inset_px`, `crop_h`, `crop_w`, `crop_aspect`, `margin_mode`, `edge_limited`. Rules apply in order on
     top of the parent's per-cell settings (a later rule overwrites an earlier one's keys; `reset`
     removes a cell's keys); the verifier compiles them into the call's `cell_overrides`.
   - StepVerifier runs the round concurrently; each call writes into its own directory,
     `<query_cache_dir>/tool_cache/crop_calls/<image>/<configuration key>/` (the tool's
     `output_subdir`), and the verifier measures every result on the loop's thread.
   - **Verify** (per call, GT-free, over all its crops before the mask-error check; every per-crop readout
     measures the cell's own label, its regionprops region): `n_crops`,
     `n_lost_to_geometry` (objects at `min_area` without a crop) and, identified per `cell_id`,
     `n_missing_crops` (those a square crop can hold at `RETENTION_FLOOR`: the guard's count) and
     `n_exempt_uncroppable` (the uncroppable ones), `n_edge_limited` (crops with `edge_limited`, step 1),
     mean / p10 / p50 / p90 of
     `neighbor_contamination`, `target_retention` (and its minimum) and `centroid_offset`,
     `mean_target_dominance` (1 - mean contamination), `mean_centered` (1 - mean offset),
     `mean_occupancy`, the share of non-square crops, the p90 aspect ratio, the median crop side, the
     share of crops per resolved margin mode and the mean `auto_tighten` iterations. The mask check (the
     mask-error check of section 2) is not run per call and is no readout: it runs once, on the delivered
     call (step 5). `scalar_score` = `score_key` (`mean_target_dominance`).
   - **Reconcile** (facts shown in the next prompt): per call, the key readouts (`key_readouts`) relative
     to its parent and to the call at the tool defaults (`default`, when it finished; not repeated for a call
     whose parent is `default`, where the comparison to the parent is that comparison), over the cells both
     cropped, as a mean with its 2SE, beyond or within the noise, and how many cells moved up or down
     (without `mean_neighbor_contamination`: per crop 1 - `target_dominance`, its difference is the
     `mean_target_dominance` one negated, cells up and down swapped); each stated expectation next to the
     observed change (`metric: "expected -> observed"`); the calls table names the readouts once, in its
     first line, and each row's readouts are the values in that order; whether the call is an admissible final choice now
     (step 5); Pareto membership on `objectives` (step 3); the calls that failed or produced no crops; the
     levers and per-cell keys not used yet; the rounds done and left (a limit, not a target), the cores
     and the cap, per-call seconds, the wall time and, when the caller knows it, the time left. Each
     prompt also carries the input image with the mask's object boundaries and the crop panel of every
     call of the previous round (the panel cells' crops with the target's contour): every question to
     the reasoner is stateless, so a prompt carries what the reasoner needs. Whether a prediction
     failed, and what follows from it, is the reasoner's judgement.
3. **Noise and the Pareto front.** The tool is deterministic (the same configuration gives the same
   crops), so the noise of a difference between two calls is the spread over cells: the larger of
   `NOISE_BAND` and 2SE of the per-cell paired difference over the cells both calls cropped. The
   Pareto front (on `objectives`) and the plateau compare the admissible calls only (step 5): each
   has a crop for every object at `min_area` but the uncroppable ones, so their means are over the
   same cells. An inadmissible call -- e.g. one whose `crop_size` cap dropped most objects, its few
   crops scoring 1.0 on every objective -- is never on the front, dominates no call and is no
   progress.
4. **Stops**, whichever comes first: the reasoner returns no moves (satisfied), at any round -- with
   rounds left it is asked once more (the shared stop check, `confirm_stop`, `stop_check_text`) and
   may continue or confirm the stop (`stop_checks`); a stopping plan may name the call to deliver
   (`deliver`; after a stop check, the confirming answer's): an admissible one is delivered without
   the final question of step 5 (`agent_choice`). The backstops: `PATIENCE` rounds in a row without
   progress -- the rounds (a round of failed calls included) after the last one with an admissible
   call better than every admissible call of the earlier rounds beyond the noise on at least one of
   `objectives` (plateau: early stopping, as SSL training's); the budget -- at most `BUDGET` rounds,
   the admission's call included (StepVerifier `max_iters` = 1 + budget; trial 0 is not a round); no
   valid move remains. A failed call, or one without crops, counts in its round and is never re-run,
   never a parent and never chosen.
5. **Final choice** by the reasoner among the admissible calls. The guard: the call has a crop for every
   object at `min_area` that a square crop can hold at `RETENTION_FLOOR` (`n_missing_crops` = 0; the
   uncroppable objects of step 0 are exempt -- human decision, 2026-09-30: no call can crop them, so they
   cannot make every call inadmissible; counted per `cell_id`, so a croppable object's loss never hides
   behind an exempt one), every crop is square (`frac_non_square` = 0, the
   human rule of 2026-09-30 held at the delivery: a call with non-square crops fails it, the failure
   naming their count; the tool writes only squares) and every crop's `target_retention` is at least
   `RETENTION_FLOOR`. The final prompt carries the finished-call table, the input image and the crop
   panels of the calls on the Pareto front (at most `FINAL_PANELS_MAX`; when none is on it -- dominance
   within the noise can cycle -- the admissible calls with the highest `mean_target_dominance`). An
   inadmissible choice is answered with the conditions it fails and asked once more; still inadmissible
   -> the admissible call with the highest `mean_target_dominance` (`conceded`). No admissible call ->
   the reasoner is not asked; the finished call with the fewest `n_missing_crops`, then the
   highest lowest retention, then the highest `mean_target_dominance` (`conceded`, fault `ceiling`).
   The prompts state both rules (`final_rule_text`). On the delivered call (the chosen call, the
   least-bad call, or `default` when the admission skips the tuning), and only on it, the
   mask check then runs (`deliver`; the mask-error check of section 2, unchanged: its three-arm data
   screen, at most `MASK_CHECK_CAP` candidates, verdicts that do not depend on the geometry cached): its
   rejects leave the call's kept-crops manifest, the decision trace is written next to the call's
   metadata, and its counts go into `details["mask_check"]` -- a measurement on what is delivered, read
   by no rule and no prompt. The delivered call's metadata, kept-crops manifest and decision trace are
   copied under a fresh name into `<query_cache_dir>/tool_cache/`, where the
   analyzer takes the newest metadata file per image; the crop paths inside point into the call's own
   directory. The outcome is a `StageResolution`; `parents` / `details` record the tree of trials, the
   rounds, the plans and hypotheses, every call's readouts and metadata path, and the choice.

Validation (deterministic; the shared checks plus the cropper's): the CARD lever names and values of
"Levers" (`Type` / `Range`); per-cell keys and values (`min_containment` at least `RETENTION_FLOOR`,
`auto_tighten` only for a cell in `mask_bbox` mode, `reset` alone in its rule) -- except that a cell a rule
of the move moves out of `mask_bbox` drops the `auto_tighten` it had in the parent (logged as
`dropped_auto_tighten`, stated in the call's change as `auto_tighten_dropped`); an `auto_tighten` the move
itself sets outside `mask_bbox`, or an inherited one on a cell the lever changes alone move out, is
rejected; a change that repeats the
parent's value is dropped (the shared layer); condition readouts that the parent provides (only `==` for
the categorical `margin_mode`, `touches_image_border` and `edge_limited`); a rule without a condition or
that matches no cell is rejected; no configuration that crops every cell like a call already run or
failed, or like a move already in the round; the round cap. The configuration key, `config_key`, is the
crops a configuration cuts: the box of every croppable cell, computed before the call by the tool's own
per-cell geometry (`Single_Cell_Cropper_Tool.cell_crop_box` on the input's mask, `auto_tighten`'s cuts
included), so two configurations are the same call exactly when every cell's box is the same -- e.g.
`auto_tighten` on a cell it does not cut, or `tight` and plain `mask_bbox` where they cut the same square;
a one-pixel difference in one cell is another call. A value is a JSON literal, a plain number (`.1`),
true / false or null; anything else is rejected with the text received. Rejected moves get one repair
question with the reasons, naming the
accepted moves that already run, while the round has room. A reasoner question that fails (a gateway
error or a malformed answer) is tried `REASONER_ATTEMPTS` times, `REASONER_RETRY_SECONDS` apart, before
it counts as failed.

Concurrency: at most `cores // THREADS_PER_CALL` calls at the same time. The tool calls run in
StepVerifier's worker threads, each in its own directory (the crop and mask file names are deterministic
per cell). Measuring stays on the loop's thread: the readouts, the mask check's montages and vision calls, the
manifest and trace writes, and every reasoner question (the batch corrector's runs deadlocked when
several scorings ran in threads of one process).

Failed reasoner questions never generate deterministic tuning calls. Exhausted nonfatal failures
are recorded in `execution_errors`; a failed admission leaves the defaults unsearched, a failed
round ends the search, and a failed final question selects no candidate. Missing required engines
and fatal authentication, permission, model or connection errors propagate. `fallback_rounds` stays zero.

Information level: the loop reads the given image and mask, the per-cell measurements, the GT-free
readouts of every call and, after the choice, the mask check's verdicts on the delivered call. Ground
truth never reaches it; the benchmark scores the delivered crops against the ground-truth masks afterwards (`benchmarks/cropper_NeurIPS22-CellSeg/
benchmark.py --compare`). These readouts are geometric: a lower contamination is a crop that isolates
its target, not a validated gain for downstream (SSL) analysis (see "Honest reporting").

`solve()` path: `integration.verify_managed_tool` hands the input record to the loop (no first call at
the planner's parameters, no OctoTools++ pass, unless the controller is enabled);
`Verifier.resolve_managed_stage` uses `verifier_cls_for(spec, tool_name)` and the verifier's
`make_adapter`; the stage's `param_map` forwards every lever, `cell_overrides`, `min_area` and
`output_subdir` to the tool (before this, `cell_overrides` and `margin_mode` never reached the tool on
this path).

## Levers

What the crops are for: each crop is one cell's input to the downstream single-cell analysis
(`Cell_State_Analyzer_Tool`). A high-quality crop is, in this order: (1) square -- every crop the tool
writes is square (height = width): the analysis resizes every crop to a 224 x 224 square without keeping
its aspect ratio, so a non-square crop's cell would be stretched (`margin_mode` below); (2) one whole
target cell, not cut off; (3) the cell at the crop's center; (4) the cell filling the main area of the
frame; (5) few or no other cells -- neighbours are reduced, never at the cost of 1-4.

Every image-level parameter of `execute()` that shapes the crops is a lever below; the agentic loop sets
them per call, and its cell rules set the per-cell keys (`cell_overrides`, "Per-cell keys" below). Each
lever has one line `Type | Range | Default` (the shared format, `schemas.lever_spec`):

- Type: `number`, `integer`, `bool` or `choice`; "... or null" allows null. Range: `[lo, hi]` or the
  choices. Default: the tool's own default -- the configuration of a move from "input".
- `null` for `margin` = the tool's rule by mask file name (`cell_mask` 1 px, `organoid_mask` 10 px, any
  other name 25 px), resolved on trial 0 and shown to the agent; a caller may fix it.
- The agent sets any value in the range, one or several levers per call; no rule-generated steps
  replace failed reasoning.

Not levers: `min_area` (fixed -- for a cell or nuclei mask the delivering segmentation's `min_size`, else the
tool's default by mask file name; the mask's objects at `min_area` are the given input), `output_format`,
`source_image_id` / `group` / `technical_metadata` (identity), `output_subdir` (each call's own directory,
set by the loop), and the internal constants of `auto_tighten` ("Per-cell keys").

### margin_mode

Type: choice | Range: fraction, tight, mask_bbox | Default: fraction

Which rule sizes each crop; every rule cuts a square of real image pixels.
- `fraction` (the default): a square around the object's bbox, centred on the bbox midpoint (on the
  centroid with `recenter`): the bbox's longer side plus the margin on each side (its half side is
  ceil(longer side / 2) + the margin: one pixel more for an odd longer side); the margin is
  max(`margin`, round(`margin_frac` x the object's equivalent diameter)). More surrounding context, and
  more neighbour pixels in a dense field.
- `tight`: the same square with `tight_buffer_px` as the margin; along the bbox's shorter side the square
  extends past the bbox, which can admit a neighbour lying just outside it.
- `mask_bbox`: the square around the object's mask bbox, its side the bbox's longer side plus 2 x
  `tight_buffer_px`, centred on the bbox midpoint -- the `tight` square without `recenter`, `crop_size`
  and `min_containment` (one pixel smaller for an odd longer side). Along the bbox's longer side nothing
  beyond the bbox (plus the buffer) is included; along its shorter side the square extends past the bbox
  and can hold neighbour pixels there. It is the mode in which `auto_tighten` ("Per-cell keys") cuts the
  square on its neighbour sides; the target is not truncated unless `auto_tighten` cuts it (or its bbox
  is longer than the image's shorter side, below). `margin`, `margin_frac`, `recenter`, `crop_size` and
  `min_containment` do not act in this mode.

Every crop is square (height = width) in every mode, at the image border too. Near the border the
same-size square shifts inside the image, so the target sits off-centre in its crop instead of the crop
being clipped. A square larger than the image's shorter side becomes the largest square that fits; an
object whose bbox is longer than that side fits in no square, and its crop is dropped unless the cell's
`min_containment` (in `mask_bbox`, the 0.90 retention floor) allows the loss. A crop never masks
neighbours out: they stay visible as real context.

### margin_frac

Type: number | Range: [0, 1] | Default: 0.2

`fraction` mode: the margin as a fraction of the object's equivalent diameter (2 sqrt(area / pi));
`margin` is its floor in pixels. A larger fraction adds surrounding context and neighbour pixels.

### margin

Type: integer or null | Range: [0, 128] | Default: null

`fraction` mode: the smallest margin in pixels.

### recenter

Type: bool | Range: false, true | Default: false

`fraction` and `tight` modes: centre the crop on the mask's centroid (its mass centre) instead of the
bbox midpoint. The square is sized from the bbox, so the object stays contained; near the image border
the same-size square shifts inside the image, as close to the centre (the centroid, or the bbox midpoint
without `recenter`) as containment allows. For an asymmetric object the mass centre and the bbox
midpoint differ.

### tight_buffer_px

Type: integer | Range: [0, 32] | Default: 2

`tight` and `mask_bbox` modes: pixels added around the bbox so the target's edge is not flush with the
crop border. More pixels re-admit neighbour context.

### crop_size

Type: integer or null | Range: [24, 512] | Default: null

`fraction` and `tight` modes: a cap on the square (its half side becomes at most max(8, crop_size //
2)); null = no cap. A cap below an object's bbox cuts the object: that crop is dropped (the object has no
crop) unless the cell's `min_containment` allows the loss.

#### Per-cell keys (`cell_overrides`, set by the agentic loop's cell rules)

A cell rule selects cells by conditions on per-cell readouts and sets these keys for them
(`tool.py::_generate_cell_crops`, per `cell_id`); a key a cell does not carry follows the image-level
value.

| key | values | acts in | effect |
|---|---|---|---|
| `margin_mode` | fraction, tight, mask_bbox | every mode | the cell's own margin mode |
| `recenter` | true, false | fraction, tight | the cell's own recenter |
| `auto_tighten` | true, false (default false) | mask_bbox | the tool cuts the square one side at a time -- the side with the most neighbour pixels, to the depth that clears them -- and cuts the same depth off the perpendicular axis, split evenly between its two sides (the target stays centred on that axis), so the crop stays square; each cut keeps at least 0.90 of the target (a cut that would not is made shallower); until the contamination is at most 0.2, a cut helps by at most 0.02, or no cut is left; the applied cuts, the paired ones included, are reported as the crop's `bbox_inset_px`; a cell a rule moves out of mask_bbox loses the auto_tighten it had in the parent |
| `min_containment` | [0.90, 1] (default 1) | fraction, tight | the square may shrink (side x sqrt(min_containment)) while this share of the target stays inside; a crop below it is dropped (the tool reads a value below 0.90 as 0.90) |
| `reset` | true | -- | removes the cell's per-cell keys |

The internal `auto_tighten` constants -- 0.2 (clean), 0.02 (plateau), the 0.90 retention floor
(`_TARGET_RETENTION_FLOOR`) and a 20-step safety cap (`tool.py`) -- are fixed engine constants, not
levers. The retention floor is one number everywhere: no crop the tool writes keeps less than 0.90 of
its target, in any mode.

---

## 2. VERDICT DECLARATION (population triage, REJECT only) -- DATA SCREENS, JUDGE DECIDES; reject = narrow, non-negotiable exception

**The primary product of this tool is a QUALITY-IMPROVED crop, not a keep/reject
verdict.** Reject exists ONLY for a crop too bad to fix by geometry ("Levers") -- STRICTLY
bounded, so it must stay the RARE exception. **The failure mode to avoid (measured, task
038's history, SS "Evidence / provenance" below): every earlier design that judged EVERY
crop, with a reject checked against the data only AFTER the judge answered, either
over-rejected (95%, then 86%), hallucinated an unsupported reject (69% of one run's
`empty_debris` claims), or simply spent tokens on crops the data never flagged in the
first place (12.8M tokens, v7, for ~0 net gain). That is backwards from this tool's
purpose and must not recur.**

- **Verdict type:** population triage (keep/reject, unchanged) PLUS, since task 038 v12
  (human-directed), an OPTIONAL geometry-quality gate (below) -- `mask_bbox`'s own
  ROUND-1 landing and `recenter` both stay 100% PURE-DATA ("Levers"/SS5, task 038 v10,
  still true) -- the judge is never asked whether to APPLY either one; an earlier,
  v9-era per-crop doublet-confirm channel used to sit alongside this section, task 038
  v10 removed it entirely (see "Evidence / provenance" below). **Task 038 v12
  (human-directed) DELIBERATELY REVERSES PART of that v10 "geometry never reaches the
  judge" framing** -- not the round-1 trigger above, but a NEW round-2+ pass (the
  `auto_tighten` cut in "Levers", reported as `bbox_inset_px`; SS5): a real v11 human
  audit of KEPT `mask_bbox` crops found the target often fills only ~30% of its own bbox
  rectangle, a neighbour INSIDE that
  rectangle untouched, and `mask_bbox` idempotent (no further retune ever proposed) --
  the human wants the loop to keep tightening, never discarding a still-dirty crop.
  The "THE BLINDED A/B GATE" bullet below is that mechanism, in its v13 (CURRENT)
  form. **Task 038 v12 R2 REPLACED the gate's OWN first design** (v12's own first
  attempt paired the judge with a per-step DIRECTION ask -- tighten/clean/over_cropped
  -- driving a FIXED-FRACTION data step; the human rejected this: "the logic of this step
  does not work; then what is the point of the VLM?" -- the mask already knows exactly where the neighbour
  pixels are, so a blind fixed fraction was arbitrary, and pairing it with a fixed
  data-step gave the judge no genuine job, a rubber-stamp) with an ABSOLUTE
  quality-control ask instead (DATA drives WHERE/HOW MUCH to cut,
  `contamination_inset_px`, SS3; the VLM judges the tightened RESULT). **Task 038 v13
  (human decision, a methodology critique) then REPLACED R2's own design again, on a
  DIFFERENT axis** -- not what the VLM is asked (still a label-only, non-magnitude
  ask), but HOW its answer is used: `crop_quality_score` was disqualified as the
  ACCEPTANCE authority entirely (the "Task 038 v13 SCOPE" note, "Development notes", explains
  why -- the self-proving trap of using the same score to select AND prove better).
  v13's fix has TWO parts, both below: **(1) acceptance is now a PARETO/CONSTRAINT
  rule** (contamination strictly down AND retention >= the floor AND the VLM does not
  prefer baseline/abstain -> accept; otherwise -> KEEP the baseline, never discard,
  SS5's own `_pareto_scalar`); **(2) the VLM's OWN job changed from an ABSOLUTE
  quality-control ask to a BLINDED, COMPARATIVE A/B ask** -- "baseline vs. candidate,
  which keeps the target more complete while containing fewer other cells?" -- a
  genuinely harder-to-game, more informative question than "does this look good"
  (which the OLD design could rubber-stamp without ever seeing what it was being
  compared against), AND it is now **VoI-GATED** (Value of Information) -- only a cut
  whose outcome is genuinely UNCERTAIN from data alone (SS3's own VoI margin) is ever
  shown to the judge at all; a comfortable, clean cut is accepted on data alone, no
  VLM call (the v7 lesson -- 12.8M tokens, ~0 gain -- applied to this channel too).
  This still follows the SAME "data screens/measures, VLM's schema can express only a
  LABEL never a magnitude, data owns the hard floor" pattern the reject channel below
  already uses, specifically to avoid repeating the v6 hallucination failure mode
  (below) in this channel.
- **DATA SCREENS which crops are even asked (task 038 v9 restructure, RELAXED; task 038
  v10 -- doublet REMOVED from this screen entirely):** a crop is a **VLM CANDIDATE**
  when its OWN mask data crosses EITHER of 2 conditions (SS3) -- `not_truncated < 0.8`
  (cut_off) or `foreground_fraction < 0.2` (empty_debris). A crop crossing NEITHER is
  **auto-kept and NEVER sent to the judge at all** -- not "sent and expected to be
  kept", genuinely never judged. On real NeurIPS22 data both floors each flag ~0% of
  crops.
- **THE GROUNDING FIX (task 038 v9 restructure -- Q3; overlay style updated task 038
  v10, human-locked):** every REJECT candidate's image is sent with its OWN target mask
  drawn on top as a semi-transparent COLORED fill + a bright contour -- a genuinely
  COLORFUL, segmentation-style overlay (`crop_triage._crop_overlay_array`/
  `_overlay_target_on_crop`, reusing `skimage.color.label2rgb`), not a flat tint -- BEFORE
  it reaches the judge, as one labeled cell in a montage (below) -- so the judge can
  reliably tell the OUTLINED target apart from a neighbour or background, mirroring why
  the segmentor's own VLM judge (which always sees a colored mask overlay) does not
  hallucinate.
- **THE JUDGE IS THE AUTHORITY on a screened candidate -- no second-guessing pass after
  it answers.** A cut_off/empty_debris candidate is placed into a labeled position of a
  grid MONTAGE (below) alongside other reject candidates; the judge confirms the
  flagged defect -> reject, or disagrees -> keep (`problem="none"` -- a RESCUE). That
  verdict lands as-is. This is a DELIBERATE departure from every earlier design in this
  CARD's history (below), which either let the judge propose freely and then checked
  the data afterward, or fused a data-derived "strength" label into the SAME prompt the
  judge had to reason past -- both added a layer of complexity that history shows did
  not reliably prevent over- or under-rejection. Screening BEFORE asking, grounding the
  image, then trusting the answer, is simpler and -- because only a data-flagged crop
  is ever a candidate in the first place -- the judge's decision space is already
  narrowed to cases where a defect is plausible.
- **THE MONTAGE FIX (task 038 montage round, human-locked -- REPLACES an earlier
  per-crop reject confirm; `crop_triage._judge_reject_montages`/`_build_montage`):** a
  real v9 cluster run found the per-crop reject ask still FALSE-REJECTS a good cell
  judged in isolation -- cell_00554's 5 rejects were all GOOD spindle cells flagged
  `empty_debris`, because a genuinely thin/elongated single cell's own
  `foreground_fraction` is naturally low (SS3's screening floor) WITHOUT the cell being
  empty/debris at all. Every cut_off/empty_debris candidate is instead BATCHED into a
  labeled grid montage (default 3x3, `CropVerifier.triage_montage_grid` -- 2x2/3x3/4x4
  are all supported): each grid cell is that candidate's own GROUNDED overlay (the SAME
  mask-outline fix above) plus a drawn INDEX label; ONE judge call scores the WHOLE
  montage (`MontageVerdict`, one `MontagePositionVerdict` per labeled position), so the
  judge SEES many candidates at once and can COMPARE them -- catching a data-screen
  false-positive a per-crop ask structurally cannot. `problem="none"` on a flagged
  position is a RESCUE (kept, not rejected) -- the exact cell_00554 case that round
  fixed. Verdicts land BY INDEX (the grid position actually drawn on the image); a
  `crop_id` mismatch is logged and the INDEX still wins (the index is what the judge
  actually looked at, `crop_id` only a restated cross-check). This was this tool's ONLY
  VLM channel from task 038 v10 through v11 -- task 038 v10 removed the earlier,
  SEPARATE per-crop doublet confirm entirely (see "Evidence / provenance" below); task
  038 v12 (below) adds a SECOND, independent channel. **Every montage actually SENT
  is ALSO persisted to disk** (task 038 v10, human-locked -- `crop_triage.
  _save_montage_png`, under `_montages_dir_for(metadata_path)`, e.g.
  `<tool_cache_dir>/montages_<hash>/montage_00.png`) so a human can see EXACTLY what the
  judge saw; each verdict's own trace record (`write_triage_trace`) carries the SAME
  `montage_path` its own index/crop_id maps back into. A no-engine or no-candidate run
  writes no montage files at all.
- **THE BLINDED A/B GATE, VoI-GATED (task 038 v13, human-directed -- `crop_triage.
  ab_recommend_crop`/`ABVerdict`; RETIRES v12 R2's own absolute-QC tighten gate,
  `tighten_recommend_crop`/`TightenVerdict`; SEPARATE and INDEPENDENT from the reject
  channel above -- its own candidate set, its own schema, its own montage subdirectory
  `montages_ab_<hash>/`):** a BLINDED, COMPARATIVE ask, not an absolute quality
  question. `tool.py::_compute_crop_qc` still computes `contamination_inset_px`
  (UNCHANGED since R2, already floor-capped there) -- but only a GENUINELY UNCERTAIN
  cut ever reaches the judge: `verifiers/crop.py::CropVerifier.adjudicate`'s round-2+
  pass VoI-SCREENS each STILL-contaminated `mask_bbox` cell's CURRENT, already-measured
  state (SS3's own VoI margin -- `target_retention` within `_VOI_RETENTION_MARGIN` of
  the floor, OR stuck at the floor still contaminated); a COMFORTABLE cut (clean data
  read, headroom to spare) commits DIRECTLY, no VLM call at all. Only the VoI-screened
  subset goes into a batched, labeled grid montage where EACH tile is one cell's own
  `[baseline | candidate]` pair -- `tool.py` renders BOTH as real images (the plain
  mask_bbox box AND the tightened result, no history threading needed), BOTH with the
  SAME colourful overlay machinery the reject channel already uses, L/R order
  RANDOMIZED per tile (a position-bias guard -- the mapping is recorded and
  de-randomized before landing the verdict). The judge answers ONE comparative
  question, per labeled position: which panel keeps the target MORE COMPLETE while
  containing FEWER other cells -- `left_better`/`right_better`/`abstain`, de-randomized
  into `"candidate"` (COMMIT the next data-computed cut, exactly as it would be
  headless, loop continues) or `"baseline"`/`"abstain"` (no clear winner -> REVERT
  FULLY to the plain mask_bbox baseline, dropping the ENTIRE accumulated inset for this
  cell back to `{}`, and FREEZE -- KEPT at its own safer baseline geometry, never
  discarded). **The v6-hallucination guardrail (below) applies here too, structurally:**
  the judge's own schema (`ABPositionVerdict`) cannot express a magnitude or a pixel
  count at all -- only one of the 3 comparative labels -- so DATA alone measures the
  cut depth and owns the hard `target_retention` floor, BOTH already computed BEFORE
  the judge is ever asked; a `"candidate"`-preferred verdict can never push retention
  below the floor, because the candidate set was already data-screened to exclude that
  possibility. **Opt-in on `llm_engine` (same hook as the reject channel); headless (no
  engine wired) falls back to committing every VoI-screened cut directly** -- the loop
  still genuinely iterates without a judge, converging to data-clean or the floor, just
  without the extra comparative confirmation. Never discards a crop for still being
  contaminated -- a cell that cannot clear the floor, or whose tightening the judge
  does not prefer, simply FREEZES at its own best-effort/baseline geometry ("Levers"/SS5),
  per the human's own "rather than just throwing it away" directive. **Replaces v12 R2's own design**
  (an ABSOLUTE quality-control ask -- good_single_cell/over_cropped/still_multi_cell --
  on the candidate ALONE, judged every round with no VoI gate), which a methodology
  critique found insufficient because `crop_quality_score` (the metric an absolute
  "does this look good" verdict implicitly reinforces) cannot be its own validator
  ("Task 038 v13 SCOPE" note, "Development notes") -- the comparative, blinded ask is harder to
  rubber-stamp, and the VoI gate keeps the token cost bounded to only the genuinely
  uncertain cuts.
- **Reject semantics:** a HIGH-confidence reject **EXCLUDES the crop from the downstream
  analysis population** (the `Cell_State_Analyzer_*` training set). Keep -- including
  every non-candidate, auto-kept crop -- is RETAINED. Keep is the OVERWHELMING default
  outcome; reject is the exception, both because most crops are never even candidates,
  and because a genuine candidate is still, on balance, more often confirmed keep. A
  contaminated/doublet crop is NEVER a reject/exclusion -- it is handled entirely by
  the PURE-DATA `mask_bbox` geometry override instead ("Levers", SS5), a categorically
  different, non-VLM mechanism.
- **Exclusion rule:** `not vision_reject` **ONLY** (`llm_recommend.write_kept_manifest`,
  the shared triage-lander the cropper's thin adapter calls -- task 038 M-B). A
  non-candidate crop never has `vision_reject` set at all, so it is correctly counted
  kept by this SAME rule -- no special-casing needed.
- **Keep-biased, confidence-gated:** only a HIGH-confidence, CLEARLY unfixable defect
  rejects; a LOW-confidence reject is treated as a keep. "Clearly unfixable" means
  geometry ("Levers") cannot rescue it -- if tight-margin/mask_bbox or recentering WOULD fix
  the problem, it is NOT reject grounds; it is geometry's job ("Levers"/SS5) instead.
- **2 REJECT reasons ONLY** (this list is EXHAUSTIVE, not illustrative -- nothing else
  is grounds for a reject). `problem` is a CLOSED enum of exactly these 2 + "none" --
  `cut_off` / `empty_debris` -- there is literally no other value to name:
  cut-off (target obviously cut off at the crop edge, a large piece missing, AND
  geometry cannot recover it) · empty_debris (there is essentially NO cell mask in the
  crop at all -- `foreground_fraction` near 0 -- effectively empty/debris-looking, NOT
  merely a minority-share target next to a tolerated neighbour). **A doublet/
  contaminated crop is NOT a reject reason -- it is a SEPARATE, PURE-DATA geometry
  mechanism** (task 038 v10, human-locked, "Levers"/SS5): a contaminated crop
  (`neighbor_contamination > 0.2`) is geometry's job to fix, landed DIRECTLY from that
  data (`mask_bbox`, "Levers") -- NO judge involved at all, unlike the earlier v9 design
  (see "Evidence / provenance" below). **`saturated` is NOT a reject reason** -- like
  blur, an ACQUISITION/EXPOSURE issue the cropper neither causes nor can fix, not this
  tool's call. **`severe_blur` is NOT a reject reason, at ANY confidence:** blur is an
  ACQUISITION/FOCUS problem the cropper neither causes nor can fix -- rejecting on it
  achieves nothing within this tool's control, and a real run (v4) found the judge
  HALLUCINATING it 178/323 times on crops with a GOOD quant sharpness (0.38-0.65, well
  above the 0.005 anomaly floor -- SS3), repeatedly misreading the sharpness scale
  itself (e.g. calling 0.64 "anomalously low"). **off-center is NEVER a valid reject
  reason, at ANY value** -- off-center is the `recenter` geometry trigger's job ("Levers"/
  SS5, pure data, no judge involved). **`target_dominance`/`neighbor_contamination`, at
  ANY value, are also NEVER valid REJECT grounds** (SS3's own note explains why -- the
  v6 hallucination finding stands unchanged). This is a DIFFERENT claim from "never a
  geometry input", though: `neighbor_contamination` DOES drive the PURE-DATA
  `mask_bbox` trigger (SS5) since task 038 v10 -- but that is a deterministic,
  non-VLM decision the judge is never asked to reason about, so the v6 hallucination
  failure mode (a JUDGE misreading dominance as reject evidence) cannot recur there. A
  reject naming off-center, blur, saturated, doublet, or target_dominance/
  neighbor_contamination is a CARD violation the judge must never produce; the correct
  verdict is KEEP.

**Evidence / provenance (history, abbreviated -- git history holds the full detail of
each round; this CARD documents only the CURRENT, human-locked design above):** a
keep-biased vision judge alone reaches balanced-acc **0.857** on 017's synthetic defect
set (catching blur 0->0.89, doublet 0->0.97, truncation 0->0.75) vs a fixed numeric QC
rule's **0.593** (catches only saturation) -- but on REAL data the standalone vision
judge drops to **0.5375** (041), and the fixed pixel gate ALONE over-rejects ~48% of real
crops -- together the original motivation for fusing vision with quant evidence rather
than either channel alone. Since then, real cluster runs found, in order: the fused
judge over-rejecting **95%** of crops in an early impact run (0 rescues, confirmed
over-rejection by human spot-check, not inferred from the rate alone); a **86%** (277/323)
over-reject after a taxonomy shrink meant to fix that (178 FALSE `severe_blur` claims
overriding a GOOD quant sharpness, 27 structurally-impossible `empty_debris` claims); an
**OVER-KEEP** the opposite direction after the reject taxonomy shrank further (a
severely edge-clipped `not_truncated=0.733` crop kept, several sub-0.40-dominance crops
kept); a **v6** run where feeding the SAME two mask floors used today into the prompt as
advisory "evidence" text, checked against the judge's answer only AFTERWARD, still let
the judge hallucinate `empty_debris` on 69% of its own rejects (`target_dominance` was
wrongly read as a signal there, even though SS3 already declares it purely
informational) and blanket-retune 128 cells net-negative; a **v7** run
(`cell_00994`, base commit 25070bc) that judged EVERY crop (262/262) regardless of
whether the data ever flagged anything, at a cost of 12.8M tokens for ~0 net geometry
gain, and never asked about geometry at all; and a **v8** FOLLOW-UP run (the SAME v7
design) that confirmed the geometry gain STAYED ~0 even after the token fix -- diagnosed
as square-tight's own inability to isolate an adjacent neighbour (`mask_bbox`, then a
rectangle tight to the bbox, did; since the human rule of 2026-09-30 every crop is a square,
`mask_bbox` included, and its square-preserving `auto_tighten` cuts remove the neighbour,
"Levers"), plus the 10%-margin naive baseline already sitting close to its own
ceiling. **The v9 design was the direct response to that full history:** screen with
the data FIRST (so only a plausibly-defective/plausibly-doublet crop is ever a
candidate), GROUND the image with a mask overlay so the judge can actually tell target
from neighbour (the missing piece every EARLIER round lacked), trust the judge's
verdict WITHOUT a second data check afterward on EITHER axis (so there is no
after-the-fact mechanism for a subtle miscalibration to hide behind), and give the
agent genuinely MORE headroom (20% starting margin, a real rectangular-bbox geometry
lever -- a square since 2026-09-30) rather than re-tuning thresholds on the same ~0-gain mechanism -- v9 also added
a per-crop VLM doublet confirm (`n_complete_cells>=2` detected the candidate, the
judge decided tighten/leave). **A real v9 cluster run then found the per-crop reject
ask itself still false-rejecting a good cell judged in isolation** (cell_00554: 5 GOOD
spindle cells flagged `empty_debris`, their own naturally-low `foreground_fraction`
misread as "empty" with no other candidate to compare against) -- the MONTAGE round was
the direct response: batch reject candidates into a labeled grid so the judge can
compare many at once, the SAME "ground the image, then trust the answer" principle
applied to a POPULATION of candidates rather than one crop in isolation. **A v10 QC
AUDIT then found the v9 doublet-confirm channel's own real-world coverage was poor:**
only 12.5% of crops got isolated even though 79% carried `neighbor_contamination>0.2`
-- because `n_complete_cells>=2` structurally MISSES a PARTIAL neighbour (merely
clipped at the crop edge, the dominant real-world case), which is exactly what the
"genuinely WHOLE second cell" trigger was designed to require. **Task 038 v10 (above,
this round) is the direct response:** REMOVE the VLM doublet channel entirely and land
`mask_bbox` DIRECTLY from `neighbor_contamination > 0.2` -- since `mask_bbox`'s own
geometry (then a rectangle tight to the bbox) was a SAFE, deterministic isolation
regardless of whether the excluded neighbour was whole or partial, and kept the whole
target, no VLM judgment was ever actually needed to apply it; the earlier VLM gate was
pure, unnecessary overhead that also happened to systematically under-trigger. (Since
the human rule of 2026-09-30 `mask_bbox` is the square around the bbox: it excludes a
neighbour beyond the bbox along its longer side, `auto_tighten`'s square-preserving cuts
remove the rest, and a bbox longer than the image's shorter side is truncated by the
largest square that fits -- "Levers".)

---

## 3. CALIBRATION (good-crop ranges -- what is anomalous)

The judge's own quant numbers are **NOT** a 0.5-is-the-midpoint scale in absolute terms.
These ranges (fit on real good crops, EXCEPT `centered` -- see its own bullet below) are what
"normal" looks like, so a number can be told apart from a genuine, unfixable defect. This is
the ONE place these numbers are written down; `quant.py` computes the raw scores but holds no
calibration constants itself. **Every threshold below is a CONSERVATIVE, rare-firing
floor/ceiling by design -- it must fire ONLY on the clearly-unfixable, not on ordinary
crop-to-crop variation; if a threshold is rejecting a large fraction of a real batch, the
THRESHOLD is miscalibrated, not the batch.**

- **sharpness (blur):** real in-focus crops span **~0.004-0.40**, HEAVILY right-skewed
  (median ~0.026, p5~0.0054) -- below **~0.005** is anomalously blurry (this floor sits AT the
  empirical p5, i.e. it fires on <=5% of a representative good population by construction).
- **single_object (target intactness):** **1.0** = the target is ONE connected piece; **~0.5**
  = the target's OWN mask is fragmented into 2+ significant pieces (a segmentation artifact --
  an exact cut at the n=2 boundary, not a fit-to-data float). **NOT a doublet signal** -- a
  MERGED doublet (segmented as one blob) scores 1.0 same as a genuinely intact single cell; a
  doublet with two labels shows in `neighbor_contamination` (below), a merged one in no mask
  readout.
- **target_dominance (mask-pixel share vs neighbours; `neighbor_contamination` is its
  near-complement, ≈`1 - target_dominance`, own bullet below) -- PURELY INFORMATIONAL TO THE
  JUDGE, NEVER a reject input:** **~1.0** = target owns the crop; below **0.5** = target is a
  literal minority ("swamped"). Recall axis 5 says "minimal", not zero -- a crop with a
  visible-but-minority neighbour is GOOD, not anomalous, **at ANY dominance value, including
  very low, as REJECT grounds**. `target_dominance` is reported to the judge purely for context
  -- it drives NEITHER the reject screen NOR anything the judge itself decides;
  `problem="target_dominance"`/"swamped"/"dominance" is not itself a value the judge's schema
  can produce. Its near-complement `neighbor_contamination` DOES drive geometry -- see its own
  bullet below -- but as a PURE-DATA trigger the judge is never asked to reason about.
- **neighbor_contamination (= 1 - target_dominance; fraction of the crop's cell
  foreground belonging to OTHER cells, whole OR partial) -- the PURE-DATA trigger for
  `mask_bbox` geometry, NEVER a reject input:** **~0** = target owns the
  crop; **above 0.2** (`verifiers/crop.py::CropVerifier._MASK_BBOX_CONTAMINATION_MIN`) lands
  `margin_mode="mask_bbox"` DIRECTLY -- no judge involved. It counts a PARTIAL neighbour
  clipped at the crop edge as well as a WHOLE second cell. `mask_bbox` is the square around
  the bbox, which excludes a neighbour beyond the bbox along its longer side only --
  `auto_tighten`'s square-preserving cuts take on the rest. Computed by
  `tool.py::_compute_crop_qc` (`other / (fg + eps)`, the SAME `fg`/`other` counts
  `target_dominance`/`foreground_fraction` already derive from -- a genuinely different
  QUESTION about the SAME pixels, not a re-derivation).
- **foreground_fraction (ALL cell-mask pixels in the crop, target's own + any neighbours', /
  crop area) -- the SCREENING floor for `empty_debris` (a reject reason):** **~1.0** = the crop
  is almost entirely cell mask; near **0** = essentially NO cell mask present at all (empty
  background/debris -- the ONLY honest "is a cell here" measure). **Below ~0.2**
  (`crop_triage._EMPTY_DEBRIS_FOREGROUND_MAX`) makes this crop an `empty_debris` CANDIDATE --
  it is sent to the judge (GROUNDED with the mask overlay) with a targeted confirm instruction;
  AT OR ABOVE it, the crop is never sent for `empty_debris` at all. Computed by
  `tool.py::_compute_crop_qc` (`fg / crop_area`, the SAME `fg` count `target_dominance` already
  derives from -- a genuinely different QUESTION about the SAME pixels, not a re-derivation).
- **contamination_side (`top`/`bottom`/`left`/`right`/`None`) -- the PURE-DATA DIRECTION input
  to the round-2+ `bbox_inset_px` tighten step (`auto_tighten`), NEVER asked of the judge:**
  which crop edge carries the MOST non-target foreground mass, relative to the target's own
  centroid -- non-target pixels are split above/below the centroid row and left/right of the
  centroid column, and the axis+direction with the larger mass wins (`None` when there is no
  non-target foreground at all, e.g. a plain, uncontaminated crop). Computed by
  `tool.py::_compute_crop_qc`, deterministic, no VLM involved in computing it -- the judge (the
  blinded A/B gate, when VoI-screened as uncertain) only ever sees the RESULT (a coloured
  overlay already tightened so far, compared against its own baseline) and reports a
  comparative LABEL on it, never computes or is told the direction itself.
- **contamination_inset_px -- the PURE-DATA MAGNITUDE input to the round-2+ tighten step,
  NEVER asked of the judge:** the pixel DEPTH to inset `contamination_side` by so the
  non-target foreground reaching furthest toward the target on that side is excluded -- scans
  inward from that edge (the SAME centroid-relative half used to pick `contamination_side`)
  and finds how deep that side's own non-target fg reaches, then returns the depth that
  clears it, CAPPED so `target_retention` never drops below `_TARGET_RETENTION_FLOOR`
  (best-effort PARTIAL removal when full exclusion would breach it, never a full discard).
  `0` means either nothing to cut (`contamination_side is None`) or no headroom left at all
  (already at/below the floor) -- the verifier reads either case as "freeze, nothing further
  to propose." The mask already knows exactly where the neighbour pixels are, so the data
  sets the depth. Square-preserving cuts (`tool.py::_clamp_directional_inset`): the tool
  pairs this cut with an equal cut on the perpendicular axis, split evenly between its two
  sides (the target stays centred on that axis), so the crop stays square, and reports both
  in `bbox_inset_px`; this single-side projection does not count the paired cut's target
  pixels, so `auto_tighten` measures each paired box against the floor and makes a breaching
  cut shallower (its binary search), and a requested `bbox_inset_px` is scaled back the same
  way.
- **target_retention (alias of `mask_containment` under a purpose-built name for this
  lever) -- the HARD FLOOR input to the round-2+ tighten step, NEVER a reject or judge
  input:** target-mask pixels inside the CURRENT crop box / target-mask pixels in the full
  bbox, in `(0, 1]`. **`_TARGET_RETENTION_FLOOR = 0.90`** (the target may be clipped by at
  most 10%), defined once, in `tool.py` (the tool-side backstop that scales back an
  over-aggressive inset request, `auto_tighten`, the floor
  `contamination_inset_px` itself caps against, and the lower bound of a cell's
  `min_containment`), and imported by `verifiers/crop.py`: `CropVerifier` (which does not
  re-check it: `contamination_inset_px` arrives capped), the agentic loop's `RETENTION_FLOOR`
  (its guard) and `is_crop_valid`'s truncation gate (`_CONTAINMENT_MIN`) -- the agent layer
  imports the tool, the tool never imports `shape.agent.*`. No cut, VLM-confirmed or
  headless, may ever push a cell's actual retention below this floor; a cell that cannot
  clear it after the tool-side scale-back still KEEPS its best-effort inset (never dropped)
  and the verifier FREEZES (stops proposing further growth) once `contamination_inset_px`
  reports 0.
- **VoI (Value-of-Information) retention margin
  (`verifiers/crop.py::CropVerifier._VOI_RETENTION_MARGIN = 0.05`) -- the SECOND tuning knob
  alongside the floor, NOT a separate absolute number:** a cut whose `target_retention` has
  landed AT OR BELOW `_TARGET_RETENTION_FLOOR + _VOI_RETENTION_MARGIN` (0.95 at the current
  0.90 floor) is genuinely UNCERTAIN -- it ate close enough to the target that a human/VLM
  check is warranted -- and becomes a candidate for the blinded A/B gate. This band TRACKS the
  floor (expressed as an offset, not a fixed window), so raising or lowering the floor moves
  the VoI trigger with it. The SECOND, independent VoI trigger is "stuck at the floor still
  contaminated" (`contamination_inset_px<=0` while `neighbor_contamination>0.2`) -- a partial
  cut that did not fully solve the problem, regardless of how comfortable its own retention
  number looks.
- **n_complete_cells (count of DISTINCT labels, including the target, whose FULL-FIELD
  mask is >=90% contained in this crop AND whose full-field area is >= `min_area`) --
  DIAGNOSTIC ONLY, a trigger for nothing:** **1** = only the target is a genuinely whole
  cell in this crop (the ordinary case); **>=2** = a genuinely SEPARATE, WHOLE second (or
  more) cell also sits inside this crop box, not merely a tolerated partial-edge fragment.
  Reported for visibility/trace only -- geometry reads `neighbor_contamination` instead
  (above), which catches BOTH whole and partial neighbours, unlike this count (which
  structurally misses a partial one). Computed by `tool.py::_compute_crop_qc` via
  `np.bincount` over the full-field labeled mask (counted once per tool call) -- see that
  function's own docstring for the exact mechanics.
- **not_truncated (1 - border-touch) -- the SCREENING floor for `cut_off` (a reject reason):**
  **~1.0** = fully inside; below **~0.95** = target noticeably touches the edge -- but `margin`
  / `margin_frac` act only in "fraction" mode (the shipped default; INERT for a cell already
  moved to "tight"/"mask_bbox" via `cell_overrides`), not automatically reject grounds. **Below
  ~0.8** (`crop_triage._SEVERE_NOT_TRUNCATED_MAX`) makes this crop a `cut_off` CANDIDATE -- it
  is sent to the judge (GROUNDED with the mask overlay) with a targeted confirm instruction; AT
  OR ABOVE it, the crop is never sent for `cut_off` at all.
- **centered (target's centroid vs crop center) -- RETUNE-ONLY signal, NEVER a reject input:**
  **~1.0** = target sits at the crop's own center; below **~0.70** (equivalently
  `centroid_offset` >= 0.30, the SAME threshold `is_crop_valid`'s existing off-center gate
  already uses -- `verifiers/crop.py::_CENTROID_OFFSET_MAX`) = noticeably off-center. Unlike
  the other ranges above, this one is not fit on good-crop data: it is the off-center gate's
  threshold (0.30). A low value is NEVER, on its own or combined with anything else, grounds to
  REJECT the crop; `problem=off_center` is not even a value the judge's schema can produce. **A
  SEPARATE, DATA-ONLY trigger for the off-center geometry recenter** (NOT the SAME number as
  the 0.30 gate above -- a DIFFERENT consumer, a DIFFERENT purpose): `centroid_offset > 0.20`
  (`CropVerifier._OFF_CENTER_MAX`) -- `CropVerifier.adjudicate` reads this value DIRECTLY and
  applies `recenter=True` to that cell, ONLY when its `neighbor_contamination` does NOT already
  trigger `mask_bbox` this iteration (above -- a contamination-triggered cell's centering is
  SUBSUMED by `mask_bbox`'s own geometry directly, pure data); the judge is never consulted for
  this trigger, at any point. A residual off-center reading on a crop already moved to
  `recenter=True` is more likely the image-border clamp than a missing lever.

---

## 4. COMPLEMENTARITY / RELIABILITY (quality assessment on the 7 axes; reject is the rare exception)

Each quant signal measures exactly ONE quality axis -- do not read a signal as evidence
for a defect it cannot see: **sharpness = focus · single_object = is the target intact
(one piece)? · target_dominance = does the target occupy the main area (INFORMATIONAL
ONLY) · neighbor_contamination = the PURE-DATA `mask_bbox` round-1 geometry trigger
(task 038 v10) · contamination_side/contamination_inset_px/target_retention = the
round-2+ `bbox_inset_px` tighten step's own DIRECTION/MAGNITUDE/FLOOR inputs (task 038
v12, R2 for the magnitude, `auto_tighten` in "Levers", SS3) · not_truncated =
is the target fully contained? · centered = is the target centered? ·
foreground_fraction = is there a cell mask here at all? · n_complete_cells =
DIAGNOSTIC ONLY (task 038 v10 -- no longer drives anything, SS3)?**

- **blur -> NEVER reject grounds, at ANY severity** (`sharpness` stays purely
  DESCRIPTIVE) -- blur is an ACQUISITION/FOCUS problem this crop did not cause and this
  tool cannot fix; rejecting on it achieves nothing within this tool's control.
  `severe_blur` is not in the reject taxonomy at all -- the judge's own schema cannot
  even produce `problem=severe_blur`. Vision is ALSO demonstrably UNRELIABLE at judging
  focus on a tiny, low-contrast grayscale crop by eye -- a real run found it
  hallucinating blur 178/323 times on crops whose OWN quant sharpness was 0.38-0.65
  (SS3's own good-crop range, nowhere near the 0.005 anomaly floor), repeatedly
  misreading the scale itself. `sharpness` still feeds the judge as evidence (and is
  still one of `crop_quality_score`'s 5 quality axes, the benchmark yardstick) -- it is
  simply NEVER a basis for a reject verdict.
- **saturated -> NEVER reject grounds either, at ANY severity** -- like blur, an
  ACQUISITION/EXPOSURE issue this crop did not cause and this tool cannot fix. A
  blank/near-empty crop where there is essentially NO cell mask at all is `empty_debris`
  reject grounds instead (below) -- a SEPARATE defect (no cell present, not an exposure
  problem) -- do not conflate the two.
- **doublet/contamination -> NOT a reject reason -- a geometry TIGHTEN instead**
  (task 038 v10, human-locked, SUPERSEDES the earlier v9 VLM-confirmed design;
  "Levers"/SS2/SS5): **`neighbor_contamination > 0.2`
  (`verifiers/crop.py::CropVerifier._MASK_BBOX_CONTAMINATION_MIN`, SS3) lands
  `margin_mode="mask_bbox"` DIRECTLY, ROUND 1, NO VLM involved** -- since the human rule
  of 2026-09-30 a SQUARE crop around the target's own mask bbox (its longer side +
  2 x `tight_buffer_px`; the v10 design was a rectangle tight to the bbox), which
  EXCLUDES any neighbour beyond the bbox along its longer side, WHOLE or PARTIAL alike,
  and does not truncate the target (unless the bbox is longer than the image's shorter
  side); a neighbour beside the bbox's shorter side can stay inside the square, for
  `auto_tighten`'s square-preserving cuts ("Levers"). This
  deliberately no longer distinguishes "genuine, separable doublet" from "a partial
  neighbour clipped at the edge" the way the earlier v9 VLM confirm tried to --
  `mask_bbox`'s own geometry (with `auto_tighten`'s cuts) treats BOTH cases identically,
  so that distinction is no longer needed to decide whether to apply it (a real v10 QC audit
  found requiring it was the earlier design's own blind spot: only 12.5% of crops
  isolated despite 79% carrying `neighbor_contamination>0.2`). **A neighbour sitting
  INSIDE the target's own bbox rectangle survives round-1 `mask_bbox` untouched** (a
  real v11 human audit finding, "Levers"/SS2) -- task 038 v12 (human-directed) adds an
  OPTIONAL, round-2+ `bbox_inset_px` DIRECTIONAL tighten, gated by the SEPARATE
  TIGHTEN VLM channel (SS2) when `llm_engine` is wired, data-only headless otherwise;
  see the `auto_tighten` per-cell key ("Levers") and SS5. geometry CAN fix a genuine or partial
  doublet this way; it just cannot fix a MERGED doublet (segmented as one blob by the
  upstream segmentation, invisible to `neighbor_contamination`/`n_complete_cells`/
  `single_object` alike, since the mask itself never distinguished the two cells) --
  that case is upstream segmentation's problem, not this tool's.
- **target fragmentation -> QUANT leads** (`single_object`) -- its OWN, distinct signal: the
  target's own segmented mask split into 2+ significant pieces (a segmentation artifact, e.g.
  an over-split nucleus) -- a DIFFERENT defect from a doublet (two cells, one or two labels)
  AND from `empty_debris` (below); do not conflate these three. Not itself one of the 2
  reject reasons (SS2) -- fragmentation alone, without an independently-flagged
  cut-off/empty_debris, is never reject grounds.
- **cut-off (truncation) -> DATA SCREENS, JUDGE DECIDES:** `not_truncated < 0.8` (SS3,
  RELAXED task 038 v9 restructure -- was 0.5) makes a crop a `cut_off` CANDIDATE -- ONLY
  THEN is it sent to the judge (as one labeled position in a grid montage, GROUNDED
  with the mask overlay, SS2), with a targeted confirm instruction naming this crop's
  own `not_truncated` value (SS2). At or above 0.8, this crop is NEVER sent for
  `cut_off` at all -- the judge's opinion on it is simply never solicited. Before
  confirming, the judge's reasoning should still consider whether geometry (padding, in
  "fraction" mode only -- INERT for a cell already moved to "tight"/"mask_bbox" via
  `cell_overrides`) could plausibly recover the missing piece; confirm ONLY when a
  genuinely large piece is missing regardless.
- **empty_debris -> DATA SCREENS, JUDGE DECIDES:** `foreground_fraction < 0.2` (SS3,
  RELAXED task 038 v9 restructure -- was 0.02) makes a crop an `empty_debris` CANDIDATE
  -- ONLY THEN is it sent to the judge (as one labeled position in a grid montage,
  GROUNDED with the mask overlay, SS2), with a targeted confirm instruction naming this
  crop's own `foreground_fraction` value (SS2). At or above 0.2, this crop is NEVER
  sent for `empty_debris` at all. `target_dominance` plays NO role in this screen, at
  any value -- it measures a DIFFERENT question (population-share among cells that ARE
  present), and using it here was a real v6 cluster run's own root cause for
  hallucinated rejects (SS2).
- **off-center -> PURELY DATA-DRIVEN geometry recenter, never a reject, never asked of
  the judge** (`centered`/`centroid_offset`) -- `centroid_offset > 0.20`
  (`CropVerifier._OFF_CENTER_MAX`, SS3) is read DIRECTLY by `CropVerifier.adjudicate`,
  which applies `recenter=True` to that cell WHEN its `neighbor_contamination` does NOT
  already trigger `mask_bbox` this iteration (SS3) -- a contamination-triggered cell's
  centering is SUBSUMED by `mask_bbox`'s own geometry directly, pure data, no VLM.
  `centered` is **NEVER**, under any circumstance, reject grounds -- off-center is not
  one of the 2 reject reasons (SS2), and the judge's schema cannot even produce
  `problem=off_center`. A residual off-center reading on a crop already moved to
  `recenter=True` is more likely the image-border clamp than a missing lever.

**CATEGORICAL RULE (non-negotiable): off-center, on its own or combined with anything
else, is NEVER reject grounds -- the verdict for an off-center-only crop MUST be
`keep`.** This is enforced structurally too (the judge's schema cannot even produce
`problem=off_center`, SS2) -- this rule is stated here as the judge's own REASONING
check, not only a schema limitation. **`target_dominance`/`neighbor_contamination` are
the SAME as far as REJECT is concerned: at ANY value, including very low/high, neither
is ever reject grounds, and the judge is never even asked to reason about either one
for geometry, at either the round-1 trigger OR the round-2+ tighten** -- REJECT is
decided ENTIRELY outside the judge's geometry involvement (task 038 v10): only
`foreground_fraction`/`centroid_offset` can drive a REJECT verdict (SS3);
`neighbor_contamination` drives `mask_bbox`'s round-1 landing PURELY as data, never via
the judge's own reasoning. The task 038 v13 BLINDED A/B gate (SS2) is a DIFFERENT
verdict schema entirely (`left_better`/`right_better`/`abstain`, de-randomized into
`candidate`/`baseline`/`abstain` -- never `keep`/`reject`) and never reads
`target_dominance`/`neighbor_contamination` as reject evidence either -- it only ever
sees the ALREADY-tightened overlay paired against its own baseline and reports a
comparative LABEL on it, never a number. `n_complete_cells` is now DIAGNOSTIC ONLY
(task 038 v10) -- it drives nothing. Blur and saturated are not in the taxonomy at all
and can never enter this list; a genuine doublet/contaminated crop is geometry's job
(above), never reject.

---

## 5. The linear loop's geometry policy (`CropVerifier`)

The linear `CropVerifier` (StepVerifier's linear loop -- the opt-in OctoTools++ controller and the
`run_shape_crop.run_shape_arm` benchmark arm use it) proposes geometry by fixed rules, one retune per
round, over the accumulated `cell_overrides` (carried from the tool's own `parameters.cell_overrides`
echo, key-merged so a cell never loses an earlier key):
- round 1, per crop not yet on `mask_bbox`: `neighbor_contamination > 0.2`
  (`_MASK_BBOX_CONTAMINATION_MIN`) -> `{margin_mode: mask_bbox, auto_tighten: true}`; otherwise
  `centroid_offset > 0.20` (`_OFF_CENTER_MAX`) -> `{recenter: true}`. `expected_effect` =
  `mean_dominance` up when a contamination cut committed, else `mean_centroid_offset` down;
- round 2 and later, per `mask_bbox` crop not yet frozen: an uncertain tightened state (retention
  within `_VOI_RETENTION_MARGIN` 0.05 of the 0.90 floor, or stuck at the floor still contaminated,
  with a baseline render) is queued for the blinded A/B gate (section 2; off by default,
  `enable_ab_channel`); a verdict for the baseline or an abstention reverts the cell to the plain
  `mask_bbox` square and freezes it (`tighten_frozen`);
- every crop of either round is square ("Levers", `margin_mode`): round 1's `mask_bbox` is the square
  around the target's bbox and `auto_tighten` pairs each cut with an equal cut on the perpendicular
  axis, split evenly between its two sides (the per-cell keys table);
- `scalar_score` = `_pareto_scalar` (the mean dominance, a floor-violating or vetoed crop counting
  -1); StepVerifier's keep-best and its reconcile stop (`hypothesis_invalidated`) apply, and the loop
  converges when nothing new changes.

The agentic loop leaves these choices to the agent; the linear geometry rules do not replace failed reasoning.

## Loop settings

Every setting of the agentic crop loop, with its value and where it lives: the shared layer's in
`base.py` (`AgenticVerifier`, the same for every tool), the cropper's in `verifiers/crop.py`
(`CropAgenticVerifier`), the tool's in `tool.py`. The consistency test (`verifiers/test/test_crop.py`)
checks each value against the code.

| setting | value | where | meaning |
|---|---|---|---|
| `BUDGET` | 8 | `shape/agent/verification/base.py` | rounds at most, a limit and not a target (each round: the calls the reasoner runs at once, at most the concurrency cap); trial 0 (the input) is not a round |
| `PATIENCE` | 3 | same | rounds in a row without progress, a round of failed calls included -> plateau stop, a backstop (`adaptive_training.early_stopping_patience(BUDGET)`; the cropper's progress: an admissible call better than every earlier admissible call beyond the noise on one of `objectives`) |
| `REASONER_ATTEMPTS` | 3 | same | tries of one reasoner question before it counts as failed |
| `REASONER_RETRY_SECONDS` | 30 | same | wait between two tries |
| `THREADS_PER_CALL` | 2 | `shape/agent/verification/verifiers/crop.py` | threads per tool call (`threads_per_call`); concurrency cap = cores // this, at least 1 |
| `NOISE_BAND` | 0.005 | same | the smallest readout difference counted (the used noise is the larger of this and the per-cell paired 2SE) |
| `RETENTION_FLOOR` | 0.9 | same | the final-choice guard: every delivered crop keeps at least this share of its target; also the `min_containment` range and the uncroppable test (imported: `tool.py::_TARGET_RETENTION_FLOOR`, the single source) |
| `PANEL_CELLS` | 9 | same | cells in each call's crop panel (the same cells for every call) |
| `PROMPT_IMAGE_PX` | 1024 | same | longest side of the input overlay and of each crop panel in a round prompt (and of the input overlay in the final prompt) |
| `FINAL_IMAGE_PX` | 512 | same | longest side of each crop panel in the final-choice prompt |
| `NN_GAP_CAP_PX` | 32 | same | `nn_gap_px` is measured up to this many pixels |
| `EDGE_TOLERANCE_PX` | 1 | same | `edge_limited`: a crop box this close to the image edge is flush, a centroid this close to the crop centre needs no move; the admission does not count an edge-limited crop as off-centre |
| `FINAL_PANELS_MAX` | 16 | same | crop panels attached to the final-choice prompt at most |
| `MASK_CHECK_CAP` | 27 | same (`CropVerifier.s1_candidate_cap`) | crops the mask check judges on the delivered call at most |
| `score_key` | mean_target_dominance | same | a call's `scalar_score` (comparison and conservative final selection) |
| `objectives` | mean_target_dominance, mean_target_retention, mean_centered | same | Pareto membership and the plateau signal (both over the admissible calls) |
| `admission_call` | true | same | round 1 is the tool at its defaults (`default`); the admission on that call decides whether the tuning starts ("Agentic VRR", step 1; the shared default is false) |
| `CropVerifier._MASK_BBOX_CONTAMINATION_MIN` | 0.2 | same (the linear `CropVerifier`, section 5) | the admission's contamination count: a crop of `default` with `neighbor_contamination` above this, which the linear loop's round-1 rule re-runs in `mask_bbox` |
| `CropVerifier._OFF_CENTER_MAX` | 0.2 | same (the linear `CropVerifier`, section 5) | the admission's centring count: a crop of `default` with `centroid_offset` above this, which that rule recentres when the crop is not contaminated |
| `_TARGET_RETENTION_FLOOR` | 0.9 | `shape/tools/single_cell_cropper/tool.py` | the cropper's one retention floor (fixed engine constant): `auto_tighten`'s cuts, the `mask_bbox` backstop, `min_containment`'s lower bound; `verifiers/crop.py` imports it (`RETENTION_FLOOR`, `CropVerifier._TARGET_RETENTION_FLOOR`, `is_crop_valid`'s `_CONTAINMENT_MIN`) |
| `_MASK_BBOX_CONTAMINATION_MIN` | 0.2 | same | `auto_tighten` stops at this contamination (fixed engine constant) |
| `_TIGHTEN_PLATEAU_EPS` | 0.02 | same | `auto_tighten` stops when a cut helps by this much or less (fixed engine constant) |

---

## Other QC gate thresholds (per-input knobs, `is_crop_valid`)
`is_crop_valid`'s three pixel-quality cuts default to a fixed fallback prior but are
agent-tunable per input -- pass an override when the input's modality/contrast context
makes the prior wrong for this batch.

### containment_min  (fallback prior 0.90, the retention floor)
- **Prior:** `_CONTAINMENT_MIN` = `tool.py::_TARGET_RETENTION_FLOOR`: a crop is `truncated` when it
  keeps LESS than the prior of its target (a crop at the prior keeps it), the same reading as the
  agentic guard's.
- **Signal:** mask-path crops from a low-contrast or otherwise atypical modality, where a
  cell's true boundary is inherently soft (containment estimates run lower even for a
  well-cropped cell).
- **Action:** lower the cutoff for that input.
- **Effect:** fewer false `truncated` rejects on that modality; a genuinely truncated crop
  (grossly below the prior) still fires.
- **`mask_bbox`-mode exemption (task 038 v12, human-locked, the §2.5 containment
  hazard):** a `margin_mode="mask_bbox"` crop is EXEMPT from this gate entirely
  (`verifiers/crop.py::is_crop_valid`), regardless of `containment_min`'s own value --
  its `mask_containment` MAY legitimately drop below any cutoff once a directional
  `bbox_inset_px` (the `auto_tighten` cut, "Levers") has been applied, which is the INTENDED
  effect of that lever (bounded separately by its own `_TARGET_RETENTION_FLOOR=0.90`, task 038 v13,
  SS3/SS5), not a
  truncation defect. A PLAIN, uninsetted `mask_bbox` crop's
  own containment is exactly 1.0 (unless its bbox is longer than the image's shorter side,
  where the largest square that fits truncates it and the 0.90 retention floor keeps or
  drops the crop), so this exemption is a no-op until an inset is actually applied.

### min_dynamic_range  (fallback prior 0.05)
- **Signal:** an inherently low-contrast modality/stain (e.g. faint fluorescence) where the
  fixed floor would flag real signal as blank.
- **Action:** lower the cutoff for that input.
- **Effect:** fewer false `blank_low_contrast` rejects; a genuinely blank/flat crop still
  fires.

### max_saturation  (fallback prior 0.5)
- **Signal:** a modality with a naturally higher bright-pixel fraction (e.g. a saturating
  stain by design) where the fixed ceiling would flag normal images as blown out.
- **Action:** raise the cutoff for that input.
- **Effect:** fewer false `saturated` rejects; a genuinely blown-out crop still fires.

On single-channel HPA these fall back to the fixed prior (no per-input reasoning signal
to act on); the per-input tuning above is exercised on multi-modal data.

## Other reject gate: `is_crop_keep` (`crop.py`, task 042) -- lenient, KEEP-BIASED
A companion to `margin_mode="tight"`: drops a crop ONLY on a "big problem" neither margin
lever can fix. NOT the same thing as `is_crop_valid` above (containment/off-center/pixel
quality) -- this is a SEPARATE, narrower, DETERMINISTIC gate for blur / target-swamp /
fragmentation, not wired as a hard default; a caller opts in. Also NOT the same thing as the
vision ARBITRATION (SS2-4 above) -- `is_crop_keep` is a deterministic pre-check some callers
use instead of/alongside the LLM judge; its thresholds are the SAME numbers as SS3
CALIBRATION, restated here as concrete defaults. Same task 038 M-C framing applies: these
defaults are already calibrated LENIENT (see below) -- do not tighten them without
re-measuring against a real good-crop population first.

### sharpness_min  (default 0.005)
- **Signal:** `quant.sharpness` for this crop is far below what real, in-focus crops score
  (SS3 CALIBRATION: real good-crop scores span ~0.004-0.4, heavily right-skewed).
- **Action:** raise ONLY if this batch's real crops still show many undetected blurry
  ones (check the batch's own good-crop sharpness floor first, the same way
  `_good_only_threshold` was calibrated in task 035's GT-free quant eval -- that
  benchmark harness was removed in the 038 benchmarks/ reorg; see git history or
  `archives/035_cropper-gtfree-quant/` for the original calibration).
- **Effect:** more blur caught; more false rejects on ordinary sharpness variation.

### single_object_min  (default 0.5)
- **Signal:** the target's OWN per-cell mask has >=2 significant (>=40% of the largest)
  connected pieces -- the target itself is fragmented, not a tolerated neighbour.
- **Action:** rarely needs tuning (it is an exact cut at the n=2 boundary, not a
  fit-to-data float); lower only if 1-significant-piece crops are still being flagged.
- **Effect:** raising catches only worse (3+ piece) fragmentation; lowering is stricter.

### target_dominance_min  (default 0.5)
- **Signal:** the MASK-based `target_dominance` (from `tool.py::_compute_crop_qc`, e.g.
  `crop_metadata["qc"]["target_dominance"]`) is below 0.5 -- OTHER cells outnumber the
  target's own pixels in the crop; the target is a literal minority ("swamped"), not just
  sharing the frame with a tolerated neighbour.
- **Action:** lower for a batch where sub-0.5 dominance is common but the target is still
  recognizable (dense fields, per the ceiling above); this is the ONE narrow exception to
  "dominance is never a gate" -- do not widen it into a general dominance cutoff.
- **Effect:** raising rejects more crops with a real second cell present.

### min_dynamic_range / max_saturation  (reused unmodified from `is_crop_valid` above)
Same fallback priors, same per-input tuning guidance as the QC gate section above --
`is_crop_keep` calls `is_crop_valid`'s pixel gates rather than reimplementing them.

## Notes
- OFF-CENTER crops are fixed by RECENTERING on the mask centroid ("Levers", PURELY
  DATA-DRIVEN via `cell_overrides`, no judge -- unless the same cell's own
  `neighbor_contamination` ALSO triggers `mask_bbox` this iteration, in which case
  `mask_bbox`'s own geometry SUBSUMES it directly, pure data, task 038 v10, "Levers"/SS5),
  never treated as reject grounds (`mean_centroid_offset`/`centered` are the metrics).
- `target_dominance` / `neighbor_contamination` are reported for visibility but are NOT
  a REJECT gate -- neighbours are TOLERATED (axis 5: "minimal", not zero), so never
  shrink the crop to chase them beyond `margin_mode="tight"`/`"mask_bbox"`.
  `target_dominance` is PURELY INFORMATIONAL in the live VLM triage (SS2/SS3/SS4) --
  NEVER a reject input, at ANY value (a real v6 run's own root cause, SS2).
  `neighbor_contamination` DOES drive the PURE-DATA `mask_bbox` geometry trigger (task
  038 v10, above) -- but as data the judge is never asked to reason about, not via the
  VLM triage at all. The ONE remaining exception is `is_crop_keep`'s own
  `target_dominance_min` above -- a SEPARATE, currently-unused-in-production
  deterministic gate (task 042).
- `empty_debris` reject grounds (SS2/SS4) come from `foreground_fraction` -- a
  SCREENING floor below `crop_triage._EMPTY_DEBRIS_FOREGROUND_MAX=0.2` (RELAXED task 038
  v9 restructure -- was 0.02) -- not from dominance; `cut_off` from `not_truncated <
  crop_triage._SEVERE_NOT_TRUNCATED_MAX=0.8` (RELAXED -- was 0.5); `mask_bbox`'s ROUND-1
  landing (a geometry action, never a reject) from `neighbor_contamination > 0.2`, PURE
  DATA, no VLM at all (task 038 v10, SUPERSEDES the earlier `n_complete_cells>=2`-gated,
  VLM-confirmed trigger) -- its OPTIONAL round-2+ `bbox_inset_px` follow-up (task 038
  v13) is the ONE geometry step that now CAN involve a judge, as a VoI-GATED,
  COMPARATIVE (blinded A/B) check only, never for the cut's own direction or magnitude
  (both data-computed, `auto_tighten` in "Levers", SS2) -- and only for the genuinely UNCERTAIN subset; a
  comfortable cut commits on data alone, no VLM call at all.
- Every REJECT candidate sent to the judge is GROUNDED with its own target mask drawn
  on top as a COLORFUL, segmentation-style overlay (`crop_triage._crop_overlay_array`/
  `_overlay_target_on_crop`, task 038 v9 restructure, colored fill added task 038 v10)
  -- the fix that lets the judge reliably tell target from neighbour/background instead
  of hallucinating. That grounded overlay is placed as one labeled cell in a batched
  grid MONTAGE (task 038 montage round, `crop_triage._judge_reject_montages`) rather
  than sent alone -- see SS2's own "THE MONTAGE FIX". This was this tool's ONLY VLM
  channel from task 038 v10 through v11 (round-1 geometry never reached the judge at
  all); task 038 v12 added a SECOND, independent VLM channel for the round-2+ tighten
  step (SS2's "THE BLINDED A/B GATE" in its CURRENT, v13 form) -- the SAME colorful
  overlay/montage machinery is reused for BOTH panels of each A/B tile, persisted in
  its own `montages_ab_<hash>/` subdirectory so it never collides with the reject
  channel's own montages for the same trial. Every montage actually sent, on EITHER
  channel, is also PERSISTED to disk (task 038 v10, `_montages_dir_for`) for a human
  to inspect exactly what the judge saw.
- `quant.py::crop_quality_score`'s COMBINED score weights ONLY `not_truncated`/
  `target_dominance`/`centered` (task 038 v9 restructure, `_WEIGHTS`) -- the 3 axes this
  tool's own geometry levers actually adjust; `sharpness`/`single_object` are still
  computed/reported but excluded from the weighted sum (no lever here ever moves them).
  **`_WEIGHTS` re-weighted task 038 v12 (human-directed):
  `{not_truncated: 0.25, target_dominance: 1.0, centered: 0.25}`** (sum 1.5, was
  `{1.0, 1.0, 0.5}` sum 2.5) -- a plain `mask_bbox` crop's `not_truncated` is a
  constant-1.0 guard under this mode (the bbox is exactly the target's own extent, never
  truncated by construction, "Levers"), so weighting it EQUALLY to `target_dominance` (the
  axis the round-2+ tighten step is actually optimizing) diluted the measured effect of
  a genuine gain; `centered` CAN legitimately drop under a directional inset (an
  asymmetric inset moves the crop's own center away from the target's centroid) so it
  stays weighted, just down, so the target_dominance/centered TRADEOFF still shows up
  as a net score change rather than being
  swamped by an axis that cannot move either way. **Task 038 v13 (human-directed):
  these values now MOOT as a ranker** -- `_WEIGHTS`/`crop_quality_score`/
  `mean_crop_quality_score_proxy` no longer drive ACCEPTANCE at all (SS2/SS5's own
  Pareto/constraint rule replaces that role entirely; `CropVerifier.scalar_score` is
  now `_pareto_scalar`, SS5) -- this reweight is exactly the FIRST of the two grounds
  the "Task 038 v13 SCOPE" note ("Development notes") cites for disqualifying the composite as a
  validator in the first place (a re-weighting done specifically to make a gain SHOW
  UP is not independent evidence that gain is real). The values are left UNCHANGED
  (harmless, still informational/diagnostic) rather than reverted, since reverting
  them would not itself fix the underlying disqualification -- the fix is removing
  their ACCEPTANCE authority, not re-tuning their own numbers again.

---

## Honest reporting (task 038 v13, human decision)

A run summary / mechanism-metrics report for this tool should state: **contamination
reduction** (`neighbor_contamination` before vs. after, per cell and in aggregate),
**`target_retention` distribution** (how close to the `_TARGET_RETENTION_FLOOR` the
kept crops actually landed), **VLM call counts** (REJECT channel + the BLINDED A/B
gate, separately -- SS2), **abstain rate** and **baseline-preferred/reverted count**
for the A/B gate specifically (`crop_triage.ab_recommend_crop`'s own returned
`{"judged", "verdicts"}` summary already carries the per-position `verdict` field
needed to compute these), and **`tool_calls`/token cost**. These are all MECHANISM
facts, honestly reportable without any downstream validation.

**Do NOT emit or headline a `crop_quality_score` (or `mean_crop_quality_score_proxy`)
delta as evidence that this tool's own tightened crops are "better."** Per the
"Task 038 v13 SCOPE" note ("Development notes"), that composite is disqualified as a validator
on two grounds (it was reweighted toward the conclusion; even un-tuned, mask geometry
is not the actual task objective) -- reporting a delta in it would silently reintroduce
the self-proving trap this round removed from the ACCEPTANCE decision. `tool_calls_per_
image` is EXPECTED to rise relative to a purely data-driven design (the VoI-gated A/B
channel adds calls for genuinely uncertain cuts) -- report this honestly as a COST, not
hide it. **The "is this crop actually better [for downstream representation learning]"
verdict is explicitly DEFERRED to a SEPARATE SSL/analyzer validation task, NOT answered
by this tool or this CARD.**

**`benchmarks/crop_NeurIPS22-CellSeg/run_shape_crop.py` (task 038 v13, a coordinator
follow-up) now complies:** its own summary reports a `"mechanism"` section
(`neighbor_contamination` reduction, `target_retention` distribution, a
`n_geometry_reverts` proxy for the A/B gate's own baseline-preferred/abstain outcomes --
see that script's own `_count_geometry_reverts` for the honest caveat on what this proxy
can and cannot distinguish from a literal per-channel VLM call/abstain count, which the
runner does not separately instrument without touching the frozen verifier internals)
as its PRIMARY reported content; `crop_quality_score` is STILL computed but its own
`"quality"` section now carries an explicit `"authoritative": False` marker.

---

## Development notes (not in the mask-check judge's prompt)

The mask check mask-error judge (`crop_triage._build_mask_check_montage_prompt`) reads only "## Purpose" and
"## 3. CALIBRATION" of this CARD (`crop_triage._cropper_judge_rules`): the rule text it needs. The
history, dates, task references, token counts and findings behind those two sections are kept here for
maintainers. The other triage channels' prompts (the reject montage, the A/B gate, S1) still carry the
whole CARD (`crop_triage._cropper_card`).

**Purpose.** The purpose and the 5-axis quality order are task 038 M-C's (human redirect, 2026-07-12).
Square, always is the human rule of 2026-09-30: `mask_bbox` was a rectangle tight to the target's mask
bbox until this rule, so in a dense field a square holds more neighbour pixels than the old rectangle did.
Since the task 038 v9 restructure the COMBINED `crop_quality_score` weights only the 3 axes the geometry
levers adjust, so sharpness/single_object (computed and reported, diagnostic) no longer dilute the "how
much did the agent's own action help" measure, since no lever here ever moves them.

**This corrects a real failure mode an impact run exposed: the vision+quant fusion rejected
95% of crops (MORE than a vision-only judge, ZERO rescues) -- reject had become the DOMINANT
behavior, exactly backwards from this tool's purpose.** SS2 restates the reject bar as
conservative and strictly bounded specifically to prevent this recurring.

**Task 038 M-A/M-B design note:** this CARD is the single source of truth for the shared-judge
contract -- Levers / VERDICT DECLARATION / CALIBRATION / COMPLEMENTARITY-
RELIABILITY / the linear loop's geometry policy (SS5). `quant.py` holds GT-free numbers ONLY, no calibration
constants; the numeric ranges of SS3 are the single copy other consumers (the shared
judge's prompt, `llm_recommend.build_judge_prompt`; `is_crop_keep`'s deterministic defaults)
read or cite, never re-embed.

**Task 038 v13 SCOPE (human decision, methodology critique -- READ BEFORE citing this tool's
own numbers as evidence of "better crops"):** this tool's own conclusion is REDUCED to a
**safe CANDIDATE-GENERATOR + constraint/Pareto SELECTION**. It reports MECHANISM metrics
honestly (contamination reduction, retention distribution, VLM call/abstain/revert counts,
cost -- SS "Honest reporting" above) and does **NOT** claim its own tightened crops are
"better" than the naive/mask_bbox baseline in any downstream sense. Why: `quant.py::
crop_quality_score` is DISQUALIFIED as that validator, on TWO independent grounds -- (a) task
038 v12 reweighted its own `_WEIGHTS` toward the conclusion the geometry levers were already
producing (`quant.py`'s own comment); (b) even an UN-TUNED
mask-geometry score is simply not the actual task objective -- only a DOWNSTREAM SSL/analyzer
run can validate whether a tighter crop genuinely helps representation quality, and **that
validation is a SEPARATE, NOT-YET-BUILT task, explicitly OUT of 038's own scope.** Using the
SAME (reweighted) score to both SELECT which crop to keep AND PROVE it is better is the exact
self-proving trap task 048's own history already showed the danger of -- v13 removes that
coupling here: acceptance is now a PARETO/CONSTRAINT rule (SS2/SS5), not a score comparison,
and the composite `crop_quality_score`/`mean_crop_quality_score_proxy` no longer drives
keep-best at all (Q1, SS5). **The "is this crop actually better" verdict is DEFERRED to the
SSL task, not answered here.**

**3. CALIBRATION.**
- The ranges were fit on task 035's real good-crop eval set
  (`archives/035_cropper-gtfree-quant/results/per_crop_scores.csv`, 43 real non-synthetic base crops,
  `centered` excepted); the conservative, rare-firing design of every threshold is task 038 M-C's.
- sharpness: the ~0.005 floor already sits AT the empirical p5 -- do not tighten it further without
  re-measuring.
- target_dominance is never a reject input (human-locked) because a real v6 cluster run showed feeding it
  to a JUDGE as reject evidence was the WRONG use: 59/86 hallucinated `empty_debris` rejects had dominance
  >= 0.40 (several = 1.000), and a 128-cell blanket-tighten regression was driven by exactly this signal (a
  JUDGE misreading it). Task 038 v10 (human-locked) made its near-complement `neighbor_contamination` a
  PURE-DATA geometry trigger, so the v6 failure mode (a judge misreading it) structurally cannot recur there.
- neighbor_contamination has been the `mask_bbox` trigger since task 038 v10 (human-locked). It REPLACES the
  v9 `n_complete_cells>=2`-gated, VLM-confirmed trigger: a real v10 QC audit found that trigger's own
  real-world coverage poor -- only 12.5% of crops isolated despite 79% carrying `neighbor_contamination>0.2`
  -- because most contamination is a PARTIAL neighbour clipped at the crop edge (`n_complete_cells=1`),
  invisible to a "genuinely WHOLE second cell" trigger. `mask_bbox`'s own geometry (the v10 rectangle tight
  to the bbox) isolated a contaminated crop regardless of whether the neighbour was whole or partial and kept
  the whole target, so no VLM confirmation was ever actually needed; since the human rule of 2026-09-30
  `mask_bbox` is the square around the bbox.
- foreground_fraction: the `empty_debris` floor (0.2) was RELAXED in the task 038 v9 restructure (it was
  0.02). On real NeurIPS22 data this floor flags ~0% of crops.
- contamination_side dates from task 038 v12.
- contamination_inset_px (task 038 v12 R2, human-directed) REPLACES an earlier fixed-fraction step the
  verifier used to project on its own. Human rationale: "the logic of this step does not work; then what is
  the point of the VLM?" -- the mask already knows exactly where the neighbour pixels are, so a blind fraction
  was arbitrary. The square-preserving paired cut is the human rule of 2026-09-30 and its even split the
  human decision of 2026-09-30.
- target_retention dates from task 038 v12. `_TARGET_RETENTION_FLOOR = 0.90` is task 038 v13's (human
  decision -- raised from 0.85, a MORE conservative floor for the "safe candidate-generator" scope: the target
  may be clipped by at most 10% now, not 15%); `contamination_inset_px` caps against it since task 038 v12 R2.
- The VoI retention margin is task 038 v13's (human decision); the band tracks the floor so the two do not
  drift out of sync across future tuning rounds.
- n_complete_cells has been DIAGNOSTIC ONLY since task 038 v10, which SUPERSEDES the earlier v9-era
  doublet-candidate trigger; its structural blind spot for a partial neighbour was the direct motivation for
  the v10 change.
- not_truncated: the `cut_off` floor (0.8) was RELAXED in the task 038 v9 restructure (it was 0.5). On real
  NeurIPS22 data this floor flags ~0% of crops.
- centered: no fresh empirical centroid-offset distribution was measured for `centered` in that task; its
  range is cross-referenced from the existing, already-validated off-center gate threshold (0.30).
- These same ranges back BOTH `is_crop_keep`'s (`crop.py`, task 042) deterministic reject-gate defaults
  (`sharpness_min`/`single_object_min`/`target_dominance_min`, SS "Other reject gate" above -- UNCHANGED, a
  separate, currently-unused-in-production deterministic gate, task 038 follow-up did not touch it) AND the
  shared judge's prompt (`llm_recommend.build_judge_prompt`, which reads this section's text verbatim rather
  than hard-coding its own copy, task 038 M-B) -- one source for both consumers.

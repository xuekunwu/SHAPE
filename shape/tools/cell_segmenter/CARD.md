---
tool: Cell_Segmenter_Tool
category: bioimage_core
stage: image_level
depends_on: []
tunable: true
metadata_consumes: []
metadata_none: the CARD default target; scale and colour measured from the image
metadata_partial: same as none
metadata_full: same as none
---
# Tuning card: Cell_Segmenter_Tool (Cellpose-SAM)

**When to use:** the single adaptive cell/nucleus segmenter (Cellpose-SAM CPSAM) for
phase-contrast / brightfield / DIC / fluorescence microscopy AND nuclei / DAPI / Hoechst
inputs -- it self-adapts across modalities, so use it for whole cells and for nuclei alike.
**When NOT:** use Organoid_Segmenter_Tool for organoids/spheroids (needs a custom model).
Input: a microscopy image. Output: a labeled cell mask + count + overlay.

Instance segmentation of cells with Cellpose-SAM (`cpsam`). Output: a label mask, one
integer id per detected cell. Use this card to choose the parameter change(s) -- one or
several levers per call -- when a segmentation has a clear, actionable problem, from the
levers whose documented effect matches what you see; if the result is already good, change
nothing.

The model is scale-robust, so prefer the smallest change that addresses the problem.
`bsize` is fixed at 256 (Cellpose-SAM requirement) and is NOT tunable.

## Levers

Each lever has one line `Type | Range | Default` (the values the agentic loop validates;
`null` = the tool's own default).

### diameter  (default 30 px; typical 10-200)
Type: number | Range: [5, 200] | Default: 30
Expected cell diameter; sets the scale the model works at.
- **Lower it** when the TARGET cells are clearly SMALLER than the masks, or small cells
  are missed because the scale is too coarse. Consequence if too low: large cells get
  fragmented into several masks.
- **Raise it** when cells are LARGER than the masks, or one cell is split into pieces.
  Consequence if too high: adjacent cells merge into one mask and small cells vanish.
- Effect to expect: lowering shrinks the median mask area; raising grows it.

### cellprob_threshold  (default 0.0; range about -6 to +6)
Type: number | Range: [-6, 6] | Default: 0
Detection sensitivity -- the cell-probability cut for keeping a pixel as foreground.
- **Lower it** when real cells are MISSED (faint cells, low contrast): admits more
  foreground, raising recall and the cell count. Consequence if too low: false masks
  appear on debris/background.
- **Raise it** when there are FALSE masks on background/debris/noise: suppresses weak
  detections, raising precision. Consequence if too high: faint real cells are dropped.
- Effect to expect: lowering raises the cell count; raising removes weak/tiny masks.

### flow_threshold  (default 0.4; the loop's range 0.1-1.5)
Type: number | Range: [0.1, 1.5] | Default: 0.4
Maximum allowed flow error per mask; also governs how readily touching cells are split.
- **Raise it** when well-formed cells are REJECTED (present in the image, no mask) or
  when crowded cells are MERGED into large blobs: accepts more masks and splits merged
  ones. Consequence if too high: noisy / poorly-formed masks slip through.
- **Lower it** when poorly-shaped / spurious masks pass: stricter QC. Consequence if too
  low: real but imperfect cells get rejected.
- Effect to expect: raising reduces the fraction of oversized (merged) masks.

### min_size  (default 15 px area)
Type: integer | Range: [1, 500] | Default: 15
Smallest mask kept; everything below is discarded.
- **Raise it** when there are many tiny spurious masks (specks, fragments). Consequence
  if too high: genuinely small cells are removed.
- **Lower it** only when real cells are tiny and being dropped.
- Effect to expect: raising reduces the tiny-mask fraction.

### niter  (default 200)
Type: integer or null | Range: [50, 3000] | Default: null
Number of dynamics iterations the flow is followed.
- **Raise it** when LARGE or elongated cells are missed or broken because the flow does
  not reach their center. Consequence if too high: slower, with little further gain.
- Lowering it is faster; the flow may then not reach the center of large or elongated cells.
- Effect to expect: raising can recover large missed cells (higher cell count).

### tile_overlap  (default 0.1)
Type: number | Range: [0, 0.5] | Default: 0.1
Overlap of the 256-px tiles the model runs on. Raise it when cells on tile seams are cut or
duplicated (a regular grid of broken masks); costs time.

### augment  (default false)
Type: bool | Range: true, false | Default: false
Test-time augmentation (flipped tiles averaged). Improves faint-cell recall and mask quality
at higher cost.

### resample  (default true)
Type: bool | Range: true, false | Default: true
Run the dynamics at the original resolution (true) or at the rescaled one (false, faster,
coarser boundaries).

### color_class_filter  (default null)
Type: choice | Range: null, keep_high_br, keep_low_br | Default: null
A post-segmentation filter: drop the masks of one color population (an Otsu cut on the
per-mask blue-minus-red value) -- `keep_high_br` keeps the bluer/purple population,
`keep_low_br` the redder/pink one. For a scene where one whole color class is not the target;
no parameter change can remove such a class.

## Default target convention

Whole cells: each cell is one instance, over its full extent. In a field where a few stained /
nucleated cells lie among many unstained or red cells, the stained cells. For a nuclear stain
only, the nuclei.

## Agentic VRR (`SegmentationAgenticVerifier`, StepVerifier wave mode)

Rules measure and guard; the agent decides. `Cell_Segmenter_Tool` is tuned by the agentic VRR
every agentically tuned tool shares, `base.AgenticVerifier` (`shape/agent/verification/base.py`:
the admission, the rounds, validation, the repair question, the plateau, the final choice and the
failure accounting). Its subclass `SegmentationAgenticVerifier` (`shape/agent/verification/verifiers/segmentation.py`,
next to the linear `SegmentationVerifier`) adds the segmenter's own part and is this tool's
verifier through `stage_map`'s `verifier_by_tool`. The organoid segmenter keeps the linear loop,
and so does this tool with the opt-in OctoTools++ controller enabled (that controller's own
system). The settings named here are listed under "Loop settings".

**Call names** (`AgenticVerifier.label`, shared by every agentically tuned tool): each call is named by
who proposed it, once it has run -- `input` (trial 0, the input; not a call), `default` (the tool at its
defaults), `reference` (the system's reference call), `agent_1`, `agent_2`, ... (the agent's calls, in
order); a second default or reference call is
`default_2` / `reference_2`. The prompts, the parents of moves, the final choice and the records use these
names.

0. **Trial 0 is the input image** (`input_record`), a measurement and not a tool call: size,
   channels, dtype, the image QC (`preprocessing.absolute_quality_metrics`) and the preflight
   read of the raw image (`image_context.read_image_context_profile` with the reasoner's engine,
   once per image through its process cache): the specimen context, the visible object classes,
   the provisional target hint and the confidence, or "unavailable (<why>)" (no engine, a failed
   or an unconfident read). The hint follows the preflight prompt's rule
   (`shape/agent/prompts/image_preprocessor/preflight.md`, STEP 5): `nuclei` only for a
   fluorescence image whose signal is a nuclear stain alone, `nucleated_cells` for a stained blood
   smear, `whole_cells` for every other image. The planner's call (at the planner's params) and the
   diameter-correction pass do not run for this verifier (`integration.verify_managed_tool`): round 1
   is the tool at its CARD defaults, and arm A's corrections are the reference call after the
   admission (step 1).
   - **The target** (`target`), stated in the task text of the round prompts and the final-choice
     prompt, before the input facts (so the preflight, an observation, and its
     target hint come after it): the text of the user's request's section headed "## Target" (any
     case; up to the next H1 / H2 heading or a `---` rule), verbatim, as "Target (from the request)";
     without one, the section "Default target convention" above, as "Target (default convention)".
     The request is the solve() query, else the step's sub_goal: `solver._execute_and_verify` ->
     `_run_managed_image` -> `integration.verify_managed_tool(request=)` ->
     `Verifier.resolve_managed_stage(request=)`, which sets it on a segmentation verifier that
     declares `request`. The details record it (`target`: source and text).
1. **Admission** (`admission_call`): round 1 is the tool at the `Default` of every lever above
   (`default`, "the tool at its defaults (the admission call)"), run before the agent is asked; it counts
   as round 1 of `BUDGET`. The admission is the linear loop's deterministic one (`admit_defaults`):
   only a measured problem of `default` starts the tuning, and the agent is not asked (each
   names the readout, its value and the threshold; `admission_problems` beyond the guard, i.e. `default`'s
   `measured_problems`, the same checks any call is measured with): the call
   failed; the guard (step 4: no cells, an invalid overlay); the linear loop's triggers on its
   readouts without its flow gate (`SegmentationVerifier.localize`, unchanged, on the readouts with
   `flow_error` set aside: the design notes; the shape gates
   `_SHAPE_GATES` on `tiny_mask_fraction`, `large_mask_fraction` and `mean_solidity`; not its
   under-detection -- `missed_foreground_fraction` and `count_miss_fraction` count every textured
   pixel no mask covers, the target's or not (unstained cells, background), so they are shown as
   information and are no problem: in the NeurIPS22 CellSeg pilots the 20 searches started on
   under-detection alone found at most +0.002 AP50 over `default`, and in the dev30 run 4 of 5 such images
   were overridden for a worse AP50, the agent growing masks to cut the readout; a dataset whose
   missed cells are real needs this re-measured); arm A's scale check (`mask_diameter_px` off from `default`'s diameter
   by `SCALE_FACTOR` or more, either way: `initial_params.diameter_correction_needed`, the linear
   loop's diameter correction in `integration.verify_managed_tool`); arm A's colour gates
   (`_color_minority`, the gates of `color_class_filter.decide` on the population it keeps): one
   color population of `default`'s masks (`color_class_filter.color_cluster_stats`) holds at most
   `_MINORITY_MAX_FRAC` and at least `_MINORITY_MIN_COUNT` of them, a colour minority ("colour
   minority: 9 of 1679 masks (0.5%) in the high (bluer) population"). Without one `default` is delivered
   as it is (satisfied, `no_tuning_needed`; no question, no round and no final question): `default` is the
   default delivery when no reference call runs (step 4), and no call has strictly fewer than no
   measured problems, so a tuning started without one could not deliver another call. The round
   prompt names why the tuning started, and its first round attaches `default`'s overlay to assess.
   - **The colour evidence** (`_color_evidence`): on
     a colour minority, an image of `default`'s masks on the raw image's own colours -- what arm A's
     vision model looks at, without its question or its rules (arm A's decision is asked separately,
     for the reference call below; its prompt reaches no agentic prompt). Two rows, the high (bluer / purpler)
     population on top and the low (redder / pinker) one below; per row up to `COLOR_TILES` masks,
     evenly spaced in the population's (B - R) order; each tile a square of the raw image around one
     mask (2 x its longer box side, at least `COLOR_TILE_MIN_PX`, clipped to the image; boxes from
     `ndimage.find_objects`), stretched as `render_evidence_png` stretches it, resized to
     `COLOR_TILE_PX` with the mask outlined thinly (on the resized tile); tiles 2 px apart; no text.
     It is attached right after the input image wherever the input image is (the round prompts, the
     final prompt), labelled "COLOUR POPULATIONS default (top row: high, <masks> masks, <fraction>, mean
     B-R <mean>; bottom row: low, ...):"; never without a colour minority. The details record the
     mask ids each row shows (`color_evidence`, null without the image). The preflight (step 0) is
     unchanged.
   - **The reference call** (arm A's corrections of `default`; `reference_moves`): after a RUN, ONE call
     from `default` holding every arm A correction that holds on `default` -- `diameter` set to `default`'s
     `mask_diameter_px` when the scale check holds, and `color_class_filter` set to arm A's colour
     rule when it applies (`_color_rule`: on a colour minority, `color_class_filter.decide` is asked
     with the reasoner's engine -- its own question, rules and gates; its `applied`, keep_high_br
     or keep_low_br, sets the filter). The question is tried as every reasoner question is
     (`REASONER_ATTEMPTS` times, `REASONER_RETRY_SECONDS` apart; decide catches the engine's errors,
     so a try without an answer -- no confidence -- is a failed one). The details record the decision
     as `color_decision`: its action, confidence, reason and applied; or "unavailable (<why>)" when
     no reasoner is wired (not asked) or every try failed (then counted in `reasoner_failures`;
     `SHAPE_REQUIRE_AGENTIC=1` raises the last error, as for the other questions). Its diagnosis
     names each rule as the system's, never "arm A" ("the system's diameter rule (mask_diameter_px
     190.7); the system's colour rule: keep_high_br (<its reason>)"). It runs as `reference`,
     one round before the agent's first (`base.AgenticVerifier`'s reference round): it counts in
     `BUDGET`, its row is in the calls table, and the round prompt's admission line adds "Then the
     system ran: reference (...)". Nothing otherwise (a SKIP, or a RUN where no correction holds).
     Otherwise `color_class_filter` is the agent's lever like any other.
   - **The reference default**: the reference call, when it finished and the guard admits it, is
     the default delivery (step 4); without one -- no reference call ran (no correction holds on
     `default`), or it failed or the guard rejects it -- the call at the defaults, `default`, is the default
     delivery when the guard admits it (`reference_default`; the prompts name it "the call at the
     defaults"). The calls table shows each call's `measured_problems` (`call_problems`): the
     guard's failures (a call without cells or with an invalid overlay), then the checks above on
     that call (`measured_problems`: the shape gates, the scale check against the diameter that call
     used, a colour minority of that call's masks). On a call with `color_class_filter` set (`default` never has it, so the admission's list is
     unchanged), three of them read the filter's own split:
     - its coverage readouts (`PREFILTER_READOUTS`: the under-detection readouts
       `UNDER_DETECTION_READOUTS`, `missed_foreground_fraction` and `count_miss_fraction`, the
       readouts `_under_detected` reads, and `missed_blob_size_ratio`) are measured on its masks
       before the filter, so the masks the filter removed count as covered (`quantify`, from the
       tool's `prefilter_mask_path`, recorded only when the filter removed masks); the other
       readouts stay on the kept masks;
     - no colour-minority check on its kept masks (their populations stay shown, as information);
     - instead, arm A's balance gate on the split the filter made (`_filter_balance`): the
       population it keeps (keep_high_br the high one, keep_low_br the low one) must hold at most
       `_MINORITY_MAX_FRAC` and at least `_MINORITY_MIN_COUNT` of its masks before the filter, the
       fraction as `decide` defines it ("colour filter keeps 1668 of 1679 masks (99.3%), above
       10%"). A call keeping the majority (red-cell) population carries this problem.
2. **Rounds** (one VRR cycle each). The agent (the multimodal reasoner) sees the input image
   (with the colour evidence after it, on a colour minority),
   the overlays of the calls just finished (in round 2, `default`'s; after the reference call, `default`'s and
   `reference`'s;
   and, for comparison, of its best earlier admissible call, else of its best earlier call marked
   not admissible), every finished call's
   configuration, mask QC readouts and `qc_score` (`SegmentationVerifier.quantify` /
   `adjudicate`; `flow_error` is measured and kept in the trial metrics but never shown: not in the
   calls table, the deltas, the predictions or the readout glossary -- the design notes),
   `mask_diameter_px` (`initial_params.diameter_from_mask_area` of the median
   mask area, the linear loop's cell-size measure), the color populations of its masks (`color_class_filter.color_cluster_stats`
   on the raw image: the Otsu split of the per-mask mean blue - red, deterministic), the QC
   deltas vs the parent, each stated prediction next to the observed
   change, its own earlier assessments, the calls that failed, the untried levers, the rounds
   done and left, the seconds of each call (the tool's report) and the wall time. It returns an
   assessment of each attached overlay (`SegPlan.assessments`, a `VerifyReport`: problem,
   severity -- recorded as the trial's `judge_severity`), its hypothesis list and the round's
   calls: at most the concurrency cap, each a parent (`input` = the tool at the `Default` of every
   lever above, or any finished call) and the levers it changes, one or several, as values within
   the lever's Type / Range. Failed predictions are facts; the agent judges them.
   **The narrowing** (`narrowing`): the agent paces its own search, broad then narrow as it judges.
   It may explore first -- several levers in one call, from any finished call -- and narrow when
   the evidence tells it which levers matter: a plan that names `focus_levers` (`SegPlan`) changes
   only those levers that round, combined as the agent chooses. When to narrow, to which levers and
   how far is the agent's decision. The shared layer's narrowing (`base.AgenticVerifier`; the levers
   above carry no Family, so its rule that a first focus needs a lever of every group changed does not
   apply here) checks one rule (`_schedule_problem`, on every agent move; a move
   that breaks it is rejected with the reason, as an invalid value is, and the repair question
   follows): once a round has run under a focus, every later plan names focus_levers again, the
   same levers or fewer -- the focus is never widened. The round prompt states this and where the
   search stands (`schedule_text`); the focus of each round is recorded (`focus_history`).
3. **Stops**: the agent returns no moves when it is satisfied, at any round; with rounds left it is
   asked once more (the shared stop check, `stop_check_text`) and may continue or confirm the stop
   (`stop_checks`). The backstops:
   `PATIENCE` rounds in a row -- the rounds (a round of failed calls included) after the last
   one that improved the best assessed severity -- without an
   improvement (plateau, early stopping; checked after the agent's plan: its assessments are the
   measurement, `plateau_after_plan`); at most `BUDGET` rounds, `default`'s included (a limit, not a
   target; StepVerifier `max_iters` = 1 + budget); no valid move remains. The admission's SKIP
   ends the search before the agent's first round.
4. **Final choice** by the agent among the calls the guard admits: cells detected and a valid
   overlay (`flow_error` is no guard: the design notes). It sees the input image (with the colour
   evidence, on a colour minority) and the overlays (at most `FINAL_IMAGE_PX`) of the admissible
   calls. An inadmissible choice is told why and asked once more. With a reference default, a choice
   of another call is delivered only when it has strictly fewer `measured_problems` than the default
   AND the agent's assessment of it is better (a lower severity) than its assessment of the default
   (`override_allowed`; a call not assessed is not better); else the reason, listing both calls'
   problems and assessments, is answered and the agent asked once more (the same retry). The final
   question first asks the assessment of each admissible call not assessed yet
   (`SegChoice.assessments`, recorded as the rounds' are). The NeurIPS22 CellSeg replay
   (`benchmarks/segmentation_NeurIPS22-CellSeg/runners/replay_shape_delivery.py`, 120 runs): the
   problem count alone overrode 15 times, 12 for a worse AP50 (0.802 vs the default's 0.825); with
   the assessment too, 2 overrides (0.822). When no answer is accepted -- an override refused or an
   inadmissible choice, in either order -- the default is delivered (`reference_default`, satisfied).
   A failed final question selects no candidate and records an execution error. Without a default,
   a choice still inadmissible delivers the admissible call
   with the best (assessment, QC score) (conceded; a call the agent has not assessed counts as
   minor). No admissible call -> no final question, the finished call with the best (assessment, QC
   score) (`least_bad`, conceded). A stopping plan (no moves) may name the call to deliver
   (`deliver`; after a stop check, the confirming answer's): an admissible one -- with a reference
   default, the default or a call `override_allowed` admits -- is delivered without the final
   question (`agent_choice`); otherwise, or with none named, the final question runs as above. The
   details record `reference_default` (the default call -- the reference call or `default` --,
   whether the delivery overrode it, the reason). The round prompts (`final_rule_text`) and the
   final prompt state these rules.
5. **Validation**: the levers of "## Levers" with their Type / Range (`schemas.lever_spec`), no
   repeated configuration, a value that equals the parent's is no change; values are JSON
   literals, plain numbers, true / false or null; one repair question names the accepted moves.
   A failed reasoner question is tried `REASONER_ATTEMPTS` times, `REASONER_RETRY_SECONDS`
   apart for nonfatal failures. Exhausted failures are recorded in `execution_errors`, never replaced
   by deterministic tuning calls. Required questions without an engine and fatal authentication,
   permission, model or connection errors propagate. A failed admission leaves the defaults
   unsearched; a failed round ends the search. `fallback_rounds` stays zero.
6. **Concurrency**: the cap is the SAM server's replicas (`sam_concurrency`: `/health`, summed
   over `$SHAPE_SAM_SERVER_URL`), else 1 (the in-process model runs one call at a time). Calls
   run in worker threads (attempt counter under a lock, PIL figures); scoring stays on the
   loop's thread.

## Cohort mode (`shape/agent/verification/segmentation_cohort.py`)

A set of images segmented by `solve()` runs as a cohort (on by default; `solve(segmentation_cohort=False)`:
every image its own VRR). The per-image VRR above is unchanged; the cohort decides which images run it and
from which parameters.

1. **Groups**: each image's preflight read (`image_context.read_image_context_profile`, the process cache
   the VRR's own preflight then reuses) gives its group, (`specimen_context`, `modality`). An image
   without a confident read, or of an unknown specimen, is ungrouped: its own VRR.
2. **Calibration** (`calibration_size`): a group of fewer than `MIN_GROUP` (10) images is not calibrated:
   every image its own VRR. Else a random sample (seeded by the group and its images, recorded) of
   5% of the first 100 images, 2.5% of the next 900 and 1% beyond, summed and rounded up, at least 3
   (1000 images -> 28), each with its own full VRR, which delivers its result.
3. **Consensus** (`consensus_params`, CPJUMP1's rule) over the agent-tuned levers -- not `diameter`
   or `color_class_filter`, which the measured corrections set per image: a lever transfers when at
   least 75% of the calibration VRRs' delivered calls changed it, a numeric one to their median with a
   relative MAD at most 0.25, a categorical one to their mode with an agreement of at least 75%. A
   lever changed often enough but dispersed makes the consensus unstable: the group's other images
   each run their own VRR.
4. **Apply** (the cohort mode, `cohort_levers`): every other image of the group runs the call at the
   defaults and its admission (SKIP: that call is delivered, as in the VRR); after a RUN, ONE reference
   call -- the image's own measured corrections (the diameter rule, the colour rule) with the
   consensus levers on top (`reference_moves`) --, then no agent round and no final question: the
   reference default is delivered ("cohort_applied").
5. **QC and routing** (`problems_of`): the admission standard on the delivered call -- its guard and its
   measured problems, and a stage that delivered nothing or conceded. An image with a problem runs its
   own full VRR from the consensus (`start_params`: the consensus in place of the CARD defaults); only
   the flagged images are routed, however many (on heterogeneous data a group's share of images that
   need tuning says little about its consensus; the others keep their cohort result).

The ungrouped images and the groups run at the same time, at most `image_workers` images at once across
them (a group's calibration does not hold the others back).

The solver records the cohort (the step's `verification["cohort"]`: the groups, each group's
calibration sample and delivered levers, the consensus per lever, the cohort levers, the problems, the
routing; the counts) and each image's `parameter_source` (`calibration` / `cohort` / `per_image_vrr`).
The CPJUMP1 harness's SHAPE arm segments its images through this cohort (`segmentation_cohort.run`,
`benchmarks/full_CPJUMP1/harness/pipeline.py::_segmentation_cohort`).

The cost-saving stop rules of the organoid verifier (`SHAPE_SEGMENTATION_EARLY_STOP`, `_CONFIRM_STOP`, `_COMPACT_PROMPTS`) are off here by
default (the organoid CARD documents them); set them in the environment to use them with this tool.

Cohort calibration remains enabled by default. `SHAPE_SEGMENTATION_CALIBRATION=skip` bypasses
calibration for eligible groups, applies each image's own corrections with empty consensus, then
routes only flagged images to full VRR. Small groups and uncertain preflight reads keep their
per-image route. The cohort record identifies the policy and the empty calibration explicitly.

## Loop settings

Every setting of the segmentation loop, with its value and where it lives. The consistency test
(`test_segmentation_golden_path.py`) checks each value against the code.

| setting | value | where | meaning |
|---|---|---|---|
| `BUDGET` | 8 | `base.py` | rounds at most, a limit and not a target (the shared agentic VRR); the admission call is round 1 |
| `PATIENCE` | 3 | same | rounds in a row without a better assessed severity -> plateau stop, a backstop (`adaptive_training.early_stopping_patience(BUDGET)`) |
| `REASONER_ATTEMPTS` | 3 | same | tries of one reasoner question |
| `REASONER_RETRY_SECONDS` | 30 | same | wait between two tries |
| `SCALE_FACTOR` | 2 | `verifiers/segmentation.py` | the admission's scale check: `mask_diameter_px` off from `default`'s diameter by this factor or more, either way (arm A's: `initial_params.diameter_correction_needed`'s `factor`); it also starts the reference call |
| `COLOR_TILES` | 8 | same | the colour evidence: masks shown per color population, evenly spaced in its (B - R) order |
| `COLOR_TILE_PX` | 96 | same | the colour evidence: side of each mask's tile |
| `COLOR_TILE_MIN_PX` | 32 | same | the colour evidence: a tile's square is 2 x the mask's longer box side, at least this, clipped to the image |
| `PROMPT_IMAGE_PX` | 1024 | same | longest side of an image in a round prompt |
| `FINAL_IMAGE_PX` | 512 | same | longest side of each overlay in the final-choice prompt |
| `admission_call` | true | `SegmentationAgenticVerifier` | round 1 is the tool at its defaults (`default`); the admission decides on that call whether the image needs tuning |
| `_MINORITY_MAX_FRAC` | 0.1 | `color_class_filter.py` | the admission's colour gates (arm A's kept-fraction gate of `decide`): a colour minority holds at most this fraction of `default`'s masks |
| `_MINORITY_MIN_COUNT` | 3 | same | and at least this many of them (arm A's kept-count gate of `decide`) |

## Development notes (not in the agentic loop's prompts)

The agentic loop's round prompt shows only "## Levers"; these notes on each lever's **Nature**
are kept here for maintainers.

`flow_error` in the agentic loop: `SegmentationAgenticVerifier` measures it (SegmentationVerifier's
readouts; kept in the trial metrics, the trials and the logs for analysis), but it is no guard, no
admission check, no input of the fallback's moves and never shown to the agent; the reason is in the
comment above `SCALE_FACTOR` in `segmentation.py`. The linear `SegmentationVerifier` (arm A) keeps its
flow gate (`flow_gate`) and its diagnosis (`flow_error` at least 0.25 -> raise `flow_threshold`).

Each lever's **Nature** (vrr.md sec 3 point 5 -- Recommend routes the VALUE by lever
nature): `continuous` levers are candidates for the golden-section optimizer (task
032 M5) WHEN re-evaluation is cheap, as it is here (re-segmentation, not retraining).

- `diameter` -- **Nature:** continuous -- but already `computable` in practice: the framework
  measures it from the first pass's mask area (`diameter_from_mask_area` in
  `shape/agent/verification/integration.py`) rather than sweeping it, so it is NOT one
  of the golden-section optimizer's targets. This holds for the linear loop only: the agentic
  loop runs no diameter-correction pass before its call at the defaults (`default`); its admission
  reports the same check on `default` as a measured problem, the system then runs `default` at the measured
  diameter in its reference call (the default delivery), and the agent overrides it only with a
  call of fewer measured problems.
- `cellprob_threshold` -- **Nature:** continuous -- GT-free quant (`quant.py`) is Verify-phase
  evidence only (the LLM may weigh it); the golden-section optimizer (task 032 M5) is NOT wired
  -- the GT-correlation gate resolved NEGATIVE (rho=-0.386 vs mean-IoU@0.5, n=20).
- `flow_threshold` -- **Nature:** continuous -- GT-free quant (`quant.py`) is Verify-phase
  evidence only (the LLM may weigh it); the golden-section optimizer (task 032 M5) is NOT wired
  -- the GT-correlation gate resolved NEGATIVE (rho=-0.386 vs mean-IoU@0.5, n=20).
- `min_size` -- **Nature:** continuous -- GT-free quant (`quant.py`) is Verify-phase evidence
  only (the LLM may weigh it); the golden-section optimizer (task 032 M5) is NOT wired -- the
  GT-correlation gate resolved NEGATIVE (rho=-0.386 vs mean-IoU@0.5, n=20).
- `niter` -- **Nature:** continuous -- but rarely tuned in practice (only large/elongated-cell
  misses call for it) and NOT a golden-section optimizer target this round.

## Notes
- The linear loop's deterministic faint-cell path may enable `augment`.
- Do NOT treat cells merely TOUCHING as a defect -- that is normal and needs no change.

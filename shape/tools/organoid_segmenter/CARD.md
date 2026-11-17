---
tool: Organoid_Segmenter_Tool
category: bioimage_core
stage: image_level
depends_on: []
tunable: true
metadata_consumes: []
metadata_none: the CARD default target at the checkpoint's native scale
metadata_partial: same as none
metadata_full: same as none
---
# Tuning card: Organoid_Segmenter_Tool (custom CO model)

**When to use:** instance segmentation of organoids / spheroids, each a large
multicellular structure. Requires a compatible, custom-trained CO Cellpose model:
pass `model_path` or set `SHAPE_ORGANOID_MODEL_PATH`. The model stays fixed during
tuning. Use `Cell_Segmenter_Tool` for individual cells or nuclei.

Input: a 2D organoid microscopy image. The tool segments channel 0 as grayscale,
preserving the original intensity range. Output: an integer label mask, organoid
count and overlay. `cell_count` is a compatibility alias for `organoid_count`.

## Levers

Each lever declares `Type | Range | Default`, parsed and validated by the shared
agentic loop. The numeric ranges bound the search, not the biological target.
`null` leaves an optional inference argument at the installed Cellpose default;
the first call preserves the tool's existing inference settings.

### diameter
Type: number or null | Range: [5, 2000] | Default: null
Cellpose rescale reference, not the organoid's pixel size: Cellpose 4 resizes
the image by 30 / diameter before inference. `null` (and 30) keep the CO
checkpoint's native training scale, its calibrated setting. Values below 30
enlarge the image; values above 30 shrink it. Never set it to the measured
organoid width: a 300 px organoid at diameter 300 is shrunk 10x and usually
returns zero organoids. Lower it (enlarge) when organoids look too small for the
model or are missed; raise it modestly (shrink) only when one organoid is broken
into pieces at native scale. Expected effect: changing diameter changes median
mask area.

### flow_threshold
Type: number | Range: [0.1, 1.5] | Default: 0.4
Maximum flow error allowed for a mask. Raise it to retain plausible organoid
masks rejected by strict flow QC; lower it to reject poor/spurious masks. A
permissive threshold can admit false positives; use the overlay to assess them.

### cellprob_threshold
Type: number | Range: [-6, 6] | Default: 0
Foreground probability threshold. Lower it to recover faint organoids or their
edges; raise it to suppress background/debris detections. Too low can include
halos and merge neighbors. Expected effect: lower values often increase count
or foreground coverage; neither increase alone proves better segmentation.

### min_size
Type: integer | Range: [1, 1000000] | Default: 15
Minimum retained mask area in pixels. Raise it to remove debris/fragments below
the organoid scale; lower it if small real organoids were discarded. The upper
range allows organoid-scale debris filtering. Raising it can remove real small
organoids, so verify the dropped population in the image.

### niter
Type: integer or null | Range: [50, 3000] | Default: null
Dynamics iterations. Raise it when flows do not reach the center of large or
elongated structures; more iterations cost time and may give no gain.

### tile_overlap
Type: number or null | Range: [0, 0.5] | Default: null
Inference tile overlap. Raise it when masks are cut or duplicated at regular
tile seams. More overlap costs time; it does not address unrelated misses.

### augment
Type: bool or null | Range: true, false | Default: null
Test-time augmentation. Try true for faint/unstable detections at greater cost.

### resample
Type: bool or null | Range: true, false | Default: null
Run dynamics at original resolution (true) or rescaled resolution (false).
False can be faster but coarsen the boundaries of small structures.

## Default target convention

Every whole organoid or spheroid is one instance over its visible extent.
Constituent cells, internal texture and debris are not separate organoids.
Use the request's nonempty `## Target` section when it specifies a different
target; otherwise use this convention. Lumens, halos, natural size variation
and truncated structures require interpretation from the raw image and overlay.

## Agentic VRR

`OrganoidSegmentationAgenticVerifier` subclasses the shared segmentation
`SegmentationAgenticVerifier` and runs through `base.AgenticVerifier` and
StepVerifier wave mode. `stage_map.verifier_by_tool` selects it for this tool.
No OctoTools++ controller or obsolete `solve(octotools_plus=...)` argument is used.

- Trial 0 measures the input image and its preflight, without a model call.
- Round 1 calls the fixed CO model at these CARD defaults (native scale). The
  segmentation admission checks the result: a failed call, no organoids,
  invalid visual output or a shape-quality complaint starts tuning. Without a
  measured problem, the default result is delivered without a tuning question.
- No scale-mismatch rule applies to this tool. The shared rule compares the
  median mask diameter with the diameter used, which assumes the model was
  trained on 30 px objects; this checkpoint was trained at native scale, so the
  rule flagged correct masks and proposed reference calls that returned zero
  organoids. `mask_diameter_px` remains a size readout only, never a diameter
  value. A reference call is made only for a cohort's calibrated levers.
- The agent sees the raw image, overlays, mask QC and predicted versus observed
  effects; it chooses subsequent inference settings and the final candidate.
  `cell_count` means organoids. Uncovered texture is evidence, not an automatic
  recall target: internal texture, halos and background can inflate it.
- Final delivery uses the existing segmentation guards and reference-override
  rule: replacing a valid reference needs fewer measured problems and a better
  visual assessment. Failure/concession and every trial remain in the stage trace.
- Colour populations and `color_class_filter` are not actions for this grayscale
  tool. Neither its input channel nor its model is an inference-search lever.

The shared limits are 8 rounds (the default and reference calls count), patience
3, and 3 reasoner attempts with 30 seconds between attempts. The in-process CO
model runs one call at a time, independent of CPSAM server replicas; the executor
caches the model and serializes access across images.

The search strategy follows the existing settings, default `legacy` (the shared
agentic VRR). Set `SHAPE_SEARCH_STRATEGY_SEGMENTATION=four_stage` to use
cover -> contrast -> narrow -> confirm. A constructor's `search_strategy` takes
precedence over that setting, then the global `SHAPE_SEARCH_STRATEGY`. Four-stage
search retains the same admission, guard, budget and delivery rules; its confirm
stage does not constitute independent held-out validation.

## Efficiency policy and VRR evidence

The segmentation efficiency policy follows SHAPE `cs-cm@092b25e8`.
Deterministic measurement and rule-computable scale corrections precede agent
search; the following switches are shared by cell and organoid segmentation:

| Setting | Default | Behavior |
|---|---|---|
| `SHAPE_SEGMENTATION_EARLY_STOP` | `1` | Deliver an admissible reference with zero measured problems without a search/final question; `0` restores the historical search. |
| `SHAPE_SEGMENTATION_CONFIRM_STOP` | `valid_choice` | Skip a second stop confirmation only when the plan names an admissible delivery satisfying the existing override rule; `always` restores reconfirmation. |
| `SHAPE_SEGMENTATION_CALIBRATION` | `full` | Calibrate eligible cohorts; `skip` uses empty consensus and routes problem images to per-image VRR. |
| `SHAPE_SEGMENTATION_COMPACT_PROMPTS` | `0` | `1` renders the calls table by columns, preserving all rows and fields. |

Early stopping follows the existing strict fewer-problems selection rule: no
candidate can have fewer than zero measured problems. It does not certify that
the organoid mask is biologically correct or optimal. If the reference fails,
has an invalid overlay or retains a measured problem, this shortcut does not
apply. A missing/invalid stopping delivery still needs confirmation and the final
selection checks remain binding. A required reasoning question without an engine
or after exhausted errors remains visible; no deterministic search silently
replaces the agent.

Verify still measures every actual result. Reconcile compares each changed
candidate to its parent using the stored measurements and stated expectations;
it needs no separate LLM request. `details.reconciliation` retains changes,
expected effects, exact before/after values, observed deltas and the loop's
hypothesis/score labels. Failed calls, missing parent readouts and unstated
predictions remain `failed`, `not_computable` or `unpredicted`, respectively,
never a fabricated successful comparison. A joint scale/consensus change is not
per-lever causal attribution. Remaining problems still reach Recommend, with the
CARD, raw image/overlays, measured effects and the original resource/delivery rules.

Policies and deterministic decisions are recorded in `efficiency_policy` and
`efficiency_events`. Per-question `llm_calls` records phase, attempt, elapsed
time, outcome and provider-reported usage, including failures and uncached
preflights; a missing token report stays unknown. Cohort grouping preflights
outside the verifier and other tool-internal calls still require engine-wide
accounting. None of these cost switches demonstrates CO accuracy/token/time
benefit without an organoid evaluation on fixed data/model/selection.

## Multiple images

`solve(images=[...], enabled_tools=["Organoid_Segmenter_Tool"])` uses the same
cohort mechanism as cell segmentation when there is more than one image and
`segmentation_cohort=True` (default). A sufficiently large, confidently grouped
set is calibrated on a sample; agreed non-scale inference levers are transferred,
and flagged images run their own VRR. Small/ungrouped sets run per image. Cohort
records identify `Organoid_Segmenter_Tool`, and use this CARD's defaults/levers;
CO and CPSAM results are not pooled. `segmentation_cohort=False` runs per-image
VRR directly. `image_workers` controls image concurrency; model calls stay serial.
With calibration skipped, there is no learned consensus to claim: each eligible
image gets its own default checks, and every flagged image keeps its
per-image VRR route. Small groups and uncertain preflight reads retain their
ordinary per-image route under either calibration policy.

Keep `enable_verification=True` and `enable_llm_verifier=True` for agentic tuning.
Configure a vision-capable reasoner as for the existing segmentation VRR. Disabling
the reasoner does not select a deterministic search replacement: a required
reasoning question without an engine fails visibly.

## Direct tool calls and rescue

Direct `tool.execute(...)` calls retain `adaptive_rescue=True`: a zero/near-zero
visual-QC failure triggers up to nine additional fixed rescue probes, with a
default ceiling of 10 calls including the first. The probes stay at native scale
except one enlarged (diameter 15) and one reduced (diameter 60) scale. Set
`adaptive_rescue=False` or `SHAPE_ORGANOID_ADAPTIVE_RESCUE=0` for a single direct
inference. This rescue is
separate from agentic search. Managed VRR calls receive `attempt_id`, disabling
internal rescue so one recorded candidate is one model evaluation. Every attempt
has its own mask/overlay files; the result's selected paths identify the delivered
candidate, while `latest_*` aliases identify the most recently executed one.


## Cohort native-trial reuse

The organoid cohort reuses Cell cs-cm's verified native-trial cache and measured
configuration handling. Only model outputs are reused; every pass still executes
Verify, admission, Reconcile and final-delivery guards. Failed/empty attempts and
changed/missing artifacts are never hits. Native files remain in immutable attempt
directories, with SHA256 checks of the input and every result artifact. Identity
includes actual loaded checkpoint hashes, SHAPE sources, package/device/precision
settings and the scoped RNG seed. A changed recipe/image/model/runtime selects a
new entry. Reused outputs retain original native cost and record actual lookup
cost separately; they do not invent additional native inferences.

solve(segmentation_cohort=True, segmentation_cache_seed=0) enables this only
inside a managed multi-image organoid cohort. Set segmentation_cache_seed=None
to disable reuse. Direct execute calls retain their existing behavior, including
adaptive rescue. Managed calls retain one native inference per new candidate;
an audited cache hit has zero native inferences. Model inference remains serial.

Only calibration deliveries without remaining guard/measured problems contribute
to the organoid consensus. If fewer than three clean calibration results remain,
the other images use full per-image VRR instead. Unstable consensus, uncertain
preflight grouping, small groups and any remaining fast-pass problem retain full
VRR. The fast-pass trace is retained under cohort.groups[].fallback_from when
routing an image to VRR, so its costs and failed candidates are not overwritten.
No Cell size-mismatch correction or colour filter is introduced for CO. Resource
limits and negative-result review are organoid-specific and do not change Cell policy.

## Negative results and bounded execution

A default zero-detection mask no longer mandates searching for a nonempty result.
The managed verifier asks one independent raw-image question, without GT or an
overlay, and records target presence, interpretability, confidence, evidence and
limitations. An interpretable, high-confidence absent answer without limitations
can admit an empty mask. A high-confidence, interpretable present answer permits
the normal repair search even when its visible boundary is uncertain.
Uncertain, malformed or uninterpretable answers retain a review requirement and
stop forced-positive tuning. Constant/nonfinite input cannot certify a negative.
The raw review is reused across inference probes on the same image. Missing model
credentials/engine remain hard errors. Confirmed negative deliveries do not
contribute levers to a positive-target cohort calibration.

Organoid search stops new questions at 24,000 reported tokens, six model/tool
requests or 300 seconds by default. These are inference-stage limits; a grouping
preflight and model startup are reported separately. The token threshold can be
exceeded by one in-flight engine call, including internal SDK/structured fallbacks.
A 16-million rescaled-pixel guard runs before Cellpose eval (30/diameter scaling),
configurable with SHAPE_ORGANOID_MAX_RESCALED_PIXELS. If a budget prevents final
selection, a completed, successful candidate that passes native artifact guards
is retained by native QC score, without GT or
another reasoner call. Failed candidates remain excluded; no admissible candidate
means no invented delivery. Retention is conceded and requires review.
Resource limits never certify segmentation quality. `quality_certified=false`,
`needs_review`, `qc_pass`,
`negative_confirmed` and `accepted_with_flags` retain their separate meanings.

For a hard wall deadline and process fault isolation, use the bounded cohort
runner in case studies/cardiac organoids/segmentation. It keeps one GPU
model process, fixed shards, per-pass journal and per-image predictions, and
explicit resume validation. The ordinary in-process solve path only checks soft
limits between operations; it cannot interrupt a native CUDA call already running.


### Experimental selective cohort review

The bounded runner's opt-in `--selective-review` keeps the legacy policy as the default.
Calibration representatives require a high-confidence whole-organoid visual assessment
(none/minor severity) before their parameters transfer. An unusable representative may be replaced at most three
times, reusing every sampled result. Other images use native QC plus
a 384-pixel dark-body coverage cue; only flagged images enter visual VRR. The cue
compares a prediction to compact dark components in the raw image, without GT. Coverage
below 0.55 or more than three instances per body triggers inspection, not automatic mask
editing. Unsupported polarity/contrast is recorded as unavailable, never quality certification.
Search, final selection and native calls retain the existing per-image budgets. This
experimental cue needs real validation on each microscopy domain; it is not an annotation.

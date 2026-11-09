---
name: agentic-bioimage-segmentation
description: >
  A platform-neutral procedure for reliable cell and organoid instance
  segmentation. It follows the current SHAPE segmentation contract: explicit
  task boundaries, image-loading checks, tool-specific first-pass segmentation,
  deterministic verification, optional high-resolution visual review, bounded
  VLRR correction, reconciliation, and provenance. It works with SHAPE, Codex,
  Claude Code, BioMNI, or another coding/tool agent without assuming a specific
  model provider, API, or controller implementation.
---

# Agentic Bioimage Segmentation

This skill defines an **execution protocol**, not a new segmentation model. Its
purpose is to make a segmentation run reproducible, diagnostically useful, and
economical. It cannot raise a model's intrinsic recall or boundary quality when
the model does not contain the missing information.

The default loop is:

```text
scope -> load/check input -> segment once -> verify -> localize -> recommend
one bounded change -> rerun -> reconcile -> accept, retain best, or escalate
```

The protocol is deliberately independent of the agent platform. An agent may
use the SHAPE runtime, a direct Python tool call, a remote job, or another
tool interface. The same evidence contract applies in all cases.

## 1. Current SHAPE contract

The current SHAPE pipeline has this relevant stage boundary:

```text
image_loading -> segmentation -> crop -> representation -> morphological_state
```

For a request whose deliverable is only a mask or cell/organoid count,
`segmentation` is the terminal stage. Do not run cropping or state analysis as
an accidental side effect of a segmentation request.

SHAPE exposes two managed segmentation tools:

| Target | Tool | Model contract | Main caution |
|---|---|---|---|
| Cells or nuclei | `Cell_Segmenter_Tool` | Cellpose-SAM (`cpsam`) | Do not use it as an organoid model. |
| Organoids or spheroids | `Organoid_Segmenter_Tool` | The configured custom organoid model | Do not substitute generic cell/cpsam defaults. |

The two tools share the SHAPE VLRR stage and evidence shape, but they do not
share a scientific quality threshold or a tuning rationale. Always read the
selected tool's tuning card and metadata before proposing a parameter change.

`Image_Preprocessor_Tool` is not part of the current segmentation tool contract.
Do not add it to `enabled_tools` or invoke it as an implicit segmentation step.
If an external wrapper performs preprocessing, record it as a separate
experimental condition; it invalidates a direct controller ON/OFF comparison
unless it is identical in both arms.

The SHAPE evidence-adaptive controller is optional and opt-in. When present, it
may compile a narrow terminal segmentation task, route verification based on
evidence, enable managed VLRR, and write a controller trace. The protocol below
remains valid when the controller, LLM, or vision API is unavailable.

## 2. Platform portability

Do not assume that an agent can call a particular SHAPE class, image API, or
vision model. First identify the available execution surface.

| Capability | SHAPE runtime | Generic coding agent | BioMNI-like tool agent |
|---|---|---|---|
| Segmentation | Call the managed SHAPE tool | Call the project tool or script | Call the registered segmentation tool |
| VLRR controller | Use the opt-in SHAPE controller | Implement the bounded loop explicitly | Use the platform's loop, with the same evidence fields |
| Visual review | Use the configured vision engine | Use an image-capable model if available | Use the available image tool/model if available |
| No visual model | Deterministic verifier | Deterministic verifier | Deterministic verifier |
| Provenance | SHAPE result and controller trace | Run manifest and artifact paths | Tool call record and run manifest |

The agent must state which capabilities were actually active. It must not claim
that a visual review occurred when only a text description or a thumbnail was
available. It must not claim that SHAPE VLRR ran when it only edited or wrote a
script.

### Minimum adapter contract

Any platform can participate if it can provide:

1. an input image path or byte-identifiable image;
2. a selected tool and model identity;
3. a labeled mask or an explicit failure;
4. an overlay or a reason it could not be produced;
5. numeric verification metrics where available;
6. the exact parameters and attempt number;
7. a machine-readable result record.

The canonical result record should contain these fields:

```json
{
  "image_id": "...",
  "task": "cell_segmentation or organoid_segmentation",
  "tool": "Cell_Segmenter_Tool",
  "model": "...",
  "attempt": 1,
  "params": {},
  "mask_path": "...",
  "overlay_path": "...",
  "status": "success or failed",
  "metrics": {},
  "decision": "accept, retune, escalate, or failed",
  "vision": {"available": false, "used": false},
  "expected_effect": {},
  "observed_effect": {},
  "code_commit": "...",
  "environment": "..."
}
```

Additional platform fields are allowed. The required fields must not be
silently renamed or omitted.

## 3. Scope and task boundary

Before running a tool, classify the request:

- **Cell segmentation:** individual cells, nuclei, or a cell count.
- **Organoid segmentation:** organoids, spheroids, or large multicellular objects.
- **Downstream analysis:** cropping, representation, morphology, phenotype, or
  comparison. These are not terminal segmentation tasks.

For CellSeg and OrganoidSeg, the output contract is a labeled mask, count, and
overlay plus metadata. A segmentation-only request stops after the segmentation
verifier accepts a result. If the request is ambiguous, preserve the boundary
and ask for the target rather than running unrelated downstream tools.

## 4. Input preparation

Input preparation is the highest-leverage part of the procedure. Apply only
operations that are explicitly recorded.

### 4.1 Inspect before transforming

Record:

- dtype and finite-value status;
- image shape and channel count;
- min, max, percentiles, and nonzero fraction;
- whether each candidate channel is near-constant;
- the original input hash or immutable source path.

Refuse or escalate non-finite input. Do not silently replace NaN or infinity.

### 4.2 Intensity and channel policy

- Pass a usable uint8 image through.
- For uint16 or float images, map the useful signal to uint8 using robust
  percentiles, normally the 1st and 99th percentiles, rather than dividing by
  the dtype maximum.
- For multichannel data, use the channel declared by the tool or experiment.
  If the declared channel is empty or near-constant, choose a signal-bearing
  channel by a deterministic rule such as highest variance and record that choice.
- Do not replace a single informative channel with an unrecorded RGB mean.

### 4.3 Preprocessing policy

Raw input is the default. Denoising, illumination correction, contrast changes,
and other transforms are allowed only when a recorded input QC failure justifies
them. Preserve a raw fallback and record:

```text
raw input -> transform name and parameters -> transformed input hash
```

Preprocessing is not a substitute for selecting the correct segmentation tool.
It must be held constant across benchmark arms.

## 5. First-pass segmentation

Run exactly one first pass with the selected tool's documented defaults unless
there is an explicit, recorded reason to override them.

### CellSeg

Use `Cell_Segmenter_Tool` for cells or nuclei. Current Cellpose-SAM guidance:

- prefer the tool default diameter on the first pass;
- use the mask-measured diameter only as a bounded scale correction;
- keep `bsize=256`, which is a Cellpose-SAM constraint;
- do not sweep parameters when the first pass has no actionable fault.

### OrganoidSeg

Use `Organoid_Segmenter_Tool` with its configured custom model. Organoids are
not cells at a larger `diameter`; the model, expected object scale, minimum area,
and touching-object behavior are different. The tool's card takes precedence.

For both tools, save the mask and overlay for every attempt. Do not overwrite a
previous attempt; use stable attempt-specific paths.

## 6. Deterministic verification

Verification is evidence collection, not a guess at AP50. Ground-truth metrics
such as AP50 are used only when labels and a fixed evaluation script are
available.

### 6.1 Hard failures

Treat any of the following as a failure or escalation, not as a candidate best
result:

- tool error or failed status;
- missing or unreadable mask;
- non-finite mask or invalid label encoding;
- zero objects, unless zero is scientifically expected and documented;
- black, empty, or dimension-mismatched overlay;
- duplicate output caused by an accidental repeated execution;
- model or checkpoint identity mismatch;
- output written outside the declared run directory without a recorded reason.

### 6.2 Common mask metrics

Compute when applicable:

- object count;
- flow error or the tool's analogous confidence/error metric;
- median object area and mask-derived diameter;
- tiny-object fraction;
- large-object or merged-object fraction;
- mean solidity;
- border-object fraction;
- foreground coverage and uncovered foreground fraction;
- cell-sized or organoid-sized missed-blob count;
- overlay validity and coverage plausibility.

The shared SHAPE verifier currently uses strict acceptance evidence for a clean
segmentation, including positive count, valid coverage, low flow error, bounded
miss evidence, bounded tiny/large/border fractions, and adequate solidity. These
are controller gates, not a universal replacement for labeled evaluation.

For CellSeg, the current starting gates are approximately:

```text
flow_error < 0.10
missed_foreground_fraction <= 0.08
count_miss_fraction <= 0.10
tiny_mask_fraction <= 0.15
large_mask_fraction <= 0.10
border_mask_fraction <= 0.40
mean_solidity >= 0.35
```

The values are conservative operational gates. They are not AP50 predictors and
must not be transferred unchanged to OrganoidSeg when object geometry or training
distribution differs. Organoid runs require tool-specific validation and, when
possible, labeled object-level metrics.

### 6.3 Under-detection is a separate risk

Good-looking mask geometry does not prove high recall. Use both:

- uncovered foreground area;
- estimated missed object count;
- overlay review when available.

These cues are noisy. Texture, halos, fluorescence, and crowded objects can make
them over-fire or under-fire. Treat them as triggers for evidence collection, not
as truth.

## 7. Optional visual review

Visual review is optional, but if used it must be real and auditable.

1. Send the full-resolution or high-detail image-plus-mask overlay. A small
   thumbnail is not equivalent evidence.
2. Ask the model to report only observable facts:

   ```json
   {
     "missed_objects": "none, few, or many",
     "false_masks": "none, few, or many",
     "usable": true,
     "problem": "..."
   }
   ```

3. Map the structured report to a deterministic action. Do not ask the vision
   model to invent parameter values.
4. Vision may tighten an acceptance decision, but it must not waive a hard
   deterministic failure.
5. Record provider, model, image detail, request id if available, and whether
   the response contained a structured verdict.

If no vision model or API is available, set `vision.available=false` and use the
deterministic path. The result is valid but has a stated limitation: recall was
not visually adjudicated.

## 8. VLRR: localize and recommend one change

VLRR means **Verify -> Localize -> Recommend -> Rerun**. It is a bounded control
loop, not an invitation to keep tuning.

### 8.1 Localize first

Use the following order:

1. invalid output or zero detection -> input/tool execution problem;
2. high error on poor input -> escalate to image loading/input preparation;
3. high error on clean input -> model or modality mismatch; do not tune blindly;
4. valid output with a specific mask fault -> choose one tool-supported lever;
5. valid output with no actionable fault -> accept.

### 8.2 CellSeg recommendation table

| Observed problem | One bounded recommendation | Expected effect |
|---|---|---|
| Zero cells | Lower `cellprob_threshold` | Count increases from zero |
| Faint missed cells | Lower `cellprob_threshold`; optionally enable `augment` once | Count/coverage increases |
| Large objects missed | Raise `niter` within the card bounds | Large-object recall increases |
| Small objects missed due to scale | Lower `diameter` modestly | Median object size and count improve |
| Flow-rejected real objects | Raise `flow_threshold` modestly | Flow-rejected count decreases |
| Merged or oversized masks | Lower `diameter`; if justified, raise `flow_threshold` | Large-mask fraction decreases |
| Specks or over-segmentation | Raise `cellprob_threshold` or `min_size` | Tiny-mask fraction decreases |

Do not change multiple unrelated knobs in one attempt. `bsize` is not a tuning
lever for Cellpose-SAM.

### 8.3 OrganoidSeg recommendation table

Use only parameters exposed by `Organoid_Segmenter_Tool` and its tuning card.
Typical directions are:

| Observed problem | Possible lever | Expected effect |
|---|---|---|
| One organoid fragmented | Increase object diameter/scale | Fewer fragments, larger coherent masks |
| Touching organoids merged | Raise the tool's separation/flow threshold if supported | Fewer merged objects |
| Faint organoid edges missed | Lower the tool's probability threshold | More boundary coverage |
| Small debris retained | Raise the tool's minimum size | Fewer debris masks |

These directions are hypotheses. Confirm that the named parameter is accepted by
the actual tool before calling it. Never pass CellSeg-only assumptions to a
custom organoid model.

## 9. Reconcile every rerun

Every recommendation must state an expected effect before the rerun. After the
rerun:

- confirm the parameter actually reached the tool;
- confirm the predicted metric moved in the predicted direction;
- compare the new result against the previous best;
- keep the best valid result, not automatically the last result;
- stop if the expected effect did not occur;
- stop on a repeated parameter state or oscillation;
- stop at the configured attempt limit, normally no more than 3 or 4 total
  segmentation attempts per image.

If a rerun changes the mask but does not improve the relevant evidence, retain the
best earlier result and record the failed hypothesis. A parameter change that did
not reach the tool is an execution defect, not evidence that the parameter is
ineffective.

## 10. Controller and API behavior

When the SHAPE evidence-adaptive controller is enabled, it may:

- compile a terminal segmentation plan for a singleton segmentation tool;
- skip an unnecessary provider round trip on a high-confidence result;
- route ambiguous or faulted results to verification;
- force a bounded confirmation/probe when strict acceptance lacks visual evidence;
- stop only after the current acceptance policy is satisfied;
- emit a per-image trace.

Skipping a vision call is an efficiency decision, not a quality result. A
controller run must report quality and efficiency separately:

```text
quality: AP50/AP75 if labels exist, otherwise verifier and visual evidence
efficiency: seconds, tool calls, provider calls, input/output tokens
control: attempts, stop reason, skipped/escalated vision, retry reason
```

For a generic coding agent, reproduce the same semantics explicitly if the SHAPE
controller is unavailable. Do not copy the internal controller name or assume
that a planner-only call exercised the segmentation loop.

## 11. Benchmark and evidence protocol

An agentic segmentation change is not validated by a single attractive overlay.
For a labeled benchmark:

1. freeze the repository commit, model/checkpoint, environment, input list, and
   preprocessing policy;
2. run controller OFF and ON on the same images, seeds, and tool version;
3. keep CellSeg and OrganoidSeg as separate cohorts;
4. record per-image masks, AP50/AP75, count, attempts, stop reason, and trace;
5. report macro per-image means as well as pooled metrics when image counts vary;
6. report quality and efficiency independently;
7. inspect at least the largest gains, losses, zero-cell cases, and early stops.

Do not attribute a change to the controller if preprocessing, tool code, model
checkpoint, enabled tools, or evaluation code changed between arms. A controller
can reduce cost while losing AP50; that is an efficiency tradeoff, not an
additional segmentation gain.

## 12. Known ceilings and escalation

Stop tuning and escalate when:

- the selected model is not fit to the modality or object type;
- clean input still produces persistently high flow/error evidence;
- crowded-object recall remains poor after one evidence-backed retry;
- boundaries remain loose despite valid object detection;
- the custom organoid model does not match the organoid distribution;
- the proposed change repeatedly fails to produce its expected effect.

These cases require a different model, checkpoint, input acquisition strategy,
annotation policy, or evaluation decision. More planner calls cannot recover
objects that the segmentation model does not represent.

## 13. Minimal portable checklist

```text
[ ] Classify CellSeg, OrganoidSeg, or downstream analysis.
[ ] Select the correct tool and read its tuning card.
[ ] Record input dtype, channels, finite status, and source identity.
[ ] Use robust intensity handling and a deterministic signal channel.
[ ] Keep preprocessing raw by default and identical across benchmark arms.
[ ] Run one first pass and save mask, overlay, params, and metadata.
[ ] Verify output validity, scale, shape, coverage, and under-detection cues.
[ ] Use high-detail visual review only when the API/model actually exists.
[ ] Recommend one tool-supported change with an expected effect.
[ ] Rerun within the attempt bound and reconcile the observed effect.
[ ] Keep the best valid result and stop on failed hypotheses or ceilings.
[ ] Emit a machine-readable record with code, model, environment, and traces.
```

The protocol is complete only when another agent can determine what ran, what
changed, why it stopped, and which limitations remain.

---
name: single_cell_cropper
description: >-
  Cut one clean single-cell crop per detected object from an image + its instance segmentation
  mask. Deterministic geometry frames each cell and trims neighbour contamination; a cheap
  three-arm data gate flags suspect crops; a keep-biased vision-model (VLM) reject channel drops
  the mask errors — debris false-positives, doublets, wrong-object labels — that geometry and
  mask-derived metrics are structurally blind to. Use after a segmenter to prepare per-cell inputs
  (SSL embeddings, morphology, classification). System-neutral recipe: runnable by any agent
  (Claude Code, Codex, Gemini, …); the only hard dependency is a multimodal VLM for the reject call.
---

# Single-cell cropper

Given a microscopy image and an **instance segmentation mask** (one integer label per object),
produce one **single-cell crop per object** that is (a) tightly framed on its own cell with minimal
neighbouring-cell signal, and (b) filtered to exclude crops whose underlying mask is a segmentation
ERROR. The keep/reject call on a flagged crop is made by a **VLM**, because the load-bearing errors
are visually — not geometrically — defined.

- **Use when:** after a segmenter, to prepare per-cell inputs for downstream single-cell analysis.
- **Requires:** a mask first. This is not a segmenter — it never re-segments; it crops what the mask
  gives and screens the mask's own mistakes.
- **Inputs:** image `I` (H×W or H×W×C, any bit depth); instance mask `M` (H×W integer labels).
- **Outputs:** kept single-cell crops (+ each crop's own mask) and a **per-crop decision trace**.

---

## Using this skill across agent systems
The body below is a self-contained, system-neutral recipe — no step depends on a specific runtime.
- **Claude Code:** place this folder in a skills directory (e.g. `.claude/skills/single_cell_cropper/`);
  the `name`/`description` frontmatter make it discoverable and it loads on demand.
- **Codex:** reference this file from `AGENTS.md` (or paste it into the agent's instruction context);
  follow the recipe directly. The YAML frontmatter is inert metadata to a plain reader.
- **Gemini (or any other agent):** reference from `GEMINI.md` / the system prompt, or include the file
  in context. Follow the same steps.
- **Capabilities the host must provide:** basic image ops (bounding box, convex-hull solidity,
  percentiles, per-region intensity stats); a montage renderer; and a **multimodal VLM with
  structured (or reliably parseable) output** for the reject call (`PROMPT.md`). Nothing else.

---

## Algorithm (precise — implement exactly; constants are documented defaults, see "Thresholds")
```
INPUT: image I, instance mask M, optional per-cell overrides
CONSTANTS: MARGIN_FRAC=0.20  MIN_AREA=50  DIM_FRAC=0.5  REF_MIN_CROPS=5  P90_FALLBACK=160.0
           SOLIDITY_MIN=0.90  DEBRIS_AREA_FRAC=0.25  FLAT_TEXTURE_MAX=10.0  VLM_CAP=27

# Stage 1 — deterministic geometry (pure data, no model)
crops = []
for each label L in M:
    if area(M==L) < MIN_AREA: continue
    bbox = bounding_box(M==L); box = expand(bbox, MARGIN_FRAC)          # margin = 20% of bbox
    (optional) box = auto_tighten(box, toward=L)  # shrink to exclude neighbours, stop before clipping L
    crops.append({label:L, crop:cut(I,box), mask:(M==L within box)})

# Stage 2 — three-arm data gate: WHICH crops the VLM should see (a loose SCREEN, never a reject)
p90s   = [ p90( I[c.crop][c.mask] ) for c in crops ]                    # 90th pct of in-mask intensity
ref_p90 = p75(p90s) if len(p90s) >= REF_MIN_CROPS else P90_FALLBACK     # per-image adaptive
median_area = median([ area(c.mask) for c in crops ])
candidates = []
for c in crops:
    dim     = readable(c) and p90(I[c.crop][c.mask]) < DIM_FRAC * ref_p90
    doublet = readable(c.mask) and solidity(c.mask) < SOLIDITY_MIN      # mask_area / convex_hull_area
    debris  = (readable(c.mask) and area(c.mask) < DEBRIS_AREA_FRAC * median_area) \
              or (readable(c) and std( I[c.crop][c.mask] ) < FLAT_TEXTURE_MAX)
    c.arms_fired = [name for name,hit in [("dim",dim),("doublet",doublet),("debris",debris)] if hit]
    if c.arms_fired: candidates.append(c)          # ANY arm ⇒ candidate; each arm FAILS CLOSED
candidates = sort_most_suspicious_first(candidates)[:VLM_CAP]           # ascending in-mask p90; cap/image

# Stage 3 — the mask check: the VLM reject channel (the decision)
for batch in montage_batches(candidates):          # e.g. a 3×3 grid per montage
    montage = render(batch)   # per cell: contour-ONLY green 1px outline of c.mask (NO fill),
                              # mask-derived contrast stretch, capped upscale, index label 0..n-1
    for c in batch: c.sent = True                  # mark sent BEFORE the call (see keep-bias)
    resp = VLM(prompt(batch), montage) -> MaskCheckVerdict      # PROMPT.md; subset responses are legal
    for v in resp.verdicts:
        if v.verdict in {"target_region_empty","wrong_object","merged"}:
            batch[v.index].reject = True; batch[v.index].reason = v.verdict

# Stage 4 — assemble (KEEP-BIASED)
kept = [c for c in crops if not c.reject]           # silence / abstain / omission / error ⇒ keep
OUTPUT: kept, and per-crop trace {geometry, arms_fired, mask_check:{sent,verdict}, final:{kept,reason}}
```

### Stage notes
- **Stage 1** alone is the naïve **MaskCrop** baseline (keep every object, fixed 20% margin). It cannot
  see mask errors — a box around a debris fragment or a doublet is geometrically fine.
- **Stage 2** is cheap and pure-data. On expert labels it catches only ~half of the truly reject-worthy
  crops: the other half are bright, textured, convex, normal-sized non-cells — geometrically identical
  to real cells. So the gate's job is to ENRICH what the VLM sees, not to classify. Send more, not fewer.
- **Stage 3**: the contour is drawn **outline-only** because a tint/fill hides the very texture the judge
  needs to tell `cell_present` from `wrong_object`/`merged`. One montage per batch; the model may answer
  a **subset** of positions — an unnamed position is simply "not rejected".

---

## Thresholds — candidate per-input knobs, not curve-fit constants
The cut-offs above are documented starting points chosen from data distributions (e.g. `FLAT_TEXTURE_MAX`
was set just below the 5th percentile of real in-mask intensity std; `MARGIN_FRAC=0.20` is a naïve
general-purpose default), **never tuned to maximise a result**. An agentic caller MAY re-propose them
per input (that is the intended use). Keep them named and commented; do not silently fit them to a metric.

---

## Design principles (load-bearing — do not drop)
- **Data screens, the VLM decides.** The mask/data is the factual anchor for *which* crops deserve a
  look; the VLM (unreliable + expensive) only CONFIRMS the reject on data-flagged crops. The VLM never
  drives geometry, and no pure-data rule makes the final reject.
- **Keep-biased on every gap.** Silence / abstain / an omitted position / a malformed response / any
  engine error ⇒ **keep**. A wrongly-kept crop is recoverable downstream noise; a wrongly-dropped good
  cell is lost data. A crop can be `sent=True, verdict=None` (sent but omitted) — still kept, still
  logged as sent (track sent-membership from the montage batch, not from whether a verdict returned).
- **Mask-derived quality metrics lie in your favour.** `neighbor_contamination` / `target_retention`
  are computed FROM the mask, so a mask ERROR can score as CLEAN as a good crop — the metric cannot see
  the error the VLM caught. Never let a contamination/retention number stand in for the VLM's judgement;
  validate kept-crop quality on the raw image or expert labels.

---

## Evaluation (how to tell it works — never self-scored)
Benchmark against the naïve MaskCrop baseline (Stage 1 only), on imperfect predicted masks, with an
independent adjudicator:
- **Crop quality (mask-derived):** `neighbor_contamination` (↓, fraction of in-frame pixels from OTHER
  cells) and `target_retention` (↑, fraction of the target cell kept in frame) for **(A)** MaskCrop-kept
  vs workflow-kept, and **(B)** workflow **kept vs rejected** — expect rejected crops to look *no worse*
  than kept on these (the mask-anchored point above; that is the proof the VLM is doing something the
  metric can't).
- **Agentic efficiency (6 metrics):** positive-tool-call rate, wasted-tool-call rate, quality-gain per
  tool-call / per 1k tokens / per minute, and tool-calls + token-cost per image.
- **Mask-error detection (needs GT masks — benchmark-only):** adjudicate each predicted object vs GT
  (`good / merged / split / false_positive`) and score the workflow's rejects as precision/recall
  against it. **GT adjudication is a benchmark signal only — it must never feed the live controller.**

---

## Reference implementation (SHAPE)
Distilled from SHAPE's `Single_Cell_Cropper_Tool`:
- deterministic engine — `shape/tools/single_cell_cropper/tool.py` (+ `CARD.md`, `quant.py`);
- three-arm gate — `shape/agent/verification/verifiers/crop.py` (`_gate_arms`, `_gate_image_stats`);
- the mask check + montage + prompt — `shape/agent/verification/crop_triage.py` (`mask_check_recommend_crop`);
- MaskCrop-vs-workflow benchmark + the metrics above — `benchmarks/cropper_NeurIPS22-CellSeg/`.
The VLM prompt, render spec, and response schema are in `PROMPT.md` beside this file.

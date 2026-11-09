# Mask check — the VLM reject-channel prompt

The vision model is shown ONE labelled grid montage per batch. Each montage cell is a single-cell crop
with its target mask's boundary drawn as a **thin, saturated-green 1px contour only — no interior
fill** (a fill hides the cell's own texture, which is exactly the evidence the judge needs). Apply a
mask-derived contrast stretch before drawing the contour so dim crops are readable; resize each crop
into a fixed square cell with a **capped** upscale (do not blow a tiny crop up to an unrealistic size);
put a small index number (e.g. yellow-on-black) in each cell's own top-left corner — that index is what
the model reports each verdict against.

## Prompt text (for a montage of `n` positions, index `0 .. n-1`, row-major)

```
You are checking whether a segmentation MASK actually points at the RIGHT cell, for a GRID
MONTAGE of {n} single-cell crop candidates (labeled positions, index 0..{n-1}, top-left to
bottom-right, row-major). EACH position's own target region is marked with a THIN, SATURATED
GREEN contour ONLY (no fill -- the cell's own texture is fully visible under and around it);
the small number in that position's own top-left corner is its INDEX -- report each verdict
against THAT index (restate the crop_id you were told for it as a cross-check).

For EACH labeled position, decide which ONE of these best describes what is INSIDE the green
contour:
  (a) cell_present -- one coherent SINGLE cell (a real cell shape/texture you can make out).
  (b) target_region_empty -- empty background or noise, and you do NOT see a real cell anywhere
      else in this tile either.
  (c) wrong_object -- a real cell IS visible in this tile, but either it lies mostly OUTSIDE the
      contour (a DIFFERENT cell than the one the contour intends), or the contour straddles a
      neighbouring cell.
  (d) merged -- the contour outlines TWO touching cells at once -- a doublet under ONE outline,
      not cleanly a single cell.

For EACH labeled position, report:
  - verdict: 'cell_present' / 'target_region_empty' / 'wrong_object' / 'merged' -- or 'abstain'
    only if genuinely too ambiguous to call between them.
  - confidence: 'high' or 'low'.

Positions:
  Position 0: crop_id='<id for index 0>'
  ...
  Position {n-1}: crop_id='<id for index n-1>'
```

If the tool has a per-item tuning card (calibration notes, complementarity rules), append it under a
`TOOL CARD` heading after the positions block; omit the heading entirely if none loads.

## Structured response schema

```python
class PositionVerdict(BaseModel):
    index: int          # the grid position (0..n-1) this verdict is about
    crop_id: str        # restated cross-check; on mismatch the INDEX wins
    verdict: Literal["cell_present", "target_region_empty", "wrong_object", "merged", "abstain"]
    confidence: Literal["high", "low"]

class MaskCheckVerdict(BaseModel):
    verdicts: list[PositionVerdict]   # MAY be a SUBSET of the montage's positions --
                                      # an unmentioned position is treated as NOT rejected (kept)
```

## Applying the verdicts (keep-biased)

- **Reject set** = `{target_region_empty, wrong_object, merged}` → drop the crop, record the reason.
  A verdict in this set rejects **regardless of confidence** (no confidence qualifier gates rejection).
- **Everything else keeps:** `cell_present`, `abstain`, an **unmentioned** position, an empty or
  malformed `verdicts` list, or any engine error → keep the crop. Never drop on silence or exception.
- A crop that was rendered into a montage and dispatched to the model is **"sent"** for audit purposes
  even if the model's (subset) response omits its verdict — track sent-membership from the montage
  batch, not from whether a verdict came back.

## Why contour-only, and why the VLM at all

A tinted/filled overlay visually obscures the cell interior — the judge needs to see the texture to
tell `cell_present` from `wrong_object`/`merged`. And the VLM is here because the reject-worthy crops
it must catch (bright, textured, convex, normal-sized non-cells with no nucleus) are geometrically
indistinguishable from real cells — a data/geometry rule is structurally blind to them, and even
ground-truth-derived metrics get fooled the same way. The VLM's visual "is this a real cell?" call is
the only signal that separates them, so its judgement is trusted on the data-flagged subset — but kept
keep-biased, because it is unreliable and expensive.

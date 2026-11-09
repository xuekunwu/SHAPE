from shape.tools.base import BaseTool
import numpy as np
import cv2
import os
import time
from PIL import Image
from uuid import uuid4
import matplotlib.pyplot as plt
from skimage.measure import label, regionprops
from pathlib import Path
import json
import zipfile
from shape.utils.agent_io import VisualizationConfig
from shape.schemas.task_state import CellCrop
import tifffile
from shape.utils.image_data import ImageData
from shape.utils.image_processor import ImageProcessor
from shape.utils.segmentation_qc import to_display_uint8


def normalize_source_image(original_img: np.ndarray) -> np.ndarray:
    """Task 038 (v9 CRITICAL fix, human-locked): the SOURCE image -> display uint8,
    ONCE, up front -- the fix for a real the cluster finding (4/10 v9 images produced
    UNUSABLE crops: cell_00232/00673/00707 ALL PURE WHITE, cell_00860 68 PURE BLACK
    crops). Root cause: every crop-SAVE path downstream did a NAIVE
    `np.clip(cell_crop, 0, 255).astype(np.uint8)` -- a clip, not a normalization.
    NeurIPS22-CellSeg is MIXED-MODALITY: a 16-bit fluorescence image (values into the
    thousands, up to 65535) clips to ALL WHITE (every value > 255 -> 255); a float
    [0, 1] image casts to ALL BLACK (`astype(uint8)` truncates every sub-1.0 value to
    0). This was PRE-EXISTING across every prior round (v5-v9) -- the mask-based QC
    (`_compute_crop_qc`) reads the LABEL mask, never `original_img`'s own pixel
    values, so target_dominance/border_touch/etc. were NEVER corrupted by this bug --
    only the saved crop IMAGES themselves (the product actually fed to SSL).

    Reuses `to_display_uint8` (`shape.utils.segmentation_qc`, the SAME canonical
    grayscale->uint8 conversion the segmentor's own overlay path already uses -- its
    own docstring already names a fixed `/65535` scale as WRONG for microscopy,
    exactly this bug) rather than reimplementing a stretch. A percentile (1-99)
    stretch, NOT a naive clip -- and a genuine no-op, byte-identical, for an ALREADY
    uint8 image (`to_display_uint8` passes uint8 through unchanged) -- so the 6
    already-good v9 images are untouched by this fix.

    Applied ONCE to the WHOLE image here (not per-crop, in the save loop) so every
    crop cut from the SAME source image shares the SAME intensity mapping -- a
    per-crop stretch would independently re-stretch each crop's own local
    min/max, destroying the relative brightness BETWEEN crops from the same image
    (bad for SSL, which relies on that consistency).

    Multi-channel (H, W, C) images are normalized PER CHANNEL, independently -- a
    single global percentile computed across channels with very different intensity
    ranges (e.g. a bright bright-field channel next to a dim fluorescence channel)
    would over/under-stretch one relative to the other. 2D (H, W) images are
    normalized directly. Returns an array of the SAME shape/ndim as `original_img`.
    """
    if original_img.ndim == 2:
        return to_display_uint8(original_img)
    return np.stack(
        [to_display_uint8(original_img[:, :, c]) for c in range(original_img.shape[2])],
        axis=-1)


# The retention floor: every crop the tool writes keeps at least this share of its target's
# mask pixels. THE single source of the value (human decision, 2026-10-01: the floor stays
# 0.90, one number everywhere) -- `verifiers/crop.py` imports it: `CropVerifier.
# _TARGET_RETENTION_FLOOR`, the agentic loop's `RETENTION_FLOOR` (its guard, the
# `min_containment` range, the uncroppable test) and `is_crop_valid`'s truncation gate
# (`_CONTAINMENT_MIN`); the agent layer may import the tool, the tool never imports
# `shape.agent.*`. Here it bounds every cut: the `mask_bbox` backstop that scales a
# directional `bbox_inset_px` back (or drops a cell no square holds at the floor),
# `auto_tighten`'s cuts, `contamination_inset_px`'s depth (`_compute_crop_qc`) and, as
# its lower bound, a cell's `min_containment`. History: task 038 v12 (human-locked
# directive -- DELIBERATELY REVERSES the earlier "geometry is 100% pure-data, no VLM"
# lock, CARD.md SS1/SS5) made it the hard floor of the directional-tighten proposals,
# then duplicated in `verifiers/crop.py`; task 038 v13 (human decision) raised it 0.85
# -> 0.90 -- a MORE conservative floor for the "safe candidate-generator" scope (the
# target may be clipped by at most 10%, not 15%), one of TWO tuning knobs of the v13
# control layer with `CropVerifier._VOI_RETENTION_MARGIN` (how close to THIS floor a cut
# must land before the blinded A/B channel treats it as uncertain enough to ask about).
_TARGET_RETENTION_FLOOR = 0.90

# Task 038 architectural FIX round (coordinator-directed -- moves the per-cell tighten
# ITERATION inside the tool, `_auto_tighten_mask_bbox` below, so the VLRR loop applies
# ONE retune per cell instead of one retune per ROUND per cell): the SAME clean
# threshold `verifiers/crop.py::CropVerifier._MASK_BBOX_CONTAMINATION_MIN` uses to
# decide a cell is contaminated enough to trigger `mask_bbox` in the first place, AND
# (v12/v13, unchanged) to decide a mask_bbox'd cell is clean enough to stop tightening
# -- duplicated here (the tool never imports `shape.agent.*`), CARD.md is the
# single source of TRUTH; keep both copies in sync.
_MASK_BBOX_CONTAMINATION_MIN = 0.2
# The SAME plateau epsilon `verifiers/crop.py::CropVerifier._TIGHTEN_PLATEAU_EPS`
# introduced (task 038 FIX round #2, per-cell plateau/regression freeze) -- duplicated
# here so `_auto_tighten_mask_bbox`'s own internal loop carries that round's semantics
# (stop when a cut helps by this much or less; never commit a cut that makes the cell
# WORSE) forward into the internal iteration, instead of losing them when the
# per-round verifier-level check that used to enforce them is removed (this same
# architectural round, below). CARD.md is the single source of TRUTH; keep both
# copies in sync.
_TIGHTEN_PLATEAU_EPS = 0.02
# A defensive LOOP-SAFETY bound on `_auto_tighten_mask_bbox`'s own internal iteration,
# NOT a tuning knob (CARD.md does not govern this value): the plateau/regression rule
# and the retention-floor backstop already terminate the loop in practice long before
# this would ever bind; it exists only so a pathological mask geometry can never spin
# forever inside one `execute()` call.
_AUTO_TIGHTEN_SAFETY_CAP = 20


def tool_cache_dir_for(query_cache_dir: str, output_subdir: str | None = None) -> str:
    """`<query_cache_dir>/tool_cache`, or `<query_cache_dir>/tool_cache/<output_subdir>` when a
    caller keeps one call's files apart from the other calls on the same query (the crop and
    mask file names are deterministic per cell, so two calls in one directory overwrite each
    other). `output_subdir` must be a relative path inside `tool_cache`."""
    base = os.path.join(query_cache_dir, "tool_cache")
    if not output_subdir:
        return base
    sub = os.path.normpath(str(output_subdir))
    if os.path.isabs(sub) or sub == ".." or sub.startswith(".." + os.sep):
        raise ValueError(f"output_subdir must be a relative path inside tool_cache, got {output_subdir!r}")
    return os.path.join(base, sub)


_INSET_SIDES = ("top", "bottom", "left", "right")


def _clamp_directional_inset(minr: int, maxr: int, minc: int, maxc: int, inset: dict,
                             min_side: int = 8) -> tuple[int, int, int, int, dict]:
    """Applies a DIRECTIONAL `{top, bottom, left, right}` inset (task 038 v12, human-
    locked) to the SQUARE `mask_bbox` box and keeps it square (human rule, 2026-09-30:
    every crop the tool writes is square -- the analyzer resizes each crop to 224 x 224
    without keeping the aspect ratio): a cut of k px on one side is paired with a k px
    cut on the perpendicular axis, split evenly between that axis's two sides so the
    target stays centred on it. Returns `(minr, maxr, minc, maxc, applied)`, `applied` =
    the sides actually cut and by how much, the paired cuts included (sides without a cut
    omitted). Negative/missing inset values are treated as 0. Pure arithmetic, no I/O --
    mirrors `compute_crop_box`'s own role as a reusable, testable geometry primitive.

    1. Clamp: the box never inverts and never shrinks below `min_side`. An axis whose
       requested cut exceeds the box's headroom (`side - min_side`) is scaled down
       PROPORTIONALLY across its two sides to exactly the headroom (integer split, the
       remainder to the bottom/right side) -- symmetric treatment when a later round's
       own `contamination_side` names BOTH opposing sides of one axis. The requested
       (primary) cuts are otherwise applied as they are.
    2. Pair: the axis with the smaller cut is cut by the difference too, so both axes lose
       the same number of pixels; the difference is split evenly between its two sides
       (human decision, 2026-09-30: the paired cut keeps the target centred), floor to
       top/left, ceil to bottom/right.

    Exact pairing is always possible: a cut only shrinks a box that already lies inside
    the image, so the image edge never blocks a cut, and step 1 clamps both axes to the
    SAME headroom, so the paired axis's total (the larger of the two) fits it too. Two
    cases cannot be met exactly and take the closest valid alternative: a cut beyond the
    headroom (`min_side`) -- both axes are cut by exactly the headroom (the cut shrinks,
    the pairing stays exact) -- and an odd difference, which cannot split exactly in half
    -- the extra pixel goes to the bottom/right side (the target off-centre by half a
    pixel)."""
    cut = {k: max(0, int(inset.get(k, 0) or 0)) for k in _INSET_SIDES}
    headroom = max(0, min(maxr - minr, maxc - minc) - min_side)
    for a, b in (("top", "bottom"), ("left", "right")):
        total = cut[a] + cut[b]
        if total > headroom:
            cut[a] = cut[a] * headroom // total
            cut[b] = headroom - cut[a]

    rows, cols = cut["top"] + cut["bottom"], cut["left"] + cut["right"]
    d = abs(cols - rows)
    if d:
        a, b = ("top", "bottom") if rows < cols else ("left", "right")
        cut[a] += d // 2
        cut[b] += d - d // 2

    applied = {k: v for k, v in cut.items() if v}
    return minr + cut["top"], maxr - cut["bottom"], minc + cut["left"], maxc - cut["right"], applied


def _save_baseline_crop(original_img, box, path, is_multi_channel, output_format):
    """Extracts + saves a crop at an arbitrary box (task 038 v13, human-directed) -- the
    mask_bbox BASELINE render for `crop_triage.ab_recommend_crop`'s blinded A/B channel,
    so the comparison has BOTH real images (baseline vs the tightened candidate) with NO
    history threading needed. `box = (minr, minc, maxr, maxc)`, the SAME ordering
    `crop_metadata["crop_bbox"]` already uses. Mirrors the PRIMARY crop's own
    channel-aware save dispatch (`_generate_cell_crops`, below -- multi-channel TIFF vs.
    1/3/4-channel PIL) in COMPACT form: this is a SECONDARY artifact for the A/B channel
    only, not the cell's own primary crop, so it skips that path's extensive
    validation/re-open verification and instead just catches+logs any failure, returning
    None -- a missing baseline simply means `crop_metadata["baseline_crop_path"]` stays
    None and the A/B channel skips that cell (never breaks the cell's own primary crop)."""
    minr, minc, maxr, maxc = box
    try:
        arr = (original_img[minr:maxr, minc:maxc, :] if original_img.ndim == 3
               else original_img[minr:maxr, minc:maxc][:, :, np.newaxis])
        if arr.size == 0:
            return None
        if is_multi_channel and arr.shape[2] > 1:
            tifffile.imwrite(path, np.moveaxis(arr, -1, 0), imagej=True)
        else:
            n_chan = arr.shape[2] if arr.ndim == 3 else 1
            mode = {1: "L", 3: "RGB", 4: "RGBA"}.get(n_chan, "L")
            flat = arr[:, :, 0] if mode == "L" and arr.ndim == 3 else arr
            if flat.dtype != np.uint8:
                flat = np.clip(flat, 0, 255).astype(np.uint8)
            fmt = "TIFF" if output_format.lower() == "tif" else (
                "JPEG" if output_format.lower() == "jpg" else "PNG")
            Image.fromarray(flat, mode=mode).save(path, fmt)
        return path if os.path.exists(path) else None
    except Exception as exc:  # pragma: no cover - defensive: a baseline render failure
                              # must never break the cell's own primary crop
        print(f"Warning: failed to save baseline crop at {path}: {exc}")
        return None


# What counts as a cell is defined once, at segmentation (Cellpose's `min_size`, a segmentation lever): the
# cropper gives every object the delivering segmentation kept a crop, or reports why it cannot -- the caller
# passes that segmentation's `min_size` as `min_area` (human decision, 2026-10-01). Without a known producer a
# cell / nuclei / unnamed mask uses this value, `Cell_Segmenter_Tool.execute`'s `min_size` default (the
# cropper used 50 before, silently dropping the 15-49 px objects the segmenter kept). Organoid masks are
# outside this rule: this tool's organoid default stays 1000 and the pipeline's cropper command keeps its
# min_area 50 (`shape/agent/executor.py`), while the organoid segmenter's min_size default is 15 -- an
# inconsistency left for later. Not imported from the segmenter (it imports cellpose and torch): a test reads
# its default with `ast`.
SEGMENTER_DEFAULT_MIN_SIZE = 15


def default_min_area_margin(mask_path) -> tuple[int, int]:
    """`execute`'s `(min_area, margin)` when the caller passes none, by mask file name: `cell_mask`
    SEGMENTER_DEFAULT_MIN_SIZE / 1 px, `organoid_mask` 1000 / 10 px, `nuclei_mask` or any other name
    SEGMENTER_DEFAULT_MIN_SIZE / 25 px (the agentic crop loop resolves the same rule on its input)."""
    name = os.path.basename(str(mask_path or "")).lower()
    if "cell_mask" in name:
        return SEGMENTER_DEFAULT_MIN_SIZE, 1
    if "organoid_mask" in name:
        return 1000, 10
    return SEGMENTER_DEFAULT_MIN_SIZE, 25


def pixel_args(min_area, margin, crop_size, tight_buffer_px):
    """`execute`'s reading of its pixel arguments, `(min_area, margin, crop_size, tight_buffer_px)`: whole
    pixels, none below 0, `crop_size` at least 16 (None = no cap). Raises TypeError / ValueError for a value
    that is not a number."""
    return (max(0, int(round(float(min_area)))), max(0, int(round(float(margin)))),
            None if crop_size is None else max(16, int(round(float(crop_size)))),
            max(0, int(round(float(tight_buffer_px)))))


def effective_margin(margin, cell_area, margin_frac=0.20, margin_mode="fraction", tight_buffer_px=2):
    """Per-cell crop margin. Two modes (task 042; DEFAULT flipped task 038 M-C/quality-geometry,
    then flipped BACK task 038 follow-up part B redesign -- see below):

    - "fraction" (DEFAULT again, task 038 follow-up part B redesign, human directive:
      "the tool's general-purpose default should be the naive baseline for ANY caller"):
      scaled by a fraction of THIS cell's mask AREA (its equivalent diameter). A fixed
      pixel margin (e.g. 1px) leaves a bare bounding box with no surrounding context,
      cutting off the phase-contrast halo that carries morphology. Scaling by a fraction
      of the cell's equivalent diameter (2*sqrt(area/pi)) gives context proportional to
      the cell's true SIZE -- not its bbox extent, which an elongated cell over-states.
      Customized per cell id; the fixed `margin` acts as a floor.
    - "tight" (opt-in -- the QUALITY-MAXIMIZING geometry, briefly the shipped default
      through the task 038 quality-geometry round; now the AGENT's own per-cell
      optimization TARGET, applied via `cell_overrides` (`margin_mode="tight",
      recenter=True` per cell) rather than the tool's own default -- see CARD.md SS1/
      SS5): the TIGHTEST per-cell margin. The target is already fully in-frame at
      margin=0 (the crop's half_side is derived from the cell's own bbox/centroid
      geometry), so the margin is a small constant anti-clip buffer (`tight_buffer_px`)
      -- just enough that the target's outer pixels are not literally flush against the
      crop border. `margin`/`margin_frac` are UNUSED in this mode (task 038 follow-up,
      item 3 part 3 made `margin_frac` ADDITIVE here as a VLRR-managed widening lever; a
      LATER round, human directive, REVERTED that -- widening margin_frac to fix low
      `target_dominance` is BACKWARDS, widening ADDS background, lowering dominance; low
      dominance is fixed DETERMINISTICALLY by the tight-bbox-square sizing itself,
      `_generate_cell_crops`, not by a margin retune). A caller may still opt IN to this
      mode explicitly (e.g. an A/B of the two crop geometries, or a per-cell
      `cell_overrides` entry) -- it is simply not the shipped default any more.
    """
    if margin_mode == "tight":
        return max(0, int(tight_buffer_px))
    equiv_diameter = 2.0 * (max(0.0, float(cell_area)) / 3.141592653589793) ** 0.5
    return max(int(margin), int(round(float(margin_frac) * equiv_diameter)))


def compute_crop_box(center_row, center_col, half_side, img_height, img_width):
    """The (possibly border-clipped) square crop box centered at (center_row,
    center_col) with the given half_side. Extracted from `_generate_cell_crops`'s
    inline geometry (task 042) so a measurement script can compute the SAME crop box
    production would, at a DIFFERENT margin, with zero risk of drifting from the tool's
    own geometry -- originally for task 042's margin/neighbour benchmark (harness
    removed in the 038 reorg; see archives/042_cropper-tight-margin-reject/); kept as a
    reusable primitive regardless of that specific benchmark's lifetime. Pure
    arithmetic, no I/O.

    No longer used to PLACE crops: its per-axis border clip returns a non-square box for
    a cell near the image border, and every crop the tool writes is square (human rule,
    2026-09-30) -- `compute_square_crop_containing_bbox` places every crop.
    """
    new_minr = int(max(center_row - half_side, 0))
    new_maxr = int(min(center_row + half_side, img_height))
    new_minc = int(max(center_col - half_side, 0))
    new_maxc = int(min(center_col + half_side, img_width))
    return new_minr, new_minc, new_maxr, new_maxc


def compute_square_crop_containing_bbox(minr, minc, maxr, maxc, center_row, center_col,
                                         target_half_side, img_height, img_width):
    """The placement of EVERY crop this tool writes (`_generate_cell_crops`, every
    `margin_mode`, with or without `recenter`; human rule, 2026-09-30: every crop is
    square and cut from real image pixels -- no padding, no masking -- because the
    analyzer resizes each crop to 224 x 224 without keeping the aspect ratio). Always
    returns a square box `(minr, minc, minr + s, minc + s)`, never None.

    The side `s` is `2 * target_half_side`, at most the image's shorter side: when no
    square of the needed side fits inside the image at all, the largest square that fits
    is used instead.
      - `s` at least the bbox's longer side: the square fully contains bbox
        [minr,maxr) x [minc,maxc), positioned as close to (center_row, center_col) as
        containment and the image allow -- near the image border the SAME-SIZE square
        shifts inside the image, so the target sits off-center instead of the crop being
        clipped (`compute_crop_box` clamps each axis INDEPENDENTLY, which clips a near-edge
        crop to a rectangle).
      - `s` below the bbox's longer side (no square of this side contains the bbox: a
        `crop_size` cap or a `min_containment` tighten below the bbox, or a bbox longer
        than the image's shorter side; an odd longer side is held in every mode since
        2026-10-01 -- `fraction` / `tight` size the half side with ceil): the square is
        centered on (center_row, center_col) as closely as the image allows -- graceful
        partial truncation around the target's center. The caller measures the ACTUAL
        containment from the real per-cell mask and its own containment /
        `min_containment` / retention-floor rules keep or drop the cell.

    Task 038 follow-up (part B, result-driven per-cell tighten): `target_half_side` may be
    DELIBERATELY smaller than the bbox itself needs -- an agent-approved tighten override
    that trades some containment for higher target_dominance (`cell_overrides`); the
    second branch above is that case (before it, the inverted containment-clamp bounds
    `lo_r > hi_r` placed the square at an arbitrary corner of the bbox).
    """
    height, width = maxr - minr, maxc - minc
    min_s = max(height, width)             # smallest square that could contain the bbox at all
    max_s = min(img_height, img_width)     # largest square that can ever fit in this image
    s = int(min(2 * target_half_side, max_s))
    if s >= min_s:
        # Full containment is achievable at this size -- ORIGINAL behavior, unchanged:
        # clamp bounds derived from "must fully contain the bbox".
        lo_r, hi_r = max(0, maxr - s), min(minr, img_height - s)
        lo_c, hi_c = max(0, maxc - s), min(minc, img_width - s)
    else:
        # `target_half_side` is deliberately below the bbox's own tight floor -- full
        # containment is no longer the constraint; only "stay inside the image" is.
        lo_r, hi_r = 0, img_height - s
        lo_c, hi_c = 0, img_width - s
    new_minr = int(min(max(center_row - s / 2.0, lo_r), hi_r))
    new_minc = int(min(max(center_col - s / 2.0, lo_c), hi_c))
    return new_minr, new_minc, new_minr + s, new_minc + s


class Single_Cell_Cropper_Tool(BaseTool):
    """MaskCrop -- SHAPE's deterministic, mask-driven single-cell cropping engine (the
    cropper's analog of Cellpose-SAM for the segmenter): per-cell, square crops cut from
    a segmentation mask. Shipped default (task 038 follow-up, part B redesign, human
    directive): the NAIVE, general-purpose baseline geometry (`margin_mode="fraction",
    margin_frac=0.20, recenter=False`) -- the SAME config as the benchmark's own MaskCrop
    control arm. `margin_frac` raised 0.10 -> 0.20 (task 038 v9 restructure, human-locked):
    a v8 cluster run confirmed the square-tight geometry lever's own quality gain is
    ~0 (it cannot isolate an adjacent neighbour lying beside the bbox's shorter side; since
    the human rule of 2026-09-30 every crop, `mask_bbox` included, is such a square, and
    `mask_bbox`'s square-preserving `auto_tighten` / `bbox_inset_px` cuts are what remove a
    neighbour -- see the `mask_bbox` branch of `_generate_cell_crops`), and the 10%
    baseline was already close to its own ceiling -- 20% gives the agent (and the naive baseline it starts from) genuinely more
    context/headroom to work with. Tight-margin + centroid-recenter geometry (CARD.md SS1 LEVERS) is now
    the AGENT's own per-cell optimization TARGET, applied via `cell_overrides`, not the
    tool's own default (it briefly WAS the shipped default, task 038 quality-geometry
    round, through the part A/B recalibration rounds -- see CARD.md's own history for the
    full sequence). Self-defined, not a third-party model."""

    def __init__(self):
        super().__init__(
            tool_name="Single_Cell_Cropper_Tool",
            tool_description="Generates individual cell/object crops from segmentation masks (nuclei, cells, or organoids) for single-cell analysis. Creates crops for each detected object with configurable margins. Supports masks from Cell_Segmenter_Tool or Organoid_Segmenter_Tool. Automatically handles image dimension mismatches by resizing masks.",
            tool_version="1.0.0",
            input_types={
                "original_image": "str - Path to the original image (brightfield/phase contrast).",
                "nuclei_mask": "str - Path to the segmentation mask image. Accepts nuclei_mask, cell_mask, or organoid_mask from segmentation tools.",
                "source_image_id": "str - Optional source image ID for cell tracking (default: extracted from image path).",
                "group": "str - Optional group/condition label for multi-group analysis (default: 'default').",
                "technical_metadata": "dict - Optional plate/well/site/batch identifiers copied to every crop.",
                "min_area": "int - Minimum object area in pixels. For a cell or nuclei mask: the delivering segmentation's min_size, so every object the segmentation kept gets a crop. Default when no producer is known, by mask file name: cell_mask, nuclei_mask or any other name = the cell segmenter's default min_size (15); organoid_mask = 1000 (organoid masks do not inherit the segmentation's min_size).",
                "margin": "int - Margin around each object for cropping (auto-detected from mask type: cell_mask=1, nuclei_mask=25, organoid_mask=10).",
                "margin_mode": "str - 'fraction' (default, task 038 follow-up part B redesign: margin scales with cell size, giving more surrounding context -- the tool's own naive, general-purpose baseline) or 'tight' (opt-in: minimum margin that keeps the target in-frame + a small anti-clip buffer, minimizing neighbor contamination -- CARD.md's quality-preferred geometry, the agent's own per-cell optimization target via cell_overrides) or 'mask_bbox' (task 038 v9 restructure, opt-in via cell_overrides: the square around the target mask's own bounding box -- its longer side + 2 x tight_buffer_px, centered on the bbox midpoint -- the mode whose per-cell auto_tighten / bbox_inset_px cuts it square-preserving on the neighbour sides; see CARD.md SS1/SS5). Every crop is square in every mode: near the image border the same-size square shifts inside the image.",
                "tight_buffer_px": "int - Anti-clip buffer in pixels used only when margin_mode='tight' or 'mask_bbox' (default: 2).",
                "recenter": "bool - Center the crop on the mask's true centroid instead of its bounding-box midpoint (default: False -- task 038 follow-up part B redesign; True is CARD.md's quality-preferred setting, the agent's own per-cell optimization target via cell_overrides; containment is preserved by growing the square window to the largest centroid-to-bbox-edge distance).",
                "output_format": "str - Output format for crops ('tif', 'png', 'jpg', default: 'png').",
                "output_subdir": "str - Optional relative directory inside tool_cache for this call's files (default: none, the files go into tool_cache itself)."
            },
            output_type="dict - Contains cropped cell images, metadata, and visualization paths.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(original_image="path/to/image.tif", nuclei_mask="path/to/nuclei_mask.png")',
                    "description": "Generate single-cell crops from nuclei segmentation mask with default parameters."
                },
                {
                    "command": 'execution = tool.execute(original_image="path/to/image.tif", nuclei_mask="path/to/cell_mask.png", min_area=15, margin=30)',
                    "description": "Generate single-cell crops from a cell segmentation mask with the segmentation's min_size (here Cell_Segmenter_Tool's default, 15) as min_area and a custom margin."
                },
                {
                    "command": 'execution = tool.execute(original_image="path/to/image.tif", nuclei_mask="path/to/organoid_mask.png")',
                    "description": "Generate organoid crops from organoid segmentation mask."
                }
            ],
            user_metadata={
                "limitation": "Requires segmentation mask results as input (from Cell_Segmenter_Tool or Organoid_Segmenter_Tool). May generate overlapping crops if objects are close together. Performance depends on image quality and segmentation accuracy.",
                "best_practice": "Use with any segmentation tool (Cell_Segmenter_Tool or Organoid_Segmenter_Tool) for optimal results. For a cell or nuclei mask, pass the segmentation's min_size as min_area, so every object the segmentation kept gets a crop (without it: the cell segmenter's default min_size, 15); organoid masks keep their own min_area (default 1000). Adjust margin based on object size and analysis requirements; for organoids, use larger margin values. Automatically handles dimension mismatches between original image and mask."
            }
        )

    def execute(self, original_image, nuclei_mask, source_image_id=None, group="default", min_area=None, margin=None, crop_size=None, margin_frac=0.20, margin_mode="fraction", tight_buffer_px=2, recenter=False, output_format='png', query_cache_dir=None, cell_overrides=None, technical_metadata=None, output_subdir=None):
        """
        Execute single-cell/object cropping from segmentation masks.

        This is a Stage 2 (cell-level) tool. It operates on segmentation results (masks)
        from Cell_Segmenter_Tool or Organoid_Segmenter_Tool,
        and produces CellCrop objects as the atomic data units for downstream analysis.

        Args:
            original_image: Path to original image (brightfield/phase contrast)
            nuclei_mask: Path to segmentation mask (accepts nuclei_mask, cell_mask, or organoid_mask)
            source_image_id: Optional source image ID for tracking (default: extracted from path)
            group: Optional group/condition label (default: 'default')
            min_area: Minimum object area: for a cell or nuclei mask the delivering segmentation's min_size
                (default: by mask file name, `default_min_area_margin` -- the cell segmenter's default
                min_size, 15); an organoid mask keeps its own (default 1000)
            margin: Margin around each object for cropping (default: auto-detected based on mask type)
            cell_overrides: Optional `dict[cell_id, {"margin_mode": str, "recenter": bool,
                "min_containment": float}]` (task 038 follow-up, part B redesign,
                VLRR-internal -- not a general planner-facing knob, see
                `_generate_cell_crops`'s own docstring): per-cell geometry override,
                moving ONE specific crop's OWN margin_mode/recenter (and optionally an
                additional containment floor, `min_containment`, read within
                [`_TARGET_RETENTION_FLOOR`, 1.0]) without perturbing the rest of the image.
                Default None = no overrides.
            margin_mode: 'fraction' (default, task 038 follow-up part B redesign -- the
                tool's own naive, general-purpose baseline; see `effective_margin`'s
                docstring) or 'tight' (opt-in -- CARD.md's quality-preferred geometry, the
                agent's own per-cell optimization target via `cell_overrides`, not the
                tool's own default any more) or 'mask_bbox' (the square around the target's
                mask bbox, see `_generate_cell_crops`). Every crop is square in every mode
                (human rule, 2026-09-30).
            tight_buffer_px: anti-clip buffer in pixels, used only when margin_mode='tight' or 'mask_bbox' (default: 2)
            recenter: center the crop on the mask's true centroid instead of the bounding-box
                midpoint (default: False, task 038 follow-up part B redesign -- True is
                CARD.md's quality-preferred setting, the agent's own per-cell optimization
                target via `cell_overrides`). Containment-preserving when True:
                the square window is sized to the largest centroid-to-bbox-edge distance, not the
                box's own half-extents, so an off-center mask within its bbox is never clipped.
                With or without recenter, `compute_square_crop_containing_bbox` places the
                square: near the image border the same-size square shifts inside the image
                (off-center is tolerated) -- a fully-contained near-edge cell is KEPT, square,
                not clipped and not dropped. Only a cell whose bbox cannot fit inside ANY
                square within the image (genuinely cut off by the image's own edge) is
                truncated by the largest square that fits, and dropped unless its
                `min_containment` allows the loss (see `_generate_cell_crops`).
            output_format: Output image format ('png', 'tif', 'jpg')
            query_cache_dir: Directory for caching results
            output_subdir: Optional relative directory inside `<query_cache_dir>/tool_cache`
                for this call's crops, masks and metadata (default None: `tool_cache`
                itself). Crop and mask file names are deterministic per cell, so callers
                that run several calls on the same query (the agentic crop loop) give each
                call its own directory.

        Returns:
            dict: Cropping results with cell/object crops, CellCrop objects, and metadata
        """
        started = time.time()
        try:
            # Without min_area / margin from the caller: the defaults by mask file name
            # (`default_min_area_margin`; min_area = the segmentation's object definition).
            mask_filename_lower = os.path.basename(nuclei_mask).lower()

            if min_area is None or margin is None:
                auto_min_area, auto_margin = default_min_area_margin(nuclei_mask)
                kind = next((k for k in ("cell_mask", "organoid_mask", "nuclei_mask") if k in mask_filename_lower), None)
                if kind:
                    print(f"Auto-detected {kind} - using min_area={auto_min_area}, margin={auto_margin}")
                else:
                    print(f"Could not detect mask type from filename, using default (nuclei_mask) parameters: min_area={auto_min_area}, margin={auto_margin}")

                # Use auto-detected values if user didn't specify
                if min_area is None:
                    min_area = auto_min_area
                if margin is None:
                    margin = auto_margin

            try:
                min_area, margin, crop_size, tight_buffer_px = pixel_args(min_area, margin, crop_size,
                                                                          tight_buffer_px)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "min_area, margin, crop_size, and tight_buffer_px must be numeric pixel values"
                ) from exc
            if margin_mode not in ("fraction", "tight", "mask_bbox"):
                raise ValueError(
                    f"margin_mode must be 'fraction', 'tight', or 'mask_bbox', got {margin_mode!r}")
            
            # Check if we should use processed image instead of original
            if query_cache_dir and os.path.exists(os.path.join(query_cache_dir, "tool_cache", "query_image_processed.png")):
                processed_image_path = os.path.join(query_cache_dir, "tool_cache", "query_image_processed.png")
                print(f"Using processed image: {processed_image_path}")
                original_img = cv2.imread(processed_image_path, cv2.IMREAD_GRAYSCALE)
                is_multi_channel = False
            else:
                # Load original image using unified abstraction layer
                try:
                    img_data = ImageProcessor.load_image(original_image)
                    print(f"Loaded image: {img_data.shape}, channels: {img_data.num_channels}")
                    
                    # Get image data in (H, W, C) format
                    original_img = img_data.data
                    is_multi_channel = img_data.is_multi_channel
                    
                    if is_multi_channel:
                        print(f"Detected multi-channel image with {img_data.num_channels} channels. Will preserve all channels in crops.")
                except Exception as load_error:
                    # Fallback to legacy loading for backward compatibility
                    print(f"Warning: Failed to load with ImageProcessor: {load_error}, trying legacy method")
                    original_image_lower = original_image.lower()
                    is_tiff = original_image_lower.endswith('.tif') or original_image_lower.endswith('.tiff')
                    
                    if is_tiff:
                        try:
                            original_img_full = tifffile.imread(original_image)
                            if original_img_full.ndim == 2:
                                original_img = original_img_full
                                is_multi_channel = False
                            elif original_img_full.ndim == 3:
                                if original_img_full.shape[2] <= 4:  # (H, W, C)
                                    original_img = original_img_full
                                    is_multi_channel = True
                                elif original_img_full.shape[0] <= 4:  # (C, H, W)
                                    original_img = np.transpose(original_img_full, (1, 2, 0))
                                    is_multi_channel = True
                                else:
                                    original_img = original_img_full[:, :, 0] if original_img_full.shape[2] < original_img_full.shape[0] else original_img_full[0, :, :]
                                    is_multi_channel = False
                            elif original_img_full.ndim == 4:
                                if original_img_full.shape[3] <= 4:  # (Z, H, W, C)
                                    original_img = original_img_full[0, :, :, :]
                                else:  # (C, Z, H, W)
                                    original_img = np.transpose(original_img_full[:, 0, :, :], (1, 2, 0))
                                is_multi_channel = True
                            else:
                                original_img = original_img_full
                                is_multi_channel = False
                        except Exception as tiff_error:
                            print(f"Warning: Failed to load TIFF: {tiff_error}, trying cv2")
                            original_img = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
                            is_multi_channel = False
                    else:
                        original_img = cv2.imread(original_image, cv2.IMREAD_GRAYSCALE)
                        is_multi_channel = False
                    
                    if original_img is None:
                        raise ValueError(f"Cannot read image: {original_image}")
                    
                    # Ensure (H, W, C) format for legacy loading
                    if original_img.ndim == 2:
                        original_img = original_img[:, :, np.newaxis]  # Add channel dimension
            
            # Setup output directory early, so we can save metadata even on errors
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            tool_cache_dir = tool_cache_dir_for(query_cache_dir, output_subdir)
            os.makedirs(tool_cache_dir, exist_ok=True)
            
            # Helper function to save error metadata
            def save_error_metadata(error_type, error_message):
                """Save metadata file even when errors occur."""
                try:
                    # Extract source_image_id if available (may not be set yet for early errors)
                    current_source_image_id = source_image_id if 'source_image_id' in locals() else None
                    if not current_source_image_id and 'original_image' in locals():
                        current_source_image_id = Path(original_image).stem
                    
                    error_metadata = {
                        'cell_count': 0,
                        'cell_crops_paths': [],
                        'cell_metadata': [],
                        'cell_crop_objects': [],
                        'statistics': {},
                        "parameters": {
                            "min_area": min_area,
                            "margin": margin
                        },
                        "execution_status": "error",
                        "error_type": error_type,
                        "error_message": error_message,
                        "source_image_id": current_source_image_id,  # Add source_image_id for deduplication
                        "group": group if 'group' in locals() else "default",
                        "original_image": original_image if 'original_image' in locals() else None
                    }
                    metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_error_{uuid4().hex[:8]}.json")
                    print(f"Single_Cell_Cropper_Tool: Saving error metadata to: {metadata_path}")
                    with open(metadata_path, 'w') as f:
                        json.dump(error_metadata, f, indent=2)
                    print(f"Single_Cell_Cropper_Tool: OK: Error metadata file saved: {metadata_path}")
                    return metadata_path
                except Exception as save_error:
                    print(f"Single_Cell_Cropper_Tool: Error: Failed to save error metadata: {save_error}")
                    return None
            
            if original_img is None:
                metadata_path = save_error_metadata("image_load_failed", f"Failed to load original image: {original_image}")
                return {
                    "error": f"Failed to load original image: {original_image}",
                    "summary": "Image loading failed",
                    "execution_status": "error",
                    "cell_crops_metadata_path": metadata_path
                }
            
            # Load segmentation mask (supports nuclei_mask, cell_mask, or organoid_mask)
            # Debug: log mask path and verify it exists
            print(f"Single_Cell_Cropper_Tool: Loading mask from: {nuclei_mask}")
            print(f"Single_Cell_Cropper_Tool: mask path isabs={os.path.isabs(nuclei_mask)}")
            print(f"Single_Cell_Cropper_Tool: mask file exists={os.path.exists(nuclei_mask)}")
            if not os.path.exists(nuclei_mask):
                print(f"WARNING: Mask file does not exist at: {nuclei_mask}")
                # Try to find the mask file in query_cache_dir if provided
                if query_cache_dir and os.path.isabs(query_cache_dir):
                    alt_mask_path = os.path.join(query_cache_dir, "output_visualizations", os.path.basename(nuclei_mask))
                    print(f"Single_Cell_Cropper_Tool: Trying alternative path: {alt_mask_path}")
                    if os.path.exists(alt_mask_path):
                        print(f"OK: Found mask at alternative path: {alt_mask_path}")
                        nuclei_mask = alt_mask_path
                    else:
                        print(f"Error: Alternative path also does not exist: {alt_mask_path}")
            
            # Load mask based on file format
            # Priority: .npy (NumPy array, preserves original data) > .tif/.tiff (16-bit TIFF) > other formats
            mask = None
            mask_path_lower = nuclei_mask.lower()
            
            if mask_path_lower.endswith('.npy'):
                # .npy format (preferred, preserves original array data)
                mask = np.load(nuclei_mask)
            elif mask_path_lower.endswith('.tif') or mask_path_lower.endswith('.tiff'):
                # .tif/.tiff format (16-bit TIFF, preserves label values up to 65535)
                try:
                    mask = tifffile.imread(nuclei_mask)
                except Exception as e:
                    print(f"Warning: Failed to load mask with tifffile: {e}, trying cv2")
                    mask = cv2.imread(nuclei_mask, cv2.IMREAD_GRAYSCALE)
            else:
                # PNG/other formats (may lose labels >255 in old masks)
                mask = cv2.imread(nuclei_mask, cv2.IMREAD_GRAYSCALE)
            
            # Unified post-processing: ensure 2D array and convert to uint16
            if mask is not None:
                mask = np.asarray(mask)
                if len(mask.shape) > 2:
                    mask = mask.squeeze()
                mask = mask.astype(np.uint16)  # Convert to uint16 for consistency (supports up to 65535 labels)
            
            if mask is None:
                metadata_path = save_error_metadata("mask_load_failed", f"Failed to load segmentation mask: {nuclei_mask}")
                return {
                    "error": f"Failed to load segmentation mask: {nuclei_mask}",
                    "summary": "Mask loading failed. Please provide a valid mask from Cell_Segmenter_Tool or Organoid_Segmenter_Tool.",
                    "execution_status": "error",
                    "cell_crops_metadata_path": metadata_path
                }
            
            # Extract source_image_id from path if not provided
            if source_image_id is None:
                source_image_id = Path(original_image).stem
            technical_metadata = dict(technical_metadata or {})
            
            # Generate cell crops and CellCrop objects
            cell_crops, cell_metadata, cell_crop_objects, stats = self._generate_cell_crops(
                original_img, mask, original_image, source_image_id, group, min_area, margin, tool_cache_dir, output_format, is_multi_channel, crop_size, margin_frac, margin_mode, tight_buffer_px, recenter, cell_overrides, technical_metadata
            )
            
            # Always save metadata file, even if no crops were generated
            # This ensures Cell_State_Analyzer_Tool knows Single_Cell_Cropper_Tool has executed
            # Combine paths and metadata for JSON output (backward compatibility)
            output_data = {
                'cell_count': stats.get('final_cell_count', 0) if cell_crops else 0,
                'cell_crops_paths': cell_crops if cell_crops else [],
                'cell_metadata': cell_metadata if cell_crops else [],
                'cell_crop_objects': [cell.to_dict() for cell in cell_crop_objects] if cell_crop_objects else [],  # Serialize CellCrop objects
                'statistics': stats if cell_crops else {},
                "parameters": {
                    "min_area": min_area,
                    "margin": margin,
                    "margin_mode": margin_mode,
                    "recenter": recenter,
                    "crop_size": crop_size,
                    # Task 038 follow-up (part B, result-driven per-cell tighten): echoed
                    # back so a caller (CropVerifier.adjudicate, via adapt_crop's own
                    # context["cell_overrides"]) can read what was ALREADY applied to
                    # produce THIS result, to merge a newly-flagged cell into it on the
                    # next retune rather than dropping a previous iteration's tighten.
                    "cell_overrides": cell_overrides or {}
                },
                "execution_status": "success" if cell_crops else "no_crops_generated",
                "source_image_id": source_image_id,  # Add source_image_id to identify which image this metadata belongs to
                "group": group,  # Add group to identify which group this metadata belongs to
                "technical_metadata": technical_metadata,
                "original_image": original_image  # Add original_image path for reference
            }
            metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_{uuid4().hex[:8]}.json")
            print(f"Single_Cell_Cropper_Tool: Saving metadata to: {metadata_path}")
            print(f"Single_Cell_Cropper_Tool: query_cache_dir={query_cache_dir}, tool_cache_dir={tool_cache_dir}")
            print(f"Single_Cell_Cropper_Tool: cell_count={output_data['cell_count']}, execution_status={output_data['execution_status']}")
            with open(metadata_path, 'w') as f:
                json.dump(output_data, f, indent=2)
            # Verify file was saved
            if os.path.exists(metadata_path):
                print(f"Single_Cell_Cropper_Tool: OK: Metadata file saved successfully: {metadata_path}")
            else:
                print(f"Single_Cell_Cropper_Tool: ERROR: Metadata file was not saved: {metadata_path}")
            
            if not cell_crops:
                return {
                    "summary": "No valid cell crops generated. Metadata file saved for tracking.",
                    "cell_count": 0,
                    "cell_crops": [],
                    "cell_metadata": [],
                    "cell_crop_objects": [],  # Empty list of CellCrop objects
                    "visual_outputs": [],
                    "cell_crops_metadata_path": metadata_path,  # Include metadata path even when no crops
                    "parameters": {
                        "min_area": min_area,
                        "margin": margin,
                        "output_format": output_format
                    },
                    "statistics": stats,
                    "execution_status": "no_crops_generated"
                }

            # Build detailed summary with processing statistics
            summary_parts = [f"Successfully generated {stats['final_cell_count']} single-cell crops."]
            
            # Add filtering statistics if available
            total_detected = stats.get('total_objects', stats.get('initial_cell_count', 0))
            filtered_by_area = stats.get('filtered_by_area', 0)
            filtered_by_border = stats.get('filtered_by_border', 0)
            invalid_crop_data = stats.get('invalid_crop_data', 0)
            total_filtered = filtered_by_area + filtered_by_border + invalid_crop_data

            # Add processing statistics to summary
            stats_text = f"\n\n**Processing Statistics:**\n"
            stats_text += f"- Total objects detected: {total_detected}\n"
            stats_text += f"- Final cell count: {stats['final_cell_count']}\n"
            stats_text += f"- Filtered by area (min_area={min_area}px): {filtered_by_area}\n"
            stats_text += f"- Filtered by border constraints: {filtered_by_border}\n"
            stats_text += f"- Invalid crop data: {invalid_crop_data}\n"
            stats_text += f"- Total filtered: {total_filtered}\n"
            stats_text += f"- Parameters: min_area={min_area}px, margin={margin}px"
            
            if total_detected > 0:
                if total_filtered > 0:
                    filter_details = []
                    if filtered_by_area > 0:
                        filter_details.append(f"{filtered_by_area} by min_area={min_area}px")
                    if filtered_by_border > 0:
                        filter_details.append(f"{filtered_by_border} by border constraints")
                    if invalid_crop_data > 0:
                        filter_details.append(f"{invalid_crop_data} invalid crops")
                    
                    filter_summary = ", ".join(filter_details)
                    summary_parts.append(f"Detected {total_detected} objects in mask, filtered out {total_filtered} ({filter_summary}).")
                    summary_parts.append(f"({total_detected} detected -> {stats['final_cell_count']} valid crops)")
                else:
                    summary_parts.append(f"All {total_detected} detected objects were successfully cropped.")
            
            summary_parts.append(f"All paths and metadata are saved in '{Path(metadata_path).name}'.")
            summary_parts.append(stats_text)
            
            # Create zip archive of all crop files for download
            crops_zip_path = None
            deliverables = []
            if cell_crops and len(cell_crops) > 0:
                try:
                    # Create zip filename with image identifier and group
                    image_name_safe = "".join(c for c in (source_image_id if source_image_id else "crops") if c.isalnum() or c in ('_', '-'))[:50]
                    group_safe = "".join(c for c in group if c.isalnum() or c in ('_', '-'))[:30] if group else "default"
                    zip_filename = f"{image_name_safe}_{group_safe}_crops.zip"
                    crops_zip_path = os.path.join(tool_cache_dir, zip_filename)
                    
                    # Create zip file with all crop files
                    with zipfile.ZipFile(crops_zip_path, 'w', zipfile.ZIP_DEFLATED) as zipf:
                        # Add all crop files
                        for crop_path in cell_crops:
                            if os.path.exists(crop_path):
                                # Add file with relative path (just filename) to keep zip structure clean
                                zipf.write(crop_path, os.path.basename(crop_path))
                        
                        # Also add metadata file to zip
                        if os.path.exists(metadata_path):
                            zipf.write(metadata_path, os.path.basename(metadata_path))
                    
                    # Verify zip file was created
                    if os.path.exists(crops_zip_path) and os.path.getsize(crops_zip_path) > 0:
                        deliverables.append(crops_zip_path)
                        print(f"OK: Created crops zip archive: {crops_zip_path} ({os.path.getsize(crops_zip_path)} bytes)")
                    else:
                        print(f"Warning: Failed to create crops zip archive")
                except Exception as zip_error:
                    print(f"Warning: Failed to create crops zip archive: {zip_error}")
                    import traceback
                    traceback.print_exc()
            
            return {
                "summary": " ".join(summary_parts),
                "cell_count": stats['final_cell_count'],
                "total_detected_objects": total_detected,
                "filtered_by_area": filtered_by_area,
                "filtered_by_border": filtered_by_border,
                "invalid_crop_data": invalid_crop_data,
                "cell_crops_metadata_path": metadata_path,
                "cell_crop_objects": cell_crop_objects,  # Return CellCrop objects for Stage 2 tools
                "visual_outputs": [],  # No summary visualization
                "crops_zip_path": crops_zip_path if crops_zip_path and os.path.exists(crops_zip_path) else None,  # Zip file path for download (not in deliverables)
                "processing_statistics": stats,  # Include processing statistics in result
                "seconds": round(time.time() - started, 1),  # this call's run time (the agentic loop shows it)
            }
            
        except Exception as e:
            print(f"Single_Cell_Cropper_Tool: Error in single-cell cropping: {e}")
            import traceback
            traceback.print_exc()
            
            # Try to save metadata file even on error, so Cell_State_Analyzer_Tool knows the tool executed
            try:
                if query_cache_dir is None:
                    query_cache_dir = "solver_cache/temp"
                try:
                    tool_cache_dir = tool_cache_dir_for(query_cache_dir, output_subdir)
                except ValueError:
                    tool_cache_dir = tool_cache_dir_for(query_cache_dir)
                os.makedirs(tool_cache_dir, exist_ok=True)

                # Extract source_image_id if available
                current_source_image_id = source_image_id if 'source_image_id' in locals() else None
                if not current_source_image_id and 'original_image' in locals():
                    current_source_image_id = Path(original_image).stem
                
                error_metadata = {
                    'cell_count': 0,
                    'cell_crops_paths': [],
                    'cell_metadata': [],
                    'cell_crop_objects': [],
                    'statistics': {},
                    "parameters": {
                        "min_area": min_area if 'min_area' in locals() else None,
                        "margin": margin if 'margin' in locals() else None
                    },
                    "execution_status": "error",
                    "error_message": str(e),
                    "source_image_id": current_source_image_id,  # Add source_image_id for deduplication
                    "group": group if 'group' in locals() else "default",
                    "original_image": original_image if 'original_image' in locals() else None
                }
                metadata_path = os.path.join(tool_cache_dir, f"cell_crops_metadata_error_{uuid4().hex[:8]}.json")
                print(f"Single_Cell_Cropper_Tool: Saving error metadata to: {metadata_path}")
                with open(metadata_path, 'w') as f:
                    json.dump(error_metadata, f, indent=2)
                print(f"Single_Cell_Cropper_Tool: OK: Error metadata file saved: {metadata_path}")
            except Exception as save_error:
                print(f"Single_Cell_Cropper_Tool: Error: Failed to save error metadata: {save_error}")
            
            return {
                "error": f"Error during single-cell cropping: {str(e)}",
                "summary": "Failed to process image",
                "execution_status": "error"
            }

    def _generate_cell_crops(self, original_img, mask, original_image_path, source_image_id, group, min_area, margin, output_dir, output_format, is_multi_channel=False, crop_size=None, margin_frac=0.20, margin_mode="fraction", tight_buffer_px=2, recenter=False, cell_overrides=None, technical_metadata=None):
        """
        Generate individual cell crops from the mask and create CellCrop objects.

        Unified logic for all image types:
        - Single-channel images: saved as single-channel (grayscale) in specified format
        - Multi-channel images: saved as multi-channel TIFF to preserve all channel information

        This method produces CellCrop objects as atomic data units for Stage 2 analysis.

        Every crop it writes is SQUARE (crop height == crop width, human rule 2026-09-30:
        the analyzer resizes every crop to 224 x 224 without keeping the aspect ratio, so a
        rectangle would stretch its cell), cut from real image pixels (no constant padding,
        no masking of neighbours): every `margin_mode`, with or without `recenter`,
        `cell_overrides`, `crop_size`, `min_containment`, `bbox_inset_px` and `auto_tighten`,
        at the image border too -- `compute_square_crop_containing_bbox` places every crop
        and `_clamp_directional_inset` keeps every directional cut square; a safety net
        (before the containment check) replaces any non-square box by the square around the
        target's bbox, so a rectangle is never written.

        `cell_overrides` (task 038 follow-up, part B redesign, result-driven per-cell
        geometry): `dict[cell_id, {"margin_mode": str, "recenter": bool,
        "min_containment": float}] | None` (all 3 keys optional per cell) -- lets a
        caller (the VLRR retune loop) move ONE specific crop's OWN geometry (by its own
        `cell_id`, the SAME string this method's own metadata reports) without
        perturbing any other cell in this image, unlike the GLOBAL `margin_mode`/
        `recenter`/`crop_size` args above (which apply uniformly to every cell). Each
        key falls back to the image-level arg above when omitted. `min_containment`
        (read within [`_TARGET_RETENTION_FLOOR`, 1.0] -- a lower value acts as the floor;
        1.0 = no floor, the default) is the additional floor this cell's
        `mask_containment` is allowed to shrink to, ON TOP of whatever `margin_mode`/
        `recenter` resolve to -- see the sizing block below for the mechanics.
        TIGHTEN-ONLY by convention (the caller decides the values; this method applies
        them, `min_containment` clamped as above) -- a caller is expected to only ever move a cell TOWARD tighter/
        recentered geometry, never wider, matching every other geometry lever in this
        tool (see CARD.md SS1/SS5).

        Args:
            original_img: Original image array (can be 2D grayscale or multi-channel)
            mask: Binary mask array
            original_image_path: Path to the original source image
            source_image_id: Source image ID for tracking
            group: Group/condition label
            min_area: Minimum area threshold
            margin: Margin around each nucleus
            output_dir: Output directory
            output_format: Output image format (for single-channel; multi-channel always uses TIFF)
            is_multi_channel: Whether the original image is multi-channel (preserve all channels in crops)
            
        Returns:
            tuple: (list of crop paths, list of metadata dicts, list of CellCrop objects, dict of statistics)
        """
        # Validate input images
        if original_img is None or mask is None:
            print("Error: Input images are None")
            return [], [], [], {}
        
        # Ensure images are numpy arrays
        original_img = np.asarray(original_img)
        mask = np.asarray(mask)

        # Task 038 (v9 CRITICAL fix): normalize the SOURCE image to display uint8 ONCE,
        # up front, before any crop is cut -- see `normalize_source_image`'s own
        # docstring for the bug this fixes (mixed-modality inputs, e.g. 16-bit
        # fluorescence or float [0,1], were blowing every crop to pure white/black
        # under the downstream per-crop NAIVE clip). Mask-based QC below reads `mask`/
        # `labeled_mask` only, never `original_img`'s pixel values -- unaffected.
        original_img = normalize_source_image(original_img)

        # For multi-channel images, extract first channel for dimension checking
        if is_multi_channel and original_img.ndim == 3:
            original_img_2d = original_img[:, :, 0]  # Use first channel (bright-field) for dimension check
        else:
            original_img_2d = original_img
        
        # Check image dimensions (compare 2D shapes)
        if original_img_2d.shape[:2] != mask.shape[:2]:
            print(f"Error: Image shape mismatch - original: {original_img.shape}, mask: {mask.shape}")
            return [], [], [], {}
        
        # Handle mask format: Cellpose generates label masks (0=background, 1-N for N cells)
        # We should preserve the original labels, not convert to binary
        # Check if mask is already a label mask (multiple unique values) or binary
        unique_values = np.unique(mask)
        num_unique = len(unique_values)
        
        if num_unique > 2:
            # Label mask from Cellpose: use directly with regionprops
            # Each unique value (except 0) represents a distinct cell/object
            print(f"Detected label mask with {num_unique - 1} unique objects (excluding background)")
            # Use the mask directly - regionprops can work with label masks
            labeled_mask = mask
        else:
            # Binary mask: need to label connected components
            print("Detected binary mask, labeling connected components")
            labeled_mask = label(mask)
        
        cell_properties = regionprops(labeled_mask)
        # Full-image work once per call, not once per cell (2026-10-01; the same values, a large image's per-cell
        # passes dominated the call): the label areas every per-crop QC counts complete cells with, and one
        # full-field target mask shared by the cells -- each cell's pixels set (255) within its bbox while it
        # is cropped, cleared at the next cell (`_show_target`), the values of a fresh `labeled_mask == label`
        # mask of the mask's dtype.
        full_field_areas = np.bincount(labeled_mask.ravel())
        cell_mask = np.zeros_like(mask)
        shown = None                                   # the region whose pixels `cell_mask` holds

        initial_cell_count = len(cell_properties)
        filtered_by_area = 0
        filtered_by_border = 0
        invalid_crop_data = 0

        cell_crops = []
        cell_metadata = []
        cell_crop_objects = []  # List of CellCrop objects

        # Extract image name for unique crop filenames/cell_id (to avoid conflicts when
        # processing multiple images) -- hoisted here (was recomputed, identically, every
        # loop iteration below) so `cell_id` can be resolved EARLY per cell, before
        # sizing, for the `cell_overrides` lookup (task 038 follow-up, part B).
        image_name = source_image_id if source_image_id else Path(original_image_path).stem
        image_name_safe = "".join(c for c in image_name if c.isalnum() or c in ('_', '-'))[:50]  # Limit length

        for idx, region in enumerate(cell_properties):
            if region.area < min_area:
                filtered_by_area += 1
                continue

            # Task 038 follow-up (part B, result-driven per-cell tighten): resolve THIS
            # cell's own cell_id EARLY (before sizing) so a caller-supplied
            # `cell_overrides[cell_id]` can affect the sizing below. cell_id depends only
            # on group/image_name (both invariant per call) and idx (stable across
            # repeated calls on the SAME mask -- regionprops iterates in the same label
            # order every time), matching the metadata's own `cell_id` field exactly.
            crop_id = f"cell_{idx:04d}"
            cell_id = f"{group}_{image_name}_{crop_id}"
            cell_override = (cell_overrides or {}).get(cell_id) or {}
            minr, minc, maxr, maxc = region.bbox

            # Task 038 architectural FIX round: `region_label`/`cell_mask` (the
            # full-field binary mask for THIS cell) -- HOISTED here from their own
            # original position further below (just before crop-save) because
            # `auto_tighten`'s internal loop (below) needs this cell's own mask to
            # recompute qc BEFORE the crop box is finalized. Same computation, same
            # cost, just earlier -- the crop-save section below re-slices this SAME
            # `cell_mask` to the FINAL box, no redundant work added. The target is the
            # region's own label (2026-10-01; it was the label under the centroid pixel,
            # which for an object whose centroid lies outside it -- an L, a ring -- is
            # background or another object, so every readout described the wrong label).
            # `cell_mask` is the call's shared target mask (above): the previous cell's
            # pixels cleared, this cell's set.
            region_label = region.label
            if shown is not None:
                self._show_target(cell_mask, shown, 0)
            self._show_target(cell_mask, region, 255)
            shown = region

            # The crop box -- the cell's geometry alone (`cell_crop_box`, which the agentic loop also runs
            # before a call to key a configuration by its crops); None: the cell is dropped.
            plan = self.cell_crop_box(
                region, labeled_mask, cell_mask, original_img.shape[:2], cell_override, idx=idx,
                min_area=min_area, margin=margin, crop_size=crop_size, margin_frac=margin_frac,
                margin_mode=margin_mode, tight_buffer_px=tight_buffer_px, recenter=recenter,
                full_field_areas=full_field_areas)
            if plan["box"] is None:
                filtered_by_border += 1
                continue
            new_minr, new_minc, new_maxr, new_maxc = plan["box"]
            cell_margin_mode, cell_recenter = plan["margin_mode"], plan["recenter"]
            applied_bbox_inset = plan["bbox_inset_px"]
            auto_tighten_iterations = plan["auto_tighten_iterations"]

            # Crop from original image (preserve all channels if multi-channel)
            # ImageData always returns (H, W, C) format where C >= 1
            # - Single-channel: (H, W, 1) when using ImageData
            # - Multi-channel: (H, W, C) where C > 1
            # Fallback loading may produce (H, W) which is converted to (H, W, 1) at line 172-173
            # So original_img should always be 3D at this point, but handle both cases for safety
            if original_img.ndim == 3:
                cell_crop = original_img[new_minr:new_maxr, new_minc:new_maxc, :]  # Preserve all channels
            elif original_img.ndim == 2:
                # Fallback: 2D grayscale (should have been converted earlier, but handle for safety)
                cell_crop = original_img[new_minr:new_maxr, new_minc:new_maxc]
                cell_crop = cell_crop[:, :, np.newaxis]  # Add channel dimension -> (H, W, 1)
            else:
                raise ValueError(f"Unexpected image dimensions: {original_img.ndim}, expected 2 or 3")
            
            # Validate crop data
            if cell_crop.size == 0 or np.isnan(cell_crop).any() or np.isinf(cell_crop).any():
                invalid_crop_data += 1
                print(f"Warning: Invalid crop data for cell {idx}, skipping (size={cell_crop.size}, has_nan={np.isnan(cell_crop).any()}, has_inf={np.isinf(cell_crop).any()})")
                continue

            # Debug: log crop shape and dtype
            print(f"Cell {idx}: crop shape={cell_crop.shape}, dtype={cell_crop.dtype}, is_multi_channel={is_multi_channel}")
            
            # Ensure crop is in valid range
            if cell_crop.dtype == np.float32 or cell_crop.dtype == np.float64:
                cell_crop = np.clip(cell_crop, 0, 255).astype(np.uint8)
            
            # image_name / image_name_safe: hoisted above the loop (task 038 follow-up,
            # part B) -- same value every iteration, unchanged here.

            # Save crop with enhanced error handling
            # For multi-channel images, always use TIFF to preserve all channel information
            # For single-channel images, use the specified output_format
            # Include image_name in filename to ensure uniqueness across different source images
            if is_multi_channel and cell_crop.ndim == 3:
                crop_filename = f"{image_name_safe}_cell_{idx:04d}_crop.tiff"
            else:
                crop_filename = f"{image_name_safe}_cell_{idx:04d}_crop.{output_format}"
            crop_path = os.path.join(output_dir, crop_filename)
            crop_path = os.path.abspath(crop_path)  # Ensure absolute path
            
            try:
                # Unified logic: preserve all channel information
                if is_multi_channel and cell_crop.ndim == 3 and cell_crop.shape[2] > 1:
                    # Multi-channel image: save as TIFF to preserve all channels
                    # Convert (H, W, C) to (C, H, W) format for ImageJ compatibility
                    # This ensures ImageJ correctly interprets channels (not as Z-stack)
                    cell_crop_chw = np.moveaxis(cell_crop, -1, 0)  # Move channel dimension to first
                    tifffile.imwrite(crop_path, cell_crop_chw, imagej=True)
                elif cell_crop.ndim == 2:
                    # Grayscale image (2D) - convert to PIL and save
                    # Ensure dtype is uint8 for PIL
                    if cell_crop.dtype != np.uint8:
                        cell_crop = np.clip(cell_crop, 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(cell_crop, mode='L')
                elif cell_crop.ndim == 3 and cell_crop.shape[2] == 1:
                    # Single-channel image stored as (H, W, 1) - squeeze and save as grayscale
                    cell_crop_2d = cell_crop.squeeze(axis=2)
                    # Ensure dtype is uint8 for PIL
                    if cell_crop_2d.dtype != np.uint8:
                        cell_crop_2d = np.clip(cell_crop_2d, 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(cell_crop_2d, mode='L')
                elif cell_crop.ndim == 3 and cell_crop.shape[2] == 3:
                    # RGB image (3 channels) - convert to PIL and save
                    # Ensure dtype is uint8 for PIL
                    if cell_crop.dtype != np.uint8:
                        cell_crop = np.clip(cell_crop, 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(cell_crop, mode='RGB')
                elif cell_crop.ndim == 3 and cell_crop.shape[2] == 4:
                    # RGBA image (4 channels) - convert to PIL and save
                    # Ensure dtype is uint8 for PIL
                    if cell_crop.dtype != np.uint8:
                        cell_crop = np.clip(cell_crop, 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(cell_crop, mode='RGBA')
                else:
                    # Fallback: try to save as grayscale by taking first channel
                    print(f"Warning: Unexpected crop shape {cell_crop.shape} for cell {idx}, converting to grayscale")
                    if cell_crop.ndim == 3:
                        cell_crop_2d = cell_crop[:, :, 0]
                    else:
                        cell_crop_2d = cell_crop
                    # Ensure dtype is uint8 for PIL
                    if cell_crop_2d.dtype != np.uint8:
                        cell_crop_2d = np.clip(cell_crop_2d, 0, 255).astype(np.uint8)
                    pil_image = Image.fromarray(cell_crop_2d, mode='L')
                
                # Save with proper format and error handling
                # Note: Multi-channel images saved with tifffile above, skip PIL save for those
                if not (is_multi_channel and cell_crop.ndim == 3):
                    # Validate PIL image
                    if pil_image.size[0] == 0 or pil_image.size[1] == 0:
                        print(f"Warning: Invalid PIL image size for cell {idx}, skipping")
                        continue
                    
                    # Save with proper format and error handling
                    if output_format.lower() == 'tif':
                        pil_image.save(crop_path, 'TIFF', compression='tiff_lzw')
                    elif output_format.lower() == 'png':
                        pil_image.save(crop_path, 'PNG', optimize=True)
                    elif output_format.lower() == 'jpg':
                        pil_image.save(crop_path, 'JPEG', quality=95, optimize=True)
                    else:
                        # Default to PNG for compatibility
                        pil_image.save(crop_path, 'PNG', optimize=True)
                
                # Verify the saved file
                if not os.path.exists(crop_path) or os.path.getsize(crop_path) == 0:
                    print(f"Warning: Failed to save crop for cell {idx} to {crop_path} (exists={os.path.exists(crop_path)}, size={os.path.getsize(crop_path) if os.path.exists(crop_path) else 0})")
                    invalid_crop_data += 1
                    continue
                
                # Test loading the saved image
                try:
                    test_image = Image.open(crop_path)
                    test_image.verify()
                    test_image.close()
                except Exception as e:
                    print(f"Warning: Saved image verification failed for cell {idx}: {e}")
                    invalid_crop_data += 1
                    continue
                
            except Exception as e:
                print(f"Error saving crop for cell {idx}: {e}")
                import traceback
                traceback.print_exc()
                invalid_crop_data += 1
                continue
            
            # cell_id: hoisted above the loop (task 038 follow-up, part B) -- same value
            # every reference here, unchanged (format: {group}_{image_name}_{crop_id},
            # e.g. "control_image1_cell_0001").

            # Create metadata for this crop (backward compatibility)
            crop_metadata = {
                "cell_id": cell_id,  # Use full cell_id string instead of idx
                "crop_path": crop_path,
                "original_bbox": [minr, minc, maxr, maxc],
                "crop_bbox": [new_minr, new_minc, new_maxr, new_maxc],
                "area": region.area,
                "centroid": list(region.centroid),  # Convert tuple to list for JSON serialization
                "crop_size": cell_crop.shape,
                "group": group,  # Include group in metadata
                "image_name": image_name,  # Include image_name in metadata
                # task 042/038 follow-up (part B): THIS cell's OWN resolved margin_mode/
                # recenter (a per-cell `cell_overrides` entry may differ from the
                # image-level defaults above -- these report what was ACTUALLY used).
                "margin_mode": cell_margin_mode,
                "recenter": cell_recenter,
                # task 038 v12: THIS cell's OWN resolved directional inset -- the
                # ACTUALLY-applied amount (after clamping AND any never-discard
                # scale-back, above), NOT necessarily the originally-requested one; {}
                # when no inset was requested at all, so a plain mask_bbox crop's
                # metadata is unchanged from before this round -- so the trace/QC can
                # see exactly what was applied.
                "bbox_inset_px": dict(applied_bbox_inset) if cell_margin_mode == "mask_bbox" else {},
                # Task 038 architectural FIX round: how many internal `_auto_tighten_
                # mask_bbox` steps this cell's own final `bbox_inset_px` above took to
                # reach (0 when `auto_tighten` was not requested, or requested but
                # nothing was cuttable) -- keeps the mechanism/trace inspectable now
                # that the round count no longer shows up as separate VLRR trials.
                "auto_tighten_iterations": auto_tighten_iterations,
            }
            # Technical identifiers are observational metadata, never biological labels.
            # Copy them per cell so downstream concatenation cannot lose image-level context.
            crop_metadata.update(dict(technical_metadata or {}))

            # Create CellCrop object (atomic data unit for Stage 2)
            # Extract individual cell mask -- `region_label`/`cell_mask` are HOISTED
            # above (task 038 architectural FIX round, `auto_tighten` needs them before
            # the crop box is finalized); re-sliced here to the FINAL box.
            cell_mask_crop = cell_mask[new_minr:new_maxr, new_minc:new_maxc]

            # VLRR per-crop QC metrics (consumed by the crop verifier via adapters)
            crop_metadata["qc"] = self._compute_crop_qc(
                region, region_label, labeled_mask, cell_mask_crop,
                new_minr, new_minc, new_maxr, new_maxc, min_area, full_field_areas=full_field_areas
            )

            # Save cell mask if needed
            mask_filename = f"cell_{idx:04d}_mask.{output_format}"
            mask_path = os.path.join(output_dir, mask_filename)
            mask_path = os.path.abspath(mask_path)  # Ensure absolute path
            try:
                # Task 038 CRITICAL fix (human-directed): `cell_mask_crop` is a slice
                # of `cell_mask = np.zeros_like(mask)` (above), which INHERITS the
                # full-field label mask's own dtype -- uint16 for NeurIPS22-CellSeg.
                # PIL's `mode='L'` is 8-BIT; forcing it onto a uint16 buffer re-reads
                # 2 bytes/pixel as 1 byte/pixel, producing a 2x-subsampled lattice (a
                # dotted grid with the right silhouette but half the pixels and no
                # connectivity), NOT the real mask -- 100% of saved masks were affected
                # in a real run (measured: mean horizontal-foreground-neighbour
                # fraction 0.00 across every sampled cell). `(cell_mask_crop > 0)` binarizes
                # REGARDLESS of the source dtype (0/255 either way) and `.astype(np.uint8)
                # * 255` casts to a REAL 8-bit buffer before `fromarray` -- no explicit
                # `mode=` (letting PIL infer it from the now-genuinely-uint8 array): an
                # explicit `mode='L'` on a future non-uint8 array is exactly the silent
                # mis-read that caused this bug, so PIL is left to infer/raise instead
                # of being told to trust a byte count that might be wrong again.
                if len(cell_mask_crop.shape) == 2:
                    mask_pil = Image.fromarray((cell_mask_crop > 0).astype(np.uint8) * 255)
                else:
                    mask_pil = Image.fromarray((cell_mask_crop > 0).astype(np.uint8) * 255)

                if output_format.lower() == 'tif':
                    mask_pil.save(mask_path, 'TIFF', compression='tiff_lzw')
                elif output_format.lower() == 'png':
                    mask_pil.save(mask_path, 'PNG', optimize=True)
                elif output_format.lower() == 'jpg':
                    mask_pil.save(mask_path, 'JPEG', quality=95, optimize=True)
                else:
                    mask_pil.save(mask_path, 'PNG', optimize=True)
            except Exception as e:
                print(f"Warning: Failed to save mask for cell {idx}: {e}")
                mask_path = None

            # Task 038 (v9 restructure): thread mask_path into the JSON-persisted
            # `cell_metadata` too (previously only on the separate `CellCrop` object,
            # below, which the crop-stage VLRR verifier's own `adapt_crop` never reads)
            # -- so `crop_triage.py`'s overlay-drawing judge evidence can find the
            # target's own per-cell mask file at judge time. None on a save failure,
            # same graceful-degrade as the CellCrop object's own field.
            crop_metadata["mask_path"] = mask_path

            # Task 038 v13 (human-directed): render + persist the mask_bbox BASELINE
            # crop AND its own matching mask -- the plain, zero-inset box (`buf_minr`/
            # etc, the mask_bbox branch's square, `cell_crop_box`'s `baseline_box`)
            # -- for every cell with a nonzero APPLIED inset this round, so
            # `crop_triage.ab_recommend_crop`'s blinded A/B channel has BOTH real
            # images to compare (baseline vs. the tightened candidate), EACH with its
            # own correctly-sized mask for the colourful overlay (a mismatched mask
            # size would silently fall back to no overlay, `_overlay_target_on_crop`'s
            # own safety net -- a SEPARATE baseline-sized mask avoids that entirely,
            # reusing the SAME `cell_mask` full-field array the primary mask above
            # already sliced, just to the baseline's own box instead). Both stay None
            # for a cell not on `mask_bbox` at all, or on `mask_bbox` with no inset
            # applied yet (baseline == candidate, nothing to compare).
            baseline_crop_path = None
            baseline_mask_path = None
            if cell_margin_mode == "mask_bbox" and any(
                    applied_bbox_inset.get(k) for k in ("top", "bottom", "left", "right")):
                baseline_box = plan["baseline_box"]
                buf_minr, buf_minc, buf_maxr, buf_maxc = baseline_box
                baseline_crop_path = _save_baseline_crop(
                    original_img, baseline_box,
                    os.path.abspath(os.path.join(
                        output_dir, f"{image_name_safe}_cell_{idx:04d}_baseline.{output_format}")),
                    is_multi_channel, output_format)
                baseline_mask_crop = cell_mask[buf_minr:buf_maxr, buf_minc:buf_maxc]
                baseline_mask_path = os.path.abspath(
                    os.path.join(output_dir, f"cell_{idx:04d}_baseline_mask.{output_format}"))
                try:
                    # Task 038 CRITICAL fix (human-directed): the SAME uint16-buffer-
                    # read-as-8-bit corruption the primary mask save above has -- this
                    # is a slice of the SAME `cell_mask` array. Same fix: binarize +
                    # cast to a real uint8 buffer, no explicit `mode=`.
                    Image.fromarray((baseline_mask_crop > 0).astype(np.uint8) * 255).save(
                        baseline_mask_path,
                        'TIFF' if output_format.lower() == 'tif' else
                        ('JPEG' if output_format.lower() == 'jpg' else 'PNG'))
                except Exception as e:
                    print(f"Warning: Failed to save baseline mask for cell {idx}: {e}")
                    baseline_mask_path = None
            crop_metadata["baseline_crop_path"] = baseline_crop_path
            crop_metadata["baseline_mask_path"] = baseline_mask_path

            cell_crop_obj = CellCrop(
                cell_id=cell_id,
                crop_id=crop_id,
                source_image_id=source_image_id,
                source_image_path=original_image_path,
                group=group,
                crop_path=crop_path,
                mask_path=mask_path,
                bbox=[new_minr, new_minc, new_maxr, new_maxc],
                centroid=list(region.centroid),
                area=int(region.area),
                metadata={
                    "original_bbox": [int(minr), int(minc), int(maxr), int(maxc)],
                    "crop_size": list(cell_crop.shape),
                    "min_area": min_area,
                    "margin": margin,
                    "margin_mode": cell_margin_mode,   # task 038 follow-up (part B): per-cell resolved
                    "recenter": cell_recenter
                }
            )
            
            cell_crops.append(crop_path)
            cell_metadata.append(crop_metadata)
            cell_crop_objects.append(cell_crop_obj)
        
        stats = {
            "total_objects": initial_cell_count,  # Total objects detected in mask
            "initial_cell_count": initial_cell_count,
            "filtered_by_area": filtered_by_area,
            "filtered_by_border": filtered_by_border,
            "invalid_crop_data": invalid_crop_data,
            "final_cell_count": len(cell_crops)
        }

        return cell_crops, cell_metadata, cell_crop_objects, stats

    def cell_crop_box(self, region, labeled_mask, cell_mask, image_shape, cell_override, *, idx, min_area,
                      margin, crop_size, margin_frac, margin_mode, tight_buffer_px, recenter,
                      full_field_areas=None):
        """The crop box of one object (`region`, a regionprops region of `labeled_mask`) -- its geometry
        alone, before any pixel is cut: the per-cell keys of `cell_override` resolved over the image-level
        arguments (read the way `execute` reads them, `pixel_args`), the square sized and placed, cut by
        `bbox_inset_px` / `auto_tighten`, and kept or dropped by the containment rules. A pure function of the
        label mask and the arguments (no image, nothing written; it prints the tool's warnings):
        `_generate_cell_crops` cuts the crop it returns, and the agentic loop runs it before a call to key a
        configuration by the crops it would cut (`verifiers/crop.py::CropAgenticVerifier.config_key`).
        `cell_mask` is the target's full-field mask, on which `auto_tighten` measures its cuts (None: built
        when needed); `full_field_areas` the label areas its QC counts complete cells with (None: counted
        there). Returns {"box": (minr, minc, maxr, maxc), None when the cell is dropped (counted with the
        border losses); "margin_mode", "recenter": the cell's resolved values; "bbox_inset_px": the
        directional cuts applied; "auto_tighten_iterations"; "baseline_box": the plain mask_bbox square,
        None outside mask_bbox}."""
        # Task 038 follow-up (part B redesign): a per-cell override may ALSO pick
        # THIS cell's own margin_mode/recenter, independent of the image-level
        # defaults (e.g. the SHAPE benchmark arm now starts every cell at the SAME
        # naive margin_mode="fraction"/recenter=False as the MaskCrop control, and
        # the agent's own per-crop geometry reasoning moves ONE bad cell to
        # tight+recenter without perturbing the rest of the image).
        cell_margin_mode = cell_override.get("margin_mode", margin_mode)
        cell_recenter = cell_override.get("recenter", recenter)
        # `min_containment` acts within [the retention floor, 1] (2026-10-01): no crop keeps
        # less than `_TARGET_RETENTION_FLOOR` of its target; a lower value acts as the floor.
        cell_min_containment = max(_TARGET_RETENTION_FLOOR,
                                   min(1.0, float(cell_override.get("min_containment", 1.0))))
        # Task 038 v12 (human-locked): a per-cell DIRECTIONAL inset -- only meaningful
        # in `margin_mode="mask_bbox"` (applied below, in that branch only); default
        # all-0 (a no-op) for any cell without one.
        cell_bbox_inset = cell_override.get("bbox_inset_px") or {}
        # The ACTUALLY-applied inset, reported into `crop_metadata` below -- starts
        # as the REQUESTED inset, but may be SCALED BACK (never dropped) if it would
        # breach the retention floor (see the containment-hazard backstop below) --
        # tracked separately so the metadata never overstates what really happened.
        applied_bbox_inset = dict(cell_bbox_inset)
        # Task 038 architectural FIX round (coordinator-directed): `auto_tighten`
        # (only meaningful in `margin_mode="mask_bbox"`, below) -- the tool now
        # ITERATES the directional inset lever internally, to convergence, rather
        # than applying one externally-supplied `bbox_inset_px` step per call.
        # `False`/absent (e.g. an A/B-veto revert, `{"auto_tighten": False}`) falls
        # through to the EXISTING `cell_bbox_inset`-driven single-shot path
        # unchanged -- so a caller that still supplies `bbox_inset_px` directly
        # (not via `auto_tighten`) keeps its own, independently-tested behavior.
        auto_tighten = bool(cell_override.get("auto_tighten"))
        auto_tighten_iterations = 0

        # Get bounding box from the mask
        minr, minc, maxr, maxc = region.bbox
        height = maxr - minr
        width = maxc - minc
        img_height, img_width = image_shape[:2]
        plan = {"box": None, "margin_mode": cell_margin_mode, "recenter": cell_recenter, "bbox_inset_px": {},
                "auto_tighten_iterations": 0, "baseline_box": None}

        if cell_margin_mode == "mask_bbox":
            # Human rule (2026-09-30, REPLACES the task 038 v9 rectangle -- the bbox
            # plus `tight_buffer_px`, not padded to a square): every crop is square,
            # because the analyzer resizes every crop to 224 x 224 without keeping the
            # aspect ratio, so a rectangle stretched its cell. `mask_bbox` is now the
            # SQUARE around the TARGET mask's own bounding box: side = the bbox's
            # longer side + 2 x `tight_buffer_px`, centered on the bbox midpoint, RAW
            # pixels (no mask-out, no padding), placed like every crop
            # (`compute_square_crop_containing_bbox`: near the image border the
            # same-size square shifts inside the image; a side above the image's
            # shorter side becomes the largest square that fits). `margin`/
            # `margin_frac`/`recenter`/`crop_size`/`min_containment` are all IGNORED
            # in this mode. Along the bbox's LONGER side nothing outside the bbox
            # (+buffer) is included; along its shorter side the square extends past
            # the bbox and can hold neighbour pixels there (the accepted cost of the
            # square, human decision) -- this mode's own feature is the directional,
            # square-preserving tightening below (`bbox_inset_px` / `auto_tighten`).
            # Contains the bbox whenever the bbox's longer side fits the image's
            # shorter side (the containment check below is then a no-op for the plain
            # square); otherwise the square truncates the target and the
            # retention-floor backstop below decides keep or drop.
            buf = max(0, int(tight_buffer_px))
            buf_minr, buf_minc, buf_maxr, buf_maxc = compute_square_crop_containing_bbox(
                minr, minc, maxr, maxc, (minr + maxr) / 2.0, (minc + maxc) / 2.0,
                (max(height, width) + 2 * buf) / 2.0, img_height, img_width)
            new_minr, new_maxr, new_minc, new_maxc = buf_minr, buf_maxr, buf_minc, buf_maxc

            # Task 038 v12 (human-locked -- DELIBERATELY REVERSES the "geometry is
            # 100% pure-data, no VLM" lock, CARD.md SS1/SS5's own updated rationale):
            # a per-cell DIRECTIONAL `bbox_inset_px` override, applied AFTER the
            # buffer above -- clips the box on the neighbour side(s) ONLY (not
            # symmetric), so the VLRR loop can keep tightening a still-contaminated
            # mask_bbox crop across MULTIPLE rounds instead of converging after one
            # shot (`verifiers/crop.py::CropVerifier.adjudicate`'s own round-2+
            # directional-tighten proposal). This is the ONLY new sizing knob this
            # round adds. Reported into `crop_metadata` below (beside `margin_mode`)
            # so the trace/QC can see exactly what was applied. `buf_minr`/etc (the
            # box BEFORE any inset -- i.e. plain mask_bbox) are kept around as the
            # SAFE fallback baseline for the retention-floor backstop below, in case
            # the verifier's own projection turns out to have been wrong. Human rule
            # (2026-09-30): every cut keeps the square -- `_clamp_directional_inset`
            # pairs a cut of k px on one side with a k px cut on the perpendicular
            # axis, split evenly between its two sides (the target stays centred on
            # that axis), and reports the paired cuts too.
            #
            # Task 038 architectural FIX round (coordinator-directed): `auto_tighten`
            # ITERATES this SAME directional lever internally, to convergence, in one
            # call (`_auto_tighten_mask_bbox`, above `_compute_crop_qc`) -- REPLACES
            # the single externally-supplied `cell_bbox_inset` step below for a caller
            # that sets it; a caller that does not (e.g. an A/B-veto revert, which sets
            # `auto_tighten: False` to go back to the plain baseline) keeps the
            # EXISTING single-shot `bbox_inset_px` path unchanged.
            if auto_tighten:
                (new_minr, new_maxr, new_minc, new_maxc, applied_bbox_inset,
                 auto_tighten_iterations) = self._auto_tighten_mask_bbox(
                    labeled_mask, region, region.label, cell_mask,
                    buf_minr, buf_maxr, buf_minc, buf_maxc, min_area, full_field_areas=full_field_areas)
            elif any(cell_bbox_inset.get(k) for k in ("top", "bottom", "left", "right")):
                new_minr, new_maxr, new_minc, new_maxc, applied_bbox_inset = _clamp_directional_inset(
                    buf_minr, buf_maxr, buf_minc, buf_maxc, cell_bbox_inset)
        else:
            # Determine half_side to ensure entire mask is included with margin on all sides.
            # Dynamic margin: scale the surrounding context with the cell's size so the
            # phase-contrast halo around the cell is captured -- a fixed 1px margin gives
            # a bare bbox with no context and hurts the morphology embedding. `margin`
            # acts as a floor (a minimum pixel pad).
            # margin scales with THIS cell's mask area (region.area), so each cell id
            # gets a customized, size-proportional context pad.
            eff_margin = effective_margin(margin, region.area, margin_frac, cell_margin_mode, tight_buffer_px)

            if cell_recenter:
                # Task 038 (quality axis 3, CENTERED): center the crop window on the
                # mask's TRUE centroid (mass center, region.centroid) rather than the
                # bounding-box midpoint -- an elongated/asymmetric cell's bbox midpoint
                # can sit well away from its actual center of mass.
                #
                # Task 038 follow-up (Part A, human-approved OPTION (a), a clean
                # deterministic ENGINE correction, NOT a candidate search): half_side is
                # now sized from the TIGHT bbox, not the multi-axis distance-to-CENTROID
                # (`max_dist`, the old formula) -- that formula OVER-SIZED the crop for
                # any cell whose centroid sits off-center within its own bbox (real-run
                # evidence, v4: 162/323 crops with target_dominance < 0.5, e.g. bbox
                # 6x10 -> a 16x16 crop, dominance 0.275). `bbox_center_row/col` is the
                # bbox's OWN discrete pixel-index midpoint -- exactly what
                # `region.centroid` itself would equal if this cell WERE perfectly
                # symmetric about its bbox (same pixel-index-mean convention
                # `regionprops.centroid` uses, where `maxr`/`maxc` are one-PAST the last
                # pixel). `tight_dist` (the max distance from THAT point to the bbox
                # edges) is therefore the MINIMUM possible value `max_dist` could ever
                # take for this exact bbox -- achieved exactly when the region is
                # symmetric, and STRICTLY exceeded by exactly the centroid's own true
                # offset from `bbox_center_row/col` otherwise. So `half_side` below is
                # BYTE-IDENTICAL to the OLD max_dist-based formula for a symmetric cell
                # (see test_generate_cell_crops_symmetric_cell_byte_identical_sizing),
                # and tighter for the vast majority of asymmetric cells (by the
                # asymmetry-driven excess the old formula paid for). NOTE: for a small
                # fraction (~0.6%, from discrete-midpoint CEIL rounding) NEW can be 1px
                # LARGER than OLD -- always a SAFE direction (never under-sizes, never
                # truncates; verified across ~38k regionprops cells). half_side is CEILed
                # so a fractional-pixel distance never rounds
                # the window a fraction of a pixel too small (containment).
                center_row, center_col = region.centroid
                bbox_center_row = minr + (height - 1) / 2.0
                bbox_center_col = minc + (width - 1) / 2.0
                tight_dist = max(bbox_center_row - minr, maxr - bbox_center_row,
                                 bbox_center_col - minc, maxc - bbox_center_col, 0.0)
                half_side = int(np.ceil(tight_dist + eff_margin))
            else:
                # Legacy bbox-midpoint centering (task 042 and earlier; recenter=False).
                # half_side = ceil(longer side / 2) + margin, so the square holds the bbox
                # for an odd longer side too (2026-10-01: `longer // 2` left it one pixel
                # short, and at margin 0 the cell was dropped).
                center_row = (minr + maxr) // 2
                center_col = (minc + maxc) // 2
                half_side = int(np.ceil(max(height, width) / 2.0)) + int(eff_margin)

            # Optional VLRR knob: cap the crop to a fixed square (used by the crop
            # verifier to shrink an over-wide box that admits neighbor cells). When
            # crop_size is None (default) the dynamic bbox+margin sizing is unchanged.
            if crop_size:
                try:
                    half_side = min(half_side, max(8, int(crop_size) // 2))
                except (TypeError, ValueError):
                    pass

            # Task 038 follow-up (part B, result-driven per-cell tighten): THIS cell has
            # an agent-approved tighten override (`cell_overrides`, resolved above) --
            # trade some containment for higher target_dominance, PER-CELL, without
            # perturbing any other cell in this image (unlike the GLOBAL `crop_size` cap
            # above). The target half-extent is ONE deterministic geometric formula, not
            # a candidate search: for a roughly compact/blob-shaped target, pixel-area
            # retention scales approximately with the SQUARE of the crop's own side
            # length, so `sqrt(cell_min_containment)` is the natural side-length scale
            # that targets `cell_min_containment` area retention -- a REASONED, not
            # fit/grid-searched, approximation (this codebase's own established
            # calibration style; see CARD.md SS3/SS5). It is only ever an APPROXIMATION
            # for an irregular mask -- the ACTUAL resulting containment is measured for
            # real, from the true per-cell mask, at the containment check below; an
            # overshoot below the floor there drops the cell, an undershoot (still
            # comfortably above the floor) simply keeps more containment than strictly
            # asked for -- neither is a correctness problem. `min(half_side, ...)` -- this
            # can only shrink further from wherever `half_side` already stands, never
            # widen it (the same TIGHTEN-ONLY direction as the `crop_size` cap above).
            if cell_min_containment < 1.0:
                tightened_half_side = int(np.ceil((cell_min_containment ** 0.5) * half_side))
                half_side = min(half_side, max(8, tightened_half_side))

            # Define new bounding box (with margin on all sides)
            # Reference: 251031_CO_inference_Mydgf.py - ensures entire mask is included with margin
            # Human rule (2026-09-30): every crop is square, at the image border too --
            # `compute_square_crop_containing_bbox` places the `2*half_side` square for
            # BOTH centerings (the centroid with recenter, the bbox midpoint without;
            # before, recenter=False used `compute_crop_box`, whose per-axis border
            # clip cut a near-edge crop to a rectangle, and recenter=True fell back to
            # it when no square fit). For a comfortably interior cell the box equals
            # `compute_crop_box`'s own square (same `2*half_side` side, same truncated
            # position -- see test_generate_cell_crops_symmetric_cell_byte_identical_
            # sizing); near the image border the SAME-SIZE square shifts inside the
            # image as close to the center as full bbox containment allows (the target
            # off-center instead of the crop clipped); a side above the image's shorter
            # side becomes the largest square that fits. The containment check below
            # decides keep or drop, unchanged.
            new_minr, new_minc, new_maxr, new_maxc = compute_square_crop_containing_bbox(
                minr, minc, maxr, maxc, center_row, center_col, half_side, img_height, img_width
            )

        # Human rule (2026-09-30): every crop the tool writes is SQUARE -- the analyzer
        # resizes every crop to 224 x 224 without keeping the aspect ratio, so a
        # rectangle would stretch its cell (the task 038 architectural FIX round had
        # accepted non-square crops in every mode). Safety net, unreachable by the code
        # above: every box is square by construction -- `compute_square_crop_containing_
        # bbox` returns `(r, c, r + s, c + s)`, and `_clamp_directional_inset` (the
        # single-shot inset, every `_auto_tighten_mask_bbox` box, and the retention
        # scale-back below, all cuts of the square `buf_*` box) cuts both axes of a square
        # by the same total. It exists so a future change can never write a rectangle: a
        # non-square box is replaced by the square placed like every crop around the
        # TARGET's bbox (side = the box's longer side, at most the image's shorter side),
        # and the containment check below then judges that square by the bbox like any
        # crop -- keep, partial keep or drop, counted. A mask_bbox cell's `bbox_inset_px`
        # is re-derived from the square (the cuts it makes into the plain mask_bbox
        # square), so the metadata describes the box written.
        box_h, box_w = new_maxr - new_minr, new_maxc - new_minc
        if box_h != box_w:
            new_minr, new_minc, new_maxr, new_maxc = compute_square_crop_containing_bbox(
                minr, minc, maxr, maxc, (new_minr + new_maxr) / 2.0, (new_minc + new_maxc) / 2.0,
                max(box_h, box_w) / 2.0, img_height, img_width)
            if cell_margin_mode == "mask_bbox":
                applied_bbox_inset = {k: v for k, v in (
                    ("top", new_minr - buf_minr), ("bottom", buf_maxr - new_maxr),
                    ("left", new_minc - buf_minc), ("right", buf_maxc - new_maxc)) if v > 0}
            print(f"Warning: Cell {idx}: crop box {box_h}x{box_w} was not square; replaced by the "
                  f"{new_maxr - new_minr}x{new_maxc - new_minc} square at ({new_minr}, {new_minc}) "
                  f"around its bbox")

        # Verify that the entire mask is included in the crop
        # At borders, margin may be reduced, but mask must be fully contained
        # Crop boundaries must fully contain mask boundaries:
        # - new_minr <= minr (crop top should be above/equal to mask top)
        # - new_maxr >= maxr (crop bottom should be below/equal to mask bottom)
        # - new_minc <= minc (crop left should be left of/equal to mask left)
        # - new_maxc >= maxc (crop right should be right of/equal to mask right)
        fully_contains_bbox = not (new_minr > minr or new_maxr < maxr
                                   or new_minc > minc or new_maxc < maxc)
        if not fully_contains_bbox:
            if cell_margin_mode == "mask_bbox":
                # Task 038 v12 (human-locked): a directional `bbox_inset_px`
                # DELIBERATELY clips the target's own bbox on the neighbour side --
                # this is the INTENDED effect (§2.5's own containment hazard), not a
                # truncation error, so `mask_bbox` mode is EXEMPT from the
                # unconditional drop below. The verifier
                # (`verifiers/crop.py::CropVerifier.adjudicate`) only ever proposes an
                # inset step that keeps `target_retention` >=
                # `_TARGET_RETENTION_FLOOR` (it projects retention BEFORE proposing),
                # so this measured check is a defensive BACKSTOP, not the primary
                # gate -- it should essentially never fire in practice.

                def _measure_retention(r0, r1, c0, c1):
                    px = int(np.count_nonzero(labeled_mask[r0:r1, c0:c1] == region.label))
                    return min(1.0, px / max(float(region.area), 1e-9))

                actual_retention = _measure_retention(new_minr, new_maxr, new_minc, new_maxc)
                if actual_retention < _TARGET_RETENTION_FLOOR:
                    # NEVER discard, per the human ("rather than just throwing it away"): the verifier's
                    # own retention projection is a REASONED APPROXIMATION and can
                    # be wrong (it does not see the real mask geometry) -- if the
                    # FULL requested inset genuinely breaches the hard floor, SCALE
                    # IT BACK instead, trying progressively smaller fractions of the
                    # SAME requested inset (re-clamped from the SAME `buf_minr`/etc
                    # baseline each time, never compounding rounding across
                    # attempts) until measured retention clears the floor again. Every
                    # scaled candidate stays square (`_clamp_directional_inset` pairs
                    # its cuts, human rule 2026-09-30). Scale 0.0 (no inset at all --
                    # the plain mask_bbox square) keeps the whole target whenever the
                    # bbox's longer side fits the image's shorter side, so this loop
                    # then always finds a KEEPABLE crop.
                    for scale in (0.75, 0.5, 0.25, 0.0):
                        scaled_inset = {k: int(round(float(v) * scale))
                                       for k, v in cell_bbox_inset.items()}
                        (cand_minr, cand_maxr, cand_minc, cand_maxc,
                         cand_applied) = _clamp_directional_inset(
                            buf_minr, buf_maxr, buf_minc, buf_maxc, scaled_inset)
                        cand_retention = _measure_retention(cand_minr, cand_maxr, cand_minc, cand_maxc)
                        if cand_retention >= _TARGET_RETENTION_FLOOR:
                            print(f"Warning: Cell {idx} bbox_inset_px scaled back to "
                                  f"{int(scale * 100)}% ({actual_retention:.3f} -> "
                                  f"{cand_retention:.3f} retention) to stay >= floor "
                                  f"{_TARGET_RETENTION_FLOOR} -- crop KEPT, not dropped.")
                            new_minr, new_maxr, new_minc, new_maxc = (
                                cand_minr, cand_maxr, cand_minc, cand_maxc)
                            actual_retention = cand_retention
                            applied_bbox_inset = cand_applied
                            break
                    else:
                        # Not even the plain mask_bbox square keeps the floor: the
                        # bbox's longer side exceeds the image's shorter side, so no
                        # square containing the target fits the image and the largest
                        # one that fits truncates it below the retention floor. The
                        # retention floor decides keep or drop (human rule 2026-09-30:
                        # squares only, no exemption at the image border) -- dropped,
                        # counted with the border losses.
                        print(f"Warning: Cell {idx}: the largest square that fits the "
                              f"{img_height}x{img_width} image keeps {cand_retention:.3f} of "
                              f"its target (bbox {height}x{width}), below the retention floor "
                              f"{_TARGET_RETENTION_FLOOR}; skipping")
                        return plan
                # else (already >= floor, or scaled back above): accept the
                # (possibly scaled-back) tightened crop -- fall through below. A
                # requested inset never drops the crop (task 038 v12).
            elif cell_min_containment < 1.0:
                # Task 038 follow-up (part B, result-driven per-cell tighten): THIS
                # cell has an agent-approved tighten override -- partial truncation
                # is EXPECTED here, not an error. Measure the ACTUAL containment
                # FRACTION (not just the rectangle-coverage boolean above) straight
                # from the real per-cell mask, and only drop the cell if it falls
                # below the override's OWN floor (never a silent, unbounded
                # truncation) -- the same `mask_containment` formula
                # `_compute_crop_qc` uses, computed here (before the mask has been
                # cropped/saved yet) so the drop decision costs no wasted file I/O.
                target_px_in_crop = int(np.count_nonzero(
                    labeled_mask[new_minr:new_maxr, new_minc:new_maxc] == region.label))
                actual_containment = min(1.0, target_px_in_crop / max(float(region.area), 1e-9))
                if actual_containment < cell_min_containment:
                    print(f"Warning: Cell {idx} tighten override dropped containment to "
                          f"{actual_containment:.3f} < floor {cell_min_containment}; skipping")
                    return plan
                # else: accept the partial-containment crop -- fall through below.
            else:
                # Crop doesn't fully contain the mask - this should not happen with proper margin calculation
                # But if it does (e.g., due to image border constraints), skip this cell
                print(f"Warning: Cell {idx} at ({center_row}, {center_col}) filtered by border constraints: "
                      f"mask bbox=({minr}, {minc}, {maxr}, {maxc}), crop bbox=({new_minr}, {new_minc}, {new_maxr}, {new_maxc})")
                return plan

        plan.update(box=(new_minr, new_minc, new_maxr, new_maxc), bbox_inset_px=applied_bbox_inset,
                    auto_tighten_iterations=auto_tighten_iterations,
                    baseline_box=((buf_minr, buf_minc, buf_maxr, buf_maxc) if cell_margin_mode == "mask_bbox"
                                  else None))
        return plan

    @staticmethod
    def _show_target(cell_mask, region, value):
        """Sets (`value` 255) or clears (0) `region`'s pixels in `cell_mask`, the full-field target mask the
        cells of a call share, within the region's bbox: O(bbox), not a full-image pass per cell."""
        minr, minc, maxr, maxc = region.bbox
        cell_mask[minr:maxr, minc:maxc][region.image] = value

    def _auto_tighten_mask_bbox(self, labeled_mask, region, region_label, cell_mask,
                                buf_minr, buf_maxr, buf_minc, buf_maxc, min_area, full_field_areas=None):
        """Task 038 architectural FIX round (coordinator-directed): iterates the
        directional `bbox_inset_px` lever for ONE `margin_mode="mask_bbox"` cell TO
        CONVERGENCE, inside this single `execute()` call, instead of relying on the
        VLRR loop's own per-round, whole-image re-render to apply one cut at a time.
        Each internal step is a cheap, mask-only `_compute_crop_qc` recompute (array
        indexing over the ALREADY-loaded `labeled_mask`/`cell_mask`, no image I/O, no
        re-render of any other cell in the image) -- the SAME computation
        `verifiers/crop.py::CropVerifier`'s own round-2+ pass used to drive one cut per
        real tool call now runs directly here, to convergence, in one pass. This is
        WHY `tool_calls_per_image` collapses from a naive ~4.67 to a genuinely lower
        figure (mask_bbox+auto_tighten [-> an optional A/B-veto revert]): the round
        count used to be driven by the SLOWEST-converging cell in the image,
        re-rendering EVERY other cell each round too; now each cell converges
        independently, in-process. Task 038 QC N2 (an independent QC of the geometry
        release candidate; the exact "~2" this comment used to cite here was STALE --
        the CURRENT measured value is `tool_calls_per_image` ~2.667, not ~2.0 -- see
        CODING_LOG.md's own QC N2 section for the full current numbers + provenance;
        the DIRECTION and MECHANISM this docstring describes are still correct, only
        the specific old headline figure was wrong).

        Carries FIX #2's own plateau/regression semantics
        (`CropVerifier._tighten_delta`) into this internal loop, UNCHANGED in spirit --
        a candidate cut that reduces `neighbor_contamination` by more than
        `_TIGHTEN_PLATEAU_EPS` (0.02) is committed and the loop continues; one that
        helps by `_TIGHTEN_PLATEAU_EPS` or less is committed (not worse, just not worth
        chasing further) and the loop stops; one that would make the cell WORSE is
        simply never committed -- there is nothing to "revert" here (no file has been
        written yet, this is in-memory box/array-index exploration only), so the loop
        just stops at the prior, already-committed state instead. Also stops on
        `neighbor_contamination <= _MASK_BBOX_CONTAMINATION_MIN` (data-clean) or
        `contamination_inset_px == 0` (no cutting headroom left, per `_compute_crop_qc`'s
        own floor-capped depth) -- matching the verifier-level design's own three stop
        conditions (clean / no-headroom / plateau-or-regression), now evaluated locally.

        A retention-floor check (`mask_containment < _TARGET_RETENTION_FLOOR`) mirrors the
        plain single-shot `mask_bbox` path's own never-truncate-past-floor check, below
        -- task 038 FIX round finding: this DOES fire in practice (a real headless run
        measured it dragging `neighbor_contamination_reduction` down, -0.0141) --
        `contamination_inset_px`'s own single-step projection is a reasoned
        APPROXIMATION and can occasionally overshoot, the SAME reason the plain
        single-shot path has its own scale-back ladder. Never simply abandons the whole
        step on a breach: binary-searches the largest additional depth on that side that
        stays at/above the floor (re-measured directly at each trial) and commits that
        instead -- see the loop body's own comment for the full reasoning.

        `_AUTO_TIGHTEN_SAFETY_CAP` bounds the iteration defensively (not a tuning
        knob) -- in practice the loop converges via one of the three stop conditions
        above long before this would ever bind.

        Square at every step (human rule, 2026-09-30: every crop the tool writes is
        square): every box this loop evaluates -- each candidate, each binary-search trial
        -- and the box it returns comes from `_clamp_directional_inset`, which pairs the
        cut of `depth` px on `contamination_side` with a `depth` px cut on the
        perpendicular axis, split evenly between its two sides (floor to top/left, ceil to
        bottom/right, so the target stays centred on that axis), and keeps both axes within
        the same `min_side` headroom, so the `mask_bbox` square stays square. The
        contaminated side and its depth are chosen as before. The pairing is part of each
        committed step: the stop rules, the retention floor and their constants are
        unchanged, and they judge the paired (square) box -- a perpendicular cut that clips
        the target is caught by the same floor check and binary search. The returned
        `applied_bbox_inset` holds the paired cuts too.

        `cell_mask` is the target's full-field mask (None: built here from `labeled_mask`); `full_field_areas`
        the label areas, passed to every `_compute_crop_qc` (None: counted there).
        Returns `(final_minr, final_maxr, final_minc, final_maxc, applied_bbox_inset,
        n_iterations)`."""
        if cell_mask is None:
            cell_mask = np.zeros_like(labeled_mask)
            cell_mask[labeled_mask == region_label] = 255
        cur_minr, cur_maxr, cur_minc, cur_maxc = buf_minr, buf_maxr, buf_minc, buf_maxc
        applied_inset: dict = {}
        cell_mask_crop = cell_mask[cur_minr:cur_maxr, cur_minc:cur_maxc]
        # NOTE: `_compute_crop_qc`'s own positional convention is (minr, minc, maxr,
        # maxc) -- DIFFERENT from `_clamp_directional_inset`'s own (minr, maxr, minc,
        # maxc) used below; this method's own local vars follow the LATTER (matching
        # `_generate_cell_crops`'s own `new_minr, new_maxr, new_minc, new_maxc` local
        # convention), so every `_compute_crop_qc` call here reorders explicitly.
        qc = self._compute_crop_qc(region, region_label, labeled_mask, cell_mask_crop,
                                   cur_minr, cur_minc, cur_maxr, cur_maxc, min_area,
                                   full_field_areas=full_field_areas)
        contamination = qc["neighbor_contamination"]
        n_iterations = 0

        while (contamination > _MASK_BBOX_CONTAMINATION_MIN
              and qc["contamination_inset_px"] > 0 and qc["contamination_side"]
              and n_iterations < _AUTO_TIGHTEN_SAFETY_CAP):
            side = qc["contamination_side"]
            depth = qc["contamination_inset_px"]
            request = dict(applied_inset)
            request[side] = int(request.get(side, 0) or 0) + depth
            cand_minr, cand_maxr, cand_minc, cand_maxc, candidate_inset = _clamp_directional_inset(
                buf_minr, buf_maxr, buf_minc, buf_maxc, request)
            cand_mask_crop = cell_mask[cand_minr:cand_maxr, cand_minc:cand_maxc]
            cand_qc = self._compute_crop_qc(region, region_label, labeled_mask, cand_mask_crop,
                                            cand_minr, cand_minc, cand_maxr, cand_maxc, min_area,
                                            full_field_areas=full_field_areas)
            if cand_qc["mask_containment"] < _TARGET_RETENTION_FLOOR:
                # FIX round (coordinator-directed, root cause CONFIRMED structurally --
                # see the coding log for the empirical investigation): `depth`'s own
                # projection (`_compute_crop_qc`'s own floor-capped budget) is a
                # reasoned APPROXIMATION and can occasionally overshoot -- the SAME
                # reason the plain single-shot mask_bbox path already SCALES BACK a
                # verifier-requested inset instead of abandoning it outright (its own
                # 0.75/0.5/0.25/0.0 ladder, below in `_generate_cell_crops`). The OLD
                # per-round verifier design tried PARTIAL fractions of the FULL
                # accumulated inset on exactly this floor breach; simply `break`-ing
                # here abandons the ENTIRE latest increment without ever trying a
                # smaller one -- structurally MORE conservative than the old design,
                # never exploring any depth between 0 and the full requested `depth`
                # (confirmed as the measured -0.0141 reduction regression's own root
                # cause). Binary-search the LARGEST additional depth on this SAME side
                # (0..depth) that keeps `mask_containment >= _TARGET_RETENTION_FLOOR`,
                # RE-MEASURED directly at each trial (never just trusting the
                # projection) -- commit THAT instead of abandoning the whole step. Each
                # trial is the paired, square box of that depth (`_clamp_directional_
                # inset`); the search keeps the deepest trial it MEASURED at/above the
                # floor -- the floor holds for whatever it commits.
                lo, hi = 0, depth
                best_depth, best = 0, None
                while lo <= hi:
                    mid = (lo + hi) // 2
                    trial_request = dict(applied_inset)
                    trial_request[side] = int(trial_request.get(side, 0) or 0) + mid
                    t_minr, t_maxr, t_minc, t_maxc, t_inset = _clamp_directional_inset(
                        buf_minr, buf_maxr, buf_minc, buf_maxc, trial_request)
                    t_mask_crop = cell_mask[t_minr:t_maxr, t_minc:t_maxc]
                    t_qc = self._compute_crop_qc(region, region_label, labeled_mask, t_mask_crop,
                                                 t_minr, t_minc, t_maxr, t_maxc, min_area,
                                                 full_field_areas=full_field_areas)
                    if t_qc["mask_containment"] >= _TARGET_RETENTION_FLOOR:
                        best_depth = mid
                        best = (t_minr, t_maxr, t_minc, t_maxc, t_inset, t_qc)
                        lo = mid + 1
                    else:
                        hi = mid - 1
                if best_depth <= 0:
                    break   # not even a partial step clears the floor -- no headroom, stop
                cand_minr, cand_maxr, cand_minc, cand_maxc, candidate_inset, cand_qc = best
            delta = contamination - cand_qc["neighbor_contamination"]
            if delta < 0:
                break   # regression -- never commit, stop at the prior committed state
            cur_minr, cur_maxr, cur_minc, cur_maxc = cand_minr, cand_maxr, cand_minc, cand_maxc
            applied_inset = candidate_inset
            qc = cand_qc
            contamination = cand_qc["neighbor_contamination"]
            n_iterations += 1
            if delta <= _TIGHTEN_PLATEAU_EPS:
                break   # plateau -- this cut IS kept (already committed above), just stop here

        return cur_minr, cur_maxr, cur_minc, cur_maxc, applied_inset, n_iterations

    @staticmethod
    def _compute_crop_qc(region, region_label, labeled_mask, cell_mask_crop,
                         new_minr, new_minc, new_maxr, new_maxc, min_area=0, full_field_areas=None):
        """Deterministic per-crop QC for the VLRR crop verifier. `full_field_areas` =
        `np.bincount(labeled_mask.ravel())` when the caller already has it (None: computed
        here; `_generate_cell_crops` counts it once per call, the agentic loop's
        configuration key once per input).

        Returns mask_containment / occupancy / centroid_offset / border_touch /
        neighbor_contamination / target_dominance / single_object /
        foreground_fraction / n_complete_cells / contamination_side / target_retention /
        contamination_inset_px.
        Computed purely from the label mask and the target cell's mask within the CURRENT
        crop box (`new_minr:new_maxr, new_minc:new_maxc` -- i.e. AFTER any directional
        `bbox_inset_px` has already been applied to it, task 038 v12) -- no LLM, no image
        content.

        `contamination_side` / `target_retention` / `contamination_inset_px` (task 038
        v12, human-locked, PURE DATA -- the directional-tighten loop's own data channel,
        `verifiers/crop.py::CropVerifier.adjudicate`'s round-2+ pass reads these):
          - `contamination_side` in {"top","bottom","left","right",None}: which crop
            edge carries the MOST non-target foreground (`labeled_mask != 0 and !=
            region_label`) -- the side the directional lever should inset next.
            Computed from the non-target-fg pixel MASS relative to the TARGET's own
            centroid within the crop: two INDEPENDENT binary splits (rows above vs
            below the centroid row; columns left vs right of the centroid col), then
            the single most extreme of the resulting 4 counts wins (a pixel counts
            toward both its own row-half AND its own column-half -- these are not 4
            mutually-exclusive quadrants, "top" vs "left" are simply different
            questions about the SAME pixel). None when there is no non-target fg at
            all (nothing to tighten).
          - `target_retention`: literally the SAME value as `mask_containment` (target-
            mask px inside the CURRENT crop box / target-mask px in the FULL bbox) --
            reported under BOTH names deliberately: `mask_containment` keeps its
            EXISTING, general-purpose role (an absolute-quality gate `is_crop_valid`
            reads across ALL margin modes); `target_retention` is the v12 lever's own
            purpose-built name for the SAME number, read specifically by the
            directional-tighten loop's own hard floor
            (`CropVerifier._TARGET_RETENTION_FLOOR`, this module's own
            `_TARGET_RETENTION_FLOOR`) -- kept as a distinct key so a future change to
            either consumer's own semantics does not have to silently share one
            overloaded name.
          - `contamination_inset_px` (task 038 v12 R2, human-directed -- REPLACES an
            earlier FIXED-FRACTION step the verifier used to project on its own; human
            rejected that design: "the logic of this step does not work; then what is the point of
            the VLM?" -- the mask
            already knows exactly where the neighbour pixels are, so a blind fraction
            was arbitrary, and a fixed data-step made the VLM gate a rubber-stamp):
            the pixel DEPTH to inset `contamination_side` by so the non-target
            foreground reaching furthest toward the target ON THAT SIDE (the SAME
            centroid-relative half used to pick `contamination_side` above) is
            excluded -- scan inward from that edge, find how deep that side's own
            non-target fg reaches, and return the depth that clears it. CAPPED so
            `target_retention` never drops below `_TARGET_RETENTION_FLOOR`: if fully
            excluding the neighbour would remove more target px than the floor
            allows, only the LARGEST depth that still clears the floor is returned
            (best-effort PARTIAL removal, never a full discard). `0` when there is
            nothing to cut (`contamination_side is None`) OR when there is no cutting
            headroom left at all (this crop is already at/below the floor) -- the
            verifier reads either case identically as "data has nothing further to
            propose here, freeze."

        `single_object` (task 038 M-A upgrade): a target-intactness / fragmentation
        estimate on the TARGET's OWN mask (`cell_mask_crop`) -- 1/n_significant,
        "significant" = a connected component with area >= 40% of the largest one's area
        (mirrors `quant.py::single_object`'s own mask-branch formula, computed
        independently here since `_compute_crop_qc` already holds the authoritative
        per-cell mask and needs no full-field context for this one value -- same "reused
        formula, independent implementation" relationship `quant.py`'s
        `_border_touch_from_mask` documents for `border_touch` above). NOT a doublet
        detector: a MERGED doublet is already one connected component in `cell_mask_crop`
        (upstream segmentation assigned it one label) and scores 1.0 -- see
        `quant.single_object`'s docstring for why, and CARD.md's COMPLEMENTARITY/
        RELIABILITY section for what DOES lead on doublet (vision +
        `n_complete_cells`, below). This REPLACES the weaker pixel-Otsu estimate
        `crop_triage._quant_evidence` previously fell back to for mask-path crops.

        `foreground_fraction`/`n_complete_cells` (task 038 follow-up, VLM-hallucination-
        fix round -- the v6 the cluster finding: pure-VLM triage hallucinated `empty_debris`/
        `doublet` unsupported by the mask; `target_dominance` was wrongly read as a reject/
        tighten driver even though it is NOT a validity gate, see this module's own
        module-level comment). These are the DATA gates `crop_triage.py`'s hard veto reads
        to CORROBORATE (never itself drive) a VLM's reject/tighten proposal:
          - `foreground_fraction` = ALL cell-mask foreground pixels in the crop / crop
            area (not just the target's own) -- near 0 only when the crop is genuinely
            empty/debris (no real cell mask at all), the ONLY thing "empty_debris" can
            honestly mean; a crop with a real target (however small-dominance amid
            tolerated neighbours) is NOT near 0.
          - `n_complete_cells` = the count of DISTINCT labels (including the target)
            present in this crop whose FULL-FIELD (whole-image) mask is >=90% CONTAINED
            inside this crop box AND whose full-field area is >= `min_area` -- a
            "complete cell" this crop genuinely captures WHOLE, not a partial sliver
            clipped at the crop's own edge. >=2 corroborates a genuine doublet (two whole
            cells actually inside this crop box, not a tolerated partial-edge neighbour).
        """
        eps = 1e-9
        crop_h = max(int(new_maxr - new_minr), 1)
        crop_w = max(int(new_maxc - new_minc), 1)
        crop_area = crop_h * crop_w

        target_in_crop = int(np.count_nonzero(cell_mask_crop))
        total_target = float(region.area) if region.area else eps
        mask_containment = min(1.0, target_in_crop / total_target)
        occupancy = target_in_crop / crop_area

        # centroid offset from crop center, normalized by half the smaller crop side
        crop_cr = (new_minr + new_maxr) / 2.0
        crop_cc = (new_minc + new_maxc) / 2.0
        cy, cx = region.centroid
        offset_px = float(np.hypot(cy - crop_cr, cx - crop_cc))
        centroid_offset = offset_px / (0.5 * min(crop_h, crop_w) + eps)

        # fraction of the target mask touching the crop border
        if cell_mask_crop.size:
            edges = np.concatenate([
                cell_mask_crop[0, :].ravel(), cell_mask_crop[-1, :].ravel(),
                cell_mask_crop[:, 0].ravel(), cell_mask_crop[:, -1].ravel(),
            ])
            border_pixels = int(np.count_nonzero(edges))
        else:
            border_pixels = 0
        border_touch = min(1.0, border_pixels / total_target)

        # fraction of foreground in the crop box that belongs to OTHER cells, and
        # the complementary share that belongs to the TARGET (its dominance).
        crop_labels = labeled_mask[new_minr:new_maxr, new_minc:new_maxc]
        fg = int(np.count_nonzero(crop_labels))
        other = int(np.count_nonzero((crop_labels != 0) & (crop_labels != region_label)))
        neighbor_contamination = other / (fg + eps)
        target_dominance = (fg - other) / (fg + eps)
        foreground_fraction = fg / crop_area

        # contamination_side / target_retention (task 038 v12, human-locked, PURE DATA --
        # see the docstring above for the full rationale).
        target_retention = mask_containment   # literally the same number, reported under
                                              # a second, purpose-built name -- see docstring
        non_target_fg = (crop_labels != 0) & (crop_labels != region_label)
        if non_target_fg.any():
            rel_cy = cy - new_minr   # target centroid, in CROP-local coordinates
            rel_cx = cx - new_minc
            rows_idx, cols_idx = np.nonzero(non_target_fg)
            side_mass = {
                "top": int(np.count_nonzero(rows_idx < rel_cy)),
                "bottom": int(np.count_nonzero(rows_idx >= rel_cy)),
                "left": int(np.count_nonzero(cols_idx < rel_cx)),
                "right": int(np.count_nonzero(cols_idx >= rel_cx)),
            }
            contamination_side = max(side_mass, key=side_mass.get)
            if side_mass[contamination_side] == 0:
                contamination_side = None
        else:
            contamination_side = None

        # contamination_inset_px (task 038 v12 R2, human-directed -- see docstring above
        # for the full rationale). Reuses the SAME centroid-relative "side" split above
        # (rows_idx/cols_idx, rel_cy/rel_cx) to find how deep, from `contamination_side`'s
        # own edge, that side's non-target fg reaches toward the target, then caps the
        # depth so removing it never drops `target_retention` below the floor.
        if contamination_side is None:
            contamination_inset_px = 0
        else:
            target_mask_bool = cell_mask_crop.astype(bool)
            if contamination_side == "top":
                side_idx = rows_idx[rows_idx < rel_cy]
                full_depth = int(side_idx.max()) + 1 if side_idx.size else 0
                depth_counts = target_mask_bool.sum(axis=1)            # row 0 (top) first
            elif contamination_side == "bottom":
                side_idx = rows_idx[rows_idx >= rel_cy]
                full_depth = crop_h - int(side_idx.min()) if side_idx.size else 0
                depth_counts = target_mask_bool.sum(axis=1)[::-1]      # bottom row first
            elif contamination_side == "left":
                side_idx = cols_idx[cols_idx < rel_cx]
                full_depth = int(side_idx.max()) + 1 if side_idx.size else 0
                depth_counts = target_mask_bool.sum(axis=0)            # col 0 (left) first
            else:  # "right"
                side_idx = cols_idx[cols_idx >= rel_cx]
                full_depth = crop_w - int(side_idx.min()) if side_idx.size else 0
                depth_counts = target_mask_bool.sum(axis=0)[::-1]      # right col first

            if full_depth <= 0:
                contamination_inset_px = 0
            else:
                budget = target_in_crop - _TARGET_RETENTION_FLOOR * total_target
                if budget <= 0:
                    contamination_inset_px = 0   # already at/below the floor -- no headroom
                else:
                    cum_removed = np.cumsum(depth_counts)   # cum_removed[d-1] = target px
                                                             # removed by depth d
                    if int(cum_removed[full_depth - 1]) <= budget:
                        contamination_inset_px = full_depth        # full exclusion clears the floor
                    else:
                        # largest depth d in [0, full_depth] with removed(d) <= budget
                        # (monotonic non-decreasing in d, so a binary search finds it).
                        lo, hi = 0, full_depth
                        while lo < hi:
                            mid = (lo + hi + 1) // 2
                            if int(cum_removed[mid - 1]) <= budget:
                                lo = mid
                            else:
                                hi = mid - 1
                        contamination_inset_px = lo

        # target intactness / fragmentation on the TARGET's OWN mask (see the docstring above)
        mask_regions = regionprops(label(cell_mask_crop > 0))
        mask_regions.sort(key=lambda r: r.area, reverse=True)
        if mask_regions:
            largest_area = mask_regions[0].area
            n_significant = sum(1 for r in mask_regions if r.area >= 0.4 * largest_area)
            single_object = 1.0 / max(1, n_significant)
        else:
            single_object = 1.0

        # n_complete_cells (see the docstring above): the full-field label areas, counted
        # ONCE via bincount (`full_field_areas`: once per tool call, by the caller), then
        # for each distinct label present in the crop (incl. the
        # target), check its full-field area meets min_area and its in-crop pixel count
        # is >=90% of that full-field area (i.e. this crop genuinely contains that cell
        # WHOLE, not a clipped sliver).
        if full_field_areas is None:
            full_field_areas = np.bincount(labeled_mask.ravel())
        n_complete_cells = 0
        for lbl in np.unique(crop_labels):
            if lbl == 0:
                continue
            full_area = int(full_field_areas[lbl]) if lbl < len(full_field_areas) else 0
            if full_area < min_area or full_area <= 0:
                continue
            in_crop_area = int(np.count_nonzero(crop_labels == lbl))
            if (in_crop_area / full_area) >= 0.9:
                n_complete_cells += 1

        return {
            "mask_containment": float(mask_containment),
            "occupancy": float(occupancy),
            "centroid_offset": float(centroid_offset),
            "border_touch": float(border_touch),
            "neighbor_contamination": float(neighbor_contamination),
            "target_dominance": float(target_dominance),
            "single_object": float(single_object),
            "foreground_fraction": float(foreground_fraction),
            "n_complete_cells": int(n_complete_cells),
            "contamination_side": contamination_side,
            "target_retention": float(target_retention),
            "contamination_inset_px": int(contamination_inset_px),
        }

    def _create_summary_visualization(self, original_img, mask, cell_crops, stats, output_dir, min_area, margin):
        """Creates a professional and highly robust summary visualization of the cropping process."""
        vis_config = VisualizationConfig()
        output_path = os.path.join(output_dir, "single_cell_cropper_summary.png")

        try:
            fig = plt.figure(figsize=(24, 16), dpi=300)
            gs = fig.add_gridspec(1, 2, width_ratios=[2, 1], hspace=0.25, wspace=0.15)
            
            fig.suptitle("Single-Cell Cropping Summary", fontsize=36, fontweight='bold', y=0.98)

            # --- Subplots ---
            ax1_container = fig.add_subplot(gs[0, 0])
            ax2 = fig.add_subplot(gs[0, 1])
            
            # 1. Sample Crops Grid (show 5 random crops)
            ax1_container.set_title("Sample Cell Crops (5 random)", fontsize=32, fontweight='bold', pad=20)
            ax1_container.axis('off')
            if cell_crops:
                num_samples = min(5, len(cell_crops))
                gs_crops = ax1_container.get_subplotspec().subgridspec(1, num_samples, wspace=0.05, hspace=0.05)
                sample_indices = np.random.choice(len(cell_crops), size=num_samples, replace=False)
                for i, idx in enumerate(sample_indices):
                    ax_grid = fig.add_subplot(gs_crops[i])
                    try:
                        crop_img = Image.open(cell_crops[idx])
                        ax_grid.imshow(crop_img, cmap='gray')
                    except (FileNotFoundError, IOError):
                        pass # Silently skip if file is missing or corrupt
                    ax_grid.axis('off')
            
            # 2. Statistics and Parameters Text - adjusted height to match cropper
            stats_text = (
                f"Initial Nuclei Detected: {stats.get('initial_cell_count', 'N/A')}\n"
                f"Filtered by Area (<{min_area}px): {stats.get('filtered_by_area', 'N/A')}\n"
                f"Filtered by Border: {stats.get('filtered_by_border', 'N/A')}\n"
                f"Final Valid Crops: {stats.get('final_cell_count', 'N/A')}\n\n"
                f"Parameters Used:\n"
                f"  - Min Area: {min_area} pixels\n"
                f"  - Margin: {margin} pixels"
            )
            ax2.set_title("Processing Statistics", fontsize=32, fontweight='bold', pad=20)
            ax2.text(
                0.5, 0.5, stats_text, ha='center', va='center', transform=ax2.transAxes,
                fontsize=28, fontweight='bold', wrap=True,
                bbox=dict(boxstyle='round,pad=1.0', facecolor='aliceblue', alpha=0.95)
            )
            ax2.axis('off')

            vis_config.save_professional_figure(fig, output_path, bbox_inches=None)
            plt.close(fig)

            print(f"OK: Summary visualization successfully created at {output_path}")
            return output_path
        except Exception as e:
            print(f"Error creating summary visualization: {e}")
            import traceback
            traceback.print_exc()
            return None

    def get_metadata(self):
        """Returns the metadata for the Single_Cell_Cropper_Tool."""
        return super().get_metadata()


if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd shape/tools/single_cell_cropper
    python tool.py
    """

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Single_Cell_Cropper_Tool
    tool = Single_Cell_Cropper_Tool()
    tool.set_custom_output_dir("single_cell_outputs")

    # Get tool metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(metadata)

    # Example execution (requires both original image and nuclei mask)
    try:
        # This would require actual image and mask files
        print("\nTool initialized successfully. Ready for execution.")
        print("Example usage:")
        print("execution = tool.execute(original_image='path/to/image.tif', nuclei_mask='path/to/mask.png')")
        print("Note: nuclei_mask parameter accepts any segmentation mask: nuclei_mask, cell_mask, or organoid_mask")
        
    except Exception as e:
        print(f"Error during tool initialization: {e}") 

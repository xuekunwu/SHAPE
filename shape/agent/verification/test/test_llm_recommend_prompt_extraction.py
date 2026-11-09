"""Task 032 (M3 scaffolding): `_build_prompt` now assembles its text from
`shape/agent/prompts/cell_segmenter/{verify,recommend}.md` instead of an inline
f-string. This test pins the extraction to byte-identical output vs the pre-
extraction golden string (captured from the ORIGINAL inline-f-string function
before this task edited it), so the "extraction" is provably behavior-preserving,
not a rewrite. Co-located with the module under test (committed -- AGENTS.md sec 6 P5).
"""

from shape.agent.verification.llm_recommend import _build_prompt

# Captured verbatim from the pre-extraction `_build_prompt` (the inline f-string
# version) for this exact (metrics, card) input -- see CODING_LOG.md sec M3.
_GOLDEN_WITH_CARD = (
    "You are the visual check of a cell-segmentation verifier. The attached image is "
    "an OVERLAY: each detected object is drawn as a colored mask on the microscopy "
    "image.\n\n"
    "STEP 1 -- recognize the image: what specimen is it, how big are the typical "
    "TARGET cells, how densely packed? Debris, stain artifacts and out-of-focus blobs "
    "are NOT target cells.\n\n"
    "STEP 2 -- find the ONE main problem, if any: target cells clearly MISSED (no "
    "mask); FALSE masks on non-cells; wrong SCALE (masks much bigger/smaller than the "
    "cells); crowded cells MERGED; or the overlay SHATTERED into far more fragments "
    "than there are real cells (over-segmentation). Do NOT nitpick fine mask SHAPE "
    "(the metrics already measure that); cells merely TOUCHING is normal.\n\n"
    "STEP 3 -- rate severity, defaulting to ACCEPT. Only a CLEARLY major problem (many "
    "cells affected) that a parameter can plausibly fix warrants a retune; tolerate "
    "minor imperfections, a few missed/false masks, and any merely good-enough result "
    "with severity='minor' or 'none' and param='none'. A retune that might not help is "
    "not worth it.\n\n"
    "STEP 4 -- if severity is 'major', use the TOOL CARD to choose the SINGLE "
    "parameter and direction whose documented effect best fixes the problem you "
    "named; otherwise param='none', direction='none'.\n"
    "\nTOOL CARD -- how this segmenter's parameters work; choose your fix from here:\n"
    "Some card text.\n"
    "\n"
    "Report: specimen; cell_size; cell_density; image_usable; problem (one short "
    "phrase or 'none'); severity; param; direction; confidence; and a one-sentence "
    "reasoning citing what you see in the overlay.\n\n"
    "Deterministic metrics -- evidence you MAY weigh (e.g. a cell_count wildly "
    "inconsistent with the cells you see corroborates over/under-segmentation); judge "
    'holistically from the overlay and do not nitpick fine mask shape: {"cell_count": 42.0, '
    '"tiny_mask_fraction": 0.12}'
)


def test_build_prompt_byte_identical_to_pre_extraction_golden_with_card():
    metrics = {"cell_count": 42.0, "tiny_mask_fraction": 0.12, "flag_bool": True}
    out = _build_prompt(metrics, "Some card text.")
    assert out == _GOLDEN_WITH_CARD


def test_build_prompt_no_card_section_omitted():
    """`card_section` is empty when no card is given -- the "{card_section}\n" slot in
    recommend.md collapses to a single blank line, exactly like the original f-string.
    (STEP 4's own text legitimately mentions "TOOL CARD" regardless -- only the
    card-section HEADING ("choose your fix from here") is conditional.)"""
    out = _build_prompt({"cell_count": 1.0}, None)
    assert "choose your fix from here" not in out
    assert out.startswith("You are the visual check")
    assert '"cell_count": 1.0' in out

"""Task 047: the batch/multi-image `solve(images=[...])` path records step verification
in a `per_image` shape -- `verification = {"tool":.., "per_image": [vrecord, ...]}` --
where each `vrecord` is the flat `{"status":.., "stages":[...]}` shape (task 045's coder
found this; see `test_verificate_memory_cpu_replay.py`'s investigation note). Before this
fix, both digest builders read `stages` at the TOP level only:
  - `Planner._format_qc` (shape/agent/planner.py, pre-fix line ~214:
    `for s in verification.get("stages", [])`)
  - `compact_verification` (shape/agent/memory.py, pre-fix line ~37:
    `for s in (v.get("stages") or [])`)
so the batch path's `vision_verdict` never reached the planner task-gate prompt (see
`solver.py::_execute_and_verify`'s "Multi-image" branch, ~line 571:
`verification={"tool": next_step.tool_name, "per_image": per_vrecords}`, where each
`per_vrecords[i]` is produced by `_managed_verification_record` -- the SAME shape as the
single-image flat record, confirmed against a real trace, see fixtures below).

Fixtures are copied (trimmed to the load-bearing fields: stage/decision/action/status/
fault + trial notes/metrics) from real 045-band `vlrr_trace.json` per_image records:
  - VRECORD_1: results/by_difficulty_ab/base/hard/shape/cell_00001/vlrr_trace.json,
    step 1 (Cell_Segmenter_Tool), `verification["per_image"][0]` -- a `conceded`/
    `retune_here` record with a real "[vision] ... sev=major" judge note.
  - VRECORD_2: results/by_difficulty_ab/base/medium/shape/cell_00026/vlrr_trace.json,
    step 0 (Cell_Segmenter_Tool), `verification["per_image"][0]` -- a `satisfied`/
    `accept` record whose FIRST trial carries a different "[vision] ... sev=major" note
    (the stage's final notes read "sev=none" -- exercising `_vision_verdict`'s "pull the
    first trial's judge note forward" behavior with a second, distinct real record).

Co-located with the modules under test (committed -- AGENTS.md sec 6 P5), not in the
gitignored `tests/` dir.
"""

from shape.agent.memory import compact_verification
from shape.agent.planner import Planner

_format_qc = Planner._format_qc


VRECORD_1 = {
    "tool": "Cell_Segmenter_Tool",
    "status": "conceded",
    "stages": [{
        "stage": "segmentation",
        "decision": "FAIL",
        "action": "retune_here",
        "status": "conceded",
        "fault": "ceiling",
        "notes": ("[vision] tok=2847 problem='target cells clearly MISSED' sev=major "
                  "fix=flow_threshold/up usable=yes (high) specimen='cell' size=large "
                  "density=sparse: The overlay shows large, distinct cells, but several "
                  "are not masked, indicating missed detections. Raising the "
                  "flow_threshold can "),
        "trials": [
            {"notes": ("[vision] tok=2847 problem='target cells clearly MISSED' sev=major "
                      "fix=flow_threshold/up usable=yes (high) specimen='cell' size=large "
                      "density=sparse: The overlay shows large, distinct cells, but several "
                      "are not masked, indicating missed detections. Raising the "
                      "flow_threshold can "),
             "metrics": {"cell_count": 10.0, "flow_error": 0.031680673717215085,
                        "tiny_mask_fraction": 0.2}},
            {"notes": ("[vision] tok=2847 problem='target cells clearly MISSED' sev=major "
                      "fix=cellprob_threshold/down usable=yes (high) specimen='cells' "
                      "size=large density=sparse: The image shows large, sparse cells with "
                      "only 10 detected, but many cells are visible and unmasked, indicating "
                      "a recall issue. Low"),
             "metrics": {"cell_count": 10.0, "flow_error": 0.031680673717215085,
                        "tiny_mask_fraction": 0.2}},
        ],
    }],
    "upstream_target": None,
}

VRECORD_2 = {
    "tool": "Cell_Segmenter_Tool",
    "status": "satisfied",
    "stages": [{
        "stage": "segmentation",
        "decision": "PASS",
        "action": "accept",
        "status": "satisfied",
        "fault": None,
        "notes": ("[vision] tok=2850 problem='none' sev=none fix=none/none usable=yes "
                  "(high) specimen='microscopy cells' size=medium density=moderate: The "
                  "segmentation overlay shows distinct masks for each cell with no clear "
                  "issues such as missed cells, false masks, or over-segmen"),
        "trials": [
            {"notes": ("[vision] tok=2850 problem='crowded cells merged' sev=major "
                      "fix=flow_threshold/up usable=yes (high) specimen='cells' size=medium "
                      "density=moderate: Several adjacent cells are merged into large blobs, "
                      "indicating a need to increase the flow threshold to better separate "
                      "touching c"),
             "metrics": {"cell_count": 32.0, "flow_error": 0.06041887607069085,
                        "tiny_mask_fraction": 0.09375}},
            {"notes": ("[vision] tok=2850 problem='none' sev=none fix=none/none usable=yes "
                      "(high) specimen='microscopy cells' size=medium density=moderate: The "
                      "segmentation overlay shows distinct masks for each cell with no clear "
                      "issues such as missed cells, false masks, or over-segmen"),
             "metrics": {"cell_count": 32.0, "flow_error": 0.06041887607069085,
                        "tiny_mask_fraction": 0.09375}},
        ],
    }],
    "upstream_target": None,
}


# ---- (c) flat single-image shape: byte-identical to current origin/main @ 2d69687, plus
# the Phase-2 refinement's `fault=` surfacing (task 045) -------------------------------
# (regression guard -- expected strings/dicts captured by running the PRISTINE pre-fix
# code, via `git stash`, on these same fixtures before the fix was applied; the `fault=`
# token was then inserted at the exact position `_format_qc` now emits it, per the
# Phase-2 refinement's diff -- not guessed.)

_FLAT_1_EXPECTED_DIGEST = (
    "segmentation=FAIL/retune_here fault=ceiling ([vision] tok=2847 problem='target "
    "cells clearly MISSED' sev=major fix=flow_threshold/up usable=yes (high) "
    "specimen='cell' size=large density=sparse: The overlay shows large, distinct "
    "cells, but several are not masked, indicating missed detections. Raising the "
    "flow_threshold can ); overall=conceded"
)
_FLAT_2_EXPECTED_DIGEST = "segmentation=PASS/accept"

_FLAT_1_EXPECTED_COMPACT = {
    "status": "conceded",
    "stages": [{
        "stage": "segmentation", "decision": "FAIL", "action": "retune_here",
        "status": "conceded", "fault": "ceiling",
        "notes": VRECORD_1["stages"][0]["notes"],
        "metrics": {"cell_count": 10.0, "flow_error": 0.0317, "tiny_mask_fraction": 0.2},
        "vision_verdict": VRECORD_1["stages"][0]["trials"][0]["notes"],
    }],
}
_FLAT_2_EXPECTED_COMPACT = {
    "status": "satisfied",
    "stages": [{
        "stage": "segmentation", "decision": "PASS", "action": "accept",
        "status": "satisfied", "fault": None,
        "notes": VRECORD_2["stages"][0]["notes"],
        "metrics": {"cell_count": 32.0, "flow_error": 0.0604, "tiny_mask_fraction": 0.0938},
        "vision_verdict": VRECORD_2["stages"][0]["trials"][0]["notes"],
    }],
}


def test_format_qc_flat_shape_unchanged():
    assert _format_qc(None, VRECORD_1) == _FLAT_1_EXPECTED_DIGEST
    assert _format_qc(None, VRECORD_2) == _FLAT_2_EXPECTED_DIGEST


def test_compact_verification_flat_shape_unchanged():
    assert compact_verification(VRECORD_1) == _FLAT_1_EXPECTED_COMPACT
    assert compact_verification(VRECORD_2) == _FLAT_2_EXPECTED_COMPACT


# ---- (a) per_image n=1 (the common case): must render IDENTICALLY to the flat shape ----

def test_format_qc_per_image_n1_matches_flat():
    per_image = {"tool": "Cell_Segmenter_Tool", "per_image": [VRECORD_1]}
    digest = _format_qc(None, per_image)
    assert digest == _format_qc(None, VRECORD_1)
    assert "[vision]" in digest        # the gap this task closes: not empty


def test_compact_verification_per_image_n1_matches_flat():
    per_image = {"tool": "Cell_Segmenter_Tool", "per_image": [VRECORD_1]}
    compacted = compact_verification(per_image)
    assert compacted == compact_verification(VRECORD_1)
    assert compacted["stages"][0]["vision_verdict"]       # present, not empty


# ---- (b) per_image N=2: each image's verdict must surface -------------------------------

def test_format_qc_per_image_n2_surfaces_each_image():
    per_image = {"tool": "Cell_Segmenter_Tool", "per_image": [VRECORD_1, VRECORD_2]}
    digest = _format_qc(None, per_image)
    d1, d2 = _format_qc(None, VRECORD_1), _format_qc(None, VRECORD_2)
    # both images' digests are present, and distinguishable (indexed).
    assert d1 in digest and d2 in digest
    assert digest == f"img0: {d1}; img1: {d2}"


def test_compact_verification_per_image_n2_surfaces_each_image():
    per_image = {"tool": "Cell_Segmenter_Tool", "per_image": [VRECORD_1, VRECORD_2]}
    compacted = compact_verification(per_image)
    c1, c2 = compact_verification(VRECORD_1), compact_verification(VRECORD_2)
    assert compacted == {"per_image": [c1, c2]}
    # both vision_verdicts present (the two records carry DIFFERENT judge notes).
    assert compacted["per_image"][0]["stages"][0]["vision_verdict"] != \
           compacted["per_image"][1]["stages"][0]["vision_verdict"]


# ---- (c) wave mode: the digest carries the DELIVERED trial's metrics ------------------------

def test_compact_verification_reports_the_delivered_trial_in_wave_mode():
    """The agentic loop's last call is not necessarily the delivered one (`details["selected"]`);
    no delivered call (None) -> the input's metrics; a linear stage (no `selected`) keeps the last."""
    def record(details):
        return {"status": "satisfied", "stages": [{
            "stage": "segmentation", "decision": "PASS", "action": "retune_here", "status": "satisfied",
            "trials": [{"metrics": {"cell_count": 0.0}}, {"metrics": {"cell_count": 64.0}},
                       {"metrics": {"cell_count": 16.0}}],
            "details": details}]}

    def metrics(details):
        return compact_verification(record(details))["stages"][0]["metrics"]

    assert metrics({"selected": 1}) == {"cell_count": 64.0}
    assert metrics({"selected": None}) == {"cell_count": 0.0}
    assert metrics({}) == {"cell_count": 16.0}

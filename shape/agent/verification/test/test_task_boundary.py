"""Task 045 Phase 3, Fix 0 -- the query task-boundary contract (TaskSpec) + scope guard.

Pure, deterministic derivation from the stage_map DAG: a counting/segmentation query
bounds scope to the segmentation prefix (crop/analysis out of scope); a cell-state /
morphology / comparison query opens the full pipeline; an unclassifiable query fails
OPEN (no boundary enforced). `tool_in_scope` blocks a managed out-of-scope tool
(cropper/analyzer on a seg query) -- the hard net against the vicious cycle.
"""

from shape.agent.verification.stage_map import (
    TaskSpec, task_spec_for, tool_in_scope,
)

_SEG_PREFIX = {"image_loading", "segmentation"}
_ALL = {"image_loading", "segmentation", "crop",
        "representation", "morphological_state"}


# -- terminal + scope derivation -------------------------------------------------

def test_counting_query_bounds_to_segmentation():
    spec = task_spec_for("Simple Counting")
    assert spec.required_terminal_stage == "segmentation"
    assert set(spec.in_scope_stages) == _SEG_PREFIX
    assert set(spec.out_of_scope_stages) == {"crop", "representation", "morphological_state"}


def test_pure_segmentation_query_bounds_to_segmentation():
    spec = task_spec_for("segmentation")
    assert spec.required_terminal_stage == "segmentation"
    assert "crop" in spec.out_of_scope_stages


def test_cell_state_query_opens_full_pipeline():
    spec = task_spec_for("Cell State Analysis")
    assert spec.required_terminal_stage == "morphological_state"
    assert set(spec.in_scope_stages) == _ALL
    assert not spec.out_of_scope_stages


def test_morphology_and_comparison_open_full_pipeline():
    for qt in ("Basic Morphology", "Comparison", "compare clusters"):
        spec = task_spec_for(qt)
        assert spec.required_terminal_stage == "morphological_state", qt
        assert not spec.out_of_scope_stages, qt


def test_unclassifiable_query_fails_open():
    spec = task_spec_for("")
    assert spec.required_terminal_stage is None
    assert set(spec.in_scope_stages) == _ALL
    assert not spec.out_of_scope_stages       # guard inert -> nothing blocked


# -- the scope guard (tool_in_scope) ---------------------------------------------

def test_seg_query_blocks_managed_out_of_scope_tools():
    spec = task_spec_for("Simple Counting")
    # the exact vicious-cycle tools must be blocked
    assert tool_in_scope("Single_Cell_Cropper_Tool", spec) is False
    assert tool_in_scope("Cell_State_Analyzer_Tool", spec) is False
    assert tool_in_scope("Cell_State_Analyzer_Tool", spec) is False
    # in-scope tools pass
    assert tool_in_scope("Cell_Segmenter_Tool", spec) is True
    assert tool_in_scope("Organoid_Segmenter_Tool", spec) is True
    # unmanaged tools are not scope-gated here
    assert tool_in_scope("Pubmed_Searcher_Tool", spec) is True


def test_cell_state_query_allows_the_full_chain():
    spec = task_spec_for("Cell State Analysis")
    for t in ("Cell_Segmenter_Tool", "Single_Cell_Cropper_Tool",
              "Cell_State_Analyzer_Tool"):
        assert tool_in_scope(t, spec) is True, t


def test_failopen_spec_blocks_nothing():
    spec = task_spec_for("")                  # unknown -> fail-open
    assert tool_in_scope("Single_Cell_Cropper_Tool", spec) is True
    assert tool_in_scope("Cell_State_Analyzer_Tool", spec) is True


def test_none_spec_blocks_nothing():
    assert tool_in_scope("Single_Cell_Cropper_Tool", None) is True


# -- the plan gate end-to-end (Verifier.verify_plan) -----------------------------

def test_verify_plan_rejects_out_of_scope_tool():
    from shape.agent.verifier import Verifier
    spec = task_spec_for("Simple Counting")
    verdict = Verifier().verify_plan("Single_Cell_Cropper_Tool", [], spec)
    assert verdict.status == "fail"
    assert "out of scope" in (verdict.fault or "").lower()


def test_verify_plan_allows_in_scope_tool():
    from shape.agent.verifier import Verifier
    spec = task_spec_for("Simple Counting")
    verdict = Verifier().verify_plan("Cell_Segmenter_Tool", [], spec)
    assert verdict.status == "pass"


def test_verify_plan_cell_state_allows_cropper_after_segmentation():
    from shape.agent.verifier import Verifier
    spec = task_spec_for("Cell State Analysis")
    verdict = Verifier().verify_plan(
        "Single_Cell_Cropper_Tool", ["Cell_Segmenter_Tool"], spec)
    assert verdict.status == "pass"


def test_verify_plan_failopen_still_enforces_ordering():
    """No task_spec -> scope guard inert, but the ORDERING check still fires
    (cropper before any segmentation) with a NON-scope fault."""
    from shape.agent.verifier import Verifier
    verdict = Verifier().verify_plan("Single_Cell_Cropper_Tool", [], None)
    assert verdict.status == "fail"
    assert "out of scope" not in (verdict.fault or "").lower()

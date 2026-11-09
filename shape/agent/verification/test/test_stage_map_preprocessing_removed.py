"""Regression gate: `Image_Preprocessor_Tool` / the `preprocessing` VLRR stage are
FUNCTIONALLY REMOVED from the agentic loop's tool discovery and stage graph (a live
083 NeurIPS run showed the planner still selecting the tool, whose command then failed
the executor's safety parser -- wasted tool call, `wasted_rate` pollution). The tool's
own code is UNCHANGED and stays directly importable/runnable (task 086's artifacts,
`shape/tools/image_preprocessor/`) -- this is an unregister, not a delete. Co-located
with the module under test (committed -- AGENTS.md sec 6 P5), not in the gitignored
`tests/` dir.
"""

from __future__ import annotations

from shape.agent.verification.stage_map import STAGES, is_managed_tool, stage_by_name


# ---- (a) planner discovery: the tool never reaches the planner's tool list --------

def test_card_index_excludes_image_preprocessor():
    from shape.tools.base import card_index
    assert "Image_Preprocessor_Tool" not in card_index()


def test_auto_discovered_available_tools_excludes_image_preprocessor():
    """`get_available_tools()` with no override is the SAME auto-discovery `solve()`
    uses by default -- the actual path the planner's tool list is built from (not
    `card_index()` directly; `shape.tools.initializer.Initializer` walks the tools/
    directory independently and must honor the same exclusion)."""
    from shape.solver import get_available_tools
    assert "Image_Preprocessor_Tool" not in get_available_tools()


def test_bioimage_tool_priorities_excludes_image_preprocessor():
    from shape.tools.tool_priority import BIOIMAGE_TOOL_PRIORITIES
    assert "Image_Preprocessor_Tool" not in BIOIMAGE_TOOL_PRIORITIES


def test_planner_rendered_tool_list_excludes_image_preprocessor():
    """The exact text the planner prompt shows the LLM (`format_tools_by_priority`,
    grouped by CARD category) must not name the tool either."""
    from shape.solver import get_available_tools
    from shape.tools.tool_priority import ToolPriorityManager

    tools = get_available_tools()
    rendered = ToolPriorityManager().format_tools_by_priority(tools)
    assert "Image_Preprocessor_Tool" not in rendered


def test_explicit_enabled_tools_no_longer_resolves_deleted_tool():
    """Physically DELETED (shape/tools/image_preprocessor/ removed) -- even an explicit
    `enabled_tools=[...]` request can no longer resolve it."""
    from shape.solver import get_available_tools
    try:
        result = get_available_tools(["Image_Preprocessor_Tool"])
    except Exception:
        return  # the tool is gone; raising on an unknown explicit request is acceptable
    assert "Image_Preprocessor_Tool" not in result


# ---- (b) the stage graph: no `preprocessing` stage, no dangling upstream ----------

def test_no_preprocessing_stage_in_stages():
    assert "preprocessing" not in {s.stage for s in STAGES}
    assert stage_by_name("preprocessing") is None


def test_no_stage_spec_references_image_preprocessor_tool():
    for s in STAGES:
        assert "Image_Preprocessor_Tool" not in s.tool_names, s.stage


def test_every_stage_upstream_points_at_an_existing_stage():
    names = {s.stage for s in STAGES}
    for s in STAGES:
        assert s.upstream is None or s.upstream in names, (s.stage, s.upstream)


def test_segmentation_upstream_is_image_loading():
    assert stage_by_name("segmentation").upstream == "image_loading"


def test_representation_upstream_is_image_loading():
    assert stage_by_name("representation").upstream == "image_loading"


def test_pipeline_is_exactly_image_loading_segmentation_crop_representation_morph():
    assert [s.stage for s in STAGES] == [
        "image_loading", "segmentation", "crop", "representation", "morphological_state",
    ]


# ---- (c) tool_has_stage equivalent (`is_managed_tool` -- see CODING_LOG.md) --------

def test_image_preprocessor_tool_has_no_managed_stage():
    assert is_managed_tool("Image_Preprocessor_Tool") is False


# ---- the tool has been physically DELETED (shape/tools/image_preprocessor/ removed) ----

def test_image_preprocessor_tool_is_deleted():
    import pytest
    with pytest.raises((ModuleNotFoundError, ImportError)):
        import shape.tools.image_preprocessor.tool  # noqa: F401

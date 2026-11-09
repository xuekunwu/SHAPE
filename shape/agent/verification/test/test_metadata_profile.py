"""The metadata profile of a solve() run (`metadata_profile`), the batch stage's inputs and task
(`batch_stage`), the managed CARDs' tier declarations, and the D3 record of a deliberately kept input."""
from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from shape.agent.verification import metadata_profile as mp
from shape.agent.verification.batch_stage import BRIEF, batch_inputs_from_h5ad, batch_task_for


@pytest.fixture
def extra(monkeypatch):
    def set_available(available=True):
        monkeypatch.setattr(mp, "batch_metrics_problem",
                            lambda: None if available else "ModuleNotFoundError: No module named 'jax'")
    set_available()
    return set_available


def _spec(image_id, plate=None, group="default", **more):
    spec = {"path": f"{image_id}.png", "image_id": image_id, "group": group, **more}
    if plate is not None:
        spec["plate"] = plate
    return spec


def _design(groups=("a", "b"), plates=("P1", "P2"), per=1, **more):
    return [_spec(f"{p}_{g}_{i}", p, g, **more) for p in plates for g in groups for i in range(per)]


@pytest.mark.parametrize("specs, tier, reason", [
    ([_spec("one", "P1", "a")], "none", "1 image(s)"),
    ([_spec("x", "P1", "a"), _spec("y", "P1", "b")], "none", "one batch value"),
    ([_spec("x", "P1", "a"), _spec("y", None, "a"), _spec("z", "P2", "a")], "none", "partial coverage"),
    ([_spec("x"), _spec("y")], "none", "no image declares a batch value"),
    (_design(plates=("P1", "P2")) + [_spec("lone", "P3", "a")], "none", "fewer than 2 images: P3"),
    (_design(groups=("default",), per=2), "partial", "fewer than 2 non-control groups"),
    # one replicated group: no biology readout, so biology preservation cannot be verified
    ([_spec("x", "P1", "a"), _spec("y", "P2", "a"), _spec("z", "P1", "b"), _spec("w", "P2", "c")], "partial",
     "biology preservation cannot be verified"),
    ([_spec("x", "P1", "a"), _spec("y", "P1", "a"), _spec("z", "P2", "b"), _spec("w", "P2", "b")], "partial",
     "fewer than 2 non-control groups"),
    (_design(), "full", None),
])
def test_the_profile_tier_table(extra, specs, tier, reason):
    profile = mp.build_metadata_profile("q", specs)
    assert profile.tier == tier and mp.batch_routed(profile) is (tier != "none")
    assert (reason is None and not profile.reasons) or any(reason in r for r in profile.reasons)


def test_a_missing_batch_metrics_extra_downgrades_full_to_partial(extra):
    extra(False)
    design = mp.build_metadata_profile("q", _design())
    assert (design.tier, design.batch_metrics, design.reasons) == ("full", None, ())     # not checked yet
    profile = mp.resolve_batch_metrics(design)
    assert profile.tier == "partial" and any("scib-metrics" in r for r in profile.reasons)
    assert profile.batch_metrics == "ModuleNotFoundError: No module named 'jax'"
    assert mp.resolve_batch_metrics(profile) is profile                              # checked once
    extra(True)
    assert mp.resolve_batch_metrics(design).tier == "full"
    assert mp.resolve_batch_metrics(design).batch_metrics == "usable"


def test_the_profile_fields(extra):
    specs = _design(groups=("a", "b"), per=2) + [_spec("c1", "P1", "dmso", control=True),
                                                 _spec("c2", "P2", "dmso", control=True)]
    profile = mp.build_metadata_profile("## Target\nnuclei", specs)
    assert profile.batch == {"available": True, "key": "plate", "labels": {s["image_id"]: s["plate"] for s in specs},
                             "n_batches": 2, "images_per_batch": {"P1": 5, "P2": 5}}
    assert profile.groups["replicated"] == ["a", "b"] and profile.groups["in_all_batches"] == ["a", "b"]
    assert profile.controls == {"images": ["c1", "c2"], "groups": ["dmso"], "per_batch": {"P1": 1, "P2": 1},
                                "in_every_batch": True}
    assert profile.request == {"target_section": True} and profile.technical == {"well": False, "site": False}
    assert profile.n_images == 10 and profile.to_dict()["reasons"] == []
    assert mp.build_metadata_profile("Compare a and b.", specs).request == {"target_section": False}


def test_the_batch_value_is_the_analyzers(extra):
    """The profile's batch label of an image is the analyzer's obs.batch for its cells
    (`resolve_technical_metadata`), for every key and precedence."""
    from shape.tools.cell_state_analyzer.tool import resolve_technical_metadata

    cases = [{"plate": "P1"}, {"plate_id": "P2"}, {"experiment": "E1"}, {"experiment_id": "E2"}, {"batch": "B1"},
             {"batch_id": "B2"}, {"plate": "P1", "experiment": "E1", "batch": "B1"}, {"experiment": "E1", "batch": "B1"},
             {"plate": " ", "batch": "B3"}, {"plate_id": "P4", "batch": "B4"}, {"plate": 7}]
    for case in cases:
        _, batches, _ = resolve_technical_metadata([case], ["img"])
        assert mp.batch_value(case)[0] == batches[0], case
    assert mp.batch_value({})[0] is None and resolve_technical_metadata([{}], ["img"])[1] == ["img"]


def _h5ad(tmp_path, specs, n=6):
    import anndata as ad

    rng = np.random.default_rng(0)
    obs = [{"batch": s.get("plate", s["image_id"]), "group": s["group"], "image_name": s["image_id"],
            "technical_group": s["image_id"]} for s in specs for _ in range(n)]
    adata = ad.AnnData(rng.normal(size=(len(obs), 4)).astype(np.float32), obs=pd.DataFrame(obs))
    adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
    adata.uns["embedding_fingerprint"] = "fp"
    path = tmp_path / "cell_state_analyzed.h5ad"
    adata.write(path)
    return str(path)


def test_the_build_base_kwargs_batch_branch(tmp_path):
    from shape.agent.memory import Memory
    from shape.agent.verification.integration import build_base_kwargs

    path = _h5ad(tmp_path, _design())
    assert build_base_kwargs("Batch_Corrector_Tool", None, None) == {}
    memory = Memory()
    assert build_base_kwargs("Batch_Corrector_Tool", None, memory) == {}
    memory.add_action(1, "Cell_State_Analyzer_Tool", "s", "c", {"per_image": [{"adata_path": path, "image_id": "x"}]})
    memory.add_action(1, "Cell_State_Analyzer_Tool", "s", "c", {"adata_path": "harmony.h5ad", "batch_corrected": True})
    kwargs = build_base_kwargs("Batch_Corrector_Tool", None, memory)
    assert set(kwargs) == {"cls_output", "metadata", "batch_key", "row_ids", "adata_path", "fingerprint"}
    assert kwargs["cls_output"].shape == (24, 4) and kwargs["cls_output"].dtype == float
    assert list(kwargs["metadata"].columns) == ["batch", "group", "image_name", "technical_group"]
    assert (kwargs["batch_key"], kwargs["adata_path"], kwargs["fingerprint"]) == ("batch", path, "fp")
    assert kwargs["row_ids"][:2] == ["cell_0", "cell_1"]


def test_batch_task_for_the_full_and_partial_tiers(tmp_path, extra):
    from shape.agent.verification.verifiers.batch_correction_unified import NOISE_BAND, admission_evidence

    specs = _design(groups=("a", "b", "c")) + [_spec("c1", "P1", "dmso", control=True),
                                               _spec("c2", "P2", "dmso", control=True), _spec("d", "P1")]
    inputs = batch_inputs_from_h5ad(_h5ad(tmp_path, specs))
    task = batch_task_for(mp.build_metadata_profile("q", specs), inputs)
    assert task.noise_band == NOISE_BAND and task.batch_column == "plate" and task.admission is None
    view = task.calibration
    assert len(view.rows) == 8 * 6 and view.label_key == "group" and view.batch_key == "batch"     # not "d"
    assert sorted(set(view.meta["Metadata_PertType"])) == ["negcon", "trt"]
    evidence = admission_evidence(task, {"detected": True})
    assert (evidence["reference_groups"], evidence["reference_groups_in_2plus_batch_groups"],
            evidence["reference_groups_in_all_batch_groups"]) == (3, 3, 3)
    assert evidence["negative_controls_per_batch_group"] == {"P1": 6, "P2": 6}
    extra(False)
    partial = batch_task_for(mp.resolve_batch_metrics(mp.build_metadata_profile("q", specs)), inputs)
    assert partial.calibration is None and admission_evidence(partial, {})["reference_groups"] == 0


def test_a_control_without_a_group_enters_the_calibration_view_as_negcon(tmp_path, extra):
    from shape.agent.verification.batch_stage import CONTROL_LABEL
    from shape.agent.verification.verifiers.batch_correction_unified import admission_evidence

    specs = _design(groups=("a", "b")) + [_spec("c1", "P1", control=True), _spec("c2", "P2", control=True)]
    profile = mp.build_metadata_profile("q", specs)
    assert profile.tier == "full" and profile.controls["in_every_batch"] is True and profile.controls["groups"] == []
    task = batch_task_for(profile, batch_inputs_from_h5ad(_h5ad(tmp_path, specs)))
    view = task.calibration
    negcon = view.meta["Metadata_PertType"] == "negcon"
    assert len(view.rows) == 6 * 6 and int(negcon.sum()) == 12
    assert set(view.meta.loc[negcon, "group"]) == {CONTROL_LABEL}
    evidence = admission_evidence(task, {"detected": True})
    assert evidence["negative_controls_per_batch_group"] == {"P1": 6, "P2": 6}
    assert evidence["reference_groups"] == 2                      # the controls are not reference groups


def test_an_unusable_batch_metrics_extra_is_the_partial_tier_with_the_reason(monkeypatch):
    def failing_import(name):
        if name == "jax":
            raise ImportError("DLL load failed while importing jaxlib")
        return __import__("types").ModuleType(name)

    mp.batch_metrics_problem.cache_clear()
    monkeypatch.setattr(mp, "_import", failing_import)
    try:
        assert mp.batch_metrics_problem() == "ImportError: DLL load failed while importing jaxlib"
        profile = mp.resolve_batch_metrics(mp.build_metadata_profile("q", _design()))
        assert profile.tier == "partial"
        assert any("not usable (ImportError: DLL load failed" in r for r in profile.reasons)
        mp.batch_metrics_problem.cache_clear()
        monkeypatch.setattr(mp, "_import", lambda name: __import__("types").ModuleType(name))
        assert "AttributeError" in mp.batch_metrics_problem()          # a module without the functions it needs
    finally:
        mp.batch_metrics_problem.cache_clear()


def test_the_check_is_lazy_and_turns_jax_preallocation_off_before_importing(monkeypatch):
    seen = []

    def recording_import(name):
        seen.append((name, mp.os.environ.get("XLA_PYTHON_CLIENT_PREALLOCATE")))
        raise ImportError(f"no {name}")

    mp.batch_metrics_problem.cache_clear()
    monkeypatch.setattr(mp, "_import", recording_import)
    monkeypatch.delenv("XLA_PYTHON_CLIENT_PREALLOCATE", raising=False)
    try:
        design = mp.build_metadata_profile("q", _design())
        assert design.tier == "full" and seen == []                    # building the profile imports nothing
        assert mp.resolve_batch_metrics(design).tier == "partial"
        assert seen == [("jax", "false")]                              # the default is set before the import
        mp.batch_metrics_problem.cache_clear()
        seen.clear()
        monkeypatch.setenv("XLA_PYTHON_CLIENT_PREALLOCATE", "true")    # a caller's own setting is kept
        mp.batch_metrics_problem()
        assert seen == [("jax", "true")]
    finally:
        mp.batch_metrics_problem.cache_clear()


def test_a_full_design_whose_cells_lost_a_group_runs_as_the_partial_tier(tmp_path, extra):
    from shape.agent.verification.batch_stage import cells_profile, replicated_groups_with_cells

    specs = _design(per=2)                                         # 2 plates x groups a, b x 2 images
    profile = mp.build_metadata_profile("q", specs)
    full = batch_inputs_from_h5ad(_h5ad(tmp_path, specs))
    assert replicated_groups_with_cells(full, profile) == ["a", "b"] and cells_profile(profile, full) is profile
    lost = batch_inputs_from_h5ad(_h5ad(tmp_path, [s for s in specs if s["group"] != "b"]))   # no "b" cell
    step = cells_profile(profile, lost)
    assert (profile.tier, step.tier) == ("full", "partial") and step.reasons[:-1] == profile.reasons
    assert step.reasons[-1] == ("after segmentation and cropping only 1 non-control groups have cells in 2 or more "
                                "batch groups (at least 2 needed): biology preservation cannot be verified")
    assert batch_task_for(step, lost).calibration is None


def test_the_h5ad_needs_two_batch_groups_of_two_images_with_cells(tmp_path):
    from shape.agent.verification.batch_stage import design_problem

    specs = _design(per=1)                                         # 2 plates x 2 groups x 1 image
    assert design_problem(batch_inputs_from_h5ad(_h5ad(tmp_path, specs))) is None
    lone = [s for s in specs if s["image_id"] != "P2_b_0"]          # an image of P2 produced no cell
    assert design_problem(batch_inputs_from_h5ad(_h5ad(tmp_path, lone))) == (
        "batch group(s) with fewer than 2 images with cells in the analyzer's h5ad: P2")
    one = [s for s in specs if s["plate"] == "P1"]
    assert design_problem(batch_inputs_from_h5ad(_h5ad(tmp_path, one))) == (
        "the analyzer's h5ad holds cells from 1 batch group(s)")


def test_a_selected_call_without_harmony_is_kept_as_the_input_with_a_note(extra):
    from shape.agent.verification.batch_stage import _batch_record
    from shape.utils.schemas import StageResolution, VLRRResult

    verdict = VLRRResult("WARNING", 0.6, {}, action="accept")
    skipped = {"status": "SKIPPED_NO_BATCH_EFFECT", "gt_free_metrics": {"batch_effect_detected": False}}
    res = StageResolution(result=skipped, status="satisfied", params={}, verdict=verdict,
                          trials=[({}, verdict, {"status": "UNCORRECTED_INPUT", "batch_effect": {"detected": True}})],
                          reason="plateau", details={"selected": 2, "admission": {"decision": "RUN"},
                                                     "rounds": [["agent_1", "agent_2"]]})
    record = _batch_record(res, mp.build_metadata_profile("q", _design()), {"adata_path": "a.h5ad"})
    assert (record["status"], record["selected_call"], record["selected_status"]) == (
        "UNCORRECTED_INPUT", "agent_2", "SKIPPED_NO_BATCH_EFFECT")
    assert "not a Harmony correction" in record["note"] and record["corrected_matrix_path"] is None


def test_the_deliverable_gives_the_reason_of_a_correction_that_was_not_written(extra):
    from shape.agent.memory import Memory
    from shape.agent.verification.batch_stage import cell_states_deliverable, record_not_routed

    profile = mp.build_metadata_profile("q", _design())
    memory = Memory()
    memory.add_action(1, "Cell_State_Analyzer_Tool", "s", "c", {"adata_path": "a.h5ad", "cluster_key": "leiden_0.5"})
    memory.add_action(1, "Batch_Corrector_Tool", "s", "c", {"status": "CORRECTED_HARMONY", "decision": "RUN",
                                                            "corrected_h5ad_error": "OSError: disk full"})
    record_not_routed(memory, 2, "this analyzer step was already handled")
    states = cell_states_deliverable(memory, profile)
    assert states["delivered"] == "uncorrected" and states["corrected"] is None
    assert (states["batch_correction"]["status"], states["batch_correction"]["reason"]) == (
        "CORRECTED_HARMONY", "OSError: disk full")
    memory = Memory()
    memory.add_action(1, "Cell_State_Analyzer_Tool", "s", "c", {"adata_path": "a.h5ad", "cluster_key": "leiden_0.5"})
    record_not_routed(memory, 1, "the analyzer conceded upstream (representation -> crop)")
    assert cell_states_deliverable(memory, profile)["batch_correction"]["reason"] == (
        "the analyzer conceded upstream (representation -> crop)")


def test_a_system_routed_pick_is_recognised_through_the_planners_name_check():
    from shape.solver import _system_routed_name
    from shape.utils.agent_io import normalize_tool_name

    toolbox = ["Cell_Segmenter_Tool", "Single_Cell_Cropper_Tool", "Cell_State_Analyzer_Tool"]
    picked = normalize_tool_name("Batch_Corrector_Tool", toolbox)
    assert picked == "No matched tool given: Batch_Corrector_Tool"
    assert _system_routed_name(picked) == _system_routed_name("batch_corrector_tool") == "Batch_Corrector_Tool"
    assert _system_routed_name(normalize_tool_name("Cell_State_Analyzer_Tool", toolbox)) is None
    assert _system_routed_name("No matched tool given: Unknown_Tool") is None
    # names compare without spaces, underscores and hyphens, case-folded
    for name in ("Batch Corrector Tool", "batch-corrector-tool", "BATCH_CORRECTOR_TOOL",
                 normalize_tool_name("Batch Corrector Tool", toolbox)):
        assert _system_routed_name(name) == "Batch_Corrector_Tool", name
    assert _system_routed_name("Batch Corrector") is None


def test_the_exact_brief(tmp_path, extra):
    specs = _design(groups=("a", "b"), per=2) + [_spec("c1", "P1", "dmso", control=True)]
    task = batch_task_for(mp.build_metadata_profile("q", specs), batch_inputs_from_h5ad(_h5ad(tmp_path, specs)))
    assert task.brief == ("Cell-level embeddings from the Cell_State_Analyzer_Tool: 54 cells from 9 images in 2 batch "
                          "groups (batch column: plate); 2 reference groups occur in 2 or more batch groups; 1 control "
                          "images.")
    assert BRIEF.count("{") == 6


def test_every_managed_card_declares_its_metadata_tiers():
    from shape.agent.verification.stage_map import MANAGED_STAGES
    from shape.tools.base import load_card

    tools = {tool for spec in MANAGED_STAGES for tool in spec.tool_names}
    assert tools == {"Batch_Corrector_Tool", "Cell_Segmenter_Tool", "Organoid_Segmenter_Tool",
                     "Single_Cell_Cropper_Tool", "Cell_State_Analyzer_Tool"}
    for tool in tools:
        stem = "_".join(p.lower() for p in tool.replace("_Tool", "").split("_"))
        header, body = load_card(stem)
        assert header["tool"] == tool
        assert isinstance(header["metadata_consumes"], list), tool
        for key in ("metadata_none", "metadata_partial", "metadata_full"):
            assert isinstance(header[key], str) and header[key], (tool, key)
        assert not any(k in body for k in ("metadata_consumes", "metadata_none", "metadata_partial",
                                            "metadata_full")), tool      # front matter only: no prompt carries it


def test_the_planner_toolbox_omits_the_system_routed_tools(monkeypatch):
    import shape.solver as solver_module
    from shape.agent.verification.stage_map import SYSTEM_ROUTED_TOOLS

    class _Initializer:
        def __init__(self, **kwargs):
            self.available_tools = ["Cell_State_Analyzer_Tool", "Batch_Corrector_Tool"]

    class _Planner:
        def __init__(self, available_tools, **kwargs):
            self.available_tools = available_tools

    monkeypatch.setattr(solver_module, "Initializer", _Initializer)
    monkeypatch.setattr(solver_module, "Planner", _Planner)
    solver = solver_module.construct_solver(llm_engine_name="gpt-4o", api_key="unused")
    assert SYSTEM_ROUTED_TOOLS == ("Batch_Corrector_Tool",)
    assert solver["planner"].available_tools == ["Cell_State_Analyzer_Tool"]
    assert solver["available_tools"] == ["Cell_State_Analyzer_Tool", "Batch_Corrector_Tool"]


# ---- D3: a deliberately kept input is not "no output" -------------------------------------------------------
def _resolution(reason, result, selected, status="conceded"):
    from shape.utils.schemas import StageResolution, VLRRResult

    verdict = VLRRResult("WARNING", 0.5, {"overall": 0.5}, action="conceded" if status == "conceded" else "accept")
    return StageResolution(result=result, status=status, params={}, verdict=verdict,
                           trials=[({}, verdict, {"status": "UNCORRECTED_INPUT"})],
                           fault="ceiling" if status == "conceded" else None, reason=reason,
                           details={"selected": selected})


@pytest.mark.parametrize("reason, result, selected, status, kept, no_output", [
    ("separability_unverifiable", None, None, "conceded", "separability_unverifiable", False),   # ABSTAIN
    ("agent_kept_input", None, None, "conceded", "agent_kept_input", False),
    ("no_admissible_call", None, None, "conceded", "no_admissible_call", False),
    ("no_batch_effect", {"status": "UNCORRECTED_INPUT"}, 0, "satisfied", "no_batch_effect", False),   # SKIP
    ("initial_trial_failed", None, None, "conceded", None, True),
    ("no_call_finished", None, None, "conceded", None, True),
    ("plateau", {"status": "CORRECTED_HARMONY"}, 3, "satisfied", None, False),
])
def test_a_kept_input_is_recorded_as_kept_not_as_no_output(reason, result, selected, status, kept, no_output):
    from shape.agent.memory import Memory
    from shape.agent.planner import Planner
    from shape.agent.verification.integration import make_verification_record

    record = make_verification_record("Batch_Corrector_Tool", [_resolution(reason, result, selected, status)], status)
    stage = record["stages"][0]
    assert (stage["kept_input"], stage["no_output"]) == (kept, no_output)
    memory = Memory()
    memory.add_action(1, "Batch_Corrector_Tool", "s", "c", {"status": "x"}, verification=record)
    digest = Planner._format_memory(object.__new__(Planner), memory)
    assert ("INPUT KEPT unchanged, deliberately" in digest) is (kept is not None)
    assert ("NO OUTPUT produced" in digest) is no_output


def test_a_linear_stage_without_output_is_still_no_output():
    from shape.agent.verification.integration import make_verification_record

    record = make_verification_record("Cell_Segmenter_Tool", [_resolution("max_iters_reached", None, None)],
                                      "conceded")
    assert (record["stages"][0]["kept_input"], record["stages"][0]["no_output"]) == (None, True)

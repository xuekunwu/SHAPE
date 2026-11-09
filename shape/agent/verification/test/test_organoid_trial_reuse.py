"""Organoid reuse keeps native identity and reruns Verify/VRR on every pass."""
from pathlib import Path
from types import SimpleNamespace
from functools import wraps
import numpy as np
import pytest
import torch

from shape.agent.verification import segmentation_cohort as cohort
from shape.agent.verification.test.test_organoid_agentic import NativeOrganoidTool, image
from shape.agent.verification.test.test_segmentation_golden_path import Engine, move
from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
from shape.utils.schemas import StageResolution, VLRRResult


def managed(image, tool, engine=None, **options):
    from shape.agent.verification.integration import verify_managed_tool
    from shape.agent.verification.stage_map import invoke_managed_tool
    executor = SimpleNamespace(enable_llm_verifier=engine is not None,
        verifier_llm_engine_mm=engine,
        run_managed_tool=lambda name, bk, params: invoke_managed_tool(tool, name, bk, params))
    return verify_managed_tool(executor, "Organoid_Segmenter_Tool",
        {"image": image, "image_id": "organoid"}, request="Segment whole organoids.",
        verifier_options=options or None)


@pytest.fixture
def cached_tool(tmp_path, monkeypatch):
    weight = tmp_path / "co.weights"
    weight.write_bytes(b"fixed custom CO checkpoint")
    tool = Organoid_Segmenter_Tool.__new__(Organoid_Segmenter_Tool)
    tool.device = torch.device("cpu")
    tool.model_path, tool._model_loaded = str(weight), True
    tool.model = SimpleNamespace(pretrained_model=str(weight), net=torch.nn.Linear(1, 1))
    calls = []
    original = Organoid_Segmenter_Tool._execute_native

    @wraps(original)
    def native(self, **kwargs):
        folder = Path(kwargs.get("query_cache_dir") or (tmp_path / ("direct_"+str(len(calls)))))
        folder.mkdir(parents=True, exist_ok=True)
        fake = NativeOrganoidTool(folder, debris=True)
        result = fake.execute(**kwargs)
        result["mask_path"] = result["organoid_mask_path"]
        calls.append(kwargs)
        return result

    monkeypatch.setattr(Organoid_Segmenter_Tool, "_execute_native", native)
    tool.enable_trial_cache(tmp_path / "native_trials", seed=7)
    return tool, calls


def test_cohort_to_full_vrr_reuses_native_files_but_still_reasons(image, tmp_path, cached_tool):
    tool, calls = cached_tool
    fast, _, _ = managed(image, tool, cohort_levers={})
    assert fast[0].details["measured_problems"]["default"]  # debris still needs VRR
    engine = Engine([[move("default", min_size=100, expected=[("tiny_mask_fraction", "down")])]],
                    final=["agent_1"], severity={"default": "major", "agent_1": "none"})
    full, result, status = managed(image, tool, engine)
    assert status == "satisfied" and full[0].details["call"] == "agent_1"
    assert len(calls) == 2  # default once plus one novel candidate, not default twice
    assert full[0].trials[1][2]["native_cache"]["hit"]
    assert engine.contents and full[0].details["reconciliation"][0]["status"] == "reconciled"
    assert not full[0].details["measured_problems"]["agent_1"]
    np.testing.assert_array_equal(np.load(result["mask_path"]), np.load(full[0].result["mask_path"]))


def test_missing_or_changed_artifact_recomputes_instead_of_reusing(image, tmp_path, cached_tool):
    tool, calls = cached_tool
    kwargs = dict(image=image, image_id="x", attempt_id=1)
    first = tool.execute(**kwargs)
    Path(first["mask_path"]).write_bytes(b"corrupt")
    second = tool.execute(**kwargs)
    assert len(calls) == 2 and not second["native_cache"]["hit"]
    assert second["mask_path"] != first["mask_path"]


def test_input_bytes_and_inference_recipe_invalidate_cache(image, cached_tool):
    tool, calls = cached_tool
    kwargs = dict(image=image, image_id="x", attempt_id=1)
    tool.execute(**kwargs)
    tool.execute(**kwargs)
    assert len(calls) == 1
    tool.execute(**kwargs, min_size=100)
    assert len(calls) == 2
    # Keep valid input image bytes while changing the actual pixels.
    from PIL import Image
    Image.fromarray(np.zeros((600, 600), dtype=np.uint8)).save(image)
    tool.execute(**kwargs)
    assert len(calls) == 3


def test_direct_rescue_is_not_cached_and_disable_restores_original_path(image, cached_tool):
    tool, calls = cached_tool
    tool.execute(image=image, image_id="x")  # no managed attempt ID
    tool.execute(image=image, image_id="x")
    assert len(calls) == 2
    tool.disable_trial_cache()
    for _ in range(2):
        tool.execute(image=image, image_id="x", attempt_id=1)
    assert len(calls) == 4


def test_dirty_calibration_never_transfers_its_parameters_to_other_organoids():
    card = OrganoidSegmentationAgenticVerifier()
    specs = [{"image_id": f"o{i}", "path": f"o{i}.png"} for i in range(12)]
    calls = []

    def run_one(spec, options):
        calls.append((spec["image_id"], options))
        res = StageResolution(result={"organoid_mask_path": "m.npy"}, status="satisfied",
            params={**card.defaults(), "cellprob_threshold": -5},
            verdict=VLRRResult("PASS", .5, {}), details={
                "call": "default", "measured_problems": {"default": ["tiny_mask_fraction 0.5 above 0.15"]}})
        return [res], res.result, res.status, None

    _, record = cohort.run(specs, run_one, lambda items, fn: [fn(s) for s in items],
        lambda _: SimpleNamespace(confidence="high", specimen_context="organoid", modality="brightfield"),
        card.defaults(), card.levers, tool_name="Organoid_Segmenter_Tool")
    group = record["groups"][0]
    assert group["routing"] == "per_image_insufficient_clean_calibration"
    assert len(group["calibration_excluded"]) == 3 and group["calibration_eligible"] == []
    assert all(options is None for _, options in calls)
    assert record["counts"] == {"calibration": 3, "per_image_vrr": 9}



@pytest.mark.parametrize("seed", [0, None])
def test_solver_cache_scope_restored_even_when_cohort_fails(tmp_path, monkeypatch, seed):
    from shape.solver import _segmentation_cohort
    disabled = []
    tool = SimpleNamespace(disable_trial_cache=lambda: disabled.append(True))
    executor = SimpleNamespace(enable_llm_verifier=False, query_cache_dir=str(tmp_path),
                               organoid_trial_cache=None, _managed_tools={"Organoid_Segmenter_Tool": tool})
    def fail(*args, **kwargs):
        if seed is None:
            assert executor.organoid_trial_cache is None
        else:
            assert executor.organoid_trial_cache == {"root": str(tmp_path/"organoid_native_trials"), "seed": 0}
        raise RuntimeError("synthetic cohort failure")
    monkeypatch.setattr(cohort, "run", fail)
    with pytest.raises(RuntimeError, match="synthetic cohort failure"):
        _segmentation_cohort({"executor": executor, "segmentation_cache_seed": seed}, [], None,
                             "Organoid_Segmenter_Tool")
    assert executor.organoid_trial_cache is None
    assert disabled == [True]

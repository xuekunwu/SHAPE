"""solve()'s multi-image chain (segment -> crop -> analyze) with SHAPE's own VRR at every step: no fallback (a
managed step that raises is a hard error), the analyzer's agentic VRR on the solve() path (trial 0 the frozen backbone, the
crops and metadata as the tool loads them, the cropper's rejected crops left out), the delivered h5ad inside the
query cache, and `image_workers` images at a time with the GPU tool calls taking turns. A stub analyzer (the real
clustering code) and fake executors / verifiers; no reasoner but stub engines.
Co-located with the module under test (committed -- AGENTS.md sec 6 P5).
"""
from __future__ import annotations

import json
import os
import threading
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest

from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool, analysis_dir

TOOL = "Cell_State_Analyzer_Tool"
FINGERPRINT = "stub-fingerprint"


# ---- the crop stage's outputs, the analyzer as a stub ---------------------------------------------------------

def _crop_cache(qcd: Path, images: list[dict], rejected=()) -> dict:
    """What the crop stage leaves in `<qcd>/tool_cache`: per image a success metadata file (crops, per-crop metadata
    with the image's technical identifiers, as the cropper copies them) and, for an image with rejected crops, the
    kept-crops manifest (arm C's exclusions). Returns {image_id: [crop paths]}."""
    cache = qcd / "tool_cache"
    cache.mkdir(parents=True, exist_ok=True)
    out = {}
    for n, image in enumerate(images):
        image_id, technical = image["image_id"], image.get("technical", {})
        crops, meta = [], []
        for c in range(image.get("cells", 3)):
            path = cache / f"{image_id}_crop_{c:03d}.png"
            path.write_bytes(b"crop")
            crops.append(str(path))
            meta.append({"cell_id": f"default_{image_id}_cell_{c:04d}", "crop_path": str(path), "area": 50 + c,
                         "group": "default", "image_name": image_id, **technical})
        record = {"source_image_id": image_id, "execution_status": "success", "group": "default",
                  "technical_metadata": technical, "cell_crops_paths": crops, "cell_metadata": meta}
        name = cache / f"cell_crops_metadata_{n:04d}.json"
        name.write_text(json.dumps(record), encoding="utf-8")
        kept = [p for p in crops if p not in rejected]
        if len(kept) != len(crops):
            (cache / name.name.replace("cell_crops_metadata_", "cell_crops_kept_manifest_")).write_text(
                json.dumps({"kept_crop_paths": kept}), encoding="utf-8")
        out[image_id] = crops
    return out


def _stub_analyzer(calls: list):
    """The analyzer as the loop sees it: an h5ad in `analysis_dir(query_cache_dir, output_subdir)` with one row per
    crop it was given (X random, the real clustering), the real fail-closed reuse on `reuse_embedding`; records the
    kwargs of every call that embeds."""
    import anndata as ad
    import pandas as pd

    class StubAnalyzer(Cell_State_Analyzer_Tool):
        def execute(self, cell_crops=None, cell_metadata=None, query_cache_dir=None, output_subdir=None,
                    cluster_resolution=0.5, n_neighbors=15, reuse_embedding=False, adaptation_depth=None,
                    learning_rate=3e-5, split_key=None, val_fraction=None, **kwargs):
            out = analysis_dir(query_cache_dir, output_subdir)
            path = os.path.join(out, "cell_state_analyzed.h5ad")
            if reuse_embedding and os.path.exists(path):
                reused = self._reuse_cached_embedding(path, FINGERPRINT, cluster_resolution, n_neighbors)
                if reused is not None:
                    return reused
            if cell_crops is None or cell_metadata is None:
                cell_crops, cell_metadata, _ = self._load_cell_data_from_metadata(query_cache_dir)
            calls.append({"cell_crops": list(cell_crops), "cell_metadata": [dict(m) for m in cell_metadata],
                          "output_subdir": output_subdir, "adaptation_depth": adaptation_depth,
                          "learning_rate": learning_rate, "split_key": split_key, "val_fraction": val_fraction})
            x = np.random.default_rng(len(calls)).normal(size=(len(cell_crops), 8)).astype(np.float32)
            obs = pd.DataFrame({"image_name": [m["image_name"] for m in cell_metadata],
                                "group": [m["group"] for m in cell_metadata],
                                "batch": [m["image_name"] for m in cell_metadata]})
            adata = ad.AnnData(x, obs=obs)
            adata.obs_names = [f"cell_{i}" for i in range(adata.n_obs)]
            adata.uns["embedding_fingerprint"] = FINGERPRINT
            key = self._compute_umap_and_clustering(adata, cluster_resolution, None, n_neighbors)
            os.makedirs(out, exist_ok=True)
            adata.write(path)
            return {"summary": "stub", "cell_count": int(adata.n_obs), "adata_path": path, "cluster_key": key,
                    "cluster_resolution": cluster_resolution, "n_neighbors": n_neighbors, "seconds": 1.0,
                    "training": {"train_seconds": 0.0}}

    return StubAnalyzer


def _executor(qcd: Path, tools: dict, engine=None):
    from shape.agent.executor import Executor

    executor = Executor("unused-model", query_cache_dir=str(qcd),
                        initializer=SimpleNamespace(_tool_classes_cache=tools))
    executor.enable_llm_verifier = engine is not None
    executor._verifier_llm_mm, executor._verifier_llm_tried = engine, True     # the stub engine, never built
    return executor


class _EmptyPlan:
    """A reasoner whose round-1 plan has no moves (the input needs no tuning); records each question."""

    def __init__(self):
        self.asked = []

    def generate(self, content, response_format=None):
        # The round's plan schema is the verifier's (`plan_schema`: the analyzer's BasedPlan since A12).
        self.asked.append((response_format.__name__, content[0]))
        return {"content": response_format(analysis="the frozen embedding is good enough", moves=[])}


@pytest.fixture()
def no_wait(monkeypatch):
    from shape.agent.verification import base

    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)


# ---- gap 2 (+ 4, 6): the analyzer's agentic VRR on the solve() path --------------------------------------------

def test_the_analyzer_is_tuned_by_the_agentic_vrr_on_the_solve_path_and_the_linear_loop_stays_the_stages():
    from shape.agent.verification.stage_map import stage_by_name, verifier_cls_for
    from shape.agent.verification.verifiers.representation import (RepresentationAgenticVerifier,
                                                                     RepresentationVerifier)

    spec = stage_by_name("representation")
    assert verifier_cls_for(spec, TOOL) is RepresentationAgenticVerifier
    assert spec.verifier_cls is RepresentationVerifier                  # the stage's own (linear) loop


def test_trial_0_is_the_frozen_backbone_on_the_kept_crops_and_a_failed_gate_delivers_it(tmp_path, no_wait):
    """Images with plate and well but no site: the planner's trained depth is not used, trial 0 runs zero_shot on
    the crops the tool's loader returns (the rejected crop left out, the identifiers the specs gave and no site),
    the gate skips (no replicate tier) and trial 0's h5ad is delivered at <qcd>/cell_state_analysis/; the
    morphology stage re-clusters it without a second embedding."""
    from shape.agent.verification import integration

    qcd = tmp_path / "cache"
    images = [{"image_id": f"P{p}_{w}", "technical": {"plate": f"P{p}", "well": w}}
              for p in (1, 2) for w in ("A01", "A02", "A03", "A04")]
    crops = _crop_cache(qcd, images, rejected={str(qcd / "tool_cache" / "P1_A01_crop_001.png")})
    calls = []
    executor = _executor(qcd, {TOOL: _stub_analyzer(calls)})
    resolutions, result, status = integration.verify_managed_tool(
        executor, TOOL, integration.build_base_kwargs(TOOL, images[0], None),
        {"adaptation_depth": "trained", "learning_rate": 3e-5, "leiden_resolution": 0.5, "n_neighbors": 15},
        request="Compare the cell states\nof these wells.")
    assert len(calls) == 1                                              # trial 0 only; the morphology reuses it
    trial0 = calls[0]
    assert trial0["adaptation_depth"] == "zero_shot" and trial0["output_subdir"] == "ssl_calls/input"
    assert trial0["val_fraction"] == 0.1 and trial0["split_key"] is None     # the tool's val wells
    expected = [p for i in images for p in crops[i["image_id"]] if not p.endswith("P1_A01_crop_001.png")]
    assert trial0["cell_crops"] == expected                             # gap 4: the rejected crop never reaches it
    assert all(m["plate"] and m["well"] and "site" not in m for m in trial0["cell_metadata"])   # nothing invented
    rep, morph = resolutions
    assert (rep.status, rep.reason) == ("satisfied", "not_measurable") and rep.details["call"] == "input"
    assert "no adopted proxy" in rep.details["admission"]["reasons"][0]
    canonical = qcd / "cell_state_analysis" / "cell_state_analyzed.h5ad"
    assert canonical.exists() and rep.result["adata_path"] == str(canonical)      # gap 6: inside the query cache
    assert not (tmp_path / "cell_state_analyzed.h5ad").exists()
    # the morphology stage tunes its own levers (filled from the planner's) on trial 0's embedding
    assert {"leiden_resolution", "n_neighbors"} <= set(morph.params) and morph.params["adaptation_depth"] == "zero_shot"
    assert morph.trials[0][0]["leiden_resolution"] == 0.5
    assert status == morph.status and isinstance(result, dict) and result.get("adata_path")


def test_a_measurable_set_asks_the_agent_with_the_request_as_its_brief(tmp_path, no_wait):
    """Plate, well and site on every crop, 2 sites per well: the gate runs the search; round 1 is the shared round
    prompt with the request as the brief and the tool's val wells; an empty plan delivers trial 0."""
    from shape.agent.verification import integration

    qcd = tmp_path / "cache"
    images = [{"image_id": f"P{p}_{w}_s{s}", "technical": {"plate": f"P{p}", "well": w, "site": f"{w}_s{s}"}}
              for p in (1, 2) for w in ("A01", "A02", "A03", "A04", "A05") for s in (1, 2)]
    _crop_cache(qcd, images)
    calls, engine = [], _EmptyPlan()
    executor = _executor(qcd, {TOOL: _stub_analyzer(calls)}, engine=engine)
    resolutions, _, _ = integration.verify_managed_tool(
        executor, TOOL, {}, {"adaptation_depth": "trained"}, request="Compare the cell states of these wells.")
    rep = resolutions[0]
    assert rep.details["admission"]["decision"] == "RUN" and rep.details["admission"]["tier"] == "replicate"
    assert (rep.status, rep.reason, rep.details["call"]) == ("satisfied", "no_tuning_needed", "input")
    assert [name for name, _ in engine.asked] == ["BasedPlan", "BasedPlan"] and len(calls) == 1
    assert rep.details["stop_checks"] == 1 and rep.details["execution_errors"] == []
    assert rep.details["fallback_rounds"] == rep.details["reasoner_failures"] == 0
    prompt = engine.asked[0][1]
    assert "Task: Compare the cell states of these wells." in prompt
    assert rep.details["split"]["val_wells_per_plate"] == {"P1": 2, "P2": 2}


# ---- gap 4: the cropper's delivered call carries its kept-crops manifest to the analyzer's loader ---------------

def test_the_delivered_crop_call_brings_its_manifest_and_the_analyzer_leaves_the_rejected_crops_out(tmp_path):
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier

    qcd = tmp_path / "cache"
    call_dir = qcd / "tool_cache" / "crop_calls" / "c01"
    call_dir.mkdir(parents=True)
    crops = []
    for c in range(3):
        path = call_dir / f"img_crop_{c}.png"
        path.write_bytes(b"crop")
        crops.append(str(path))
    metadata = call_dir / "cell_crops_metadata_abc.json"
    metadata.write_text(json.dumps({
        "source_image_id": "img", "execution_status": "success", "cell_crops_paths": crops,
        "cell_metadata": [{"cell_id": f"c{c}", "crop_path": p} for c, p in enumerate(crops)]}), encoding="utf-8")
    (call_dir / "cell_crops_kept_manifest_abc.json").write_text(json.dumps({"kept_crop_paths": crops[:2]}),
                                                                encoding="utf-8")
    result, details = {"cell_crops_metadata_path": str(metadata)}, {}
    state = SimpleNamespace(trials=[({}, None, {}), ({"output_subdir": "crop_calls/c01"}, None, result)])
    CropAgenticVerifier().deliver(state, 1, details)
    assert Path(details["delivered_metadata_path"]).parent == qcd / "tool_cache"
    loaded, meta, _ = Cell_State_Analyzer_Tool()._load_cell_data_from_metadata(str(qcd))
    assert loaded == crops[:2] and [m["cell_id"] for m in meta] == ["c0", "c1"]


# ---- gap 6: the representation's delivery never links above the query cache -----------------------------------

def test_the_delivered_h5ad_resolves_up_its_output_subdir_only_when_it_lies_there(tmp_path):
    from shape.agent.verification.verifiers.representation import RepresentationAgenticVerifier

    qcd = tmp_path / "cache"
    inside = qcd / "cell_state_analysis" / "ssl_calls" / "input" / "cell_state_analyzed.h5ad"
    elsewhere = qcd / "cell_state_analysis" / "cell_state_analyzed.h5ad"   # a tool that ignored output_subdir
    for path in (inside, elsewhere):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(b"h5ad")
    verifier, params = RepresentationAgenticVerifier(), {"output_subdir": "ssl_calls/input"}
    details = {}
    verifier.deliver(SimpleNamespace(trials=[(params, None, {"adata_path": str(inside)})]), 0, details)
    assert details["delivered"]["adata_path"] == str(elsewhere)
    details = {}
    verifier.deliver(SimpleNamespace(trials=[(params, None, {"adata_path": str(elsewhere)})]), 0, details)
    assert details["delivered"]["adata_path"] == str(elsewhere)          # itself, never <tmp_path>/...
    assert not (tmp_path / "cell_state_analyzed.h5ad").exists()


# ---- no fallback: a managed step that raises is a hard error with its cause --------------------------------

class _Recorder:
    def __init__(self):
        self.actions = []

    def add_action(self, **kwargs):
        self.actions.append(kwargs)

    def get_actions(self, llm_safe=False):
        return list(self.actions)


def _step_solver(llm_path):
    executor = SimpleNamespace(
        generate_tool_command=lambda **kw: llm_path.append(kw) or SimpleNamespace(command="tool.execute()"),
        execute_tool_command=lambda **kw: {"summary": "llm path"})
    return {"planner": SimpleNamespace(recommended_params={"diameter": 30.0}), "executor": executor,
            "initializer": SimpleNamespace(get_tool_metadata=lambda name: {}), "available_tools": []}


def _execute(solver, memory, images, tool="Cell_Segmenter_Tool"):
    from shape.agent.verifier import Verdict, Verifier
    from shape.solver import _execute_and_verify

    step = SimpleNamespace(tool_name=tool, sub_goal="segment", context="", justification="")
    return _execute_and_verify(solver, step, 1, "segment the cells", images[0]["path"], memory, True, {},
                               Verifier(), Verdict(scope="plan", status="pass"), images=images)


def _images(n=2):
    return [{"path": f"img{i}.png", "image_id": f"img{i}", "group": "default"} for i in range(n)]


def test_an_error_in_the_managed_path_is_a_hard_error_never_the_llm_command(monkeypatch):
    from shape.agent.verification import integration

    def boom(*args, **kwargs):
        raise RuntimeError("crop: the round question failed 3 times; last error: TimeoutError: gateway down")

    monkeypatch.setattr(integration, "verify_managed_tool", boom)
    llm_path, memory = [], _Recorder()
    verdict = _execute(_step_solver(llm_path), memory, _images())
    assert llm_path == [] and verdict.status == "fail" and verdict.stop_reason == "hard_error"
    error = memory.actions[0]["result"]
    assert error["status"] == "failed" and error["error"].startswith("Cell_Segmenter_Tool failed: RuntimeError: crop:")
    assert "TimeoutError: gateway down" in error["error"]                     # the cause


def test_a_failed_batch_follow_up_is_a_hard_error_recorded_once(monkeypatch):
    import shape.solver as solver_module
    from shape.agent.verification import batch_stage

    def recorded(solver, step, memory, request=None, picked=None):        # run_followup records, then raises
        memory.add_action(step_count=step, tool_name=batch_stage.TOOL, sub_goal="b", command="c",
                          result={"status": "failed", "error": "ValueError: bad labels"})
        raise ValueError("bad labels")

    def unrecorded(solver, step, memory, request=None, picked=None):
        raise KeyError("metadata_profile")

    for follow_up, n_actions in ((recorded, 1), (unrecorded, 1)):
        monkeypatch.setattr(solver_module, "_batch_followup", follow_up)
        memory = _Recorder()
        verdict = solver_module._batch_step({}, 3, memory)
        assert verdict.stop_reason == "hard_error" and len(memory.actions) == n_actions
        assert memory.actions[0]["tool_name"] == batch_stage.TOOL and memory.actions[0]["result"]["status"] == "failed"
    monkeypatch.setattr(solver_module, "_batch_followup", lambda *a, **k: None)
    assert solver_module._batch_step({}, 3, _Recorder()) is None


# ---- gap 5: image_workers images at a time, the GPU tool calls take turns -------------------------------------

def test_image_workers_runs_the_images_at_once_in_order_and_one_after_the_other_by_default():
    from shape.solver import _run_images

    specs = _images(3)
    barrier = threading.Barrier(3, timeout=10)

    def at_once(spec):
        barrier.wait()                                                  # passes only when all three run at once
        return spec["image_id"]

    executor = SimpleNamespace()
    solver = {"image_workers": 3, "executor": executor}
    assert _run_images(solver, "Cell_Segmenter_Tool", specs, at_once) == ["img0", "img1", "img2"]
    assert set(executor.gpu_slots) == {"Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"}
    order = []
    assert _run_images({"executor": SimpleNamespace()}, "Cell_Segmenter_Tool", specs,
                       lambda s: order.append(threading.current_thread().name) or s["image_id"]) == [
        "img0", "img1", "img2"]
    assert len(set(order)) == 1                                         # default: this thread, one after the other


def test_the_gpu_tool_calls_of_concurrent_images_take_turns_and_the_cropper_does_not(tmp_path, monkeypatch):
    from shape.solver import _gpu_slots

    monkeypatch.delenv("SHAPE_SAM_SERVER_URL", raising=False)
    running, peak, lock = {"n": 0}, {}, threading.Lock()

    def tool_class(name):
        class Tool:
            def execute(self, image=None, **kwargs):
                with lock:
                    running["n"] += 1
                    peak[name] = max(peak.get(name, 0), running["n"])
                time.sleep(0.2)
                with lock:
                    running["n"] -= 1
                return {"ok": image}
        return Tool

    executor = _executor(tmp_path, {"Cell_Segmenter_Tool": tool_class("Cell_Segmenter_Tool"),
                                    "Organoid_Segmenter_Tool": tool_class("Organoid_Segmenter_Tool"),
                                    "Single_Cell_Cropper_Tool": tool_class("Single_Cell_Cropper_Tool")})
    executor.gpu_slots = _gpu_slots()
    for name in ("Cell_Segmenter_Tool", "Organoid_Segmenter_Tool", "Single_Cell_Cropper_Tool"):
        threads = [threading.Thread(target=executor.run_managed_tool, args=(name, {"image": f"i{i}"}, {}))
                   for i in range(3)]
        for t in threads:
            t.start()
        for t in threads:
            t.join()
    assert peak == {"Cell_Segmenter_Tool": 1, "Organoid_Segmenter_Tool": 1, "Single_Cell_Cropper_Tool": 3}


def test_a_multi_image_step_with_image_workers_runs_its_images_at_once_and_records_them_in_order(monkeypatch):
    from shape.agent.verification import integration
    from shape.utils.schemas import StageResolution, VLRRResult

    barrier = threading.Barrier(2, timeout=10)

    def verify(executor, tool_name, base_kwargs, *args, **kwargs):
        barrier.wait()
        res = StageResolution(result={"mask_path": f"{base_kwargs['image_id']}.npy"}, status="satisfied", params={},
                              verdict=VLRRResult("PASS", 1.0, {}, action="accept"), reason="agent_choice")
        return [res], res.result, res.status

    monkeypatch.setattr(integration, "verify_managed_tool", verify)
    solver, memory = dict(_step_solver([]), image_workers=2), _Recorder()
    solver["executor"].gpu_slots = {}
    verdict = _execute(solver, memory, _images(2))
    assert verdict.stop_reason is None
    assert [e["image_id"] for e in memory.actions[0]["result"]["per_image"]] == ["img0", "img1"]


# ---- no engine: the per-image vision calls raise (a system error), never silently skipped ----------------------

def test_the_per_image_vision_calls_raise_without_an_engine(tmp_path):
    from shape.agent.verification.crop_vision import crop_vision_profile
    from shape.agent.verification.verifiers.segmentation import SegmentationVerifier
    from shape.tools.cell_segmenter import color_class_filter as ccf

    with pytest.raises(RuntimeError, match="color-class decide: no vision engine is configured"):
        ccf.decide(None, np.zeros((8, 8, 3), np.uint8), np.zeros((8, 8), np.int32))
    with pytest.raises(RuntimeError, match="crop-vision profile: no vision engine is configured"):
        crop_vision_profile([], {}, engine=None)
    metrics = {"cell_count": 20.0, "flow_error": 0.1, "tiny_mask_fraction": 0.0, "large_mask_fraction": 0.0,
               "mean_solidity": 0.95, "border_mask_fraction": 0.0, "mask_area_median": 300.0,
               "missed_foreground_fraction": 0.0, "count_miss_fraction": 0.0, "missed_blob_size_ratio": 1.0}
    # The legacy linear verifier permits deterministic-only use. An actual vision question requires an engine.
    from shape.agent.verification.llm_recommend import vision_recommend_segmentation
    from shape.utils.schemas import VLRRResult
    with pytest.raises(RuntimeError, match="no vision engine is configured"):
        vision_recommend_segmentation(None, {}, dict(metrics), VLRRResult("PASS", 1.0, {}))
    quiet = SegmentationVerifier()
    quiet.uses_llm = False                                    # the explicit opt-out: the deterministic floor
    assert quiet.adjudicate(dict(metrics), "this_step", [], {}).action in ("accept", "retune_here")


# ---- the segmentation cohort on the solve path (`segmentation_cohort`) ------------------------------------------

def _stage(image_id):
    from shape.utils.schemas import StageResolution, VLRRResult
    res = StageResolution(result={"mask_path": f"{image_id}.npy"}, status="satisfied", params={},
                          verdict=VLRRResult("PASS", 1.0, {}, action="accept"), reason="agent_choice")
    return [res], res.result, res.status


@pytest.mark.parametrize(("tool", "diameter"), [("Cell_Segmenter_Tool", 30), ("Organoid_Segmenter_Tool", None)])
def test_a_multi_image_segmentation_step_runs_as_a_cohort_and_records_it(monkeypatch, tool, diameter):
    from shape.agent.verification import integration, segmentation_cohort

    seen = {}

    def run(specs, run_one, map_images, read_profile, defaults, levers, workers, *, tool_name):
        seen.update(defaults=defaults, levers=list(levers), workers=workers, tool_name=tool_name)
        runs = [run_one(s, {"cohort_levers": {"cellprob_threshold": -1.0}}) for s in specs]
        return runs, {"parameter_source": {s["image_id"]: "cohort" for s in specs}, "counts": {"cohort": len(specs)}}

    options = []
    monkeypatch.setattr(segmentation_cohort, "run", run)
    monkeypatch.setattr(integration, "verify_managed_tool",
                        lambda executor, tool, base, *a, **k: options.append(k["verifier_options"]) or
                        _stage(base["image_id"]))
    memory = _Recorder()
    verdict = _execute(_step_solver([]), memory, _images(3), tool=tool)
    assert verdict.stop_reason is None and options == [{"cohort_levers": {"cellprob_threshold": -1.0}}] * 3
    action = memory.actions[0]
    assert action["verification"]["cohort"]["counts"] == {"cohort": 3}
    assert [e["parameter_source"] for e in action["result"]["per_image"]] == ["cohort"] * 3
    assert seen["defaults"]["diameter"] == diameter and "cellprob_threshold" in seen["levers"]
    assert seen["tool_name"] == tool
    if tool == "Organoid_Segmenter_Tool":
        assert "color_class_filter" not in seen["levers"]
    assert seen["workers"] == 1                          # the run's image_workers (solve() default 1)


def test_the_cohort_off_or_a_single_image_runs_every_image_its_own_vrr(monkeypatch):
    from shape.agent.verification import integration, segmentation_cohort

    monkeypatch.setattr(segmentation_cohort, "run", lambda *a, **k: pytest.fail("the cohort ran"))
    options = []
    monkeypatch.setattr(integration, "verify_managed_tool",
                        lambda executor, tool, base, *a, **k: options.append(k["verifier_options"]) or
                        _stage(base["image_id"]))
    memory = _Recorder()
    _execute(dict(_step_solver([]), segmentation_cohort=False), memory, _images(2))
    assert options == [None, None] and "cohort" not in memory.actions[0]["verification"]
    assert "parameter_source" not in memory.actions[0]["result"]["per_image"][0]
    _execute(dict(_step_solver([]), segmentation_cohort=True), _Recorder(), _images(1))     # one image: no cohort
    _execute(dict(_step_solver([]), cropping_cohort=False), _Recorder(), _images(2),
             tool="Single_Cell_Cropper_Tool")                                               # the cropper, cohort off


def test_the_cohort_groups_by_the_preflight_read_with_the_runs_engine(monkeypatch):
    from shape.agent.verification import image_context, integration
    from shape.solver import _segmentation_cohort
    from shape.utils.image_processor import ImageProcessor

    reads = []
    monkeypatch.setattr(ImageProcessor, "load_image", staticmethod(lambda path: path))
    monkeypatch.setattr(image_context, "read_image_context_profile",
                        lambda img, engine: reads.append((img, engine)) or SimpleNamespace(
                            specimen_context="cell_culture", modality="fluorescence", confidence="high"))
    executor = SimpleNamespace(verifier_llm_engine_mm="ENGINE", enable_llm_verifier=True)
    runs, record = _segmentation_cohort({"executor": executor}, _images(3), lambda spec, options: _stage(spec["image_id"]))
    assert reads == [(f"img{i}.png", "ENGINE") for i in range(3)] and len(runs) == 3
    assert record["groups"][0]["group"] == {"specimen_context": "cell_culture", "modality": "fluorescence"}
    assert record["groups"][0]["routing"] == "per_image_small_group"             # 3 images: below 10

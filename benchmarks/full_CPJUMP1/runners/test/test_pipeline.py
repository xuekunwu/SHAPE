import json
import time
from pathlib import Path
import numpy as np
import pandas as pd
import pytest
import tifffile

from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
from benchmarks.full_CPJUMP1.harness.pipeline import (
    assert_search_clean, calibration_view, cropping, read_embedding, shape_search_policy,
)


CLEAN = {"admission": {"decision": "SKIP"}, "reasoner_failures": 0, "fallback_rounds": 0, "execution_errors": []}


@pytest.mark.parametrize("bad", [{"fallback_rounds": 1}, {"status": "failed"}, {"status": "not_started"},
                                 {"stop": True}])
def test_a_rule_fallback_or_failed_search_is_never_valid_delivery(bad):
    assert assert_search_clean(CLEAN) is None
    with pytest.raises(RuntimeError):
        assert_search_clean({**CLEAN, **bad})


@pytest.mark.parametrize("failed", [
    {"reasoner_failures": 1, "stop_reason": "invalid_search_plan",
     "execution_errors": [{"question": "search_repair", "cause": "deferred parameters must be known dimensions"}]},
    {"stop_reason": "execution_error"},
    {"execution_errors": [{"question": "mask_check", "cause": "vision call failed"}]},
])
def test_failed_in_run_calls_are_recorded_not_fatal(failed):
    warning = assert_search_clean({**CLEAN, **failed})
    assert warning == {k: {**CLEAN, **failed}.get(k) for k in ("reasoner_failures", "execution_errors", "stop_reason")}


def test_missing_health_counters_are_not_zero():
    with pytest.raises(RuntimeError, match="missing"):
        assert_search_clean({"admission": {"decision": "SKIP"}})


def test_smoke_limits_shape_search_without_changing_full_policy():
    assert shape_search_policy({}, 1000) == (1000, None)
    policy = {"smoke_policy": {"shape_search_images": 1, "shape_search_rounds": 2}}
    assert shape_search_policy(policy, 20) == (1, 2)
    with pytest.raises(ValueError, match="at least two"):
        shape_search_policy({"smoke_policy": {"shape_search_rounds": 1}}, 20)


def test_full_shape_segmentation_runs_the_shared_cohort(tmp_path, monkeypatch):
    """The full SHAPE arm segments through SHAPE's shared cohort (segmentation_cohort.run): one group, a
    calibration sample with its own VRR, the consensus in the cohort mode for the rest, only the flagged image
    routed to its own VRR from the consensus; the records compare.segment_routed reads."""
    from types import SimpleNamespace
    import threading
    import benchmarks.full_HCMA.harness.shape_arm as hcma
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks.full_CPJUMP1.harness.compare import segment_routed
    from shape.agent.verification import segmentation_cohort
    from shape.tools.cell_segmenter import tool as segmenter_module

    prep, run = tmp_path / "prep", tmp_path / "run"
    (prep / "public").mkdir(parents=True)
    (run / "logs").mkdir(parents=True)
    rows = []
    for i in range(12):
        name = f"img_{i:02d}.ome.tif"
        tifffile.imwrite(prep / "public" / name, np.zeros((2, 8, 8), np.uint16), photometric="minisblack")
        rows.append({"image_id": f"img_{i:02d}", "segment_file": name})
    ids = [r["image_id"] for r in rows]
    key = ("cell_culture", "fluorescence")
    sample = segmentation_cohort.draw(ids, segmentation_cohort.calibration_size(12), key)
    flagged = next(i for i in ids if i not in sample)

    class FakeSegmenter:
        device = type("Device", (), {"type": "cuda"})()

        def execute(self, **kwargs):
            mask = Path(kwargs["query_cache_dir"]) / "mask.npy"
            mask.parent.mkdir(parents=True, exist_ok=True)
            np.save(mask, np.ones((1080, 1080), np.int32))
            return {"mask_path": str(mask), "overlay_path": None, "cell_count": 1}

    calls, lock = [], threading.Lock()

    def segment_resolution(C, tool, item, levers, defaults, engine, scratch, log_dir, model_lock,
                           verifier_options=None):
        lever = next(k for k in levers if k not in segmentation_cohort.MEASURED_LEVERS
                     and isinstance(defaults[k], float))
        options = verifier_options or {}
        mode = "cohort" if "cohort_levers" in options else "routed" if "start_params" in options else "search"
        with lock:
            calls.append((item["image_id"], mode, dict(options)))
        changed = {lever: defaults[lever] + 0.1} if mode != "cohort" else dict(options["cohort_levers"])
        bad = mode == "cohort" and item["image_id"] == flagged
        mask = Path(scratch) / "selected.npy"
        mask.parent.mkdir(parents=True, exist_ok=True)
        np.save(mask, np.full((1080, 1080), 2, np.int32))
        res = SimpleNamespace(details={"admission": {"decision": "RUN"}, "call": "c1",
                                       "measured_problems": {"c1": ["bad"] if bad else []}},
                              params={**defaults, **changed}, result={"mask_path": str(mask)}, status="satisfied",
                              reason="satisfied", fault=None)
        record = {"status": "searched", "levers": changed, "mask_npy": str(mask), "n_calls": 1,
                  "reasoner_failures": 0, "fallback_rounds": 0, "execution_errors": [], "stop_reason": "satisfied"}
        return record, res

    monkeypatch.setattr(segmenter_module, "Cell_Segmenter_Tool", FakeSegmenter)
    monkeypatch.setattr(hcma, "segment_resolution", segment_resolution)
    monkeypatch.setattr(segmentation_cohort, "preflight_read", lambda path, engine, image_id=None: SimpleNamespace(
        specimen_context=key[0], modality=key[1], confidence="high"))
    config = {"arm": "shape", "deadline": time.time() + 120, "cores": 4}
    pipeline._segmentation_cohort(prep, run, config, rows, object())

    cohort = read_json(run / "segment_COHORT.json")
    assert cohort["workers"] == 4 and cohort["groups"][0]["calibration"] == sample
    assert cohort["groups"][0]["routed"] == [flagged]
    source = {r["image_id"]: r["parameter_source"] for r in read_json(run / "segmentation.json")}
    assert source == cohort["parameter_source"]
    assert {i for i, s in source.items() if s == "calibration"} == set(sample) and source[flagged] == "per_image_vrr"
    consensus = cohort["groups"][0]["cohort_levers"]
    assert consensus and [c for c in calls if c[1] == "routed"] == [(flagged, "routed", {"start_params": consensus})]
    assert {c[0] for c in calls if c[1] == "cohort"} == set(ids) - set(sample)
    artifacts = {p.relative_to(run).as_posix() for p in run.rglob("*") if p.is_file()}
    assert segment_routed(run, ids, artifacts) == set(sample) | {flagged}
    assert {p.parent.name for p in (run / "segment").glob("*/SEARCH.json")} == set(sample) | {flagged}
    assert all((run / "segment" / i / f"cell_mask_{i}.tif").is_file() for i in ids)


def _crop_fixture(tmp_path, monkeypatch, n_images, record_for):
    """n images and masks, a stub cropper and a stub crop_one (record_for(image_id) -> record changes)."""
    import threading
    import benchmarks.full_HCMA.harness.shape_arm as hcma
    from benchmarks.full_CPJUMP1.harness import pipeline
    from shape.tools.single_cell_cropper import tool as cropper_module

    prep, run = tmp_path / "prep", tmp_path / "run"
    (prep / "public").mkdir(parents=True)
    (run / "logs").mkdir(parents=True)
    rows, masks = [], []
    for i in range(n_images):
        ident, name = f"img_{i}", f"img_{i}.ome.tif"
        tifffile.imwrite(prep / "public" / name, np.zeros((5, 32, 32), np.uint16), photometric="minisblack")
        mask = tmp_path / f"mask_{i}.tif"
        tifffile.imwrite(mask, np.ones((32, 32), np.uint16))
        rows.append({"image_id": ident, "file": name, "plate": f"p{i % 2}", "well": f"w{i}",
                     "usage": "evaluation_only", "negative_control": False})
        masks.append({"image_id": ident, "mask": str(mask), "sha256": digest(mask), "min_area": 15})
    write_json(run / "segmentation.json", masks)

    class FakeCropper:
        def execute(self, **kwargs):
            ident = kwargs["source_image_id"]
            folder = Path(kwargs["query_cache_dir"])
            folder.mkdir(parents=True, exist_ok=True)
            display = folder / f"{ident}.tif"
            tifffile.imwrite(display, np.zeros((5, 16, 16), np.uint16), photometric="minisblack")
            metadata = folder / f"{ident}.json"
            write_json(metadata, {"cell_metadata": [{"cell_id": f"{ident}_c{k}", "crop_path": str(display),
                        "area": 100, "crop_bbox": [0, 0, 16, 16]} for k in range(2)]})
            return {"cell_crops_metadata_path": str(metadata)}

    state = {"active": 0, "peak": 0, "calls": []}
    lock = threading.Lock()

    def crop_one(C, tool_class, item, engine, deadline, cores, scratch, log_dir, budget=None):
        with lock:
            state["active"] += 1
            state["peak"] = max(state["peak"], state["active"])
            state["calls"].append((item["image_id"], cores, str(scratch)))
        time.sleep(0.2 if item["image_id"] == "img_0" else 0.05)        # the first image finishes last
        with lock:
            state["active"] -= 1
        return {"status": "searched", "delivered": "agent_1", "levers": {}, "cell_overrides": {},
                "rejected_tool_cells": [f"{item['image_id']}_c1"], "boxes": {}, "n_calls": 2,
                "reasoner_failures": 0, "fallback_rounds": 0, "execution_errors": [], "stop_reason": "satisfied",
                **record_for(item["image_id"])}

    monkeypatch.setattr(cropper_module, "Single_Cell_Cropper_Tool", FakeCropper)
    monkeypatch.setattr(hcma, "crop_one", crop_one)
    monkeypatch.setattr(pipeline, "engine_for", lambda config, stage: object())
    return pipeline, prep, run, rows, state


def test_every_crop_runs_its_own_vrr_in_parallel_in_image_order(tmp_path, monkeypatch):
    invalid = {"reasoner_failures": 1, "stop_reason": "invalid_search_plan",
               "execution_errors": [{"question": "search_repair", "cause": "unknown focus dimensions"}]}
    pipeline, prep, run, rows, state = _crop_fixture(
        tmp_path, monkeypatch, 6, lambda ident: invalid if ident == "img_3" else {})
    config = {"arm": "shape", "deadline": time.time() + 120, "cores": 6,
              "shape_policy": {"crop_vrr_workers": 3}}
    pipeline.cropping(prep, run, config, rows, threads=True)            # the stubs live in this process
    assert sorted(i for i, _, _ in state["calls"]) == [r["image_id"] for r in rows]      # no cohort, no calibration
    assert all(cores == 2 and "calibration" not in scratch for _, cores, scratch in state["calls"])
    assert 1 < state["peak"] <= 3
    assert [c["cell_id"] for c in read_json(run / "cells.json")] == [f"img_{i}_c0" for i in range(6)]
    assert read_json(run / "excluded_cells.json") == [f"img_{i}_c1" for i in range(6)]
    assert all((run / "crop" / r["image_id"] / "SEARCH.json").is_file() for r in rows)
    lines = [json.loads(x) for x in (run / "SEARCH_WARNINGS.jsonl").read_text(encoding="utf-8").splitlines()]
    assert lines == [{"stage": "crop", "image_id": "img_3", **invalid}]


def test_crop_workers_never_exceed_the_cores(tmp_path, monkeypatch):
    pipeline, prep, run, rows, state = _crop_fixture(tmp_path, monkeypatch, 4, lambda ident: {})
    config = {"arm": "shape", "deadline": time.time() + 120, "cores": 2, "shape_policy": {"crop_vrr_workers": 16}}
    pipeline.cropping(prep, run, config, rows, threads=True)            # the stubs live in this process
    assert state["peak"] <= 2 and {cores for _, cores, _ in state["calls"]} == {1}
    assert not (run / "SEARCH_WARNINGS.jsonl").exists()


def test_a_crop_search_with_a_fallback_round_is_fatal(tmp_path, monkeypatch):
    pipeline, prep, run, rows, _ = _crop_fixture(
        tmp_path, monkeypatch, 3, lambda ident: {"fallback_rounds": 1} if ident == "img_1" else {})
    config = {"arm": "shape", "deadline": time.time() + 120, "cores": 3, "shape_policy": {"crop_vrr_workers": 3}}
    with pytest.raises(RuntimeError, match="fallback_rounds"):
        pipeline.cropping(prep, run, config, rows, threads=True)            # the stubs live in this process
    assert not (run / "cells.json").exists()


def test_batch_view_only_calibration_and_negatives():
    meta = pd.DataFrame({"well": ["a", "b", "c"], "plate": ["p1", "p2", "p2"],
                         "calibration_group": ["reference_001", "", ""], "negative_control": [False, False, True]})
    view = calibration_view(meta)
    assert view.rows.tolist() == [0, 2]
    assert view.meta.Metadata_JCP2022.tolist() == ["reference_001", "DMSO"]
    assert view.meta.Metadata_PertType.tolist() == ["trt", "negcon"]
    assert view.meta.Metadata_Plate.tolist() == ["p1", "p2"]


def test_embedding_reorders_ids_and_rejects_duplicates(tmp_path):
    import anndata as ad
    p = tmp_path / "data.h5ad"
    data = ad.AnnData(np.array([[2., 3.], [0., 1.]], dtype=np.float32), obs=pd.DataFrame({"cell_id": ["b", "a"]}))
    data.write_h5ad(p)
    np.testing.assert_array_equal(read_embedding({"adata_path": str(p)}, [{"cell_id": "a"}, {"cell_id": "b"}]), [[0, 1], [2, 3]])
    data.obs["cell_id"] = ["a", "a"]
    data.write_h5ad(p)
    with pytest.raises(ValueError, match="identities"):
        read_embedding({"adata_path": str(p)}, [{"cell_id": "a"}, {"cell_id": "b"}])


def test_real_cropper_keeps_five_channels_and_cell_margin(tmp_path):
    prep, run = tmp_path / "prep", tmp_path / "run"
    (prep / "public").mkdir(parents=True)
    run.mkdir()
    image = np.stack([np.full((64, 64), 4000 + i * 9000, np.uint16) for i in range(5)])
    tifffile.imwrite(prep / "public" / "image.ome.tif", image, ome=True, photometric="minisblack", metadata={"axes": "CYX"})
    mask = np.zeros((64, 64), np.uint32)
    mask[20:38, 20:38] = 1
    p = run / "cell_mask_img_0000.tif"
    tifffile.imwrite(p, mask)
    write_json(run / "segmentation.json", [{"image_id": "img_0000", "mask": str(p), "sha256": digest(p), "min_area": 15}])
    images = [{"image_id": "img_0000", "file": "image.ome.tif", "plate": "p1", "well": "w1", "usage": "evaluation_only"}]
    cropping(prep, run, {"arm": "control", "deadline": time.time() + 120, "cores": 1}, images)
    cells = read_json(run / "cells.json")
    assert len(cells) == 1
    data = tifffile.imread(cells[0]["crop_path"])
    assert data.shape[0] == 5 and data.dtype == np.uint16
    assert data.shape[1] < 50  # generic mask names would silently select a 25-pixel margin
    for i in range(5):
        assert np.all(data[i] == 4000 + i * 9000)


def test_mask_check_rejections_removed_before_embedding(tmp_path, monkeypatch):
    import benchmarks.full_HCMA.harness.shape_arm as hcma
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
    from benchmarks.full_CPJUMP1.harness import pipeline
    prep, run = tmp_path / "prep", tmp_path / "run"
    (prep / "public").mkdir(parents=True)
    run.mkdir()
    crop_paths = []
    tifffile.imwrite(prep / "public" / "image.tif", np.zeros((5, 32, 32), np.uint16), photometric="minisblack")
    for i in range(2):
        p = tmp_path / f"crop{i}.tif"
        tifffile.imwrite(p, np.zeros((5, 16, 16), np.uint16), photometric="minisblack")
        crop_paths.append(p)
    metadata = tmp_path / "crop_metadata.json"
    write_json(metadata, {"cell_metadata": [{"cell_id": f"c{i}", "crop_path": str(p), "area": 100,
                                           "crop_bbox": [0, 0, 16, 16]} for i, p in enumerate(crop_paths)]})
    monkeypatch.setattr(pipeline, "engine_for", lambda cfg, stage: object())
    monkeypatch.setattr(hcma, "crop_one", lambda *a, **kw: {"levers": {}, "cell_overrides": {},
                       "rejected_tool_cells": ["c0"], "boxes": {}, "reasoner_failures": 0,
                       "fallback_rounds": 0, "execution_errors": []})
    monkeypatch.setattr(Single_Cell_Cropper_Tool, "execute", lambda *a, **kw: {"cell_crops_metadata_path": str(metadata)})
    mask = tmp_path / "cell_mask_i.tif"
    tifffile.imwrite(mask, np.ones((32, 32), dtype=np.uint16))
    write_json(run / "segmentation.json", [{"image_id": "i", "mask": str(mask), "sha256": digest(mask), "min_area": 15}])
    cropping(prep, run, {"arm": "shape", "deadline": time.time() + 120, "cores": 1},
             [{"image_id": "i", "file": "image.tif", "plate": "p", "well": "w", "usage": "evaluation_only"}])
    assert [c["cell_id"] for c in read_json(run / "cells.json")] == ["c1"]
    assert read_json(run / "excluded_cells.json") == ["c0"]


@pytest.mark.parametrize("trained", [False, True])
@pytest.mark.parametrize("arm", ["shape", "codex"])
def test_ssl_fit_development_only_then_frozen_all_cells(tmp_path, monkeypatch, trained, arm):
    from types import SimpleNamespace
    import anndata as ad
    from benchmarks.full_CPJUMP1.harness import pipeline
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    cells, images, calls = [], [], []
    for well, usage in (("train", "development_train"), ("val", "development_val"), ("held", "evaluation_only")):
        for site in range(2):
            ident = f"{well}_{site}"
            images.append(dict(image_id=ident, well=well, plate="p1", usage=usage,
                               calibration_group="", negative_control=False))
            path = tmp_path / f"{ident}.tif"
            tifffile.imwrite(path, np.ones((5, 16, 16), np.uint16), photometric="minisblack")
            cells.append(dict(cell_id=ident, image_id=ident, well=well, plate="p1", usage=usage,
                              area=100, crop_path=str(path), sha256=digest(path)))
    write_json(tmp_path / "cells.json", cells)
    checkpoint = tmp_path / "checkpoint.pt"
    checkpoint.write_bytes(b"synthetic checkpoint")

    def execute(self, **kwargs):
        calls.append(kwargs)
        meta = kwargs["cell_metadata"]
        p = tmp_path / f"embedding_{len(calls)}.h5ad"
        data = ad.AnnData(np.ones((len(meta), 3), dtype=np.float32), obs=pd.DataFrame({"cell_id": [r["cell_id"] for r in meta]}))
        data.write_h5ad(p)
        return {"adata_path": str(p), "resume_best": str(checkpoint) if kwargs.get("adaptation_depth") == "trained" else None}

    def resolve(self, verifier, adapter, runner, initial, baseline, **kw):
        verifier.admission = {"decision": "RUN" if trained else "SKIP"}
        params = {**initial, "adaptation_depth": "trained", "max_epochs": 1} if trained else initial
        result = runner(params) if trained else baseline
        return SimpleNamespace(params=params, result=result, details={}, status="satisfied", reason="test", fault=None)

    monkeypatch.setattr(pipeline, "engine_for", lambda cfg, stage: object())
    monkeypatch.setattr(Cell_State_Analyzer_Tool, "execute", execute)
    monkeypatch.setattr(StepVerifier, "resolve_stage", resolve)
    config = dict(arm=arm, deadline=time.time() + 3600, cores=1, ssl_search_deadline=time.time() + 3600 * 0.65,
                  train_cells_per_site=128, inference_chunk=16, seed=0,
                  tool_params={"adaptation_depth": "trained" if trained else "zero_shot"})
    if arm == "codex":
        pipeline.representation(tmp_path, tmp_path, {**config, "fit_only": True}, images)
        assert not (tmp_path / "embedding.npy").exists()
        assert all(not any("held" in p for p in call["cell_crops"]) for call in calls)
        config["ssl_selection"] = read_json(tmp_path / "SSL_SELECTION.json")
    pipeline.representation(tmp_path, tmp_path, config, images)
    for call in calls[:-1]:
        assert len(call["cell_metadata"]) == 4
        assert not any("held" in p for p in call["cell_crops"])
        assert call["split_key"] == "split"
        assert {m["split"] for m in call["cell_metadata"]} == {"train", "val"}
    inference = calls[-1]
    assert len(inference["cell_metadata"]) == 6
    assert all("split" not in r for r in inference["cell_metadata"])
    assert "split_key" not in inference
    if trained and arm == "codex":
        assert 0 < calls[0]["max_minutes"] <= 3600 * 0.65 / 60 - 15
    if trained:
        assert inference["embed_only"] is True and inference["init_checkpoint"] == str(checkpoint)
    else:
        assert inference["adaptation_depth"] == "zero_shot"
    assert np.load(tmp_path / "profiles.npy").shape == (3, 3)


@pytest.mark.parametrize("rows,cores", [(500, 8), (60_000, 8), (60_000, 2)])
def test_batch_stage_blas_limit_is_the_verifiers_threads_per_call(tmp_path, monkeypatch, rows, cores):
    from types import SimpleNamespace
    from threadpoolctl import threadpool_info
    from benchmarks.full_CPJUMP1.harness import pipeline
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.agent.verification.verifiers.batch_correction_unified import call_threads
    from shape.tools.batch_corrector.tool import DEFAULT_CONFIG, Batch_Corrector_Tool

    rng = np.random.default_rng(0)
    x = rng.normal(size=(rows, 4)).astype(np.float32)
    np.save(tmp_path / "profiles.npy", x)
    pd.DataFrame({"well": [f"w{i}" for i in range(rows)], "plate": ["p1", "p2"] * (rows // 2),
                  "calibration_group": "", "negative_control": [i % 10 == 0 for i in range(rows)]}).to_csv(
        tmp_path / "profiles.csv", index=False)
    seen, resolved = [], {}

    def execute(self, features, metadata, **kwargs):
        # harmonypy 2.0.0's bundled OpenBLAS (Linux: harmonypy.libs/libopenblas*) is single-threaded under any limit
        seen.append({i["filepath"]: i["num_threads"] for i in threadpool_info()
                     if i["user_api"] == "blas" and "harmonypy" not in i["filepath"]})
        return {"corrected_cls_output": np.asarray(features), "status": "CORRECTED_HARMONY"}

    def resolve(self, verifier, adapter, runner, initial, baseline, **kw):
        verifier.begin(kw["upstream_ctx"]["batch_task"])
        resolved.update(threads=verifier.threads_per_call, cap=verifier.cap)
        result = runner({"harmony_params": dict(DEFAULT_CONFIG)})
        return SimpleNamespace(params={}, result=result, details={}, status="satisfied", reason="test", fault=None)

    monkeypatch.setattr(pipeline, "engine_for", lambda config, stage: object())
    monkeypatch.setattr(Batch_Corrector_Tool, "execute", execute)
    monkeypatch.setattr(StepVerifier, "resolve_stage", resolve)
    pipeline.batch_correction(tmp_path, tmp_path, {"arm": "shape", "cores": cores,
                                                    "deadline": time.time() + 600}, [])
    threads = call_threads(x, cores)
    assert threads == {500: 1, 60_000: min(3, cores)}[rows] and resolved == {"threads": threads, "cap": cores // threads}
    assert seen and all(set(limits.values()) == {threads} for limits in seen)     # every BLAS library, every call


def test_crop_images_run_in_worker_processes_with_the_real_cropper(tmp_path):
    """Two images, two spawned worker processes (no stubs: the real cropper), outputs in image order."""
    prep, run = tmp_path / "prep", tmp_path / "run"
    (prep / "public").mkdir(parents=True)
    run.mkdir()
    masks, images = [], []
    for i in range(2):
        tifffile.imwrite(prep / "public" / f"image{i}.ome.tif", np.full((5, 64, 64), 4000 + i, np.uint16), ome=True,
                         photometric="minisblack", metadata={"axes": "CYX"})
        mask = np.zeros((64, 64), np.uint32)
        mask[20:38, 20:38] = 1
        p = run / f"cell_mask_img_{i}.tif"
        tifffile.imwrite(p, mask)
        masks.append({"image_id": f"img_{i}", "mask": str(p), "sha256": digest(p), "min_area": 15})
        images.append({"image_id": f"img_{i}", "file": f"image{i}.ome.tif", "plate": "p1", "well": f"w{i}",
                       "usage": "evaluation_only"})
    write_json(run / "segmentation.json", masks)
    cropping(prep, run, {"arm": "control", "deadline": time.time() + 600, "cores": 2,
                         "shape_policy": {"crop_vrr_workers": 2}}, images)
    cells = read_json(run / "cells.json")
    assert [c["image_id"] for c in cells] == ["img_0", "img_1"]
    assert [int(tifffile.imread(c["crop_path"])[0, 0, 0]) for c in cells] == [4000, 4001]


def test_a_worker_processs_token_usage_is_added_to_the_stage(monkeypatch):
    from shape.engine import usage
    for name in ("_TOTAL", "_FIELD_CALLS"):                 # this process's counters, restored after the test
        monkeypatch.setattr(usage, name, dict(getattr(usage, name)))
    monkeypatch.setattr(usage, "_OBSERVED", set(usage._OBSERVED))
    before = usage.snapshot()
    worker = {"total_tokens": 30, "prompt_tokens": 20, "completion_tokens": 10, "calls": 2,
              "usage_observed_fields": ["total_tokens"], "usage_field_observed_calls": {"total_tokens": 2}}
    usage.absorb(worker)
    after = usage.snapshot()
    assert after["total_tokens"] - before["total_tokens"] == 30 and after["calls"] - before["calls"] == 2
    assert "total_tokens" in after["usage_observed_fields"]

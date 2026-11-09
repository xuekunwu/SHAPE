"""harness/shape_arm.py: the SHAPE arm's driver.

The chain test builds the workspace with the real setup_run (test_control's synthetic prep) and runs every
delivered call through the real tool CLIs in subprocesses (SHAPE tool classes replaced by test/tool_stubs.py); the
segmentation child and the per-image crop loop are replaced by fakes that return chosen decisions (the masks and
crop boxes they report are computed with the same tool code, so the reproduction checks must hold); the
representation verifier, the batch rule and MorphologicalStateVerifier are the real ones; the reasoner is a fake
engine (its smoke calls and its usage records). Other tests run the real per-image segmentation loop with a stub
reasoner on one stub image, the deadline cut-off, a fatal LLM error stopping the run and any other search failure
recorded while the run goes on, the tier mapping, the embed lever translation and the refusals."""
from __future__ import annotations

import json
from functools import wraps
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_HCMA.harness import ledger, process, score, shape_arm, table
from benchmarks.full_HCMA.test.test_control import _allow_stubs, fast_scib, root  # noqa: F401 -- fixtures

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
_OPENED: list[str] = []
_WATCH = {"on": False}


def _audit(event, args):
    if _WATCH["on"] and event == "open" and args and isinstance(args[0], (str, bytes, os.PathLike)):
        _OPENED.append(os.fsdecode(args[0]) if isinstance(args[0], bytes) else os.fspath(args[0]))


sys.addaudithook(_audit)


class FakeEngine:
    """The reasoner: answers the smoke calls (the plain one, and the structured one with the schema's instance),
    records its usage as SHAPE's engines do."""

    def __init__(self, fail: bool = False, structured: bool = True):
        self.fail, self.structured, self.prompts = fail, structured, []

    def generate(self, content, response_format=None, **kwargs):
        from shape.engine import usage

        self.prompts.append(content)
        if self.fail:
            raise ConnectionError("gateway down")
        usage.record({"input_tokens": 11, "output_tokens": 2, "total_tokens": 13})
        answer = "OK"
        if response_format is not None and self.structured:
            answer = response_format(answer="OK", number=7)
        return {"content": answer, "usage": {"input_tokens": 11, "output_tokens": 2, "total_tokens": 13}}


@pytest.fixture()
def env(monkeypatch):
    """The monkeypatch; os.environ as the driver and the tools' setup change it, restored after the test."""
    saved = dict(os.environ)
    monkeypatch.setenv(shape_arm._ORIGINAL_CUDA, "<unset>")
    yield monkeypatch
    os.environ.clear()
    os.environ.update(saved)


def _stub_mask(image: str, levers: dict, out: Path, image_id: str) -> str:
    from benchmarks.full_HCMA.test import tool_stubs

    out.mkdir(parents=True, exist_ok=True)
    result = tool_stubs.Cell_Segmenter_Tool().execute(image=image, query_cache_dir=str(out), image_id=image_id,
                                                      **levers)
    return result["mask_path"]


def _fake_search(seen: dict):
    """The segmentation child: img_0001 gets its own levers (its mask made by the stub segmenter at them), img_0002
    was left at the defaults by the deadline, the others were searched and kept the defaults."""
    def search(job):
        seen["job"] = job
        out = {}
        for item in job["items"]:
            image_id = item["image_id"]
            if image_id == "img_0001":
                levers = {"diameter": 20.0, "min_size": 5}
                mask = _stub_mask(item["path"], {**job["defaults"], **levers}, Path(job["scratch"]) / image_id, image_id)
                out[image_id] = {"status": "searched", "levers": levers, "mask_npy": mask, "delivered": "c03"}
            elif image_id == "img_0002":
                out[image_id] = {"status": "left_at_defaults", "levers": {}, "note": "deadline"}
            else:
                out[image_id] = {"status": "searched", "levers": {}, "delivered": "c01"}
        return {"images": out, "usage": {"calls": 4, "input_tokens": 400, "output_tokens": 40, "total_tokens": 440}}
    return search


def _fake_crop(seen: dict):
    """The per-image crop loop: img_0001 gets margin_frac 0.3, a cell rule and one rejected cell; the boxes are
    the real cropper's at that configuration."""
    def crop_one(C, tool_class, item, engine, deadline, cores, scratch, log_dir):
        seen.setdefault("items", {})[item["image_id"]] = item
        if item["image_id"] != "img_0001":
            return {"status": "searched", "levers": {}, "cell_overrides": {}, "rejected_tool_cells": [], "boxes": {}}
        overrides = {"default_img_0001_cell_0000": {"margin_mode": "mask_bbox", "auto_tighten": True}}
        scratch.mkdir(parents=True, exist_ok=True)
        alias = shape_arm._link(Path(item["mask"]), scratch / "masks" / "cell_mask_img_0001.tif")
        result = tool_class().execute(original_image=item["path"], nuclei_mask=str(alias), source_image_id="img_0001",
                                      group="default", output_format="png", query_cache_dir=str(scratch),
                                      min_area=item["min_area"], **{**item["defaults"], "margin_frac": 0.3},
                                      cell_overrides=overrides)
        metadata = json.loads(Path(result["cell_crops_metadata_path"]).read_text(encoding="utf-8"))
        return {"status": "searched", "levers": {"margin_frac": 0.3}, "cell_overrides": overrides,
                "rejected_tool_cells": ["default_img_0001_cell_0001"],
                "boxes": {m["cell_id"]: m["crop_bbox"] for m in metadata["cell_metadata"]}}
    return crop_one


def _commands(run: Path) -> list[dict]:
    return [json.loads(line) for line in (run / "logs" / "control_commands.jsonl").read_text().splitlines()]


def _arg(argv: list, flag: str):
    return argv[argv.index(flag) + 1]


def test_the_shape_chain_runs_through_the_tool_clis(root, env, fast_scib):  # noqa: F811
    _allow_stubs(env)
    seen = {}
    env.setattr(shape_arm, "crop_one", _fake_crop(seen))
    engine = FakeEngine()
    run = root / "shape-run"
    (run / "logs").mkdir(parents=True)                # as a launcher leaves the run dir before the driver
    (run / "cache" / "tmp").mkdir(parents=True)
    real_build = shape_arm.build_engine

    def watched(model, *, engine=None):          # from here on SHAPE's code runs (the harness's checks are before)
        _WATCH["on"] = True
        return real_build(model, engine=engine)

    env.setattr(shape_arm, "build_engine", watched)
    _OPENED.clear()
    try:
        shape_arm.run_shape(root, REPO, run, model="fake-reasoner", budget_hours=2, engine=engine,
                            segment_search_fn=_fake_search(seen), crop_workers=2)
    finally:
        _WATCH["on"] = False
    # SHAPE read no label: neither the private labels nor the repository's HCMA label table
    assert not [p for p in _OPENED if "labels" in Path(p).name], [p for p in _OPENED if "labels" in p]
    assert engine.prompts[0] == [shape_arm.SMOKE_PROMPT]                       # the smoke calls before step 2
    from shape.agent.verification.vision_engine import SMOKE_QUESTION
    assert engine.prompts[1] == [SMOKE_QUESTION]
    commands = _commands(run)
    assert [(c["tool"], c["argv"][1]) for c in commands] == [
        ("segment", "defaults"), ("segment", "shape_seg"), ("crop", "shape_crop"), ("embed", "shape_zero_shot"),
        ("batch_correct", "shape_batch"), ("cluster", "shape_states")]
    assert all(c["returncode"] == 0 for c in commands)
    records = ledger.calls(run)
    by_name = {r["start"]["name"]: (cid, r) for cid, r in records.items()}

    # 2: the search saw the images' facts in its request, never a label; S002 = the delivered levers
    job = seen["job"]
    assert [i["image_id"] for i in job["items"]] == ["img_0001", "img_0002", "img_0003", "img_0004"]
    assert "## Target" in job["items"][0]["request"] and "image modality: brightfield" in job["items"][0]["request"]
    assert job["defaults"]["diameter"] == 30.0 and job["model"] == "fake-reasoner"
    s002, rec = by_name["shape_seg"]
    assert s002 == "S002" and rec["start"]["inputs"]["images"] == ["img_0001"]
    assert set(rec["start"]["set_levers"]) == {"diameter", "min_size"}
    assert json.loads((rec["dir"] / "inputs" / "per_image.json").read_text()) == {"img_0001": {"diameter": 20.0,
                                                                                                 "min_size": 5}}
    record = json.loads((run / shape_arm.RUN_RECORD).read_text(encoding="utf-8"))
    seg = record["stages"]["segment"]
    assert seg["changed"] == ["img_0001"] and seg["left_at_defaults"] == ["img_0002"]
    assert seg["reproduction"]["img_0001"]["identical"] is True and seg["mismatches"] == []

    # 3: the crop call reads S002's mask for img_0001 and carries the delivered levers and cell rules
    c001, rec = by_name["shape_crop"]
    masks_map = json.loads(Path(_arg(rec["start"]["argv"], "--masks-map")).read_text())
    assert masks_map == {"img_0001": "S002", "img_0002": "S001", "img_0003": "S001", "img_0004": "S001"}
    item = seen["items"]["img_0001"]
    assert item["defaults"]["margin"] is None and item["min_area"] == 5         # the CLI's defaults, S002's min_size
    assert seen["items"]["img_0003"]["min_area"] == 15
    meta = json.loads((rec["dir"] / "meta.json").read_text())
    assert meta["per_image"]["values"] == {"img_0001": {"margin_frac": 0.3}}
    assert meta["cell_overrides"]["values"] == {"img_0001": {"default_img_0001_cell_0000": {
        "margin_mode": "mask_bbox", "auto_tighten": True}}}
    crop = record["stages"]["crop"]
    assert crop["reproduction"]["img_0001"]["identical"] is True and crop["mismatches"] == []
    assert crop["n_excluded_cells"] == 1

    # 4: arm C's reject is excluded from the embed call (a miss for the scorer); the gate SKIPs on the batch tier
    e001, rec = by_name["shape_zero_shot"]
    crops = pd.read_csv(records[c001]["dir"] / "crops.csv", dtype=str)
    rejected = crops.loc[(crops["image_id"] == "img_0001") & (crops["tool_cell_id"] == "default_img_0001_cell_0001"),
                         "cell_id"].tolist()
    assert (run / "workspace" / rec["start"]["inputs"]["exclude_cells"]).read_text().split() == rejected
    assert rec["start"]["params"]["adaptation_depth"] == "zero_shot"
    rep = record["stages"]["representation"]
    assert rep["tier"] == "batch" and rep["gate"]["decision"] == "SKIP" and rep["delivered"] == e001
    assert rep["resolution"]["reason"] == "not_measurable" and "no adopted proxy" in rep["gate"]["reasons"][0]
    b001, rec = by_name["shape_batch"]
    assert _arg(rec["start"]["argv"], "--embedding") == e001 and _arg(rec["start"]["argv"], "--batch-key") == "source"
    assert rec["end"]["summary"]["corrected"] is False and rec["end"]["summary"]["tier"] == "partial"
    k001, rec = by_name["shape_states"]
    morph = record["stages"]["morphology"]
    assert rec["start"]["params"]["resolution"] == [morph["resolution"]]
    assert rec["start"]["params"]["n_neighbors"] == morph["n_neighbors"] and morph["trials"]
    choice = json.loads((run / "workspace" / "submission" / "choice.json").read_text())
    assert choice["cluster_call"] == k001 and choice["resolution"] == morph["resolution"]

    # the records: a SHAPE run, not a Claude run; its chain verifies from the driver's command log
    run_meta = json.loads((run / "RUN_META.json").read_text())
    assert run_meta["arm"] == "shape" and run_meta["model"] == "fake-reasoner" and run_meta["gpu_slug"] == "testgpu"
    assert not (run / "AGENT_RUN.json").exists() and not list((run / "traces").glob("round_*.jsonl"))
    assert ledger.run_arm(run) == "shape" and not ledger.is_agent_run(run)
    assert ledger.chain(run, choice)["ok"], ledger.chain(run, choice)["problems"]
    assert record["usage"]["segment"]["total_tokens"] == 440 and record["usage"]["smoke_call"]["calls"] == 2
    assert record["usage"]["total"]["calls"] == 6 and record["usage"]["total"]["input_tokens"] == 422
    assert record["status"] == "delivered" and record["wall_hours"] > 0

    # scored, process metrics and table rows as an arm of its own
    result = score.score_run(run, root, forbidden=[])
    assert result["status"] == "OK" and result["chain"]["ok"]
    report = process.write(run)
    assert report["arm"] == "shape" and report["end_state"]["end_state"] == "delivered"
    assert report["D1_adaptivity"]["segment"]["share_images_with_own_setting"] == 0.25
    assert report["D1_adaptivity"]["crop"]["share_images_with_cell_overrides"] == 0.25
    llm = report["E1_cost"]["llm"]
    assert llm["usd_source"] == "shape_usage" and llm["total_cost_usd"] is None and llm["n_calls"] == 6
    assert llm["tokens"]["input_tokens"] == 422 and report["claims"]["applicable"] is False
    assert report["F3_failures"]["shape_llm"] == {"failed_searches": {}, "failed_questions": {},
                                                  "n_failed_questions": 0, "stopped": None}
    row = table.run_row(run)
    assert row["arm"] == "shape" and row["model"] == "fake-reasoner" and row["tokens_in"] == 422
    assert row["gpu"] == "testgpu" and row["usd_estimate"] is None


class _SegAgent:
    """A stub reasoner of the segmentation loop: round 1 diameter 20 from c01, then no moves; the final choice is
    the first admissible call; the preflight read gets an unconfident answer."""

    def __init__(self):
        self.rounds = [[("default", {"diameter": 20.0})]]

    def generate(self, content, response_format=None):
        import ast
        import re

        from shape.agent.verification.verifiers.segmentation import SegPlan
        from shape.utils.schemas import LeverChange, WaveMove

        if response_format is SegPlan:
            moves = [WaveMove(parent=parent, changes=[LeverChange(lever=k, value=json.dumps(v))
                                                      for k, v in changes.items()])
                     for parent, changes in (self.rounds.pop(0) if self.rounds else [])]
            return {"content": SegPlan(analysis="stub", moves=moves)}
        if "call" in getattr(response_format, "model_fields", {}):
            found = re.search(r"Admissible calls \(.*?\): (\[[^\]]*\])", content[0], re.DOTALL)
            admissible = ast.literal_eval(found.group(1)) if found else []
            return {"content": response_format(call=admissible[0] if admissible else None, reasoning="stub")}
        from shape.agent.verification.image_context import ImageContextProfile, default_unknown_profile
        if response_format is ImageContextProfile:   # the preflight read: an unconfident answer
            return {"content": default_unknown_profile("stub")}
        raise RuntimeError(f"no answer for {response_format}")


def test_the_real_segmentation_loop_reuses_s001_for_the_defaults_call(tmp_path, env):
    """SegmentationAgenticVerifier with a stub reasoner on one stub image: its call at the defaults is S001's result
    (no in-process call at the defaults); every other call is the segmenter in this process."""
    import tifffile

    from benchmarks.full_HCMA.test import tool_stubs
    from benchmarks.full_HCMA.test.test_control import _image

    env.setenv("FULL_HCMA_TOOL_STUBS", str(HERE / "tool_stubs.py"))
    C = shape_arm.tools_common()
    levers = list(C.tool_levers("segment", REPO))
    defaults = {k: C.normal_default(v) for k, v in C.tool_levers("segment", REPO).items()}
    image = tmp_path / "img_0001.png"
    _image(image, 5, 0)
    s001 = tmp_path / "s001"
    result = tool_stubs.Cell_Segmenter_Tool().execute(image=str(image), query_cache_dir=str(s001), image_id="img_0001")
    tifffile.imwrite(tmp_path / "s001.tif", np.load(result["mask_path"]).astype(np.uint16))
    record = {k: v for k, v in result.items() if k not in ("mask_path", "overlay_path", "output_path",
                                                             "visual_outputs", "deliverables")}
    item = {"image_id": "img_0001", "path": str(image), "request": shape_arm.segment_request({"image_id": "img_0001"}),
            "s001_record": record, "s001_mask": str(tmp_path / "s001.tif"), "s001_overlay": result["overlay_path"]}
    calls = []

    class Counting(tool_stubs.Cell_Segmenter_Tool):
        @wraps(tool_stubs.Cell_Segmenter_Tool.execute)
        def execute(self, **kwargs):
            calls.append(kwargs)
            return super().execute(**kwargs)

    outcome = shape_arm.segment_one(C, Counting(), item, levers, defaults, _SegAgent(), tmp_path / "search",
                                    tmp_path / "logs")
    assert outcome["status"] == "searched" and outcome["defaults_call_reused"] is True
    assert calls and any(k["diameter"] == 20.0 for k in calls)
    assert outcome["reasoner_failures"] == outcome["fallback_rounds"] == 0
    assert all(not shape_arm._same_levers(C, levers, k, defaults) for k in calls)   # never a call at the defaults
    assert all(k["model_type"] == "cpsam" and k["bsize"] == 256 for k in calls)
    assert outcome["admission"]["call"] == "default" and outcome["n_calls"] == len(calls) + 1
    assert (tmp_path / "logs" / "img_0001.details.json").exists()
    assert set(outcome["levers"]) <= set(levers)


def test_segment_search_starts_no_search_after_the_deadline(tmp_path, env):
    env.setenv("FULL_HCMA_TOOL_STUBS", str(HERE / "tool_stubs.py"))
    env.setattr(shape_arm, "segment_one", lambda *a, **k: pytest.fail("no search after the deadline"))
    config = tmp_path / "_config.json"
    config.write_text(json.dumps({"repo": str(REPO), "env": {}, "allow_tool_stubs": True}))
    job = {"config": str(config), "items": [{"image_id": "img_0001"}, {"image_id": "img_0002"}], "levers": [],
           "defaults": {}, "stop_at": 0, "model": None, "scratch": str(tmp_path), "log_dir": str(tmp_path)}
    out = shape_arm.segment_search(job)
    assert {i: o["status"] for i, o in out["images"].items()} == {"img_0001": "left_at_defaults",
                                                                   "img_0002": "left_at_defaults"}
    assert all("deadline" in o["note"] for o in out["images"].values())


def test_real_crop_loop_exports_mask_check_rejections_and_failure_telemetry(tmp_path, monkeypatch):
    import time
    from shape.agent.verification import base
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier
    from shape.agent.verification.verifiers.test.test_crop import (
        _Engine, _make_input, _mask_check_verdicts, _move,
    )
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool

    monkeypatch.setattr(base, "REASONER_RETRY_SECONDS", 0)
    image, mask = _make_input(tmp_path / "data")
    item = {"image_id": "img", "path": str(image), "mask": str(mask), "min_area": 50,
            "defaults": CropAgenticVerifier().defaults(), "brief": "test"}
    engine = _Engine([[_move(margin_mode="mask_bbox")]], final=["agent_1"],
                     mask_check=_mask_check_verdicts(reject_first=True))
    outcome = shape_arm.crop_one(shape_arm.tools_common(), Single_Cell_Cropper_Tool, item, engine,
                                  time.time() + 60, 4, tmp_path / "search", tmp_path / "logs")
    details = json.loads((tmp_path / "logs" / "img.details.json").read_text())["details"]
    assert outcome["rejected_tool_cells"] and outcome["rejected_tool_cells"] == details["mask_check"]["rejected_cells"]
    assert outcome["reasoner_failures"] == outcome["fallback_rounds"] == 0
    assert outcome["execution_errors"] == []

    class Down(_Engine):
        def generate(self, *args, **kwargs):
            raise TimeoutError("nonfatal gateway timeout")

    failed = shape_arm.crop_one(shape_arm.tools_common(), Single_Cell_Cropper_Tool, item, Down(),
                                time.time() + 60, 4, tmp_path / "failed", tmp_path / "failed-logs")
    assert failed["reasoner_failures"] > 0 and failed["fallback_rounds"] == 0
    assert failed["execution_errors"][0]["question"] == "round"
    run = tmp_path / "run"
    run.mkdir()
    (run / shape_arm.RUN_RECORD).write_text(json.dumps({"stages": {"crop": {"per_image": {"img": failed}}}}))
    assert process.shape_llm_failures(run)["n_failed_questions"] > 0


def test_the_segmentation_child_runs_as_its_own_process(tmp_path, env):
    """`python -m benchmarks.full_HCMA.harness.shape_arm segment-search JOB OUT` (the process that holds the GPU)."""
    env.setenv("FULL_HCMA_TOOL_STUBS", str(HERE / "tool_stubs.py"))
    config = tmp_path / "_config.json"
    config.write_text(json.dumps({"repo": str(REPO), "env": {}, "allow_tool_stubs": True}))
    job = {"config": str(config), "items": [{"image_id": "img_0001"}], "levers": [], "defaults": {}, "stop_at": 0,
           "model": None, "scratch": str(tmp_path), "log_dir": str(tmp_path), "workers": 2}
    (tmp_path / "job.json").write_text(json.dumps(job))
    out = shape_arm._run_child_search(tmp_path / "job.json", tmp_path / "out.json", REPO, tmp_path / "child.log")
    assert out["images"]["img_0001"]["status"] == "left_at_defaults" and "calls" in out["usage"]
    (tmp_path / "bad.json").write_text("{}")
    with pytest.raises(SystemExit, match="SHAPE_FAILED: the segmentation child exited"):
        shape_arm._run_child_search(tmp_path / "bad.json", tmp_path / "out2.json", REPO, tmp_path / "child2.log")


def test_the_crop_stage_starts_no_search_after_its_deadline(root, env, fast_scib):  # noqa: F811
    _allow_stubs(env)
    env.setattr(shape_arm, "SEGMENT_SHARE", 0.0)
    env.setattr(shape_arm, "CROP_SHARE", 0.0)
    env.setattr(shape_arm, "crop_one", lambda *a, **k: pytest.fail("no crop search after the deadline"))
    run = shape_arm.run_shape(root, REPO, root / "late", model="fake", budget_hours=2, engine=FakeEngine(),
                              segment_search_fn=lambda job: shape_arm.segment_search(job, engine=FakeEngine()))
    record = json.loads((run / shape_arm.RUN_RECORD).read_text(encoding="utf-8"))
    assert record["stages"]["segment"]["left_at_defaults"] == ["img_0001", "img_0002", "img_0003", "img_0004"]
    assert record["stages"]["crop"]["left_at_defaults"] == ["img_0001", "img_0002", "img_0003", "img_0004"]
    assert [c["argv"][1] for c in _commands(run)] == ["defaults", "shape_crop", "shape_zero_shot", "shape_batch",
                                                      "shape_states"]          # no S002, no per-image files
    crop_argv = _commands(run)[1]["argv"]
    assert "--per-image" not in crop_argv and "--cell-overrides" not in crop_argv


def _record(run: Path) -> dict:
    return json.loads((run / shape_arm.RUN_RECORD).read_text(encoding="utf-8"))


def _auth_error(question: str):
    """A fatal LLM error as `AgenticVerifier._ask` raises it (tagged with its question)."""
    import httpx
    import openai

    request = httpx.Request("POST", "https://gateway.example/v1/chat/completions")
    error = openai.AuthenticationError("invalid key", response=httpx.Response(401, request=request), body=None)
    error.shape_question = question
    return error


def test_a_fatal_llm_error_in_a_segment_search_stops_the_run(root, env, fast_scib, capsys):  # noqa: F811
    """A fatal LLM error (by type: here an invalid key) stops the run: no S002, no crop, the images after it not
    started, `stopped` (stage, image, question, cause) in SHAPE_RUN.json and in the log."""
    _allow_stubs(env)
    env.setattr(shape_arm, "SEGMENT_WORKERS", 1)

    def segment_one(C, tool, item, *args, **kwargs):
        if item["image_id"] == "img_0002":
            raise _auth_error("round")
        return {"status": "searched", "levers": {}, "delivered": "c01"}

    env.setattr(shape_arm, "segment_one", segment_one)
    env.setattr(shape_arm, "crop_one", lambda *a, **k: pytest.fail("no crop search after a fatal error"))
    run = root / "seg-stopped"
    with pytest.raises(SystemExit, match="SHAPE_FAILED: the segment stage stopped on image img_0002, the round "
                                         "question: AuthenticationError: invalid key"):
        shape_arm.run_shape(root, REPO, run, model="fake", budget_hours=2, engine=FakeEngine(),
                            segment_search_fn=lambda job: shape_arm.segment_search(job, engine=FakeEngine()))
    record = _record(run)
    stop = {"stage": "segment", "image": "img_0002", "question": "round", "cause": "AuthenticationError: invalid key"}
    assert record["status"] == "failed" and record["stopped"] == stop
    seg = record["stages"]["segment"]
    assert seg["stopped"] == stop and seg["failed"] == ["img_0002"] and seg["not_started"] == ["img_0003", "img_0004"]
    assert seg["per_image"]["img_0001"]["status"] == "searched" and seg["left_at_defaults"] == []
    assert "crop" not in record["stages"] and "choice" not in record
    assert [c["argv"][1] for c in _commands(run)] == ["defaults"]          # nothing after S001
    assert "SHAPE_FAILED: the segment stage stopped on image img_0002, the round question" in capsys.readouterr().out


def test_a_non_fatal_search_failure_is_recorded_and_the_run_goes_on(root, env, fast_scib, capsys):  # noqa: F811
    """Any other error of an image's search is that image's outcome "failed" (the image at the defaults): the other
    images are searched and the chain is delivered; the process metrics count it."""
    _allow_stubs(env)
    env.setattr(shape_arm, "SEGMENT_WORKERS", 1)
    seen = {}

    def segment_one(C, tool, item, *args, **kwargs):
        if item["image_id"] == "img_0002":
            raise RuntimeError("segmentation: no usable call left (execution error: TimeoutError)")
        return {"status": "searched", "levers": {}, "delivered": "c01",
                "execution_errors": [{"status": "execution_error", "question": "final", "cause": "ValueError: x"}]}

    env.setattr(shape_arm, "segment_one", segment_one)
    env.setattr(shape_arm, "crop_one", _fake_crop(seen))
    run = shape_arm.run_shape(root, REPO, root / "seg-goes-on", model="fake", budget_hours=2, engine=FakeEngine(),
                              segment_search_fn=lambda job: shape_arm.segment_search(job, engine=FakeEngine()))
    record = _record(run)
    assert record["status"] == "delivered" and "stopped" not in record
    seg = record["stages"]["segment"]
    assert seg["failed"] == ["img_0002"] and seg["not_started"] == [] and "stopped" not in seg
    assert "no usable call left" in seg["per_image"]["img_0002"]["error"]
    assert sorted(seen["items"]) == ["img_0001", "img_0002", "img_0003", "img_0004"]   # every image cropped
    assert "segment search img_0002 failed (the run goes on)" in capsys.readouterr().out
    shape_llm = process.write(run)["F3_failures"]["shape_llm"]
    assert shape_llm["failed_searches"] == {"segment": ["img_0002"]} and shape_llm["stopped"] is None
    assert shape_llm["failed_questions"] == {"segment": {"final": 3}} and shape_llm["n_failed_questions"] == 3


def test_a_fatal_crop_search_error_stops_the_run_before_the_crop_call(root, env, fast_scib, capsys):  # noqa: F811
    _allow_stubs(env)
    from shape.engine.errors import NoEngineError

    def crop_one(C, tool_class, item, *args, **kwargs):
        if item["image_id"] == "img_0002":
            raise NoEngineError("crop-triage: arm C (the mask-error check): no vision engine is configured")
        if item["image_id"] == "img_0001":
            raise RuntimeError("not fatal")              # recorded, the stage would have gone on
        return {"status": "searched", "levers": {}, "cell_overrides": {}, "rejected_tool_cells": [], "boxes": {}}

    env.setattr(shape_arm, "crop_one", crop_one)
    run = root / "crop-stopped"
    with pytest.raises(SystemExit, match="SHAPE_FAILED: the crop stage stopped on image img_0002: NoEngineError"):
        shape_arm.run_shape(root, REPO, run, model="fake", budget_hours=2, engine=FakeEngine(),
                            segment_search_fn=_fake_search({}), crop_workers=1)
    record = _record(run)
    assert record["status"] == "failed" and record["stopped"]["stage"] == "crop"
    assert record["stopped"]["question"] is None and "no vision engine" in record["stopped"]["cause"]
    crop = record["stages"]["crop"]
    assert crop["failed"] == ["img_0001", "img_0002"] and crop["not_started"] == ["img_0003", "img_0004"]
    assert "stop" not in crop["per_image"]["img_0001"]
    assert [c["argv"][1] for c in _commands(run)] == ["defaults", "shape_seg"]      # no crop call
    assert "SHAPE_FAILED: the crop stage stopped on image img_0002" in capsys.readouterr().out


def test_the_structured_smoke_call_refuses_a_reasoner_without_structured_output(root, env):  # noqa: F811
    with pytest.raises(SystemExit, match="NOT_READY: the structured smoke call to fake failed"):
        shape_arm.run_shape(root, REPO, root / "no-structured", model="fake", engine=FakeEngine(structured=False))
    assert not (root / "no-structured").exists()


def test_a_stage_level_fatal_error_is_recorded_as_what_stopped_the_run(root, env, fast_scib):  # noqa: F811
    _allow_stubs(env)
    env.setattr(shape_arm, "crop_one", _fake_crop({}))

    def represent(self, c001, excluded):
        raise _auth_error("admission")

    env.setattr(shape_arm.Arm, "represent", represent)
    run = root / "rep-stopped"
    import openai
    with pytest.raises(openai.AuthenticationError):
        shape_arm.run_shape(root, REPO, run, model="fake", budget_hours=2, engine=FakeEngine(),
                            segment_search_fn=_fake_search({}), crop_workers=1)
    record = _record(run)
    assert record["status"] == "failed"
    assert record["stopped"] == {"stage": "representation", "image": None, "question": "admission",
                                 "cause": "AuthenticationError: invalid key"}


def test_the_tier_mapping_site_image_well_micrograph_plate_source():
    from shape.tools.cell_state_analyzer import quant

    def cells(images):
        return pd.DataFrame({"cell_id": [f"{i}_c{k}" for i in images for k in range(3)],
                             "image_id": [i for i in images for _ in range(3)]})

    # pairs: every micrograph contributes one image -> no well with 2 sites -> batch tier (no adopted proxy)
    rows = [{"image_id": f"i{n}", "parent_id": f"p{n}", "source": f"s{n % 2}"} for n in range(8)]
    meta, split = shape_arm.representation_meta(cells([r["image_id"] for r in rows]), rows)
    assert meta[0] == {"cell_id": "i0_c0", "image_name": "i0", "site": "i0", "well": "p0", "plate": "s0",
                       "split": meta[0]["split"]}
    assert quant.tier_of(meta) == "batch"
    # the full atlas: 289 of 916 micrographs have 2 panels (< half) -> still the batch tier
    rows = [{"image_id": f"i{n}", "parent_id": f"p{n // 2 if n < 578 else n - 289}", "source": "s"} for n in range(1205)]
    meta, _ = shape_arm.representation_meta(cells([r["image_id"] for r in rows]), rows)
    assert len({m["well"] for m in meta}) == 916 and quant.tier_of(meta) == "batch"
    # more than half of the micrographs with 2 panels -> the replicate tier
    rows = [{"image_id": f"i{n}", "parent_id": f"p{n // 2}", "source": "s"} for n in range(12)]
    assert quant.tier_of(shape_arm.representation_meta(cells([r["image_id"] for r in rows]), rows)[0]) == "replicate"
    # metadata none: no well, no plate -> the image tier
    rows = [{"image_id": f"i{n}", "parent_id": "", "source": ""} for n in range(4)]
    meta, split = shape_arm.representation_meta(cells([r["image_id"] for r in rows]), rows)
    assert quant.tier_of(meta) == "image" and meta[0]["well"] is None and split["rule"].startswith("every cell train")


def test_embed_trials_translate_to_cli_levers_or_fail():
    C = shape_arm.tools_common()
    defaults = {"view_pairing": "off", "trainable_pct": 0, "learning_rate": 3e-5}
    assert shape_arm.embed_set_args(C, {"learning_rate": 5e-5, "view_pairing": "off", "trainable_pct": 0,
                                        "output_subdir": "ssl_calls/x", "split_key": "split", "max_minutes": 9,
                                        "init_checkpoint": None}, defaults) == ["learning_rate=5e-05"]
    with pytest.raises(ValueError, match="no lever for view_pairing='same_well'"):
        shape_arm.embed_set_args(C, {"view_pairing": "same_well"}, defaults)
    with pytest.raises(ValueError, match="resume state"):
        shape_arm.embed_set_args(C, {"init_checkpoint": "/x/resume_best.pt"}, defaults)


def test_refusals_without_a_reasoner(root, env, monkeypatch):  # noqa: F811
    monkeypatch.delenv(shape_arm.MODEL_ENV, raising=False)
    with pytest.raises(SystemExit, match=shape_arm.MODEL_ENV):
        shape_arm.run_shape(root, REPO, root / "no-model", model=None, engine=FakeEngine())
    with pytest.raises(SystemExit, match="smoke call to fake failed: ConnectionError"):
        shape_arm.run_shape(root, REPO, root / "down", model="fake", engine=FakeEngine(fail=True))
    assert not (root / "no-model").exists() and not (root / "down").exists()
    from benchmarks.full_HCMA.harness import cli

    with pytest.raises(SystemExit, match=shape_arm.MODEL_ENV):
        cli.main(["shape", "--root", str(root), "--repo", str(REPO), "--out", str(root / "cli")])


def test_run_arm_tells_shape_from_claude_runs(tmp_path):
    shape = tmp_path / "shape"
    shape.mkdir()
    (shape / "RUN_META.json").write_text(json.dumps({"arm": "shape"}))
    assert ledger.run_arm(shape) == "shape"
    (shape / "traces").mkdir()
    (shape / "traces" / "round_00.jsonl").write_text("")
    assert ledger.run_arm(shape) == "agent"                  # a traced round: a Claude run whatever RUN_META says
    claude = tmp_path / "claude"
    claude.mkdir()
    (claude / "RUN_META.json").write_text(json.dumps({"agent": "claude"}))
    assert ledger.run_arm(claude) == "agent" and ledger.run_arm(tmp_path) == "control"


def test_stage_deadlines_leave_time_for_the_calls_after_the_searches():
    d = shape_arm.stage_deadlines(0.0, 4.0)                       # a long budget: the shares apply
    assert d == {"segment": 0.35 * 4 * 3600, "crop": 0.75 * 4 * 3600, "run": 4 * 3600}
    d = shape_arm.stage_deadlines(0.0, 1.0)                       # a short one: the tail caps the crop search
    assert d["crop"] == 3600 - shape_arm.TAIL_MINUTES * 60 and d["segment"] == 0.35 * 3600
    d = shape_arm.stage_deadlines(0.0, 0.6)                       # shorter still: segment never after crop
    assert d["segment"] <= d["crop"] == 0.6 * 3600 - shape_arm.TAIL_MINUTES * 60

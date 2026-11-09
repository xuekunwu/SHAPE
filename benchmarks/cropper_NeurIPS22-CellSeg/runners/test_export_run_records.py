"""export_run_records.py (the audit records, the copy into data/), the seed parameter and the export wiring of
the cluster scripts. Offline: synthetic run dirs in the format before and after the run-time recording."""

from __future__ import annotations

import csv
import json
import re
import shutil
import subprocess
from pathlib import Path

import pytest

import run_benchmark

err = run_benchmark.load_exporter()     # by path: other benchmarks have an export_run_records.py too

_RUNNERS = Path(__file__).resolve().parent
COMMIT = "a" * 40
IMAGES = ("img_a", "img_b")
BASH = shutil.which("bash")
needs_bash = pytest.mark.skipif(BASH is None, reason="bash not found")
REF = {"label": "INPUT OVERLAY", "sha256": "0" * 64, "bytes": 12}


def _write(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(payload if isinstance(payload, str) else json.dumps(payload), encoding="utf-8")


def _jsonl(path: Path, rows) -> None:
    _write(path, "".join(json.dumps(r) + "\n" for r in rows))


def _metrics(**extra) -> dict:
    return {"neighbor_contamination": 0.2, "target_retention": 0.95, "runtime_per_image": 3.0,
            "tool_calls_per_image": 2, "tokens_per_image": 100, "n_crops": 3, "n_expected_crops": 3,
            "crop_coverage": 1.0, "n_triage_rejected": None, "model_calls_per_image": 2,
            "input_tokens_per_image": 80, "output_tokens_per_image": 20, "cached_tokens_per_image": 0,
            "reasoning_tokens_per_image": 5, "delivery": None, **extra}


def make_run(root: Path, name: str, new: bool, names: tuple[str, str] | None = None) -> Path:
    """A two-image run dir with all five arms; `new` = with the fields of the run-time recording. `names`: the
    calls' names (the tool at its defaults, then the agent's call): the role names by default for a new run,
    c01 / c02 for an old one (the records before the role names, e.g. the reported run made at ea78bb4b)."""
    D, A = names or (("default", "agent_1") if new else ("c01", "c02"))
    run = root / name
    seed = 1881694274 if new else 0
    _write(run / "selection.json", {"created_utc": "2026-10-06T20:00:00+00:00" if new else "2026-10-05T20:00:00+00:00",
                                    "mask_source": "ground_truth", "seed": seed,
                                    "selected_images": [{"image_id": i, "image_path": f"/d/images/{i}.png",
                                                         "gt_path": f"/d/labels/{i}_label.tiff",
                                                         "mask_path": f"/d/labels/{i}_label.tiff"} for i in IMAGES]})
    _write(run / "provenance.json", {"git_commit": COMMIT, "status": "failed", "problems": ["gpt records=2"],
                                     **({"seed": seed} if new else {})})
    for arm in err.ARMS:
        meta = {"arm": arm, "model": "gpt-5.6-luna", "git_commit": COMMIT, "started_utc": "t0", "min_area": 15,
                "margin": 1, "maskcrop_margin_frac": 0.2, "cores": 8, "concurrency_cap": 4, "max_rounds": 8,
                "react_tool_budget": 29, "gpt_batch_size": 64}
        if new:
            meta.update(host="sh03", python="3.11.9", packages={"numpy": "1.26"}, seed=seed,
                        llm={"model": "gpt-5.6-luna", "provider": "openai"} if arm in ("gpt", "react", "shape") else None)
        _write(run / arm / "run_meta.json", meta)
    _jsonl(run / "maskcrop" / "records.jsonl", [
        {"image_id": i, "status": "ok", "metrics": _metrics(tokens_per_image=0),
         "trace": {"result": {"summary": "ok", "cell_count": 3, "seconds": 0.1}}} for i in IMAGES])
    _jsonl(run / "cellprofiler" / "records.jsonl", [
        {"image_id": i, "status": "ok", "metrics": _metrics(),
         "trace": {"backend": "CellProfiler", "module": "SaveCroppedObjects", "stdout": "OK"}} for i in IMAGES])
    if new:
        call = {"batch_index": 0, "targets": [1, 2, 3], "prompt": "Return only a JSON list", "images": [REF],
                "params": {"model": "gpt-5.6-luna", "max_tokens": 8000, "response_format": None},
                "usage": {"total_tokens": 50}, "seconds": 1.5}
        gpt_ok = {"model": "gpt-5.6-luna", "batches": [{"index": 0, "n_targets": 3, "attempts": 2}],
                  "calls": [{**call, "attempt": 1, "reply": "[]", "error": "gpt returned 0 boxes"},
                            {**call, "attempt": 2, "reply": "[{...}]", "error": None}], "image_attached": True}
        gpt_err = {**gpt_ok, "calls": [{**call, "attempt": a, "reply": "x", "error": "no JSON"} for a in (1, 2, 3)]}
    else:
        gpt_ok = {"model": "gpt-5.6-luna", "batches": [{"index": 0, "n_targets": 3, "attempts": 2}],
                  "image_attached": True}
        gpt_err = None
    _jsonl(run / "gpt" / "records.jsonl", [
        {"image_id": "img_a", "status": "ok", "metrics": _metrics(), "trace": gpt_ok},
        {"image_id": "img_b", "status": "error", "error": "GPT batch 1 failed after 3 attempts: no JSON",
         "metrics": {"runtime_per_image": 9.0}, **({"trace": gpt_err} if gpt_err else {})}])
    model_extra = {"prompt": "You are a ReAct agent", "images": [REF], "seconds": 2.0} if new else {}
    reply1 = ({"thought": "T1", "action": "CALL"} if new else f"thought='T1' action='CALL' parent='{D}'")
    reply2 = ({"thought": "T2", "action": "FINISH", "deliver": A} if new
              else f"thought='T2' action='FINISH' deliver='{A}'")
    react_trace = [
        {"step": 0, "action": "maskcrop", "call": D, "parent": "input", "change": "none (defaults)",
         "readouts": {"mean_target_dominance": 0.5}, "guard": "admissible", "seconds": 1.0},
        {"step": 1, "attempt": 1, **model_extra, "raw_reply": {"content": reply1, "usage": {"total_tokens": 9}},
         "action": "call"},
        {"step": 2, "action": "maskcrop", "call": A, "parent": D, "change": {"margin": 0},
         "cell_rules": [], "thought": "T1", "readouts": {"mean_target_dominance": 0.7}, "guard": "admissible",
         "seconds": 1.0},
        {"step": 3, "attempt": 1, **model_extra, "raw_reply": {"content": "{", "usage": None}, "error": "no JSON"},
        {"step": 4, "attempt": 2, **model_extra, "raw_reply": {"content": reply2, "usage": {"total_tokens": 9}},
         "action": "finish", "deliver": A}]
    _jsonl(run / "react" / "records.jsonl", [
        {"image_id": i, "status": "ok", "metrics": _metrics(delivery="finish"),
         "trace": {"delivered_call": A, "delivery": "finish", "image_attached": True, "trace": react_trace}}
        for i in IMAGES])
    _jsonl(run / "shape" / "records.jsonl", [
        {"image_id": i, "status": "ok", "metrics": _metrics(delivery="agent_choice", n_triage_rejected=1,
                                                             reasoner_calls_per_image=3, vision_calls_per_image=1),
         "trace": {"shape_loop": "agentic", "status": "satisfied", "reason": "plan_delivery",
                   "delivered_call": A, "delivery": "agent_choice", "delivered_metadata_path": "/m.json"}}
        for i in IMAGES])
    images = {"images": [REF]} if new else {}
    plan = {"analysis": "contamination is high", "hypotheses": [{"id": "H1"}], "moves": [{"parent": D}],
            "deliver": None}
    log = [{"t": 1, "kind": "admission", "text": json.dumps({"call": D, "decision": "RUN"})},
           {"t": 2, "kind": "prompt", "label": "round", "text": "You tune a fixed tool", **images},
           {"t": 3, "kind": "usage", "label": "round", "text": json.dumps({"total_tokens": 70})},
           {"t": 4, "kind": "reasoning", "label": "round", "text": json.dumps(plan)},
           {"t": 5, "kind": "rejected_move", "text": "already run"},
           {"t": 6, "kind": "prompt", "label": "round", "text": "You tune a fixed tool (round 3)", **images},
           {"t": 7, "kind": "reasoner_error", "label": "round", "text": "try 1: RuntimeError: 502",
            **({"reply": None} if new else {})},
           {"t": 8, "kind": "usage", "label": "round", "text": json.dumps({"total_tokens": 60})},
           {"t": 9, "kind": "reasoning", "label": "round",
            "text": json.dumps({"analysis": f"{A} is best", "moves": [], "deliver": A})},
           {"t": 10, "kind": "plan_delivery", "text": json.dumps({"call": A, "delivered": True})},
           {"t": 11, "kind": "choice", "text": json.dumps({"chosen": A, "how": "agent_choice"})}]
    mask_check = {"call": A, "mask_check_candidates": 2.0, "mask_check_judged": 2.0, "mask_check_rejected": 1.0,
                  "rejected_cells": ["cell_0001"]}
    if new:
        mask_check.update(
            montages=[{"index": 0, "prompt": "Judge each crop", "image": REF, "montage_path": "/mc/montage_000.png",
                       "reply": {"verdicts": [{"position": 1, "verdict": "wrong_mask"}]}, "error": None}],
            verdicts=[{"crop_id": "cell_0001", "verdict": "wrong_mask", "confidence": 0.9, "reject": True,
                       "montage_path": "/mc/montage_000.png"}])
    details = {"status": "satisfied", "reason": "plan_delivery", "rounds": [[D], [A]],
               "entries": [None, {"index": 1, "param_delta": {}}, {"index": 2, "param_delta": {"margin": 0},
                                                                   "outcome": "effective"}],
               "calls": {D: {"parent": "input", "configuration": {"image_level": {}},
                             "readouts": {"mean_target_dominance": 0.5}},
                         A: {"parent": D, "configuration": {"image_level": {"margin": 0}},
                             "readouts": {"mean_target_dominance": 0.7}}},
               "admissible": [D, A], "failed": [], "admission": {"call": D}, "plans": [plan],
               "call": A, "choice": "agent_choice", "plan_delivery": {"call": A, "delivered": True},
               "final_answers": [{"call": A, "reasoning": f"{A} is best"}], "reasoner_failures": 0,
               "fallback_rounds": 0, "mask_check": mask_check}
    for i in IMAGES:
        _jsonl(run / "shape" / "images" / i / "agent_log.jsonl", log)
        _write(run / "shape" / "images" / i / "agent_details.json", details)
        for pixel in ("tool_cache/crop_0001.tiff", "mask_check/montage_000.png", "tool_cache/crops.zip"):
            _write(run / "shape" / "images" / i / pixel, "pixels")
        _write(run / "maskcrop" / "images" / i / "tool_cache" / "crop.tif", "pixels")
    _write(run / "compare" / "summary.json", {"arms": {}})
    _write(run / "compare" / "per_image.csv", "arm\n")
    _write(run / "compare" / "thumb.png", "pixels")
    job = f"crop2-all-{'200' if new else '100'}"
    _write(root / f"{job}.out", f"hint: x\n== stage=all arm= n_images=2 seed={seed} commit={COMMIT} branches= "
                                f"run_dir=/scratch/results/{name} cores=8\nVALID_RUN_FAIL gpt records=2\n")
    _write(root / f"{job}.err", "VALID_RUN_FAIL 1 problem(s)\n")
    _write(root / f"{job}-shape.log", "WARNING crop-triage: montage judging failed\nshape img_a: ok\n"
                                      "Cell 1: crop shape=(9, 9)\nshape img_b: ok\nwrote records\n")
    _write(root / f"{job}-gpt.log", "gpt img_a: ok\nError in generate method: 502\ngpt img_b: error\n")
    _write(root / "crop2-all-999.out", "== stage=all arm= n_images=2 commit=b branches= run_dir=/s/other_run cores=8\n")
    _write(root / "crop2-all-999-shape.log", "ERROR not this run\nshape img_a: ok\n")
    return run


def _record(out: Path, arm: str, image: str) -> dict:
    return json.loads((out / arm / f"{image}.json").read_text(encoding="utf-8"))


def _missing_text(record: dict) -> str:
    return "\n".join(record["sources"]["missing"])


@pytest.mark.parametrize("names", [("default", "agent_1"), ("c01", "c02")])
def test_new_format_records(tmp_path, names):
    """The run-time recording, with the role names and with the index names of the reported run (ea78bb4b):
    a call's trial index is read from the record's rounds either way."""
    D, A = names
    run = make_run(tmp_path, "aaaaaaaa_n2_s1881694274", new=True, names=names)
    summary = err.export(run)
    out = run / "audit"
    assert summary["n_records"] == 10 and summary["arms"] == list(err.ARMS)
    assert summary["logs"]["jobs"] == ["crop2-all-200"]
    assert any("VALID_RUN_FAIL" in e for e in summary["logs"]["run_events"])
    index = json.loads((out / "index.json").read_text(encoding="utf-8"))
    assert index["schema"] == "cropper-run-record/1" and len(index["records"]) == 10
    for arm in err.ARMS:
        for image in IMAGES:
            r = _record(out, arm, image)
            assert {"schema", "input", "provenance", "config", "steps", "final_output", "score", "cost", "status",
                    "sources"} <= set(r)
            assert r["input"]["seed"] == 1881694274 and r["input"]["selection_index"] == IMAGES.index(image)
            assert r["provenance"]["commit"] == COMMIT and r["provenance"]["host"] == "sh03"
            assert r["sources"]["missing"] == [], _missing_text(r)
            assert all(not p.startswith(str(tmp_path)) for p in r["sources"]["used"])   # relative to the run dir
    gpt = _record(out, "gpt", "img_a")
    assert [s["attempt"] for s in gpt["steps"]] == [1, 2]
    assert gpt["steps"][0]["prompt"] == "Return only a JSON list" and gpt["steps"][0]["images"] == [REF]
    assert gpt["steps"][0]["result"].startswith("rejected") and gpt["steps"][1]["result"] == "accepted"
    assert gpt["status"]["retries"] == 1 and gpt["config"]["params"]["max_tokens"] == 8000
    assert gpt["status"]["log_events"] == []
    failed = _record(out, "gpt", "img_b")                       # a failed image keeps its requests
    assert failed["status"]["status"] == "error" and len(failed["steps"]) == 3
    assert failed["status"]["log_events"] == ["Error in generate method: 502"]
    react = _record(out, "react", "img_a")
    assert [s["step"] for s in react["steps"]] == ["tool_call", "llm_call", "tool_call", "llm_call", "llm_call"]
    assert react["steps"][1]["prompt"] == "You are a ReAct agent" and react["steps"][1]["rationale"] == "T1"
    assert react["steps"][4]["deliver"] == A and react["status"]["retries"] == 1
    assert react["final_output"]["delivered_call"] == A
    shape = _record(out, "shape", "img_a")
    kinds = [s["step"] for s in shape["steps"]]
    assert kinds == ["tool_call", "admission", "llm_call", "rejected_move", "tool_call", "llm_call",
                     "plan_delivery", "choice", "final_choice", "mask_check", "vision_call"]
    q1, q2 = (s for s in shape["steps"] if s["step"] == "llm_call")
    assert q1["round"] == 2 and q1["images"] == [REF] and q1["rationale"] == "contamination is high"
    assert q1["reply"]["moves"] == [{"parent": D}] and q1["usage"] == [{"total_tokens": 70}]
    assert q2["errors"] == [{"error": "try 1: RuntimeError: 502", "reply": None}] and q2["reply"]["deliver"] == A
    call = shape["steps"][4]
    assert call["call"] == A and call["round"] == 2 and call["tool_call"]["change"] == {"margin": 0}
    assert shape["steps"][9]["verdicts"][0]["reject"] is True and shape["steps"][10]["image"] == REF
    assert shape["status"]["retries"] == 1 and shape["final_output"]["delivered_call"] == A
    assert shape["status"]["log_events"] == ["WARNING crop-triage: montage judging failed"]
    assert _record(out, "shape", "img_b")["status"]["log_events"] == []


def test_old_format_degrades_to_sources_missing(tmp_path):
    run = make_run(tmp_path, "bbbbbbbb_n2", new=False)
    out = tmp_path / "audit_out"
    err.export(run, out_dir=out)
    assert not (run / "audit").exists()
    gpt = _record(out, "gpt", "img_a")
    assert gpt["steps"][0]["step"] == "llm_batch" and gpt["status"]["retries"] == 1
    assert "trace.calls" in _missing_text(gpt) and "[llm]: not recorded in this run" in _missing_text(gpt)
    failed = _record(out, "gpt", "img_b")
    assert failed["steps"] == [] and "[trace]: not recorded in this run" in _missing_text(failed)
    react = _record(out, "react", "img_a")
    model = [s for s in react["steps"] if s["step"] == "llm_call"]
    assert all(s["prompt"] == err.NOT_RECORDED for s in model) and model[0]["rationale"] == "T1"
    assert "3 model step(s)" in _missing_text(react)
    shape = _record(out, "shape", "img_a")
    assert all(s["images"] == err.NOT_RECORDED for s in shape["steps"] if s["step"] == "llm_call")
    assert "prompt images of 2 question(s)" in _missing_text(shape)
    assert "mask_check.montages, verdicts" in _missing_text(shape)
    assert shape["steps"][-1]["step"] == "mask_check" and shape["steps"][-1]["verdicts"] == err.NOT_RECORDED
    # the oldest layout: no shape/images, no job logs
    shutil.rmtree(run / "shape" / "images")
    for log in tmp_path.glob("crop2-all-100*"):
        log.unlink()
    err.export(run, out_dir=out)
    shape = _record(out, "shape", "img_a")
    assert [s["step"] for s in shape["steps"]] == ["final_choice"] and shape["steps"][0]["call"] == "c02"
    assert "agent_details.json" in _missing_text(shape) and "job logs" in _missing_text(shape)


def test_logs_in_the_run_dir_and_a_separate_logs_dir(tmp_path):
    """The unified layout: per-arm stage logs in <run_dir>/logs/<arm>.log, the Slurm .out in a separate --logs-dir."""
    run = make_run(tmp_path / "results", "aaaaaaaa_n2_s1881694274", new=True)
    for log in (tmp_path / "results").glob("crop2-all-*"):
        log.unlink()
    _write(run / "logs" / "shape.log", "shape img_a: ok\nWARNING stage log\nshape img_b: ok\n")
    slurm = tmp_path / "slurm_logs"
    _write(slurm / "crop2_300.out", f"== stage=all arm= n_images=2 seed=1 commit={COMMIT} branches= "
                                    f"run_dir=/x/{run.name} cores=8\nVALID_RUN=OK /x\n")
    summary = err.export(run, logs_dir=slurm)
    assert summary["logs"]["jobs"] == ["crop2_300", "logs/<arm>.log"]
    assert summary["logs"]["run_events"] == ["crop2_300.out: VALID_RUN=OK /x"]
    assert _record(run / "audit", "shape", "img_b")["status"]["log_events"] == ["WARNING stage log"]
    dest = err.into_data(run, tmp_path / "data", summary, run / "audit")
    assert {p.name for p in (dest / "logs").iterdir()} == {"crop2_300.out", "shape.log"}


def test_into_data_copies_non_pixel_files_and_is_idempotent(tmp_path):
    root = tmp_path / "results"
    new = make_run(root, "aaaaaaaa_n2_s1881694274", new=True)
    old = make_run(root, "bbbbbbbb_n2", new=False)
    data = tmp_path / "data"
    for _ in range(2):                                          # a re-export replaces the run's copy and rows
        dest = err.into_data(new, data, err.export(new), new / "audit")
    files = sorted(p.relative_to(dest).as_posix() for p in dest.rglob("*") if p.is_file())
    assert not [f for f in files if Path(f).suffix in err.PIXEL_SUFFIXES]
    for expected in ("selection.json", "provenance.json", "gpt/run_meta.json", "shape/records.jsonl",
                     "compare/summary.json", "compare/per_image.csv", "shape/images/img_a/agent_log.jsonl",
                     "shape/images/img_b/agent_details.json", "logs/crop2-all-200.out", "logs/crop2-all-200.err",
                     "logs/crop2-all-200-shape.log", "audit/index.json", "audit/shape/img_a.json"):
        assert expected in files, expected
    assert not any("999" in f for f in files)

    def rows():
        with (data / "cropper-run_image_metrics.csv").open(newline="", encoding="utf-8") as fh:
            return list(csv.DictReader(fh))

    assert len(rows()) == 10 and set(rows()[0]) == set(err.CSV_FIELDS)
    err.into_data(old, data, err.export(old), old / "audit")
    assert len(rows()) == 20 and {r["run_id"] for r in rows()} == {new.name, old.name}
    shape_row = next(r for r in rows() if r["run_id"] == new.name and r["arm"] == "shape")
    assert (shape_row["seed"], shape_row["commit"], shape_row["n_images"]) == ("1881694274", COMMIT, "2")
    assert shape_row["delivery"] == "agent_choice" and shape_row["n_triage_rejected"] == "1"
    index = json.loads((data / "runs" / "index.json").read_text(encoding="utf-8"))
    assert [r["run_id"] for r in index["runs"]] == [old.name, new.name]          # by date
    assert index["runs"][1]["status"] == "failed" and index["runs"][1]["problems"] == ["gpt records=2"]
    seeds = {s["seed"]: s for s in json.loads((data / "seeds.json").read_text(encoding="utf-8"))["seeds"]}
    assert seeds[0]["how"] == "default, the original pilot selection" and seeds[0]["runs"] == [old.name]
    assert seeds[1881694274]["drawn"] == "2026-10-06" and seeds[1881694274]["runs"] == [new.name]
    # seeds.json keeps a hand-written entry; exporting the data copy itself copies nothing
    payload = json.loads((data / "seeds.json").read_text(encoding="utf-8"))
    payload["seeds"][0]["drawn"] = "2026-09-01"
    (data / "seeds.json").write_text(json.dumps(payload), encoding="utf-8")
    copy = data / "runs" / old.name
    err.into_data(copy, data, err.export(copy), copy / "audit")
    seeds = {s["seed"]: s for s in json.loads((data / "seeds.json").read_text(encoding="utf-8"))["seeds"]}
    assert seeds[0]["drawn"] == "2026-09-01" and len(rows()) == 20
    assert (copy / "logs" / "crop2-all-100.out").is_file()    # the copy's own logs are found in <run>/logs
    assert _record(copy / "audit", "shape", "img_a")["status"]["log_events"] == [
        "WARNING crop-triage: montage judging failed"]


# --------------------------------------------------------------------------- #
# the cluster scripts
# --------------------------------------------------------------------------- #
def _bash(script: str, tmp_path: Path, **env) -> subprocess.CompletedProcess:
    path = tmp_path / "t.sh"
    with path.open("w", encoding="utf-8", newline="\n") as fh:
        fh.write(script)
    import os
    return subprocess.run([BASH, str(path)], capture_output=True, text=True,
                          env={**os.environ, **{k: str(v) for k, v in env.items()}}, timeout=60)


def test_the_exporter_is_loaded_by_path_and_never_takes_the_bare_module_name():
    """The segmentation and batch-correction benchmarks have an export_run_records.py too: this benchmark's copy is
    loaded under its own name, so a test session importing theirs by the bare name gets theirs."""
    import sys

    assert err.__name__ == "cropper_export_run_records"
    assert Path(err.__file__).resolve() == (_RUNNERS.parent / "export_run_records.py").resolve()
    assert sys.modules.get("export_run_records") is not err
    assert run_benchmark.PIXEL_SUFFIXES is err.PIXEL_SUFFIXES

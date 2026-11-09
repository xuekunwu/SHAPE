"""export_run_records.py on a synthetic experiment dir: whitelist, caps, key scan, summaries, --check, determinism."""
import csv
import json
import os
from pathlib import Path

import pytest

from benchmarks.full_CPJUMP1 import export_run_records as ex

MB = 1024 * 1024


def put(path: Path, content="{}", binary=False):
    path.parent.mkdir(parents=True, exist_ok=True)
    if binary:
        path.write_bytes(content if isinstance(content, bytes) else b"\x00\x01")
    else:
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8", newline="\n")
    return path


def result(arm, commit, main=0.1):
    metrics = {m: {"main": {"value": main + i}, "guard": {"value": main + i + 0.5}}
               for i, m in enumerate(ex.VERDICT_METRICS)}
    return {"status": "complete", "arm": arm, "provenance": {"commit": commit, "model": "gpt-test"},
            "metrics": {e: {"mAP": 0.01 * (i + 1)} for i, e in enumerate(ex.RETRIEVAL)},
            "full_metrics": {"metrics": metrics}}


def stage_done(seconds, tokens=None, reasoning=None):
    usage = {} if tokens is None else {"total_tokens": tokens}
    if reasoning is not None:
        usage["reasoning_tokens"] = reasoning
    return {"status": "complete", "seconds": seconds, "usage": usage}


def search(delivered, levers, reason, calls, failures=0, errors=(), seconds=10.0, mode="search"):
    return {"status": "searched", "delivered": delivered, "levers": levers, "mode": mode,
            "resolution": {"status": "satisfied", "reason": reason, "fault": None}, "stop_reason": reason,
            "n_calls": calls, "reasoner_failures": failures, "fallback_rounds": 0,
            "execution_errors": list(errors), "seconds": seconds}


def make_arm(arm_dir: Path, arm, commit, shape_like=False, agent=False):
    put(arm_dir / "RESULT.json", result(arm, commit))
    put(arm_dir / "RESULT.md", f"# {arm}\n")
    put(arm_dir / "CONFIG.json", {"arm": arm, "model": "gpt-test", "reasoning": {"stages": {
        "segment": {"effort": "medium", "max_tokens": 4000, "timeout_seconds": 120},
        "embed": {"effort": "high", "max_tokens": 16000, "timeout_seconds": 900}, "cluster": None}}})
    put(arm_dir / "CPJUMP1_COMPLETE", commit + " \n")
    put(arm_dir / "RUN_META.json", {"status": "complete"})
    put(arm_dir / "DELIVERY.json", {"delivered": True})
    put(arm_dir / "SEARCH_WARNINGS.jsonl", '{"w": 1}\n')
    put(arm_dir / "segment_COMPLETE.json", stage_done(100.5, 1000, 300))
    put(arm_dir / "crop_COMPLETE.json", stage_done(50.0, 0))
    put(arm_dir / "embed_COMPLETE.json", stage_done(70.0, 500))
    for name in ("profiles.csv", "states.csv", "umap.csv"):
        put(arm_dir / name, "a,b\n1,2\n")
    put(arm_dir / "cells.json", "[" + "0," * 50 + "0]")                      # per-cell table: never
    put(arm_dir / "development_cells.json", "[]")
    for name in ("embedding.npy", "corrected.npy", "profiles.npy"):
        put(arm_dir / name, b"\x93NUMPY", binary=True)
    put(arm_dir / "profiles.h5ad", b"HDF", binary=True)
    put(arm_dir / "evaluation" / "delivered_cross_guide_cross_plate.csv", "gene,ap\nA,0.1\n")
    put(arm_dir / "evaluation" / "genes.csv", "gene\nA\n")
    put(arm_dir / "evaluation" / "full_metrics_draws.npz", b"PK\x03\x04npz", binary=True)
    put(arm_dir / "evaluation" / "full_metrics_draws.npz.json", {"n": 1000})
    put(arm_dir / "evaluation" / "scratch.npy", b"x", binary=True)
    put(arm_dir / "logs" / "segment.log", "line one\nline two\n")
    put(arm_dir / "logs" / "batch.jsonl", '{"e": 1}\n')
    put(arm_dir / "logs" / "crop" / "img_0000.jsonl", '{"e": 2}\n')                # per-image log: not copied
    put(arm_dir / "ssl" / "call_000.params.json", {"lr": 1})
    put(arm_dir / "ssl" / "call_000.result.json", {"loss": 1})
    put(arm_dir / "ssl" / "call_000.pt", b"w", binary=True)
    put(arm_dir / "ssl" / "cell_state_analysis" / "x.h5ad", b"h", binary=True)
    put(arm_dir / "batch" / "call_000" / "params.json", {"k": 1})
    put(arm_dir / "batch" / "call_000" / "result.json", {"ok": 1})
    put(arm_dir / "batch" / "call_000" / "embedding.npy", b"e", binary=True)
    put(arm_dir / "batch" / "call_000" / "BATCH_CORRECTOR_RESULT.json", {"x": 1})   # not on the whitelist
    put(arm_dir / "inference" / "chunk_00000000.json", {"c": 1})
    put(arm_dir / "private" / "RESULT.json", {"labels": "secret"})
    for stage in ("segment", "crop"):
        put(arm_dir / stage / "img_0000" / "mask.tif", b"II*\x00", binary=True)
        put(arm_dir / stage / "img_0000" / "overlay.png", b"\x89PNG", binary=True)
    if agent:
        for stage in ("segment", "crop"):
            base = arm_dir / "agent" / stage
            put(base / "CALLS.json", [{"call": "c1"}])
            put(base / "ROUNDS.json", [{"round": 1}])
            put(base / "SELECTED.json", {"call": "c1"})
            put(base / "auth" / "config.toml", "token = 1\n")
            put(base / "traces" / "round_000.json", {"t": 1})
        put(arm_dir / "agent" / "embed" / "INFER_CONFIG.json", {"i": 1})
        call = arm_dir / "agent" / "embed" / "calls" / "c1"
        put(call / "ssl_result.json", {"r": 1})
        put(call / "FEEDBACK.json", {"f": 1})
        put(call / "CONFIG.json", {"c": 1})
        put(call / "cells.json", "[]")
        put(call / "best_model.pth", b"w", binary=True)
        put(call / "tool.log", "heavy log\n")
        put(call / "private" / "CONFIG.json", {"p": 1})
    if shape_like:
        put(arm_dir / "segment_COHORT.json", {
            "counts": {"cohort": 1, "per_image_vrr": 1, "calibration": 1},
            "parameter_source": {"img_0000": "cohort", "img_0001": "per_image_vrr", "img_0002": "calibration"}})
        seg = arm_dir / "segment"
        put(seg / "img_0000" / "COHORT_SEARCH.json",
            search("reference", {}, "cohort_applied", 2, seconds=1.0, mode="cohort"))
        put(seg / "img_0001" / "COHORT_SEARCH.json", search("default", {}, "no_tuning_needed", 1, mode="cohort"))
        put(seg / "img_0001" / "SEARCH.json", search("agent_2", {"diameter": 50, "flow_threshold": 0.3},
                                                      "satisfied", 5, failures=1, seconds=20.0))
        put(seg / "img_0002" / "SEARCH.json", search("default", {}, "plateau", 4, errors=[{"question": "repair"}],
                                                      seconds=30.0))
        crop = arm_dir / "crop"
        put(crop / "img_0000" / "SEARCH.json", search("agent_1", {"margin": 2, "margin_mode": "mask_bbox"},
                                                       "satisfied", 3, seconds=5.0))
        put(crop / "img_0001" / "SEARCH.json", search("agent_1", {"margin": 4, "margin_mode": "mask_bbox"},
                                                       "no_valid_move", 3, seconds=7.0))


def make_run(root: Path) -> Path:
    run = root / "full_abc12345_20261012T165818Z_codexfirst"
    make_arm(run / "control", "control", "c0ffee", )
    make_arm(run / "shape", "shape", "c0ffee", shape_like=True)
    make_arm(run / "codex", "codex", "c0ffee", agent=True)
    make_arm(run / "shape_rerun_x_1234", "shape", "d00d", shape_like=True)
    put(run / "shape_rerun_x_1234" / "RERUN.json", {"rerun": 1})
    put(run / "shape_rerun_x_1234" / "RERUN_COMPLETE", "d00d \n")
    failed = run / "shape_failed_gateway_20261012T222009Z"
    put(failed / "RUN_META.json", {"status": "failed", "provenance": {"commit": "c0ffee"}})
    put(failed / "CONFIG.json", {"arm": "shape", "model": "gpt-test"})
    put(failed / "PREFLIGHT.json", {"p": 1})
    put(failed / "logs" / "segment.log", "boom\n")
    put(failed / "shape.err", "Traceback\n")
    put(failed / "segment" / "img_0000" / "SEARCH.json", search("default", {}, "x", 1))
    put(failed / "segment_USAGE.json", {"u": 1})
    put(run / "comparison" / "COMPARISON.json", {"verdict": 1})
    put(run / "comparison" / "comparison.csv", "arm,map\nshape,0.1\n")
    put(run / "comparison" / "COMPARISON_COMPLETE", "c0ffee \n")
    put(run / "comparison_all_arms" / "COMPARISON.json", {"verdict": 2})
    post = run / "posthoc_rerun_main_2f34db04"
    put(post / "POSTHOC_COMPARISON.json", {"p": 1})
    put(post / "POSTHOC_COMPARISON.md", "# posthoc\n")
    put(post / "scores" / "shape_rerun" / "RESULT.json", result("shape", "d00d"))
    put(post / "scores" / "shape_rerun" / "RESULT.md", "# r\n")
    put(post / "scores" / "shape_rerun" / "evaluation" / "full_metrics_draws.npz", b"PK\x03\x04", binary=True)
    put(post / "scores" / "shape_rerun" / "evaluation" / "genes.csv", "gene\nA\n")
    put(run / "compare.log", "compare done\n")
    put(run / "control.out", "")
    put(run / "VM_RESOURCES.txt", "cores 24\n")
    put(run / "shape.noimg.tgz", b"tgz", binary=True)
    return run


@pytest.fixture()
def run(tmp_path):
    return make_run(tmp_path / "vm")


def exported(out: Path) -> set:
    return {p.relative_to(out).as_posix() for p in out.rglob("*") if p.is_file()}


def test_whitelist_and_exclusions(run, tmp_path):
    out = tmp_path / "out"
    manifest = ex.export(run, out)
    got = exported(out)
    assert got == {f["path"] for f in manifest["files"]} | {"MANIFEST.json"}
    # control / shape arm: the records, the evaluation set, the call JSON, the generated summaries
    for rel in ("shape/RESULT.json", "shape/RESULT.md", "shape/CONFIG.json", "shape/CPJUMP1_COMPLETE",
                "shape/RUN_META.json", "shape/DELIVERY.json", "shape/SEARCH_WARNINGS.jsonl",
                "shape/segment_COMPLETE.json", "shape/segment_COHORT.json", "shape/profiles.csv", "shape/states.csv",
                "shape/umap.csv", "shape/evaluation/delivered_cross_guide_cross_plate.csv",
                "shape/evaluation/genes.csv", "shape/evaluation/full_metrics_draws.npz",
                "shape/evaluation/full_metrics_draws.npz.json", "shape/logs/segment.log", "shape/logs/batch.jsonl",
                "shape/ssl/call_000.params.json", "shape/ssl/call_000.result.json", "shape/batch/call_000/params.json",
                "shape/batch/call_000/result.json", "shape/search_summary.json", "control/RESULT.json",
                "shape_rerun_x_1234/RERUN.json", "shape_rerun_x_1234/RERUN_COMPLETE", "summary.csv"):
        assert rel in got, rel
    assert "control/search_summary.json" not in got                      # no SEARCH.json records, no summary
    # the agent arm: stage records and the small per-call JSON only
    for rel in ("codex/agent/segment/CALLS.json", "codex/agent/crop/ROUNDS.json", "codex/agent/crop/SELECTED.json",
                "codex/agent/embed/INFER_CONFIG.json", "codex/agent/embed/calls/c1/ssl_result.json",
                "codex/agent/embed/calls/c1/FEEDBACK.json", "codex/agent/embed/calls/c1/CONFIG.json"):
        assert rel in got, rel
    # comparison, posthoc, driver logs
    for rel in ("compare/comparison/COMPARISON.json", "compare/comparison/comparison.csv",
                "compare/comparison/COMPARISON_COMPLETE", "compare/comparison_all_arms/COMPARISON.json",
                "posthoc/posthoc_rerun_main_2f34db04/POSTHOC_COMPARISON.json",
                "posthoc/posthoc_rerun_main_2f34db04/POSTHOC_COMPARISON.md",
                "posthoc/posthoc_rerun_main_2f34db04/scores/shape_rerun/RESULT.json",
                "posthoc/posthoc_rerun_main_2f34db04/scores/shape_rerun/RESULT.md",
                "logs/compare.log", "logs/VM_RESOURCES.txt"):
        assert rel in got, rel
    # a failed run dir keeps RUN_META.json, CONFIG.json, the error output and its logs
    failed = {r for r in got if r.startswith("shape_failed_gateway_20261012T222009Z/")}
    assert failed == {f"shape_failed_gateway_20261012T222009Z/{n}" for n in
                      ("RUN_META.json", "CONFIG.json", "shape.err", "logs/segment.log")}
    # never: per-cell tables, arrays, images, per-image trees, private paths, non-whitelisted files
    for rel in got:
        parts = rel.split("/")
        assert parts[-1] not in ("cells.json", "development_cells.json"), rel
        assert not rel.endswith((".npy", ".h5ad", ".pt", ".pth", ".tif", ".png")), rel
        assert "private" not in parts and "segment" not in parts[1:2] and "crop" not in parts[1:2], rel
    assert "shape/logs/crop/img_0000.jsonl" not in got
    assert "shape/batch/call_000/BATCH_CORRECTOR_RESULT.json" not in got
    assert "shape/inference/chunk_00000000.json" not in got
    assert "codex/agent/segment/auth/config.toml" not in got and "codex/agent/segment/traces/round_000.json" not in got
    for rel in ("codex/agent/embed/calls/c1/cells.json", "codex/agent/embed/calls/c1/best_model.pth",
                "codex/agent/embed/calls/c1/tool.log", "codex/agent/embed/calls/c1/private/CONFIG.json",
                "shape.noimg.tgz", "logs/shape.noimg.tgz"):
        assert rel not in got, rel
    reasons = {e["path"]: e["reason"] for e in manifest["excluded"]}
    assert reasons["shape/cells.json"] == "per-cell record" and reasons["shape/private/"] == "private"
    assert reasons["shape/segment/"].startswith("per-image tree") and reasons["shape.noimg.tgz"] == "not a driver log"
    assert manifest["excluded"] == sorted(manifest["excluded"], key=lambda e: e["path"])


def test_denylist_is_independent_of_the_whitelist():
    for rel in ("a/cells.json", "a/development_cells.json", "a/x.npy", "a/x.h5ad", "a/m.pt", "a/m.pth", "a/i.tif",
                "a/i.tiff", "a/i.png", "a/private/RESULT.json", "Private/x.json", "a/b/PRIVATE/x.json"):
        assert ex.denied(rel), rel
    for rel in ("a/RESULT.json", "a/evaluation/full_metrics_draws.npz", "a/privately.json"):
        assert not ex.denied(rel), rel


def test_per_file_cap_lists_the_skipped_file_with_its_size(run, tmp_path):
    big = "x" * int(1.5 * MB) + "\n"
    put(run / "shape" / "logs" / "crop.log", big)
    put(run / "codex" / "agent" / "crop" / "CALLS.json", big)
    out = tmp_path / "out"
    manifest = ex.export(run, out, cap_mb=80, file_cap_mb=1)
    skipped = {s["path"]: s for s in manifest["skipped"]}
    assert set(skipped) == {"shape/logs/crop.log", "codex/agent/crop/CALLS.json"}
    for s in skipped.values():
        assert s["bytes"] == len(big) and "per-file cap" in s["reason"]
    assert manifest["totals"]["skipped"] == 2
    assert not (out / "shape/logs/crop.log").exists() and not (out / "codex/agent/crop/CALLS.json").exists()
    assert "shape/logs/crop.log" not in {f["path"] for f in manifest["files"]}


def test_per_call_json_over_one_mb_is_skipped(run, tmp_path):
    put(run / "codex" / "agent" / "embed" / "calls" / "c1" / "CONFIG.json", "y" * (MB + 10))
    manifest = ex.export(run, tmp_path / "out", file_cap_mb=8)
    assert [s["path"] for s in manifest["skipped"]] == ["codex/agent/embed/calls/c1/CONFIG.json"]


def test_total_cap_skips_the_larger_files_and_lists_them(run, tmp_path):
    put(run / "shape" / "logs" / "embed.log", "e" * 3000)
    put(run / "shape" / "logs" / "representation.jsonl", "r" * 5000)
    out = tmp_path / "out"
    full = ex.export(run, tmp_path / "full")
    cap_bytes = full["totals"]["bytes"] - 4000          # the 5000-byte file no longer fits, the small ones do
    manifest = ex.export(run, out, cap_mb=cap_bytes / MB)
    assert manifest["totals"]["bytes"] <= cap_bytes
    skipped = {s["path"]: s for s in manifest["skipped"]}
    assert "shape/logs/representation.jsonl" in skipped and "total cap" in skipped["shape/logs/representation.jsonl"]["reason"]
    assert skipped["shape/logs/representation.jsonl"]["bytes"] == 5000
    assert "shape/RESULT.json" in {f["path"] for f in manifest["files"]}       # small files are kept first
    assert sum(f["bytes"] for f in manifest["files"]) == manifest["totals"]["bytes"]
    assert sum(p.stat().st_size for p in out.rglob("*") if p.is_file() and p.name != "MANIFEST.json") == \
        manifest["totals"]["bytes"]


@pytest.mark.parametrize("planted", ["sk-" + "A1b2C3d4E5f6G7h8", "Authorization: Bearer abc", "LLM_API_KEY=abc123",
                                     "source /home/u/api.env"])
def test_key_scan_fails_naming_the_file_and_writes_nothing(run, tmp_path, planted):
    put(run / "codex" / "logs" / "segment.log", f"ok\n{planted}\nok\n")
    out = tmp_path / "out"
    with pytest.raises(ex.ExportError, match="codex/logs/segment.log"):
        ex.export(run, out)
    assert not out.exists() or not any(out.iterdir())


def test_key_scan_covers_json_extra_logs_and_the_cli(run, tmp_path, capsys):
    put(run / "shape" / "CONFIG.json", {"note": "sk-" + "Z" * 20})
    with pytest.raises(ex.ExportError, match="shape/CONFIG.json"):
        ex.export(run, tmp_path / "o1")
    put(run / "shape" / "CONFIG.json", {"note": "clean"})
    extra = put(tmp_path / "driver.log", "export LLM_API_KEY=abc\n")
    with pytest.raises(ex.ExportError, match="logs/driver.log"):
        ex.export(run, tmp_path / "o2", extra_logs=[extra])
    rc = ex.main(["--run", str(run), "--out", str(tmp_path / "o3"), "--extra-log", str(extra)])
    assert rc == 2 and "logs/driver.log" in capsys.readouterr().err


def test_key_like_text_that_is_not_a_key_passes(run, tmp_path):
    put(run / "shape" / "logs" / "segment.log", "sk-short and LLM_API_KEY= and a bearer token\n")
    ex.export(run, tmp_path / "out")


def test_extra_logs_are_copied_whole_and_collisions_refused(run, tmp_path):
    extra = put(tmp_path / "run_after_codex.log", "a very long line " * 5000 + "\nsecond\n")
    manifest = ex.export(run, tmp_path / "out", extra_logs=[extra])
    assert (tmp_path / "out" / "logs" / "run_after_codex.log").read_bytes() == extra.read_bytes()
    assert [f["kind"] for f in manifest["files"] if f["path"] == "logs/run_after_codex.log"] == ["log"]
    with pytest.raises(ex.ExportError, match="already exists"):
        ex.export(run, tmp_path / "o2", extra_logs=[put(tmp_path / "x" / "compare.log", "z")])
    with pytest.raises(ex.ExportError, match="not a file"):
        ex.export(run, tmp_path / "o3", extra_logs=[tmp_path / "missing.log"])


def test_refuses_a_non_empty_output(run, tmp_path):
    out = put(tmp_path / "out" / "something.txt", "x").parent
    with pytest.raises(ex.ExportError, match="not empty"):
        ex.export(run, out)


def read_rows(out):
    with open(out / "summary.csv", encoding="utf-8", newline="") as fh:
        return {r["run"]: r for r in csv.DictReader(fh)}


def test_summary_csv_has_one_row_per_arm_run_with_absolute_values(run, tmp_path):
    out = tmp_path / "out"
    ex.export(run, out)
    text = (out / "summary.csv").read_bytes()
    assert b"\r" not in text
    rows = read_rows(out)
    assert set(rows) == {"control", "shape", "codex", "shape_rerun_x_1234", "shape_failed_gateway_20261012T222009Z"}
    shape = rows["shape"]
    assert shape["arm"] == "shape" and shape["status"] == "complete" and shape["commit"] == "c0ffee"
    assert float(shape["mAP_main"]) == pytest.approx(0.1) and float(shape["mAP_guard"]) == pytest.approx(0.6)
    assert float(shape["macro_f1_main"]) == pytest.approx(1.1) and float(shape["balanced_accuracy_guard"]) == pytest.approx(2.6)
    for i, e in enumerate(ex.RETRIEVAL):
        assert float(shape[f"retrieval_{e}_mAP"]) == pytest.approx(0.01 * (i + 1))
    assert float(shape["segment_seconds"]) == 100.5 and shape["segment_total_tokens"] == "1000"
    assert shape["segment_reasoning_tokens"] == "300" and shape["crop_total_tokens"] == "0"
    assert shape["crop_reasoning_tokens"] == "" and shape["batch_seconds"] == ""      # no batch_COMPLETE.json
    assert shape["reasoning_segment"] == "medium/4000/120" and shape["reasoning_embed"] == "high/16000/900"
    assert shape["reasoning_crop"] == "" and shape["reasoning_cluster"] == ""
    assert not [c for c in shape if "delta" in c or "vs_control" in c]
    assert rows["shape_rerun_x_1234"]["commit"] == "d00d"          # the rerun's own RESULT.json provenance
    failed = rows["shape_failed_gateway_20261012T222009Z"]
    assert failed["status"] == "failed" and failed["commit"] == "c0ffee" and failed["mAP_main"] == ""


def test_summary_commit_falls_back_to_the_completion_marker(run, tmp_path):
    r = json.loads((run / "codex" / "RESULT.json").read_text(encoding="utf-8"))
    r["provenance"].pop("commit")
    put(run / "codex" / "RESULT.json", r)
    put(run / "codex" / "CPJUMP1_COMPLETE", "feedbeef trailing\n")
    ex.export(run, tmp_path / "out")
    assert read_rows(tmp_path / "out")["codex"]["commit"] == "feedbeef"


def test_search_summary_counts_the_per_image_records(run, tmp_path):
    out = tmp_path / "out"
    ex.export(run, out)
    assert not (out / "shape_failed_gateway_20261012T222009Z" / "search_summary.json").exists()
    s = json.loads((out / "shape" / "search_summary.json").read_text(encoding="utf-8"))
    assert s["run"] == "shape" and set(s["stages"]) == {"segment", "crop"}
    seg, crop = s["stages"]["segment"], s["stages"]["crop"]
    assert seg["images"] == 3 and seg["record_source"] == {"COHORT_SEARCH.json": 1, "SEARCH.json": 2}
    assert seg["resolution_reason"] == {"cohort_applied": 1, "plateau": 1, "satisfied": 1}
    assert seg["stop_reason"] == {"cohort_applied": 1, "plateau": 1, "satisfied": 1}
    assert seg["calls_per_image"] == {"distribution": {"2": 1, "4": 1, "5": 1}, "total": 11, "mean": 3.6667}
    assert seg["delivered_call"] == {"agent_2": 1, "default": 1, "reference": 1}
    assert seg["delivered_kind"] == {"agent": 1, "default": 1, "reference": 1}
    assert seg["delivered_lever_sets"] == {"(none)": 2, "diameter+flow_threshold": 1}
    assert seg["reasoner_failures"] == {"total": 1, "images": 1}
    assert seg["execution_errors"] == {"total": 1, "images": 1, "by_question": {"repair": 1}}
    assert seg["seconds"] == {"total": 51.0, "mean": 17.0}
    assert seg["parameter_source"] == {"calibration": 1, "cohort": 1, "per_image_vrr": 1}
    assert seg["parameter_source_from"] == "segment_COHORT.json"
    assert crop["images"] == 2 and crop["record_source"] == {"SEARCH.json": 2}
    assert crop["resolution_reason"] == {"no_valid_move": 1, "satisfied": 1}
    assert crop["delivered_lever_sets"] == {"margin+margin_mode": 2} and "parameter_source" not in crop
    assert crop["execution_errors"] == {"total": 0, "images": 0, "by_question": {}}


def test_search_summary_parameter_source_falls_back_to_segmentation_json(run, tmp_path):
    (run / "shape" / "segment_COHORT.json").unlink()
    put(run / "shape" / "segmentation.json", [{"image_id": "img_0000", "parameter_source": "cohort"},
                                              {"image_id": "img_0001", "parameter_source": "cohort"},
                                              {"image_id": "img_0002"}])
    ex.export(run, tmp_path / "out")
    seg = json.loads((tmp_path / "out" / "shape" / "search_summary.json").read_text(encoding="utf-8"))["stages"]["segment"]
    assert seg["parameter_source"] == {"cohort": 2} and seg["parameter_source_from"] == "segmentation.json"


def test_per_image_records_are_not_copied(run, tmp_path):
    out = tmp_path / "out"
    ex.export(run, out)
    assert not any(p.name in ("SEARCH.json", "COHORT_SEARCH.json") for p in out.rglob("*"))


def test_check_passes_and_detects_modified_missing_and_unlisted_files(run, tmp_path, capsys):
    out = tmp_path / "out"
    assert ex.main(["--run", str(run), "--out", str(out)]) == 0
    assert ex.check(out) == [] and ex.main(["--check", str(out)]) == 0
    assert "OK" in capsys.readouterr().out
    put(out / "README.md", "hand written\n")                                   # the README is not checked
    put(out / "analysis" / "pairwise.json", "[]")                              # nor is a derived analysis/ folder
    assert ex.check(out) == []
    target = out / "shape" / "RESULT.json"
    original = target.read_bytes()
    target.write_bytes(original.replace(b"complete", b"c0mplete"))              # same size, other content
    assert ex.check(out) == ["modified (sha256): shape/RESULT.json"]
    assert ex.main(["--check", str(out)]) == 1 and "shape/RESULT.json" in capsys.readouterr().out
    target.write_bytes(original + b" ")
    assert ex.check(out) == [f"modified (size {len(original) + 1} != {len(original)}): shape/RESULT.json"]
    target.write_bytes(original)
    (out / "summary.csv").unlink()
    put(out / "shape" / "extra.json", "{}")
    assert ex.check(out) == ["missing: summary.csv", "unlisted: shape/extra.json"]
    (out / "MANIFEST.json").unlink()
    assert "missing or not a manifest" in ex.check(out)[0]


def test_two_exports_have_identical_manifests_and_trees(run, tmp_path):
    ex.export(run, tmp_path / "a")
    for p in run.rglob("*"):                                                    # other mtimes, same content
        os.utime(p, (1_000_000_000, 1_000_000_000))
    ex.export(run, tmp_path / "b")
    assert (tmp_path / "a" / "MANIFEST.json").read_bytes() == (tmp_path / "b" / "MANIFEST.json").read_bytes()
    assert exported(tmp_path / "a") == exported(tmp_path / "b")
    for rel in exported(tmp_path / "a"):
        assert (tmp_path / "a" / rel).read_bytes() == (tmp_path / "b" / rel).read_bytes(), rel
    manifest = json.loads((tmp_path / "a" / "MANIFEST.json").read_text(encoding="utf-8"))
    paths = [f["path"] for f in manifest["files"]]
    assert paths == sorted(paths) and manifest["source"] == run.name
    assert b"\r" not in (tmp_path / "a" / "MANIFEST.json").read_bytes()


def test_cli_prints_the_totals(run, tmp_path, capsys):
    assert ex.main(["--run", str(run), "--out", str(tmp_path / "out"), "--cap-mb", "10", "--file-cap-mb", "2"]) == 0
    assert "exported" in capsys.readouterr().out
    with pytest.raises(SystemExit):
        ex.main(["--run", str(run)])
    assert ex.main(["--run", str(tmp_path / "missing"), "--out", str(tmp_path / "o2")]) == 2

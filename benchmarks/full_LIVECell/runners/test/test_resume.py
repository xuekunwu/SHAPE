"""Continuing a failed Codex run from its completed segment stage (harness/resume.py), on a synthetic failed run."""
import hashlib
import json
from pathlib import Path
from types import SimpleNamespace

import pytest

from benchmarks.full_CPJUMP1.harness import cli as base
from benchmarks.full_CPJUMP1.harness import rerun as posthoc
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
from benchmarks.full_LIVECell.harness import resume
from benchmarks.full_LIVECell.harness.spec import LIVECELL

N = LIVECELL.n_images
IMAGES = [f"img_{i:04d}" for i in range(N)]
SOURCE_COMMIT, NEW_COMMIT = "2" * 40, "3" * 40
MODEL = "gpt-5.6-luna"


def put(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(obj if isinstance(obj, str) else json.dumps(obj, sort_keys=True), encoding="utf-8", newline="\n")
    return path


def failed_source(root: Path, prep: Path) -> Path:
    """A Codex run whose segment stage completed and whose crop failed: partial agent/crop and crop/, no delivery."""
    put(prep / "PREPARE.json", {"status": "complete"})
    usage = {"total_tokens": 1000, "calls": 7, "usage_observed_fields": ["total_tokens"]}
    config = {"arm": "codex", "dataset": "LIVECell", "model": MODEL, "seed": 0, "cores": 1, "budget_hours": 11,
              "train_cells_per_site": 128, "inference_chunk": 2048, "images": IMAGES, "smoke": False,
              "assets_root": "/assets", "deadline": 1.0, "reasoning": {"protocol": "p", "stages": {}}}
    put(root / "CONFIG.json", config)
    put(root / "PREFLIGHT.json", {"commit": SOURCE_COMMIT, "branch": "e2e", "model": MODEL})
    put(root / "RUN_META.json", {"status": "failed", "arm": "codex", "error": "RuntimeError: Codex CLI failed",
                                 "prepare_sha256": digest(prep / "PREPARE.json"), "config": config})
    put(root / "segment_COMPLETE.json", {"status": "complete", "stage": "segment", "seconds": 2820.0, "usage": usage})
    put(root / "segment_USAGE.json", usage)
    stage, rows, calls, deliveries = root / "agent" / "segment", [], {}, {}
    for image in IMAGES:
        folder = stage / "waves" / "w000" / image / "c0"
        mask = folder / f"cell_mask_{image}.tif"
        mask.parent.mkdir(parents=True)
        mask.write_bytes(hashlib.sha256(image.encode()).digest())
        sha = digest(mask)
        rows.append({"image_id": image, "mask": str(mask), "sha256": sha, "min_area": 15, "n_cells": 3, "call": "c0"})
        calls[image] = {"c0": {"status": "complete", "folder": f"waves/w000/{image}/c0",
                               "artifacts": {mask.name: sha}}}
        deliveries[image] = {"call": "c0", "source": "agent", "session": "group"}
    put(root / "segmentation.json", rows)
    put(stage / "CALLS.json", calls)
    put(stage / "ROUNDS.json", [{"session": "group", "round": 0, "action": "finish", "outcome": "finished"}])
    put(stage / "SELECTED.json", {"deliveries": deliveries})
    put(stage / "auth" / "config.toml", "secret-looking home\n")
    put(root / "agent" / "crop" / "ROUNDS.json", [{"partial": True}])
    put(root / "agent" / "crop" / "calls" / "c1" / "x.json", {})
    put(root / "crop" / "img_0000" / "cell.tif", "partial crop")
    return root


@pytest.fixture
def failed(tmp_path):
    return failed_source(tmp_path / "exp" / "codex", tmp_path / "prep"), tmp_path / "prep"


def tree_digest(root: Path) -> dict:
    return {p.relative_to(root).as_posix(): digest(p) for p in sorted(root.rglob("*")) if p.is_file()}


def test_a_failed_source_with_a_complete_segment_stage_is_accepted(failed):
    source, prep = failed
    found = resume.check_source(source, MODEL, prep)
    assert found["commit"] == SOURCE_COMMIT and found["segment"]["seconds"] == 2820.0


@pytest.mark.parametrize("damage,match", [
    ("no_segment_complete", "segment_COMPLETE.json is missing"),
    ("mask_changed", "missing or changed"),
    ("later_complete", "a later stage is complete"),
    ("not_failed", "not a failed run"),
    ("other_model", "is not --model"),
    ("usage_differs", "not complete with its usage"),
    ("call_artifact_changed", "artifact changed"),
])
def test_a_source_that_is_not_a_clean_segment_failure_is_refused(failed, damage, match):
    source, prep = failed
    stage = source / "agent" / "segment"
    first = IMAGES[0]
    if damage == "no_segment_complete":
        (source / "segment_COMPLETE.json").unlink()
    elif damage == "mask_changed":
        (stage / "waves" / "w000" / first / "c0" / f"cell_mask_{first}.tif").write_bytes(b"changed")
    elif damage == "later_complete":
        put(source / "crop_COMPLETE.json", {"status": "complete", "stage": "crop"})
    elif damage == "not_failed":
        meta = read_json(source / "RUN_META.json")
        write_json(source / "RUN_META.json", {**meta, "status": "complete"})
    elif damage == "other_model":
        config = read_json(source / "CONFIG.json")
        write_json(source / "CONFIG.json", {**config, "model": "other"})
    elif damage == "usage_differs":
        put(source / "segment_USAGE.json", {"total_tokens": 1})
    else:       # the delivered call's recorded artifact hash no longer matches (the mask check is satisfied)
        calls = read_json(stage / "CALLS.json")
        calls[first]["c0"]["artifacts"][f"cell_mask_{first}.tif"] = "0" * 64
        write_json(stage / "CALLS.json", calls)
    with pytest.raises(ValueError, match=match):
        resume.check_source(source, MODEL, prep)


def test_only_the_segment_stage_is_placed_rewired_and_recorded(failed, tmp_path):
    source, prep = failed
    before = tree_digest(source)
    out = tmp_path / "exp" / "codex_resume_33333333"
    out.mkdir(parents=True)
    record = resume.reuse_segment(resume.check_source(source, MODEL, prep), out)
    assert tree_digest(source) == before                      # the source is only read
    placed = tree_digest(out)
    assert not any(name.startswith(("agent/crop", "crop/", "agent/segment/auth")) for name in placed)
    assert {"segment_COMPLETE.json", "segment_USAGE.json", "segmentation.json", "RERUN.json",
            "agent/segment/CALLS.json", "agent/segment/ROUNDS.json", "agent/segment/SELECTED.json"} <= set(placed)
    rows = read_json(out / "segmentation.json")
    assert len(rows) == N and all(Path(r["mask"]).is_relative_to(out) and digest(Path(r["mask"])) == r["sha256"]
                                  for r in rows)
    assert record["reused_stages"] == ["segment"] and record["source_commit"] == SOURCE_COMMIT
    assert "prompt-size limit" in record["reason"] and record["not_placed_irregular"] == []
    mask_rel = f"agent/segment/waves/w000/{IMAGES[0]}/c0/cell_mask_{IMAGES[0]}.tif"
    assert record["files"][mask_rel]["method"] in ("hardlink", "copy")
    assert record["files"]["agent/segment/CALLS.json"]["method"] == "copy"
    assert record["rewired"]["segmentation.json"]["sha256"] == digest(out / "segmentation.json")


class Delivered:
    """Stands in for the worker stages and the delivery of `base`: the rest of the continuation runs for real."""

    def __init__(self):
        self.marker = self.control = None

    def run_preflight(self, args, spec):
        provenance = {"commit": NEW_COMMIT, "branch": "e2e", "model": MODEL, "gpu": "g", "assets": {}}
        write_json(args.out / "PREFLIGHT.json", provenance)
        return provenance

    def run_stages(self, args, spec, config, record, stages):
        assert list(stages) == ["crop", "embed", "batch", "cluster"] and config["deadline"] > 0
        for stage in stages:
            usage = {"total_tokens": 10, "calls": 1}
            put(args.out / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage, "seconds": 5.0,
                                                      "usage": usage})
            put(args.out / f"{stage}_USAGE.json", usage)
            record["stages"][stage] = read_json(args.out / f"{stage}_COMPLETE.json")

    def deliver(self, args, spec, record, provenance, marker, control=None):
        self.marker, self.control = marker, control
        names = ["segmentation.json", "RERUN.json", *[p.name for p in args.out.glob("*_COMPLETE.json")],
                 *[p.name for p in args.out.glob("*_USAGE.json")]]
        names += [p.relative_to(args.out).as_posix() for p in (args.out / "agent").rglob("*.json")
                  if p.name in ("CALLS.json", "ROUNDS.json", "SELECTED.json")]
        delivery = {"status": "complete", "arm": args.arm, "provenance": provenance, "prepare_sha256":
                    record["prepare_sha256"], "smoke": False,
                    "artifacts": {n: digest(args.out / n) for n in sorted(set(names))}}
        write_json(args.out / "DELIVERY.json", delivery)
        record["status"] = "complete"
        write_json(args.out / "RUN_META.json", record)
        (args.out / marker).write_text(args.expected_commit + "\n", encoding="ascii", newline="\n")


def run_resume(monkeypatch, source, prep, out, **override):
    delivered = Delivered()
    for name in ("run_preflight", "run_stages", "deliver"):
        monkeypatch.setattr(base, name, getattr(delivered, name))
    monkeypatch.setattr(posthoc, "verify_run", lambda *a, **k: {"path": a[0]})      # the control
    args = SimpleNamespace(source=source, out=out, prep=prep, control=out.parent / "control", model=MODEL,
                           arm="codex", seed=0, cores=None, train_cells_per_site=128, inference_chunk=2048,
                           budget_hours=11.0, assets_root=Path("/assets"), expected_commit=NEW_COMMIT,
                           expected_branch="e2e", reason=resume.REASON, from_stage="crop", **override)
    resume.rerun(args)
    return delivered


def test_the_continuation_runs_the_later_stages_and_is_marked_with_the_rerun_marker(failed, tmp_path, monkeypatch):
    source, prep = failed
    out = tmp_path / "exp" / "codex_resume_33333333"
    delivered = run_resume(monkeypatch, source, prep, out)
    assert delivered.marker == base.RERUN_MARKER and delivered.control == out.parent / "control"
    assert (out / base.RERUN_MARKER).read_text(encoding="ascii").strip() == NEW_COMMIT
    assert not (out / "LIVECELL_COMPLETE").exists()
    config = read_json(out / "CONFIG.json")
    info = config["rerun"]
    assert info["source_commit"] == SOURCE_COMMIT and info["commit"] == NEW_COMMIT
    assert info["reused_stages"] == ["segment"] and info["rerun_stages"] == ["crop", "embed", "batch", "cluster"]
    assert info["budget"]["reused_stage_seconds"] == 2820.0 and info["budget"]["budget_hours"] == 11.0
    assert info["budget"]["hours"] == pytest.approx(11 - 2820 / 3600) and config["budget_hours"] == info["budget"]["hours"]
    assert (config["seed"], config["cores"], config["train_cells_per_site"], config["inference_chunk"]) == (0, 1, 128, 2048)
    assert not (out / "agent" / "crop").exists() and not (out / "crop").exists()
    assert read_json(source / "RUN_META.json")["status"] == "failed"         # the source is untouched


def test_settings_that_differ_from_the_source_and_a_missing_source_stage_refuse_before_anything_is_written(
        failed, tmp_path, monkeypatch):
    source, prep = failed
    out = tmp_path / "exp" / "codex_resume_33333333"
    with pytest.raises(ValueError, match="--seed 1 differs"):
        resume.rerun(SimpleNamespace(source=source, out=out, prep=prep, control=tmp_path, model=MODEL, arm="codex",
                                     seed=1, cores=None, train_cells_per_site=128, inference_chunk=2048,
                                     budget_hours=11.0))
    assert not out.exists()
    with pytest.raises(FileExistsError):
        run_resume(monkeypatch, source, prep, source.parent)


def test_the_post_hoc_comparison_records_the_reused_stage_and_the_commits(failed, tmp_path, monkeypatch):
    source, prep = failed
    out = tmp_path / "exp" / "codex_resume_33333333"
    real_verify = posthoc.verify_run
    run_resume(monkeypatch, source, prep, out)

    def runs(arm_dir, arm):
        arm_dir.mkdir(parents=True)
        for stage in base.STAGES:
            put(arm_dir / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage, "seconds": 1.0,
                                                     "usage": {"total_tokens": 1, "calls": 1}})
        return {"path": arm_dir, "config": {"arm": arm, "seed": 0, "cores": 1, "budget_hours": 11, "images": IMAGES,
                                            "train_cells_per_site": 128, "inference_chunk": 2048, "model": MODEL},
                "commit": "1" * 40, "delivery_sha256": "d" * 64}

    stubs = {"control": runs(tmp_path / "exp" / "control", "control"), "shape": runs(tmp_path / "exp" / "shape", "shape")}
    monkeypatch.setattr(posthoc, "verify_run", lambda run, arm, *a, **k: stubs[arm] if arm in stubs else real_verify(
        run, arm, *a, **k))      # control and SHAPE are stubs; the continuation is verified for real
    verdict = {"delta": 0.01, "se": 0.01, "ci95": [0, 0], "guard_delta": 0.0, "verdict": "BETTER"}
    metrics = {m: {"main": {"value": 0.5}, "vs_control": verdict} for m in ("mAP", "macro_f1", "balanced_accuracy")}
    import benchmarks.full_LIVECell.harness.score as score_module
    monkeypatch.setattr(score_module, "score", lambda *a, **k: {"full_metrics": {"metrics": metrics}})
    summary = resume.compare(prep, out, tmp_path / "exp" / "control", tmp_path / "exp" / "shape",
                             tmp_path / "exp" / "comparison_rerun")
    assert summary["strict_comparison"] is False and summary["reused_stages"] == ["segment"]
    assert summary["commits"] == {"control": "1" * 40, "shape": "1" * 40, "codex_resumed": NEW_COMMIT}
    assert summary["source"]["commit"] == SOURCE_COMMIT and summary["source"]["status"] == "failed"
    stages = summary["runs"]["codex_resumed"]["stages"]
    assert stages["segment"]["ran"] == "source run" and stages["segment"]["commit"] == SOURCE_COMMIT
    assert stages["crop"]["ran"] == "this run" and stages["crop"]["commit"] == NEW_COMMIT
    assert "budget_hours" in summary["config_differences"]
    assert (tmp_path / "exp" / "comparison_rerun" / "POSTHOC_COMPARISON.json").is_file()

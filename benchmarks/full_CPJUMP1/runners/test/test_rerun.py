"""Post-hoc partial reruns (cli.rerun, harness/rerun.py) and their comparison (compare-rerun), on CPU stand-ins."""
import json
import os
import shutil
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_CPJUMP1.harness import cli, compare, data, pipeline, rerun as posthoc
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
from benchmarks.full_CPJUMP1.harness.pipeline import SSL_EXPERT_CONFIG, reasoning_record

from .test_data import fixed_manifest
from .test_score import complete_design

OLD, NEW = "b" * 40, "a" * 40                   # the source runs' commit, the rerun's
REASONER_ENV = ("SHAPE_REASONING_EFFORT", "SHAPE_LLM_TIMEOUT")
HIGH = {"effort": "high", "max_tokens": 16000, "timeout_seconds": 900}
CLEAN = {"admission": {"decision": "SKIP"}, "reasoner_failures": 0, "fallback_rounds": 0, "execution_errors": [],
         "stop_reason": "satisfied"}
SECONDS = {"segment": 3600, "crop": 7200, "embed": 600, "batch": 60, "cluster": 5}
UNBOUND = ("raw_crops", "delivered")            # crop files: bound by cells.json's sha256, as in a real delivery


def design(root):
    """complete_design's prepared data (1000 views, 500 wells) with the public fields the stages read, and its
    delivered outputs (a gene-perfect representation) as a source of cells, vectors and profiles."""
    (root / "design").mkdir()
    prep, base = complete_design(root / "design")
    wells = {f"{w['plate']}_{w['well']}": w for w in fixed_manifest()["wells"]}
    rows = read_json(prep / "public" / "images.json")
    for row in rows:
        w = wells[row["well"]]
        row.update(plate=w["plate"], usage=w["usage"], negative_control=w["pert_type"] == "negcon",
                   calibration_group=w["guide_sequence"] if w["role"] == "calibration" else "")
    write_json(prep / "public" / "images.json", rows)
    record = read_json(prep / "PREPARE.json")
    record["images_sha256"] = digest(prep / "public" / "images.json")
    write_json(prep / "PREPARE.json", record)
    return prep, base


def finish(run, prep, arm, commit, config, branch="e2e", meta=None):
    """A complete full run's records, delivery and marker around the files already in `run`."""
    provenance = {"branch": branch, "commit": commit, "model": config.get("model"), "gpu": "test-gpu",
                  "assets": {"sha": "assets"}}
    write_json(run / "CONFIG.json", config)
    write_json(run / "PREFLIGHT.json", provenance)
    write_json(run / "RUN_META.json", {"status": "complete", "arm": arm, "config": config, "provenance": provenance,
                                       **(meta or {})})
    for stage in cli.STAGES:
        tokens = 0 if arm == "control" or stage == "cluster" else 100
        usage = {"total_tokens": tokens, "prompt_tokens": tokens, "completion_tokens": 0, "calls": 1}
        write_json(run / f"{stage}_USAGE.json", usage)
        write_json(run / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage, "seconds": SECONDS[stage],
                                                    "usage": usage})
    artifacts = {p.relative_to(run).as_posix(): digest(p) for p in sorted(run.rglob("*"))
                 if p.is_file() and p.name not in ("DELIVERY.json", "CPJUMP1_COMPLETE")
                 and not set(UNBOUND) & set(p.relative_to(run).parts)}
    write_json(run / "DELIVERY.json", {"status": "complete", "arm": arm, "smoke": False, "provenance": provenance,
                                       "prepare_sha256": digest(prep / "PREPARE.json"), "artifacts": artifacts})
    (run / "CPJUMP1_COMPLETE").write_text(commit + "\n", encoding="ascii")
    return run


def outputs(run, base):
    """The delivered stage outputs every scored fixture run carries."""
    run.mkdir(parents=True)
    for name in ("profiles.csv", "profiles.npy", "corrected.npy", "embedding.npy", "excluded_cells.json"):
        shutil.copyfile(base / name, run / name)
    write_json(run / "SSL_SELECTION.json", {"mode": "zero_shot", "checkpoint": None})
    write_json(run / "batch_SEARCH.json", CLEAN)
    (run / "profiles.h5ad").write_bytes(b"h5ad")
    (run / "states.csv").write_text("well,state\n", encoding="ascii")


def config_of(prep, arm, images, **extra):
    return {"arm": arm, "model": None if arm == "control" else cli.ARM_MODELS[arm], "seed": 0, "cores": 1,
            "budget_hours": 11, "train_cells_per_site": 4, "inference_chunk": 2048, "images": images,
            "smoke": False, "assets_root": str(prep), **extra}


def plain_run(root, prep, base, arm):
    """A complete control or agent-arm run (its stage outputs: the design's)."""
    run = root / arm
    outputs(run, base)
    shutil.copyfile(base / "cells.json", run / "cells.json")
    write_json(run / "segmentation.json", [])
    images = [r["image_id"] for r in read_json(prep / "public" / "images.json")]
    return finish(run, prep, arm, OLD, config_of(prep, arm, images))


def source_run(root, prep, base, rows=None):
    """A complete SHAPE run at OLD: per-image masks under segment/, raw and display crops under crop/ (unbound, as a
    real delivery leaves them), cells.json naming them, search records, and warnings of several stages."""
    run = root / "shape"
    outputs(run, base)
    rows = rows or read_json(prep / "public" / "images.json")
    usage = {r["image_id"]: r["usage"] for r in rows}
    segmentation = []
    for r in rows:
        ident = r["image_id"]
        mask = run / "segment" / ident / f"cell_mask_{ident}.tif"
        mask.parent.mkdir(parents=True)
        mask.write_bytes(f"mask {ident}".encode())
        segmentation.append({"image_id": ident, "mask": str(mask.resolve()), "sha256": digest(mask), "min_area": 15,
                             "n_cells": 2, "parameter_source": "per_image_vrr"})
    write_json(run / "segmentation.json", segmentation)
    # One cell per site (the design's first): every site keeps a retained cell, half the crop files to write.
    design_cells = read_json(base / "cells.json")
    keep = [i for i, c in enumerate(design_cells) if c["image_id"] in usage and c["cell_id"].endswith("_0")]
    np.save(run / "embedding.npy", np.load(base / "embedding.npy")[keep])
    cells = []
    for index, cell in enumerate(design_cells[i] for i in keep):
        folder = run / "crop" / cell["image_id"]
        raw, display = folder / "raw_crops" / f"cell_{index:06d}.ome.tif", folder / "delivered" / f"{index}.tif"
        for path, text in ((raw, f"raw {cell['cell_id']}"), (display, f"display {cell['cell_id']}")):
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(text.encode())
        cells.append({**cell, "usage": usage[cell["image_id"]], "area": 100, "crop_path": str(raw.resolve()),
                      "sha256": digest(raw), "display_crop_path": str(display.resolve()), "crop_bbox": [0, 0, 8, 8]})
    write_json(run / "cells.json", cells)
    for r in rows[:3]:
        write_json(run / "segment" / r["image_id"] / "SEARCH.json", CLEAN)
        write_json(run / "crop" / r["image_id"] / "SEARCH.json", CLEAN)
    write_json(run / "representation_SEARCH.json", CLEAN)
    (run / "SEARCH_WARNINGS.jsonl").write_text(
        "".join(json.dumps({"stage": s, "image_id": None, "reasoner_failures": 1}) + "\n"
                for s in ("segment", "crop", "embed", "batch")), encoding="utf-8", newline="\n")
    images = [r["image_id"] for r in read_json(prep / "public" / "images.json")]
    return finish(run, prep, "shape", OLD, config_of(prep, "shape", images))


def snapshot(root):
    """Every file's size and modification time (any write changes them; cheaper than hashing every file)."""
    return {p.relative_to(root).as_posix(): (p.stat().st_size, p.stat().st_mtime_ns)
            for p in sorted(root.rglob("*")) if p.is_file()}


# ---------------------------------------------------------------- the CLI


def test_the_rerun_and_compare_rerun_commands():
    base = ["rerun", "--source", "s", "--from-stage", "embed", "--arm", "shape", "--prep", "p", "--assets-root", "a",
            "--out", "o", "--expected-commit", NEW, "--expected-branch", "main"]
    args = cli.parser().parse_args(base)
    assert (args.budget_hours, args.cores, args.control, args.reason) == (None, None, None, cli.RERUN_REASON)
    assert args.model == "gpt-5.6-luna" and args.expected_branch == "main"
    for flag, value in (("--arm", "codex"), ("--arm", "control"), ("--from-stage", "crop"), ("--from-stage", "batch")):
        argv = list(base)
        argv[argv.index(flag) + 1] = value
        with pytest.raises(SystemExit):
            cli.parser().parse_args(argv)
    args = cli.parser().parse_args(["compare-rerun", "--prep", "p", "--rerun", "r", "--control", "c",
                                    "--arm", "codex=x", "--arm", "claude=y", "--out", "o"])
    assert args.arm == ["codex=x", "claude=y"] and args.source is None
    assert set(posthoc.named_runs(args.arm)) == {"codex", "claude"}
    for bad in (["shape=x"], ["control=x"], ["codex"], ["codex=x", "codex=y"]):
        with pytest.raises(ValueError, match="NAME=DIR"):
            posthoc.named_runs(bad)


# ---------------------------------------------------------------- refusals


@pytest.fixture
def records_only(tmp_path):
    """A complete SHAPE run that holds only its records (refusals happen before any artifact is read)."""
    tmp_path = tmp_path.resolve()
    prep = tmp_path / "prep"
    prep.mkdir()
    write_json(prep / "PREPARE.json", {"status": "complete"})
    run = tmp_path / "shape"
    run.mkdir()
    write_json(run / "representation_SEARCH.json", CLEAN)
    finish(run, prep, "shape", OLD, config_of(prep, "shape", [f"img_{i:04d}" for i in range(1000)]))
    return prep, run


def edit(path, **changes):
    write_json(path, {**read_json(path), **changes})


def test_a_complete_source_is_accepted(records_only):
    prep, run = records_only
    found = posthoc.check_source(run, "shape", "gpt-5.6-luna", prep)
    assert found["commit"] == OLD and found["config_sha256"] == digest(run / "CONFIG.json")


@pytest.mark.parametrize("case,match", [
    ("no_marker", "not complete: no CPJUMP1_COMPLETE"), ("incomplete", "not complete"),
    ("other_arm", "not a shape run"), ("smoke", "smoke"), ("other_prep", "another preparation"),
    ("is_rerun", "post-hoc rerun"), ("rerun_config", "post-hoc rerun"), ("marker_commit", "marker commit"),
    ("tampered", "artifact changed"), ("model", "is not --model"), ("views", "1000 views"),
    ("unclean_search", "fallback_rounds")])
def test_the_rerun_refuses_a_source_that_is_not_a_complete_full_run_of_the_arm(records_only, case, match):
    prep, run = records_only
    if case == "no_marker":
        (run / "CPJUMP1_COMPLETE").unlink()
    elif case == "incomplete":
        edit(run / "DELIVERY.json", status="failed")
    elif case == "other_arm":
        for name in ("DELIVERY", "RUN_META", "CONFIG"):
            edit(run / f"{name}.json", arm="control")
    elif case == "smoke":
        edit(run / "DELIVERY.json", smoke=True)
    elif case == "other_prep":
        write_json(prep / "PREPARE.json", {"status": "complete", "other": True})
    elif case == "is_rerun":
        (run / cli.RERUN_MARKER).write_text(OLD, encoding="ascii")
    elif case == "rerun_config":
        edit(run / "CONFIG.json", rerun={"source_run": "x"})
    elif case == "marker_commit":
        (run / "CPJUMP1_COMPLETE").write_text(NEW, encoding="ascii")
    elif case == "tampered":
        write_json(run / "representation_SEARCH.json", {**CLEAN, "stop_reason": "edited"})
    elif case == "model":
        finish(run, prep, "shape", OLD, {**read_json(run / "CONFIG.json"), "model": "claude-sonnet-5"})
    elif case == "views":
        finish(run, prep, "shape", OLD, {**read_json(run / "CONFIG.json"), "images": ["img_0000"]})
    elif case == "unclean_search":
        write_json(run / "representation_SEARCH.json", {**CLEAN, "fallback_rounds": 1})
        finish(run, prep, "shape", OLD, read_json(run / "CONFIG.json"))
    before = snapshot(run)
    with pytest.raises((ValueError, RuntimeError), match=match):
        posthoc.check_source(run, "shape", "gpt-5.6-luna", prep)
    assert snapshot(run) == before


def test_the_rerun_refuses_an_existing_output_before_anything_else(tmp_path):
    (tmp_path / "out").mkdir()
    args = SimpleNamespace(out=tmp_path / "out")
    with pytest.raises(FileExistsError):
        cli.rerun(args)


def test_budget_and_resources(records_only):
    _, run = records_only
    found = {"path": run, "config": read_json(run / "CONFIG.json")}
    assert posthoc.budget(found, ("segment", "crop"), None) == {
        "hours": 8.0, "rule": "source budget_hours less the reused stages' seconds", "source_budget_hours": 11,
        "reused_stage_seconds": 10800.0}
    assert posthoc.budget(found, ("segment", "crop"), 2.5)["hours"] == 2.5
    for bad in (0, -1, float("inf")):
        with pytest.raises(ValueError, match="positive"):
            posthoc.budget(found, ("segment", "crop"), bad)
    from shape.agent.verification.base import available_cores
    assert posthoc.resources(None, 1) == {"cores": 1, "available_cores": available_cores(), "source": "source_run"}
    assert posthoc.resources(1, 64)["source"] == "flag"
    with pytest.raises(ValueError, match="cores this process may use"):
        posthoc.resources(available_cores() + 1, 1)


# ---------------------------------------------------------------- reuse of the upstream artifacts


@pytest.fixture
def small_source(tmp_path):
    """A SHAPE source run on three of the design's views (reuse_upstream does not check the view count)."""
    tmp_path = tmp_path.resolve()
    prep, base = design(tmp_path)
    rows = read_json(prep / "public" / "images.json")[:3]
    return source_run(tmp_path, prep, base, rows), tmp_path


def found(run):
    return {"path": run, "delivery": read_json(run / "DELIVERY.json")}


@pytest.mark.parametrize("linkable", [True, False])
def test_reuse_places_rewires_and_records_the_upstream_artifacts(small_source, monkeypatch, linkable):
    source, root = small_source
    if not linkable:                                       # another filesystem: copies, never symlinks
        monkeypatch.setattr(posthoc.os, "link", lambda src, dst: (_ for _ in ()).throw(OSError("cross-device")))
    before = snapshot(source)
    out = root / "rerun"
    out.mkdir()
    record = posthoc.reuse_upstream(found(source), out, ("segment", "crop"))
    assert snapshot(source) == before
    delivered = read_json(source / "DELIVERY.json")["artifacts"]
    owned = {rel for rel in delivered if posthoc.owner(rel, ("segment", "crop"))}
    assert owned == {"segmentation.json", "cells.json", "excluded_cells.json", "segment_COMPLETE.json",
                     "segment_USAGE.json", "crop_COMPLETE.json", "crop_USAGE.json",
                     *{f"segment/{i}/cell_mask_{i}.tif" for i in ("P1_W0_0", "P1_W0_1", "P1_W1_0")},
                     *{f"{s}/{i}/SEARCH.json" for s in ("segment", "crop") for i in ("P1_W0_0", "P1_W0_1", "P1_W1_0")}}
    assert set(record["files"]) == owned - {"segmentation.json", "cells.json"}
    for rel, entry in record["files"].items():
        assert entry["sha256"] == delivered[rel] == digest(out / rel) and not (out / rel).is_symlink()
        assert entry["method"] == ("hardlink" if linkable and rel.endswith(".tif") else "copy")
    for rel in ("embedding.npy", "profiles.npy", "corrected.npy", "SSL_SELECTION.json", "embed_COMPLETE.json"):
        assert not (out / rel).exists()                    # the rerun stages' outputs are its own
    rows = read_json(out / "segmentation.json")
    assert all(Path(r["mask"]).parent.parent.parent == out and digest(Path(r["mask"])) == r["sha256"] for r in rows)
    cells = read_json(out / "cells.json")
    assert len(cells) == 3 and record["crops"]["crop_path"] == {
        "n": 3, "hardlink": 3 if linkable else 0, "copy": 0 if linkable else 3}
    for cell, old in zip(cells, read_json(source / "cells.json")):
        assert Path(cell["crop_path"]).is_relative_to(out) and digest(Path(cell["crop_path"])) == cell["sha256"]
        assert Path(cell["display_crop_path"]).is_relative_to(out)
        assert {k: v for k, v in cell.items() if not k.endswith("crop_path")} == {
            k: v for k, v in old.items() if not k.endswith("crop_path")}
    assert record["rewired"]["cells.json"] == {"fields": ["crop_path", "display_crop_path"],
                                               "source_sha256": delivered["cells.json"],
                                               "sha256": digest(out / "cells.json"), "from_root": str(source),
                                               "to_root": str(out)}
    assert record["recorded_source_root"] == str(source) and record["source_run"] == str(source)
    lines = (out / "SEARCH_WARNINGS.jsonl").read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["stage"] for line in lines] == ["segment", "crop"]
    assert record["search_warnings"]["dropped"] == 2 and read_json(out / cli.RERUN_RECORD) == record


def test_reuse_follows_a_moved_source_and_refuses_changed_bytes(small_source):
    source, root = small_source
    moved = root / "archive" / "shape"
    moved.parent.mkdir()
    source.rename(moved)                                  # its records still name the old directory
    out = root / "rerun"
    out.mkdir()
    record = posthoc.reuse_upstream(found(moved), out, ("segment", "crop"))
    assert record["recorded_source_root"] == str(source) and record["source_run"] == str(moved)
    assert all(Path(c["crop_path"]).is_relative_to(out) for c in read_json(out / "cells.json"))
    crop = next((moved / "crop").rglob("cell_000001.ome.tif"))      # the second view's crop
    crop.write_bytes(b"changed")                          # a crop that is not the one cells.json hashed
    shutil.rmtree(out)
    out.mkdir()
    with pytest.raises(ValueError, match="does not match its recorded hash"):
        posthoc.reuse_upstream(found(moved), out, ("segment", "crop"))
    assert crop.read_bytes() == b"changed"


def test_a_placed_file_is_never_written_through(tmp_path):
    src, dst = tmp_path / "src.bin", tmp_path / "out" / "dst.bin"
    src.write_bytes(b"source bytes")
    assert posthoc._place(src, dst, None, link=True)["sha256"] == digest(src)
    again = posthoc._place(src, dst, digest(src), link=False)        # already there: not copied onto the link
    assert again["sha256"] == digest(src) and src.read_bytes() == b"source bytes"
    other = tmp_path / "other.bin"
    other.write_bytes(b"other bytes")
    with pytest.raises(ValueError, match="already placed with other bytes"):
        posthoc._place(other, dst, None, link=False)
    assert src.read_bytes() == b"source bytes"


# ---------------------------------------------------------------- the rerun, delivered, scored and compared


class Recorder:
    model_string = "gpt-5.6-luna"

    def generate(self, content, **kwargs):
        return {"content": "OK"}


@pytest.fixture
def stand_ins(monkeypatch, tmp_path):
    """The rerun's stage workers in this process with the real stages on CPU stand-ins: the SSL tool returns each
    cell's design vector, the verifiers' search returns its input, cluster writes its files."""
    import anndata as ad
    from benchmarks.full_HCMA.harness import shape_arm
    from shape.agent.verification import vision_engine
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.engine import usage
    from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

    seen = {"env": {}, "search": {}, "crops": set()}
    vectors = {}
    tmp_path = tmp_path.resolve()
    for name in REASONER_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setattr(cli, "load_prepared", lambda prep, verify_pixels=False: data.load_prepared(prep))
    monkeypatch.setattr(cli, "asset_environment", lambda root: {})
    monkeypatch.setattr(cli, "code_identity", lambda commit, branch: {"commit": commit, "branch": branch})
    monkeypatch.setattr(vision_engine, "build_vision_engine", lambda model: Recorder())
    monkeypatch.setattr(shape_arm, "build_engine", lambda model, engine=None: engine)

    def execute(self, **kwargs):
        meta = kwargs["cell_metadata"]
        seen["crops"].update(kwargs["cell_crops"])
        folder = Path(kwargs["query_cache_dir"])
        folder.mkdir(parents=True, exist_ok=True)
        path = folder / f"embedding_{len(list(folder.glob('*.h5ad')))}.h5ad"
        ad.AnnData(np.stack([vectors[m["cell_id"]] for m in meta]).astype(np.float32),
                   obs=pd.DataFrame({"cell_id": [m["cell_id"] for m in meta]})).write_h5ad(path)
        return {"adata_path": str(path), "resume_best": None}

    def resolve(self, verifier, adapter, runner, initial, baseline, **kw):
        stage = "embed" if "representation_task" in kw["upstream_ctx"] else "batch"
        seen["search"][stage] = {"verifier": type(verifier).__name__,
                                 "expert_config": getattr(verifier, "expert_config", None),
                                 "max_tokens": verifier.llm_engine.max_tokens,
                                 "effort": os.environ.get("SHAPE_REASONING_EFFORT"),
                                 "timeout": os.environ.get("SHAPE_LLM_TIMEOUT")}
        verifier.admission = {"decision": "SKIP"}
        usage.record({"total_tokens": 50, "prompt_tokens": 30, "completion_tokens": 20, "reasoning_tokens": 7})
        return SimpleNamespace(params=initial or None, result=None if stage == "batch" else baseline, details={},
                               status="satisfied", reason="test", fault=None)

    def cluster(prep, run, config, rows):
        (run / "profiles.h5ad").write_bytes(b"h5ad")
        (run / "states.csv").write_text("well,state\n", encoding="ascii")

    def child(command, log, timeout, *, new_session=True, env=None):
        if command[3] == "preflight":
            write_json(Path(command[command.index("--out") + 1]),
                       {"branch": command[command.index("--expected-branch") + 1],
                        "commit": command[command.index("--expected-commit") + 1], "model": "gpt-5.6-luna",
                        "gpu": "test-gpu", "assets": {"sha": "assets"}})
            return
        stage = command[4]
        seen["env"][stage] = {k: env.get(k) for k in REASONER_ENV}
        for name in REASONER_ENV:
            if env.get(name) is None:
                monkeypatch.delenv(name, raising=False)
            else:
                monkeypatch.setenv(name, env[name])
        usage.reset()
        try:
            cli.stage(SimpleNamespace(stage=stage, prep=Path(command[command.index("--prep") + 1]),
                                      run=Path(command[command.index("--run") + 1])))
        finally:
            usage.reset()
            for name in REASONER_ENV:
                monkeypatch.delenv(name, raising=False)

    monkeypatch.setattr(Cell_State_Analyzer_Tool, "execute", execute)
    monkeypatch.setattr(StepVerifier, "resolve_stage", resolve)
    monkeypatch.setitem(pipeline.STAGES, "cluster", cluster)
    monkeypatch.setattr(cli, "child", child)
    prep, base = design(tmp_path)
    rng = np.random.default_rng(3)
    for cell, vector in zip(read_json(base / "cells.json"), np.load(base / "embedding.npy")):
        vectors[cell["cell_id"]] = vector + rng.normal(scale=0.3, size=vector.shape)
    return SimpleNamespace(prep=prep, base=base, root=tmp_path, seen=seen)


def test_a_rerun_reuses_segment_and_crop_reruns_the_rest_delivers_scores_and_compares(stand_ins):
    from benchmarks.full_CPJUMP1.harness.score import score
    prep, root, seen = stand_ins.prep, stand_ins.root, stand_ins.seen
    control = plain_run(root, prep, stand_ins.base, "control")
    score(prep, control)                                   # a complete control run is scored at its end
    codex = plain_run(root, prep, stand_ins.base, "codex")
    source = source_run(root, prep, stand_ins.base)
    before = {run.name: snapshot(run) for run in (control, codex, source)}
    out = root / "rerun"
    cli.main(["rerun", "--source", str(source), "--from-stage", "embed", "--arm", "shape", "--prep", str(prep),
              "--assets-root", str(root), "--out", str(out), "--expected-commit", NEW, "--expected-branch", "main",
              "--control", str(control)])
    assert before["shape"] == snapshot(source)              # the source is never modified

    # the marker: never the full-run one
    assert (out / "RERUN_COMPLETE").read_text(encoding="ascii").strip() == NEW
    assert not (out / "CPJUMP1_COMPLETE").exists() and not (out / "SMOKE_COMPLETE").exists()

    # provenance
    config, meta = read_json(out / "CONFIG.json"), read_json(out / "RUN_META.json")
    rerun = config["rerun"]
    assert {k: rerun[k] for k in ("source_run", "source_commit", "source_config_sha256", "source_delivery_sha256",
                                  "reused_stages", "rerun_stages", "reason", "commit", "branch")} == {
        "source_run": str(source), "source_commit": OLD, "source_config_sha256": digest(source / "CONFIG.json"),
        "source_delivery_sha256": digest(source / "DELIVERY.json"), "reused_stages": ["segment", "crop"],
        "rerun_stages": ["embed", "batch", "cluster"], "reason": cli.RERUN_REASON, "commit": NEW, "branch": "main"}
    assert rerun["budget"]["hours"] == 8.0 == config["budget_hours"] and rerun["record_sha256"] == digest(
        out / "RERUN.json")
    assert rerun["source_reasoning"] is None                 # the source's commit recorded no reasoning table
    assert config["reasoning"] == reasoning_record("shape")
    assert config["reasoning"]["stages"]["embed"] == config["reasoning"]["stages"]["batch"] == HIGH
    assert config["ssl_expert_config"] == [dict(r) for r in SSL_EXPERT_CONFIG]
    assert {k: config[k] for k in ("seed", "train_cells_per_site", "inference_chunk", "images", "cores")} == {
        k: read_json(source / "CONFIG.json")[k] for k in ("seed", "train_cells_per_site", "inference_chunk",
                                                           "images", "cores")}
    assert config["resources"]["source"] == "source_run" and config["ssl_epoch_budget"]["n_train"] > 0
    assert meta["status"] == "complete" and meta["config"] == config and meta["rerun"] == rerun
    assert meta["provenance"] == read_json(out / "PREFLIGHT.json") and meta["provenance"]["commit"] == NEW
    for stage in ("segment", "crop"):                        # the upstream stages' records stay the source's
        assert meta["stages"][stage] == read_json(source / f"{stage}_COMPLETE.json")
        assert meta["stage_code"][stage] == {"run": "source", "commit": OLD}
    assert meta["stage_code"]["embed"] == {"run": "rerun", "commit": NEW}

    # each rerun stage's reasoner: its own environment, the table's cap, the expert start
    assert seen["env"] == {"embed": {"SHAPE_REASONING_EFFORT": "high", "SHAPE_LLM_TIMEOUT": "900"},
                           "batch": {"SHAPE_REASONING_EFFORT": "high", "SHAPE_LLM_TIMEOUT": "900"},
                           "cluster": {"SHAPE_REASONING_EFFORT": None, "SHAPE_LLM_TIMEOUT": None}}
    assert seen["search"]["embed"] == {"verifier": "FixedEpochVerifier",
                                       "expert_config": [dict(r) for r in SSL_EXPERT_CONFIG],
                                       "max_tokens": 16000, "effort": "high", "timeout": "900"}
    assert {k: v for k, v in seen["search"]["batch"].items() if k != "expert_config"} == {
        "verifier": "BatchCorrectionVerifier", "max_tokens": 16000, "effort": "high", "timeout": "900"}
    for stage in ("embed", "batch"):
        assert read_json(out / f"{stage}_USAGE.json")["reasoning"] == HIGH
        assert read_json(out / f"{stage}_USAGE.json")["reasoning_tokens"] == 7

    # the embed stage read the rerun's own copies of the crops
    assert seen["crops"] and all(Path(p).is_relative_to(out) for p in seen["crops"])

    # delivery: self-contained, the reused artifacts bound with the source's hashes
    delivery = read_json(out / "DELIVERY.json")
    source_delivery = read_json(source / "DELIVERY.json")["artifacts"]
    assert delivery["status"] == "complete" and delivery["provenance"]["commit"] == NEW
    for rel in ("segment_COMPLETE.json", "crop_USAGE.json", "excluded_cells.json", "segment/P1_W0_0/SEARCH.json",
                "segment/P1_W0_0/cell_mask_P1_W0_0.tif"):
        assert delivery["artifacts"][rel] == source_delivery[rel]
    assert {"RERUN.json", "embed_COMPLETE.json", "representation_SEARCH.json"} <= set(delivery["artifacts"])
    assert delivery["artifacts"]["embedding.npy"] != source_delivery["embedding.npy"]
    for rel in delivery["artifacts"]:
        assert data.within(out, rel).is_relative_to(out)

    # scoring with the control's verdicts
    result = read_json(out / "RESULT.json")
    assert result["arm"] == "shape" and set(result["full_metrics"]["verdicts"]) == {
        "mAP", "macro_f1", "balanced_accuracy"}

    # the strict compare refuses the rerun; the post-hoc checks accept it
    with pytest.raises(ValueError, match="post-hoc rerun"):
        compare.validate_run(out, "shape", NEW, "main")
    with pytest.raises(ValueError, match="post-hoc rerun"):
        posthoc.verify_run(out, "shape", digest(prep / "PREPARE.json"))

    # the post-hoc comparison
    before.update(rerun=snapshot(out))
    comparison = root / "posthoc"
    cli.main(["compare-rerun", "--prep", str(prep), "--rerun", str(out), "--control", str(control),
              "--arm", f"codex={codex}", "--out", str(comparison)])
    for run in (control, codex, source, out):                # the compared runs are not written
        assert snapshot(run) == before[run.name]
    summary = read_json(comparison / "POSTHOC_COMPARISON.json")
    assert summary["strict_comparison"] is False and summary["kind"] == "posthoc_rerun_comparison"
    assert set(summary["runs"]) == {"control", "shape_source", "shape_rerun", "codex"}
    assert set(summary["verdicts"]) == {"shape_source", "shape_rerun", "codex"}
    assert summary["commits"] == {"control": OLD, "shape_source": OLD, "shape_rerun": NEW, "codex": OLD}
    assert "NOT THE STRICT COMPARISON" in summary["notes"][0]
    assert "differ in code and in reasoning" in summary["notes"][0]
    for label in ("shape_source", "shape_rerun", "codex"):
        for metric in ("mAP", "macro_f1", "balanced_accuracy"):
            vs = summary["runs"][label]["metrics"][metric]["vs_control"]
            assert set(vs) == {"delta", "se", "ci95", "guard_delta", "verdict"}
            # the same control draws as the rerun's own RESULT.json
            if label == "shape_rerun":
                assert vs["delta"] == pytest.approx(result["full_metrics"]["metrics"][metric]["vs_control"]["delta"])
    assert all(summary["runs"]["codex"]["metrics"][m]["vs_control"]["delta"] == 0 for m in summary["primary"])
    assert set(summary["rerun_vs_source"]) == {"mAP", "macro_f1", "balanced_accuracy"}
    assert summary["rerun_vs_source"]["mAP"]["delta"] == pytest.approx(
        summary["runs"]["shape_rerun"]["metrics"]["mAP"]["vs_control"]["delta"]
        - summary["runs"]["shape_source"]["metrics"]["mAP"]["vs_control"]["delta"])
    assert set(summary["runs"]["shape_rerun"]["retrieval"]) == {
        f"{v}_{g}_cross_plate" for v in ("uncorrected", "delivered") for g in ("cross_guide", "same_guide")}
    stages = summary["runs"]["shape_rerun"]["stages"]
    assert stages["segment"] == {"ran": "source run", "commit": OLD, "reasoning": None, "seconds": 3600,
                                 "total_tokens": 100, "reasoning_tokens": None, "calls": 1}
    assert stages["embed"]["ran"] == "this run" and stages["embed"]["commit"] == NEW
    assert stages["embed"]["reasoning"] == HIGH and stages["batch"]["reasoning"] == HIGH
    assert stages["embed"]["total_tokens"] == 50 and stages["embed"]["reasoning_tokens"] == 7
    assert summary["runs"]["shape_rerun"]["reasoning_tokens"] == 14
    assert summary["runs"]["shape_source"]["reasoning_tokens"] is None
    assert summary["runs"]["shape_rerun"]["ssl_expert_config"] == [dict(r) for r in SSL_EXPERT_CONFIG]
    assert summary["config_differences"]["budget_hours"]["shape_rerun"] == 8.0
    assert "images" not in summary["config_differences"]
    text = (comparison / "POSTHOC_COMPARISON.md").read_text(encoding="utf-8")
    assert "not the strict comparison" in text and "| shape_rerun | embed | this run |" in text
    assert (comparison / "scores" / "control" / "evaluation" / "full_metrics_draws.npz").is_file()
    with pytest.raises(FileExistsError):
        posthoc.compare_rerun(prep, out, control, {}, comparison)


def test_compare_rerun_refuses_a_source_that_is_not_the_reruns(tmp_path):
    """The source must be the one whose DELIVERY.json and CONFIG.json hashes the rerun recorded."""
    tmp_path = tmp_path.resolve()
    prep = tmp_path / "prep"
    prep.mkdir()
    write_json(prep / "PREPARE.json", {"status": "complete"})
    other = tmp_path / "other"
    other.mkdir()
    write_json(other / "representation_SEARCH.json", CLEAN)
    finish(other, prep, "shape", OLD, config_of(prep, "shape", [f"img_{i:04d}" for i in range(1000)]))
    run = tmp_path / "rerun"
    run.mkdir()
    write_json(run / "representation_SEARCH.json", CLEAN)
    write_json(run / cli.RERUN_RECORD, {"files": {}, "rewired": {}})
    info = {"source_run": str(other), "source_commit": OLD, "source_config_sha256": "0" * 64,
            "source_delivery_sha256": digest(other / "DELIVERY.json"), "reused_stages": ["segment", "crop"],
            "rerun_stages": ["embed", "batch", "cluster"], "reason": "test",
            "record_sha256": digest(run / cli.RERUN_RECORD)}
    finish(run, prep, "shape", NEW, config_of(prep, "shape", [f"img_{i:04d}" for i in range(1000)], rerun=info,
                                              reasoning=reasoning_record("shape")), meta={"rerun": info})
    (run / "CPJUMP1_COMPLETE").rename(run / cli.RERUN_MARKER)
    with pytest.raises(ValueError, match="not this rerun's source run"):
        posthoc.verify_rerun(run, digest(prep / "PREPARE.json"))
    with pytest.raises(ValueError, match="is not a post-hoc rerun"):
        posthoc.verify_run(other, "shape", digest(prep / "PREPARE.json"), marker=cli.RERUN_MARKER)

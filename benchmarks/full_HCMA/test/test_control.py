"""harness/control.py: the default chain driver.

The first test runs the real workspace tools as subprocesses (the SHAPE tool classes replaced by
test/tool_stubs.py, FULL_HCMA_TOOL_STUBS; the batch corrector's detector and the analyzer's metadata
handling are the real code) in a workspace built by the real setup_run, at both metadata levels; the
others inject a fake tool runner (the synthetic calls of _synthetic_chain) to test the command
sequence and the failure handling quickly. Two tests call the real SHAPE classes directly: the batch
corrector on one all-empty batch column, the analyzer's reading of blank technical metadata."""
from __future__ import annotations

import json
import stat
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_HCMA.harness import control, hardware, ledger, score, subset, workspace
from benchmarks.full_HCMA.test._synthetic_chain import (Chain, build_subset, digest, informative, workspace_data,
                                                        write_class_map)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
IMAGES_COLUMNS = ["image_id", "file", "parent_id", "source", "image_modality", "magnification", "height", "width",
                  "channels"]
# two classes, three micrographs each, one image per micrograph, folds 0..2; `pairs`: two per class
STATES = ["epithelial", "fibroblast"] * 3
PAIRS = ["img_0001", "img_0002", "img_0003", "img_0004"]
CLASSES = {"epithelial": "sheet", "fibroblast": "spindle"}            # a synthetic morph_classes.csv


def _image(path: Path, n: int, seed: int) -> None:
    from PIL import Image

    rng = np.random.default_rng(seed)
    yy, xx = np.mgrid[:96, :96]
    pixels = np.full((96, 96), 20, dtype=np.uint8)
    for r, c in rng.integers(12, 84, size=(n, 2)):
        pixels[(yy - r) ** 2 + (xx - c) ** 2 <= 36] = 200
    Image.fromarray(pixels).save(path)


@pytest.fixture()
def root(tmp_path, monkeypatch):
    monkeypatch.setattr(score, "CLASS_MAP", write_class_map(tmp_path / "morph_classes.csv", CLASSES))
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "testgpu")
    monkeypatch.setattr(hardware, "gpu_slots", lambda: 1)
    monkeypatch.setattr(hardware, "node_facts", lambda: {"gpu": "none", "gpu_mem_gb": None, "cpus": 2, "mem_gb": 8.0})
    root = tmp_path / "full_HCMA"
    (root / "prep" / "images").mkdir(parents=True)
    rows, labels = [], []
    for i, state in enumerate(STATES):
        image_id, parent = f"img_{i + 1:04d}", f"par_{i + 1:04d}"
        _image(root / "prep" / "images" / f"{image_id}.png", 4 + i % 3, i)
        rows.append({"image_id": image_id, "file": f"images/{image_id}.png", "parent_id": parent,
                     "source": f"vendor_{i % 2}", "image_modality": "brightfield", "magnification": "",
                     "height": 96, "width": 96, "channels": 1})
        labels.append({"image_id": image_id, "orig_image_id": f"o{i}", "image_filename": f"o{i}.png",
                       "parent_id": parent, "orig_parent_image_id": f"op{i}", "source": f"vendor_{i % 2}",
                       "cell_type": f"{state} cell", "cell_state": state, "organ_category": f"organ{i % 2}",
                       "fold": i // 2})
    pd.DataFrame(rows, columns=IMAGES_COLUMNS).to_csv(root / "prep" / "images.csv", index=False)
    (root / "prep" / "PREPARE.json").write_text(json.dumps({"n_images": len(rows)}))
    private = root / "private"
    private.mkdir()
    prep = root / "prep"
    (private / "PREPARE_PRIVATE.json").write_text(json.dumps({"hcma_root": "/data/HCMA", "prep_sha256": {
        f.relative_to(prep).as_posix(): digest(f) for f in sorted(prep.rglob("*")) if f.is_file()}}))
    table = pd.DataFrame(labels)
    table.drop(columns="fold").to_csv(private / "labels.csv", index=False)
    table[["parent_id", "fold"]].to_csv(private / "folds.csv", index=False)
    table[["image_id", "cell_state", "parent_id", "fold"]].to_csv(private / "scorable.csv", index=False)
    build_subset(root, "pairs", PAIRS)
    return root


@pytest.fixture()
def fast_scib(monkeypatch):
    monkeypatch.setattr(score, "recluster", lambda x, ids, resolutions=score.SCIB_RESOLUTIONS, n_neighbors=15, timings=None: {
        r: np.argmax(np.asarray(x), axis=1).astype(str) for r in resolutions})


def _allow_stubs(monkeypatch):
    """setup_run, then let the tools load test/tool_stubs.py (a run's setup never allows it) and the
    chain accept calls made with the stubs."""
    monkeypatch.setattr(ledger, "ALLOW_STUBS", True)
    real = workspace.setup_run

    def setup(run, root, repo, **kwargs):
        ws = real(run, root, repo, **kwargs)
        tools = ws / "tools"
        tools.chmod(stat.S_IRWXU)
        (tools / "_config.json").chmod(stat.S_IRUSR | stat.S_IWUSR)
        config = json.loads((tools / "_config.json").read_text(encoding="utf-8"))
        config["allow_tool_stubs"] = True
        (tools / "_config.json").write_text(json.dumps(config, indent=1), encoding="utf-8")
        (Path(run) / "TOOLS_SHA256.json").write_text(json.dumps(workspace.tool_hashes(tools)), encoding="utf-8")
        return ws

    monkeypatch.setattr(workspace, "setup_run", setup)
    monkeypatch.setenv("FULL_HCMA_TOOL_STUBS", str(HERE / "tool_stubs.py"))


def _starts(run: Path) -> list[dict]:
    lines = (run / "logs" / "tool_calls.jsonl").read_text(encoding="utf-8").splitlines()
    return [e for e in map(json.loads, lines) if e["event"] == "start"]


@pytest.mark.parametrize("metadata", ["full", "none"])
def test_control_runs_the_real_tools_in_subprocesses(root, monkeypatch, metadata):
    _allow_stubs(monkeypatch)
    run = control.run_control(root, REPO, seeds=[0], metadata=metadata)       # the default prep: pairs
    assert run == root / "control" / "pairs" / metadata / "testgpu"
    starts = _starts(run)
    assert [(e["call"], e["tool"], e["name"]) for e in starts] == [
        ("S001", "segment", "defaults"), ("C001", "crop", "defaults"),
        ("E001", "embed", "defaults_s0"), ("B001", "batch_correct", "defaults_s0"), ("K001", "cluster", "defaults_s0")]
    argv = {e["call"]: e["argv"] for e in starts}
    assert argv["C001"][argv["C001"].index("--masks") + 1] == "S001"
    assert argv["E001"][argv["E001"].index("--set") + 1:][:2] == ["seed=0", "adaptation_depth=zero_shot"]
    assert argv["B001"][-2:] == ["--batch-key", "source"] and "E001" in argv["B001"]
    assert argv["K001"][-2:] == ["--embedding", "B001"]
    # every tool at its defaults, except embed adaptation_depth=zero_shot
    assert {e["tool"]: e["set_levers"] for e in starts} == {"segment": [], "crop": [], "embed": ["adaptation_depth"],
                                                             "batch_correct": [], "cluster": []}
    choice = json.loads((run / "workspace" / "submission" / "choice.json").read_text(encoding="utf-8"))
    assert choice["cluster_call"] == "K001" and choice["resolution"] == 0.5
    result = json.loads((run / "RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "OK" and result["seed"] == 0 and result["chain"]["ok"]
    assert result["coverage"]["n_images_with_cells"] == len(PAIRS) == result["coverage"]["n_images"]
    assert result["inputs"]["ok"] and result["prep_name"] == "pairs" and result["metadata"] == metadata
    assert result["expected_set"] == "prep" and result["expected_image_ids"] is None
    assert result["control_chain"] == {"embed": "zero_shot", "rule": "the five tools at their defaults, except embed "
                                                                     "adaptation_depth=zero_shot"}
    assert result["class_map"]["ok"] and result["primary"]["n_classes"] == 2 and result["fine"]["n_types"] == 2
    for name in ("umap_scored.png", "per_class.csv", "per_type.csv"):
        assert (run / name).exists()
    assert result["tool_intact"]["problem"] == "tools/_config.json allows the test stubs"
    logged = [json.loads(line) for line in (run / "logs" / "control_commands.jsonl").read_text().splitlines()]
    assert [(e["tool"], e["argv"][:2], e["returncode"]) for e in logged][0] == ("segment", ["--name", "defaults"], 0)
    assert len(logged) == 5
    assert 0.0 <= result["primary"]["value"] <= 1.0
    for name in ("RESULT_s0.json", "per_image.csv", "per_image_s0.csv", "per_class_s0.csv", "per_type_s0.csv",
                 "RESULT.md"):
        assert (run / name).exists()
    ws = run / "workspace"
    seen = pd.read_csv(ws / "data" / "images.csv", dtype=str, keep_default_na=False)
    assert seen.equals(workspace.blank_metadata(pd.read_csv(root / "prep_pairs" / "images.csv", dtype=str,
                                                            keep_default_na=False), metadata))
    if metadata == "none":
        # one empty batch: no batch labels, no calibration view; the input comes back unchanged and the chain goes on
        b001 = next((ws / "calls").glob("B001_*"))
        tool_result = json.loads((b001 / "tool_result.json").read_text(encoding="utf-8"))
        assert tool_result["status"] == "UNCORRECTED_INPUT" and tool_result["tier"] == "none"
        assert tool_result["corrected"] is False and tool_result["reason"].startswith("no calibration reference")
        assert np.array_equal(np.load(b001 / "embedding.npy"), np.load(next((ws / "calls").glob("E001_*")) / "embedding.npy"))
        # embed --meta with blank values: the analyzer's real metadata code treats them as absent
        code = control.subprocess_runner(REPO)(ws, "embed", ["--name", "meta_blank", "--crops", "C001",
                                                             "--meta", "plate=source"], run / "logs" / "meta.log")
        assert code == 0, (run / "logs" / "meta.log").read_text(encoding="utf-8")
        import anndata as ad

        obs = ad.read_h5ad(next((ws / "calls").glob("E002_meta_blank")) / "cell_state_analyzed.h5ad").obs
        assert (obs["batch"] == obs["image_name"]).all()          # no plate: each image is its own batch


def _fake_setup(monkeypatch, *, gpu_exclusive: bool = False, gpu_mem_gb: float | None = None):
    monkeypatch.setattr(hardware, "node_facts", lambda: {"gpu": "fake", "gpu_mem_gb": gpu_mem_gb, "cpus": 2,
                                                         "mem_gb": 8.0, "gpu_exclusive": gpu_exclusive})

    def setup(run, root, repo, *, budget_hours, gpu_slots, template=None, prep=None, metadata="full"):
        assert budget_hours is None
        ws = Path(run) / "workspace"
        for sub in ("calls", "submission", "data", "tools"):
            (ws / sub).mkdir(parents=True)
        (Path(run) / "logs").mkdir()
        workspace_data(run, root, Path(prep).name, metadata)
        (ws / "tools" / "_config.json").write_text(json.dumps({"gpu_exclusive": gpu_exclusive, "prep": str(prep),
                                                               "prep_name": subset.prep_name(prep),
                                                               "metadata": metadata}))
        (Path(run) / "TOOLS_SHA256.json").write_text(json.dumps({"_config.json": digest(ws / "tools" / "_config.json")}))
        return ws

    monkeypatch.setattr(workspace, "setup_run", setup)


def _fake_runner(root: Path, fail: str | None = None):
    """A tool runner that records the commands and writes synthetic calls instead of running tools."""
    labels = pd.read_csv(root / "private" / "labels.csv", dtype=str)
    seen, chains = [], {}

    def run(ws: Path, tool: str, args: list, log: Path) -> int:
        seen.append((tool, list(args)))
        log.write_text("fake tool log\nboom\n" if tool == fail else "fake tool log\n", encoding="utf-8")
        if tool == fail:
            return 1
        chain = chains.setdefault(ws, Chain(ws.parent, traced=False))       # provenance: control_commands.jsonl
        name = args[args.index("--name") + 1]
        value = lambda flag: args[args.index(flag) + 1]                 # noqa: E731
        if tool == "segment":
            images = value("--images").split(",") if "--images" in args else \
                list(pd.read_csv(ws / "data" / "images.csv", dtype=str)["image_id"])
            chain.segment(images, name=name)
        elif tool == "crop":
            images = pd.read_csv(next((ws / "calls").glob(f"{value('--masks')}_*")) / "per_image.csv")["image_id"]
            chain.crop(value("--masks"), {i: 3 for i in images}, name=name)
        elif tool == "embed":
            chain.embed(value("--crops"), lambda cells: informative(labels, cells), name=name)
        elif tool == "batch_correct":
            chain.batch_correct(value("--embedding"), name=name)
        else:
            chain.cluster(value("--embedding"), name=name)
        return 0

    return run, seen


def test_smoke_uses_the_first_images_and_writes_the_choice(root, monkeypatch, fast_scib):
    _fake_setup(monkeypatch)
    runner, seen = _fake_runner(root)
    run = control.run_control(root, REPO, seeds=[0, 1], n_images=3, runner=runner, prep_name="full", metadata="none")
    assert run.parent == root / "smoke" / "full" / "none"
    assert seen[0] == ("segment", ["--name", "defaults", "--images", "img_0001,img_0002,img_0003", "--workers", "1"])
    assert seen[1] == ("crop", ["--name", "defaults", "--masks", "S001", "--workers", "2"])
    assert [t for t, _ in seen] == ["segment", "crop", *["embed", "batch_correct", "cluster"] * 2]
    assert seen[2][1] == ["--name", "defaults_s0", "--crops", "C001", "--set", "seed=0", "adaptation_depth=zero_shot"]
    assert seen[3][1] == ["--name", "defaults_s0", "--embedding", "E001", "--batch-key", "source"]
    assert seen[4][1] == ["--name", "defaults_s0", "--embedding", "B001"]
    assert seen[5][1] == ["--name", "defaults_s1", "--crops", "C001", "--set", "seed=1", "adaptation_depth=zero_shot"]
    assert seen[7][1] == ["--name", "defaults_s1", "--embedding", "B002"]
    choice = json.loads((run / "workspace" / "submission" / "choice.json").read_text(encoding="utf-8"))
    assert choice["cluster_call"] == "K001" and choice["resolution"] == 0.5
    result = json.loads((run / "RESULT.json").read_text(encoding="utf-8"))
    assert result["status"] == "OK" and result["coverage"]["n_images_with_cells"] == 3
    assert result["prep_name"] == "full"
    # a smoke is scored over its own images only: no missing-image misses for the images it did not run
    assert result["expected_set"] == "smoke" and result["expected_image_ids"] == ["img_0001", "img_0002", "img_0003"]
    assert result["coverage"]["n_images"] == 3 == result["n_expected_images"]
    assert result["primary"]["n_images_missing"] == 0 and result["primary"]["n_classes"] == 1
    # img_0002 is the smoke's only fibroblast micrograph: unscorable at both levels within the smoke's images
    assert result["unscorable_classes"]["morph_class"] == [{"morph_class": "spindle", "n_images": 1, "n_micrographs": 1}]
    assert "smoke: scored over its 3 images only" in (run / "RESULT.md").read_text(encoding="utf-8")
    assert result["metadata"] == "none" and result["settings_from"]["metadata"] == "the caller"
    assert result["n_images_requested"] == 3 and set(result["seed_spread"]["values"]) == {"0", "1"}
    assert result["smoke"]["types"] == ["epithelial", "fibroblast"]          # the cell_states of the first 3 ids
    assert result["smoke"]["morph_classes"] == {"epithelial": "sheet", "fibroblast": "spindle"}
    assert result["control_chain"]["embed"] == "zero_shot"                   # the smoke records the chain too
    s1 = json.loads((run / "RESULT_s1.json").read_text(encoding="utf-8"))
    assert s1["seed"] == 1 and s1["choice"]["cluster_call"] == "K002"
    assert result["seed_spread"]["values"]["1"] == s1["primary"]["value"]


@pytest.mark.parametrize("exclusive, mem, workers", [(False, 80.0, "4"), (True, 80.0, "1"), (False, 16.0, "2")])
def test_segment_workers_follow_the_gpu(root, monkeypatch, fast_scib, exclusive, mem, workers):
    _fake_setup(monkeypatch, gpu_exclusive=exclusive, gpu_mem_gb=mem)
    runner, seen = _fake_runner(root)
    run = control.run_control(root, REPO, seeds=[0], runner=runner)
    assert seen[0][1][-2:] == ["--workers", workers]
    assert json.loads((run / "RESULT.json").read_text(encoding="utf-8"))["chain"]["ok"]


def test_a_failing_stage_stops_the_control(root, monkeypatch, fast_scib):
    _fake_setup(monkeypatch)
    runner, seen = _fake_runner(root, fail="embed")
    with pytest.raises(SystemExit) as stop:
        control.run_control(root, REPO, seeds=[0], runner=runner)
    message = str(stop.value)
    assert message.startswith("CONTROL_FAILED at embed --name defaults_s0: exit code 1")
    assert "boom" in message and [t for t, _ in seen] == ["segment", "crop", "embed"]
    assert not (root / "control" / "pairs" / "full" / "testgpu" / "RESULT.json").exists()
    with pytest.raises(SystemExit, match="exists; move it aside"):
        control.run_control(root, REPO, seeds=[0], runner=runner)


def test_the_control_stops_before_any_tool_without_a_class_map(root, monkeypatch, tmp_path):
    _fake_setup(monkeypatch)
    runner, seen = _fake_runner(root)
    monkeypatch.setattr(score, "CLASS_MAP", tmp_path / "missing.csv")
    with pytest.raises(SystemExit, match="CONTROL_FAILED: NO_CLASS_MAP: no class map at"):
        control.run_control(root, REPO, seeds=[0], runner=runner)
    monkeypatch.setattr(score, "CLASS_MAP", write_class_map(tmp_path / "partial.csv", {"epithelial": "sheet"}))
    with pytest.raises(SystemExit, match="no morph_class row"):
        control.run_control(root, REPO, seeds=[0], runner=runner)
    assert not seen and not (root / "control").exists()


def test_the_real_batch_corrector_skips_a_single_empty_batch(tmp_path):
    from shape.tools.batch_corrector.tool import SKIPPED_INSUFFICIENT_BATCHES, Batch_Corrector_Tool

    x = np.random.default_rng(0).normal(size=(40, 8))
    result = Batch_Corrector_Tool().execute(cls_output=x, metadata=pd.DataFrame({"source": [""] * 40}),
                                            batch_key="source", row_ids=[f"c{i}" for i in range(40)],
                                            harmony_params=None, output_dir=str(tmp_path / "tool"))
    assert result["status"] == SKIPPED_INSUFFICIENT_BATCHES and result["harmony"] is None
    assert result["batch_effect"]["n_batches"] == 1 and not result["batch_effect"]["detected"]
    assert np.array_equal(result["corrected_cls_output"], x)


def test_the_analyzer_reads_blank_technical_metadata_as_absent():
    from shape.tools.cell_state_analyzer.tool import resolve_plate_nuisance, resolve_technical_metadata

    metadata = [{"plate": "", "site": "  ", "cell_id": f"c{i}"} for i in range(6)]
    names = ["img_1", "img_1", "img_2", "img_2", "img_3", "img_3"]
    groups, batches, report = resolve_technical_metadata(metadata, names)
    assert groups == names and batches == names
    assert report["scale_and_balance"]["metadata_availability"]["plate_id"] == "unavailable"
    nuisance = resolve_plate_nuisance(metadata, 1.0, 1.0)
    assert not nuisance["eligible"] and nuisance["plate_mean_weight"] == 0.0 == nuisance["plate_adversarial_weight"]


def test_a_subset_smoke_takes_whole_type_pairs(root, monkeypatch, fast_scib):
    _fake_setup(monkeypatch)
    runner, seen = _fake_runner(root)
    run = control.run_control(root, REPO, seeds=[0], n_images=2, runner=runner)          # the default prep: pairs
    assert run.parent == root / "smoke" / "pairs" / "full"
    assert seen[0][1][:4] == ["--name", "defaults", "--images", "img_0001,img_0003"]       # both epithelial images
    result = json.loads((run / "RESULT.json").read_text(encoding="utf-8"))
    assert result["smoke"]["types"] == ["epithelial"] and result["smoke"]["images"] == ["img_0001", "img_0003"]
    assert result["smoke"]["morph_classes"] == {"epithelial": "sheet"}
    assert result["expected_set"] == "smoke" and result["coverage"]["n_images"] == 2
    assert result["primary"]["n_classes"] == 1 and result["primary"]["n_images_missing"] == 0   # spindle: not in the smoke
    assert result["status"] == "OK" and result["coverage"]["n_images_with_cells"] == 2


def test_smoke_selection_round_robins_over_the_classes(tmp_path):
    rows = [("i1", "a1"), ("i2", "a1"), ("i3", "a2"), ("i4", "a2"), ("i5", "b1"), ("i6", "b1"), ("i7", "c1"), ("i8", "c1")]
    (tmp_path / "private").mkdir()
    (tmp_path / "private" / "SUBSET_x.json").write_text(json.dumps({"selection": [
        {"image_id": i, "cell_state": t, "parent_id": f"p{i}", "source": "s"} for i, t in rows]}))
    classes = {"a1": "A", "a2": "A", "b1": "B", "c1": score.UNSCORED}
    pick = lambda n: control.smoke_selection(tmp_path, "x", n, classes)                   # noqa: E731
    assert pick(4)["types"] == ["a1", "b1"] and pick(4)["images"] == ["i1", "i2", "i5", "i6"]
    assert pick(6)["types"] == ["a1", "b1", "c1"] and pick(8)["types"] == ["a1", "b1", "c1", "a2"]
    assert pick(1)["types"] == ["a1"]                     # never empty: the first pair even when it exceeds N

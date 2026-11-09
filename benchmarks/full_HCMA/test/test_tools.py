"""The workspace tools (workspace_template/tools/*.py) end to end on CPU, with the SHAPE tool classes
replaced by test/tool_stubs.py (FULL_HCMA_TOOL_STUBS; the cropper and the analyzer's clustering are the
real code). One workspace with four tiny synthetic images (the fourth without cells) serves the module:
the `pipe` fixture runs a chain of calls once, the tests read its ledger and files and make more calls.

    PYTHONPATH=<repo> python -m pytest benchmarks/full_HCMA/test/test_tools.py -q
"""
from __future__ import annotations

import csv
import importlib
import importlib.util
import json
import os
import shutil
import sys
import time
import types
from pathlib import Path

import numpy as np
import pytest

HERE = Path(__file__).resolve().parent
TEMPLATE_TOOLS = HERE.parent / "workspace_template" / "tools"
REPO = HERE.parents[2]
CLIS = ("segment", "crop", "embed", "batch_correct", "cluster")
CENTRES = {"img_0001": [(15, 15), (15, 50), (50, 20), (50, 60), (78, 40), (30, 80)],
           "img_0002": [(20, 20), (20, 70), (70, 25), (72, 72), (45, 45)],
           "img_0003": [(16, 30), (40, 70), (75, 18), (60, 50), (25, 55), (82, 80)],
           "img_0004": []}
SOURCES = {"img_0001": "vendor_a", "img_0002": "vendor_a", "img_0003": "vendor_b", "img_0004": "vendor_b"}


def _image(path: Path, centres, radius: int = 6) -> None:
    from PIL import Image

    yy, xx = np.mgrid[:96, :96]
    pixels = np.full((96, 96), 20, dtype=np.uint8)
    for r, c in centres:
        pixels[(yy - r) ** 2 + (xx - c) ** 2 <= radius ** 2] = 200
    Image.fromarray(pixels).save(path)


class Workspace:
    def __init__(self, run: Path, modules: dict):
        self.run, self.workspace, self.modules = run, run / "workspace", modules
        self.tools = self.workspace / "tools"

    def call(self, tool: str, *argv) -> int:
        """One tool call in this process (os.environ restored after it); returns the exit code."""
        env = dict(os.environ)
        try:
            return self.modules[tool].main([str(a) for a in argv])
        except SystemExit as error:          # argparse
            return error.code
        finally:
            os.environ.clear()
            os.environ.update(env)

    def ledger(self) -> list[dict]:
        path = self.run / "logs" / "tool_calls.jsonl"
        return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()] \
            if path.exists() else []

    def last(self, tool: str) -> dict:
        """The newest call of `tool`: {"id", "start", "end", "dir"}."""
        starts = [e for e in self.ledger() if e["event"] == "start" and e["tool"] == tool]
        start = starts[-1]
        end = next((e for e in self.ledger() if e["event"] == "end" and e["call"] == start["call"]), None)
        return {"id": start["call"], "start": start, "end": end, "dir": self.workspace / start["call_dir"]}

    def record(self, call_id: str) -> dict:
        start = next(e for e in self.ledger() if e["event"] == "start" and e["call"] == call_id)
        end = next((e for e in self.ledger() if e["event"] == "end" and e["call"] == call_id), None)
        return {"id": call_id, "start": start, "end": end, "dir": self.workspace / start["call_dir"]}

    def write(self, name: str, content) -> Path:
        path = self.run / "inputs" / name
        path.parent.mkdir(exist_ok=True)
        path.write_text(content if isinstance(content, str) else json.dumps(content), encoding="utf-8")
        return path

    def snapshot(self) -> tuple[int, list[str]]:
        return len(self.ledger()), sorted(p.name for p in (self.workspace / "calls").iterdir())


def _rows(path: Path) -> list[dict]:
    with open(path, newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _load(path: Path, name: str):
    """A module from a file, without writing __pycache__ next to it (the template stays clean)."""
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    dont_write, sys.dont_write_bytecode = sys.dont_write_bytecode, True
    try:
        spec.loader.exec_module(module)
    finally:
        sys.dont_write_bytecode = dont_write
    return module


@pytest.fixture(scope="module")
def ws(tmp_path_factory):
    run = tmp_path_factory.mktemp("run")
    workspace = run / "workspace"
    for sub in ("data/images", "calls", "submission", "logs"):
        (workspace / sub).mkdir(parents=True)
    (run / "logs").mkdir()
    shutil.copytree(TEMPLATE_TOOLS, workspace / "tools", ignore=shutil.ignore_patterns("__pycache__", "*.pyc"))
    with open(workspace / "data" / "images.csv", "w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle, lineterminator="\n")
        writer.writerow(["image_id", "file", "parent_id", "source", "image_modality", "magnification", "height",
                         "width", "channels"])
        for n, (image_id, centres) in enumerate(CENTRES.items()):
            _image(workspace / "data" / "images" / f"{image_id}.png", centres)
            writer.writerow([image_id, f"images/{image_id}.png", f"par_{n % 2 + 1:04d}", SOURCES[image_id],
                             "brightfield", "", 96, 96, 1])
    config = {"run": str(run), "ledger": str(run / "logs" / "tool_calls.jsonl"), "images_csv": "data/images.csv",
              "images_dir": "data/images", "gpu_slots": 1, "deadline_file": ".deadline", "min_start_seconds": 600,
              "repo": str(REPO), "env": {"SHAPE_FAST_FIGURES": "1", "MPLBACKEND": "Agg"}, "allow_tool_stubs": True}
    (workspace / "tools" / "_config.json").write_text(json.dumps(config), encoding="utf-8")

    names = ("_common", *CLIS)
    saved = {name: sys.modules.pop(name) for name in names if name in sys.modules}
    sys.path.insert(0, str(workspace / "tools"))
    old_stub = os.environ.get("FULL_HCMA_TOOL_STUBS")
    os.environ["FULL_HCMA_TOOL_STUBS"] = str(HERE / "tool_stubs.py")
    modules = {}
    for name in names:          # _common drops tools/ from sys.path after it is imported
        sys.path.insert(0, str(workspace / "tools"))
        modules[name] = importlib.import_module(name)
    modules["time_left"] = _load(workspace / "tools" / "time_left.py", "full_hcma_test_time_left")
    modules["lever_coverage"] = _load(workspace / "tools" / "lever_coverage.py", "full_hcma_test_lever_coverage")
    try:
        yield Workspace(run, modules)
    finally:
        for name in names:
            sys.modules.pop(name, None)
        sys.modules.update(saved)
        sys.path.remove(str(workspace / "tools"))
        if old_stub is None:
            os.environ.pop("FULL_HCMA_TOOL_STUBS", None)
        else:
            os.environ["FULL_HCMA_TOOL_STUBS"] = old_stub


@pytest.fixture(scope="module")
def pipe(ws):
    """A chain with every input kind: two segment calls, two crop calls (one combining both segment calls
    with cell overrides, in two worker processes), two embed calls (one combining both crop calls, with
    an excluded cell), two batch corrections, two clusterings."""
    ids = {}

    def step(key, tool, *argv):
        assert ws.call(tool, "--name", key, *argv) == 0, f"{tool} {key} failed"
        ids[key] = ws.last(tool)["id"]

    step("seg_all", "segment")
    step("seg_sub", "segment", "--images", "img_0002", "--per-image",
         ws.write("per_image.json", {"img_0002": {"diameter": 50, "min_size": 5}, "img_0001": {"diameter": 60}}))
    step("crop_base", "crop", "--masks", ids["seg_all"])
    masks_map = ws.write("masks_map.json", {"img_0001": ids["seg_all"], "img_0002": ids["seg_sub"],
                                            "img_0003": ids["seg_all"]})
    overrides = ws.write("cell_overrides.json", {"img_0001": {
        "default_img_0001_cell_0000": {"margin_mode": "mask_bbox", "auto_tighten": True},
        "default_img_0001_cell_0001": {"reset": True}}})
    step("crop_mixed", "crop", "--masks-map", masks_map, "--cell-overrides", overrides, "--set", "margin_frac=0.3",
         "--workers", "2")
    step("embed_base", "embed", "--crops", ids["crop_base"])
    excluded = next(r["cell_id"] for r in _rows(ws.record(ids["crop_mixed"])["dir"] / "crops.csv")
                    if r["image_id"] == "img_0003")
    ids["excluded"] = excluded
    crops_map = ws.write("crops_map.json", {"img_0001": ids["crop_mixed"], "img_0002": ids["crop_base"],
                                            "img_0003": ids["crop_mixed"]})
    step("embed_mixed", "embed", "--crops-map", crops_map, "--exclude-cells", ws.write("exclude.txt", excluded + "\n"),
         "--meta", "plate=source", "--set", "seed=1", "augment_color=false")
    step("batch_source", "batch_correct", "--embedding", ids["embed_base"], "--batch-key", "source")
    cells = _rows(ws.record(ids["embed_mixed"])["dir"] / "cells.csv")
    batch_file = ws.write("batches.csv", "cell_id,batch\n" + "".join(f"{c['cell_id']},{c['image_id']}\n" for c in cells))
    step("batch_file", "batch_correct", "--embedding", ids["embed_mixed"], "--batch-file", batch_file,
         "--set", "theta=3")
    step("states_base", "cluster", "--embedding", ids["batch_source"])
    step("states_multi", "cluster", "--embedding", ids["embed_mixed"], "--set", "resolution=[0.5,1.0]", "n_neighbors=5")
    return ids


# ---------------------------------------------------------------- the chain

def test_ids_ledger_and_statuses(ws, pipe):
    assert [pipe[k] for k in ("seg_all", "seg_sub", "crop_base", "crop_mixed", "embed_base", "embed_mixed",
                              "batch_source", "batch_file", "states_base", "states_multi")] == \
        ["S001", "S002", "C001", "C002", "E001", "E002", "B001", "B002", "K001", "K002"]
    for key, status in (("seg_all", "partial"), ("seg_sub", "done"), ("crop_base", "done"), ("crop_mixed", "done"),
                        ("embed_base", "done"), ("embed_mixed", "done"), ("batch_source", "done"),
                        ("batch_file", "done"), ("states_base", "done"), ("states_multi", "done")):
        rec = ws.record(pipe[key])
        assert rec["end"]["status"] == status, (key, rec["end"])
        result = json.loads((rec["dir"] / "result.json").read_text(encoding="utf-8"))
        assert result["status"] == status and result["outputs"] == rec["end"]["outputs"]
        meta = json.loads((rec["dir"] / "meta.json").read_text(encoding="utf-8"))
        assert meta["id"] == rec["id"] and meta["argv"][:2] == ["--name", key]
        assert rec["start"]["argv"] == meta["argv"] and "set_levers" in rec["start"]
        assert rec["dir"].name == f"{rec['id']}_{key}" and not (rec["dir"] / "cache").exists()
        common = ws.modules["_common"]
        assert rec["start"]["tool_sha256"] == common.sha256_file(ws.tools / f"{rec['start']['tool']}.py")
        assert rec["start"]["common_sha256"] == common.sha256_file(ws.tools / "_common.py")
        assert rec["start"]["config_sha256"] == common.sha256_file(ws.tools / "_config.json")
        assert rec["start"]["stubs_active"] is True
        for rel, digest in rec["end"]["outputs"].items():       # every recorded output, as recorded
            assert ws.modules["_common"].sha256_file(rec["dir"] / rel) == digest
    s001 = ws.record(pipe["seg_all"])
    assert s001["end"]["summary"]["failed"] == ["img_0004"] and s001["end"]["summary"]["n_cells"] == 17


def test_segment_outputs_and_per_image(ws, pipe):
    import tifffile

    s001, s002 = ws.record(pipe["seg_all"]), ws.record(pipe["seg_sub"])
    assert sorted(s001["end"]["outputs"]) == ["masks/img_0001.tif", "masks/img_0002.tif", "masks/img_0003.tif",
                                               "per_image.csv"]
    mask = tifffile.imread(s001["dir"] / "masks" / "img_0001.tif")
    assert mask.dtype == np.uint16 and mask.shape == (96, 96) and len(np.unique(mask)) == 7
    assert (s001["dir"] / "overlays" / "img_0001.png").exists()
    rows = {r["image_id"]: r for r in _rows(s001["dir"] / "per_image.csv")}
    assert rows["img_0004"]["status"] == "failed" and "cells=0" in rows["img_0004"]["error"]
    assert rows["img_0001"]["diameter"] == "30.0" and rows["img_0001"]["cell_count"] == "6"
    assert {r["device"] for r in rows.values()} == {"cuda"}         # the stub tool's reported device
    assert s001["end"]["summary"]["devices"] == ["cuda"] and s001["end"]["summary"]["cpu_fallback"] is False
    assert s001["start"]["set"] == {} and s001["start"]["set_levers"] == []
    sub = _rows(s002["dir"] / "per_image.csv")
    assert [r["image_id"] for r in sub] == ["img_0002"] and sub[0]["diameter"] == "50.0" and sub[0]["min_size"] == "5"
    # the per-image file's img_0001 entry is not applied (img_0001 is not in this call)
    assert s002["start"]["set_levers"] == ["diameter", "min_size"] and s002["start"]["per_image_sha256"]
    assert (s002["dir"] / "inputs" / "per_image.json").exists()
    assert len(list((s002["dir"]).glob("tool_results.jsonl"))) == 1


def test_crop_mask_name_rule_and_cell_ids(ws, pipe):
    import tifffile

    c001 = ws.record(pipe["crop_base"])
    per_image = _rows(c001["dir"] / "per_image.csv")
    assert [r["image_id"] for r in per_image] == ["img_0001", "img_0002", "img_0003"]
    # the CLI's default margin is the tool's rule (1 px for a cell_mask), the segment call's min_size the min_area
    assert {(r["margin"], r["margin_resolved"], r["min_area_passed"], r["min_area_resolved"]) for r in per_image} == \
        {("", "1", "15", "15")}
    assert {r["mask_call"] for r in per_image} == {pipe["seg_all"]}
    crops = _rows(c001["dir"] / "crops.csv")
    assert len(crops) == 17 and len({r["cell_id"] for r in crops}) == 17
    mask = tifffile.imread(ws.record(pipe["seg_all"])["dir"] / "masks" / "img_0001.tif")
    for row in crops:
        image_id, mask_call, label = row["cell_id"].rsplit("_", 2)
        assert image_id == row["image_id"] and mask_call == pipe["seg_all"] and int(label) == int(row["mask_label"])
        assert row["tool_cell_id"].startswith(f"default_{image_id}_cell_")
        assert ws.modules["_common"].sha256_file(c001["dir"] / row["crop_path"]) == row["sha256"]
        assert row["crop_h"] == row["crop_w"] and "qc_neighbor_contamination" in row
        if image_id == "img_0001":
            r, c = (int(round(v)) for v in json.loads(row["centroid"]))
            assert mask[r, c] == int(row["mask_label"])
    assert c001["start"]["inputs"]["from_calls"] == [pipe["seg_all"]]


def test_crop_masks_map_and_cell_overrides(ws, pipe):
    c002 = ws.record(pipe["crop_mixed"])
    per_image = {r["image_id"]: r for r in _rows(c002["dir"] / "per_image.csv")}
    assert {i: r["mask_call"] for i, r in per_image.items()} == {
        "img_0001": pipe["seg_all"], "img_0002": pipe["seg_sub"], "img_0003": pipe["seg_all"]}
    assert per_image["img_0001"]["n_cell_overrides"] == "1" and per_image["img_0001"]["margin_frac"] == "0.3"
    # each image's min_area is the min_size of the segment call that delivered its mask
    assert {i: r["min_area_resolved"] for i, r in per_image.items()} == {"img_0001": "15", "img_0002": "5",
                                                                         "img_0003": "15"}
    crops = {r["tool_cell_id"]: r for r in _rows(c002["dir"] / "crops.csv")}
    assert crops["default_img_0001_cell_0000"]["margin_mode"] == "mask_bbox"
    assert crops["default_img_0001_cell_0001"]["margin_mode"] == "fraction"       # reset: no keys
    assert c002["start"]["set"] == {"margin_frac": 0.3}
    assert c002["start"]["set_levers"] == ["cell_overrides", "margin_frac"]
    assert c002["start"]["cell_overrides_sha256"] and sorted(c002["start"]["inputs"]["from_calls"]) == \
        sorted([pipe["seg_all"], pipe["seg_sub"]])
    assert c002["end"]["summary"]["n_cell_overrides"] == 1
    # the same mask gives the same cell ids in another crop call
    base = {r["cell_id"] for r in _rows(ws.record(pipe["crop_base"])["dir"] / "crops.csv") if r["image_id"] == "img_0001"}
    assert base == {r["cell_id"] for r in crops.values() if r["image_id"] == "img_0001"}


def test_embed_outputs_crops_map_exclude_and_meta(ws, pipe):
    e001, e002 = ws.record(pipe["embed_base"]), ws.record(pipe["embed_mixed"])
    cells = _rows(e001["dir"] / "cells.csv")
    crops = _rows(ws.record(pipe["crop_base"])["dir"] / "crops.csv")
    assert [c["cell_id"] for c in cells] == [r["cell_id"] for r in crops]
    assert np.load(e001["dir"] / "embedding.npy").shape == (17, 16)
    assert (e001["dir"] / "cell_state_analyzed.h5ad").exists() and (e001["dir"] / "loss_history.json").exists()
    assert json.loads((e001["dir"] / "tool_result.json").read_text(encoding="utf-8"))["mode"] == "training"
    mixed = _rows(e002["dir"] / "cells.csv")
    assert pipe["excluded"] not in {c["cell_id"] for c in mixed}
    assert {(c["image_id"], c["crop_call"]) for c in mixed} == {
        ("img_0001", pipe["crop_mixed"]), ("img_0002", pipe["crop_base"]), ("img_0003", pipe["crop_mixed"])}
    assert e002["start"]["set_levers"] == ["augment_color"] and e002["start"]["params"]["seed"] == 1
    assert e002["start"]["inputs"]["exclude_cells"].endswith("inputs/exclude_cells.txt")
    assert (ws.workspace / e002["start"]["inputs"]["exclude_cells"]).read_text(encoding="utf-8").strip() == pipe["excluded"]
    analyzer = ws.modules["_common"].tool_class("Cell_State_Analyzer_Tool", {"allow_tool_stubs": True})
    last = analyzer.calls[-1]
    assert last["seed"] == 1 and last["augment_color"] is False and last["adaptation_depth"] == "trained"
    first = last["metadata"]
    assert first["cell_id"] == first["crop_id"] and first["plate"] == "vendor_a" and first["group"] == "default"


def test_batch_correct(ws, pipe):
    # images.csv has no group / treatment / control column: no calibration view, the input comes back unchanged
    b001, b002 = ws.record(pipe["batch_source"]), ws.record(pipe["batch_file"])
    e001, e002 = ws.record(pipe["embed_base"]), ws.record(pipe["embed_mixed"])
    assert (b001["dir"] / "cells.csv").read_bytes() == (e001["dir"] / "cells.csv").read_bytes()
    assert np.load(b001["dir"] / "embedding.npy").shape == (17, 16)
    assert np.array_equal(np.load(b001["dir"] / "embedding.npy"), np.load(e001["dir"] / "embedding.npy"))
    assert np.array_equal(np.load(b002["dir"] / "embedding.npy"), np.load(e002["dir"] / "embedding.npy"))
    batch_correct = ws.modules["batch_correct"]
    # tier none: vendor_b holds one image with cells (img_0004 has none); the batch file makes each image a batch
    for rec, tier in ((b001, "none"), (b002, "none")):
        result = json.loads((rec["dir"] / "tool_result.json").read_text(encoding="utf-8"))
        assert result["status"] == "UNCORRECTED_INPUT" and result["corrected"] is False
        assert result["reason"] == batch_correct.NO_CALIBRATION and result["tier"] == tier
        assert result["levers_applied"] is False and result["harmony_params_passed"] is None
        assert rec["end"]["summary"]["corrected"] is False and rec["end"]["summary"]["tier"] == tier
        assert rec["end"]["summary"]["reason"] == batch_correct.NO_CALIBRATION
        assert not (rec["dir"] / "tool").exists()
    assert b002["start"]["set_levers"] == ["theta"]                   # set, recorded, not applied
    assert b002["start"]["inputs"]["batch_file"].endswith("inputs/batch_file.csv")


def test_cluster(ws, pipe):
    import anndata as ad

    k002 = ws.record(pipe["states_multi"])
    states = _rows(k002["dir"] / "states.csv")
    assert list(states[0]) == ["cell_id", "image_id", "state_0.5", "state_1.0"]
    adata = ad.read_h5ad(k002["dir"] / "states.h5ad")
    np.testing.assert_array_equal(adata.obsm["X_embedding"], np.load(ws.record(pipe["embed_mixed"])["dir"] / "embedding.npy"))
    assert adata.obsm["X_umap"].shape == (len(states), 2) and list(adata.obs["cell_id"]) == [s["cell_id"] for s in states]
    assert list(adata.uns["chain"]) == [pipe["embed_mixed"], pipe["crop_base"], pipe["crop_mixed"], pipe["seg_all"],
                                        pipe["seg_sub"]]
    assert (k002["dir"] / "umap.png").exists()
    summary = k002["end"]["summary"]
    assert set(summary["resolutions"]) == {"0.5", "1.0"} and summary["n_neighbors_used"] == 5
    assert sum(summary["resolutions"]["0.5"]["sizes"]) == len(states)
    k001 = ws.record(pipe["states_base"])
    assert k001["start"]["set_levers"] == [] and k001["start"]["inputs"]["from_calls"] == [pipe["batch_source"]]
    assert list(ad.read_h5ad(k001["dir"] / "states.h5ad").uns["chain"]) == [
        pipe["batch_source"], pipe["embed_base"], pipe["crop_base"], pipe["seg_all"]]


def test_chain_verifies_with_the_harness_ledger(ws, pipe, monkeypatch):
    ledger = pytest.importorskip("benchmarks.full_HCMA.harness.ledger")
    # the chain's provenance check reads the start lines' code identity: stubbed calls are refused ...
    result = ledger.chain(ws.run, {"cluster_call": pipe["states_base"], "resolution": 0.5})
    assert any("stubs_active is True" in p for p in result["problems"])
    assert not any("tool_sha256" in p or "common_sha256" in p for p in result["problems"])
    # ... and accepted where the tests allow stubs, with the commands that ran the calls logged
    log = ws.run / "logs" / ledger.CONTROL_COMMANDS
    try:
        log.write_text("".join(json.dumps({"tool": e["tool"], "argv": e["argv"]}) + "\n"
                               for e in ws.ledger() if e["event"] == "start"), encoding="utf-8")
        monkeypatch.setattr(ledger, "ALLOW_STUBS", True)
        for key in ("states_base", "states_multi"):
            result = ledger.chain(ws.run, {"cluster_call": pipe[key], "resolution": 0.5})
            assert result["ok"], result["problems"]
    finally:
        log.unlink(missing_ok=True)


# ---------------------------------------------------------------- refusals

@pytest.mark.parametrize("tool, argv", [
    ("segment", ["--set", "diameter_px=30"]),                       # unknown lever
    ("segment", ["--set", "diameter=1000"]),                        # out of range
    ("segment", ["--set", "augment=1"]),                            # wrong type
    ("segment", ["--set", "min_size=2.5"]),                         # not an integer
    ("segment", ["--set", "color_class_filter=keep_all"]),          # not a choice
    ("segment", ["--images", "img_9999"]),                          # unknown image
    ("crop", ["--masks", "S001", "--set", "margin_mode=wide"]),
    ("crop", ["--masks", "S001", "--set", "cell_overrides=1"]),
    ("crop", ["--masks", "C001"]),                                  # not a segment call
    ("crop", ["--masks", "S099"]),                                  # no such call
    ("crop", ["--masks", "S001", "--images", "img_0004"]),          # no mask for it
    ("embed", ["--crops", "C001", "--set", "learning_rate=0.1"]),
    ("embed", ["--crops", "C001", "--set", "adaptation_depth=deep"]),
    ("embed", ["--crops", "C001", "--meta", "donor=source"]),
    ("batch_correct", ["--embedding", "E001", "--batch-key", "source", "--set", "theta=-1"]),
    ("batch_correct", ["--embedding", "E001", "--batch-key", "no_such_column"]),
    ("cluster", ["--embedding", "E001", "--set", "resolution=[0.5,0.5]"]),
    ("cluster", ["--embedding", "E001", "--set", "n_neighbors=1"]),
    ("cluster", ["--embedding", "C001"]),
])
def test_invalid_calls_exit_2_and_write_nothing(ws, pipe, tool, argv):
    before = ws.snapshot()
    assert ws.call(tool, "--name", "bad", *argv) == 2
    assert ws.snapshot() == before


def test_invalid_files_and_names(ws, pipe):
    before = ws.snapshot()
    assert ws.call("segment", "--name", "no spaces") == 2
    assert ws.call("segment", "--name", "x", "--per-image", ws.write("pi_bad.json", {"img_0001": {"diameter": "big"}})) == 2
    assert ws.call("crop", "--name", "x", "--masks", "S001", "--cell-overrides",
                   ws.write("co_bad1.json", {"img_0001": {"default_img_0001_cell_0000": {"auto_tighten": True}}})) == 2
    assert ws.call("crop", "--name", "x", "--masks", "S001", "--cell-overrides",
                   ws.write("co_bad2.json", {"img_0001": {"default_img_0001_cell_0099": {"recenter": True}}})) == 2
    assert ws.call("crop", "--name", "x", "--masks", "S001", "--cell-overrides",
                   ws.write("co_bad3.json", {"img_0001": {"default_img_0001_cell_0000": {"min_containment": 0.5}}})) == 2
    assert ws.call("crop", "--name", "x", "--masks", "S001", "--cell-overrides",
                   ws.write("co_bad4.json", {"img_0001": {"default_img_0001_cell_0000": {"reset": True, "recenter": True}}})) == 2
    assert ws.call("embed", "--name", "x", "--crops", "C001", "--exclude-cells", ws.write("ex_bad.txt", "no_such_cell\n")) == 2
    assert ws.call("embed", "--name", "x", "--crops", "C001", "--crops-map", ws.write("cm.json", {"img_0001": "C001"})) == 2
    assert ws.call("batch_correct", "--name", "x", "--embedding", "E001", "--batch-file",
                   ws.write("bf_bad.csv", "cell_id,batch\nonly_one,a\n")) == 2
    assert ws.call("segment", "--name", "x", "--config", ws.write("cfg_bad.json", {"set": {}, "levers": {}})) == 2
    assert ws.snapshot() == before


def test_dry_run_writes_nothing(ws, pipe, capsys):
    before = ws.snapshot()

    def dry(tool, *argv):
        capsys.readouterr()
        assert ws.call(tool, "--name", "dry", *argv, "--dry-run") == 0
        return json.loads(capsys.readouterr().out)

    out = dry("segment", "--set", "diameter=45")
    assert out["dry_run"] and out["set"] == {"diameter": 45.0} and out["params"]["cellprob_threshold"] == 0.0
    out = dry("crop", "--masks", "S001", "--set", "margin_mode=tight")
    assert out["set_levers"] == ["margin_mode"] and set(out["masks"]) == {"img_0001", "img_0002", "img_0003"}
    out = dry("cluster", "--embedding", "B001", "--config", ws.write("cfg.json", {"set": {"resolution": 2}}))
    assert out["params"]["resolution"] == [2.0] and out["set_levers"] == ["resolution"]
    out = dry("embed", "--crops", "C001", "--set", "seed=3")
    assert out["set_levers"] == [] and out["params"]["seed"] == 3
    assert ws.snapshot() == before


def test_changed_inputs_exit_4(ws, pipe):
    before = ws.snapshot()
    embedding = ws.record(pipe["embed_base"])["dir"] / "embedding.npy"
    original = embedding.read_bytes()
    try:
        embedding.write_bytes(original[:-4] + b"\x00\x00\x80\x3f")
        assert ws.call("cluster", "--name", "x", "--embedding", pipe["embed_base"]) == 4
        assert ws.call("batch_correct", "--name", "x", "--embedding", pipe["embed_base"], "--batch-key", "source") == 4
    finally:
        embedding.write_bytes(original)
    c001 = ws.record(pipe["crop_base"])
    crop = c001["dir"] / _rows(c001["dir"] / "crops.csv")[0]["crop_path"]
    original = crop.read_bytes()
    try:
        crop.write_bytes(original + b"\x00")
        assert ws.call("embed", "--name", "x", "--crops", pipe["crop_base"]) == 4
    finally:
        crop.write_bytes(original)
    mask = ws.record(pipe["seg_all"])["dir"] / "masks" / "img_0002.tif"
    original = mask.read_bytes()
    try:
        mask.write_bytes(original + b"\x00")
        assert ws.call("crop", "--name", "x", "--masks", pipe["seg_all"]) == 4
    finally:
        mask.write_bytes(original)
    assert ws.snapshot() == before


def test_deadline_exit_3(ws, pipe):
    before = ws.snapshot()
    deadline = ws.workspace / ".deadline"
    try:
        deadline.write_text(f"{time.time() + 300:.0f}\n", encoding="utf-8")
        assert ws.call("segment", "--name", "late") == 3
        assert ws.call("cluster", "--name", "late", "--embedding", pipe["embed_base"]) == 3
        assert ws.snapshot() == before
        assert ws.call("time_left") == 0
    finally:
        deadline.unlink()


def test_failed_tool_call_is_recorded(ws, pipe, monkeypatch):
    # an exception inside the call (here the cluster tool's clustering): failed, with the error and its traceback
    def boom(*args, **kwargs):
        raise RuntimeError("no clustering today")

    monkeypatch.setattr(ws.modules["cluster"], "compute_states", boom)
    code = ws.call("cluster", "--name", "broken", "--embedding", pipe["embed_base"])
    assert code == 1
    rec = ws.last("cluster")
    assert rec["start"]["name"] == "broken" and rec["end"]["status"] == "failed"
    assert "no clustering today" in rec["end"]["error"]
    assert json.loads((rec["dir"] / "result.json").read_text(encoding="utf-8"))["status"] == "failed"
    assert (rec["dir"] / "error.txt").exists()


# ---------------------------------------------------------------- describe, registry, helpers

def test_describe_prints_the_card_sections(ws, capsys):
    common = ws.modules["_common"]
    for tool, heading, needle in (("segment", "## Levers", "### diameter"), ("crop", "## Levers", "#### Per-cell keys"),
                                  ("batch_correct", "## Levers", "### theta"),
                                  ("embed", "## Tunable parameters", "### adaptation_depth")):
        capsys.readouterr()
        assert ws.call(tool, "--describe") == 0
        out = capsys.readouterr().out
        section = common.card_section(REPO, common.CARD_DIRS.get(tool, common.ANALYZER_DIR), heading)
        assert out.startswith(section) and needle in section and "## Usage" in out
    assert ws.call("cluster", "--describe") == 0
    out = capsys.readouterr().out
    assert "| resolution |" in out and "| n_neighbors |" in out


def test_tool_levers_from_the_real_cards(ws):
    levers = ws.modules["_common"].tool_levers
    assert list(levers("segment", REPO)) == ["diameter", "cellprob_threshold", "flow_threshold", "min_size", "niter",
                                             "tile_overlap", "augment", "resample", "color_class_filter"]
    assert list(levers("crop", REPO)) == ["margin_mode", "margin_frac", "margin", "recenter", "tight_buffer_px",
                                          "crop_size", "cell_overrides"]
    assert list(levers("batch_correct", REPO)) == [
        "theta", "sigma", "lambda_value", "nclust", "input_pca", "input_scale", "max_iter_harmony", "epsilon_harmony",
        "max_iter_kmeans", "epsilon_cluster", "block_size", "tau", "random_state"]
    assert list(levers("embed", REPO)) == ["adaptation_depth", "learning_rate", "max_epochs", "augment_color",
                                           "field_aware", "plate_mean_weight", "plate_adversarial_weight"]
    assert list(levers("cluster", REPO)) == ["resolution", "n_neighbors"]
    seg = levers("segment", REPO)
    assert (seg["diameter"].kind, seg["diameter"].bounds, seg["diameter"].default) == ("number", (5.0, 200.0), 30.0)
    assert seg["niter"].nullable and seg["niter"].kind == "integer"
    assert levers("crop", REPO)["cell_overrides"].kind == "per_cell"
    margin = levers("crop", REPO)["margin"]          # the CARD's null: the tool's rule by mask file name
    assert (margin.default, margin.nullable, margin.bounds) == (None, True, (0.0, 128.0))
    # the batch corrector's input_pca default (0 = off) lies outside its Range and is still valid
    common = ws.modules["_common"]
    assert common.check_value(levers("batch_correct", REPO)["input_pca"], 0) == 0.0


def test_cluster_states_match_the_analyzer_clustering():
    """cluster.py's graph + UMAP + Leiden on the REAL analyzer class equal the analyzer's own
    _compute_umap_and_clustering, at the first resolution and at a further one (same graph)."""
    pytest.importorskip("leidenalg")
    import anndata as ad

    path = list(sys.path)
    try:                 # cluster.py puts its directory on sys.path; the workspace's tools must stay first
        sys.path.insert(0, str(REPO))
        from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool

        cluster = _load(TEMPLATE_TOOLS / "cluster.py", "full_hcma_test_cluster_real")
    finally:
        sys.path[:] = path
    rng = np.random.default_rng(0)
    centres = rng.normal(scale=4, size=(4, 16))
    embedding = (centres[rng.integers(0, 4, 200)] + rng.normal(size=(200, 16))).astype(np.float32)
    ids = [f"c{n:03d}" for n in range(200)]
    adata, keys = cluster.compute_states(Cell_State_Analyzer_Tool, embedding, ids, [0.5, 1.0], 15)
    assert adata.obsm["X_umap"].shape == (200, 2)
    for resolution in (0.5, 1.0):
        reference = ad.AnnData(X=embedding.copy())
        reference.obs_names = ids
        key = Cell_State_Analyzer_Tool()._compute_umap_and_clustering(reference, resolution, None, 15, n_pcs=None)
        assert list(adata.obs[keys[resolution]]) == list(reference.obs[key])
    assert adata.obs[keys[0.5]].nunique() >= 2


def test_coverage_view_prints_coverage_and_the_chain_only_with_chain(ws, monkeypatch, capsys):
    seen = []
    fake = types.ModuleType("benchmarks.full_HCMA.harness.coverage")
    fake.lever_groups = lambda run: ["segment.diameter", "segment.niter", "crop.margin", "embed.seed_free"]
    fake.calls = lambda run: {"S002": {"run": run}}
    fake.coverage = lambda ledger, groups: {"segment.diameter": ["S002"], "segment.niter": [], "crop.margin": [],
                                            "embed.seed_free": []}
    fake.read_choice = lambda run: {"skipped": {"crop": "why not", "segment.niter": "  "}}
    fake.final_check = lambda run: seen.append(run) or {"accepted": False}
    fake.describe = lambda check: f"FULL CHECK {check}"
    monkeypatch.setitem(sys.modules, "benchmarks.full_HCMA.harness.coverage", fake)
    capsys.readouterr()
    assert ws.call("lever_coverage") == 0
    out = capsys.readouterr().out
    assert not seen and "segment.diameter: tested in S002" in out and "crop.margin: untested, skipped: why not" in out
    assert "segment.niter: UNTESTED" in out and "untested without a reason: segment.niter, embed.seed_free" in out
    assert "FINAL acceptance also verifies" in out and "--chain" in out
    assert ws.call("lever_coverage", "--chain") == 0
    assert capsys.readouterr().out.startswith("FULL CHECK") and [Path(r) for r in seen] == [ws.run]


def test_coverage_view_with_the_harness_coverage(ws, pipe, capsys):
    pytest.importorskip("benchmarks.full_HCMA.harness.coverage")
    capsys.readouterr()
    assert ws.call("lever_coverage") == 0
    out = capsys.readouterr().out
    assert f"segment.diameter: tested in {pipe['seg_sub']}" in out and "crop.cell_overrides: tested in" in out
    assert "cluster.n_neighbors: tested in" in out and "FINAL acceptance also verifies" in out
    assert ws.call("lever_coverage", "--chain") == 0
    assert "choice.json chain:" in capsys.readouterr().out


def test_exclusive_gpu_allows_one_segment_worker(ws, pipe, capsys):
    config_path = ws.tools / "_config.json"
    original = config_path.read_text(encoding="utf-8")
    before = ws.snapshot()
    try:
        config_path.write_text(json.dumps({**json.loads(original), "gpu_exclusive": True}), encoding="utf-8")
        capsys.readouterr()
        assert ws.call("segment", "--name", "excl", "--workers", "2") == 2
        assert "exclusive-process mode" in capsys.readouterr().err
        assert ws.call("segment", "--name", "excl", "--workers", "1", "--dry-run") == 0
    finally:
        config_path.write_text(original, encoding="utf-8")
    assert ws.snapshot() == before


def test_segment_reports_a_cpu_fallback(ws, pipe, monkeypatch):
    import torch

    monkeypatch.setenv("FULL_HCMA_STUB_DEVICE", "cpu")
    assert ws.call("segment", "--name", "oncpu", "--images", "img_0003") == 0
    rec = ws.last("segment")
    assert rec["start"]["name"] == "oncpu" and _rows(rec["dir"] / "per_image.csv")[0]["device"] == "cpu"
    summary = rec["end"]["summary"]
    assert summary["devices"] == ["cpu"] and summary["cpu_fallback"] is bool(torch.cuda.is_available())


def test_time_left_without_deadline(ws, capsys):
    capsys.readouterr()
    assert ws.call("time_left") == 0
    assert "no deadline" in capsys.readouterr().out


# ---------------------------------------------------------------- termination, dying workers, locks

def test_terminated_and_killed_calls_end_failed(ws, pipe):
    import signal

    common = ws.modules["_common"]
    cfg = common.load_config()
    seen = {}

    def body(call):
        seen["handler"] = signal.getsignal(signal.SIGTERM)
        seen["handler"](signal.SIGTERM, None)          # what a SIGTERM does inside the call

    call = common.Call(cfg, "cluster", "term", note="", argv=["--name", "term"], set_={}, set_levers=[], params={},
                       inputs={"from_calls": []})
    assert common.run_call(call, body, gpu=False) == 1
    assert seen["handler"] is common._terminate and signal.getsignal(signal.SIGTERM) is not common._terminate
    rec = ws.last("cluster")
    assert rec["start"]["name"] == "term" and rec["end"]["status"] == "failed" and rec["end"]["error"] == "terminated"
    assert json.loads((rec["dir"] / "result.json").read_text(encoding="utf-8"))["error"] == "terminated"
    # a call killed outright (no handler runs) keeps the result.json written at its start: failed; it has
    # no ledger end, so no later call can consume it
    killed = common.Call(cfg, "embed", "killed", note="", argv=[], set_={}, set_levers=[], params={},
                         inputs={"from_calls": []})
    killed.start()
    assert json.loads((killed.dir / "result.json").read_text(encoding="utf-8"))["status"] == "failed"
    before = ws.snapshot()
    assert ws.call("cluster", "--name", "x", "--embedding", killed.id) == 2
    assert ws.snapshot() == before


def test_a_dying_worker_fails_its_images_without_hanging(ws, pipe, monkeypatch):
    monkeypatch.setenv("FULL_HCMA_STUB_CRASH", "img_0002")
    started = time.time()
    code = ws.call("segment", "--name", "crash", "--images", "img_0001,img_0002", "--workers", "2")
    assert time.time() - started < 300
    rec = ws.last("segment")
    assert rec["start"]["name"] == "crash" and rec["end"]["status"] in ("partial", "failed")
    assert code == (0 if rec["end"]["status"] == "partial" else 1)
    rows = {r["image_id"]: r for r in _rows(rec["dir"] / "per_image.csv")}
    assert rows["img_0002"]["status"] == "failed" and "BrokenProcessPool" in rows["img_0002"]["error"]


def test_file_locks_and_gpu_slots_with_flock(ws, tmp_path, monkeypatch):
    import errno

    held = {}

    def flock(handle, flags):
        owner = held.get(handle.name)
        if owner is not None and owner is not handle and not owner.closed:
            if flags & 4:
                raise BlockingIOError(errno.EWOULDBLOCK, "locked")
            raise AssertionError("a blocking flock would wait forever here")
        held[handle.name] = handle

    monkeypatch.setitem(sys.modules, "fcntl", types.SimpleNamespace(LOCK_EX=2, LOCK_NB=4, flock=flock))
    common = ws.modules["_common"]
    first, second = common.FileLock(tmp_path / "a.lock"), common.FileLock(tmp_path / "a.lock")
    assert first.acquire(blocking=False) and not second.acquire(blocking=False)
    first.release()
    assert second.acquire(blocking=False)
    second.release()
    cfg = {"run": str(tmp_path), "gpu_slots": 2}
    waited = []
    with common.gpu_slot(cfg, lambda: waited.append(1)) as slot_a:
        with common.gpu_slot(cfg, lambda: waited.append(1)) as slot_b:
            assert {slot_a, slot_b} == {0, 1}
    with common.gpu_slot(cfg, lambda: waited.append(1)) as slot_c:
        assert slot_c == 0
    assert not waited and sorted(p.name for p in (tmp_path / "logs").iterdir()) == ["gpu_slot_0.lock", "gpu_slot_1.lock"]


def test_deadline_file_from_the_config_and_missing_file_refused(ws, pipe, tmp_path, capsys):
    config_path = ws.tools / "_config.json"
    original = config_path.read_text(encoding="utf-8")
    deadline = tmp_path / "outside" / "deadline"
    before = ws.snapshot()
    try:
        config_path.write_text(json.dumps({**json.loads(original), "deadline_file": str(deadline),
                                           "has_deadline": True}), encoding="utf-8")
        capsys.readouterr()
        assert ws.call("segment", "--name", "nodeadline") == 3            # the run has one, its file is gone
        assert "missing" in capsys.readouterr().err
        assert ws.call("time_left") == 0 and "refused" in capsys.readouterr().out
        deadline.parent.mkdir()
        deadline.write_text(f"{time.time() + 120:.0f}\n", encoding="utf-8")
        assert ws.call("cluster", "--name", "late", "--embedding", pipe["embed_base"]) == 3
    finally:
        config_path.write_text(original, encoding="utf-8")
    assert ws.snapshot() == before


def test_tools_run_without_pythonpath(ws, pipe):
    """The tools add `_config.json: repo` to sys.path themselves (the agent's environment has no PYTHONPATH)."""
    import subprocess

    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    for argv in (["segment.py", "--name", "nopath", "--dry-run"],
                 ["crop.py", "--name", "nopath", "--masks", pipe["seg_all"], "--dry-run"]):
        done = subprocess.run([sys.executable, f"tools/{argv[0]}", *argv[1:]], cwd=ws.workspace, env=env,
                              capture_output=True, text=True, timeout=600)
        assert done.returncode == 0, done.stderr[-2000:]
        assert json.loads(done.stdout)["dry_run"]


def test_no_tool_file_shadows_an_installed_module():
    """Each tool runs with tools/ first on sys.path, so a tool file named like an installed module
    (the earlier tools/coverage.py hid the `coverage` package from torch on the cluster) breaks its imports."""
    import importlib.util

    tools_dir = REPO / "benchmarks" / "full_HCMA" / "workspace_template" / "tools"
    clashes = []
    for path in sorted(tools_dir.glob("*.py")):
        name = path.stem
        if name in sys.stdlib_module_names:
            clashes.append(f"{name} (stdlib)")
            continue
        saved = list(sys.path)
        try:
            sys.path[:] = [p for p in saved if Path(p or ".").resolve() != tools_dir.resolve()]
            spec = importlib.util.find_spec(name) if name not in sys.modules else None
        finally:
            sys.path[:] = saved
        # a tools/ copy (template or a test workspace, both hold _common.py) is ours, not an installed module
        if spec is not None and spec.origin and not (Path(spec.origin).parent / "_common.py").exists():
            clashes.append(f"{name} ({spec.origin})")
    assert not clashes, f"tool files shadow installed modules: {clashes}"

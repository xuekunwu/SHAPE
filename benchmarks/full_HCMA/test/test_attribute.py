"""harness/attribute.py: the 2 x 2 objects x representation factorial (synthetic chains, a fake tool
runner); the pinned verdict lives in test_framework_metrics.py."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from benchmarks.full_HCMA.harness import attribute, control, coverage, hardware, ledger, score, subset, workspace
from benchmarks.full_HCMA.test._synthetic_chain import (Chain, build_private, digest, informative, workspace_data,
                                                        write_class_map)

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]


@pytest.fixture()
def root(tmp_path, monkeypatch):
    path = tmp_path / "root"
    build_private(path)
    monkeypatch.setattr(score, "CLASS_MAP", write_class_map(tmp_path / "morph_classes.csv"))
    monkeypatch.setattr(hardware, "gpu_slug", lambda: "testgpu")
    monkeypatch.setattr(hardware, "gpu_slots", lambda: 1)
    monkeypatch.setattr(hardware, "node_facts", lambda: {"gpu": "fake", "gpu_mem_gb": None, "cpus": 2, "mem_gb": 8.0})
    monkeypatch.setattr(score, "recluster", lambda x, ids, resolutions=score.SCIB_RESOLUTIONS, n_neighbors=15, timings=None: {
        r: np.argmax(np.asarray(x), axis=1).astype(str) for r in resolutions})
    levers = {"segment": {"diameter": None}, "crop": {"margin": None}, "embed": {"learning_rate": None},
              "batch_correct": {"theta": None}, "cluster": {"resolution": None}}
    monkeypatch.setattr(coverage, "_registry", lambda tools_dir: (tuple(levers), lambda tool, repo: levers[tool]))

    def setup(run, root, repo, *, budget_hours, gpu_slots, template=None, prep=None, metadata="full"):
        ws = Path(run) / "workspace"
        for sub in ("calls", "submission", "data", "tools"):
            (ws / sub).mkdir(parents=True)
        (Path(run) / "logs").mkdir()
        workspace_data(run, root, Path(prep).name, metadata)
        (ws / "tools" / "_config.json").write_text(json.dumps({"prep": str(prep), "prep_name": subset.prep_name(prep),
                                                               "metadata": metadata, "repo": str(repo)}))
        (Path(run) / "TOOLS_SHA256.json").write_text(json.dumps({"_config.json": digest(ws / "tools" / "_config.json")}))
        return ws

    monkeypatch.setattr(workspace, "setup_run", setup)
    return path


def _labels(root):
    return pd.read_csv(root / "private" / "labels.csv", dtype=str, keep_default_na=False)


class FakeTools:
    """A tool runner: parses each command like the tools do and writes a synthetic call (the
    embedding carries the class signal strongly unless adaptation_depth is zero_shot)."""

    def __init__(self, root: Path, crops_per_image):
        self.labels, self.crops_per_image, self.chains, self.seen = _labels(root), crops_per_image, {}, []

    def __call__(self, ws: Path, tool: str, args: list, log: Path) -> int:
        self.seen.append((ws, tool, list(args)))
        log.write_text("fake tool log\n", encoding="utf-8")
        chain = self.chains.setdefault(ws, Chain(ws.parent, traced=False))
        self.run_tool(chain, ws, tool, args)
        return 0

    @staticmethod
    def options(args: list) -> dict:
        out, key = {}, None
        for item in args:
            if item.startswith("--"):
                key = item[2:]
                out.setdefault(key, [])
            elif key is not None:
                out[key].append(item)
        return out

    def run_tool(self, chain: Chain, ws: Path, tool: str, args: list) -> str:
        o = self.options(args)
        name = o["name"][0]
        given = {k: json.loads(v) if v[:1] in "[{0123456789-\"" or v in ("true", "false", "null") else v
                 for k, _, v in (item.partition("=") for item in o.get("set", []))}
        if tool == "segment":
            return chain.segment(list(pd.read_csv(ws / "data" / "images.csv", dtype=str)["image_id"]), name=name)
        if tool == "crop":
            images = pd.read_csv(next((ws / "calls").glob(f"{o['masks'][0]}_*")) / "per_image.csv")["image_id"]
            return chain.crop(o["masks"][0], {i: self.crops_per_image(i) for i in images}, name=name)
        if tool == "embed":
            seed = int(given.pop("seed", 0))
            depth = given.get("adaptation_depth", "trained")
            labels = self.labels
            x_fn = lambda cells: informative(labels, cells, signal=6.0 if depth != "zero_shot" else 0.8,  # noqa: E731
                                             seed=seed)
            crops_map = json.loads((ws / o["crops-map"][0]).read_text()) if "crops-map" in o else None
            exclude = ([line.strip() for line in (ws / o["exclude-cells"][0]).read_text().splitlines() if line.strip()]
                       if "exclude-cells" in o else None)
            meta = dict(item.split("=", 1) for item in o.get("meta", []))
            return chain.embed(o["crops"][0] if "crops" in o else None, x_fn, name=name, crops_map=crops_map,
                               exclude=exclude, exclude_file=o["exclude-cells"][0] if exclude else None,
                               set_values=given, seed=seed, meta=meta, argv=args)
        if tool == "batch_correct":
            if "batch-file" in o:
                return chain.batch_correct(o["embedding"][0], name=name, set_values=given, argv=args,
                                           batch_file=pd.read_csv(ws / o["batch-file"][0], dtype=str))
            return chain.batch_correct(o["embedding"][0], name=name, batch_key=o["batch-key"][0], set_values=given,
                                       argv=args)
        resolutions = given.get("resolution", 0.5)
        resolutions = resolutions if isinstance(resolutions, list) else [resolutions]
        return chain.cluster(o["embedding"][0], name=name, resolutions=tuple(resolutions), set_values=given, argv=args)


def _control(root, runner) -> Path:
    run = control.run_control(root, REPO, seeds=[0], runner=runner, prep_name="full")
    assert run == root / "control" / "full" / "full" / "testgpu"
    return run


def _agent_run(root, name="claude-1", *, batch="key", exclude=True) -> Path:
    """An agent run on all images (6 crops each): a trained embedding with two excluded cells, a
    batch correction (by key, or by a batch file) and a cluster call at [0.5]."""
    run = root / "runs" / name
    workspace.setup_run(run, root, REPO, budget_hours=1, gpu_slots=1, prep=root / "prep")
    (run / "RUN_META.json").write_text(json.dumps({"gpu": "testgpu", "prep_name": "full", "metadata": "full"}))
    tools = FakeTools(root, lambda image_id: 6)
    chain = Chain(run, traced=True)
    s = tools.run_tool(chain, run / "workspace", "segment", ["--name", "seg"])
    c = tools.run_tool(chain, run / "workspace", "crop", ["--name", "crop", "--masks", s])
    cells = chain.crop_cells(c)
    excluded = list(cells["cell_id"].iloc[:2]) if exclude else []
    if excluded:
        (run / "workspace" / "drop.txt").write_text("\n".join(excluded) + "\n")
    e = tools.run_tool(chain, run / "workspace", "embed",
                       ["--name", "emb", "--crops", c, *(["--exclude-cells", "drop.txt"] if excluded else []),
                        "--set", "adaptation_depth=trained", "learning_rate=0.0001", "seed=3"])
    if batch == "key":
        b = chain.batch_correct(e, name="bc", batch_key="source", set_values={"theta": 2.0})
    elif batch in ("file_source", "file_random"):
        ecells = pd.read_csv(chain._dir(e) / "cells.csv", dtype=str)
        source = ecells["image_id"].map(_labels(root).set_index("image_id")["source"])
        values = source if batch == "file_source" else pd.Series(np.arange(len(ecells)) % 2).astype(str)
        b = chain.batch_correct(e, name="bc", batch_file=pd.DataFrame({"cell_id": ecells["cell_id"], "batch": values}))
    else:
        b = e
    k = chain.cluster(b, name="k", resolutions=(0.5,), set_values={"resolution": [0.5]})
    chain.choose({"cluster_call": k, "resolution": 0.5, "reason": "test", "skipped": {}})
    result = score.score_run(run, root, forbidden=[])
    assert result["status"] == "OK", result["chain"]["problems"]
    return run


def _control_crops(image_id: str) -> int:
    return 0 if image_id in ("img_0001", "img_0002") else 3          # the control's crops miss one c0 micrograph


def test_the_four_cells_and_the_decomposition(tmp_path, root):
    runner = FakeTools(root, _control_crops)
    ctrl = _control(root, runner)
    run = _agent_run(root)
    record = attribute.attribute(run, ctrl, tmp_path / "attr", root=root, repo=REPO, runner=runner)
    assert record["complete"]
    y00 = json.loads((ctrl / "RESULT.json").read_text())
    y11 = json.loads((run / "RESULT.json").read_text())
    for metric, key in (("primary", score.PRIMARY), ("cross_source", score.CROSS)):
        block = record["metrics"][metric]
        y = block["cells"]
        assert y["Y00"] == pytest.approx(y00[key]) and y["Y11"] == pytest.approx(y11[key])
        d = block["decomposition"]
        assert d["total"]["value"] == pytest.approx(d["phi_O"]["value"] + d["phi_R"]["value"])
        assert d["total"]["value"] == pytest.approx(y["Y11"] - y["Y00"])
        assert d["interaction"]["value"] == pytest.approx(y["Y11"] - y["Y10"] - y["Y01"] + y["Y00"])
        assert d["phi_O"]["value"] == pytest.approx(0.5 * ((y["Y10"] - y["Y00"]) + (y["Y11"] - y["Y01"])))
        # the type bootstrap reproduces the four values on the full sample; the CIs contain the point
        for cell in attribute.CELLS:
            assert block["bootstrap"]["full_sample"][cell] == pytest.approx(y[cell])
        for part in d.values():
            assert part["ci95"][0] - 1e-9 <= part["value"] <= part["ci95"][1] + 1e-9
    d = record["metrics"]["primary"]["decomposition"]
    assert d["phi_R"]["value"] > 0.1                   # the trained recipe beats zero-shot
    assert d["phi_O"]["value"] > 0                     # the run's crops cover the micrograph the control missed
    # the transfers: exclusions travel with O (Y10) and are dropped from Y01, reported
    t10, t01 = record["transfers"]["Y10"], record["transfers"]["Y01"]
    assert "run_exclude_cells" in t01["dropped"] and "2 cells" in t01["dropped"]["run_exclude_cells"]
    starts = lambda cell: [json.loads(line) for line in (Path(record["transfers"][cell]["replay"]) / "logs"  # noqa: E731
                                                          / "tool_calls.jsonl").read_text().splitlines()
                           if json.loads(line)["event"] == "start"]
    e10 = next(s for s in starts("Y10") if s["tool"] == "embed")
    e01 = next(s for s in starts("Y01") if s["tool"] == "embed")
    assert "--exclude-cells" in e10["argv"] and "--exclude-cells" not in e01["argv"]
    assert e10["set"] == {"adaptation_depth": "zero_shot"} and e10["params"]["seed"] == 0
    assert e01["set"] == {"adaptation_depth": "trained", "learning_rate": 0.0001} and e01["params"]["seed"] == 3
    b01 = next(s for s in starts("Y01") if s["tool"] == "batch_correct")
    assert b01["inputs"]["batch_key"] == "source" and b01["set"] == {"theta": 2.0}
    imported = [s["call"] for s in starts("Y10") if s.get(ledger.IMPORTED)]
    assert imported == t10["imported"] == ["S001", "C001"] and set(t10["calls"]) == {"embed", "batch_correct", "cluster"}
    commands = (Path(t10["replay"]) / "logs" / ledger.CONTROL_COMMANDS).read_text().splitlines()
    assert [json.loads(c)["tool"] for c in commands] == ["embed", "batch_correct", "cluster"]
    out = tmp_path / "attr"
    assert (out / "attribution.png").stat().st_size > 0
    md = (out / "ATTRIBUTION.md").read_text()
    assert "| O1: run objects | Y10" in md and "- phi_O = " in md and "(95 % CI" in md
    assert json.loads((out / "ATTRIBUTION.json").read_text())["metrics"]["primary"]["cells"] == record["metrics"]["primary"]["cells"]


def test_a_batch_file_falls_back_to_its_batch_key(tmp_path, root):
    runner = FakeTools(root, _control_crops)
    ctrl = _control(root, runner)
    run = _agent_run(root, "claude-file", batch="file_source")
    record = attribute.attribute(run, ctrl, tmp_path / "attr", root=root, repo=REPO, runner=runner)
    assert record["complete"] and "replaced by --batch-key source" in record["transfers"]["Y01"]["dropped"]["batch_file"]
    starts = [json.loads(line) for line in (Path(record["transfers"]["Y01"]["replay"]) / "logs"
                                             / "tool_calls.jsonl").read_text().splitlines()]
    b = next(s for s in starts if s["event"] == "start" and s["tool"] == "batch_correct")
    assert "--batch-key" in b["argv"] and b["argv"][b["argv"].index("--batch-key") + 1] == "source"


def test_a_batch_file_without_a_key_leaves_y01_incomplete(tmp_path, root):
    runner = FakeTools(root, _control_crops)
    ctrl = _control(root, runner)
    run = _agent_run(root, "claude-random", batch="file_random")
    record = attribute.attribute(run, ctrl, tmp_path / "attr", root=root, repo=REPO, runner=runner)
    assert not record["complete"] and "Y01" not in record["cells"]
    assert record["transfers"]["Y01"]["incomplete"] and record["metrics"]["primary"]["decomposition"] is None
    assert not (tmp_path / "attr" / "attribution.png").exists() and "INCOMPLETE" in (tmp_path / "attr" / "ATTRIBUTION.md").read_text()


def test_an_uncorrected_run_transfers_no_batch_call(tmp_path, root):
    runner = FakeTools(root, _control_crops)
    ctrl = _control(root, runner)
    run = _agent_run(root, "claude-nobatch", batch=None, exclude=False)
    record = attribute.attribute(run, ctrl, tmp_path / "attr", root=root, repo=REPO, runner=runner)
    assert set(record["transfers"]["Y01"]["calls"]) == {"embed", "cluster"}
    assert set(record["transfers"]["Y10"]["calls"]) == {"embed", "batch_correct", "cluster"}
    assert record["transfers"]["Y01"]["dropped"] == {}


@pytest.mark.parametrize("change, reason", [
    ({"expected_images_sha256": "0" * 64}, "expected image set sha256"),
    ({"control_chain": {"embed": "trained", "rule": "fine-tune"}}, "not the zero-shot control"),
])
def test_refusals(tmp_path, root, change, reason):
    runner = FakeTools(root, _control_crops)
    ctrl = _control(root, runner)
    run = _agent_run(root, "claude-refused")
    path = ctrl / "RESULT.json"
    path.write_text(json.dumps({**json.loads(path.read_text()), **change}))
    with pytest.raises(SystemExit, match=reason):
        attribute.attribute(run, ctrl, tmp_path / "attr", root=root, repo=REPO, runner=runner)
    assert not (tmp_path / "attr" / "replay_Y10").exists()


def test_another_gpu_model_is_refused(tmp_path, root, monkeypatch):
    runner = FakeTools(root, _control_crops)
    ctrl = _control(root, runner)
    run = _agent_run(root, "claude-gpu")
    (run / "RUN_META.json").write_text(json.dumps({"gpu": "Other GPU", "prep_name": "full", "metadata": "full"}))
    with pytest.raises(SystemExit, match="GPU models differ"):
        attribute.attribute(run, ctrl, tmp_path / "attr", root=root, repo=REPO, runner=runner)


def test_imports_are_verified(tmp_path, root):
    runner = FakeTools(root, _control_crops)
    ctrl = _control(root, runner)
    run = _agent_run(root, "claude-import")
    record = attribute.attribute(run, ctrl, tmp_path / "attr", root=root, repo=REPO, runner=runner)
    replay = Path(record["transfers"]["Y10"]["replay"])
    choice = json.loads((replay / "workspace" / "submission" / "choice.json").read_text())
    assert ledger.chain(replay, choice)["ok"]
    # an imported ledger line that differs from the donor's is refused by the chain walker
    path = replay / "logs" / "tool_calls.jsonl"
    lines = [json.loads(line) for line in path.read_text().splitlines()]
    for line in lines:
        if line.get(ledger.IMPORTED) and line["call"] == "C001" and line["event"] == "start":
            line["note"] = "edited after the import"
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n")
    verified = ledger.chain(replay, choice)
    assert not verified["ok"] and any("ledger lines differ from the donor's" in p for p in verified["problems"])
    # a donor output that no longer matches its recorded sha256 is refused at the import
    crops = next((run / "workspace" / "calls").glob("C001_*")) / "crops.csv"
    crops.write_text(crops.read_text() + "\n")
    (tmp_path / "replay" / "logs").mkdir(parents=True)
    (tmp_path / "replay" / "workspace" / "calls").mkdir(parents=True)
    with pytest.raises(SystemExit, match="does not match its recorded sha256"):
        attribute.import_calls(run, tmp_path / "replay", ["C001"])


def test_decompose_arithmetic():
    parts = attribute.decompose({"Y00": 0.4, "Y10": 0.5, "Y01": 0.6, "Y11": 0.9})
    assert parts["phi_O"] == pytest.approx(0.5 * (0.1 + 0.3)) and parts["phi_R"] == pytest.approx(0.5 * (0.2 + 0.4))
    assert parts["interaction"] == pytest.approx(0.2) and parts["total"] == pytest.approx(parts["phi_O"] + parts["phi_R"])

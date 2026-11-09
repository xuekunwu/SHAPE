"""harness/ledger.py: ledger reading and the chain walk behind choice.json."""
from __future__ import annotations

import json

import numpy as np

from benchmarks.full_HCMA.harness import ledger
from benchmarks.full_HCMA.test._synthetic_chain import Chain, build_private, full_chain, informative


def _labels(tmp_path):
    return build_private(tmp_path / "root", classes=2, parents_per_class=2)


def _cells(labels, n=3):
    return {image_id: n for image_id in labels["image_id"]}


def _x(labels):
    return lambda cells: informative(labels, cells)


def test_read_ledger_skips_broken_lines(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    full_chain(run, labels, _cells(labels), _x(labels))
    with open(run / "logs" / "tool_calls.jsonl", "a", encoding="utf-8") as handle:
        handle.write("not json\n[1, 2]\n")
    entries = ledger.read_ledger(run)
    assert len(entries) == 10                      # five calls, a start and an end each
    records = ledger.calls(run)
    assert sorted(records) == ["B001", "C001", "E001", "K001", "S001"]
    assert records["E001"]["dir"].name.startswith("E001_")
    assert records["K001"]["end"]["status"] == "done"


def test_valid_chain_with_and_without_batch_correct(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    chain, ids = full_chain(run, labels, _cells(labels), _x(labels))
    result = ledger.chain(run, {"cluster_call": ids["K"], "resolution": 0.5})
    assert result["ok"], result["problems"]
    assert set(result["calls"]) == {"S001", "C001", "E001", "B001", "K001"}
    assert result["embedding_call"] == "B001" and result["embed_call"] == "E001"
    assert result["state_column"] == "state_0.5" and result["n_cells"] == 3 * len(labels)
    assert result["n_excluded"] == 0

    direct = chain.cluster(ids["E"], name="direct", resolutions=(0.5, 1.0))
    result = ledger.chain(run, {"cluster_call": direct, "resolution": 1})
    assert result["ok"], result["problems"]
    assert result["embedding_call"] == result["embed_call"] == "E001"
    assert result["state_column"] == "state_1.0"


def test_tampered_files_break_the_chain(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    chain, ids = full_chain(run, labels, _cells(labels), _x(labels))
    e_dir = next((run / "workspace" / "calls").glob("E001_*"))
    x = np.load(e_dir / "embedding.npy")
    np.save(e_dir / "embedding.npy", x + 1)
    result = ledger.chain(run, {"cluster_call": ids["K"], "resolution": 0.5})
    assert not result["ok"]
    assert any("E001: embedding.npy changed" in p for p in result["problems"])

    run2 = tmp_path / "run2"
    chain, ids = full_chain(run2, labels, _cells(labels), _x(labels))
    crop = next((run2 / "workspace" / "calls").glob("C001_*/crops/*/*.png"))
    crop.write_bytes(b"edited")
    result = ledger.chain(run2, {"cluster_call": ids["K"], "resolution": 0.5})
    assert not result["ok"]
    assert any("C001: crop files missing or changed" in p for p in result["problems"])

    run3 = tmp_path / "run3"
    chain, ids = full_chain(run3, labels, _cells(labels), _x(labels))
    mask = next((run3 / "workspace" / "calls").glob("S001_*/masks/*.tif"))
    mask.write_bytes(b"edited")
    result = ledger.chain(run3, {"cluster_call": ids["K"], "resolution": 0.5})
    assert not result["ok"]
    assert any(p.startswith("S001: masks/") and "changed" in p for p in result["problems"])


def test_cluster_input_must_be_a_completed_call(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    chain, ids = full_chain(run, labels, _cells(labels), _x(labels))
    failed = chain.embed(ids["C"], _x(labels), name="broken", status="failed")
    k = chain.cluster(failed, name="on_failed")
    result = ledger.chain(run, {"cluster_call": k, "resolution": 0.5})
    assert not result["ok"]
    assert any(p.startswith(f"{failed}: status failed") for p in result["problems"])

    partial = chain.embed(ids["C"], _x(labels), name="partial", status="partial")
    k = chain.cluster(partial, name="on_partial")
    assert ledger.chain(run, {"cluster_call": k, "resolution": 0.5})["ok"]

    k_partial = chain.cluster(ids["E"], name="k_partial", status="partial")
    result = ledger.chain(run, {"cluster_call": k_partial, "resolution": 0.5})
    assert not result["ok"] and any("needs done" in p for p in result["problems"])


def test_resolution_must_be_one_of_the_cluster_call(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    _, ids = full_chain(run, labels, _cells(labels), _x(labels))
    result = ledger.chain(run, {"cluster_call": ids["K"], "resolution": 1.0})
    assert not result["ok"]
    assert any("resolution 1.0 is not one of K001's resolutions" in p for p in result["problems"])
    assert not ledger.chain(run, {"cluster_call": ids["K"], "resolution": "0.5"})["ok"]
    assert not ledger.chain(run, {"cluster_call": "K009", "resolution": 0.5})["ok"]
    assert not ledger.chain(run, {})["ok"]


def test_excluded_cells_are_accounted(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    chain = Chain(run)
    s = chain.segment(list(labels["image_id"]))
    c = chain.crop(s, _cells(labels))
    cells = chain.crop_cells(c)["cell_id"].tolist()

    e = chain.embed(c, _x(labels), name="excl", exclude=cells[:2])
    k = chain.cluster(e, name="k_excl")
    result = ledger.chain(run, {"cluster_call": k, "resolution": 0.5})
    assert result["ok"], result["problems"]
    assert result["n_excluded"] == 2 and result["n_cells"] == len(cells) - 2
    assert result["excluded_per_image"] == {cells[0].rsplit("_", 2)[0]: 2}

    e = chain.embed(c, _x(labels), name="dropped", drop=cells[:1])
    k = chain.cluster(e, name="k_dropped")
    result = ledger.chain(run, {"cluster_call": k, "resolution": 0.5})
    assert not result["ok"]
    assert any("neither embedded nor excluded" in p and cells[0] in p for p in result["problems"])

    e = chain.embed(c, _x(labels), name="edited_exclude", exclude=cells[:2])
    (run / "workspace" / "exclude_edited_exclude.txt").write_text(cells[0] + "\n", encoding="utf-8")
    k = chain.cluster(e, name="k_edited")
    result = ledger.chain(run, {"cluster_call": k, "resolution": 0.5})
    assert not result["ok"]
    assert any("exclude-cells file" in p and "changed" in p for p in result["problems"])


def test_states_must_cover_the_embedding_cells(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    chain, ids = full_chain(run, labels, _cells(labels), _x(labels))
    k_dir = next((run / "workspace" / "calls").glob(f"{ids['K']}_*"))
    import pandas as pd

    states = pd.read_csv(k_dir / "states.csv", dtype=str).iloc[1:]
    states.to_csv(k_dir / "states.csv", index=False)
    # rewrite the ledger digest so only the cell mismatch remains
    import json

    from benchmarks.full_HCMA.test._synthetic_chain import digest

    path = run / "logs" / "tool_calls.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        if line["call"] == ids["K"] and line["event"] == "end":
            line["outputs"]["states.csv"] = digest(k_dir / "states.csv")
    path.write_text("\n".join(json.dumps(line) for line in lines) + "\n", encoding="utf-8")
    result = ledger.chain(run, {"cluster_call": ids["K"], "resolution": 0.5})
    assert not result["ok"]
    assert any("its states cover other cells" in p for p in result["problems"])


def _rewrite(run, call_id, event, **fields):
    path = run / "logs" / "tool_calls.jsonl"
    lines = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]
    for line in lines:
        if line["call"] == call_id and line["event"] == event:
            line.update(fields)
    path.write_text("".join(json.dumps(line) + "\n" for line in lines), encoding="utf-8")


def test_a_forged_call_does_not_verify(tmp_path):
    """Files and ledger lines written without running the tool: no traced command runs them."""
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    chain, ids = full_chain(run, labels, _cells(labels), _x(labels))
    forger = Chain(run, traced=False)
    forger.counter = dict(chain.counter)
    forged = forger.embed(ids["C"], _x(labels), name="mine")
    k = chain.cluster(forged, name="k_mine")
    result = ledger.chain(run, {"cluster_call": k, "resolution": 0.5})
    assert not result["ok"]
    assert result["problems"] == [f"{forged}: no traced agent Bash command runs tools/embed.py --name mine"]


def test_code_identity_and_duplicate_lines(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "run"
    _, ids = full_chain(run, labels, _cells(labels), _x(labels))
    choice = {"cluster_call": ids["K"], "resolution": 0.5}
    _rewrite(run, "E001", "start", tool_sha256="0" * 64)
    _rewrite(run, "C001", "start", stubs_active=True)
    _rewrite(run, "S001", "start", common_sha256=None)
    with open(run / "logs" / "tool_calls.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "end", "call": "B001", "tool": "batch_correct", "status": "done"}) + "\n")
    problems = ledger.chain(run, choice)["problems"]
    assert "E001: tool_sha256 is not the sha256 of the template's embed.py" in problems
    assert "C001: stubs_active is True (needs false)" in problems
    assert "S001: common_sha256 is not the sha256 of the template's _common.py" in problems
    assert "B001: 1 more start / end line(s) in the ledger" in problems and len(problems) == 4


def test_control_runs_are_checked_against_the_driver_log(tmp_path):
    labels = _labels(tmp_path)
    run = tmp_path / "control"
    chain = Chain(run, traced=False)
    s = chain.segment(list(labels["image_id"]), name="defaults")
    c = chain.crop(s, _cells(labels), name="defaults")
    e = chain.embed(c, _x(labels), name="defaults_s0")
    k = chain.cluster(e, name="defaults_s0")
    problems = ledger.chain(run, {"cluster_call": k, "resolution": 0.5})["problems"]
    assert len(problems) == 4 and all("no line of logs/control_commands.jsonl" in p for p in problems)
    with open(run / "logs" / "control_commands.jsonl", "w", encoding="utf-8") as handle:
        for tool, name in (("segment", "defaults"), ("crop", "defaults"), ("embed", "defaults_s0"),
                           ("cluster", "defaults_s0")):
            handle.write(json.dumps({"tool": tool, "argv": ["--name", name], "returncode": 0}) + "\n")
    assert ledger.chain(run, {"cluster_call": k, "resolution": 0.5})["ok"]
    assert ledger.ran_by_command("embed", "a", ["nohup python tools/embed.py --name 'a' --crops C1 > logs/a.log &"])
    assert not ledger.ran_by_command("embed", "a", ["python tools/embed.py --name ab"])
    assert not ledger.ran_by_command("embed", "a", ["python tools/cluster.py --name a"])

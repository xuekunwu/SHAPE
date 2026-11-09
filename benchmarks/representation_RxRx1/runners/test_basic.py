"""Local gates use the benchmark's synthetic, label-free fixtures."""
import json
import os
import shutil
import subprocess
from pathlib import Path

import pytest

from benchmarks.representation_RxRx1.runners import basic
from benchmarks.representation_RxRx1.runners.test_benchmark import fake_cp_measure, prepared, stub

# The mocked shell scripts' hang guard: Git Bash's process starts slow down several-fold on a loaded
# Windows host (a 5 s script ran past 30 s under the parallel suite).
BASH_TIMEOUT = 300


def test_control_reuse_preserves_evidence(prepared, stub):
    from benchmarks.representation_RxRx1.runners import control

    root, _, _ = prepared
    assert not basic.check_control(root)
    control.control(root, seeds=(0,))
    path = root / "control" / "CONTROL.json"
    before, calls = path.read_bytes(), len(stub.calls)
    assert basic.check_control(root)
    assert path.read_bytes() == before and len(stub.calls) == calls
    ledger = json.loads(before)
    ledger["provenance"]["tool_sha256"] = "stale"
    path.write_text(json.dumps(ledger))
    with pytest.raises(SystemExit, match="CONTROL_STALE"):
        basic.check_control(root)


def test_the_dinov3_arm_kwargs_are_the_epoch_budget_without_a_floor():
    """A23: dinov3 (seed 0) and dinov3_s<k> train at default_epochs(E) = E x the round budget with min_epochs 1 (no
    floor); split_key stays (the tool writes resume_best only with a split; the input holds train crops only)."""
    from benchmarks.representation_RxRx1.runners import control
    from benchmarks.representation_RxRx1.runners.config import DEFAULT

    planned = control.arms((0, 2), "split", 9)
    assert set(planned) == {"frozen", "dinov3", "dinov3_s2"}
    assert planned["frozen"] == {"adaptation_depth": "zero_shot", "split_key": "split"}
    for name, seed in (("dinov3", 0), ("dinov3_s2", 2)):
        assert planned[name] == {"seed": seed, "split_key": "split", "max_epochs": 9 * DEFAULT.max_rounds,
                                 "min_epochs": 1}
    assert control.default_epochs(13) == 13 * DEFAULT.max_rounds
    assert [control.is_dinov3(n) for n in ("dinov3", "dinov3_s1", "dinov3_s", "dinov3_sx", "default_s0")] == [
        True, True, False, False, False]
    assert (control.input_splits("frozen"), control.input_splits("dinov3")) == (("train", "val"), ("train",))


def test_a_kept_dinov3_arm_with_other_kwargs_or_another_input_is_stale(prepared, stub):
    from benchmarks.representation_RxRx1.runners import control

    root, _, _ = prepared
    record = control.control(root, seeds=(0,))
    cap = json.loads((root / "TRAINING_BUDGET.json").read_text())["max_epochs"]
    assert record["arms"]["dinov3"]["kwargs"] == {"seed": 0, "split_key": "split",
                                                  "max_epochs": control.default_epochs(cap), "min_epochs": 1}
    path = root / "control" / "CONTROL.json"
    current = json.loads(path.read_text())
    old = json.loads(path.read_text())
    old["arms"]["dinov3"]["kwargs"] = {"seed": 0, "split_key": "split",
                                       "max_epochs": control.default_epochs(cap)}       # with the tool's floor
    path.write_text(json.dumps(old))
    with pytest.raises(SystemExit, match="CONTROL_STALE: dinov3 kwargs differ"):
        basic.check_control(root)
    del old["complete"]
    path.write_text(json.dumps(old))
    calls = len(stub.calls)
    with pytest.raises(SystemExit, match="CONTROL_STALE: .* holds dinov3 with other kwargs or another input"):
        control.control(root, seeds=(0,))
    assert len(stub.calls) == calls
    other = json.loads(json.dumps(current))
    other["arms"]["dinov3"]["input"] = dict(current["arms"]["frozen"]["input"])       # A22's input: train + val
    path.write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="CONTROL_STALE: dinov3 trained on another input"):
        basic.check_control(root)
    del other["complete"]
    path.write_text(json.dumps(other))
    with pytest.raises(SystemExit, match="CONTROL_STALE: .* holds dinov3 with other kwargs or another input"):
        control.control(root, seeds=(0,))
    assert len(stub.calls) == calls


def test_a_cached_default_s0_control_is_stale_and_preserved(prepared, stub):
    """A23: a CONTROL.json of A22 (frozen + default_s0, val-proxy selection) is reported stale, never reused or
    recomputed in place, complete or not."""
    from benchmarks.representation_RxRx1.runners import control

    root, _, _ = prepared
    record = control.control(root, seeds=(0,))
    path = root / "control" / "CONTROL.json"
    arms = dict(record["arms"])
    kwargs = {"seed": 0, "split_key": "split", "max_epochs": arms["dinov3"]["kwargs"]["max_epochs"]}   # A22's
    arms["default_s0"] = {**arms.pop("dinov3"), "kwargs": kwargs}
    legacy = {**record, "arms": arms}
    path.write_text(json.dumps(legacy))
    before = path.read_bytes()
    with pytest.raises(SystemExit, match=r"CONTROL_STALE: cached control holds the retired \['default_s0'\]"):
        basic.check_control(root)
    del legacy["complete"]
    path.write_text(json.dumps(legacy))
    before, calls = path.read_bytes(), len(stub.calls)
    with pytest.raises(SystemExit, match="CONTROL_STALE: .* holds the retired"):
        control.control(root, seeds=(0,))
    assert path.read_bytes() == before and len(stub.calls) == calls


def test_control_missing_artifact_and_extra_seed_fail_closed(prepared, stub):
    from benchmarks.representation_RxRx1.runners import control

    root, _, _ = prepared
    record = control.control(root, seeds=(0,))
    path = root / "control" / "CONTROL.json"
    extra = {**record, "arms": {**record["arms"], "dinov3_s1": record["arms"]["dinov3"]}}
    path.write_text(json.dumps(extra))
    with pytest.raises(SystemExit, match="NOT_BASIC"):
        basic.check_control(root)
    path.write_text(json.dumps(record))
    Path(record["arms"]["dinov3"]["resume_best"]).unlink()
    with pytest.raises(SystemExit, match="CONTROL_MISSING"):
        basic.check_control(root)


def test_external_checks_real_integrity(prepared, fake_cp_measure, tmp_path):
    from benchmarks.representation_RxRx1.runners import external

    root, _, _ = prepared
    assert not basic.check_external(root)
    external.run(root, root.parent, ("cellprofiler",), workers=1)
    assert not basic.check_external(root)
    path = root / "external" / "cellprofiler" / "EMBED.json"
    record = json.loads(path.read_text())
    record["external_sha256"] = "stale"
    path.write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="EXTERNAL_STALE"):
        basic.check_external(root)


def test_external_complete_and_wrong_fit(monkeypatch, tmp_path):
    from benchmarks.representation_RxRx1.runners import external, prepare

    monkeypatch.setattr(prepare, "load", lambda _: ({}, [1, 2]))
    records = {"openphenom": {"status": "scored", "settings": {"fit": "pad"}},
               "cellprofiler": {"status": "scored"}}
    monkeypatch.setattr(external, "load_embeddings", lambda *args: ({}, records))
    assert basic.check_external(tmp_path)
    records["openphenom"]["settings"]["fit"] = "resize"
    with pytest.raises(SystemExit, match="EXTERNAL_STALE"):
        basic.check_external(tmp_path)


@pytest.fixture()
def complete_result(tmp_path):
    tables = {name: {split: {"value": 0.2} for split in ("seen", "heldout")} for name in basic.ARMS}
    record = {"default_arms": ["dinov3"], "decision": {"kind": "final"}, "tables": tables,
              "matched": {"tables": json.loads(json.dumps(tables))},
              "llm_arms": {name: {"status": "delivered"} for name in ("llm", "react")},
              "external_arms": {name: {"status": "scored"} for name in ("openphenom", "cellprofiler")}}
    (tmp_path / "RESULT.json").write_text(json.dumps(record))
    return tmp_path, record


def test_complete_result(complete_result):
    run, _ = complete_result
    basic.check_result(run)


@pytest.mark.parametrize("fault", ["missing_arm", "missing_map", "nan_map", "provisional", "failed_llm",
                                   "skipped_external", "extra_seed"])
def test_result_rejects_incomplete_or_nonbasic(complete_result, fault):
    run, record = complete_result
    if fault == "missing_arm":
        del record["matched"]["tables"]["react"]
    elif fault == "missing_map":
        record["tables"]["llm"]["heldout"]["value"] = None
    elif fault == "nan_map":
        record["tables"]["llm"]["heldout"]["value"] = float("nan")
    elif fault == "provisional":
        record["decision"]["kind"] = "provisional"
    elif fault == "failed_llm":
        record["llm_arms"]["llm"]["status"] = "failed"
    elif fault == "skipped_external":
        record["external_arms"]["cellprofiler"]["status"] = "skipped"
    else:
        record["default_arms"].append("dinov3_s1")
    (run / "RESULT.json").write_text(json.dumps(record))
    with pytest.raises(SystemExit, match="INCOMPLETE|NOT_BASIC"):
        basic.check_result(run)


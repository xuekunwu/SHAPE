"""harness/coverage.py: lever groups from the tools' registries, counting, skips, the FINAL check."""
from __future__ import annotations

import json

import pytest

from benchmarks.full_HCMA.harness import coverage
from benchmarks.full_HCMA.test._synthetic_chain import Chain, build_private, informative

LEVERS = {"segment": ["diameter", "flow_threshold"], "crop": ["margin"], "embed": ["learning_rate"],
          "batch_correct": ["theta"], "cluster": ["resolution"]}


@pytest.fixture()
def stub(monkeypatch):
    monkeypatch.setattr(coverage, "_registry",
                        lambda tools_dir: (tuple(LEVERS), lambda tool, repo: {n: None for n in LEVERS[tool]}))


def _run(tmp_path, *, skipped=None, resolution=0.5):
    """A run whose chain is valid; segment.diameter set per image, crop.cell_overrides set, a failed
    embed call that set learning_rate (does not count), a partial batch_correct that set theta."""
    labels = build_private(tmp_path / "root", classes=2, parents_per_class=2)
    run = tmp_path / "run"
    chain = Chain(run)
    s = chain.segment(list(labels["image_id"]), set_levers=["diameter"])
    c = chain.crop(s, {i: 2 for i in labels["image_id"]}, set_levers=["cell_overrides"])
    chain.embed(c, lambda cells: informative(labels, cells), name="lr", set_levers=["learning_rate"], status="failed")
    e = chain.embed(c, lambda cells: informative(labels, cells))
    b = chain.batch_correct(e, set_levers=["theta"], status="partial")
    k = chain.cluster(b, set_levers=["resolution", "not_a_lever"])
    chain.choose({"cluster_call": k, "resolution": resolution, "reason": "test", "skipped": skipped or {}})
    return run


def test_groups_come_from_the_registry_plus_cell_overrides(tmp_path, stub):
    assert coverage.lever_groups(tmp_path) == [
        "segment.diameter", "segment.flow_threshold", "crop.margin", "crop.cell_overrides", "embed.learning_rate",
        "batch_correct.theta", "cluster.resolution"]


def test_completed_calls_count_failed_calls_do_not(tmp_path, stub):
    check = coverage.final_check(_run(tmp_path))
    assert check["tested"]["segment.diameter"] == ["S001"]
    assert check["tested"]["crop.cell_overrides"] == ["C001"]
    assert check["tested"]["batch_correct.theta"] == ["B001"]          # partial counts
    assert check["tested"]["cluster.resolution"] == ["K001"]
    assert check["tested"]["embed.learning_rate"] == []                # only the failed call set it
    assert check["missing"] == ["segment.flow_threshold", "crop.margin", "embed.learning_rate"]
    assert check["chain_problems"] == [] and not check["accepted"]
    text = coverage.describe(check)
    assert "segment.flow_threshold: UNTESTED" in text and "FINAL would NOT be accepted" in text


def test_skips_by_group_and_by_tool(tmp_path, stub):
    run = _run(tmp_path, skipped={"segment.flow_threshold": "diameter dominates", "embed": "no time to train",
                                  "crop.margin": "  "})
    check = coverage.final_check(run)
    assert check["skipped"] == {"segment.flow_threshold": "diameter dominates",
                                "embed.learning_rate": "no time to train"}
    assert check["missing"] == ["crop.margin"]                          # a blank reason is no reason
    choice_path = run / "workspace" / "submission" / "choice.json"
    choice = json.loads(choice_path.read_text(encoding="utf-8"))
    choice["skipped"]["crop.margin"] = "the default margin keeps neighbours out"
    choice_path.write_text(json.dumps(choice), encoding="utf-8")
    check = coverage.final_check(run)
    assert check["accepted"] and check["missing"] == []
    assert coverage.describe(check).endswith("FINAL would be accepted.")


def test_final_needs_a_valid_choice(tmp_path, stub):
    run = _run(tmp_path, skipped={"segment": "a", "crop": "b", "embed": "c"}, resolution=2.0)
    check = coverage.final_check(run)
    assert check["missing"] == [] and not check["accepted"]
    assert any("resolution 2.0" in p for p in check["chain_problems"])
    assert "the choice does not verify" in coverage.describe(check)
    (run / "workspace" / "submission" / "choice.json").unlink()
    check = coverage.final_check(run)
    assert check["chain_problems"] == ["no readable submission/choice.json"] and not check["accepted"]


def test_groups_from_the_real_tool_registry(tmp_path):
    """Against the workspace tools' own _common.py (CARD levers read from the repo)."""
    if not (coverage.TEMPLATE_TOOLS / "_common.py").exists():
        pytest.skip("the workspace tools are not there yet")
    groups = coverage.lever_groups(tmp_path)
    for group in ("segment.diameter", "segment.color_class_filter", "crop.margin_mode", "crop.cell_overrides",
                  "embed.adaptation_depth", "embed.plate_adversarial_weight", "batch_correct.theta",
                  "batch_correct.random_state", "cluster.resolution", "cluster.n_neighbors"):
        assert group in groups
    assert "embed.seed" not in groups and len(groups) == len(set(groups))


def test_the_first_end_line_counts(tmp_path, stub):
    """As in the chain: a second end line of a failed call does not make it count."""
    run = _run(tmp_path)
    with open(run / "logs" / "tool_calls.jsonl", "a", encoding="utf-8") as handle:
        handle.write(json.dumps({"event": "end", "call": "E001", "tool": "embed", "status": "done"}) + "\n")
    assert coverage.final_check(run)["tested"]["embed.learning_rate"] == []

"""The Codex arm's prompt protections (table aggregation, the inspected-JSON budget, the decision echo, the whole-prompt
guard, prompt_chars) for the Claude Code and Biomni arms: one StageRun prompt for all three arms, sized for 1310 images x 8
calls with 7 inspected 300k JSON files, and the small-sample (smoke) path of every stage with the backends mocked."""
from pathlib import Path

import pytest

from benchmarks.full_CPJUMP1.harness import cli
from shape.agent.arms import biomni as biomni_backend, claude as claude_backend, codex as codex_backend, protocol
from benchmarks.full_CPJUMP1.harness import codex_arm as agent
from benchmarks.full_CPJUMP1.harness.data import read_json

from .test_agent_arms import claude_env, fake_claude, fake_driver   # noqa: F401  (pytest fixtures)
from .test_codex import (CODEX_INPUT_LIMIT, FakeCodex, d, dataset_stage, gateway, image_stage,   # noqa: F401
                         inspect_run, old_group_prompt, table_run, livecell_record)

ARMS = ("codex", "claude", "biomni")


def limit_of(arm):
    return min(agent.PROMPT_CHAR_LIMIT, agent.ARM_PROMPT_LIMIT.get(arm, agent.PROMPT_CHAR_LIMIT))


def test_the_guard_limits_follow_each_arms_input_path():
    assert limit_of("codex") == limit_of("claude") == 900_000 < CODEX_INPUT_LIMIT
    assert limit_of("biomni") == 800_000
    assert agent.INSPECT_JSON_BUDGET + 400_000 + agent.DECISIONS_ECHO_BUDGET + 15_000 < limit_of("biomni")   # 15k: the measured facts (crop 13.7k)


@pytest.mark.parametrize("arm", ARMS)
def test_1310_images_8_calls_7_inspected_300k_json_files_stay_below_the_arms_limit_with_margin(tmp_path, arm):
    run, session, items = inspect_run(tmp_path, n_calls=8, facts_chars=30_000)
    run.arm = arm
    for image in run.images[::3]:                       # the flagged rows fill their budget: the largest table
        run.calls[image]["c3"] = livecell_record(3, image, "error")
    run.inspect(session, items, 0)
    session.decisions = [{"round": k, "action": "run", "request_json": "x" * 60_000, "reason": "r", "outcome": "executed",
                          "message": "m" * 30_000} for k in range(8)]
    prompt = run.prompt(session, 7, 32)
    inspected = sum(len(t) for t in session.texts)
    print(f"ARM={arm} PROMPT_CHARS={len(prompt)} inspected_json={inspected} limit={limit_of(arm)}")
    assert session.table_info["table_mode"] == "aggregated" and session.prompt_chars == len(prompt)
    assert inspected < agent.INSPECT_JSON_BUDGET + 7 * 200
    assert len(prompt) < limit_of(arm) - 150_000          # a wide margin for the facts, the CLI's own system prompt, steps
    assert len(prompt) < CODEX_INPUT_LIMIT - 400_000


@pytest.mark.parametrize("arm", ("claude", "biomni"))
def test_a_prompt_that_fits_is_byte_identical_to_the_codex_arms(monkeypatch, arm):
    monkeypatch.setattr(agent.time, "time", lambda: 1000.0)
    run = table_run(5, 3)
    run.arm = arm
    session = agent.Session(Path("x"), "group", run.images)
    assert run.prompt(session, 2, 32) == old_group_prompt(run, session, 2, 32)
    assert session.table_info["table_mode"] == "full"


@pytest.mark.parametrize("arm", ARMS)
def test_each_arms_guard_cuts_the_inspected_json_then_refuses_naming_the_part_sizes(tmp_path, monkeypatch, arm):
    run, session, items = inspect_run(tmp_path, n_images=20, n_calls=2, n_json=4, json_chars=100_000)
    run.arm = arm
    run.inspect(session, items, 0)
    uncut = len(run.prompt(session, 1, 32))
    monkeypatch.setattr(agent, "ARM_PROMPT_LIMIT", {arm: uncut - 60_000})
    cut = run.prompt(session, 1, 32)
    assert len(cut) <= uncut - 60_000 and cut.count("cut to fit the prompt limit") == 4
    monkeypatch.setattr(agent, "ARM_PROMPT_LIMIT", {arm: 5_000})
    with pytest.raises(RuntimeError, match=f"{agent.ARM_NAMES[arm]} prompt of .* exceeds the harness limit 5000") as error:
        run.prompt(session, 1, 32)
    assert "'facts':" in str(error.value) and "'table':" in str(error.value)


def test_claude_guard_raises_before_the_cli_runs(image_stage, fake_claude, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=2)
    config.update(arm="claude", model=claude_backend.MODEL)
    monkeypatch.setattr(agent, "PROMPT_CHAR_LIMIT", 2_000)
    with pytest.raises(RuntimeError, match="Claude Code prompt of .* exceeds the harness limit 2000"):
        agent.run_stage("segment", prep, run, config)
    assert fake_claude.calls == []


def test_biomni_guard_raises_before_the_driver_runs(image_stage, fake_driver, monkeypatch):
    prep, run, config, rows, waves = image_stage("segment", n=2)
    config.update(arm="biomni", model=biomni_backend.MODEL)
    monkeypatch.setattr(agent, "ARM_PROMPT_LIMIT", {"biomni": 2_000})
    with pytest.raises(RuntimeError, match="Biomni prompt of .* exceeds the harness limit 2000"):
        agent.run_stage("segment", prep, run, config)
    assert not list(run.rglob("biomni_requests.jsonl")) and not list(run.rglob("traces"))


# ---- the small-sample (smoke) path of every stage, the backends mocked --------------------------------------------

def mock_backend(monkeypatch, arm, fake):
    """The arm's real backend module with only its CLI / driver call replaced: route, setup and the usage fields stay."""
    module = agent.backend(arm)
    monkeypatch.setattr(module, "invoke", fake)
    return module


SMOKE = {"smoke": True, "smoke_policy": dict(cli.SMOKE_POLICY)}      # codex_rounds: 2


@pytest.mark.parametrize("arm", ("claude", "biomni"))
@pytest.mark.parametrize("stage", ("segment", "crop"))
def test_smoke_image_stages_reach_delivery_and_record_prompt_chars(image_stage, claude_env, monkeypatch, arm, stage):
    prep, run, config, rows, waves = image_stage(stage, n=2)
    config.update(arm=arm, model=cli.ARM_MODELS[arm], **SMOKE)
    params = {"diameter": 12} if stage == "segment" else {"margin_frac": 0.3}
    fake = FakeCodex({"group": [d("run", [{"images": "all", "params": params}]), d("finish", {"all": "c1"})]})
    mock_backend(monkeypatch, arm, fake)
    agent.run_stage(stage, prep, run, config)
    rounds = read_json(run / "agent" / stage / "ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["executed", "finished"]            # the smoke's two rounds
    assert [r["prompt_chars"] for r in rounds] == [len(p) for p in fake.prompts["group"]]
    assert all(r["table_mode"] == "full" for r in rounds)
    assert "This is the last round of this session: only finish is accepted." in fake.prompts["group"][1]
    assert fake.prompts["group"][0].startswith(f"You are the {agent.ARM_NAMES[arm]} arm of the CPJUMP1 benchmark")
    selected = read_json(run / "agent" / stage / "SELECTED.json")
    assert {v["call"] for v in selected["deliveries"].values()} == {"c1"}
    assert (run / ("segmentation.json" if stage == "segment" else "cells.json")).is_file()


@pytest.mark.parametrize("arm", ("claude", "biomni"))
@pytest.mark.parametrize("stage", ("embed", "batch"))
def test_smoke_dataset_stages_reach_delivery_and_record_prompt_chars(dataset_stage, claude_env, monkeypatch, arm, stage):
    run, config, workers = dataset_stage(stage)
    config.update(arm=arm, model=cli.ARM_MODELS[arm], **SMOKE)
    params = {"view_pairing": "same_well"} if stage == "embed" else {}
    fake = FakeCodex({"dataset": [d("run", [{"params": params}]), d("finish", {"all": "c1"})]})
    mock_backend(monkeypatch, arm, fake)
    agent.run_stage(stage, run, run, config)
    rounds = read_json(run / "agent" / stage / "ROUNDS.json")
    assert [r["outcome"] for r in rounds] == ["executed", "finished"]
    assert [r["prompt_chars"] for r in rounds] == [len(p) for p in fake.prompts["dataset"]]
    assert all("table_mode" not in r for r in rounds)                           # a dataset session has no call table
    assert read_json(run / "agent" / stage / "SELECTED.json")["deliveries"]["dataset"]["call"] == "c1"
    assert all(c["agent_arm"] == arm and c["arm"] == "codex" for _, c in workers)

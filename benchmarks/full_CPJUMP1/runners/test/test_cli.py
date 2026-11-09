from types import SimpleNamespace

import pytest

from benchmarks.full_CPJUMP1.harness import cli
from benchmarks.full_CPJUMP1.harness.data import read_json, write_json


@pytest.mark.parametrize("command", ["run", "preflight"])
@pytest.mark.parametrize("arm", ["control", "shape", "codex"])
def test_active_arms_default_to_exact_model(command, arm):
    argv = [command, "--arm", arm, "--expected-commit", "a" * 40,
            "--assets-root", "assets", "--out", "run"]
    if command == "run":
        argv += ["--prep", "prep"]
    assert cli.parser().parse_args(argv).model == "gpt-5.6-luna"
    with pytest.raises(SystemExit):
        cli.parser().parse_args(argv + ["--model", "gpt-5.6"])


def test_agent_arms_are_cli_choices_with_their_own_models():
    """Claude and biomni are active arms; an unknown arm is not."""
    base = ["preflight", "--expected-commit", "a" * 40, "--assets-root", "assets", "--out", "run"]
    assert cli.parser().parse_args(base + ["--arm", "claude", "--model", "claude-sonnet-5"]).model == "claude-sonnet-5"
    assert cli.parser().parse_args(base + ["--arm", "biomni"]).model == "gpt-5.6-luna"
    assert cli.ARM_MODELS == {"shape": "gpt-5.6-luna", "codex": "gpt-5.6-luna", "claude": "claude-sonnet-5",
                              "biomni": "gpt-5.6-luna"}
    for arm in ("react", "claude-code"):
        with pytest.raises(SystemExit):
            cli.parser().parse_args(base + ["--arm", arm])


@pytest.mark.parametrize("arm,model", [("claude", "gpt-5.6-luna"), ("biomni", "claude-sonnet-5"),
                                       ("codex", "claude-sonnet-5"), ("shape", "claude-sonnet-5")])
def test_preflight_checks_each_arms_model_before_anything_else(arm, model):
    args = SimpleNamespace(arm=arm, model=model, expected_commit="a" * 40)
    with pytest.raises(ValueError, match="exact model ID"):
        cli.preflight(args)


def test_compare_requires_codex_directory():
    argv = ["compare", "--prep", "prep", "--control", "control", "--shape", "shape",
            "--out", "comparison", "--expected-commit", "a" * 40]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(argv)
    args = cli.parser().parse_args(argv + ["--codex", "codex"])
    assert str(args.codex) == "codex" and args.claude is None and args.biomni is None
    args = cli.parser().parse_args(argv + ["--codex", "codex", "--claude", "claude", "--biomni", "biomni"])
    assert (str(args.claude), str(args.biomni)) == ("claude", "biomni")


def test_compare_passes_only_the_given_arms(tmp_path, monkeypatch):
    from benchmarks.full_CPJUMP1.harness import compare
    seen = []
    monkeypatch.setattr(compare, "compare", lambda prep, runs, out, expected, branch: seen.append(set(runs)))
    argv = ["compare", "--prep", str(tmp_path), "--control", "c", "--shape", "s", "--codex", "x",
            "--out", str(tmp_path / "out"), "--expected-commit", "a" * 40]
    cli.main(argv)
    cli.main(argv + ["--biomni", "b"])
    assert seen == [{"control", "shape", "codex"}, {"control", "shape", "codex", "biomni"}]


def test_verify_prepared_requires_complete_fixed_subset(tmp_path, monkeypatch):
    private = tmp_path / "private"
    private.mkdir()
    (private / "labels.json").write_text("labels", encoding="ascii")
    (private / "raw_hashes.json").write_text("hashes", encoding="ascii")
    record = {"images_sha256": "images",
              "labels_sha256": cli.digest(private / "labels.json"),
              "raw_hashes_sha256": cli.digest(private / "raw_hashes.json")}
    write_json(tmp_path / "PREPARE.json", record)
    monkeypatch.setattr(cli, "load_prepared", lambda prep, verify_pixels: [{}] * 1000)

    result = cli.verify_prepared(tmp_path, tmp_path / "receipt.json")

    assert result["status"] == "verified"
    assert result["views"] == 1000
    assert read_json(tmp_path / "receipt.json") == result

    monkeypatch.setattr(cli, "load_prepared", lambda prep, verify_pixels: [{}] * 999)
    with pytest.raises(ValueError, match="1000-view"):
        cli.verify_prepared(tmp_path, tmp_path / "invalid.json")


def test_codex_stage_dispatch_and_usage(tmp_path, monkeypatch):
    from benchmarks.full_CPJUMP1.harness import codex_arm
    monkeypatch.setattr(cli, "asset_environment", lambda root: {})
    monkeypatch.setattr(cli, "load_prepared", lambda root: [])
    config = {"arm": "codex", "model": "gpt-5.6-luna", "assets_root": str(tmp_path), "images": []}
    write_json(tmp_path / "CONFIG.json", config)
    calls = []
    selected = {"trial": "t000", "total_tokens": 12, "response_model_attested": False}

    def run_stage(stage, prep, run, cfg):
        calls.append((stage, cfg))
        write_json(run / "agent" / stage / "SELECTED.json", selected)

    monkeypatch.setattr(codex_arm, "run_stage", run_stage)
    cli.stage(SimpleNamespace(stage="embed", prep=tmp_path, run=tmp_path))
    assert calls == [("embed", config)]
    assert read_json(tmp_path / "embed_USAGE.json") == selected
    assert read_json(tmp_path / "embed_COMPLETE.json")["usage"] == selected


@pytest.mark.parametrize("arm", ["control", "shape", "claude"])
def test_native_trial_only_accepts_codex(tmp_path, arm):
    write_json(tmp_path / "CONFIG.json", {"arm": arm})
    with pytest.raises(ValueError, match="Codex"):
        cli.native_trial(SimpleNamespace(config=None, run=tmp_path, prep=tmp_path, stage="embed"))


@pytest.mark.parametrize("agent_arm,accepted", [("codex", True), ("claude", True), ("biomni", True),
                                                ("shape", False), ("control", False)])
def test_native_trial_takes_every_agent_arms_worker_config(tmp_path, monkeypatch, agent_arm, accepted):
    """An agent arm's worker CONFIG.json is arm "codex" (pipeline.py's agent-arm path) plus agent_arm."""
    monkeypatch.setattr(cli, "asset_environment", lambda root: (_ for _ in ()).throw(LookupError("assets")))
    write_json(tmp_path / "CONFIG.json", {"arm": "codex", "agent_arm": agent_arm, "assets_root": str(tmp_path)})
    with pytest.raises(LookupError if accepted else ValueError):
        cli.native_trial(SimpleNamespace(config=None, run=tmp_path, prep=tmp_path, stage="embed"))


@pytest.mark.parametrize("arm", ["claude", "biomni"])
def test_new_agent_arm_stage_dispatch_and_usage(tmp_path, monkeypatch, arm):
    from benchmarks.full_CPJUMP1.harness import codex_arm
    monkeypatch.setattr(cli, "asset_environment", lambda root: {})
    monkeypatch.setattr(cli, "load_prepared", lambda root: [])
    config = {"arm": arm, "model": cli.ARM_MODELS[arm], "assets_root": str(tmp_path), "images": []}
    write_json(tmp_path / "CONFIG.json", config)
    selected = {"total_tokens": 0}
    monkeypatch.setattr(codex_arm, "run_stage",
                        lambda stage, prep, run, cfg: write_json(run / "agent" / stage / "SELECTED.json", selected))
    with pytest.raises(RuntimeError, match=f"{arm} stage reported zero tokens"):
        cli.stage(SimpleNamespace(stage="segment", prep=tmp_path, run=tmp_path))
    selected["total_tokens"] = 7
    cli.stage(SimpleNamespace(stage="crop", prep=tmp_path, run=tmp_path))
    assert read_json(tmp_path / "crop_USAGE.json") == selected

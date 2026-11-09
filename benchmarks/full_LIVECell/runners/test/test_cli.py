"""The full_LIVECell CLI: arms, models, resources, dispatch and the shared run discipline under the LIVECell spec."""
from pathlib import Path
import sys
from types import SimpleNamespace

import pytest

from benchmarks.full_CPJUMP1.harness import cli as base
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
from benchmarks.full_LIVECell.harness import cli, data
from benchmarks.full_LIVECell.harness.spec import LIVECELL

COMMIT = "a" * 40
MODULE = "benchmarks.full_LIVECell.harness.cli"


def argv_run(tmp_path, *extra):
    return ["run", "--arm", "control", "--expected-commit", COMMIT, "--assets-root", str(tmp_path / "assets"),
            "--out", str(tmp_path / "run"), "--prep", str(tmp_path / "prep"), *extra]


@pytest.mark.parametrize("arm", ["control", "shape", "codex"])
def test_arms_are_this_rounds_with_the_exact_model(arm):
    args = cli.parser().parse_args(["preflight", "--arm", arm, "--expected-commit", COMMIT, "--assets-root", "a",
                                    "--out", "o"])
    assert args.model == "gpt-5.6-luna"
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["preflight", "--arm", arm, "--model", "claude-sonnet-5", "--expected-commit", COMMIT,
                                 "--assets-root", "a", "--out", "o"])
    for other in ("claude", "biomni", "react"):
        with pytest.raises(SystemExit):
            cli.parser().parse_args(["preflight", "--arm", other, "--expected-commit", COMMIT, "--assets-root", "a",
                                     "--out", "o"])


def test_run_flags_match_cpjump1_and_cores_default_to_the_hardware(tmp_path):
    args = cli.parser().parse_args(argv_run(tmp_path))
    assert (args.budget_hours, args.seed, args.cores, args.train_cells_per_site, args.inference_chunk, args.smoke) == (
        11, 0, None, 128, 2048, False)
    from shape.agent.verification.base import available_cores
    assert cli.resources(None) == {"cores": available_cores(), "available_cores": available_cores(),
                                   "source": "hardware"}
    assert cli.resources(1)["source"] == "flag"
    for bad in (0, available_cores() + 1):
        with pytest.raises(ValueError, match="cores"):
            cli.resources(bad)


def test_compare_needs_the_three_arms_and_delegates(tmp_path, monkeypatch):
    from benchmarks.full_LIVECell.harness import compare
    base_argv = ["compare", "--prep", str(tmp_path), "--control", "c", "--shape", "s", "--out", str(tmp_path / "o"),
                 "--expected-commit", COMMIT]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(base_argv)
    seen = []
    monkeypatch.setattr(compare, "compare", lambda prep, runs, out, expected, branch: seen.append(
        (set(runs), expected, branch)))
    cli.main(base_argv + ["--codex", "x"])
    cli.main(base_argv + ["--codex", "x", "--expected-branch", "main"])
    assert seen == [({"control", "shape", "codex"}, COMMIT, "e2e"), ({"control", "shape", "codex"}, COMMIT, "main")]


def test_data_commands_and_workers_dispatch(tmp_path, monkeypatch):
    calls = []
    monkeypatch.setattr(data, "select", lambda raw, manifest: calls.append(("select", raw.name, manifest.name)))
    monkeypatch.setattr(data, "prepare", lambda raw, out, manifest: calls.append(("prepare", raw.name, out.name,
                                                                                   manifest)))
    monkeypatch.setattr(base, "stage", lambda args: calls.append(("stage", args.stage)))
    monkeypatch.setattr(base, "native_trial", lambda args: calls.append(("native", args.stage, args.config)))
    cli.main(["select", "--raw", str(tmp_path / "raw"), "--manifest", str(tmp_path / "manifest.json")])
    cli.main(["prepare", "--raw", str(tmp_path / "raw"), "--out", str(tmp_path / "prep")])
    cli.main(["stage", "batch", "--prep", str(tmp_path), "--run", str(tmp_path)])
    cli.main(["native-trial", "crop", "--prep", str(tmp_path), "--run", str(tmp_path)])
    assert calls == [("select", "raw", "manifest.json"), ("prepare", "raw", "prep", None), ("stage", "batch"),
                     ("native", "crop", None)]
    with pytest.raises(SystemExit):
        cli.parser().parse_args(["native-trial", "cluster", "--prep", "p", "--run", "r"])


def rows(n=4, shape=(1, 520, 704), dtype="uint8"):
    return [{"image_id": f"img_{i:04d}", "well": f"unit_{i // 2:04d}", "plate": f"stage_{i % 2}",
             "usage": "train", "shape": list(shape), "dtype": dtype} for i in range(n)]


def test_prepared_facts_require_the_spec_shape_and_one_native_dtype(tmp_path):
    write_json(tmp_path / "PREPARE.json", {"image_shape": [520, 704], "channels": ["Phase"], "dtype": "uint8"})
    assert cli.prepared_facts(tmp_path, rows()) == {"shape": [1, 520, 704], "dtype": "uint8"}
    for bad in (rows(shape=(1, 704, 520)), rows(dtype="uint16"), rows(shape=(3, 520, 704))):
        with pytest.raises(ValueError, match="images differ"):
            cli.prepared_facts(tmp_path, bad)
    write_json(tmp_path / "PREPARE.json", {"image_shape": [1080, 1080], "channels": ["Phase"], "dtype": "uint8"})
    with pytest.raises(ValueError, match="LIVECell spec"):
        cli.prepared_facts(tmp_path, rows())


def test_stage_workers_load_the_livecell_rows(tmp_path, monkeypatch):
    """The shared stage worker reads the dataset from CONFIG.json and loads the LIVECell prepared rows."""
    from benchmarks.full_CPJUMP1.harness import pipeline
    loaded, ran = [], []
    monkeypatch.setattr(base, "asset_environment", lambda root: {})
    monkeypatch.setattr(base, "load_prepared", lambda prep: pytest.fail("CPJUMP1 rows for a LIVECell run"))
    monkeypatch.setattr(data, "load_prepared", lambda prep: loaded.append(prep) or rows())
    monkeypatch.setitem(pipeline.STAGES, "cluster", lambda prep, run, config, images: ran.append(len(images)))
    write_json(tmp_path / "CONFIG.json", {"arm": "control", "assets_root": str(tmp_path), "dataset": "LIVECell",
                                          "images": ["img_0000", "img_0002"]})
    base.stage(SimpleNamespace(stage="cluster", prep=tmp_path, run=tmp_path))
    assert loaded == [tmp_path] and ran == [2]


@pytest.fixture
def fake_run(tmp_path, monkeypatch):
    """base.run under the LIVECell spec with every child process faked: (prep, commands)."""
    prep = tmp_path / "prep"
    prep.mkdir()
    write_json(prep / "PREPARE.json", {"image_shape": [520, 704], "channels": ["Phase"], "dtype": "uint8"})
    images = rows(48)
    commands, scored, identities = [], [], []
    monkeypatch.setattr(data, "load_prepared", lambda prep, verify_pixels=False: images)
    monkeypatch.setattr(data, "smoke_rows", lambda prep, rows: rows[:24])
    monkeypatch.setattr(base, "load_prepared", lambda *a, **k: pytest.fail("CPJUMP1 rows for a LIVECell run"))
    monkeypatch.setattr(base, "code_identity", lambda expected, branch: identities.append(branch))
    from benchmarks.full_LIVECell.harness import score
    monkeypatch.setattr(score, "score", lambda prep, run: scored.append(run))

    def child(command, log, timeout, **kwargs):
        commands.append(command)
        if "preflight" in command:
            write_json(Path(command[command.index("--out") + 1]), {"commit": COMMIT, "gpu": "fake"})
            return
        run, stage = Path(command[command.index("--run") + 1]), command[command.index("stage") + 1]
        if stage == "segment":
            mask = run / "mask.tif"
            mask.write_bytes(b"mask")
            write_json(run / "segmentation.json", [{"mask": str(mask)}])
        if stage == "cluster":
            for name in base.ARTIFACTS:
                if not (run / name).exists():
                    (run / name).write_bytes(b"x")
            write_json(run / "SSL_SELECTION.json", {"checkpoint": None})
        write_json(run / f"{stage}_COMPLETE.json", {"status": "complete", "stage": stage})

    monkeypatch.setattr(base, "child", child)
    return prep, commands, scored, identities


@pytest.mark.parametrize("smoke", [False, True])
def test_run_records_the_livecell_dataset_and_marker(tmp_path, fake_run, capsys, smoke):
    prep, commands, scored, identities = fake_run
    cli.main(argv_run(tmp_path, "--cores", "1", *(["--smoke"] if smoke else [])))
    run = tmp_path / "run"
    config = read_json(run / "CONFIG.json")
    assert config["dataset"] == "LIVECell" and config["cores"] == 1 and config["resources"]["source"] == "flag"
    assert config["raw"] == {"shape": [1, 520, 704], "dtype": "uint8"}
    assert len(config["images"]) == (24 if smoke else 48) and config["smoke"] is smoke
    assert all(c[:3] == [sys.executable, "-m", MODULE] for c in commands)
    assert [c[4] for c in commands if c[3] == "stage"] == list(base.STAGES)
    meta, delivery = read_json(run / "RUN_META.json"), read_json(run / "DELIVERY.json")
    assert meta["status"] == "complete" and meta["config"] == config and delivery["smoke"] is smoke
    assert all(digest(run / name) == sha for name, sha in delivery["artifacts"].items())
    marker = "SMOKE_COMPLETE" if smoke else "LIVECELL_COMPLETE"
    assert (run / marker).read_text(encoding="ascii").strip() == COMMIT
    assert not (run / ("LIVECELL_COMPLETE" if smoke else "SMOKE_COMPLETE")).exists()
    assert scored == ([] if smoke else [run])
    out = capsys.readouterr().out
    assert f"{marker} arm=control" in out and "RESOURCES cores=1" in out
    if smoke:
        assert "LIVECELL_COMPLETE" not in out


def test_run_passes_the_expected_branch_to_preflight_and_the_final_identity_check(tmp_path, fake_run):
    prep, commands, scored, identities = fake_run
    cli.main(argv_run(tmp_path, "--cores", "1", "--expected-branch", "main"))
    preflight = next(c for c in commands if c[3] == "preflight")
    assert preflight[preflight.index("--expected-branch") + 1] == "main" and identities == ["main"]
    args = cli.parser().parse_args(["preflight", "--arm", "shape", "--expected-commit", COMMIT, "--assets-root", "a",
                                    "--out", "o", "--expected-branch", "main"])
    assert args.expected_branch == "main"
    assert cli.parser().parse_args(argv_run(tmp_path)).expected_branch == "e2e"


@pytest.mark.parametrize("branch,accepted", [("main", True), ("e2e", False)])
def test_code_identity_checks_the_expected_branch_and_its_origin(monkeypatch, branch, accepted):
    """A checkout of `main` at the pinned commit passes only with --expected-branch main; the record names it."""
    answers = {("rev-parse", "HEAD"): COMMIT, ("branch", "--show-current"): "main", ("status", "--porcelain"): "",
               ("rev-parse", "origin/main"): COMMIT, ("rev-parse", "origin/e2e"): "b" * 40}
    monkeypatch.setattr(base, "git", lambda *args: answers[args])
    if accepted:
        assert base.code_identity(COMMIT, branch)["branch"] == "main"
    else:
        with pytest.raises(RuntimeError, match="expected e2e"):
            base.code_identity(COMMIT, branch)
    answers[("rev-parse", "origin/main")] = "c" * 40
    with pytest.raises(RuntimeError, match="origin/main does not match" if accepted else "expected e2e"):
        base.code_identity(COMMIT, branch)


def test_a_failed_stage_is_recorded_and_raised(tmp_path, fake_run, monkeypatch):
    prep, commands, scored, identities = fake_run
    original = base.child

    def failing(command, log, timeout, **kwargs):
        if "embed" in command:
            raise RuntimeError("stage failed (exit 1)")
        original(command, log, timeout, **kwargs)

    monkeypatch.setattr(base, "child", failing)
    with pytest.raises(RuntimeError, match="stage failed"):
        cli.main(argv_run(tmp_path, "--cores", "1"))
    meta = read_json(tmp_path / "run" / "RUN_META.json")
    assert meta["status"] == "failed" and meta["current_stage"] == "embed"
    assert not (tmp_path / "run" / "LIVECELL_COMPLETE").exists() and not scored
    with pytest.raises(FileExistsError):
        cli.main(argv_run(tmp_path, "--cores", "1"))


def test_verify_prepared_checks_the_ground_truth_and_the_smoke_list(tmp_path, monkeypatch):
    (tmp_path / "private" / "gt").mkdir(parents=True)
    gt = tmp_path / "private" / "gt" / "img_0000.npz"
    gt.write_bytes(b"gt")
    for name, text in (("labels.json", "labels"), ("raw_hashes.json", "hashes")):
        (tmp_path / "private" / name).write_text(text, encoding="ascii")
    write_json(tmp_path / "private" / "gt_index.json", [{"image_id": "img_0000", "file": "gt/img_0000.npz",
                                                         "sha256": digest(gt)}])
    record = {"image_shape": [520, 704], "channels": ["Phase"], "dtype": "uint8", "images_sha256": "i",
              "smoke_sha256": "s", "gt_index_sha256": digest(tmp_path / "private" / "gt_index.json"),
              "labels_sha256": digest(tmp_path / "private" / "labels.json"),
              "raw_hashes_sha256": digest(tmp_path / "private" / "raw_hashes.json")}
    write_json(tmp_path / "PREPARE.json", record)
    monkeypatch.setattr(data, "load_prepared", lambda prep, verify_pixels=False: rows(LIVECELL.n_images))
    monkeypatch.setattr(data, "smoke_rows", lambda prep, rows: rows[:24])
    result = cli.verify_prepared(tmp_path, tmp_path / "receipt.json")
    assert result["views"] == 1310 and result["dtype"] == "uint8" and read_json(tmp_path / "receipt.json") == result
    gt.write_bytes(b"changed")
    with pytest.raises(ValueError, match="ground truth changed"):
        cli.verify_prepared(tmp_path, tmp_path / "receipt2.json")
    gt.write_bytes(b"gt")
    monkeypatch.setattr(data, "load_prepared", lambda prep, verify_pixels=False: rows(1309))
    with pytest.raises(ValueError, match="1310-view"):
        cli.verify_prepared(tmp_path, tmp_path / "receipt3.json")


@pytest.mark.parametrize("arm,model", [("codex", "gpt-5.6-luna"), ("claude", "claude-sonnet-5"),
                                       ("biomni", "gpt-5.6-luna")])
def test_the_agent_arms_run_with_their_models_and_compare_takes_optional_claude_and_biomni(arm, model, tmp_path):
    parsed = cli.parser().parse_args(["run", "--arm", arm, "--model", model, "--expected-commit", "a" * 40,
                                      "--assets-root", str(tmp_path), "--out", str(tmp_path / "o"),
                                      "--prep", str(tmp_path)])
    assert (parsed.arm, parsed.model) == (arm, model)
    compare = cli.parser().parse_args(["compare", "--prep", str(tmp_path), "--control", "c", "--shape", "s", "--codex", "x",
                                       "--claude", "cl", "--out", "o", "--expected-commit", "a" * 40])
    assert compare.claude == Path("cl") and compare.biomni is None
    assert cli.ARMS == ("control", "shape", "codex", "claude", "biomni")

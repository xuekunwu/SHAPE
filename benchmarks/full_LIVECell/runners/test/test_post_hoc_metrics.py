"""run_post_hoc_metrics.sh and process_readouts.py of full_LIVECell: script contracts and one run on a tiny experiment dir."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.full_LIVECell.runners import process_readouts

RUNNERS = Path(__file__).resolve().parents[1]
REPO = RUNNERS.parents[2]
SCRIPT = RUNNERS / "run_post_hoc_metrics.sh"
STAGES = ("segment", "crop", "embed", "batch", "cluster")


def bash():
    if Path("C:/Program Files/Git/bin/bash.exe").is_file():
        return "C:/Program Files/Git/bin/bash.exe"
    found = shutil.which("bash")
    if not found:
        pytest.skip("bash unavailable")
    return found


def tiny_experiment(root: Path, arms=("control", "shape")) -> Path:
    for name in arms:
        arm = root / name
        arm.mkdir(parents=True)
        files = {"RUN_META.json": {"status": "running", "arm": name, "provenance": {"commit": "a" * 40}},
                 "CONFIG.json": {"arm": name, "seed": 0}}
        for i, stage in enumerate(STAGES[:2]):
            files[f"{stage}_COMPLETE.json"] = {"status": "complete", "stage": stage, "seconds": 10.0 + i, "usage": {}}
        for fname, body in files.items():
            (arm / fname).write_text(json.dumps(body), encoding="utf-8", newline="\n")
    return root


def run(*args):
    import os
    env = {**os.environ, "PYTHON": Path(sys.executable).as_posix(), "FULL_LIVECELL_REPO": REPO.as_posix()}
    return subprocess.run([bash(), SCRIPT.as_posix(), *args], capture_output=True, text=True, env=env, cwd=str(REPO))


def test_script_syntax_line_endings_and_shell_options():
    subprocess.run([bash(), "-n", SCRIPT.as_posix()], check=True, capture_output=True)
    raw = SCRIPT.read_bytes()
    assert b"\r" not in raw
    text = raw.decode()
    assert "set -eo pipefail" in text and not re.search(r"set -\w*u", text)
    assert "pgrep" not in text and "pkill" not in text
    assert "extra_readouts.py" in text and "process_readouts.py" in text and "LIVECELL_COMPLETE" in text


def test_usage_and_bad_arguments():
    out = run("--help")
    assert out.returncode == 0 and "--run EXPERIMENT_DIR" in out.stdout
    assert run().returncode == 2
    assert run("--run", str(REPO), "--bogus").returncode == 2


def test_wrapper_defaults_are_the_dataset_ones():
    assert process_readouts.MARKERS == ("LIVECELL_COMPLETE", "RERUN_COMPLETE")
    assert process_readouts.STAGES == STAGES and process_readouts.DEFAULT_OUT == "process_readouts.json"


def test_incomplete_and_absent_arms_are_skipped_and_said(tmp_path):
    exp = tiny_experiment(tmp_path / "exp")                 # control and shape without LIVECELL_COMPLETE, no codex
    done = run("--run", exp.as_posix())
    assert done.returncode == 0, done.stderr + done.stdout
    assert "extra_readouts_control: skipped" in done.stdout and "incomplete" in done.stdout
    assert "extra_readouts_codex: skipped (run dir" in done.stdout and "absent" in done.stdout
    lines = (exp / "analysis" / "COMMANDS.txt").read_text(encoding="utf-8").splitlines()
    skips = [ln for ln in lines if "skipped:" in ln]
    assert len(skips) == 3
    steps = [ln for ln in lines if ln.startswith("# ") and "skipped" not in ln]
    assert len(steps) == 1 and "step=process_readouts" in steps[0]
    assert re.match(r"# \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ commit=([0-9a-f]{40}|unknown) step=", steps[0])
    command = lines[lines.index(steps[0]) + 1]
    assert command.startswith("PYTHONPATH=") and "process_readouts.py" in command and "--md" in command
    for name in ("process_readouts.json", "process_readouts.csv", "process_readouts.md"):
        assert (exp / "analysis" / name).is_file()


def test_complete_arm_without_prep_stops_with_a_message(tmp_path):
    exp = tiny_experiment(tmp_path / "exp", arms=("control",))
    (exp / "control" / "LIVECELL_COMPLETE").write_text("x\n", encoding="ascii", newline="\n")
    (exp / "control" / "RESULT.json").write_text("{}", encoding="utf-8", newline="\n")
    done = run("--run", exp.as_posix(), "--prep", (tmp_path / "no_prep").as_posix(), "--arms", "control")
    assert done.returncode == 2 and "extra_readouts.py needs the prepared tree" in done.stderr

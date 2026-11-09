"""run_post_hoc_metrics.sh and process_readouts.py of full_CPJUMP1: script contracts and one run on a tiny experiment dir."""
import json
import re
import shutil
import subprocess
import sys
from pathlib import Path

import pytest

from benchmarks.full_CPJUMP1.runners import process_readouts

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


def tiny_experiment(root: Path) -> Path:
    arm = root / "control"
    arm.mkdir(parents=True)
    files = {"RUN_META.json": {"status": "complete", "arm": "control", "provenance": {"commit": "a" * 40}},
             "CONFIG.json": {"arm": "control", "seed": 0}}
    for i, stage in enumerate(STAGES):
        files[f"{stage}_COMPLETE.json"] = {"status": "complete", "stage": stage, "seconds": 10.0 + i, "usage": {}}
    for name, body in files.items():
        (arm / name).write_text(json.dumps(body), encoding="utf-8", newline="\n")
    (arm / "CPJUMP1_COMPLETE").write_text("a" * 40 + "\n", encoding="ascii", newline="\n")
    return root


def run(*args, env_extra=None):
    import os
    env = {**os.environ, "PYTHON": Path(sys.executable).as_posix(), "FULL_CPJUMP1_REPO": REPO.as_posix(),
           **(env_extra or {})}
    return subprocess.run([bash(), SCRIPT.as_posix(), *args], capture_output=True, text=True, env=env, cwd=str(REPO))


def test_script_syntax_line_endings_and_shell_options():
    subprocess.run([bash(), "-n", SCRIPT.as_posix()], check=True, capture_output=True)
    raw = SCRIPT.read_bytes()
    assert b"\r" not in raw
    text = raw.decode()
    assert "set -eo pipefail" in text and not re.search(r"set -\w*u", text)
    assert "pgrep" not in text and "pkill" not in text
    for call in ("export_run_records.py", "pairwise_deltas.py", "process_readouts.py"):
        assert call in text


def test_usage_and_bad_arguments():
    out = run("--help")
    assert out.returncode == 0 and "--run EXPERIMENT_DIR" in out.stdout
    assert run().returncode == 2
    assert run("--run", str(REPO), "--bogus").returncode == 2


def test_wrapper_defaults_are_the_dataset_ones():
    assert process_readouts.MARKERS == ("CPJUMP1_COMPLETE", "RERUN_COMPLETE")
    assert process_readouts.STAGES == STAGES and process_readouts.DEFAULT_OUT == "process_readouts.json"


def test_full_run_records_every_step(tmp_path):
    exp = tiny_experiment(tmp_path / "exp")
    out = tmp_path / "results" / "exp"
    done = run("--run", exp.as_posix(), "--out", out.as_posix())
    assert done.returncode == 0, done.stderr + done.stdout
    lines = (out / "analysis" / "COMMANDS.txt").read_text(encoding="utf-8").splitlines()
    headers = [ln for ln in lines if ln.startswith("# ")]
    assert [re.search(r"step=(\S+)", h).group(1) for h in headers] == [
        "export_run_records", "export_run_records_check", "pairwise_deltas", "process_readouts"]
    assert "skipped: no --pair given" in headers[2]
    for h in headers:
        assert re.match(r"# \d{4}-\d\d-\d\dT\d\d:\d\d:\d\dZ commit=([0-9a-f]{40}|unknown) step=", h)
    commands = [ln for ln in lines if not ln.startswith("# ")]
    assert "export_run_records.py" in commands[0] and "--run" in commands[0] and "/exp --out " in commands[0]
    assert "--check" in commands[1] and "process_readouts.py" in commands[2] and "--md" in commands[2]
    assert all(ln.startswith("PYTHONPATH=") for ln in commands)
    assert (out / "analysis" / "process_readouts.json").is_file() and (out / "analysis" / "process_readouts.csv").is_file()
    assert (out / "analysis" / "process_readouts.md").is_file()
    from benchmarks.full_CPJUMP1 import export_run_records as ex
    assert ex.check(out) == []                       # the exporter's --check ignores analysis/


def test_skip_export_and_failed_pair_stops(tmp_path):
    exp = tiny_experiment(tmp_path / "exp")
    out = tmp_path / "res"
    out.mkdir()
    done = run("--run", exp.as_posix(), "--out", out.as_posix(), "--skip-export")
    assert done.returncode == 0, done.stderr
    text = (out / "analysis" / "COMMANDS.txt").read_text(encoding="utf-8")
    assert "export_run_records skipped: --skip-export" in text and "step=process_readouts" in text
    bad = run("--run", exp.as_posix(), "--out", out.as_posix(), "--skip-export", "--pair", "control:nosuch")
    assert bad.returncode != 0

import importlib.util
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location("runtime_environment", Path(__file__).parents[1] / "runtime_environment.py")
runtime = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runtime)


def test_exact_environment_rejects_missing_changed_and_extra(tmp_path):
    lock = tmp_path / "requirements.lock.txt"
    lock.write_text("numpy==1.26.4\ntorch==2.8.0+cu128\n")
    exact = {"numpy": "1.26.4", "torch": "2.8.0+cu128", "shape": "0.1.0"}
    assert runtime.check_lock(lock, exact)["package_count"] == 2
    for installed in ({"numpy": "1.26.4"}, {**exact, "torch": "2.8.0"}, {**exact, "unknown": "1"}):
        with pytest.raises(RuntimeError, match="ENVIRONMENT_DRIFT"):
            runtime.check_lock(lock, installed)


def test_lock_rejects_unpinned_and_duplicate_requirements(tmp_path):
    lock = tmp_path / "requirements.lock.txt"
    for text in ("numpy>=1", "torch==2\ntorch==2\n", "-e /another/source"):
        lock.write_text(text)
        with pytest.raises(ValueError):
            runtime.check_lock(lock, {})


def test_chip_info_is_rendered_from_actual_record():
    evidence = {"hardware": {"gpus": [{"name": "NVIDIA RTX PRO 6000 Blackwell Server Edition",
        "memory_mib": 97887, "driver": "test-driver"}], "cpu_model": "CPU model", "available_cpus": 24,
        "logical_cpus": 24, "ram_gib": 214.3}, "torch": {"version": "2.8.0+cu128", "cuda_runtime": "12.8"},
        "python": {"version": "3.12.3"}, "os": {"system": "Linux", "release": "test-kernel"},
        "lock": {"sha256": "abc"}, "external_tools": {"codex": "0.162.0"}}
    table = runtime.markdown(evidence)
    assert "RTX PRO 6000 Blackwell" in table and "97887 MiB" in table
    assert "24 / logical 24" in table and "12.8" in table and "test-driver" in table

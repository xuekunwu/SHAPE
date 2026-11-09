"""Public, credential-free runtime evidence shared by benchmark runners."""
from __future__ import annotations

import csv
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import platform
import re
import subprocess
import sys


def packages():
    found = {}
    for dist in importlib.metadata.distributions():
        name = re.sub(r"[-_.]+", "-", dist.metadata["Name"]).lower()
        if name in found and found[name] != dist.version:
            raise RuntimeError(f"multiple installed versions of {name}")
        found[name] = dist.version
    return dict(sorted(found.items()))


def check_lock(path, installed=None):
    """Exact package set, excluding the separately source-bound SHAPE distribution."""
    path = Path(path)
    required = {}
    for line in path.read_text(encoding="utf-8-sig").splitlines():
        line = line.strip()
        if not line or line.startswith("#"):
            continue
        match = re.fullmatch(r"([A-Za-z0-9_.-]+)==([^\s;]+)", line)
        if not match:
            raise ValueError(f"lock must contain exact package pins: {line}")
        name = re.sub(r"[-_.]+", "-", match[1]).lower()
        if name in required:
            raise ValueError(f"duplicate lock entry: {name}")
        required[name] = match[2]
    actual = dict(packages() if installed is None else installed)
    actual.pop("shape", None)
    wrong = {k: {"expected": v, "installed": actual.get(k)} for k, v in required.items() if actual.get(k) != v}
    extra = sorted(set(actual) - set(required))
    if wrong or extra:
        raise RuntimeError(f"ENVIRONMENT_DRIFT: changed={wrong}, unexpected={extra}")
    return {"file": path.name, "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "package_count": len(required), "verified": True}


def command(argv):
    try:
        return subprocess.check_output(argv, text=True, stderr=subprocess.DEVNULL, timeout=15).strip()
    except (OSError, subprocess.SubprocessError):
        return None


def capture(lock=None):
    """Capture scientific runtime facts; never enumerate environment variables or auth files."""
    installed = packages()
    cpu = platform.processor() or platform.machine()
    memory = None
    proc_cpu = Path("/proc/cpuinfo")
    if proc_cpu.is_file():
        match = re.search(r"^model name\s*:\s*(.+)$", proc_cpu.read_text(), re.M)
        if match:
            cpu = match[1]
    meminfo = Path("/proc/meminfo")
    if meminfo.is_file():
        match = re.search(r"^MemTotal:\s*(\d+) kB", meminfo.read_text(), re.M)
        if match:
            memory = round(int(match[1]) / 1024 ** 2, 3)
    gpu_text = command(["nvidia-smi", "--query-gpu=name,memory.total,driver_version,uuid", "--format=csv,noheader,nounits"])
    gpus = []
    for row in csv.reader((gpu_text or "").splitlines(), skipinitialspace=True):
        if len(row) == 4:
            gpus.append(dict(name=row[0], memory_mib=int(row[1]), driver=row[2], uuid=row[3]))
    torch_info = {}
    try:
        import torch
        torch_info = {"version": str(torch.__version__), "cuda_runtime": torch.version.cuda,
                      "cudnn": torch.backends.cudnn.version(), "cuda_available": torch.cuda.is_available(),
                      "compute_capability": [list(torch.cuda.get_device_capability(i)) for i in range(torch.cuda.device_count())]}
    except ImportError:
        pass
    return {"schema": "shape-runtime-v1", "python": {"version": platform.python_version(), "executable": sys.executable},
            "os": {"system": platform.system(), "release": platform.release(), "machine": platform.machine()},
            "hardware": {"cpu_model": cpu, "logical_cpus": os.cpu_count(),
                         "available_cpus": len(os.sched_getaffinity(0)) if hasattr(os, "sched_getaffinity") else os.cpu_count(),
                         "ram_gib": memory, "gpus": gpus},
            "torch": torch_info, "packages": installed,
            "lock": check_lock(lock, installed) if lock else None,
            "external_tools": {"codex": command(["codex", "--version"]), "uv": command(["uv", "--version"])}}


def markdown(record):
    hw, torch = record["hardware"], record["torch"]
    chips = "; ".join(f"{g['name']} ({g['memory_mib']} MiB; driver {g['driver']})" for g in hw["gpus"]) or "not detected"
    lock = record.get("lock") or {}
    rows = [("GPU / chip", f"{len(hw['gpus'])} × {chips}"),
            ("CPU", f"{hw['cpu_model']}; available {hw['available_cpus']} / logical {hw['logical_cpus']}"),
            ("RAM (GiB)", hw['ram_gib']), ("Python", record['python']['version']),
            ("PyTorch / CUDA runtime / cuDNN", f"{torch.get('version')} / {torch.get('cuda_runtime')} / {torch.get('cudnn')}"),
            ("OS / kernel", f"{record['os']['system']} / {record['os']['release']}"),
            ("Dependency lock SHA256", lock.get('sha256', 'not enforced')),
            ("CODEX CLI", record['external_tools'].get('codex'))]
    return "\n".join(["| Runtime | Recorded value |", "|---|---|", *[f"| {k} | {v} |" for k,v in rows],
                      "", "All installed package versions: [ENVIRONMENT.json](ENVIRONMENT.json)."])


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--lock", type=Path)
    parser.add_argument("--out", required=True, type=Path)
    args = parser.parse_args()
    args.out.write_text(json.dumps(capture(args.lock), indent=2) + "\n", encoding="utf-8")

"""GPU and node facts via nvidia-smi and Slurm variables (never a CUDA context in the harness
process: on an exclusive-process GPU that would lock the tool calls out)."""
from __future__ import annotations

import os
import re
import subprocess

SLOT_GB = 16          # one GPU tool call (Cellpose-SAM or a DINOv3 fine-tune) plus headroom
MAX_SLOTS = 4


def _query(fields: str) -> list[str]:
    try:
        out = subprocess.run(["nvidia-smi", f"--query-gpu={fields}", "--format=csv,noheader,nounits"],
                             capture_output=True, text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return []
    line = out.stdout.strip().splitlines()[0] if out.returncode == 0 and out.stdout.strip() else ""
    return [part.strip() for part in line.split(",")] if line else []


def gpu_name() -> str:
    fields = _query("name")
    return fields[0] if fields else "cpu"


def slug(name: str) -> str:
    """A GPU name as a path part: 'NVIDIA H100 80GB HBM3' -> 'nvidia-h100-80gb-hbm3'; '' -> 'cpu'."""
    return re.sub(r"[^a-z0-9]+", "-", name.lower()).strip("-") or "cpu"


def gpu_slug() -> str:
    """e.g. 'nvidia-h100-80gb-hbm3', 'nvidia-geforce-rtx-2080-ti', 'cpu'."""
    return slug(gpu_name())


def gpu_slots() -> int:
    """How many tool calls may share the GPU: 1 in exclusive-process mode, else by memory."""
    fields = _query("memory.total,compute_mode")
    if len(fields) < 2:
        return 1
    memory_mib, mode = fields
    if "exclusive" in mode.lower():
        return 1
    try:
        return max(1, min(MAX_SLOTS, int(float(memory_mib) / 1024 // SLOT_GB)))
    except ValueError:
        return 1


def _mem_gb(cpus: int) -> float | None:
    """Memory of the allocation: Slurm's per-node or per-cpu limit, else the node's RAM."""
    per_node, per_cpu = os.environ.get("SLURM_MEM_PER_NODE"), os.environ.get("SLURM_MEM_PER_CPU")
    if per_node and per_node.isdigit():
        return round(int(per_node) / 1024, 1)
    if per_cpu and per_cpu.isdigit():
        return round(int(per_cpu) * cpus / 1024, 1)
    try:
        return round(os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES") / 1024 ** 3, 1)
    except (AttributeError, ValueError, OSError):
        return None


def node_facts() -> dict:
    """{gpu, gpu_mem_gb, gpu_compute_mode, gpu_exclusive, cpus, mem_gb}: what TASK.md tells the
    agent about its node (first GPU; compute mode e.g. Default / Exclusive_Process, "" without one)."""
    fields = _query("name,memory.total,compute_mode")
    gpu, mem, mode = (fields[0], fields[1], fields[2]) if len(fields) >= 3 else ("none", None, "")
    try:
        gpu_mem_gb = round(float(mem) / 1024, 1) if mem is not None else None
    except ValueError:
        gpu_mem_gb = None
    slurm_cpus = os.environ.get("SLURM_CPUS_ON_NODE", "")
    cpus = int(slurm_cpus) if slurm_cpus.isdigit() else (os.cpu_count() or 1)
    return {"gpu": gpu, "gpu_mem_gb": gpu_mem_gb, "gpu_compute_mode": mode,
            "gpu_exclusive": "exclusive" in mode.lower(), "cpus": cpus, "mem_gb": _mem_gb(cpus)}

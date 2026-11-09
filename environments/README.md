# Pinned benchmark runtimes

One directory per runtime the benchmarks and case studies are pinned to. Each holds
`requirements.lock.txt` (every package at an exact version) and `PROFILE.json` (the Python,
platform, CUDA and tool versions, the lock's sha256, and how the runtime was checked).

| Profile | For |
|---|---|
| `benchmark/linux-py312-cu128` | the default: Linux x86_64, Python 3.12, CUDA 12.8, torch 2.8 |
| `benchmark/linux-py312-cu130-h200` | an H200 node: CUDA 13.0, torch 2.14, cellpose 4.2 |

A run does not install from these files; it records what was actually installed and compares it
with the lock, so a run whose environment drifted is visible in its own records
(`shape/utils/runtime_environment.py`: `packages()`, `check_lock()`). The case-study runners read
the cu128 lock by default and `SHAPE_BENCHMARK_LOCK` selects another one:

```bash
export SHAPE_BENCHMARK_LOCK=environments/benchmark/linux-py312-cu130-h200/requirements.lock.txt
```

To install one of them instead of resolving dependencies yourself:

```bash
python -m pip install -r environments/benchmark/linux-py312-cu128/requirements.lock.txt
python -m pip install -e . --no-deps
```

The lock pins the CPU/CUDA build of torch that profile was built with; on another CUDA version
install torch from the index that matches your driver first.

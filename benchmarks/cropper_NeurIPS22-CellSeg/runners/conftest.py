"""Import the benchmark's modules by name: runners/ (the arms, the config, the legacy runners) and the
top level (run_benchmark.py), from this hyphenated directory."""

from pathlib import Path
import sys

_RUNNERS = Path(__file__).resolve().parent
sys.path[:0] = [str(_RUNNERS), str(_RUNNERS.parent)]

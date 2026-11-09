#!/usr/bin/env python3
"""Process readouts (L4 protocol friction P1-P4, L5 efficiency E1-E3, E6) of a full_CPJUMP1 experiment directory or of
its curated results folder. A thin entry over benchmarks/_shared/agentic_process_readouts.py (all logic lives there).

    python process_readouts.py --run <experiment dir or results folder> [--out OUT.json] [--csv OUT.csv] [--md]

Defaults: completion markers CPJUMP1_COMPLETE and RERUN_COMPLETE; stages segment, crop, embed, batch, cluster; output
<run>/analysis/process_readouts.json (a curated results folder's `--check` ignores analysis/). Needs the repo on
PYTHONPATH (benchmarks._shared)."""
from __future__ import annotations

from benchmarks._shared import agentic_process_readouts as shared

MARKERS = ("CPJUMP1_COMPLETE", "RERUN_COMPLETE")
STAGES = ("segment", "crop", "embed", "batch", "cluster")
DEFAULT_OUT = "process_readouts.json"


def main(argv=None) -> int:
    return shared.main(argv, markers=MARKERS, stages=STAGES, default_out=DEFAULT_OUT,
                       description=__doc__.split("\n")[0])


if __name__ == "__main__":
    raise SystemExit(main())

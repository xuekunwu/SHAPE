"""Benchmark adapter abstraction. An adapter turns a benchmark into three things the harness
needs: (1) build an ISOLATED workdir the agent runs in (inputs + TASK.md, never SHAPE code),
(2) the task prompt, (3) a scorer over the delivered artifacts using OUR metrics, plus (4) a
feedback string for the outer loop. Add a new benchmark by subclassing this.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path


@dataclass
class ScoreResult:
    summary: dict            # macro metrics (the headline the loop optimizes)
    per_item: dict           # per-image/-crop detail
    n_delivered: int
    n_expected: int


class BenchmarkAdapter:
    id = "benchmark"

    def __init__(self, data_root: Path):
        self.data_root = Path(data_root)

    def select_items(self, limit: int | None, seed: int = 0) -> list[str]:
        """Return the item ids (image stems) to run."""
        raise NotImplementedError

    def prepare_workdir(self, workdir: Path, items: list[str]) -> None:
        """Populate ``workdir`` with inputs + TASK.md + an empty deliver/ tree (no SHAPE code)."""
        raise NotImplementedError

    def task_prompt(self, workdir: Path, items: list[str]) -> str:
        """The first-turn instruction handed to the agent."""
        raise NotImplementedError

    def score(self, workdir: Path, items: list[str]) -> ScoreResult:
        """Grade the delivered artifacts in ``workdir/deliver`` with our own metric code."""
        raise NotImplementedError

    def feedback_prompt(self, score: ScoreResult, history: list | None = None) -> str:
        """The follow-up instruction for the next loop iteration, given the best score so
        far and the full per-iteration ``history`` (each entry has a ``summary`` dict) so
        the prompt can show the trajectory and discourage stopping while still improving."""
        raise NotImplementedError

    def materialize(self, deliver_dir: Path, items: list[str], out_dir: Path,
                    score: ScoreResult) -> None:
        """Write the SHAPE-parallel per-item deliver channel into ``out_dir/images/<id>/``:
        a per-item ``metrics.json`` and a ``deliverable_seg.png`` visualization. Default no-op."""
        return None

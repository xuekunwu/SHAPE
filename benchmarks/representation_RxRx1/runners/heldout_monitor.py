"""Held-out monitor of SHAPE runs with early stopping. Never read by any agent.

For every finished call of every shape run under $BENCH, embed a fixed subsample with the call's
resume_best state (the scorer's own embedding path, `score.embed`, embed_only) and write the well mAP of the
held-out sample (10 % of held-out wells) and of the val wells to <run>/MONITOR.jsonl. Loops until
$BENCH/monitor.stop exists.

  python -m benchmarks.representation_RxRx1.runners.heldout_monitor [--once] [--heldout-wells 0.10]
"""
import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

BENCH = Path(os.environ["BENCH"])        # the benchmark results root holding the run dirs and cache/
STOP = BENCH / "monitor.stop"
SEED = 20261019


def find_resume(obj):
    """The first resume_best path inside a (nested) tool-call record."""
    if isinstance(obj, dict):
        for k, v in obj.items():
            if k in ("resume_best", "call_resume_best") and isinstance(v, str) and v.endswith(".pt"):
                return v
            found = find_resume(v)
            if found:
                return found
    elif isinstance(obj, list):
        for v in obj:
            found = find_resume(v)
            if found:
                return found
    return None


def finished_calls(run: Path) -> list[tuple[str, str]]:
    path = run / "logs" / "tool_calls.jsonl"
    if not path.exists():
        return []
    out = []
    for line in path.read_text(encoding="utf-8").splitlines():
        try:
            rec = json.loads(line)
        except json.JSONDecodeError:
            continue
        if rec.get("event") == "start" or rec.get("status") == "crashed":
            continue
        resume = find_resume(rec)
        call = rec.get("call")
        if call and resume and Path(resume).exists():
            out.append((call, resume))
    return out


class Cell:
    def __init__(self, cell_type: str, heldout_wells: float):
        from benchmarks.representation_RxRx1.runners import prepare, score

        self.root = BENCH / "cache" / f"{cell_type}_full_v5"
        self.summary, cells = prepare.load(self.root)
        sirna = score.read_labels(self.root, cells, self.summary)
        rng = np.random.default_rng(SEED)
        held = sorted(set(cells.well[cells.split == "heldout"]))
        pick = set(rng.choice(held, size=int(round(heldout_wells * len(held))), replace=False).tolist())
        mask = ((cells.split == "heldout") & cells.well.isin(pick)) | (cells.split == "val")
        self.cells = cells[mask.to_numpy()].reset_index(drop=True)
        self.sirna = sirna[mask.to_numpy()]
        self.keep = score.scored_cells(self.sirna)

    def scores(self, vectors) -> dict:
        from benchmarks.representation_RxRx1.runners import score

        out = {}
        split, well = self.cells.split.to_numpy(), self.cells.well.to_numpy()
        for name, splits in (("heldout", ("heldout",)), ("val", ("val",))):
            m = np.isin(split, splits) & self.keep
            entry = score.well_map(vectors[m], self.sirna[m], well[m])
            out[name] = None if entry is None else entry["value"]
        return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--once", action="store_true")
    ap.add_argument("--heldout-wells", type=float, default=0.10)
    args = ap.parse_args()
    from benchmarks.representation_RxRx1.runners import prepare

    os.environ.update(prepare.check_assets(BENCH / "cache"))
    from benchmarks.representation_RxRx1.runners import score

    cells: dict[str, Cell] = {}
    while True:
        did = 0
        for run in sorted(p for p in BENCH.iterdir() if p.is_dir() and p.name.split("_")[0] == "shape"):
            cell_type = run.name.split("_")[1]
            out = run / "MONITOR.jsonl"
            done = {json.loads(l)["call"] for l in out.read_text().splitlines()} if out.exists() else set()
            for call, resume in finished_calls(run):
                if call in done:
                    continue
                if cell_type not in cells:
                    cells[cell_type] = Cell(cell_type, args.heldout_wells)
                c = cells[cell_type]
                arm_dir = run / "monitor_embed" / call
                started = time.time()
                vectors = score.embed(c.summary, c.cells, arm_dir, {"init_checkpoint": resume, "embed_only": True})
                rec = {"call": call, "resume_best": resume, **c.scores(vectors),
                       "n_cells": len(c.cells), "seconds": round(time.time() - started, 1), "time": time.time()}
                with out.open("a", encoding="utf-8") as f:
                    f.write(json.dumps(rec) + "\n")
                print(run.name, json.dumps(rec), flush=True)
                # the h5ad of the monitor embedding is large and never read again
                for h5 in arm_dir.rglob("*.h5ad"):
                    h5.unlink()
                did += 1
        if args.once or STOP.exists():
            return
        if not did:
            time.sleep(120)


if __name__ == "__main__":
    sys.exit(main())

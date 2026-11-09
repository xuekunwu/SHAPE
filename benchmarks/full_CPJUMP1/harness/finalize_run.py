"""Finalize a STAGED run into a scored delivery.

The monolithic ``cli run`` ends with ``deliver()``: it binds every artifact by hash in
DELIVERY.json and scores the run. The staged runner (run_staged.sh) runs the stages separately --
often across several commits and on different nodes/days -- so it never reaches that tail, and
``deliver()``'s ``code_identity`` (HEAD == one pinned commit, clean tree, origin match) cannot
hold for a staged dev run by construction. This writes the SAME DELIVERY.json artifact set as
``deliver()`` (so ``score()``'s own hash + prepare checks still bind the exact bytes) but WITHOUT
the submission-grade commit provenance, then scores. The result is a research mAP, not a
submission; the provenance block records that it was finalized from staged artifacts.

    python -m benchmarks.full_CPJUMP1.harness.finalize_run --prep PREP --run RUN [--control CONTROL_RUN]

--control: a finalized+scored control run of the same preparation, for the delta/SE/verdict block
(omit for the control arm itself).
"""
from __future__ import annotations

import argparse
from pathlib import Path

from .cli import ARTIFACTS, RERUN_RECORD
from .data import digest, read_json, within, write_json
from .pipeline import CPJUMP1


def gather_artifacts(out: Path) -> set[Path]:
    """Exactly deliver()'s artifact set (cli.py): the fixed ARTIFACTS that exist, the stage markers and
    usages, every per-image search/cohort record, the agent-arm round traces, the delivered masks, and the
    SSL checkpoint. Missing optional files are skipped (a control has no agent/ tree, say)."""
    artifacts = {out / name for name in ARTIFACTS if (out / name).exists()}
    artifacts.update(out.glob("*_COMPLETE.json"))
    artifacts.update(out.glob("*_USAGE.json"))
    artifacts.update(out.rglob("*SEARCH.json"))
    artifacts.update(out.glob("*_COHORT.json"))
    artifacts.update(out.glob("SEARCH_WARNINGS.jsonl"))
    artifacts.update(out.glob(RERUN_RECORD))
    agent = out / "agent"
    if agent.exists():
        for pattern in ("TRIALS.json", "CALLS.json", "ROUNDS.json", "SELECTED.json",
                        "round_*.jsonl", "round_*.json", "failed_round_*.jsonl", "failed_round_*.json"):
            artifacts.update(agent.rglob(pattern))
    seg = out / "segmentation.json"
    if seg.exists():
        artifacts.update(Path(row["mask"]) for row in read_json(seg))
    sel = out / "SSL_SELECTION.json"
    if sel.exists():
        checkpoint = read_json(sel).get("checkpoint")
        if checkpoint and Path(checkpoint).exists():
            artifacts.add(Path(checkpoint))
    return artifacts


def main(argv=None) -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--prep", type=Path, required=True)
    ap.add_argument("--run", type=Path, required=True)
    ap.add_argument("--control", type=Path, default=None,
                    help="a finalized control run for the verdict block (omit for the control arm)")
    a = ap.parse_args(argv)
    prep, out = a.prep.resolve(), a.run.resolve()

    config = read_json(out / "CONFIG.json")
    if config.get("smoke"):
        raise SystemExit("refusing to finalize a smoke run (score needs a full delivery)")
    need = ("segment", "crop", "embed", "batch", "cluster")
    missing = [s for s in need if not (out / f"{s}_COMPLETE.json").exists()]
    if missing:
        raise SystemExit(f"cannot finalize: missing stage markers {missing}")

    artifacts = gather_artifacts(out)
    # Only hash-bind artifacts UNDER the run dir. A run that REUSES another run's stage outputs (e.g.
    # shape_perimage reuses shape's segmentation masks, which live under results/shape) references
    # them by absolute path in segmentation.json; those external files are already hash-verified in
    # their own run's delivery, and score() reads cells.json / embedding.npy / profiles (not the
    # masks), so leave the out-of-tree paths out of this manifest instead of crashing on relative_to.
    external = sorted(p for p in artifacts if out not in p.resolve().parents and p.resolve() != out)
    delivery = {
        "status": "complete", "arm": config["arm"], "smoke": False,
        "provenance": {"finalized_from": "staged", "note": "staged dev run; submission-grade "
                       "code_identity NOT enforced (stages ran separately, possibly across commits)",
                       **({"reused_external_artifacts": [str(p) for p in external]} if external else {})},
        "prepare_sha256": digest(prep / "PREPARE.json"),
        "artifacts": {p.relative_to(out).as_posix(): digest(p)
                      for p in sorted(artifacts) if out in p.resolve().parents},
    }
    write_json(out / "DELIVERY.json", delivery)
    print(f"[finalize] DELIVERY.json written: arm={config['arm']} artifacts={len(delivery['artifacts'])}")

    import importlib
    verdicts = {} if a.control is None else {"control": a.control.resolve()}
    importlib.import_module(CPJUMP1.score).score(prep, out, **verdicts)
    print(f"[finalize] scored -> {out}/RESULT.json  ({out}/RESULT.md)")


if __name__ == "__main__":
    main()

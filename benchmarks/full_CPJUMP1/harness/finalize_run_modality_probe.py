"""Finalize a staged ModalityProbe/ModalityProbe2 run into a scored delivery (same DELIVERY.json artifact
set and semantics as finalize_run.py, which is hardcoded to CPJUMP1.score -- this dispatches the scorer
through pipeline.dataset(config) instead, so it works for either ModalityProbe variant's own scorer).

    python -m benchmarks.full_CPJUMP1.harness.finalize_run_modality_probe --prep PREP --run RUN [--control CONTROL_RUN]

--control: a finalized+scored control run of the same preparation, for the delta/SE/verdict block (omit for
the control arm itself).
"""
from __future__ import annotations

import argparse
import importlib
from pathlib import Path

from .data import digest, read_json, write_json
from .finalize_run import gather_artifacts
from .pipeline import dataset


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

    spec = dataset(config)
    artifacts = gather_artifacts(out)
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
    print(f"[finalize] DELIVERY.json written: arm={config['arm']} dataset={spec.name} "
          f"artifacts={len(delivery['artifacts'])}")

    verdicts = {} if a.control is None else {"control": a.control.resolve()}
    importlib.import_module(spec.score).score(prep, out, **verdicts)
    print(f"[finalize] scored -> {out}/RESULT.json  ({out}/RESULT.md)")


if __name__ == "__main__":
    main()

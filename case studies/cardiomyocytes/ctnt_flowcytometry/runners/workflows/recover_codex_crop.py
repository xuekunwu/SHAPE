"""Restart Codex crop after an oversized inspect input; retain verified segmentation."""

# Internal worker CLI; the supported public entrypoint is run.py.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    __package__ = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows'

import argparse
from datetime import datetime, timezone
import os
from pathlib import Path
import shutil
import sys
from types import SimpleNamespace

from ..dataset import preparation as data
from . import experiment_runner as e2e
from ..dataset import sampling as pilot
from ..runtime.source_snapshot import verify_source


def run(args):
    from benchmarks.full_CPJUMP1.harness.cli import child
    from benchmarks.full_CPJUMP1.harness import pipeline
    verify_source(args.expected_source)
    target = args.suite/"codex"
    config = data.read_json(target/"CONFIG.json")
    meta = data.read_json(target/"RUN_META.json")
    if (meta["status"] != "failed" or meta["current_stage"] != "crop" or config["arm"] != "codex" or
            config["source_sha256"] != args.previous_source or (target/"RESUME.json").exists()):
        raise ValueError("expected unrecovered Codex crop failure at previous source")
    if config["deadline"] is not None or config["budget_hours"] is not None:
        raise ValueError("restart must preserve unlimited arm policy")
    if any((target/f"{stage}_COMPLETE.json").exists() for stage in ("crop", "embed", "batch", "predict")):
        raise ValueError("unexpected downstream completion")
    traces = list((target/"agent"/"crop").rglob("*.stderr"))
    if not any("input_too_large" in p.read_text(errors="replace") for p in traces):
        raise ValueError("restart only accepts the diagnosed oversized input failure")
    if data.read_json(target/"segment_COMPLETE.json")["status"] != "complete":
        raise ValueError("segmentation incomplete")
    if data.digest(args.prep/"PREPARE.json") != meta["prepare_sha256"]:
        raise ValueError("preparation changed")
    sampling = pilot.verify(args.prep, target/"PILOT.json")
    if set(sampling["images"]) != set(config["images"]) or data.digest(target/"PILOT.json") != config["pilot_sha256"]:
        raise ValueError("pilot changed")
    masks = data.read_json(target/"segmentation.json")
    if len(masks) != len(config["images"]) or {r["image_id"] for r in masks} != set(config["images"]):
        raise ValueError("segmentation does not cover the pilot")
    for mask in masks:
        path = Path(mask["mask"]).resolve()
        if not path.is_relative_to(target.resolve()) or data.digest(path) != mask["sha256"]:
            raise ValueError("segmentation mask changed")
    print(f"CODEX_SEGMENTATION_VERIFIED images={len(masks)}", flush=True)
    if args.audit_only:
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = target/"attempts"/f"input_too_large_{stamp}"
    backup.mkdir(parents=True)
    hashes = {}
    for name in ("CONFIG.json", "RUN_META.json", "PREFLIGHT.json", "segment_COMPLETE.json", "segmentation.json"):
        shutil.copy2(target/name, backup/name)
        hashes[name] = data.digest(backup/name)
    shutil.copy2(target/"logs"/"crop.log", backup/"crop.log")
    shutil.copy2(args.suite/"codex.log", backup/"codex.log")
    preflight = backup/"RESTART_PREFLIGHT.json"
    child([sys.executable, "-m", e2e.MODULE, "preflight", "--arm", "codex", "--model", config["model"],
           "--assets-root", config["assets_root"], "--expected-source", args.expected_source,
           "--out", str(preflight)], backup/"preflight.log", 1200)
    # Preserve the entire unsuccessful crop attempt, including its default calls and CLI trace.
    for relative in ("agent/crop", "crop"):
        src = target/relative
        if src.exists():
            dest = backup/relative
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
    if (target/"crop_USAGE.json").exists():
        shutil.copy2(target/"crop_USAGE.json", backup/"crop_USAGE.json")
    lineage = {"policy":"codex-crop-input-bound-v1", "source_sha256":args.expected_source,
               "previous_source_sha256":args.previous_source, "backup":str(backup),
               "original_artifact_sha256":hashes, "started_utc":stamp, "reused_stage":"segment",
               "note":"Crop restarts from defaults after input-bound fix; prior failed crop time/tokens are retained in backup, excluded from new crop_COMPLETE."}
    data.write_json(target/"RESUME.json", lineage)
    config.update(source_sha256=args.expected_source, recovery_policy=lineage["policy"])
    data.write_json(target/"CONFIG.json", config)
    shutil.copy2(preflight, target/"PREFLIGHT.json")
    meta.update(status="running", config=config, provenance=data.read_json(preflight), resume=lineage)
    meta.pop("error", None)
    try:
        from .stage_sequence import run_stages
        run_stages(('crop','embed','batch','predict'),e2e.MODULE,args.prep,target,config,meta)
        e2e.freeze(SimpleNamespace(out=target, prep=args.prep, arm="codex", expected_source=args.expected_source,
                                   smoke=False, pilot_manifest=target/"PILOT.json"), meta)
        meta["current_stage"] = "readouts"
        data.write_json(target/"RUN_META.json", meta)
        from ..evaluation.stage_quality import collect
        from ..evaluation.heldout_evaluation import score
        collect(args.prep, target)
        meta.update(status="complete", result=score(args.prep, target))
        data.write_json(target/"RUN_META.json", meta)
        (target/"CTNT_FLOW_PILOT_COMPLETE").write_text(args.expected_source+'\n')
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        data.write_json(target/"RUN_META.json", meta)
        raise
    from ..evaluation.arm_comparison import compare
    compare(args.prep, {a:args.suite/a for a in ("control", "shape", "codex")},
            args.suite/"comparison_resumed", args.expected_source, allow_resumed=True)
    (args.suite/"PILOT_SUITE_COMPLETE").write_text(args.expected_source+'\n')
    print("PILOT_SUITE_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--suite", type=Path, required=True)
    parser.add_argument("--prep", type=Path, required=True)
    parser.add_argument("--expected-source", required=True)
    parser.add_argument("--previous-source", required=True)
    parser.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    args.suite, args.prep = args.suite.resolve(), args.prep.resolve()
    run(args)


if __name__=='__main__':main()

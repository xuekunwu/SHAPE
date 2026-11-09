"""Recover a timed-out SHAPE crop stage, then finish the pilot without an arm time limit.

Reused image outputs are checked against raw pixels and selected search geometry.
Original timeout evidence is retained. The completed control is not rewritten.
"""
from __future__ import annotations

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
import re
import shutil
import sys
import time
from types import SimpleNamespace

from ..dataset import preparation as data
from . import experiment_runner as e2e
from ..dataset import sampling as pilot
from ..runtime.source_snapshot import verify_source

MODULE = "case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows.recover_shape_crop"


def inside(run, path):
    path = Path(path).resolve()
    if not path.is_relative_to(run.resolve()) or not path.is_file():
        raise ValueError(f"missing artifact or outside run: {path}")
    return path


def recover_image(prep, run, config, row, mask):
    """Read and validate a finished crop. Never rewrite its image/search/raw files."""
    import numpy as np
    import tifffile
    from benchmarks.full_CPJUMP1.harness import pipeline

    ident = row["image_id"]
    folder = run/"crop"/ident
    record = data.read_json(folder/"SEARCH.json")
    pipeline.assert_search_clean(record)
    result = data.read_json(folder/"result.json")
    metadata = inside(run, result["cell_crops_metadata_path"])
    crops = data.read_json(metadata)["cell_metadata"]
    boxes = {c["cell_id"]: c.get("crop_bbox") for c in crops}
    if record.get("boxes") and boxes != record["boxes"]:
        raise ValueError(f"recovered boxes differ from selected search: {ident}")
    rejected = set(record["rejected_tool_cells"])
    if rejected - set(boxes):
        raise ValueError(f"unknown rejected cells: {ident}")
    mask_path = inside(run, mask["mask"])
    if data.digest(mask_path) != mask["sha256"]:
        raise ValueError(f"changed segmentation: {ident}")
    spec = pipeline.dataset(config)
    labels = pipeline.mask_labels(tifffile.imread(mask_path), crops)
    raw = pipeline.raw_stack(data.within(prep/"public", row["file"]), spec)
    kept, excluded = [], []
    for index, crop in enumerate(crops):
        if crop["cell_id"] in rejected:
            excluded.append(crop["cell_id"])
            continue
        path = inside(run, folder/"raw_crops"/f"cell_{index:06d}.ome.tif")
        display = inside(run, crop["crop_path"])
        box = crop["crop_bbox"]
        if len(box) != 4 or any(int(v) != v for v in box):
            raise ValueError(f"invalid recovered box: {ident}")
        y0, x0, y1, x1 = map(int, box)
        if not (0 <= y0 < y1 <= raw.shape[1] and 0 <= x0 < x1 <= raw.shape[2]) or y1-y0 != x1-x0:
            raise ValueError(f"out-of-range recovered box: {ident}")
        pixels = tifffile.imread(path)
        expected = raw[:, y0:y1, x0:x1]
        if pixels.dtype != raw.dtype or pixels.shape not in (expected.shape, expected.shape[1:]) or not np.array_equal(
                pixels.reshape(expected.shape), expected):
            raise ValueError(f"recovered raw pixels differ from source: {ident}/{index}")
        kept.append({"cell_id": crop["cell_id"], "crop_path": str(path), "sha256": data.digest(path),
                     "area": crop["area"], "display_crop_path": str(display), "crop_bbox": box,
                     **{k: row[k] for k in ("image_id", "plate", "well", "usage")},
                     "mask_label": labels[crop["cell_id"]]})
    if not kept:
        raise ValueError(f"empty recovered image: {ident}")
    return kept, excluded


def audit(prep, run, previous_source):
    """Fail closed before any run files are changed."""
    meta = data.read_json(run/"RUN_META.json")
    config = data.read_json(run/"CONFIG.json")
    if (meta["status"] != "failed" or meta["current_stage"] != "crop" or
            "TimeoutExpired" not in meta.get("error", "") or config["arm"] != "shape" or
            config["source_sha256"] != previous_source or (run/"RESUME.json").exists()):
        raise ValueError("expected an unrecovered SHAPE crop timeout at the specified source")
    if data.read_json(run/"segment_COMPLETE.json")["status"] != "complete":
        raise ValueError("segmentation is not complete")
    if any((run/f"{s}_COMPLETE.json").exists() for s in ("crop", "embed", "batch", "predict")):
        raise ValueError("unexpected downstream stage artifacts")
    if meta["prepare_sha256"] != data.digest(prep/"PREPARE.json"):
        raise ValueError("prepared dataset identity changed")
    sampling = pilot.verify(prep, run/"PILOT.json")
    if set(config["images"]) != set(sampling["images"]) or data.digest(run/"PILOT.json") != config["pilot_sha256"]:
        raise ValueError("pilot identity changed")
    rows = data.load_prepared(prep, verify_pixels=True)
    rows = [r for r in rows if r["image_id"] in set(config["images"])]
    masks = {r["image_id"]: r for r in data.read_json(run/"segmentation.json")}
    if set(masks) != set(config["images"]):
        raise ValueError("incomplete segmentation")
    for mask in masks.values():
        if data.digest(inside(run, mask["mask"])) != mask["sha256"]:
            raise ValueError("segmentation digest mismatch")
    finished = set(re.findall(r"^CROP (\S+) retained=", (run/"logs"/"crop.log").read_text(), re.M))
    reusable, cells, excluded, hashes = [], [], [], {}
    for row in rows:
        ident = row["image_id"]
        if ident not in finished:
            continue
        kept, rejected = recover_image(prep, run, config, row, masks[ident])
        reusable.append(ident)
        cells.extend(kept)
        excluded.extend(rejected)
        for filename in ("SEARCH.json", "result.json"):
            path = run/"crop"/ident/filename
            hashes[path.relative_to(run).as_posix()] = data.digest(path)
    missing = [r["image_id"] for r in rows if r["image_id"] not in reusable]
    if not reusable or not missing:
        raise ValueError("recovery needs both completed and interrupted images")
    return meta, config, rows, {"reused_images": reusable, "rerun_images": missing,
                               "reused_cells": cells, "reused_excluded_cells": excluded,
                               "reused_record_hashes": hashes}


def crop_worker(prep, run):
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from shape.engine import usage

    config = data.read_json(run/"CONFIG.json")
    verify_source(config["source_sha256"])
    asset_environment(Path(config["assets_root"]))
    protocol = config["reasoning"]["protocol"]
    pipeline.check_stage_env("shape", "crop", protocol)
    recovered = data.read_json(run/"crop_RECOVERY.json")
    for relative, digest in recovered["reused_record_hashes"].items():
        if data.digest(data.within(run, relative)) != digest:
            raise ValueError("recovered search record changed")
    for cell in recovered["reused_cells"]:
        if data.digest(inside(run, cell["crop_path"])) != cell["sha256"]:
            raise ValueError("recovered crop changed after audit")
    for ident in recovered["reused_images"]:
        pipeline.record_search(run, "crop", ident, data.read_json(run/"crop"/ident/"SEARCH.json"))
    rows = [r for r in data.load_prepared(prep) if r["image_id"] in set(recovered["rerun_images"])]
    started = time.time()
    try:
        pipeline.cropping(prep, run, config, rows)
    finally:
        data.write_json(run/"crop_RESUME_USAGE.json", usage.snapshot())
    if usage.total_tokens() <= 0:
        raise RuntimeError("resumed SHAPE search has no observed positive token usage")
    cells = recovered["reused_cells"] + data.read_json(run/"cells.json")
    order = {ident: i for i, ident in enumerate(config["images"])}
    cells.sort(key=lambda c: (order[c["image_id"]], c["cell_id"]))
    if set(c["image_id"] for c in cells) != set(order) or len({c["cell_id"] for c in cells}) != len(cells):
        raise ValueError("recovered cell table has missing images or duplicate cells")
    data.write_json(run/"cells.json", cells)
    data.write_json(run/"excluded_cells.json", recovered["reused_excluded_cells"] +
                    data.read_json(run/"excluded_cells.json"))
    tokens = {**usage.snapshot(), "usage_complete": False,
              "coverage": "resume workers only; killed original workers did not persist aggregate usage",
              "original_attempt_tokens": None}
    data.write_json(run/"crop_USAGE.json", tokens)
    elapsed = time.time()-started
    data.write_json(run/"crop_COMPLETE.json", {"status": "complete", "stage": "crop",
                    "seconds": recovered["original_crop_seconds"]+elapsed, "resume_seconds": elapsed,
                    "original_attempt_seconds": recovered["original_crop_seconds"],
                    "usage": tokens, "reasoning": pipeline.stage_reasoning("shape", "crop", protocol)})


def resume(args):
    from benchmarks.full_CPJUMP1.harness.cli import child
    from benchmarks.full_CPJUMP1.harness import pipeline

    identity = verify_source(args.expected_source)
    run = args.suite/"shape"
    if (args.suite/"codex").exists():
        raise ValueError("Codex output already exists; do not overwrite it")
    meta, config, rows, recovery = audit(args.prep, run, args.previous_source)
    print(f'RECOVERY_VERIFIED reuse={len(recovery["reused_images"])} rerun={recovery["rerun_images"]}', flush=True)
    if args.audit_only:
        return
    stamp = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')
    backup = run/"attempts"/f"timeout_{stamp}"
    backup.mkdir(parents=True)
    original = {}
    for name in ("CONFIG.json", "RUN_META.json", "PREFLIGHT.json", "segment_COMPLETE.json", "segmentation.json"):
        shutil.copy2(run/name, backup/name)
        original[name] = data.digest(backup/name)
    shutil.copy2(run/"logs"/"crop.log", backup/"crop.log")
    shutil.copy2(args.suite/"shape.log", backup/"shape.log")
    preflight = backup/"RESUME_PREFLIGHT.json"
    child([sys.executable, "-m", e2e.MODULE, "preflight", "--arm", "shape", "--model", config["model"],
           "--assets-root", config["assets_root"], "--expected-source", args.expected_source,
           "--out", str(preflight)], backup/"preflight.log", 1200)
    # Archive only interrupted image directories; the 111 verified outputs remain byte-for-byte intact.
    for ident in recovery["rerun_images"]:
        src = run/"crop"/ident
        if src.exists():
            dest = backup/"partial_crop"/ident
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
        for src in (run/"logs"/"crop").glob(ident+".*"):
            dest = backup/"partial_logs"/src.name
            dest.parent.mkdir(parents=True, exist_ok=True)
            src.rename(dest)
    original_seconds = re.search(r"timed out after ([\d.]+) seconds", meta["error"])
    if original_seconds is None:
        raise ValueError("missing original crop elapsed time")
    recovery["original_crop_seconds"] = float(original_seconds[1])
    data.write_json(run/"crop_RECOVERY.json", recovery)
    lineage = {"policy": "resume-crop-no-walltime-v1", "previous_source_sha256": args.previous_source,
               "source_sha256": args.expected_source, "source_identity": identity, "started_utc": stamp,
               "backup": str(backup), "original_artifact_sha256": original,
               "reused_images": recovery["reused_images"], "rerun_images": recovery["rerun_images"],
               "budget_hours": None, "original_budget_hours": config["budget_hours"],
               "note": "Historical segment/crop decisions retained; uninterrupted-run/resource equality is not claimed."}
    data.write_json(run/"RESUME.json", lineage)
    config.update(source_sha256=args.expected_source, deadline=None, budget_hours=None,
                  recovery_policy=lineage["policy"])
    data.write_json(run/"CONFIG.json", config)
    shutil.copy2(preflight, run/"PREFLIGHT.json")
    meta.update(status="running", config=config, provenance=data.read_json(preflight), resume=lineage)
    meta.pop("error", None)
    try:
        from .stage_sequence import run_stages
        def recovery_command(stage):
            if stage == 'crop':
                return [sys.executable,'-m',MODULE,'crop-worker','--prep',str(args.prep),'--run',str(run)]
            return None
        run_stages(('crop','embed','batch','predict'),e2e.MODULE,args.prep,run,config,meta,
                   command_for=recovery_command)
        e2e.freeze(SimpleNamespace(out=run, prep=args.prep, arm="shape", expected_source=args.expected_source,
                                   smoke=False, pilot_manifest=run/"PILOT.json"), meta)
        meta["current_stage"] = "readouts"
        data.write_json(run/"RUN_META.json", meta)
        from ..evaluation.stage_quality import collect
        from ..evaluation.heldout_evaluation import score
        collect(args.prep, run)
        result = score(args.prep, run)
        meta.update(status="complete", result=result)
        data.write_json(run/"RUN_META.json", meta)
        (run/"CTNT_FLOW_PILOT_COMPLETE").write_text(args.expected_source+'\n')
        print("SHAPE_RESUME_COMPLETE", flush=True)
    except BaseException as error:
        meta.update(status="failed", error=f"{type(error).__name__}: {error}")
        data.write_json(run/"RUN_META.json", meta)
        raise
    # Same pilot/model/rounds/epochs; Codex has no arm wall-clock cutoff either.
    child([sys.executable, "-m", e2e.MODULE, "run", "--prep", str(args.prep), "--out", str(args.suite/"codex"),
           "--arm", "codex", "--model", config["model"], "--assets-root", config["assets_root"],
           "--expected-source", args.expected_source, "--pilot-manifest", str(run/"PILOT.json"),
           "--cores", str(config["cores"]), "--seed", str(config["seed"]), "--no-time-limit"],
          args.suite/"codex.log", None)
    from ..evaluation.arm_comparison import compare
    compare(args.prep, {a: args.suite/a for a in ("control", "shape", "codex")},
            args.suite/"comparison_resumed", args.expected_source, allow_resumed=True)
    (args.suite/"PILOT_SUITE_COMPLETE").write_text(args.expected_source+'\n')
    print("PILOT_SUITE_COMPLETE", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest="command", required=True)
    worker = commands.add_parser("crop-worker")
    worker.add_argument("--run", type=Path, required=True)
    worker.add_argument("--prep", type=Path, required=True)
    runner = commands.add_parser("resume-pilot")
    runner.add_argument("--suite", type=Path, required=True)
    runner.add_argument("--prep", type=Path, required=True)
    runner.add_argument("--expected-source", required=True)
    runner.add_argument("--previous-source", required=True)
    runner.add_argument("--audit-only", action="store_true")
    args = parser.parse_args()
    for key, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, key, value.resolve())
    if args.command == "crop-worker":
        crop_worker(args.prep, args.run)
    else:
        resume(args)


if __name__ == '__main__':
    main()

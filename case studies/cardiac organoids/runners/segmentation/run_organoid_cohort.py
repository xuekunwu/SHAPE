#!/usr/bin/env python3
"""Fixed-manifest CO cohort evaluation, with selective per-image VRR and full audits."""
from __future__ import annotations
import argparse
from dataclasses import asdict
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import threading
import time

from common import (MODEL, REPO, configure, dump, read_manifest, protect_output,
                    sha256, source_identity, versions, cellseg)
from audit import llm_audit, tool_audit, safe
from run_organoid_segmentation import evaluate_pair

TOOL = "Organoid_Segmenter_Tool"


def main():
    if "--bounded" in sys.argv:
        sys.argv.remove("--bounded")
        from run_organoid_bounded import main as bounded_main
        return bounded_main()
    cli = argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest", required=True, type=Path)
    cli.add_argument("--run-dir", required=True, type=Path)
    cli.add_argument("--model-path", required=True, type=Path)
    cli.add_argument("--expected-commit", required=True)
    cli.add_argument("--image-workers", type=int, default=4)
    cli.add_argument("--cache-seed", type=int, default=0)
    cli.add_argument("--disable-native-cache", action="store_true")
    cli.add_argument("--cohort-min-group", type=int, default=10,
                     help="Production default 10; set 5 only for the five-image technical smoke")
    args = cli.parse_args()
    if args.image_workers < 1 or args.cohort_min_group < 3:
        raise ValueError("Positive image workers and cohort support >=3 required")
    if not os.environ.get("SLURM_JOB_ID"):
        raise RuntimeError("Run this evaluation in a Slurm GPU allocation")
    head = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=REPO, text=True).strip()
    dirty = subprocess.check_output(["git", "status", "--porcelain"], cwd=REPO, text=True).strip()
    if head != args.expected_commit or dirty:
        raise RuntimeError("Evaluation requires the clean expected source commit")
    manifest = read_manifest(args.manifest)
    pairs = manifest["pairs"]
    weights = args.model_path.resolve()
    if not weights.is_file():
        raise ValueError("An exact local CO checkpoint file is required")
    output = args.run_dir.resolve()
    protect_output(output, pairs, weights)
    output.mkdir(parents=True, exist_ok=False)
    settings = configure("legacy")
    os.environ["SHAPE_ORGANOID_MODEL_PATH"] = str(weights)
    os.environ["ORGANOID_EXPECTED_SHA256"] = sha256(weights)
    os.environ["ORGANOID_REQUIRE_GPU"] = "1"
    os.environ["SHAPE_ORGANOID_ADAPTIVE_RESCUE"] = "0"
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("A working CUDA GPU is required; CPU fallback is forbidden")
    probe = torch.ones(1, device="cuda")
    probe.square().sum().item()
    torch.cuda.synchronize()
    from shape.agent.executor import Executor
    from shape.agent.verification.integration import verify_managed_tool
    from shape.agent.verification import segmentation_cohort
    from shape.solver import _segmentation_cohort, _gpu_slots
    from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
    from arms import gateway_preflight, probe_dependencies
    segmentation_cohort.MIN_GROUP = args.cohort_min_group
    meta = dict(schema=1, status="RUNNING", model=MODEL, reasoning_effort="high",
        manifest=manifest, source=source_identity(), command=sys.argv, settings=settings,
        checkpoint_sha256=sha256(weights), packages=versions(), host=platform.node(),
        python=sys.version, job_id=os.environ["SLURM_JOB_ID"],
        cpus=os.environ.get("SLURM_CPUS_PER_TASK"), image_workers=args.image_workers,
        cohort_min_group=args.cohort_min_group, cache_seed=None if args.disable_native_cache else args.cache_seed,
        hardware=dict(gpu=torch.cuda.get_device_name(0), cuda=torch.version.cuda,
            gpu_memory_bytes=torch.cuda.get_device_properties(0).total_memory),
        started_unix=time.time(),
        cost_scope="Direct managed segmentation stage; excludes outer solve planning. Global audit includes preflight and every cohort/fallback call.")
    dump(output / "run_meta.json", meta)
    executor = Executor(MODEL, query_cache_dir=str(output / "managed"), enable_signal=False)
    executor.set_query_cache_dir(str(output / "managed"))
    executor.gpu_slots = _gpu_slots()
    counts, passes, lock = {}, {}, threading.Lock()

    def run_one(spec, options):
        started = time.time()
        with lock:
            index = counts.get(spec["image_id"], 0) + 1
            counts[spec["image_id"]] = index
        resolutions, result, status = verify_managed_tool(executor, TOOL,
            {"image": spec["path"], "image_id": spec["image_id"]},
            request="Segment every whole cardiac organoid or spheroid as one instance. Preserve border organoids; do not segment constituent cells.",
            verifier_options=options)
        folder = output / "passes" / spec["image_id"] / f"pass_{index}"
        folder.mkdir(parents=True, exist_ok=False)
        record = dict(image_id=spec["image_id"], pass_index=index, options=options,
            seconds=time.time()-started, status=status, result=result,
            resolutions=[asdict(r) for r in resolutions])
        dump(folder / "stage.json", safe(record))
        with lock:
            passes.setdefault(spec["image_id"], []).append(record)
        print(f"PASS {spec['image_id']} {index} {status}", flush=True)
        return resolutions, result, status, None

    started = time.time()
    llm, native, tools = [], [], []
    try:
        probe_dependencies("shape")
        model = Organoid_Segmenter_Tool(model_path=str(weights))
        model._load_model()
        if not str(model.device).startswith("cuda"):
            raise RuntimeError("CO model failed GPU loading; no paid preflight or CPU inference was attempted")
        executor._managed_tools[TOOL] = model
        with llm_audit(output) as llm:
            preflight = output / "preflight"
            preflight.mkdir()
            engine = gateway_preflight(preflight)
            executor._verifier_llm_mm, executor._verifier_llm_tried = engine, True
            specs = [{"path": p["image"], "image_id": p["image_id"]} for p in pairs]
            with tool_audit(output, Organoid_Segmenter_Tool) as (tools, native):
                runs, cohort = _segmentation_cohort(
                    {"executor": executor, "image_workers": args.image_workers,
                     "segmentation_cache_seed": None if args.disable_native_cache else args.cache_seed},
                    specs, run_one, TOOL)
        dump(output / "cohort.json", safe(cohort))
        if any(not str(r["device"]).startswith("cuda") for r in native):
            raise RuntimeError("A native CO call fell back to CPU")
        if any(r.get("weights_sha256") != meta["checkpoint_sha256"] for r in native):
            raise RuntimeError("A native CO call used the wrong checkpoint")
        if sha256(weights) != meta["checkpoint_sha256"]:
            raise RuntimeError("CO checkpoint changed during evaluation")
        records = []
        for pair, (resolutions, result, status, _) in zip(pairs, runs):
            folder = output / "images" / pair["image_id"]
            folder.mkdir(parents=True)
            path = (result or {}).get("mask_path") or (result or {}).get("organoid_mask_path")
            error = (result or {}).get("error")
            if status != "satisfied":
                error = error or f"Managed VRR ended {status}"
            prediction = cellseg.load_mask(path) if path and not error else None
            record = evaluate_pair(pair, prediction, folder, error=error)
            record.update(arm="shape_cohort", seconds=sum(p["seconds"] for p in passes[pair["image_id"]]),
                          n_managed_passes=len(passes[pair["image_id"]]),
                          parameter_source=cohort["parameter_source"][pair["image_id"]])
            records.append(record)
            dump(folder / "record.json", record)
        summary = dict(n_planned=len(pairs), n_recorded=len(records),
            n_delivered=sum(r["delivered"] for r in records),
            n_failed=sum(bool(r["error"]) for r in records),
            stage_wall_seconds=time.time()-started, native_calls=len(native),
            tool_calls=len(tools), llm_calls=len(llm), tokens=sum(r.get("tokens") or 0 for r in llm),
            native_cache_hits=sum(bool((r.get("result") or {}).get("native_cache", {}).get("hit")) for r in tools),
            parameter_sources=cohort["counts"])
        for key in ("ap50", "ap75", "map_50_95"):
            summary[key] = sum(r[key] for r in records)/len(records)
        with (output / "records.jsonl").open("w", encoding="utf-8") as stream:
            for record in records:
                stream.write(json.dumps(record, allow_nan=False)+"\n")
        dump(output / "summary.json", summary)
        meta["status"] = "COMPLETE" if not summary["n_failed"] else "COMPLETED_WITH_FAILURES"
        meta["summary"] = summary
        print("COHORT_COMPLETE", json.dumps(summary), flush=True)
        return 0 if meta["status"] == "COMPLETE" else 2
    except BaseException as exc:
        meta["status"] = "FAILED"
        meta["error"] = safe(f"{type(exc).__name__}: {exc}")
        raise
    finally:
        meta.update(ended_unix=time.time(), observed_native_calls=len(native),
                    observed_llm_calls=len(llm), observed_tokens=sum(r.get("tokens") or 0 for r in llm))
        dump(output / "run_meta.json", safe(meta))


if __name__ == "__main__":
    raise SystemExit(main())

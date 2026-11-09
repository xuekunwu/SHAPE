#!/usr/bin/env python3
"""Seven-arm organoid evaluation. Run each arm as a separate process in shape_bench."""
from __future__ import annotations

import argparse
import csv
import json
import os
import platform
import sys
import time
from contextlib import ExitStack
from pathlib import Path
from uuid import uuid4

from common import (ARMS, CO_ARMS, DATA_ROOT, EXTRA_ARMS, LABEL_ROOT, LLM_ARMS, MODEL, REPO, WEIGHTS,
                    append, cellseg, configure, digest, dump, labels, prepare, protect_output,
                    read_manifest, scoring, sha256, source_identity, versions)

sys.path.insert(0, str(REPO))
import numpy as np


def read_records(path):
    if not Path(path).is_file():
        return []
    # Refuse corrupt/truncated records; never silently count a partially written image.
    rows = []
    for line in Path(path).read_text(encoding="utf-8").splitlines():
        if line.strip():
            rows.append(json.loads(line))
    if len({r["image_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate image records")
    return rows


def validate_mask(mask, shape):
    mask = np.asarray(mask)
    if (mask.shape != tuple(shape) or mask.ndim != 2 or not np.issubdtype(mask.dtype, np.integer)
            or (mask < 0).any()):
        raise ValueError(f"Invalid prediction: shape={mask.shape}, dtype={mask.dtype}; expected {shape}")
    # Compact IDs before serialization/scoring; IDs do not encode class or confidence.
    values, inverse = np.unique(mask, return_inverse=True)
    mapped = np.arange(len(values), dtype=np.uint32) + (0 if values[0] == 0 else 1)
    return mapped[inverse].reshape(mask.shape)


def evaluate_pair(pair, prediction, out, *, error=None):
    gt = labels(pair["label"])
    delivered = prediction is not None
    try:
        pred = validate_mask(prediction, gt.shape) if delivered else np.zeros_like(gt)
    except Exception as exc:
        delivered, error = False, f"{type(exc).__name__}: {exc}"
        pred = np.zeros_like(gt)
    if not delivered:
        error = error or "No mask delivered"
    else:
        import tifffile
        tifffile.imwrite(out / "mask.tif", pred)
    metrics = scoring.evaluate_instance_segmentation(pred, gt)
    return {"image_id": pair["image_id"], "delivered": delivered, "error": error,
            "mask_sha256": sha256(out / "mask.tif") if delivered else None, **metrics}


def run_image(arm, pair, out, name):
    from audit import llm_audit, safe, tool_audit
    out.mkdir(parents=True, exist_ok=False)
    start = time.time()
    tool_calls, evaluations, llm = [], [], []
    result, error = {}, None
    try:
        with ExitStack() as stack:
            if name in LLM_ARMS:
                llm = stack.enter_context(llm_audit(out))
            if name in CO_ARMS or name == "cellpose_sam":
                if name == "cellpose_sam":
                    from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool as cls
                else:
                    from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool as cls
                tool_calls, evaluations = stack.enter_context(tool_audit(out, cls))
            result = arm(Path(pair["image"]), out)
        if name in LLM_ARMS and (not llm or any(r.get("error") for r in llm)):
            raise RuntimeError("LLM calls failed or were absent; inspect LLM_CALLS.jsonl")
        if name in CO_ARMS or name == "cellpose_sam":
            for call in tool_calls:
                output = call.get("result")
                failed = (call.get("error") or not isinstance(output, dict)
                          or output.get("error") or output.get("status") == "failed")
                count = call["native_requests"]
                if count > 1 or (count == 0 and not failed):
                    raise RuntimeError("Unaccounted native inference or hidden rescue within one tool call")
            if result.get("mask") is not None and not evaluations:
                raise RuntimeError("A delivered model mask requires an observed native inference")
            if name in ("cellpose_sam", "shape_seg_co") and (len(evaluations) != 1 or len(tool_calls) != 1):
                raise RuntimeError("Direct baselines must execute one native inference per image")
            if name in CO_ARMS and any(r.get("weights_sha256") != os.environ.get("ORGANOID_EXPECTED_SHA256")
                                       for r in evaluations):
                raise RuntimeError("Native CO weights differ from the fixed benchmark asset")
            if os.environ.get("ORGANOID_REQUIRE_GPU") == "1" and any(not r["device"].startswith("cuda") for r in evaluations):
                raise RuntimeError("Cellpose fell back to CPU despite --require-gpu")
    except Exception as exc:
        error = safe(f"{type(exc).__name__}: {exc}")
        # Keep partial audit evidence and costs; failed execution never supplies a delivered mask.
        result["mask"] = None
        if name in LLM_ARMS:
            from shape.engine.errors import is_fatal_llm_error
            if is_fatal_llm_error(exc):
                dump(out / "FATAL.json", {"error": error})
                raise
    dump(out / "trace.json", safe(result.get("trace", {})))
    record = evaluate_pair(pair, result.get("mask"), out, error=error)
    record.update(arm=name, started_unix=start, seconds=time.time() - start,
                  tool_calls=len(tool_calls) if name in CO_ARMS or name == "cellpose_sam" else result.get("tool_calls", 0),
                  native_calls=len(evaluations) if name in CO_ARMS or name == "cellpose_sam" else result.get("tool_calls", 0),
                  tokens=sum(r.get("tokens") or 0 for r in llm), llm_calls=len(llm), attempt_dir=str(out),
                  native_seconds=sum(r["seconds"] for r in evaluations) if evaluations else None,
                  llm_seconds=sum(r["seconds"] for r in llm))
    trajectory = []
    for call in tool_calls:
        snapshot = call.get("snapshot")
        score, candidate_error = None, None
        if snapshot:
            try:
                score = scoring.evaluate_instance_segmentation(labels(snapshot), labels(pair["label"]))
            except Exception as exc:
                candidate_error = safe(f"{type(exc).__name__}: {exc}")
        trajectory.append({"ap50": score["ap50"] if score else None,
                           "error": candidate_error or call.get("error") or (call.get("result") or {}).get("error"),
                           "parameters": call.get("parameters"), "snapshot": snapshot})
    record["call_trajectory"] = trajectory
    dump(out / "record.json", record)
    return record


def run(args):
    from arms import gateway_preflight, make_arm, probe_dependencies
    from audit import llm_audit
    manifest = read_manifest(args.manifest)
    pairs = manifest["pairs"]
    weights = Path(args.model_path).resolve() if args.arm in CO_ARMS else None
    if weights and not weights.is_file():
        raise ValueError("--model-path must identify the exact trained checkpoint file, not a directory")
    root = Path(args.run_dir).resolve()
    protect_output(root, pairs, weights)
    settings = configure(args.search_strategy, args.vrr_policy, args.max_completion_tokens, args.reasoning_effort)
    os.environ["ORGANOID_REQUIRE_GPU"] = "1" if args.require_gpu else "0"
    if weights:
        os.environ["SHAPE_ORGANOID_MODEL_PATH"] = str(weights)
        os.environ["ORGANOID_EXPECTED_SHA256"] = sha256(weights)
    source = source_identity()
    identity = {"manifest": manifest["fingerprint"], "source": source["source_sha256"], "arm": args.arm,
                "weights": sha256(weights) if weights else None, "settings": settings,
                "max_steps": args.max_steps, "require_gpu": args.require_gpu,
                "python": sys.executable, "packages": versions()}
    arm_dir = root / args.arm
    arm_dir.mkdir(parents=True, exist_ok=True)
    meta_path = arm_dir / "run_meta.json"
    old = {}
    if meta_path.exists():
        old = json.loads(meta_path.read_text(encoding="utf-8"))
        if not args.resume or old["identity"] != identity:
            raise ValueError("Existing arm requires --resume and identical data/code/model/settings/environment")
    elif args.resume:
        raise ValueError("Cannot resume an arm without its run_meta.json")
    meta = {"identity": identity, "source": source, "manifest": manifest, "model": MODEL if args.arm in LLM_ARMS else None,
            "reasoning_effort": args.reasoning_effort if args.arm in LLM_ARMS else None, "host": platform.node(),
            "python": sys.version, "environment": sys.prefix, "job_id": os.environ.get("SLURM_JOB_ID"),
            "command": sys.argv, "started_unix": time.time(), "status": "RUNNING",
            "preflights": old.get("preflights", []),
            "previous_sessions": old.get("previous_sessions", []) + ([{k: old.get(k) for k in
                ("started_unix", "ended_unix", "status", "host", "job_id")} ] if old else [])}
    dump(meta_path, meta)
    try:
        import torch
        meta["cuda"] = {"available": torch.cuda.is_available(), "version": torch.version.cuda,
                        "device": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None}
    except ImportError:
        meta["cuda"] = None
    if args.arm == "stardist":
        try:
            import tensorflow as tf
            devices = tf.config.list_physical_devices("GPU")
            meta["cuda"] = {"available": bool(devices), "backend": "tensorflow", "devices": [d.name for d in devices]}
        except ImportError:
            meta["cuda"] = None
    dump(meta_path, meta)
    if args.require_gpu and not (meta["cuda"] or {}).get("available"):
        meta["status"] = "PREFLIGHT_FAILED"
        dump(meta_path, meta)
        raise RuntimeError("GPU required but unavailable in this arm's environment")
    records_path = arm_dir / "records.jsonl"
    records = read_records(records_path)
    if any(r["image_id"] not in {p["image_id"] for p in pairs} for r in records):
        raise ValueError("Record does not belong to the frozen manifest")
    done = {r["image_id"] for r in records}
    pending = [p for p in pairs if p["image_id"] not in done]
    for record in records:
        if record["delivered"] and sha256(Path(record["attempt_dir"]) / "mask.tif") != record["mask_sha256"]:
            raise ValueError("Previously delivered mask changed")
    try:
        engine = None
        if pending or args.preflight_only:
            meta["import_origins"] = probe_dependencies(args.arm)
        if args.arm in LLM_ARMS and (pending or args.preflight_only):
            probe = arm_dir / ("preflight_" + uuid4().hex[:12])
            probe.mkdir()
            meta["preflights"].append(str(probe))
            with llm_audit(probe, strict_structured=True):
                engine = gateway_preflight(probe)
        if args.preflight_only:
            meta["status"] = "PREFLIGHT_ONLY"
            dump(meta_path, meta)
            return meta
        init_start = time.time()
        arm = make_arm(args.arm, weights, engine, args.max_steps) if pending else None
        meta["adapter_initialization_seconds"] = time.time() - init_start
        for pair in pending:
            out = arm_dir / "images" / pair["image_id"] / ("attempt_" + uuid4().hex[:12])
            out.parent.mkdir(parents=True, exist_ok=True)
            record = run_image(arm, pair, out, args.arm)
            append(records_path, record)
            records.append(record)
            print(f"{args.arm} {pair['image_id']}: AP50={record['ap50']:.4f} "
                  f"calls={record['native_calls']} tokens={record['tokens']} error={record['error']}", flush=True)
        if weights and sha256(weights) != identity["weights"]:
            raise ValueError("CO checkpoint changed during the run")
        meta["status"] = "COMPLETE" if all(not r["error"] for r in records) else "COMPLETED_WITH_FAILURES"
    except BaseException:
        meta["status"] = "INTERRUPTED_OR_FAILED"
        raise
    finally:
        meta["ended_unix"] = time.time()
        meta["n_recorded"] = len(records)
        dump(meta_path, meta)
    return meta


def compare(args):
    root = Path(args.run_dir).resolve()
    manifest = read_manifest(args.manifest)
    protect_output(root, manifest["pairs"])
    expected = {p["image_id"] for p in manifest["pairs"]}
    by_arm, identities, preflight_costs = {}, {}, {}
    for arm in args.arms:
        meta = json.loads((root / arm / "run_meta.json").read_text(encoding="utf-8"))
        if meta["status"] not in ("COMPLETE", "COMPLETED_WITH_FAILURES"):
            raise ValueError(f"Incomplete arm: {arm}")
        ident = meta["identity"]
        if ident["manifest"] != manifest["fingerprint"]:
            raise ValueError(f"Different manifest: {arm}")
        rows = read_records(root / arm / "records.jsonl")
        if {r["image_id"] for r in rows} != expected or any(r["arm"] != arm for r in rows):
            raise ValueError(f"Incomplete/mislabelled records: {arm}")
        for row in rows:
            if row["delivered"] and sha256(Path(row["attempt_dir"]) / "mask.tif") != row["mask_sha256"]:
                raise ValueError(f"Delivered mask changed: {arm}/{row['image_id']}")
        by_arm[arm], identities[arm] = rows, ident
        probes = [record for p in meta.get("preflights", []) for record in read_llm_calls(Path(p) / "LLM_CALLS.jsonl")]
        preflight_costs[arm] = {"tokens": sum(r.get("tokens") or 0 for r in probes),
                                "seconds": sum(r["seconds"] for r in probes), "calls": len(probes)}
    if len({i["source"] for i in identities.values()}) != 1:
        raise ValueError("Arms were run with different source code")
    if len({digest(i["settings"]) for i in identities.values()}) != 1:
        raise ValueError("Arms were run with different experiment settings")
    if len({i["weights"] for a, i in identities.items() if a in CO_ARMS}) > 1:
        raise ValueError("CO model hash differs across direct/ReAct/SHAPE arms")
    if len({i["max_steps"] for i in identities.values()}) != 1:
        raise ValueError("Different agent budgets")
    if len({i["require_gpu"] for arm, i in identities.items() if arm != "llm"}) > 1:
        raise ValueError("Different required device policies")
    baseline = {r["image_id"]: r for r in by_arm.get("shape_seg_co", [])}
    summary = {}
    for arm, rows in by_arm.items():
        n = len(rows)
        calls, tokens, seconds = (sum(r[k] for r in rows) for k in ("native_calls", "tokens", "seconds"))
        gains = [r["ap50"] - baseline[r["image_id"]]["ap50"] for r in rows] if baseline else []
        retunes = [(prev, cur) for r in rows for prev, cur in zip(r["call_trajectory"], r["call_trajectory"][1:])]
        observed = [(p["ap50"], c["ap50"]) for p, c in retunes
                    if p["ap50"] is not None and c["ap50"] is not None and not p["error"] and not c["error"]]
        summary[arm] = {"n_images": n, "n_delivered": sum(r["delivered"] for r in rows),
                        "n_errors": sum(bool(r["error"]) for r in rows),
                        **{key: float(np.mean([r[key] for r in rows])) for key in
                           ("ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50", "f1_at_50")},
                        "native_calls_per_image": calls / n, "tokens_per_image": tokens / n,
                        "llm_calls_per_image": sum(r["llm_calls"] for r in rows) / n, "seconds_per_image": seconds / n,
                        "native_seconds_per_image": sum(r["native_seconds"] for r in rows) / n
                            if all(r["native_seconds"] is not None for r in rows) else None,
                        "llm_seconds_per_image": sum(r["llm_seconds"] for r in rows) / n,
                        "retunes": len(retunes), "evaluable_retunes": len(observed),
                        "positive_retune_rate": sum(c - p >= .01 for p, c in observed) / len(observed) if observed else None,
                        "wasted_retune_rate": sum(abs(c - p) < .01 for p, c in observed) / len(observed) if observed else None,
                        "ap50_gain_vs_direct_co": float(np.mean(gains)) if gains else None,
                        "delivered_gain_per_1k_tokens": sum(gains) * 1000 / tokens if gains and tokens else None}
        # An interrupted image can spend tokens before it has a completed record.
        # Retain that cost across --resume rather than charging only successful retries.
        selected_attempts = {Path(r["attempt_dir"]).resolve() for r in rows}
        orphan_llm, orphan_native = [], []
        for attempt in (root / arm / "images").glob("*/attempt_*"):
            if attempt.resolve() in selected_attempts:
                continue
            for file, bucket in (("LLM_CALLS.jsonl", orphan_llm), ("NATIVE_CALLS.jsonl", orphan_native)):
                if (attempt / file).is_file():
                    bucket.extend(read_llm_calls(attempt / file))
        orphan_tokens = sum(r.get("tokens") or 0 for r in orphan_llm)
        summary[arm].update(preflight_tokens=preflight_costs[arm]["tokens"],
                            preflight_seconds=preflight_costs[arm]["seconds"],
                            interrupted_attempt_tokens=orphan_tokens,
                            interrupted_attempt_native_calls=len(orphan_native),
                            total_tokens_including_preflight=tokens + preflight_costs[arm]["tokens"] + orphan_tokens)
    report = {"manifest": manifest["fingerprint"], "summary": summary,
              "note": "Failure-inclusive macro instance AP; empty/missing predictions have all FN. No GT used by agents."}
    dump(root / "comparison.json", report)
    with (root / "comparison.csv").open("w", newline="", encoding="utf-8") as fh:
        writer = csv.DictWriter(fh, fieldnames=["arm", *next(iter(summary.values()))])
        writer.writeheader()
        writer.writerows({"arm": arm, **values} for arm, values in summary.items())
    return report


def read_llm_calls(path):
    return [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]


def parser():
    cli = argparse.ArgumentParser(description=__doc__)
    sub = cli.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="Freeze and validate image/GT pairs (no inference)")
    prep.add_argument("--data-root", default=DATA_ROOT)
    prep.add_argument("--image-root")
    prep.add_argument("--label-root", help=f"Default for the study data: {LABEL_ROOT}; otherwise auto-discover labels/")
    prep.add_argument("--label-suffix", default="_seg")
    prep.add_argument("--manifest", required=True)
    prep.add_argument("--sample-size", type=int)
    prep.add_argument("--seed", type=int, default=112)
    runner = sub.add_parser("run", help="Run one arm in the active shape_bench environment")
    runner.add_argument("--manifest", required=True)
    runner.add_argument("--run-dir", required=True)
    runner.add_argument("--arm", required=True, choices=ARMS + EXTRA_ARMS)
    runner.add_argument("--model-path", default=WEIGHTS)
    runner.add_argument("--max-steps", type=int, default=10)
    runner.add_argument("--search-strategy", choices=("legacy", "four_stage"), default="legacy")
    runner.add_argument("--vrr-policy", choices=("legacy", "balanced_v1"), default="legacy")
    runner.add_argument("--max-completion-tokens", type=int, help="pin every LLM reply's completion cap")
    runner.add_argument("--reasoning-effort", choices=("low", "medium", "high"), default="high")
    runner.add_argument("--require-gpu", action="store_true")
    runner.add_argument("--resume", action="store_true")
    runner.add_argument("--preflight-only", action="store_true", help="Check inputs/GPU and LLM text+vision, then stop before inference")
    comp = sub.add_parser("compare", help="Require every requested arm on exactly the same pairs/code/settings")
    comp.add_argument("--manifest", required=True)
    comp.add_argument("--run-dir", required=True)
    comp.add_argument("--arms", nargs="+", choices=ARMS + EXTRA_ARMS, default=list(ARMS))
    return cli


def main():
    args = parser().parse_args()
    if args.command == "prepare":
        if args.data_root == DATA_ROOT and not args.label_root:
            args.label_root = LABEL_ROOT
        result = prepare(args)
        print(f"PREPARED {len(result['pairs'])} pairs; fingerprint={result['fingerprint']}")
    elif args.command == "run":
        if args.max_steps < 1:
            raise ValueError("max-steps must be positive")
        result = run(args)
        print(result["status"])
        if result["status"] == "COMPLETED_WITH_FAILURES":
            raise SystemExit(2)
    else:
        result = compare(args)
        print(json.dumps(result["summary"], indent=2))


if __name__ == "__main__":
    main()

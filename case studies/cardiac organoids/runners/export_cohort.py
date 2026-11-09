#!/usr/bin/env python3
"""Export a completed managed CO cohort run without rescoring or inventing per-image LLM costs."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path

SCORES = ("ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50", "f1_at_50")
COUNTS = ("ground_truth_cell_count", "predicted_cell_count", "mean_matched_iou_at_50", "cell_count_error_fraction")
IDENTITY = ("source_commit", "source_sha256", "manifest_fingerprint", "checkpoint_sha256")
COLUMNS = ("image_id", "arm", "evaluation_status", "delivered", *IDENTITY,
    "image_height_px", "image_width_px", "image_sha256", "ground_truth_sha256",
    *COUNTS, *SCORES, "managed_pass_seconds", "n_managed_passes", "parameter_source",
    "tool_calls", "native_calls", "native_cache_hits", "mask_sha256", "error")


def sha(data):
    return hashlib.sha256(data).hexdigest()


def encode(value):
    return (json.dumps(value, indent=2, sort_keys=True, allow_nan=False)+"\n").encode()


def build(folder, expected_commit, verify_masks=False):
    folder = Path(folder)
    read = lambda name: json.loads((folder/name).read_text(encoding="utf-8"))
    lines = lambda name: [json.loads(s) for s in (folder/name).read_text(encoding="utf-8").splitlines() if s]
    meta, summary = read("run_meta.json"), read("summary.json")
    if meta["status"] not in ("COMPLETE", "COMPLETED_WITH_FAILURES"):
        raise ValueError("Run is unfinished; never export missing predictions as zero")
    source, manifest = meta["source"], meta["manifest"]
    if source["commit"] != expected_commit or source["dirty"]:
        raise ValueError("Unexpected or dirty inference source")
    digest = lambda o: sha(json.dumps(o, sort_keys=True).encode())
    if not source["files"] or digest(source["files"]) != source["source_sha256"]:
        raise ValueError("Source-file attestation is inconsistent")
    if digest({k:v for k,v in manifest.items() if k != "fingerprint"}) != manifest["fingerprint"]:
        raise ValueError("Manifest fingerprint mismatch")
    pairs = manifest["pairs"]
    ids = [p["image_id"] for p in pairs]
    records, tools, native, llm = [lines(n+".jsonl") for n in ("records", "TOOL_CALLS", "NATIVE_CALLS", "LLM_CALLS")]
    if len(set(ids)) != len(ids) or [r["image_id"] for r in records] != ids:
        raise ValueError("Missing, duplicate, reordered or unexpected image records")
    if meta.get("summary") != summary:
        raise ValueError("Final metadata and summary disagree")
    if any(not r["device"].startswith("cuda") or r.get("weights_sha256") != meta["checkpoint_sha256"] for r in native):
        raise ValueError("Native calls must use CUDA and the attested checkpoint")
    if any(r.get("model") != meta["model"] or r.get("reasoning_effort") != meta["reasoning_effort"] for r in llm):
        raise ValueError("Audited LLM configuration drift")
    if any(not r.get("error") and (not isinstance(r.get("tokens"), int) or r["tokens"] <= 0) for r in llm):
        raise ValueError("Successful LLM call lacks positive token usage")
    if (len(native),len(tools),len(llm),sum(r.get("tokens") or 0 for r in llm)) != tuple(summary[k] for k in ("native_calls","tool_calls","llm_calls","tokens")):
        raise ValueError("Summary and audit call/usage totals disagree")
    by_image = defaultdict(list)
    for call in tools:
        image = call["parameters"]["image_id"]
        if image not in ids or not isinstance(call["native_requests"], int) or call["native_requests"] < 0:
            raise ValueError("Unattributed or invalid native-call count")
        by_image[image].append(call)
    if sum(r["native_requests"] for r in tools) != len(native):
        raise ValueError("Per-image native accounting does not reconcile")
    counts = dict(n_planned=len(pairs), n_recorded=len(records),
                  n_delivered=sum(bool(r["delivered"]) for r in records), n_failed=sum(bool(r["error"]) for r in records))
    if any(summary[k] != n for k,n in counts.items()):
        raise ValueError("Coverage totals disagree")
    if (meta["status"] == "COMPLETE") != (counts["n_failed"] == 0):
        raise ValueError("Completion status and failure count disagree")
    for key in SCORES:
        if any(not math.isfinite(r[key]) or not 0 <= r[key] <= 1 for r in records):
            raise ValueError("Invalid score")
    for key in SCORES[:3]:
        if not math.isclose(sum(r[key] for r in records)/len(records), summary[key], abs_tol=1e-12):
            raise ValueError("Aggregate score mismatch")
    output = []
    for pair,record in zip(pairs,records):
        delivered = bool(record["delivered"])
        if delivered == bool(record["error"]):
            raise ValueError("Delivery/error mismatch")
        if verify_masks and delivered:
            if sha((folder/"images"/record["image_id"]/"mask.tif").read_bytes()) != record["mask_sha256"]:
                raise ValueError("Delivered mask hash mismatch")
        calls = by_image[record["image_id"]]
        row = dict(image_id=record["image_id"], arm="shape_cohort", evaluation_status="success" if delivered else "failed",
            delivered=str(delivered).lower(), source_commit=source["commit"], source_sha256=source["source_sha256"],
            manifest_fingerprint=manifest["fingerprint"], checkpoint_sha256=meta["checkpoint_sha256"],
            image_height_px=pair["shape"][0], image_width_px=pair["shape"][1], image_sha256=pair["image_sha256"],
            ground_truth_sha256=pair["label_sha256"], managed_pass_seconds=record["seconds"],
            n_managed_passes=record["n_managed_passes"], parameter_source=record["parameter_source"],
            tool_calls=len(calls), native_calls=sum(c["native_requests"] for c in calls),
            native_cache_hits=sum(bool((c.get("result") or {}).get("native_cache",{}).get("hit")) for c in calls),
            mask_sha256=record["mask_sha256"], error=record["error"])
        row.update({k:record[k] for k in (*COUNTS,*SCORES)})
        for threshold,metrics in record["_thresholds"].items():
            for key,value in metrics.items():
                if key != "matched_ious":
                    row["iou_"+threshold.replace(".", "")+"_"+key] = value
        output.append(row)
    if sum(r["native_cache_hits"] for r in output) != summary["native_cache_hits"]:
        raise ValueError("Cache-hit totals disagree")
    if dict(Counter(r["parameter_source"] for r in output)) != summary["parameter_sources"]:
        raise ValueError("Parameter-source totals disagree")
    fields = [*COLUMNS, *sorted(k for k in output[0] if k.startswith("iou_"))]
    stream = io.StringIO(newline="")
    writer = csv.DictWriter(stream, fieldnames=fields, lineterminator="\n")
    writer.writeheader(); writer.writerows(output)
    data = stream.getvalue().encode()
    audit_files = {name:dict(sha256=sha((folder/name).read_bytes()),bytes=(folder/name).stat().st_size)
                   for name in ("run_meta.json","summary.json","cohort.json","records.jsonl","TOOL_CALLS.jsonl","NATIVE_CALLS.jsonl","LLM_CALLS.jsonl")}
    receipt = dict(schema=1, arm="shape_cohort", exporter_sha256=sha(Path(__file__).read_bytes()), inference_source=source, model=meta["model"],
        checkpoint_sha256=meta["checkpoint_sha256"], manifest_fingerprint=manifest["fingerprint"],
        selection=manifest.get("selection"), packages=meta["packages"], hardware=meta["hardware"], job_id=meta["job_id"],
        summary=summary, audited_errors={k:sum(bool(r.get("error")) for r in rows) for k,rows in (("llm",llm),("native",native),("tool",tools))},
        mask_files_rehashed=verify_masks, source_artifacts=audit_files,
        llm_calls_without_reported_usage=sum(not isinstance(r.get("tokens"),int) or r["tokens"]<=0 for r in llm),
        tokens_scope="Reported usage only; errored calls without usage remain unmeasured.",
        file=dict(name="shape_cohort.csv",sha256=sha(data),bytes=len(data),rows=len(output)),
        cost_scope=meta["cost_scope"],
        cost_note="managed_pass_seconds sums this image's managed passes; shared LLM usage is reported globally, never allocated to images; summed concurrent seconds are not wall time. stage_wall_seconds includes model setup, gateway preflight, cohort/VRR and scoring, excluding outer solve planning and scheduler wait.")
    return data, encode(receipt)


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--run-dir", required=True, type=Path)
    cli.add_argument("--expected-commit", required=True)
    cli.add_argument("--output-dir", required=True, type=Path)
    cli.add_argument("--verify-mask-files", action="store_true", help="Rehash all delivered masks; perform large exports inside Slurm")
    args=cli.parse_args()
    data, receipt=build(args.run_dir,args.expected_commit,args.verify_mask_files)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    (args.output_dir/"shape_cohort.csv").write_bytes(data)
    (args.output_dir/"manifest.json").write_bytes(receipt)
    print("EXPORTED",json.loads(receipt)["file"])


if __name__=="__main__":
    main()

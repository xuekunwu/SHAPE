#!/usr/bin/env python3
"""Compare fixed, failure-inclusive organoid cohort exports with frozen historical arms."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import hashlib
import io
import json
import math
from pathlib import Path
import statistics
import evaluation as old

SOURCE = "a879dca9619645fdd670eb7818c4efc3bbf19203"
FINGERPRINTS = {150: "ff0a8c0fddaa585752c36d2e443b1c815bb0cb977e89989b7012aa6ddbfae6df",
                5: "db383d109ba94fb127f53f5c8b3e5b2188977a48ad8bdf8147589b0ae3e4a43e"}
SELECTION_SHA = "080591cfcb4ecf97df914968975c379bca7c7bf72b89c4849adec27e3b0442b3"
SCORES = old.SCORES[:3]


def load_export(folder, expected_ids, historical):
    folder = Path(folder)
    raw_meta = (folder/"manifest.json").read_bytes()
    meta = json.loads(raw_meta)
    raw = (folder/"shape_cohort.csv").read_bytes()
    if old.sha256(raw) != meta["file"]["sha256"] or len(raw) != meta["file"]["bytes"]:
        raise ValueError("Cohort CSV hash or size mismatch")
    rows = list(csv.DictReader(io.StringIO(raw.decode())))
    if [r["image_id"] for r in rows] != list(expected_ids) or len(set(expected_ids)) != len(rows):
        raise ValueError("Fixed cohort membership or ordering mismatch")
    source = meta["inference_source"]
    if source["commit"] != SOURCE or source["dirty"]:
        raise ValueError("Inference source mismatch")
    if old.sha256(json.dumps(source["files"],sort_keys=True).encode()) != source["source_sha256"]:
        raise ValueError("Source attestation mismatch")
    selection = meta["selection"]
    if selection["hf_revision"] != old.HF_REVISION or selection["selection_sha256"] != SELECTION_SHA:
        raise ValueError("Frozen selection mismatch")
    if meta["manifest_fingerprint"] != FINGERPRINTS[len(rows)]:
        raise ValueError("Unexpected input manifest")
    if meta["model"] != "gpt-5.6-luna" or meta["hardware"]["cuda"] is None:
        raise ValueError("Model or CUDA provenance mismatch")
    for row in rows:
        i=row["image_id"]
        if any(row[k] != historical["shape"][i][k] for k in old.IDENTITY):
            raise ValueError("Image/GT identity mismatch")
        if row["checkpoint_sha256"] != historical["shape_seg_co"][i]["checkpoint_sha256"]:
            raise ValueError("CO checkpoint mismatch")
        for key,value in (("source_commit", SOURCE), ("source_sha256",source["source_sha256"]),
                          ("manifest_fingerprint",meta["manifest_fingerprint"]), ("checkpoint_sha256",meta["checkpoint_sha256"])):
            if row[key] != value: raise ValueError("Row attestation mismatch")
        success=row["evaluation_status"]=="success"
        if row["evaluation_status"] not in ("success","failed") or row["delivered"] != str(success).lower() or bool(row["error"])==success:
            raise ValueError("Inconsistent delivery status")
        if row["arm"] != "shape_cohort": raise ValueError("Unexpected arm")
        for key in old.SCORES:
            if not math.isfinite(float(row[key])) or not 0 <= float(row[key]) <= 1:
                raise ValueError("Invalid score")
    summary=meta["summary"]
    expected=dict(n_planned=len(rows),n_recorded=len(rows),n_delivered=sum(r["delivered"]=="true" for r in rows),
                  n_failed=sum(r["evaluation_status"]=="failed" for r in rows))
    if meta["file"]["rows"] != len(rows) or any(summary[k]!=v for k,v in expected.items()):
        raise ValueError("Coverage mismatch")
    for key in SCORES:
        if not math.isclose(old.mean(rows,key),summary[key],abs_tol=1e-12): raise ValueError("Score reconciliation failed")
    for key in ("tool_calls","native_calls","native_cache_hits"):
        if sum(int(r[key]) for r in rows)!=summary[key]: raise ValueError("Cost reconciliation failed")
    if dict(Counter(r["parameter_source"] for r in rows))!=summary["parameter_sources"]:
        raise ValueError("Routing count mismatch")
    return meta,rows,old.sha256(raw_meta)


def build(historical_folder, cohort_folder, hf_revision):
    if len(hf_revision)!=40 or any(c not in '0123456789abcdef' for c in hf_revision):
        raise ValueError("Use an immutable HF revision")
    _, historical=old.load_inputs(historical_folder)
    old.cohorts(historical)
    ids=[i for i,r in historical["shape"].items() if old.observed(r)][:150]
    if len(ids)!=150: raise ValueError("Historical fixed cohort is incomplete")
    full,rows,meta_sha=load_export(Path(cohort_folder)/"full150",ids,historical)
    smoke_ids=[ids[i] for i in (0,37,74,111,149)]
    smoke,smoke_rows,smoke_sha=load_export(Path(cohort_folder)/"smoke5",smoke_ids,historical)
    summaries,_=old.summarize(historical,ids,(*old.BASELINES,"shape"))
    summary_rows=[{k:r[k] for k in ("arm","n_images","n_delivered","n_failed",*SCORES)} for r in summaries]
    summary_rows.append(dict(arm="shape_cohort",n_images=150,n_delivered=full["summary"]["n_delivered"],
                             n_failed=full["summary"]["n_failed"],**{k:old.mean(rows,k) for k in SCORES}))
    paired=[]
    for ref in (*old.BASELINES,"shape"):
        result=dict(arm="shape_cohort",reference=ref,n_pairs=150)
        for key in SCORES:
            delta=[float(r[key])-float(historical[ref][r["image_id"]][key]) for r in rows]
            result.update({key+"_mean_delta":statistics.fmean(delta),key+"_wins":sum(d>1e-12 for d in delta),
                           key+"_ties":sum(abs(d)<=1e-12 for d in delta),key+"_losses":sum(d< -1e-12 for d in delta)})
        paired.append(result)
    notes=["Fixed first 150 observed historical SHAPE records, including the historical failure; no outcome-based filtering.",
           "Recorded failures remain in the 150-image denominator. Operational delivery is distinct from segmentation quality.",
           "AP(t)=TP/(TP+FP+FN); mAP averages IoU thresholds .50:.05:.95. These are not confidence-ranked COCO AP.",
           "New cost scope includes model setup, gateway preflight, managed cohort/VRR and scoring; excludes outer solve planning and scheduler wait.",
           "Historical costs cover a different full-pipeline scope on different hardware. No causal speedup or matched cost reduction is established.",
           "Audited logical LLM calls may include internal structured-output fallback requests; tokens are reported engine usage, not a provider billing audit.",
           "Mask hashes were recorded during inference; these exported files do not independently rehash full150 masks.",
           "LLM/ReAct canaries have no overlapping records on this fixed 150-image cohort and are omitted.",
           "Training/evaluation non-overlap and biological replicate independence are unverified. No significance or generalization claim."]
    report=dict(schema=1,historical_source=dict(repo=old.HF_REPO,revision=old.HF_REVISION,manifest_sha256=old.MANIFEST_SHA256),
                cohort_source=dict(repo=old.HF_REPO,revision=hf_revision,folder="cardiac_organoids_segmentation/cohort_20261020",
                    csv_sha256=full["file"]["sha256"],export_manifest_sha256=meta_sha,smoke_export_manifest_sha256=smoke_sha),
                n_images=150,ordered_membership_sha256=old.sha256(json.dumps(ids).encode()),summaries=summary_rows,paired_deltas=paired,
                full150={k:full[k] for k in ("job_id","hardware","model","checkpoint_sha256","manifest_fingerprint","inference_source",
                    "packages","summary","audited_errors","mask_files_rehashed","llm_calls_without_reported_usage","cost_scope","cost_note")},
                smoke5=dict(summary=smoke["summary"],hardware=smoke["hardware"],mask_files_rehashed=smoke["mask_files_rehashed"]),
                failure=dict(count=1,statuses=dict(Counter(r["error"] for r in rows if r["error"])),
                    note="Full per-image VRR exhausted its accepted candidates and returned conceded; failed scores retained."),limitations=notes)
    url=f"https://huggingface.co/datasets/{old.HF_REPO}/tree/{hf_revision}/cardiac_organoids_segmentation/cohort_20261020"
    text=["# Optimized cardiac organoid cohort: fixed 150 images", "",
          "150/150 recorded, **149 delivered and one recorded VRR failure**. Of the 150 final results, 121 used cohort parameters, "
          "six were calibration images and 23 used per-image VRR. All native inference used the pinned CO checkpoint on CUDA.","",
          "| Arm | Images | Failed | AP50 | AP75 | mAP50-95 |", "| --- | ---: | ---: | ---: | ---: | ---: |"]
    labels=dict(old.LABELS,shape="SHAPE historical",shape_cohort="SHAPE cohort")
    for r in summary_rows:
        text.append(f"| {labels[r['arm']]} | {r['n_images']} | {r['n_failed']} | {r['ap50']:.6f} | {r['ap75']:.6f} | {r['map_50_95']:.6f} |")
    d=next(r for r in paired if r["reference"]=="shape")
    text += ["",f"Cohort minus historical SHAPE: AP50 {d['ap50_mean_delta']:+.6f}, AP75 {d['ap75_mean_delta']:+.6f}, "
             f"mAP {d['map_50_95_mean_delta']:+.6f}. microSAM has the highest mAP on this cohort. These are descriptive paired results.","",
             "## Routing, cost and failure", "",
             f"The managed evaluation took {full['summary']['stage_wall_seconds']/60:.2f} minutes on {full['hardware']['gpu']}. "
             f"There were {full['summary']['native_calls']} native calls, {full['summary']['tool_calls']} tool calls, "
             f"{full['summary']['native_cache_hits']} native cache hits, {full['summary']['llm_calls']} audited logical LLM calls "
             f"and {full['summary']['tokens']:,} reported engine tokens. Shared usage is reported globally rather than invented per-image allocations.", "",
             "The failed image completed its per-image VRR search with status `conceded`. Candidate masks failed visual QC. "
             "The run intentionally exited 2 (`COMPLETED_WITH_FAILURES`), so Slurm marks job 48641513 FAILED despite all 150 records being finalized. "
             "The log also contains recovered structured-output token-limit exceptions; they are not evidence that this recorded failure was an API failure.","",
             "The 5-image smoke passed 5/5 with mAP 0.46, identical to historical SHAPE on those images. It used only per-image VRR. "
             "The 150-image evaluation demonstrates cohort reuse and VRR routing; a matched-budget, matched-hardware timing experiment is still needed to quantify acceleration.","",
             "## Sources and replay", "", f"Per-image CSVs and export manifests: [immutable private HF snapshot]({url}). "
             "Historical eight-arm files remain at their original immutable revision. Raw image IDs, masks, logs and weights are excluded from this Git report.","",
             "Reproduce with `compare_cohort.py` in the [runners directory](../../runners/README.md#compare-the-fixed-150-image-cohort). "
             "`scores.csv` contains unrounded macro scores, `paired.csv` contains failure-inclusive paired deltas and win/tie/loss counts, "
             "and `comparison.json` contains inference, input, hardware and cost attestations. `EXPORT.json` hashes every generated report file.","",
             "## Interpretation limits", ""]
    text += ["- "+n for n in notes]
    files={"README.md":("\n".join(text)+"\n").encode(),"scores.csv":old.csv_bytes(summary_rows),
           "paired.csv":old.csv_bytes(paired),"comparison.json":old.encoded(report)}
    files["EXPORT.json"]=old.encoded(dict(schema=1,builder_sha256=old.sha256(Path(__file__).read_bytes()),
        files={n:dict(sha256=old.sha256(b),bytes=len(b)) for n,b in files.items()}))
    return files


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--historical-dir",type=Path,required=True)
    p.add_argument("--cohort-dir",type=Path,required=True,help="Contains smoke5/ and full150/ exports")
    p.add_argument("--hf-revision",required=True)
    p.add_argument("--output-dir",type=Path,required=True)
    p.add_argument("--verify",action="store_true",help="Verify byte-identical existing reports")
    args=p.parse_args()
    files=build(args.historical_dir,args.cohort_dir,args.hf_revision)
    if args.verify:
        for n,b in files.items():
            if (args.output_dir/n).read_bytes()!=b: raise ValueError("Report mismatch: "+n)
        print("VERIFIED",len(files),"byte-identical reports")
    else:
        args.output_dir.mkdir(parents=True,exist_ok=False)
        for n,b in files.items(): (args.output_dir/n).write_bytes(b)
        print("WROTE",len(files),"reports")

if __name__=="__main__": main()

#!/usr/bin/env python3
"""Replay a fixed development smoke against hash-verified historical per-arm exports."""
import argparse,json,shutil
from pathlib import Path
from compare_bounded_validation import ARMS,HF_REVISION,FROZEN_MANIFEST_SHA,load,sha,compare,write_csv

def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--source-dir",required=True,type=Path)
    p.add_argument("--baseline-dir",required=True,type=Path)
    p.add_argument("--previous-csv",action="append",required=True,type=Path)
    p.add_argument("--output-dir",required=True,type=Path)
    a=p.parse_args()
    if a.output_dir.exists():raise ValueError("Use a fresh output directory")
    rows=load(a.source_dir/"shape_cohort_bounded.csv")
    manifest=json.loads((a.source_dir/"pairs.json").read_text())
    identity=json.loads((a.source_dir/"identity.json").read_text())
    expected={r["image_id"]:r for r in manifest["pairs"]}
    if [r["image_id"] for r in rows]!=list(expected):raise ValueError("Fixed smoke membership/order differs")
    if identity["manifest"]!=manifest["fingerprint"]:raise ValueError("Run/manifest identity differs")
    for r in rows:
        pair=expected[r["image_id"]]
        if (r["image_sha256"]!=pair["image_sha256"] or r["ground_truth_sha256"]!=pair["label_sha256"]
            or [int(r["image_height_px"]),int(r["image_width_px"])]!=pair["shape"]
            or r["source_commit"]!=identity["source"]["commit"]):raise ValueError("Smoke provenance differs")
    bm=a.baseline_dir/"manifest.json"
    if sha(bm)!=FROZEN_MANIFEST_SHA:raise ValueError("Unexpected baseline snapshot")
    bm=json.loads(bm.read_text());baselines={}
    for arm in ARMS:
        f=a.baseline_dir/(arm+".csv")
        if sha(f)!=bm["files"][f.name]["sha256"]:raise ValueError("Baseline changed")
        baselines[arm]=load(f)
    previous=[]
    for f in a.previous_csv:previous.extend(load(f))
    if len({r["image_id"] for r in previous})!=len(previous):raise ValueError("Duplicate previous image IDs")
    baselines["shape_cohort_previous"]=previous
    exports,scores,paired=compare(rows,baselines)
    a.output_dir.mkdir(parents=True)
    for arm,data in exports.items():write_csv(a.output_dir/(arm+".csv"),data)
    write_csv(a.output_dir/"paired.csv",paired)
    result=dict(schema=1,n=len(rows),source_commit=identity["source"]["commit"],manifest_fingerprint=identity["manifest"],
        source_csv_sha256=sha(a.source_dir/"shape_cohort_bounded.csv"),baseline_hf_revision=HF_REVISION,
        source_previous_csv_hashes=[sha(f) for f in a.previous_csv],arms=scores,
        scope="Outcome-informed development smoke; failure scores retained, missing historical observations never imputed",
        summary=json.loads((a.source_dir/"summary.json").read_text()))
    (a.output_dir/"comparison.json").write_text(json.dumps(result,indent=2)+"\n")
    shutil.copytree(a.source_dir,a.output_dir/"source")
    print(json.dumps({k:v["scores"] for k,v in scores.items()}))

if __name__=="__main__":main()

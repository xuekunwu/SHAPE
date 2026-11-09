#!/usr/bin/env python3
"""Hash-checked paired comparison of schema-2 cohort CSVs against frozen arm exports."""
import argparse,csv,hashlib,json,math
from pathlib import Path

ARMS=("shape_seg_co","cellpose_sam","microsam","organoid_id","stardist","shape","llm","react")
IDENTITY=("image_sha256","ground_truth_sha256","image_height_px","image_width_px","ground_truth_cell_count")
METRICS=("ap50","ap75","map_50_95")
FROZEN_MANIFEST_SHA="bbafaee99e0b918b6fc16b3b59eaaacbd82ba4a848004cc4423df79de0dec5d8"
HF_REVISION="cd0ab63f74fc304d481dad55912cc6afca00e780"
PREVIOUS_REVISION="67ed1fd763820ce46fc9551723dcb71258157046"
PREVIOUS_MANIFEST_SHA="d42ec0c73b13818cf180732e4f69e86ddb5121c9cf9c0ed5d3c22bc5143f64da"


def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def truth(value):return str(value).lower() in ("true","1")
def load(path):
    with Path(path).open(encoding="utf-8",newline="") as f:rows=list(csv.DictReader(f))
    if len({r["image_id"] for r in rows})!=len(rows):raise ValueError("Duplicate image IDs")
    return rows
def number(row,key):
    value=row.get(key)
    if value in (None,""):return None
    value=float(value)
    if not math.isfinite(value):raise ValueError("Nonfinite metric")
    return value
def write_csv(path,rows):
    fields=sorted({k for r in rows for k in r})
    with Path(path).open("w",encoding="utf-8",newline="") as f:
        w=csv.DictWriter(f,fieldnames=fields,lineterminator="\n");w.writeheader();w.writerows(rows)


def compare(rows,baselines):
    """Never drop a recorded failure or turn an unrecorded prediction into a zero."""
    if len({r["image_id"] for r in rows})!=len(rows):raise ValueError("Duplicate cohort IDs")
    outputs={"shape_cohort_bounded":rows};summaries={};paired=[]
    for arm,source in baselines.items():
        byid={r["image_id"]:r for r in source};export=[]
        for current in rows:
            old=byid.get(current["image_id"])
            if old:
                if any(str(current[k])!=str(old[k]) for k in IDENTITY):raise ValueError("Image/GT identity mismatch: "+current["image_id"])
                if arm=="shape_seg_co" and current["checkpoint_sha256"]!=old["checkpoint_sha256"]:raise ValueError("CO checkpoint mismatch")
                value=dict(old)
            else:
                value={k:current[k] for k in ("image_id",)+IDENTITY}
                value.update(arm=arm,evaluation_status="not_evaluated",delivered="",error="No historical observation on this image")
            export.append(value)
            item=dict(image_id=current["image_id"],baseline=arm,baseline_delivered=value.get("delivered"),
                baseline_evaluation_status=value.get("evaluation_status"),shape_delivered=current["delivered"],
                shape_needs_review=current["needs_review"])
            for key in METRICS:
                a,b=number(current,key),number(value,key)
                item["shape_"+key]=a;item["baseline_"+key]=b
                item["delta_"+key]=None if a is None or b is None else a-b
            paired.append(item)
        outputs[arm]=export
    for arm,data in outputs.items():
        observed=[r for r in data if number(r,"map_50_95") is not None]
        comparisons=[p for p in paired if p["baseline"]==arm and p["delta_map_50_95"] is not None]
        summaries[arm]=dict(n_planned=len(rows),n_observed=len(observed),
            n_delivered=sum(truth(r.get("delivered")) for r in observed),
            n_recorded_failure=sum(not truth(r.get("delivered")) for r in observed),
            n_unobserved=len(rows)-len(observed),
            scores={k:sum(number(r,k) for r in observed)/len(observed) if observed else None for k in METRICS},
            paired_n=len(comparisons),wins=sum(p["delta_map_50_95"]>1e-9 for p in comparisons),
            ties=sum(abs(p["delta_map_50_95"])<=1e-9 for p in comparisons),
            losses=sum(p["delta_map_50_95"]< -1e-9 for p in comparisons))
    return outputs,summaries,paired


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--review-dir",type=Path,required=True)
    cli.add_argument("--baseline-dir",type=Path,required=True)
    cli.add_argument("--output-dir",type=Path,required=True)
    cli.add_argument("--previous-dir",type=Path,help="Frozen pre-optimization HF export, revision "+PREVIOUS_REVISION)
    args=cli.parse_args()
    if args.output_dir.exists():raise ValueError("Use a new comparison directory")
    manifest_path=args.baseline_dir/"manifest.json"
    if sha(manifest_path)!=FROZEN_MANIFEST_SHA:raise ValueError("Unexpected frozen baseline manifest")
    manifest=json.loads(manifest_path.read_text())
    baselines={}
    for arm in ARMS:
        path=args.baseline_dir/(arm+".csv")
        if sha(path)!=manifest["files"][path.name]["sha256"]:raise ValueError("Baseline export changed")
        baselines[arm]=load(path)
    previous_manifest=None
    if args.previous_dir:
        path=args.previous_dir/"publication_manifest.json"
        if sha(path)!=PREVIOUS_MANIFEST_SHA:raise ValueError("Unexpected previous bounded publication manifest")
        previous_manifest=json.loads(path.read_text())
    index=json.loads((args.review_dir/"review_index.json").read_text())
    result=dict(schema=1,baseline_hf_revision=HF_REVISION,baseline_manifest_sha256=FROZEN_MANIFEST_SHA,
        review_index_sha256=sha(args.review_dir/"review_index.json"),cohorts={},
        metrics="AP(t)=TP/(TP+FP+FN), t=.50:.05:.95; not confidence-ranked COCO AP",
        missing="Unobserved predictions stay missing; recorded failures retain original scores",
        timing="Historical arm runtimes and GPU models differ; no causal speedup comparison")
    if previous_manifest is not None:result.update(previous_hf_revision=PREVIOUS_REVISION,previous_publication_manifest_sha256=PREVIOUS_MANIFEST_SHA)
    args.output_dir.mkdir(parents=True)
    for split,expected in (("challenge",24),("pilot",64)):
        source=args.review_dir/split/"shape_cohort_bounded.csv";rows=load(source)
        summary=json.loads((args.review_dir/split/"summary.json").read_text())
        identity=json.loads((args.review_dir/split/"identity.json").read_text())
        if len(rows)!=expected or len(rows)!=summary["n_recorded"]:raise ValueError("Incomplete fixed cohort")
        if any(r["source_commit"]!=identity["source"]["commit"] or r["manifest_fingerprint"]!=identity["manifest"] for r in rows):raise ValueError("Cohort provenance differs")
        current_baselines=dict(baselines)
        if previous_manifest is not None:
            relative=split+"/shape_cohort_bounded.csv"
            path=args.previous_dir/relative
            if sha(path)!=previous_manifest["files"][relative]["sha256"]:raise ValueError("Previous bounded CSV changed")
            previous=load(path)
            if [r["image_id"] for r in previous]!=[r["image_id"] for r in rows]:raise ValueError("Previous cohort membership/order differs")
            if any(r["manifest_fingerprint"]!=identity["manifest"] for r in previous):raise ValueError("Previous cohort manifest differs")
            current_baselines["previous_bounded"]=previous
        outputs,scores,paired=compare(rows,current_baselines)
        folder=args.output_dir/split;folder.mkdir()
        for arm,data in outputs.items():write_csv(folder/(arm+".csv"),data)
        write_csv(folder/"paired.csv",paired)
        result["cohorts"][split]=dict(n=expected,source_commit=identity["source"]["commit"],
            manifest_fingerprint=identity["manifest"],source_csv_sha256=sha(source),
            summary=summary,arms=scores)
    result["files"]={str(p.relative_to(args.output_dir)).replace("\\","/"):sha(p) for p in args.output_dir.rglob("*.csv")}
    (args.output_dir/"comparison.json").write_text(json.dumps(result,indent=2,allow_nan=False)+"\n")
    print(json.dumps({s:v["arms"] for s,v in result["cohorts"].items()},indent=2))


if __name__=="__main__":main()

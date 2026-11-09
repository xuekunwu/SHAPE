#!/usr/bin/env python3
"""Run one pinned bounded cohort and verify completed replay without claiming biological quality."""
import argparse,hashlib,json,os,subprocess,sys,time,traceback
from pathlib import Path

STABLE=("n_planned","n_recorded","n_delivered","n_operational_success","n_needs_review","llm_calls","tokens","native_calls","tool_calls","tool_requests_started","unknown_usage","incomplete_audit_records","incomplete_tool_requests","unmeasured_interrupted_passes","ap50","ap75","map_50_95")


def check_complete(summary,n):
    for key in ("n_planned","n_recorded","n_delivered"):
        if summary[key]!=n:raise ValueError("Incomplete delivery: "+key)
    for key in ("unknown_usage","incomplete_audit_records","incomplete_tool_requests","unmeasured_interrupted_passes"):
        if summary[key]!=0:raise ValueError("Unconfirmed usage or completion: "+key)


def check_replay(before,after,csv_before,csv_after):
    if any(before[k]!=after[k] for k in STABLE):raise ValueError("Completed replay changed results or model costs")
    if csv_before!=csv_after:raise ValueError("Completed replay changed CSV bytes")


def main():
    cli=argparse.ArgumentParser(description=__doc__)
    cli.add_argument("--manifest",required=True,type=Path)
    cli.add_argument("--run-dir",required=True,type=Path)
    cli.add_argument("--expected-commit",required=True)
    cli.add_argument("--receipt",required=True,type=Path)
    cli.add_argument("--model-path",required=True,type=Path)
    cli.add_argument("--n-expected",required=True,type=int)
    cli.add_argument("--min-map",type=float,help="Explicit development regression floor; not a biological quality gate")
    a=cli.parse_args()
    if not os.environ.get("SLURM_JOB_ID"):raise RuntimeError("Run inside Slurm")
    if a.receipt.exists() or a.run_dir.exists():raise ValueError("Use fresh run and receipt paths")
    cmd=[sys.executable,str(Path(__file__).with_name("run_organoid_bounded.py")),"--manifest",str(a.manifest),
        "--run-dir",str(a.run_dir),"--model-path",str(a.model_path),"--expected-commit",a.expected_commit]
    receipt=dict(schema=1,source_commit=a.expected_commit,job_id=os.environ["SLURM_JOB_ID"],
        n_expected=a.n_expected,started_utc=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime()),
        technical_gate_passed=False,quality_gate_passed=False,min_map_regression_floor=a.min_map)
    rc=1
    try:
        with a.receipt.with_suffix(".inference.log").open("x") as f:code=subprocess.run(cmd,stdout=f,stderr=subprocess.STDOUT).returncode
        receipt["inference_exit_code"]=code
        if code not in (0,2):raise RuntimeError("Inference runtime failure; inspect inference log")
        before=json.loads((a.run_dir/"summary.json").read_text());check_complete(before,a.n_expected)
        receipt["before_resume_summary"]=before
        csv_before=(a.run_dir/"shape_cohort_bounded.csv").read_bytes()
        with a.receipt.with_suffix(".resume.log").open("x") as f:code=subprocess.run(cmd+["--resume"],stdout=f,stderr=subprocess.STDOUT).returncode
        receipt["completed_resume_exit_code"]=code
        if code not in (0,2):raise RuntimeError("Completed replay runtime failure")
        after=json.loads((a.run_dir/"summary.json").read_text())
        check_complete(after,a.n_expected);check_replay(before,after,csv_before,(a.run_dir/"shape_cohort_bounded.csv").read_bytes())
        receipt.update(technical_gate_passed=True,completed_resume_no_new_model_calls=True,
            csv_sha256=hashlib.sha256(csv_before).hexdigest(),summary=after,
            regression_gate_passed=a.min_map is None or after["map_50_95"]>=a.min_map)
        if not receipt["regression_gate_passed"]:raise ValueError("Development regression floor failed; do not advance")
        rc=0
    except Exception as exc:
        receipt["error"]=str(exc);traceback.print_exc()
    finally:
        receipt["finished_utc"]=time.strftime("%Y-%m-%dT%H:%M:%SZ",time.gmtime())
        a.receipt.write_text(json.dumps(receipt,indent=2)+"\n")
        print(json.dumps(receipt),flush=True)
    return rc

if __name__=="__main__":raise SystemExit(main())

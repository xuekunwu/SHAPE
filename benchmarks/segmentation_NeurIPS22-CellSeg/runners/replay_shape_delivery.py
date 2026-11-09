#!/usr/bin/env python3
"""Offline replay of the shape arm's delivery choice: what each delivery policy would have delivered.

For every shape image of the given run dirs, every segmenter call of its agentic VRR (the trials in
shape/<band>/<image>/verification_summary.json; their masks are the run's own files) is scored against the
ground truth (AP50, mAP@0.50:0.95). Each policy then picks one admissible call per image, from what the run
recorded; no tool or LLM is run again:

    current        the call the run delivered (the reference default unless the agent's choice had strictly
                   fewer measured problems)
    reference      the reference default (the system's reference call when one ran, else the default call)
    agent_first    the agent's first final answer (it was asked knowing the current rule, so an approximation
                   of the agent deciding alone); else the delivered call
    agent_assessed the admissible call the agent assessed best (severity none < minor < major < severe; ties:
                   the higher QC score); else the delivered call
    fewest_problems the admissible call with the fewest measured problems (ties: the higher QC score)
    defaults       the default call, the tool at its defaults
    oracle         the admissible call with the best AP50 (the upper bound of any choice among the calls)

usage: python runners/replay_shape_delivery.py --run-dir R1 [--run-dir R2 ...] --data-root DATA --out OUT.json
Writes OUT.json (per image: every call's scores and each policy's pick) and OUT.md (per policy: mean AP50 /
mAP per band and pooled, and wins / ties / losses against `current`). Labels are read only to score.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE))
import cellseg_arms as arms                 # noqa: E402
import run_cellpose_sam_only as score       # noqa: E402

BANDS = ("hard", "medium", "easy")
POLICIES = ("current", "reference", "agent_first", "agent_assessed", "fewest_problems", "defaults", "oracle")
_SEVERITY = {"none": 0, "minor": 1, "major": 2, "severe": 3}
_EPS = 1e-9


def call_names(details: dict) -> list[str]:
    """Trial index -> the call's name, from the record's rounds: "input", then each round's calls in order (the
    names the run used: "default", "reference", "agent_<n>", ..., or c01, c02, ... in records before the role
    names)."""
    return ["input"] + [c for calls in details.get("rounds") or [] for c in calls]


def _call(names: list[str], index: int) -> str:
    return names[index] if 0 <= index < len(names) else f"trial_{index}"


def _segmentation_stage(vs: list) -> dict | None:
    for group in vs or []:
        for pi in (group.get("per_image") or [group]):
            for stage in pi.get("stages") or []:
                if stage.get("stage") == "segmentation":
                    return stage
    return None


def score_calls(stage: dict, gt: np.ndarray) -> dict:
    """call -> {ap50, map_50_95, qc_score, n_problems, severity, error}, for every trial after the input."""
    d = stage.get("details") or {}
    problems = d.get("measured_problems") or {}
    assessments = d.get("assessments") or {}
    names, out = call_names(d), {}
    for i, t in enumerate(stage.get("trials") or []):
        if i == 0:
            continue
        call = _call(names, i)
        r = t.get("result") or {}
        row = {"qc_score": t.get("score"), "n_problems": len(problems.get(call) or []),
               "severity": (assessments.get(call) or {}).get("severity"), "error": None}
        try:
            m = score.evaluate_instance_segmentation(arms.load_mask(r["mask_path"]), gt)
            row.update(ap50=m["ap50"], map_50_95=m["map_50_95"])
        except Exception as exc:
            row.update(ap50=None, map_50_95=None, error=f"{type(exc).__name__}: {exc}")
        out[call] = row
    return out


def _best(calls: dict, pool: list, key) -> str | None:
    pool = [c for c in pool if c in calls]
    return min(pool, key=key) if pool else None


def picks(stage: dict, calls: dict) -> dict:
    """policy -> the call it delivers (None when the image has no scorable call)."""
    d = stage.get("details") or {}
    selected, names = d.get("selected"), call_names(d)
    defaults = names[1] if len(names) > 1 and names[1] in calls else None     # round 1: the tool at its defaults
    delivered = _call(names, selected) if isinstance(selected, int) else defaults
    admissible = [c for c in (d.get("admissible") or list(calls)) if c in calls] or list(calls)
    reference = (d.get("reference_default") or {}).get("call") or defaults
    answers = d.get("final_answers") or []
    first = answers[0].get("call") if answers and isinstance(answers[0], dict) else None

    def qc(c):
        return -(calls[c].get("qc_score") or 0.0)

    assessed = [c for c in admissible if calls[c].get("severity") in _SEVERITY]
    out = {
        "current": delivered,
        "reference": reference if reference in calls else delivered,
        "agent_first": first if first in calls else delivered,
        "agent_assessed": (_best(calls, assessed, lambda c: (_SEVERITY[calls[c]["severity"]], qc(c)))
                           or delivered),
        "fewest_problems": _best(calls, admissible, lambda c: (calls[c]["n_problems"], qc(c))) or delivered,
        "defaults": defaults or delivered,
        "oracle": _best(calls, admissible, lambda c: -(calls[c].get("ap50") or 0.0)) or delivered,
    }
    return out


def replay(run_dirs: list[Path], data_root: Path) -> dict:
    rows = []
    for run in run_dirs:
        cp = {}
        for band in BANDS:
            f = run / "cellpose_sam" / band / "records.jsonl"
            if f.is_file():
                for ln in f.read_text(encoding="utf-8").splitlines():
                    if ln.strip():
                        r = json.loads(ln)
                        cp[r["image"]] = r.get("ap50")
        for vs_path in sorted(run.glob("shape/*/*/verification_summary.json")):
            band, image = vs_path.parent.parent.name, vs_path.parent.name
            stage = _segmentation_stage(json.loads(vs_path.read_text(encoding="utf-8")))
            if stage is None:
                continue
            gt = score.load_label_mask(data_root / "labels" / f"{image}_label.tiff")
            calls = score_calls(stage, gt)
            if not calls:
                continue
            chosen = picks(stage, calls)
            rows.append({"run": run.name, "band": band, "image": image, "calls": calls, "picks": chosen,
                         "cellpose_sam_ap50": cp.get(image),
                         "ap50": {p: (calls.get(c) or {}).get("ap50") for p, c in chosen.items()},
                         "map_50_95": {p: (calls.get(c) or {}).get("map_50_95") for p, c in chosen.items()}})
    return {"run_dirs": [str(r) for r in run_dirs], "n_images": len(rows), "rows": rows,
            "summary": summarize(rows)}


def summarize(rows: list[dict]) -> dict:
    out = {}
    for scope in list(BANDS) + ["all"]:
        rs = [r for r in rows if scope == "all" or r["band"] == scope]
        if not rs:
            continue
        block = {"n": len(rs)}
        for p in POLICIES:
            a = [r["ap50"][p] for r in rs if r["ap50"][p] is not None]
            m = [r["map_50_95"][p] for r in rs if r["map_50_95"][p] is not None]
            cur = [(r["ap50"][p], r["ap50"]["current"]) for r in rs
                   if r["ap50"][p] is not None and r["ap50"]["current"] is not None]
            block[p] = {"ap50": float(np.mean(a)) if a else None, "map_50_95": float(np.mean(m)) if m else None,
                        "vs_current": {"wins": sum(1 for x, y in cur if x > y + _EPS),
                                       "ties": sum(1 for x, y in cur if abs(x - y) <= _EPS),
                                       "losses": sum(1 for x, y in cur if x < y - _EPS)}}
        cp = [r["cellpose_sam_ap50"] for r in rs if r["cellpose_sam_ap50"] is not None]
        block["cellpose_sam_ap50"] = float(np.mean(cp)) if cp else None
        out[scope] = block
    return out


def markdown(result: dict) -> str:
    lines = ["# shape delivery replay", "", f"{result['n_images']} shape images from: "
             + ", ".join(Path(r).name for r in result["run_dirs"]), ""]
    for scope, block in result["summary"].items():
        lines += [f"## {scope} (n = {block['n']}; cellpose_sam AP50 {block['cellpose_sam_ap50']:.3f})"
                  if block["cellpose_sam_ap50"] is not None else f"## {scope} (n = {block['n']})", "",
                  "| policy | AP50 | mAP@0.50:0.95 | vs current: wins / ties / losses |", "|---|---|---|---|"]
        for p in POLICIES:
            b = block[p]
            v = b["vs_current"]
            fmt = (lambda x: "n/a" if x is None else f"{x:.3f}")
            lines.append(f"| {p} | {fmt(b['ap50'])} | {fmt(b['map_50_95'])} | {v['wins']} / {v['ties']} / {v['losses']} |")
        lines.append("")
    return "\n".join(lines)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--run-dir", type=Path, action="append", required=True)
    ap.add_argument("--data-root", type=Path, required=True, help="holds labels/<image>_label.tiff")
    ap.add_argument("--out", type=Path, required=True, help="OUT.json (and OUT.md beside it)")
    args = ap.parse_args()
    result = replay(args.run_dir, args.data_root)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(result, indent=1), encoding="utf-8")
    md = markdown(result)
    args.out.with_suffix(".md").write_text(md, encoding="utf-8")
    print(md)
    print(f"[replay] {result['n_images']} images -> {args.out}")


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Agent efficiency metrics for segmentation runs -- compare different agents on a test set.

These metrics weigh ACCURACY GAIN against COST (tool calls, tokens, time). They are computed
from a per-image RECORD; the loop's per-call AP50 trajectory is scored offline against the GT
(the agent never sees GT live).

Task 038: `call_deltas`/`retune_deltas`/`total_gain`/`compute_metrics` all take a `metric_key`
param (default `"ap50"`, byte-identical to before this existed) so the SAME engine serves any
other tool's per-call quality metric -- e.g. `crop_NeurIPS22-CellSeg/run_shape_crop.py` calls
`compute_metrics(records, metric_key="crop_quality_score")` rather than reimplementing this
aggregation for the cropper. The record schema below is the `metric_key="ap50"` instance;
another metric_key reads/writes `baseline_<metric_key>`/`call_<metric_key>` and the metric
names below with `ap50` substituted for `<metric_key>` (e.g. `crop_quality_score_gain_per_tool_call`).

Per-image RECORD schema (one dict per image, metric_key="ap50"):
    {
      "image_id":     str,
      "baseline_ap50": float|None,   # AP50 of the no-agent default (Cellpose alone)
      "call_ap50":    [float, ...],  # AP50 AFTER each segmentation tool call, in order
      "tokens":       int|None,      # total LLM tokens spent on this image
      "seconds":      float|None,    # wall-clock seconds for this image
    }
ΔAP50 of call i = call_ap50[i] - (call_ap50[i-1], or baseline_ap50 for the first call). A call
is "evaluable" when that ΔAP50 is computable (both AP50s known).

Metrics (direction = the better way; ↑ higher-better, ↓ lower-better):
  tool_calls_per_image     ↓  operation complexity (all segmenter calls incl. the initial one)
  retunes_per_image        ↓  agent interventions per image (calls AFTER the first)
  positive_tool_call_rate  ↑  fraction of RETUNES with ΔAP50 >= eps (effective decisions)
  wasted_tool_call_rate    ↓  fraction of RETUNES with |ΔAP50| < eps (no-op retunes)
  token_cost_per_image     ↓  LLM cost
  time_per_image_s         ↓  runtime efficiency
  ap50_gain_per_tool_call  ↑  total ΔAP50 / tool calls
  ap50_gain_per_1k_tokens  ↑  total ΔAP50 / tokens * 1000
  ap50_gain_per_minute     ↑  total ΔAP50 / minutes

Usage:
  # build a records file from a benchmark run folder (cellpose/<id>/ + shape/<id>/ + GT)
  python benchmarks/segmentation_metrics.py build \
      --run-root /scratch/.../<run> --label-dir /scratch/.../labels \
      --agent shape_v1 --out shape_v1_records.json
  # compare one or more agents' records and print the table
  python benchmarks/segmentation_metrics.py compare \
      shape_v1=shape_v1_records.json baseline=baseline_records.json --out metrics.json
"""

from __future__ import annotations

import argparse
import glob
import json
import os
import re
import sys
from pathlib import Path

EPS = 0.01   # |ΔAP50| below this is "no change" (positive/wasted threshold)

# metric -> True if HIGHER is better
_HIGHER_BETTER = {
    "tool_calls_per_image": False,
    "retunes_per_image": False,
    "positive_tool_call_rate": True,
    "wasted_tool_call_rate": False,
    "token_cost_per_image": False,
    "time_per_image_s": False,
    "ap50_gain_per_tool_call": True,
    "ap50_gain_per_1k_tokens": True,
    "ap50_gain_per_minute": True,
}
_METRIC_ORDER = list(_HIGHER_BETTER)


# ---- pure metric engine -----------------------------------------------------
#
# Task 038: generalized with a `metric_key` param (default "ap50", preserving the
# segmenter's own field names / output keys byte-for-byte) so the SAME engine serves
# any per-call quality metric -- e.g. `benchmarks/crop_NeurIPS22-CellSeg/run_shape_crop.py`
# calls this with metric_key="crop_quality_score" rather than reimplementing the same
# aggregation logic for the cropper. Record schema per metric_key: `baseline_<metric_key>`
# (no-agent reference score) / `call_<metric_key>` (the agent's own per-call trajectory).

def call_deltas(record, eps=EPS, metric_key="ap50"):
    """Per-call delta list for one image (only the EVALUABLE calls, i.e. those with a
    reference score before them)."""
    base = record.get(f"baseline_{metric_key}")
    calls = list(record.get(f"call_{metric_key}") or [])
    seq = ([base] + calls) if base is not None else calls
    return [seq[i] - seq[i - 1] for i in range(1, len(seq))]


def retune_deltas(record, metric_key="ap50"):
    """Per-RETUNE delta for one image: consecutive call-score diffs (a2 vs a1, a3 vs a2,
    ...). EXCLUDES the mandatory first tool call (a1 vs baseline ~= 0), so the
    positive/wasted rates measure the agent's actual DECISIONS, not the unavoidable
    initial call."""
    calls = list(record.get(f"call_{metric_key}") or [])
    return [calls[i] - calls[i - 1] for i in range(1, len(calls))]


def total_gain(record, metric_key="ap50"):
    """final score - baseline score for one image, or None if either is missing."""
    base = record.get(f"baseline_{metric_key}")
    calls = record.get(f"call_{metric_key}") or []
    final = calls[-1] if calls else base
    if base is None or final is None:
        return None
    return final - base


def compute_metrics(records, eps=EPS, metric_key="ap50"):
    """The efficiency metrics aggregated over a list of per-image records.

    A metric is None when the data it needs is absent for the whole set (e.g. no tokens),
    so partial logs still yield the metrics they CAN support. `metric_key` selects which
    per-call quality field to read (default "ap50" -- byte-identical output keys/values to
    before this param existed, so the segmenter's own callers are unaffected).
    """
    n = len(records)
    total_calls = sum(len(r.get(f"call_{metric_key}") or []) for r in records)
    # the agent's DECISIONS are the RETUNES (calls after the first); the initial tool call is
    # mandatory, not a decision -> the rate metrics below are computed over the retunes only.
    rdeltas = [d for r in records for d in retune_deltas(r, metric_key=metric_key)]
    retunes = len(rdeltas)
    positive = sum(1 for d in rdeltas if d >= eps)
    wasted = sum(1 for d in rdeltas if abs(d) < eps)

    gains = [g for g in (total_gain(r, metric_key=metric_key) for r in records) if g is not None]
    sum_gain = sum(gains)

    toks = [r.get("tokens") for r in records]
    have_tok = all(t is not None for t in toks) and bool(toks)
    sum_tok = sum(t for t in toks if t is not None)
    secs = [r.get("seconds") for r in records]
    have_sec = all(s is not None for s in secs) and bool(secs)
    sum_sec = sum(s for s in secs if s is not None)

    def ratio(num, den):
        return (num / den) if den else None

    return {
        "n_images": n,
        f"total_{metric_key}_gain": round(sum_gain, 4),
        "tool_calls_per_image": ratio(total_calls, n),
        "retunes_per_image": ratio(retunes, n),
        "positive_tool_call_rate": ratio(positive, retunes),
        "wasted_tool_call_rate": ratio(wasted, retunes),
        "token_cost_per_image": ratio(sum_tok, n) if have_tok else None,
        "time_per_image_s": ratio(sum_sec, n) if have_sec else None,
        f"{metric_key}_gain_per_tool_call": ratio(sum_gain, total_calls),
        # efficiency = gain / EXTRA tool calls (the retunes; the mandatory first call is
        # not "extra"). This is the "gain per retune" figure: how much AP each extra
        # tool call bought on average.
        f"{metric_key}_gain_per_extra_call": ratio(sum_gain, retunes),
        f"{metric_key}_gain_per_1k_tokens": ratio(sum_gain * 1000.0, sum_tok) if have_tok else None,
        f"{metric_key}_gain_per_minute": ratio(sum_gain, sum_sec / 60.0) if have_sec else None,
        "evaluable_retunes": retunes,
    }


def format_comparison(metrics_by_agent):
    """Aligned table: rows = metrics (with ↑/↓), columns = agents."""
    agents = list(metrics_by_agent)
    width = max([len(m) for m in _METRIC_ORDER] + [24])
    head = "metric".ljust(width + 3) + "  ".join(a.center(12) for a in agents)
    lines = [head, "-" * len(head)]
    for m in _METRIC_ORDER:
        arrow = "↑" if _HIGHER_BETTER[m] else "↓"
        cells = []
        for a in agents:
            v = metrics_by_agent[a].get(m)
            cells.append(("n/a" if v is None else f"{v:.4f}").center(12))
        lines.append(f"{(arrow + ' ' + m).ljust(width + 3)}" + "  ".join(cells))
    return "\n".join(lines)


# ---- build records from a benchmark run folder ------------------------------

def _import_scorer():
    sys.path.insert(0, os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))),
                                    "segmentation_NeurIPS22-CellSeg", "runners"))
    from run_shape_neurips_cellseg import evaluate_instance_segmentation
    return evaluate_instance_segmentation


def _try_index(path):
    m = re.search(r"__try(\d+)", os.path.basename(path))
    return int(m.group(1)) if m else -1


def _ordered_attempt_masks(image_dir):
    """Per-attempt masks (…__try{N}.npy) under an image's shape/<id>/ folder, in call order."""
    masks = [str(p) for p in Path(image_dir).rglob("*__try*.npy")]
    return [p for _, p in sorted((_try_index(p), p) for p in masks)]


def attempt_masks_for(final_mask_path):
    """Ordered per-attempt masks for ONE image: its `..._a{N}` siblings (a1, a2, a3 ...), in
    attempt order. The cell segmenter writes a DISTINCT file per VLRR attempt, so these are the
    real per-call trajectory. Falls back to [final] if the `_a{N}` tag is absent."""
    if not final_mask_path:
        return []
    p = str(final_mask_path)
    base = os.path.basename(p)
    m = re.search(r"^(.*_a)\d+(\.[^.]+)$", base)
    if not m:
        return [p]
    prefix, ext = m.group(1), m.group(2)
    sibs = []
    for f in glob.glob(os.path.join(os.path.dirname(p), f"{glob.escape(prefix)}*{ext}")):
        mm = re.search(r"_a(\d+)" + re.escape(ext) + r"$", os.path.basename(f))
        if mm:
            sibs.append((int(mm.group(1)), f))
    return [f for _, f in sorted(sibs)] or [p]


def _first_mask(image_dir):
    masks = sorted(str(p) for p in Path(image_dir).rglob("*.npy") if "__try" not in p.name)
    return masks[0] if masks else None


def build_records_from_run(run_root, label_dir, *, label_suffix="_label", label_ext=".tiff"):
    """Assemble per-image records from a benchmark run laid out as
    `<run_root>/{cellpose,shape}/<image_id>/...` (the param-restructured NeurIPS benchmark).

    - baseline_ap50: score the Cellpose-alone mask in `cellpose/<id>/`;
    - call_ap50: score each archived per-attempt mask in `shape/<id>/` (needs the per-attempt
      mask archiving). Falls back to the single final mask when no `__try` masks exist.
    - seconds: per-image `seconds` from `run_summary.json` (if present); tokens: not logged yet.
    """
    evaluate = _import_scorer()
    run_root = Path(run_root)
    secs_by_id = {}
    sp = run_root / "run_summary.json"
    if sp.exists():
        d = json.loads(sp.read_text(encoding="utf-8"))
        for im in (d.get("segmentation_evaluation", {}) or {}).get("per_image", []) or []:
            if im.get("image_name") is not None and im.get("seconds") is not None:
                secs_by_id[im["image_name"]] = im["seconds"]

    def _ap50(mask_path, gt_path):
        try:
            return float(evaluate(mask_path, gt_path).get("ap50"))
        except Exception:
            return None

    records = []
    shape_root = run_root / "shape"
    for image_dir in sorted(p for p in shape_root.iterdir() if p.is_dir()) \
            if shape_root.is_dir() else []:
        image_id = image_dir.name
        gt = Path(label_dir) / f"{image_id}{label_suffix}{label_ext}"
        if not gt.is_file():
            continue
        attempts = _ordered_attempt_masks(image_dir) or (
            [m] if (m := _first_mask(image_dir)) else [])
        call_ap50 = [a for a in (_ap50(m, gt) for m in attempts) if a is not None]
        base_dir = run_root / "cellpose" / image_id
        base_mask = _first_mask(base_dir) if base_dir.is_dir() else None
        records.append({
            "image_id": image_id,
            "baseline_ap50": _ap50(base_mask, gt) if base_mask else None,
            "call_ap50": call_ap50,
            "tokens": None,                       # TODO: needs per-image token logging
            "seconds": secs_by_id.get(image_id),
        })
    return records


# ---- CLI --------------------------------------------------------------------

def _cmd_build(args):
    records = build_records_from_run(
        args.run_root, args.label_dir,
        label_suffix=args.label_suffix, label_ext=args.label_ext)
    Path(args.out).write_text(json.dumps(
        {"agent": args.agent, "records": records}, indent=2, default=str), encoding="utf-8")
    m = compute_metrics(records)
    print(f"[{args.agent}] {m['n_images']} images, {m['evaluable_calls']} evaluable calls")
    print(format_comparison({args.agent: m}))
    print(f"Records -> {args.out}")


def _load_records(spec):
    """spec = 'name=path.json'; returns (name, records)."""
    name, _, path = spec.partition("=")
    if not path:
        name, path = Path(spec).stem, spec
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    return name, (d.get("records") if isinstance(d, dict) and "records" in d else d)


def _cmd_compare(args):
    metrics_by_agent = {}
    for spec in args.records:
        name, records = _load_records(spec)
        metrics_by_agent[name] = compute_metrics(records, eps=args.eps)
    print(format_comparison(metrics_by_agent))
    if args.out:
        Path(args.out).write_text(json.dumps(metrics_by_agent, indent=2), encoding="utf-8")
        print(f"\nMetrics -> {args.out}")


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__,
                                formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = p.add_subparsers(dest="cmd", required=True)

    b = sub.add_parser("build", help="Build a records file from a benchmark run folder.")
    b.add_argument("--run-root", required=True, type=Path)
    b.add_argument("--label-dir", required=True, type=Path)
    b.add_argument("--label-suffix", default="_label")
    b.add_argument("--label-ext", default=".tiff")
    b.add_argument("--agent", default="agent")
    b.add_argument("--out", required=True, type=Path)
    b.set_defaults(func=_cmd_build)

    c = sub.add_parser("compare", help="Compare agents' records and print the metric table.")
    c.add_argument("records", nargs="+", help="name=records.json (or just records.json)")
    c.add_argument("--eps", type=float, default=EPS)
    c.add_argument("--out", type=Path)
    c.set_defaults(func=_cmd_compare)

    args = p.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()

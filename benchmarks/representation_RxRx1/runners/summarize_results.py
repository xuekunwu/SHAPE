"""Offline summary of a scored RxRx1 comparison from the per-arm score cache: no re-embedding and no refit.

Every arm is scored once into the cache by the `score-arm` stage (score.py); this script reads the newest cache entry
of each arm and table, computes the paired bands from the cached per-siRNA APs and the probe's fixed bootstrap draws,
and lists the tuning trajectory of the SHAPE and ReAct runs with each call's run-prep well mAP (`score.diagnostic`,
which reads each call's own h5ad):

    python -m benchmarks.representation_RxRx1.runners.summarize_results \
        --root <results root>/representation_RxRx1/cache/HEPG2 --out <dir> --shape-run <shape run dir> --react-run <react run dir> --llm-run <llm run dir> [--title TEXT]

--root is the cell-type output root (it holds score_cache/, prep/ and private/). Writes <out>/compare/: summary.md,
scores.csv (well mAP per arm, table and group), paired_vs_shape.csv, probe.csv and calls.csv. The arms and the pairs
are `ARMS` and `PAIRS`; an arm without a cache entry is left out. It reads the labels (`score.read_labels`), so like
score.py it runs only where the private labels are. The reported comparison's copy of this script (rxrx_summary.py in
results/final-7arm-hepg2-a21-51ce0c5c-dinov3-a1fba795/) is the exact one that wrote that folder, with its paths fixed
in the code; this file is the same computation with the paths as arguments.
"""
import argparse
import csv
import glob
import json
import os
from pathlib import Path

import numpy as np

from benchmarks.representation_RxRx1.runners import prepare, score

ARMS = ["frozen", "dinov3", "default_s0", "openphenom", "cellprofiler", "llm", "shape", "react"]
PAIRS = [("shape", "dinov3"), ("shape", "react"), ("shape", "frozen"), ("shape", "llm"), ("shape", "cellprofiler"),
         ("shape", "openphenom"), ("shape", "default_s0"), ("dinov3", "frozen"), ("react", "frozen")]
GROUPS = list(score.GROUPS) + list(score.AUDIT_GROUPS)
PROBE_COLUMNS = [("well", "knn", "balanced_accuracy"), ("well", "knn", "macro_f1"),
                 ("well", "logreg", "balanced_accuracy"), ("well", "logreg", "macro_f1"),
                 ("cell", "knn", "balanced_accuracy"), ("cell", "logreg", "balanced_accuracy")]
TITLE = "RxRx1 HEPG2: the 7-arm comparison (SHAPE A21 51ce0c5c; dinov3 control A23 a1fba795; ReAct A14 f080a296)"


def latest_entries(cache: Path) -> dict:
    """{(arm, table): decoded entry}: the newest cache entry per arm and table."""
    found = {}
    for path in sorted(glob.glob(str(cache / "*.json")), key=os.path.getmtime):
        try:
            stored = json.loads(Path(path).read_text(encoding="utf-8"))
        except ValueError:
            continue
        arm, table = stored.get("arm"), (stored.get("key") or {}).get("table")
        if arm in ARMS and table and "evaluation" in stored:
            found[(arm, table)] = score._decode(stored)
    return found


def main(root, out, shape_run, react_run, llm_run, title=TITLE):
    root = Path(root)
    out = Path(out) / "compare"
    out.mkdir(parents=True, exist_ok=True)
    entries = latest_entries(root / score.SCORE_CACHE)
    md = [f"# {title}", "",
          f"SHAPE run `{Path(shape_run).name}`; ReAct run `{Path(react_run).name}`; llm reused `{Path(llm_run).name}`.",
          "Scores from the per-arm score cache (each arm scored once by `score-arm`); the paired bands below are "
          "computed from the cached per-siRNA APs and the probe's fixed bootstrap draws (no refit). n = 1 run.", ""]
    rows, paired_rows, probe_rows = [], [], []
    for table in score.TABLES:
        tabs = {a: entries[(a, table)]["evaluation"] for a in ARMS if (a, table) in entries}
        md += [f"## {table} table: well mAP", "", "| arm | " + " | ".join(GROUPS) + " |",
               "| --- |" + " --- |" * len(GROUPS)]
        for arm in ARMS:
            if arm in tabs:
                vals = [(tabs[arm].get(g) or {}).get("value") for g in GROUPS]
                md.append(f"| {arm} | " + " | ".join("-" if v is None else f"{v:.4f}" for v in vals) + " |")
                rows += [{"table": table, "arm": arm, "group": g, "well_map": v} for g, v in zip(GROUPS, vals)]
        chance = [(tabs["frozen"].get(g) or {}).get("chance") for g in GROUPS]
        md.append("| chance | " + " | ".join("-" if v is None else f"{v:.4f}" for v in chance) + " |")
        md += ["", "| group | comparison | delta +- 2SE (paired over siRNA) | significant |",
               "| --- | --- | --- | --- |"]
        for g in score.READ:
            for a, b in PAIRS:
                ea, eb = (tabs.get(a) or {}).get(g), (tabs.get(b) or {}).get(g)
                if not ea or not eb:
                    continue
                unit_group = {**eb["unit_group"], **ea["unit_group"]}
                band = score.paired_band(eb["per_unit"], ea["per_unit"], unit_group)
                md.append(f"| {g} | {a}_vs_{b} | {band['delta']:+.4f} +- {band['band_2se']:.4f} | "
                          f"{band['significant']} |")
                paired_rows.append({"table": table, "group": g, "comparison": f"{a}_vs_{b}", **{
                    k: band.get(k) for k in ("delta", "band_2se", "significant", "n_units_paired")}})
        md.append("")
        # the probe: point, 95 % interval over the fixed bootstrap draws, paired delta vs SHAPE on the same draws
        # the columns of the probe runs the cache holds (the full-set preps skip the cell unit: score.PROBE_SKIPPED)
        held = {run for arm in ARMS for run in ((entries.get((arm, table)) or {}).get("probe") or {})}
        cols = [c for c in PROBE_COLUMNS if c[:2] in held]
        md += [f"### {table} table: siRNA probe across experiments (balanced accuracy / macro F1)", "",
               "| arm | " + " | ".join(f"{u} {c} {'BA' if m == 'balanced_accuracy' else 'F1'}" for u, c, m in cols)
               + " |", "| --- |" + " --- |" * len(cols)]
        if held and not any(c[0] == "cell" for c in cols):
            md[-2:-2] = ["The cell unit is not probed for this prep (skipped: score.PROBE_SKIPPED).", ""]
        for arm in ARMS:
            probe = (entries.get((arm, table)) or {}).get("probe")
            if not probe:
                continue
            cells = []
            for unit, clf, metric in cols:
                r = probe[(unit, clf)]
                point, draws = r["point"][metric], np.asarray(r["draws"][metric])
                lo, hi = np.percentile(draws, [2.5, 97.5])
                cells.append(f"{point:.3f} [{lo:.3f}, {hi:.3f}]")
                probe_rows.append({"table": table, "arm": arm, "unit": unit, "classifier": clf, "metric": metric,
                                   "value": point, "ci_low": lo, "ci_high": hi})
            md.append(f"| {arm} | " + " | ".join(cells) + " |")
        shape_probe = (entries.get(("shape", table)) or {}).get("probe")
        if shape_probe:
            md += ["", "shape minus arm, paired over the same bootstrap draws (95 % interval):", ""]
            for arm in ("dinov3", "react", "frozen", "default_s0"):
                probe = (entries.get((arm, table)) or {}).get("probe")
                if not probe:
                    continue
                parts = []
                for unit, clf, metric in PROBE_COLUMNS[:3]:
                    d = (np.asarray(shape_probe[(unit, clf)]["draws"][metric])
                         - np.asarray(probe[(unit, clf)]["draws"][metric]))
                    pd = shape_probe[(unit, clf)]["point"][metric] - probe[(unit, clf)]["point"][metric]
                    lo, hi = np.percentile(d, [2.5, 97.5])
                    parts.append(f"{unit} {clf} {metric}: {pd:+.3f} [{lo:+.3f}, {hi:+.3f}]")
                md.append(f"- vs {arm}: " + "; ".join(parts))
        md.append("")
    # the tuning trajectories of the two agentic arms
    calls = []
    for arm, run in (("shape", shape_run), ("react", react_run)):
        ends, starts = {}, []
        with open(Path(run) / "logs" / "tool_calls.jsonl", encoding="utf-8") as handle:
            for line in handle:
                d = json.loads(line)
                (starts.append(d) if d["event"] == "start" else ends.__setitem__(d["call"], d))
        for s in starts:
            e = ends.get(s["call"], {})
            tool = e.get("tool") or {}
            training = tool.get("training") or {}
            calls.append({"arm": arm, "call": s["call"], "parent": s.get("parent"), "init": s.get("init"),
                          "changes": json.dumps({k: v for k, v in (s.get("set") or {}).items()}),
                          "val_proxy_comparison_draws": (e.get("metrics") or {}).get("sibling_site_map"),
                          "epochs_trained": training.get("epochs_trained"), "best_epoch": training.get("best_epoch"),
                          "status": e.get("status")})
    summary_prep, cells = prepare.load(root)
    sirna = score.read_labels(root, cells, summary_prep)
    for arm, run in (("shape", shape_run), ("react", react_run)):
        diag = score.diagnostic(cells, sirna, score.finished_calls(Path(run)), score.moves(Path(run)))
        target = {p["call"]: p["run_prep_well_map"] for p in diag["points"]}
        for c in calls:
            if c["arm"] == arm:
                c["run_prep_well_map"] = target.get(c["call"])
        md += [f"{arm}: Spearman rho (val proxy vs run-prep well mAP over the calls) = {diag['spearman_rho']}", ""]
    md += ["## Tuning trajectories", "",
           "| arm | call | parent | init | changes | val proxy | run-prep well mAP | epochs | best |",
           "| --- | --- | --- | --- | --- | --- | --- | --- | --- |"]
    for c in calls:
        proxy = "-" if c["val_proxy_comparison_draws"] is None else f"{c['val_proxy_comparison_draws']:.4f}"
        target = "-" if c.get("run_prep_well_map") is None else f"{c['run_prep_well_map']:.4f}"
        md.append(f"| {c['arm']} | {c['call']} | {c['parent']} | {c['init']} | `{c['changes'][:120]}` | {proxy} | "
                  f"{target} | {c['epochs_trained']} | {c['best_epoch']} |")
    for name, data in (("scores.csv", rows), ("paired_vs_shape.csv", paired_rows), ("probe.csv", probe_rows),
                       ("calls.csv", calls)):
        if data:
            with open(out / name, "w", newline="", encoding="utf-8") as h:
                w = csv.DictWriter(h, fieldnames=list(data[0]))
                w.writeheader()
                w.writerows(data)
    (out / "summary.md").write_text("\n".join(md) + "\n", encoding="utf-8")
    print(out / "summary.md")
    return out


def cli(argv=None):
    p = argparse.ArgumentParser(prog="summarize_results", description=__doc__.split("\n\n")[0])
    p.add_argument("--root", required=True, help="the cell-type output root (score_cache/, prep/, private/)")
    p.add_argument("--out", required=True, help="the results folder; <out>/compare/ is written")
    p.add_argument("--shape-run", required=True, help="the SHAPE run directory")
    p.add_argument("--react-run", required=True, help="the ReAct run directory")
    p.add_argument("--llm-run", required=True, help="the llm run directory (named in the header only)")
    p.add_argument("--title", default=TITLE, help="the summary's heading")
    args = p.parse_args(argv)
    return main(args.root, args.out, args.shape_run, args.react_run, args.llm_run, title=args.title)


if __name__ == "__main__":
    cli()

"""python -m benchmarks.full_HCMA.harness.cli <command> ...

Stages, in the order they run:
  prepare   anonymized prep ($ROOT/prep) + private labels, folds, scorable set ($ROOT/private)
  subset    a named subset of the prep ($ROOT/prep_<name>, e.g. pairs: 2 images per cell type from
            2 micrographs; selection in $ROOT/private/SUBSET_<name>.json)
  verify-prep   a prep against its recorded sha256s (exit 1 on any change)
  class-map     the scorer's morph_classes.csv against a prep's cell_states (score.class_map_check;
                exit 1 = NO_CLASS_MAP: a run on that prep could not be scored)
  assets    pinned model caches ($ROOT/assets): DINOv3 hub dir, HF-layout weights, cpsam
  smoke     the default chain on the first N images of a prep ($ROOT/smoke/<prep-name>/<metadata>/<utc>/),
            scored, traced, exported
  control   the default chain on every image of a prep ($ROOT/control/<prep-name>/<metadata>/<gpu-slug>/),
            scored per seed
  setup     an agent run's workspace (TASK.md, data, tools, _config.json, TOOLS_SHA256.json)
  agent     the headless Claude loop
  shape     the SHAPE arm (harness/shape_arm.py): SHAPE's VRR loops choose the chain's parameters, every delivered
            call through the tool CLIs; reasoner SHAPE_LLM_MODEL (--model), --images N = a smoke on whole type pairs
  score     a run (--run) or an AnnData of another arm (--h5ad)
  trace, export   the run's trace.md / trace.json and its audit tarball
  process   process metrics of a run (PROCESS.json / PROCESS.md: adaptivity, substantive lever coverage,
            note traceability, cost, failure taxonomy, end state, completion claims)
  table     the BIABench-style summary over run dirs (TABLE.md, table.csv, runs.csv, pareto.png)
  attribute a run's difference to the control split into objects (segment + crop + exclusions) and
            representation (embed + batch + cluster): a 2 x 2 factorial with two replay chains on the GPU
            (ATTRIBUTION.json / ATTRIBUTION.md / attribution.png)
A prep is named by its directory: `full` = $ROOT/prep, `<name>` = $ROOT/prep_<name>. The metadata level
(--metadata) is `full` (the prep's technical metadata) or `none` (workspace images.csv without
parent_id, source, image_modality, magnification: workspace.blank_metadata).
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def _seeds(text: str) -> list[int]:
    return [int(s) for s in text.split(",") if s.strip()]


def main(argv: list[str] | None = None) -> None:
    p = argparse.ArgumentParser(prog="full_HCMA")
    sub = p.add_subparsers(dest="cmd", required=True)
    pr = sub.add_parser("prepare")
    pr.add_argument("--hcma-root", required=True, help="dir with images/ and HCMA_metadata.csv")
    pr.add_argument("--root", required=True)
    pr.add_argument("--seed", type=int, default=0)
    su = sub.add_parser("subset")
    su.add_argument("--root", required=True)
    su.add_argument("--name", default="pairs")
    su.add_argument("--per-class", type=int, default=2)
    su.add_argument("--seed", type=int, default=0)
    vp = sub.add_parser("verify-prep")
    vp.add_argument("--root", required=True)
    vp.add_argument("--prep-name", required=True, help="full or a subset name (pairs)")
    cm = sub.add_parser("class-map")
    cm.add_argument("--root", required=True)
    cm.add_argument("--prep-name", required=True, help="full or a subset name (pairs)")
    cm.add_argument("--map", default=None, help="the class map (default: the versioned morph_classes.csv)")
    at = sub.add_parser("assets")
    at.add_argument("--root", required=True)
    at.add_argument("--dinov3-repo", required=True)
    at.add_argument("--weights", required=True)
    at.add_argument("--cpsam", default=str(Path.home() / ".cellpose" / "models" / "cpsam"),
                    help="cpsam weights to copy (cellpose downloads them when absent)")
    st = sub.add_parser("setup")
    st.add_argument("--run", required=True)
    st.add_argument("--root", required=True)
    st.add_argument("--repo", required=True)
    st.add_argument("--budget-hours", type=float, default=None)
    st.add_argument("--gpu-slots", type=int, default=None, help="default: hardware.gpu_slots()")
    st.add_argument("--template", default=None)
    st.add_argument("--prep", default=None, help="prep dir (default: $ROOT/prep_pairs, as smoke / control)")
    st.add_argument("--metadata", choices=("full", "none"), default="full",
                    help="none: images.csv without parent_id, source, image_modality, magnification")
    ag = sub.add_parser("agent")
    ag.add_argument("--run", required=True)
    ag.add_argument("--model", default=None)
    ag.add_argument("--budget-hours", type=float, required=True)
    ag.add_argument("--claude-settings", default=None)
    ag.add_argument("--add-dir", action="append", default=[])
    sh = sub.add_parser("shape")
    sh.add_argument("--root", required=True)
    sh.add_argument("--repo", required=True)
    sh.add_argument("--out", required=True, help="the run directory (a new one)")
    sh.add_argument("--prep-name", default="pairs", help="full or a subset name")
    sh.add_argument("--metadata", choices=("full", "none"), default="full")
    sh.add_argument("--model", default=None, help="the reasoner's gateway model id (default: $SHAPE_LLM_MODEL)")
    sh.add_argument("--budget-hours", type=float, default=None, help="default: $BUDGET_HOURS, else 4")
    sh.add_argument("--images", type=int, default=None, help="a smoke on the prep's first whole type pairs")
    for name in ("control", "smoke"):
        c = sub.add_parser(name)
        c.add_argument("--root", required=True)
        c.add_argument("--repo", required=True)
        c.add_argument("--seeds", type=_seeds, default=[0], help="comma-separated, e.g. 0,1,2")
        c.add_argument("--prep-name", default="pairs", help="full or a subset name")
        c.add_argument("--metadata", choices=("full", "none"), default="full")
        c.add_argument("--images", type=int, default=8 if name == "smoke" else None,
                       help="only the first N image_ids (writes under $ROOT/smoke/)")
        c.add_argument("--out", default=None)
    sc = sub.add_parser("score")
    which = sc.add_mutually_exclusive_group(required=True)
    which.add_argument("--run")
    which.add_argument("--h5ad")
    sc.add_argument("--root", required=True)
    sc.add_argument("--forbidden", action="append", default=[], help="leak-audit needle (run mode)")
    sc.add_argument("--all-candidates", action="store_true",
                    help="run mode: also score every embedding and cluster candidate of the ledger")
    sc.add_argument("--state-key", default=None, help="--h5ad: obs column of the states")
    sc.add_argument("--embedding-key", default=None, help="--h5ad: obsm key of the embedding")
    sc.add_argument("--out", default=None, help="--h5ad: output dir (default: next to the h5ad)")
    sc.add_argument("--prep-name", default="pairs", help="--h5ad: the prep the arm ran on")
    sc.add_argument("--metadata", choices=("full", "none"), default="full", help="--h5ad: the arm's metadata level")
    for name in ("trace", "export"):
        sub.add_parser(name).add_argument("--run", required=True)
    ps = sub.add_parser("process")
    ps.add_argument("--run", required=True)
    ps.add_argument("--out", default=None, help="output dir (default: the run dir)")
    tb = sub.add_parser("table")
    tb.add_argument("--runs", nargs="+", default=[], help="run dirs (each with RESULT.json)")
    tb.add_argument("--control-root", default=None, help="also every dir with a RESULT.json under this root")
    tb.add_argument("--out", required=True)
    ab = sub.add_parser("attribute")
    ab.add_argument("--run", required=True)
    ab.add_argument("--control", required=True, help="the zero-shot control run dir of the same prep / metadata / GPU")
    ab.add_argument("--out", required=True)
    ab.add_argument("--root", default=None, help="default: the run's prep dir's parent")
    ab.add_argument("--repo", default=None, help="default: the run's tools/_config.json repo")
    a = p.parse_args(argv)

    if a.cmd == "prepare":
        from .prepare import prepare

        prepare(Path(a.hcma_root), Path(a.root), seed=a.seed)
    elif a.cmd == "subset":
        from .subset import subset

        subset(Path(a.root), a.name, per_class=a.per_class, seed=a.seed)
    elif a.cmd == "verify-prep":
        from .subset import prep_problems

        problems = prep_problems(Path(a.root), a.prep_name)
        print("\n".join(problems[:20]) or f"PREP {a.prep_name} verifies")
        if problems:
            raise SystemExit(1)
    elif a.cmd == "class-map":
        from .score import class_map_check

        check = class_map_check(Path(a.root), a.prep_name, Path(a.map) if a.map else None)
        print("\n".join(check["problems"]) if check["problems"] else
              f"CLASS_MAP ok for prep {a.prep_name}: {check['path']} "
              f"(sha256 {check['sha256']}, {check['n_rows']} rows)")
        if not check["ok"]:
            raise SystemExit(1)
    elif a.cmd == "assets":
        from .assets import build

        build(Path(a.root), dinov3_repo=Path(a.dinov3_repo), weights=Path(a.weights), cpsam=Path(a.cpsam))
    elif a.cmd == "setup":
        from .hardware import gpu_slots
        from .subset import prep_dir
        from .workspace import setup_run

        print(setup_run(Path(a.run), Path(a.root), Path(a.repo), budget_hours=a.budget_hours,
                        gpu_slots=a.gpu_slots if a.gpu_slots is not None else gpu_slots(),
                        template=Path(a.template) if a.template else None,
                        prep=Path(a.prep) if a.prep else prep_dir(Path(a.root), "pairs"), metadata=a.metadata))
    elif a.cmd == "agent":
        from .agents import run_loop

        summary = run_loop(Path(a.run), budget_hours=a.budget_hours, model=a.model, add_dirs=a.add_dir,
                           claude_settings=a.claude_settings)
        print(json.dumps(summary, indent=1))
    elif a.cmd == "shape":
        import os

        from .shape_arm import BUDGET_DEFAULT, MODEL_ENV, run_shape

        budget = a.budget_hours if a.budget_hours is not None else float(os.environ.get("BUDGET_HOURS") or BUDGET_DEFAULT)
        run = run_shape(Path(a.root), Path(a.repo), Path(a.out), model=a.model or os.environ.get(MODEL_ENV),
                        budget_hours=budget, prep_name=a.prep_name, metadata=a.metadata, n_images=a.images)
        print(f"SHAPE_RUN={run}")
    elif a.cmd in ("control", "smoke"):
        from .control import run_control

        run = run_control(Path(a.root), Path(a.repo), seeds=a.seeds, n_images=a.images,
                          out=Path(a.out) if a.out else None, prep_name=a.prep_name, metadata=a.metadata)
        print(f"{a.cmd.upper()}_RUN={run}")
        if a.cmd == "smoke":         # the agent-run machinery on the real tool outputs
            from .coverage import describe, final_check
            from .export import export
            from .trace_md import write_all

            print(describe(final_check(run)))
            write_all(run)
            export(run)
    elif a.cmd == "score":
        from . import score

        if a.run:
            result = score.score_run(Path(a.run), Path(a.root), forbidden=a.forbidden,
                                    all_candidates=a.all_candidates)
        else:
            if not (a.state_key and a.embedding_key):
                p.error("score --h5ad needs --state-key and --embedding-key")
            h5ad = Path(a.h5ad)
            out = Path(a.out) if a.out else h5ad.parent / f"score_{h5ad.stem}"
            result = score.score_h5ad(h5ad, Path(a.root), out, state_key=a.state_key,
                                      embedding_key=a.embedding_key, prep_name=a.prep_name, metadata=a.metadata)
        print(json.dumps({k: v for k, v in result.items() if isinstance(v, (str, int, float, bool)) or v is None},
                         indent=1))
    elif a.cmd == "trace":
        from .trace_md import write_all

        write_all(Path(a.run))
        print(Path(a.run) / "trace.md", Path(a.run) / "trace.json")
    elif a.cmd == "process":
        from .process import write

        out = Path(a.out) if a.out else Path(a.run)
        write(Path(a.run), out)
        print(out / "PROCESS.json", out / "PROCESS.md")
    elif a.cmd == "table":
        from .table import table

        table([Path(r) for r in a.runs], Path(a.out), Path(a.control_root) if a.control_root else None)
        print(Path(a.out) / "TABLE.md")
    elif a.cmd == "attribute":
        from .attribute import attribute

        record = attribute(Path(a.run), Path(a.control), Path(a.out), root=Path(a.root) if a.root else None,
                           repo=Path(a.repo) if a.repo else None)
        print(Path(a.out) / "ATTRIBUTION.md")
        print(json.dumps({m: b.get("decomposition") for m, b in record["metrics"].items()}, indent=1))
    elif a.cmd == "export":
        from .export import export

        export(Path(a.run))


if __name__ == "__main__":
    main()

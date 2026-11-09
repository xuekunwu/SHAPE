"""The stage CLI of the RxRx1 representation benchmark: the seven arms (frozen, dinov3, shape, react, llm, openphenom,
cellprofiler) on Cell_State_Analyzer_Tool and their scoring.

    python benchmarks/representation_RxRx1/run_rxrx1.py <stage> [--prep-set v3|full_v4|full_v5] ...
        prepare  [--cell-type HEPG2] [--data-root DIR] [--output-root DIR] [--dinov3-repo DIR --weights PTH]
        assets   [--output-root DIR] [--openphenom-source DIR]                (a login node: it downloads)
        control  [--cell-type HEPG2] [--output-root DIR] [--seeds 0,1,2]
        external [--cell-type HEPG2] [--output-root DIR] [--arms openphenom,cellprofiler] [--workers N]
                 [--input-size 256] [--fit resize|pad] [--batch-size 64] [--limit N]
        shape    --run RUN [--cell-type HEPG2] [--output-root DIR] [--model gpt-5.6-luna] [--max-rounds 8]
                 [--no-engine] [--resume] [--seed 0] [--early-stop]  
        llm      --run RUN [--cell-type HEPG2] [--output-root DIR] [--model gpt-5.6-luna] [--max-rounds 8] [--resume]
        react    --run RUN [--cell-type HEPG2] [--output-root DIR] [--model gpt-5.6-luna] [--resume]
        score-arm --arm ARM [--run RUN] [--cell-type HEPG2] [--output-root DIR]     (one arm into the score cache)
        score    --run RUN [--cell-type HEPG2] [--output-root DIR] [--llm-run LLM_RUN] [--react-run REACT_RUN]
        finish   --run RUN [--cell-type HEPG2] [--output-root DIR] [--llm-run LLM_RUN] [--react-run REACT_RUN]
                                                                         (score + export)

prepare (CPU) turns the task-116 v3 preps of the cell type into tool inputs under <output root>/<cell type>/ and
stages the pinned DINOv3 assets under <output root>/assets/; each part is kept when it exists; it writes no crop
file. assets (a login node) stages the pinned OpenPhenom snapshot under <output root>/assets/openphenom/ (external.py).
external computes the label-free external arms (openphenom on the GPU, 256 px only; cellprofiler on CPU workers),
raw, into <output root>/<cell type>/external/<arm>/ (with --limit N: a pilot on the first N cells into
external_pilot/<arm>[_<fit>]_n<N>/, one per setting, never overwritten, never scored). control, shape and score (GPU)
give the tool the FULL images.npy as its packed-array input (crop_array) and load DINOv3 offline from those assets.
control runs frozen and dinov3 (config.py); shape runs the SHAPE arm into RUN (a new directory); llm and react
run the LLM comparison arms (the same model without SHAPE's loop) into RUN (a new directory): llm (llm_arms.py,
contract amendment A5) has the model write a hand-crafted feature module and runs it on every FULL cell (no analyzer,
no DINOv3, no training), react (react_arm.py, benchmarks/_shared/REACT_ARM_CONTRACT.md, amendment A14) is the shared
ReAct runner on the analyzer: SHAPE's task text, SHAPE's lever set as name / type / default, the tool's own outputs,
at most 8 rounds after trial 0 (the frozen default call), with the same dataset-derived fixed per-call epoch cap as
SHAPE (A6); score embeds every FULL cell per arm (with --llm-run / --react-run also those runs' deliveries: react's
call, llm's own features) and writes RUN/RESULT.json / RESULT.md (the native and the matched table), every arm's scores
read from (or computed into) the per-arm score cache <output root>/<cell type>/score_cache/ (score.py); score-arm
scores one arm into that cache as soon as it exists and prints its headline (ARM: frozen, dinov3, openphenom,
cellprofiler; llm, shape or react with RUN = that arm's run directory);
finish also
writes RUN/EXPORT_<run>.tar.gz (no arrays, credentials masked), and the same export into each given llm / react
run directory (<that run>/EXPORT_<that run>.tar.gz). With RXRX_EXPECTED_COMMIT set, every stage refuses a
checkout at another commit. `shape --no-engine` runs the deterministic fallback (a smoke only). `--resume` (contract
amendment A8, the VM mode: RXRX_PLATFORM=vm) resumes the unfinished run in RUN in place (resume.py): shape and react by
replay, llm by restarting its current attempt.

`--prep-set` (every stage; config.py PREP_SETS): v3 (the default, the task-116 preps; --cell-type HEPG2, HUVEC or RPE)
or full_v4 (the full-set preps; also U2OS), whose outputs go to <output root>/<cell type>_full_v4/ (full_v5, their
well-level resplit, alike: <cell type>_full_v5/). The three LLM
arms (shape, react, llm) get the same reasoner (`reasoner`): REASONING_EFFORT high and MAX_COMPLETION_TOKENS.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

from benchmarks.representation_RxRx1.runners.config import DEFAULT, PREP_SETS, BenchmarkConfig  # noqa: E402
from benchmarks.representation_RxRx1.runners.external import (EXTERNAL_ARMS, OPENPHENOM_BATCH,  # noqa: E402
                                                               OPENPHENOM_FIT, OPENPHENOM_INPUT_SIZE)


def main(argv=None) -> None:
    p = argparse.ArgumentParser(prog="run_rxrx1")
    sub = p.add_subparsers(dest="cmd", required=True)
    stages = {name: sub.add_parser(name) for name in ("prepare", "assets", "control", "external", "shape", "llm",
                                                      "react", "score-arm", "score", "finish")}
    cell_types = sorted({cell_type for preps in PREP_SETS.values() for cell_type in preps})
    for name, sp in stages.items():
        sp.add_argument("--cell-type", default=DEFAULT.cell_type, choices=cell_types)
        sp.add_argument("--prep-set", default=DEFAULT.prep_set, choices=sorted(PREP_SETS),
                        help="config.py PREP_SETS: v3 (the task-116 preps), full_v4 or full_v5 (the full-set preps)")
        sp.add_argument("--output-root", default=DEFAULT.output_root)
        if name in ("shape", "llm", "react", "score", "finish"):
            sp.add_argument("--run", required=True, help="the run directory (new for shape, llm and react)")
        if name in ("llm", "react"):
            sp.add_argument("--model", default=DEFAULT.model)
        if name in ("shape", "llm", "react"):
            sp.add_argument("--resume", action="store_true", help="resume the unfinished run in RUN (A8, the VM)")
        if name == "score-arm":
            sp.add_argument("--arm", required=True, help="frozen, dinov3, dinov3_s<k>, openphenom, cellprofiler, llm, "
                                                         "shape or react")
            sp.add_argument("--run", help="the arm's run directory (llm, shape, react)")
        if name in ("score", "finish"):
            sp.add_argument("--llm-run", help="the llm arm's run directory (amendment A5), scored with this run")
            sp.add_argument("--react-run", help="the react arm's run directory (react_arm.py, A14), scored with this "
                                                "run")
    stages["prepare"].add_argument("--data-root", default=DEFAULT.data_root, help="the task-116 preps")
    stages["prepare"].add_argument("--dinov3-repo", help="the pinned DINOv3 checkout (when the assets are missing)")
    stages["prepare"].add_argument("--weights", help="the pinned DINOv3 ViT-B/16 weights (when the assets are missing)")
    stages["control"].add_argument("--seeds", default=",".join(map(str, DEFAULT.control_seeds)))
    stages["assets"].add_argument("--openphenom-source", help="a local copy of the OpenPhenom snapshot (else download)")
    stages["external"].add_argument("--arms", default=",".join(EXTERNAL_ARMS))
    stages["external"].add_argument("--workers", type=int, default=1, help="cellprofiler CPU processes")
    stages["external"].add_argument("--input-size", type=int, default=OPENPHENOM_INPUT_SIZE,
                                    help="openphenom input px (256 only; another value is refused)")
    stages["external"].add_argument("--fit", default=OPENPHENOM_FIT, choices=("resize", "pad"),
                                    help="how a 64 px crop enters openphenom")
    stages["external"].add_argument("--batch-size", type=int, default=OPENPHENOM_BATCH)
    stages["external"].add_argument("--limit", type=int, help="a pilot on the first N cells (external_pilot/)")
    stages["shape"].add_argument("--model", default=DEFAULT.model)
    stages["shape"].add_argument("--max-rounds", type=int, default=DEFAULT.max_rounds)
    stages["llm"].add_argument("--max-rounds", type=int, default=DEFAULT.max_rounds,
                               help="the llm arm's attempts: SHAPE's max_rounds (A5 as amended 2026-10-13)")
    stages["shape"].add_argument("--no-engine", action="store_true", help="deterministic fallback only (a smoke)")
    stages["shape"].add_argument("--seed", type=int, default=0, help="the analyzer's seed of every SHAPE call (A18)")
    stages["shape"].add_argument("--early-stop", action="store_true",
                                 help="SHAPE-ES: search-level and within-call early stopping (shape_arm.EARLY_STOP_*)")
    a = p.parse_args(argv)
    if a.cell_type not in PREP_SETS[a.prep_set]:
        p.error(f"--cell-type {a.cell_type} has no {a.prep_set} prep (choose from {', '.join(PREP_SETS[a.prep_set])})")

    from benchmarks.representation_RxRx1.runners import control
    from benchmarks.representation_RxRx1.runners.prepare import build_assets, check_assets, outside_repo, prepare

    expected = os.environ.get("RXRX_EXPECTED_COMMIT")
    if expected and control.git_commit() != expected:
        raise SystemExit(f"STALE_CODE: {REPO} is at {control.git_commit()}, not RXRX_EXPECTED_COMMIT {expected}")
    cfg = BenchmarkConfig(cell_type=a.cell_type, prep_set=a.prep_set, output_root=a.output_root,
                          data_root=getattr(a, "data_root", DEFAULT.data_root))
    output_root, root = outside_repo(Path(cfg.output_root)), cfg.root()

    if a.cmd == "prepare":
        full, run_prep = cfg.preps()
        if (root / "prep" / "PREPARE.json").exists() and (root / "private").exists():
            print(f"prep exists, kept: {root / 'prep'}")
        else:
            prepare(full, run_prep, root)
        if (output_root / "assets" / "ASSETS.json").exists():
            print(f"assets exist, kept: {output_root / 'assets'}")
        else:
            if not (a.dinov3_repo and a.weights):
                raise SystemExit("NOT_READY: no assets yet; pass --dinov3-repo and --weights")
            build_assets(output_root, dinov3_repo=Path(a.dinov3_repo), weights=Path(a.weights))
        return

    if a.cmd in ("assets", "external"):              # no DINOv3 here: the external arms load their own assets
        from benchmarks.representation_RxRx1.runners import external

        if a.cmd == "assets":
            source = Path(a.openphenom_source) if a.openphenom_source else None
            external.stage_openphenom(output_root, source=source)
        else:
            arms = tuple(s.strip() for s in a.arms.split(",") if s.strip())
            external.run(root, output_root, arms, workers=a.workers, input_size=a.input_size, fit=a.fit,
                         batch_size=a.batch_size, limit=a.limit)
        return
    if a.cmd == "llm":                               # no DINOv3 here: LLM-written features, no pretrained weights (A5)
        run_llm_arm(a, root, cfg, outside_repo(Path(a.run)))
        return
    os.environ.update(check_assets(output_root))     # before the analyzer (and huggingface_hub) is imported
    if a.cmd == "control":
        seeds = tuple(int(s) for s in a.seeds.split(",") if s.strip())
        control.control(root, seeds=seeds, split_key=cfg.split_key)
        return
    if a.cmd == "score-arm":
        from benchmarks.representation_RxRx1.runners.score import score_arm

        score_arm(root, a.arm, outside_repo(Path(a.run)) if a.run else None)
        return
    run = outside_repo(Path(a.run))
    if a.cmd == "shape":
        from benchmarks.representation_RxRx1.runners.shape_arm import run as run_arm

        engine = None
        if not a.no_engine:
            engine = reasoner(a.model)
        final = run_arm(root, run, cell_type=cfg.cell_type, model=a.model if engine else "none",
                        max_rounds=a.max_rounds, engine=engine, no_engine=a.no_engine, resume=a.resume,
                        seed=a.seed, early_stop=a.early_stop)
        print(json.dumps({k: final.get(k) for k in ("use", "case", "status", "reason", "provisional")}))
        return
    if a.cmd == "react":
        run_llm_arm(a, root, cfg, run)
        return
    from benchmarks.representation_RxRx1.runners.score import final as score

    arm_runs = [outside_repo(Path(p)) if p else None for p in (a.llm_run, a.react_run)]
    score(root, run, llm_run=arm_runs[0], react_run=arm_runs[1])
    if a.cmd == "finish":
        from benchmarks.batch_correction_JUMP.runners.trace_md import export

        export(run)
        for arm_run in filter(None, arm_runs):       # each LLM-arm run: its own EXPORT_<run>.tar.gz next to it
            export(arm_run)


# A20: the reasoner of the three LLM arms (SHAPE, ReAct and llm, the same for each): high reasoning effort, and room
# for it in the completion-token cap (reasoning included; the engine's default of 4000 would cut a high-effort reply
# off)
REASONING_EFFORT = "high"
MAX_COMPLETION_TOKENS = 16000


class Budgeted:
    """A reasoner engine whose every generate call carries the completion-token cap (unless the caller sets one)."""

    def __init__(self, engine, max_tokens: int):
        self._engine, self.max_tokens = engine, max_tokens

    def generate(self, content, **kwargs):
        kwargs.setdefault("max_tokens", self.max_tokens)
        return self._engine.generate(content, **kwargs)

    def __getattr__(self, name):
        return getattr(self._engine, name)


def reasoner(model: str):
    """The SHAPE / ReAct / llm reasoner engine: REASONING_EFFORT (SHAPE_REASONING_EFFORT, read by the engine per call)
    and MAX_COMPLETION_TOKENS (`Budgeted`)."""
    from shape.agent.verification.vision_engine import build_vision_engine

    os.environ["SHAPE_REASONING_EFFORT"] = REASONING_EFFORT
    engine = build_vision_engine(model)
    if engine is None:
        raise SystemExit(f"NOT_READY: no reasoner engine for {model} (LLM_API_KEY / LLM_BASE_URL)")
    return Budgeted(engine, MAX_COMPLETION_TOKENS)


def run_llm_arm(a, root: Path, cfg: BenchmarkConfig, run: Path) -> None:
    """The llm or react stage: one run of that LLM comparison arm into the new directory `run` (llm: llm_arms.py;
    react: react_arm.py, the shared ReAct runner), both with the SHAPE arm's reasoner (`reasoner`)."""
    from benchmarks.representation_RxRx1.runners import llm_arms, react_arm

    engine = reasoner(a.model)
    if a.cmd == "react":
        final = react_arm.run(root, run, cell_type=cfg.cell_type, model=a.model, engine=engine, resume=a.resume)
    else:
        final = llm_arms.run(root, run, arm=a.cmd, cell_type=cfg.cell_type, model=a.model, engine=engine,
                             resume=a.resume, max_attempts=a.max_rounds)
    print(json.dumps({k: final.get(k) for k in ("use", "case", "status", "reason")}))


if __name__ == "__main__":
    main()

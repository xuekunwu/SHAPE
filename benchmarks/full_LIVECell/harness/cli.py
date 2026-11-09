"""python -m benchmarks.full_LIVECell.harness.cli --help

The full_CPJUMP1 harness's run discipline on LIVECell: the same stages, arms, provenance,
RUN_META / DELIVERY records and fail-closed checks, with the LIVECell dataset spec (`spec.LIVECELL`)."""
from __future__ import annotations

import argparse
from pathlib import Path

from benchmarks.full_CPJUMP1.harness import cli as base
from benchmarks.full_CPJUMP1.harness.data import digest, read_json, within, write_json

from . import data
from .spec import LIVECELL

ARMS = ("control", "shape", *base.AGENT_ARMS)    # codex, claude, biomni: the agent arms (full_CPJUMP1 AGENT_ARMS)
MODEL = base.DEFAULT_MODEL


def resources(cores: int | None) -> dict:
    """The arm's cores: --cores, else every core this process may use; never more than those."""
    from shape.agent.verification.base import available_cores
    available = available_cores()
    resolved = available if cores is None else int(cores)
    if not 1 <= resolved <= available:
        raise ValueError(f"--cores {resolved} is outside the {available} cores this process may use")
    return {"cores": resolved, "available_cores": available, "source": "flag" if cores is not None else "hardware"}


def prepared_facts(prep: Path, rows: list[dict]) -> dict:
    """Every image is the PREPARE.json shape and dtype (one channel, 520 x 704, the native dtype)."""
    record = read_json(prep / "PREPARE.json")
    shape = [len(LIVECELL.channels), *LIVECELL.image_shape]
    if list(record["image_shape"]) != list(LIVECELL.image_shape) or list(record["channels"]) != list(LIVECELL.channels):
        raise ValueError("PREPARE.json channels or image shape differ from the LIVECell spec")
    bad = [r["image_id"] for r in rows if list(r["shape"]) != shape or r["dtype"] != record["dtype"]]
    if bad:
        raise ValueError(f"images differ from {shape} {record['dtype']}: {bad[:5]}")
    return {"shape": shape, "dtype": record["dtype"]}


def verify_prepared(prep: Path, out: Path) -> dict:
    """VM admission: the full CPJUMP1 check (all pixels, labels, raw hashes) plus the LIVECell shapes, the GT index,
    every GT file and the smoke list."""
    record = read_json(prep / "PREPARE.json")
    rows = data.load_prepared(prep)
    facts = prepared_facts(prep, rows)
    if digest(prep / "private" / "gt_index.json") != record["gt_index_sha256"]:
        raise ValueError("prepared private metadata changed: gt_index.json")
    for entry in read_json(prep / "private" / "gt_index.json"):
        if digest(within(prep / "private", entry["file"])) != entry["sha256"]:
            raise ValueError(f"prepared ground truth changed: {entry['image_id']}")
    data.smoke_rows(prep, rows)
    result = base.verify_prepared(prep, out, LIVECELL)
    result.update(facts, gt_index_sha256=record["gt_index_sha256"], smoke_sha256=record["smoke_sha256"])
    write_json(out, result)
    return result


def run(args) -> None:
    """One arm's run under base.run: the LIVECell spec, the smoke units of public/smoke.json, the resolved cores."""
    resolved = resources(args.cores)
    args.cores = resolved["cores"]
    facts = prepared_facts(args.prep, data.load_prepared(args.prep))
    print(f"RESOURCES cores={resolved['cores']} available={resolved['available_cores']} source={resolved['source']} "
          f"dtype={facts['dtype']}", flush=True)
    base.run(args, LIVECELL, smoke=data.smoke_rows, extra={"resources": resolved, "raw": facts})


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="LIVECell phase-contrast full pipeline (full_CPJUMP1 harness)")
    sub = p.add_subparsers(dest="command", required=True)
    select = sub.add_parser("select", help="build manifest.json from the downloaded files")
    select.add_argument("--raw", type=Path, required=True)
    select.add_argument("--manifest", type=Path, required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--raw", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--manifest", type=Path, help="default: <raw>/../manifest.json")
    verify = sub.add_parser("verify-prepared", help="verify transferred prepared data without rebuilding it")
    verify.add_argument("--prep", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    for name in ("run", "preflight"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--arm", choices=ARMS, required=True)
        cmd.add_argument("--model", default=MODEL, choices=sorted(set(base.ARM_MODELS.values())),
                         help="the arm's exact model (ARM_MODELS): claude claude-sonnet-5, the others gpt-5.6-luna")
        cmd.add_argument("--expected-commit", required=True)
        cmd.add_argument("--expected-branch", default=base.DEFAULT_BRANCH)
        cmd.add_argument("--assets-root", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        if name == "run":
            cmd.add_argument("--prep", type=Path, required=True)
            cmd.add_argument("--budget-hours", type=float, default=11)
            cmd.add_argument("--seed", type=int, default=0)
            cmd.add_argument("--cores", type=int, help="default: every core this process may use")
            cmd.add_argument("--train-cells-per-site", type=int, default=128)
            cmd.add_argument("--inference-chunk", type=int, default=2048)
            cmd.add_argument("--smoke", action="store_true")
    redo = sub.add_parser("rerun", help="continue a failed Codex run from its completed segment stage (not a full run)")
    redo.add_argument("--source", type=Path, required=True, help="the failed Codex run dir (read only)")
    redo.add_argument("--from-stage", choices=("crop",), required=True)
    redo.add_argument("--arm", choices=("codex",), required=True)
    redo.add_argument("--model", default=MODEL, choices=[MODEL])
    redo.add_argument("--prep", type=Path, required=True)
    redo.add_argument("--assets-root", type=Path, required=True)
    redo.add_argument("--out", type=Path, required=True)
    redo.add_argument("--expected-commit", required=True)
    redo.add_argument("--expected-branch", default=base.DEFAULT_BRANCH)
    redo.add_argument("--control", type=Path, required=True, help="the complete control run: the verdicts against it")
    redo.add_argument("--budget-hours", type=float, default=11, help="the full budget; the reused stage's time is deducted")
    redo.add_argument("--seed", type=int, default=0)
    redo.add_argument("--cores", type=int, help="default: the source run's cores")
    redo.add_argument("--train-cells-per-site", type=int, default=128)
    redo.add_argument("--inference-chunk", type=int, default=2048)
    redo.add_argument("--reason", default=None)
    posthoc = sub.add_parser("compare-rerun", help="post-hoc comparison of the continuation with control and SHAPE")
    posthoc.add_argument("--prep", type=Path, required=True)
    posthoc.add_argument("--rerun", type=Path, required=True)
    posthoc.add_argument("--control", type=Path, required=True)
    posthoc.add_argument("--shape", type=Path, required=True)
    posthoc.add_argument("--source", type=Path, help="the failed source run (default: the path the rerun records)")
    for arm in ("claude", "biomni"):
        posthoc.add_argument(f"--{arm}", type=Path, help=f"a complete {arm} full run to compare as well")
    posthoc.add_argument("--out", type=Path, required=True)
    st = sub.add_parser("stage", help="internal stage worker")
    st.add_argument("stage", choices=base.STAGES)
    st.add_argument("--prep", type=Path, required=True)
    st.add_argument("--run", type=Path, required=True)
    native = sub.add_parser("native-trial", help="trusted Codex native-tool worker")
    native.add_argument("stage", choices=base.STAGES[:-1])
    native.add_argument("--prep", type=Path, required=True)
    native.add_argument("--run", type=Path, required=True)
    native.add_argument("--config", type=Path)
    compare = sub.add_parser("compare", help="verify and compare the three complete arms")
    compare.add_argument("--prep", type=Path, required=True)
    for arm in ARMS:
        compare.add_argument(f"--{arm}", type=Path, required=arm in ("control", "shape", "codex"))
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--expected-commit", required=True)
    compare.add_argument("--expected-branch", default=base.DEFAULT_BRANCH)
    scoring = sub.add_parser("score")
    scoring.add_argument("--prep", type=Path, required=True)
    scoring.add_argument("--run", type=Path, required=True)
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.command == "select":
        data.select(args.raw, args.manifest)
    elif args.command == "prepare":
        data.prepare(args.raw, args.out, args.manifest)
    elif args.command == "verify-prepared":
        verify_prepared(args.prep, args.out)
    elif args.command == "preflight":
        write_json(args.out, base.preflight(args))
        print("PREFLIGHT_OK")
    elif args.command == "run":
        run(args)
    elif args.command == "stage":
        base.stage(args)
    elif args.command == "native-trial":
        base.native_trial(args)
    elif args.command == "rerun":
        from . import resume
        args.reason = args.reason or resume.REASON
        resume.rerun(args)
    elif args.command == "compare-rerun":
        from . import resume
        resume.compare(args.prep, args.rerun, args.control, args.shape, args.out, args.source,
                       agents={arm: getattr(args, arm) for arm in ("claude", "biomni") if getattr(args, arm)})
    elif args.command == "compare":
        from .compare import compare
        compare(args.prep, {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm) is not None}, args.out,
                args.expected_commit, args.expected_branch)
    else:
        from .score import score
        print(score(args.prep, args.run))


if __name__ == "__main__":
    main()

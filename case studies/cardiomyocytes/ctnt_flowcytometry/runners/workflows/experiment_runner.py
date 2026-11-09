"""VM runner: prepare -> native SHAPE segment/crop/embed/batch -> well regression -> frozen evaluation."""
from __future__ import annotations

# Internal worker CLI; the supported public entrypoint is run.py.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    __package__ = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows'


import argparse
import importlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import socket
import sys
import time

from ..runtime import reasoning
from ..dataset import preparation as data
from ..dataset import sampling as pilot
from ..runtime.source_snapshot import pack
from ..runtime.source_snapshot import verify_source

MODULE = "case studies.cardiomyocytes.ctnt_flowcytometry.runners.workflows.experiment_runner"
STAGES = ("segment", "crop", "embed", "batch", "predict")
MODEL = "gpt-5.6-luna"
AGENT_ARMS = ("codex", "biomni", "claude")
ARM_MODELS = {"shape": MODEL, "codex": MODEL, "biomni": "gpt-5.6-luna", "claude": "claude-sonnet-5"}


def configure_agent_reasoning(arm):
    if arm in ("biomni", "claude"):
        from benchmarks.full_CPJUMP1.harness.agent_reasoning import EFFORT_ENV
        os.environ[EFFORT_ENV] = "high"


def reasoning_record(arm, protocol):
    from benchmarks.full_CPJUMP1.harness import pipeline
    record = pipeline.reasoning_record(arm, protocol)
    if arm in ("biomni", "claude"):
        record["stages"] = {s: {"reasoning_effort": "high"} for s in STAGES if s != "predict"}
    return record


def requested_protocol(args):
    profile = getattr(args, "reasoning_profile", "all-high")
    if profile != "all-high" and args.arm != "shape":
        raise ValueError("image-medium is a separate SHAPE experiment")
    return reasoning.reasoning_protocol(profile)


def preflight(args) -> dict:
    protocol = requested_protocol(args)
    if args.arm != "control" and args.model != ARM_MODELS[args.arm]:
        raise ValueError(f"{args.arm} requires {ARM_MODELS[args.arm]}")
    configure_agent_reasoning(args.arm)
    identity = verify_source(args.expected_source)
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    import shape
    import torch

    if not Path(shape.__file__).resolve().is_relative_to(data.REPO):
        raise RuntimeError("SHAPE imported from another checkout")
    assets = asset_environment(args.assets_root)
    for module in ("tifffile", "sklearn", "anndata", "harmonypy", "shape.tools.cell_segmenter.tool",
                   "shape.tools.single_cell_cropper.tool", "shape.tools.cell_state_analyzer.tool"):
        importlib.import_module(module)
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA required for this E2E test")
    torch.zeros(1, device="cuda")
    result = {**identity, "assets": assets, "host": socket.gethostname(), "python": sys.executable,
              "repo": str(data.REPO), "gpu": torch.cuda.get_device_name(), "arm": args.arm}
    result["packages"] = {m: importlib.metadata.version(m) for m in
                          ("torch", "numpy", "pandas", "scikit-learn", "tifffile", "cellpose", "huggingface-hub")}
    if args.arm == "shape":
        if not os.environ.get("LLM_API_KEY") or os.environ.get("LLM_BASE_URL", "").rstrip("/") not in (
                "https://llm-gateway.example.edu/v1", "https://api.openai.com/v1"):
            raise RuntimeError("require SHAPE LLM_API_KEY and a verified LLM_BASE_URL")
        from benchmarks.full_HCMA.harness.shape_arm import build_engine
        from shape.engine import usage
        before = usage.total_tokens()
        from benchmarks.full_CPJUMP1.harness import pipeline
        os.environ.update(pipeline.stage_env("shape", "segment", protocol))
        pipeline.engine_for({"arm":"shape", "model":args.model,
                             "reasoning":{"protocol":protocol}}, "segment")
        tokens = usage.total_tokens()-before
        if tokens <= 0:
            raise RuntimeError("LLM preflight has no positive token usage")
        result.update(model=args.model, preflight_tokens=tokens, key_variable="LLM_API_KEY",
                      base_url=os.environ["LLM_BASE_URL"], response_model_attested=False)
    elif args.arm == "codex":
        os.environ["CTNT_CODEX_MODEL"] = args.model
        os.environ["CTNT_CODEX_BASE_URL"] = os.environ.get("LLM_BASE_URL", "").rstrip("/")
        from benchmarks.full_CPJUMP1.harness.codex_arm import backend
        result.update(backend("codex").preflight(args.out.parent/"codex_preflight", args.model))
        result.update(model=args.model, base_url=os.environ["CTNT_CODEX_BASE_URL"], response_model_attested=False)
    elif args.arm in ("biomni", "claude"):
        from benchmarks.full_CPJUMP1.harness.codex_arm import backend
        result.update(backend(args.arm).preflight(args.out.parent/f"{args.arm}_preflight", args.model))
        if result.get("reasoning_effort") != "high":
            raise RuntimeError("agent preflight did not retain high effort")
    result["reasoning_profile"] = getattr(args, "reasoning_profile", "all-high")
    result["reasoning_effort"] = ("mixed" if result["reasoning_profile"] == "image-medium" else "high") if args.arm != "control" else None
    result["reasoning_protocol"] = protocol
    return result


def stage(args):
    config = data.read_json(args.run / "CONFIG.json")
    verify_source(config["source_sha256"])
    if config["arm"] == "codex":
        os.environ["CTNT_CODEX_MODEL"] = config["model"]
        os.environ["CTNT_CODEX_BASE_URL"] = os.environ.get("LLM_BASE_URL", "").rstrip("/")
    configure_agent_reasoning(config["arm"])
    if args.stage == "predict":
        started = time.time()
        from ..prediction.well_predictor import predict
        predict(args.prep, args.run)
        data.write_json(args.run / "predict_COMPLETE.json", {"status": "complete", "stage": "predict",
                                                            "seconds":time.time()-started})
        return
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from benchmarks.full_CPJUMP1.harness import pipeline
    from shape.engine import usage
    asset_environment(Path(config["assets_root"]))
    selected = set(config["images"])
    rows = [r for r in data.load_prepared(args.prep) if r["image_id"] in selected]
    protocol = config["reasoning"]["protocol"]
    if protocol != reasoning.reasoning_protocol(config.get("reasoning_profile", "all-high")):
        raise ValueError("unexpected reasoning protocol")
    pipeline.check_stage_env(config["arm"], args.stage, protocol)
    started = time.time()
    try:
        if config["arm"] in AGENT_ARMS:
            from benchmarks.full_CPJUMP1.harness.codex_arm import run_stage
            run_stage(args.stage, args.prep, args.run, config)
        else:
            pipeline.STAGES[args.stage](args.prep, args.run, config, rows)
    finally:
        data.write_json(args.run / f"{args.stage}_USAGE.json", usage.snapshot())
    if config["arm"] == "shape" and usage.total_tokens() <= 0:
        raise RuntimeError("SHAPE stage has no positive reasoner token usage")
    tokens = usage.snapshot()
    if config["arm"] in AGENT_ARMS:
        tokens = data.read_json(args.run/"agent"/args.stage/"SELECTED.json")
        if tokens["total_tokens"] <= 0:
            raise RuntimeError("Agent stage has no positive reasoner token usage")
        data.write_json(args.run/f"{args.stage}_USAGE.json", tokens)
    if args.stage == "batch":
        import numpy as np
        record = data.read_json(args.run / "batch_SEARCH.json")
        if record["admission"]["decision"] not in ("ABSTAIN", "SKIP"):
            raise RuntimeError("batch correction admitted without validated cross-plate reference groups")
        if not np.array_equal(np.load(args.run / "profiles.npy"), np.load(args.run / "corrected.npy")):
            raise RuntimeError("abstention must preserve well profiles")
    data.write_json(args.run / f"{args.stage}_COMPLETE.json",
                    {"status": "complete", "stage": args.stage, "seconds": time.time()-started,
                     "usage": tokens, "reasoning":reasoning_record(config["arm"],protocol)["stages"].get(args.stage)})


def delivery_files(run: Path) -> set[Path]:
    """The shared harness's delivery boundary: scientific outputs and decisions, not redundant display caches.

    Every raw crop's hash is in cells.json and the embedding stage verifies those
    bytes before consumption. The immutable raw images are verified again here.
    """
    names = ("cells.json", "excluded_cells.json", "embedding.npy", "profiles.npy", "profiles.csv",
             "corrected.npy", "SSL_SELECTION.json", "batch_SEARCH.json", "CONFIG.json", "PREFLIGHT.json",
             "segmentation.json", "predictions.json", "REGRESSOR.json", "regressor.npz")
    files = {run / n for n in names}
    files.update(run / f"{s}_COMPLETE.json" for s in STAGES)
    files.update(run / f"{s}_USAGE.json" for s in STAGES if s != "predict")
    files.update(run.rglob("*SEARCH.json"))
    files.update(run.glob("*_COHORT.json"))
    files.update(run.glob("SEARCH_WARNINGS.jsonl"))
    files.update(run.glob("RESUME.json"))
    files.update(run.glob("crop_RECOVERY.json"))
    files.update((run/"agent").rglob("SELECTED.json"))
    files.update((run/"agent").rglob("CALLS.json"))
    files.update((run/"agent").rglob("ROUNDS.json"))
    if (run/"PILOT.json").exists():
        files.add(run/"PILOT.json")
    files.update(Path(r["mask"]) for r in data.read_json(run / "segmentation.json"))
    checkpoint = data.read_json(run / "SSL_SELECTION.json").get("checkpoint")
    if checkpoint:
        files.add(Path(checkpoint))
    for p in files:
        if not p.resolve().is_relative_to(run.resolve()) or not p.is_file():
            raise ValueError(f"missing delivery artifact or path outside run: {p}")
    return files


def freeze(args, record):
    verify_source(args.expected_source)
    data.load_prepared(args.prep, verify_pixels=True)
    artifacts = {p.relative_to(args.out).as_posix(): data.digest(p) for p in sorted(delivery_files(args.out))}
    data.write_json(args.out / "DELIVERY.json", {"status": "complete", "arm": args.arm,
                    "source_sha256": args.expected_source, "smoke": args.smoke,
                    "pilot": bool(args.pilot_manifest),
                    "prepare_sha256": record["prepare_sha256"], "artifacts": artifacts,
                    "raw_crop_integrity": "hashes in cells.json; verified before embedding",
                    "boundary": "final arrays, masks, selected checkpoint, parameters and search decisions"})


def run(args):
    protocol = requested_protocol(args)
    from benchmarks.full_CPJUMP1.harness.cli import child
    from benchmarks.full_CPJUMP1.harness import pipeline
    data.fresh_output(args.out, args.prep)
    if not math.isfinite(args.budget_hours) or args.budget_hours<=0 or args.cores<1:
        raise ValueError("positive finite runtime and core budget required")
    if args.train_cells_per_site<1 or args.inference_chunk<16:
        raise ValueError("invalid embedding sample/chunk size")
    verify_source(args.expected_source)
    rows = data.load_prepared(args.prep, verify_pixels=True)
    data.verify_private(args.prep, rows)
    sampling = pilot.verify(args.prep, args.pilot_manifest) if args.pilot_manifest else None
    if sampling:
        wanted = set(sampling["images"])
        rows = [r for r in rows if r["image_id"] in wanted]
    if args.smoke:
        rows = data.smoke_rows(rows)
    args.out.mkdir(parents=True)
    (args.out / "logs").mkdir()
    if sampling:
        data.write_json(args.out/"PILOT.json", sampling)
    record = {"status": "preflight", "prepare_sha256": data.digest(args.prep / "PREPARE.json"), "stages": {}}
    try:
        child([sys.executable, "-m", MODULE, "preflight", "--arm", args.arm, "--model", args.model,
               "--assets-root", str(args.assets_root), "--expected-source", args.expected_source,
               "--reasoning-profile", getattr(args, "reasoning_profile", "all-high"),
               "--out", str(args.out / "PREFLIGHT.json")], args.out / "logs" / "preflight.log", 1200)
        config = {"dataset": "ctnt_flowcytometry", "arm": args.arm, "model": args.model, "seed": args.seed,
                  "source_sha256": args.expected_source,
                  "deadline": None if args.no_time_limit else time.time()+args.budget_hours*3600,
                  "budget_hours": None if args.no_time_limit else args.budget_hours,
                  "cores": args.cores, "assets_root": str(args.assets_root),
                  "train_cells_per_site": args.train_cells_per_site, "inference_chunk": args.inference_chunk,
                  "images": [r["image_id"] for r in rows], "smoke": args.smoke,
                  "reasoning_profile": getattr(args, "reasoning_profile", "all-high"),
                  "reasoning": reasoning_record(args.arm, protocol),
                  "pilot_sha256": data.digest(args.out/"PILOT.json") if sampling else None,
                  "ssl_epoch_budget": pipeline.ssl_epoch_budget(rows, args.train_cells_per_site),
                  "shape_policy": {"crop_vrr_workers": args.cores},
                  "evaluation": "well-disjoint 40/10/50; date/plate-balanced; seen train/val and held-out evaluation",
                  "protocol": data.PROTOCOL, "target": data.TARGET, "target_unit": "percent",
                  "batch_fields": ["experiment_date", "plate_id"],
                  "segmentation_regime": "per-image" if args.smoke or args.arm != "shape" else
                      "joint unlabeled cohort calibration across all splits (transductive preprocessing)"}
        if args.smoke:
            config["smoke_policy"] = {"shape_search_images": 1, "shape_search_rounds": 2, "codex_rounds":2}
        data.write_json(args.out / "CONFIG.json", config)
        record.update(status="running", config=config, provenance=data.read_json(args.out / "PREFLIGHT.json"))
        from .stage_sequence import run_stages
        run_stages(STAGES, MODULE, args.prep, args.out, config, record)
        freeze(args, record)
        from ..evaluation.stage_quality import collect
        record["current_stage"] = "readouts"
        data.write_json(args.out/"RUN_META.json", record)
        collect(args.prep,args.out)
        from ..evaluation.heldout_evaluation import score
        result = score(args.prep, args.out)
        record.update(status="complete", result=result)
        data.write_json(args.out / "RUN_META.json", record)
        marker = "CTNT_FLOW_SMOKE_COMPLETE" if args.smoke else "CTNT_FLOW_PILOT_COMPLETE" if sampling else "CTNT_FLOW_COMPLETE"
        (args.out / marker).write_text(args.expected_source+"\n", encoding="ascii")
        print(f"{marker} run={args.out}", flush=True)
    except BaseException as e:
        record.update(status="failed", error=f"{type(e).__name__}: {e}")
        data.write_json(args.out / "RUN_META.json", record)
        raise


def parser():
    p = argparse.ArgumentParser(description="ctnt_flowcytometry: " + __doc__,
        epilog="Other run.py commands: suite, progress, recover, hybrid, crop-from-masks, materialize, smoke-image-stages, scripts, results, replay, inspect, fetch-metadata, audit-metadata. Use COMMAND --help.")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare", help="verify all 600 raw TIFFs; date/plate-balanced 40/10/50 well split")
    prep.add_argument("--data-root", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--seed", type=int, default=20261015)
    verify = sub.add_parser("verify-prepared")
    verify.add_argument("--prep", type=Path, required=True)
    snapshot = sub.add_parser("pack", help="package exact working code for VM testing, without a Git commit/push")
    snapshot.add_argument("--out", type=Path, required=True)
    sample = sub.add_parser("make-pilot", help="freeze a 20%% whole-well sample within the existing splits")
    sample.add_argument("--prep", type=Path, required=True)
    sample.add_argument("--out", type=Path, required=True)
    sample.add_argument("--seed", type=int, default=pilot.SEED)
    for name in ("run", "preflight"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--arm", choices=("shape", "control", *AGENT_ARMS), default="shape")
        cmd.add_argument("--model", default=MODEL)
        cmd.add_argument("--reasoning-profile", choices=("all-high", "image-medium"), default="all-high",
                         help="image-medium: SHAPE segment/crop medium, embed/batch high")
        cmd.add_argument("--expected-source", required=True, help="64-character source SHA256 emitted by pack")
        cmd.add_argument("--assets-root", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        if name == "run":
            cmd.add_argument("--prep", type=Path, required=True)
            cmd.add_argument("--budget-hours", type=float, default=12)
            cmd.add_argument("--cores", type=int, default=4)
            cmd.add_argument("--no-time-limit", action="store_true", help="no arm wall-clock deadline; retain search/epoch budgets")
            cmd.add_argument("--train-cells-per-site", type=int, default=128)
            cmd.add_argument("--inference-chunk", type=int, default=2048)
            cmd.add_argument("--seed", type=int, default=20261015)
            mode = cmd.add_mutually_exclusive_group()
            mode.add_argument("--smoke", action="store_true")
            mode.add_argument("--pilot-manifest", type=Path)
    worker = sub.add_parser("stage", help=argparse.SUPPRESS)
    worker.add_argument("stage", choices=STAGES)
    worker.add_argument("--prep", type=Path, required=True)
    worker.add_argument("--run", type=Path, required=True)
    native = sub.add_parser("native-trial", help=argparse.SUPPRESS)
    native.add_argument("stage", choices=STAGES[:-1])
    native.add_argument("--prep", type=Path, required=True)
    native.add_argument("--run", type=Path, required=True)
    native.add_argument("--config", type=Path)
    compare = sub.add_parser("compare-pilot", help="verify and compare three completed arms of the same pilot")
    compare.add_argument("--prep", type=Path, required=True)
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--expected-source", required=True)
    for arm in ("control","shape","codex"):
        compare.add_argument(f"--{arm}", type=Path, required=True)
    return p


def main(argv=None):
    args = parser().parse_args(argv)
    for k,v in vars(args).items():
        if isinstance(v, Path):
            setattr(args, k, v.resolve())
    if args.command == "prepare":
        print(json.dumps(data.prepare(args.data_root, args.out, args.seed), indent=2))
    elif args.command == "verify-prepared":
        rows = data.load_prepared(args.prep, verify_pixels=True)
        data.verify_private(args.prep, rows)
        print(f"PREPARED_VERIFIED images={len(rows)} sha256={data.digest(args.prep/'PREPARE.json')}")
    elif args.command == "pack":
        print(json.dumps(pack(args.out), indent=2))
    elif args.command == "make-pilot":
        if args.out.exists():
            raise FileExistsError(args.out)
        data.write_json(args.out, pilot.manifest(args.prep, args.seed))
        print(json.dumps(pilot.verify(args.prep,args.out),indent=2))
    elif args.command == "preflight":
        if args.out.exists():
            raise FileExistsError(args.out)
        data.write_json(args.out, preflight(args))
        print("PREFLIGHT_OK")
    elif args.command == "stage":
        stage(args)
    elif args.command == "native-trial":
        config = data.read_json(args.config or args.run/"CONFIG.json")
        verify_source(config["source_sha256"])
        from benchmarks.full_CPJUMP1.harness.cli import native_trial
        native_trial(args)
    elif args.command == "compare-pilot":
        from ..evaluation.arm_comparison import compare
        compare(args.prep,{arm:getattr(args,arm) for arm in ("control","shape","codex")},args.out,args.expected_source)
    else:
        run(args)


if __name__ == "__main__":
    main()

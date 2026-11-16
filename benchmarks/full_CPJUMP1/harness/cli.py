"""python -m benchmarks.full_CPJUMP1.harness.cli --help"""
from __future__ import annotations

import argparse
import importlib
import json
import math
import os
from pathlib import Path
import signal
import socket
import subprocess
import sys
import time

from shape.agent.arms import AGENT_ARMS, ARM_MODELS as _AGENT_ARM_MODELS, ARM_ALLOWED as _AGENT_ARM_ALLOWED
from .data import MANIFEST_SHA256, digest, load_prepared, prepare, read_json, write_json
from .pipeline import CPJUMP1, dataset, reasoning_record, stage_env, stage_reasoning

REPO = Path(__file__).resolve().parents[3]
DEFAULT_MODEL = "gpt-6-sol"
DEFAULT_BRANCH = "e2e"         # --expected-branch: the branch a run's checkout and its comparison must be on
# The agent arms (shape/agent/arms) share codex_arm's stage protocol; each has its own backend and model.
ARMS = ("control", "shape", *AGENT_ARMS)
ARM_MODELS = {"shape": DEFAULT_MODEL, **_AGENT_ARM_MODELS}       # + shape, which runs the SHAPE pipeline, not a backend
# Every model id each arm accepts (shape runs the pipeline reasoner on the gateway (gpt-*) or Anthropic-direct
# (claude-*); the agent arms carry their own ARM_ALLOWED). --model must be in the arm's set.
ARM_ALLOWED = {"shape": frozenset({"gpt-6-sol", "claude-opus-5", "zai-org/GLM-5.3-Flash",
                                   "deepseek-ai/DeepSeek-V4.1-Flash"}), **_AGENT_ARM_ALLOWED}
STAGES = ("segment", "crop", "embed", "batch", "cluster")
SMOKE_POLICY = {"shape_search_images": 1, "shape_search_rounds": 2, "codex_rounds": 2}
ARTIFACTS = ("cells.json", "excluded_cells.json", "embedding.npy", "profiles.npy", "profiles.csv",
             "corrected.npy", "profiles.h5ad", "states.csv", "SSL_SELECTION.json", "batch_SEARCH.json",
             "CONFIG.json", "PREFLIGHT.json", "segmentation.json")
RERUN_MARKER = "RERUN_COMPLETE"   # a post-hoc rerun's completion marker (rerun()), never a dataset's full-run marker
RERUN_RECORD = "RERUN.json"       # a rerun's record of what it reused from its source run (rerun.reuse_upstream)
RERUN_FROM = ("embed",)           # the stages a rerun may start from
RERUN_REASON = ("post-hoc re-evaluation of the arm with the current code on the source run's segment and crop "
                "outputs: the embed (SSL: representation_RxRx1 A21 with its expert start), batch and cluster stages "
                "rerun under the current per-stage reasoning table, before any new full run")


def shape_policy(cores: int) -> dict:
    """SHAPE's image-stage policy for a run on `cores` cores: every crop runs its own VRR, `cores` images at a time.
    Segmentation runs SHAPE's shared segmenter cohort with `cores` images at a time (pipeline._segmentation_cohort)."""
    return {"crop_vrr_workers": cores}


def git(*args: str) -> str:
    return subprocess.check_output(["git", "-C", str(REPO), *args], text=True).strip()


def code_identity(expected: str, branch: str = DEFAULT_BRANCH) -> dict:
    """The checkout is `branch` at the pinned commit, clean, equal to origin/<branch>, and SHAPE is imported from it."""
    import shape
    if len(expected) != 40 or any(c not in "0123456789abcdef" for c in expected):
        raise ValueError("--expected-commit must be a full commit SHA")
    if git("rev-parse", "HEAD") != expected or git("branch", "--show-current") != branch:
        raise RuntimeError(f"INVALID_REMOTE_SYNC: expected {branch} at the pinned commit")
    if git("status", "--porcelain"):
        raise RuntimeError("INVALID_REMOTE_SYNC: checkout is not clean")
    if git("rev-parse", f"origin/{branch}") != expected:
        raise RuntimeError(f"INVALID_REMOTE_SYNC: origin/{branch} does not match; fetch before submission")
    if not Path(shape.__file__).resolve().is_relative_to(REPO):
        raise RuntimeError("SHAPE is imported from another checkout")
    return {"branch": branch, "commit": expected, "repo": str(REPO), "python": sys.executable,
            "job_id": os.environ.get("SLURM_JOB_ID"), "host": socket.gethostname()}


def asset_environment(root: Path) -> dict:
    from benchmarks.full_HCMA.harness.assets import DINOV3_COMMIT, WEIGHTS_SHA256, tree_digest, tool_env, check_hf_offline
    record = read_json(root / "assets" / "ASSETS.json")
    hub = record["dinov3_hub"]
    if hub["commit"] != DINOV3_COMMIT or tree_digest(Path(hub["path"]))[0] != hub["tree_sha256"]:
        raise ValueError("pinned DINOv3 source changed")
    for key in ("dinov3_weights", "cpsam"):
        if digest(Path(record[key]["path"])) != record[key]["sha256"]:
            raise ValueError(f"model asset changed: {key}")
    if record["dinov3_weights"]["sha256"] != WEIGHTS_SHA256:
        raise ValueError("incorrect DINOv3 weights")
    env = tool_env(root)
    if Path(env["SHAPE_DINOV3_REPO"]).resolve() != Path(hub["path"]).resolve():
        raise ValueError("pinned DINOv3 source path does not match benchmark environment")
    # Do not allow a preexisting alternate cache to bypass the pinned assets. A remote SAM
    # service ($SHAPE_SAM_SERVER_URL) is NOT a bypass: it runs the SAME pinned cpsam weights on
    # a GPU node so a CPU orchestration node can delegate segmentation (the cell_state_HCMA
    # pattern). It is therefore kept when set; the DINOv3 inference server url is still dropped
    # because the embed stage trains and must use the local pinned backbone, not a remote one.
    for key in ("HF_HUB_CACHE", "HUGGINGFACE_HUB_CACHE", "SHAPE_DINO_SERVER_URL"):
        os.environ.pop(key, None)
    os.environ.update(env)
    check_hf_offline(root / "assets" / "hf", Path(record["dinov3_weights"]["path"]))
    return {"manifest_sha256": digest(root / "assets" / "ASSETS.json"),
            "dinov3_commit": hub["commit"], "weights_sha256": record["dinov3_weights"]["sha256"],
            "cpsam_sha256": record["cpsam"]["sha256"]}


def preflight(args) -> dict:
    if args.arm != "control" and args.model not in ARM_ALLOWED[args.arm]:
        raise ValueError(f"the {args.arm} arm accepts models {sorted(ARM_ALLOWED[args.arm])}, not {args.model!r}")
    identity = code_identity(args.expected_commit, args.expected_branch)
    assets = asset_environment(args.assets_root)
    for module in ("torch", "anndata", "scanpy", "leidenalg", "harmonypy", "copairs", "scib_metrics",
                   "shape.tools.cell_segmenter.tool", "shape.tools.single_cell_cropper.tool",
                   "shape.tools.cell_state_analyzer.tool", "shape.tools.batch_corrector.tool"):
        importlib.import_module(module)
    import torch
    if not torch.cuda.is_available():
        raise RuntimeError("GPU allocation required")
    torch.zeros(1, device="cuda")
    from shape.agent.verification.four_stage import strategy_name
    strategies = {k: strategy_name(k) for k in ("segmentation", "crop", "representation", "batch_correction")}
    result = {**identity, "assets": assets, "gpu": torch.cuda.get_device_name(),
              "gpu_memory_bytes": torch.cuda.get_device_properties(0).total_memory, "search_strategies": strategies}
    if args.arm == "shape":
        from benchmarks.full_HCMA.harness.shape_arm import build_engine
        from shape.engine import usage
        from shape.engine.factory import provider_of
        # The reasoner's provider follows the model id and the environment: gpt-* -> the OpenAI-compatible
        # gateway (LLM_BASE_URL + LLM_API_KEY); claude-* -> Anthropic-direct (ANTHROPIC_API_KEY) ONLY when
        # LLM_BASE_URL is unset (factory.provider_of routes claude-* through the gateway when it is set, and
        # the gateway serves no claude model, so a claude-* SHAPE run must unset LLM_BASE_URL first).
        provider = provider_of(args.model)
        if provider == "anthropic":
            if not os.environ.get("ANTHROPIC_API_KEY"):
                raise RuntimeError("the SHAPE arm on a claude-* model requires ANTHROPIC_API_KEY and LLM_BASE_URL "
                                   "unset (Anthropic-direct; the gateway serves no claude model)")
            base_url, key_variable = "https://api.anthropic.com", "ANTHROPIC_API_KEY"
        else:
            if not os.environ.get("LLM_API_KEY") or not os.environ.get("LLM_BASE_URL"):
                raise RuntimeError("the SHAPE arm on a gpt-* model requires LLM_API_KEY and the gateway LLM_BASE_URL")
            base_url, key_variable = os.environ["LLM_BASE_URL"], "LLM_API_KEY"
        before = usage.total_tokens()
        build_engine(args.model)
        tokens = usage.total_tokens() - before
        if tokens <= 0:
            raise RuntimeError("model preflight did not report positive token usage")
        result.update(model=args.model, requested_model=args.model, response_model=None,
                      response_model_attested=False, model_provenance="requested_engine_config",
                      preflight_tokens=tokens, base_url=base_url, key_variable=key_variable)
    elif args.arm in AGENT_ARMS:
        from shape.agent.arms import backend
        result.update(backend(args.arm).preflight(args.out.parent / f"{args.arm}_preflight", args.model))
    return result


def verify_prepared(prep: Path, out: Path, spec=CPJUMP1) -> dict:
    """Perform full prepared-data verification at VM admission."""
    rows = spec.loader(load_prepared)(prep, verify_pixels=True)
    if len(rows) != spec.n_images:
        raise ValueError(f"expected the complete {spec.n_images}-view subset, found {len(rows)}")
    record = read_json(prep / "PREPARE.json")
    for name, key in (("labels.json", "labels_sha256"), ("raw_hashes.json", "raw_hashes_sha256")):
        if digest(prep / "private" / name) != record[key]:
            raise ValueError(f"prepared private metadata changed: {name}")
    result = {"status": "verified", "views": len(rows),
              "prepare_sha256": digest(prep / "PREPARE.json"),
              "images_sha256": record["images_sha256"],
              "labels_sha256": record["labels_sha256"],
              "raw_hashes_sha256": record["raw_hashes_sha256"]}
    out.parent.mkdir(parents=True, exist_ok=True)
    write_json(out, result)
    print(f"PREPARED_VERIFIED views={len(rows)} sha256={result['prepare_sha256']}", flush=True)
    return result


def smoke_images(rows: list[dict], usages: tuple = CPJUMP1.usages) -> list[dict]:
    """Whole wells from both plates, with development train/val and held-out sites."""
    chosen = set()
    for plate in sorted({r["plate"] for r in rows}):
        for usage in usages:
            wells = sorted({r["well"] for r in rows if r["plate"] == plate and r["usage"] == usage and not r["negative_control"]})
            chosen.update(wells[:2 if usage == usages[1] else 1])
        chosen.update(sorted({r["well"] for r in rows if r["plate"] == plate and r["negative_control"]})[:1])
    return [r for r in rows if r["well"] in chosen]


def child(command: list[str], log: Path, timeout: float | None, *, new_session: bool = True, env: dict | None = None) -> None:
    """Run one worker process; `env`: its whole environment (None: this process's)."""
    if timeout is not None and timeout <= 0:
        raise TimeoutError("arm deadline exceeded before starting stage")
    if timeout == math.inf:
        timeout = None
    with log.open("w", encoding="utf-8") as handle:
        proc = subprocess.Popen(command, cwd=REPO, stdout=handle, stderr=subprocess.STDOUT, env=env,
                                start_new_session=os.name == "posix" and new_session)
        try:
            rc = proc.wait(timeout=timeout)
            if rc:
                raise RuntimeError(f"stage failed (exit {rc}); see {log}")
        finally:
            if os.name == "posix" and new_session:
                # Reap descendants even if the immediate worker already exited.
                try:
                    os.killpg(proc.pid, signal.SIGKILL)
                except ProcessLookupError:
                    pass
            elif proc.poll() is None:
                proc.kill()
            if proc.poll() is None:
                proc.wait()


def run_preflight(args, spec=CPJUMP1) -> dict:
    """The arm's preflight in its own process (it opens CUDA there; the driver never competes for an exclusive GPU);
    returns its PREFLIGHT.json."""
    child([sys.executable, "-m", spec.cli, "preflight", "--expected-commit", args.expected_commit,
           "--expected-branch", args.expected_branch,
           "--assets-root", str(args.assets_root), "--arm", args.arm, "--model", args.model or "",
           "--out", str(args.out / "PREFLIGHT.json")], args.out / "logs" / "preflight.log", 600)
    return read_json(args.out / "PREFLIGHT.json")


def run_stages(args, spec, config: dict, record: dict, stages) -> None:
    """Each of `stages` in its own worker process under the arm deadline; each stage's record in RUN_META.json."""
    for stage in stages:
        record["current_stage"] = stage
        write_json(args.out / "RUN_META.json", record)
        started = time.time()
        # The stage's reasoner settings go in its own worker's environment only (pipeline.stage_env).
        extra_env = stage_env(args.arm, stage)
        print(f"STAGE_START {stage} reasoning={json.dumps(config['reasoning']['stages'][stage])} "
              f"env={json.dumps(extra_env)}", flush=True)
        child([sys.executable, "-m", spec.cli, "stage", stage, "--prep", str(args.prep), "--run", str(args.out)],
              args.out / "logs" / f"{stage}.log", config["deadline"] - time.time(),
              env={**os.environ, **extra_env})
        record["stages"][stage] = read_json(args.out / f"{stage}_COMPLETE.json")
        print(f"STAGE_COMPLETE {stage} seconds={time.time() - started:.1f}", flush=True)


def deliver(args, spec, record: dict, provenance: dict, marker: str, control: Path | None = None) -> None:
    """After the last stage: the code identity and the prepared pixels checked again, every artifact bound by hash in
    DELIVERY.json, the run scored unless a smoke (`control`: a scored control run for the verdicts), RUN_META.json
    complete, and `marker` holding the commit."""
    code_identity(args.expected_commit, args.expected_branch)
    spec.loader(load_prepared)(args.prep, verify_pixels=True)
    artifacts = {args.out / name for name in ARTIFACTS}
    artifacts.update(args.out.glob("*_COMPLETE.json"))
    artifacts.update(args.out.glob("*_USAGE.json"))
    artifacts.update(args.out.rglob("*SEARCH.json"))
    artifacts.update(args.out.glob("*_COHORT.json"))
    artifacts.update(args.out.glob("SEARCH_WARNINGS.jsonl"))
    artifacts.update(args.out.glob(RERUN_RECORD))                  # a rerun's record of what it reused
    for pattern in ("TRIALS.json", "CALLS.json", "ROUNDS.json", "SELECTED.json", "round_*.jsonl", "round_*.json",
                    "failed_round_*.jsonl", "failed_round_*.json"):
        artifacts.update((args.out / "agent").rglob(pattern))
    artifacts.update(Path(row["mask"]) for row in read_json(args.out / "segmentation.json"))
    checkpoint = read_json(args.out / "SSL_SELECTION.json").get("checkpoint")
    if checkpoint:
        artifacts.add(Path(checkpoint))
    delivery = {"status": "complete", "arm": args.arm, "provenance": provenance,
                "prepare_sha256": record["prepare_sha256"], "smoke": args.smoke,
                "artifacts": {p.relative_to(args.out).as_posix(): digest(p) for p in sorted(artifacts)}}
    write_json(args.out / "DELIVERY.json", delivery)
    if not args.smoke:
        verdicts = {} if control is None else {"control": control}
        importlib.import_module(spec.score).score(args.prep, args.out, **verdicts)
    record["status"] = "complete"
    write_json(args.out / "RUN_META.json", record)
    (args.out / marker).write_text(args.expected_commit + "\n", encoding="ascii")
    print(f"{marker} arm={args.arm} run={args.out}", flush=True)


def run(args, spec=CPJUMP1, smoke=None, extra=None) -> None:
    """One arm's full or smoke run. spec: the dataset; smoke(prep, rows): the smoke run's rows (None: smoke_images);
    extra: the dataset CLI's further CONFIG.json fields (a dataset other than CPJUMP1 is named in `dataset`)."""
    if args.out.exists():
        raise FileExistsError("use a fresh output directory; partial runs are never reused as evidence")
    if not math.isfinite(args.budget_hours) or args.budget_hours <= 0:
        raise ValueError("budget-hours must be positive and finite")
    if args.train_cells_per_site < 1 or args.inference_chunk < 16 or args.cores < 1:
        raise ValueError("invalid crop cap, chunk size or core count")
    load = spec.loader(load_prepared)
    rows = load(args.prep, verify_pixels=True)
    args.out.mkdir(parents=True)
    (args.out / "logs").mkdir()
    record = {"status": "preflight", "arm": args.arm}
    try:
        provenance = run_preflight(args, spec)
        if args.smoke:
            rows = smoke_images(rows, spec.usages) if smoke is None else smoke(args.prep, rows)
        config = {"arm": args.arm, "model": args.model, "seed": args.seed, "cores": args.cores,
                  "deadline": time.time() + args.budget_hours * 3600, "budget_hours": args.budget_hours,
                  "train_cells_per_site": args.train_cells_per_site, "inference_chunk": args.inference_chunk,
                  "images": [r["image_id"] for r in rows], "smoke": args.smoke, "assets_root": str(args.assets_root)}
        if spec is not CPJUMP1:
            config.update(dataset=spec.name, **(extra or {}))
        from .pipeline import ssl_epoch_budget
        config["ssl_epoch_budget"] = ssl_epoch_budget(rows, args.train_cells_per_site, spec.usages)   # every arm's fixed epochs
        if args.arm == "shape":
            config["shape_policy"] = shape_policy(args.cores)
        config["reasoning"] = reasoning_record(args.arm)        # pipeline.REASONING, per stage, for this arm
        if args.smoke:
            config["smoke_policy"] = dict(SMOKE_POLICY)
        write_json(args.out / "CONFIG.json", config)
        record.update(status="running", provenance=provenance, config=config,
                      prepare_sha256=digest(args.prep / "PREPARE.json"), stages={})
        run_stages(args, spec, config, record, STAGES)
        deliver(args, spec, record, provenance, "SMOKE_COMPLETE" if args.smoke else spec.marker)
    except BaseException as error:
        if args.arm in AGENT_ARMS and os.name == "posix":
            from benchmarks.full_HCMA.harness.agents import kill_run_processes
            for root in [args.out / f"{args.arm}_preflight", *[args.out / "agent" / s for s in STAGES[:-1]]]:
                kill_run_processes(root, grace=2)
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.out / "RUN_META.json", record)
        raise


def rerun(args, spec=CPJUMP1) -> None:
    """A post-hoc partial rerun of a completed full run (rerun.py). The source run's stages before --from-stage are
    reused: their outputs copied or hard-linked into the new run, hash-verified, the paths the later stages read
    rewired to the copies (RERUN_RECORD records the mapping); the source is never modified. The later stages run with
    the current code exactly as run() runs them, then the run is delivered and scored as run() does (--control: the
    verdicts against that control) and marked RERUN_MARKER, never the full-run marker, so the strict compare cannot
    take it for a clean full run."""
    from . import rerun as posthoc
    if args.out.exists():
        raise FileExistsError("use a fresh output directory; partial runs are never reused as evidence")
    first = STAGES.index(args.from_stage)
    reused, stages = STAGES[:first], STAGES[first:]
    source = posthoc.check_source(args.source, args.arm, args.model, args.prep, spec)
    old = source["config"]
    load = spec.loader(load_prepared)
    rows = load(args.prep, verify_pixels=True)
    if set(old["images"]) != {r["image_id"] for r in rows}:
        raise ValueError("the source run's views are not the preparation's")
    budget = posthoc.budget(source, reused, args.budget_hours)
    resources = posthoc.resources(args.cores, old["cores"])
    if args.control is not None:
        posthoc.verify_run(args.control, "control", digest(args.prep / "PREPARE.json"), spec)
    args.cores, args.smoke = resources["cores"], False
    print(f"RERUN source={args.source} commit={source['commit']} reused={','.join(reused)} "
          f"rerun={','.join(stages)} budget_hours={budget['hours']:.3f} cores={args.cores} "
          f"available_cores={resources['available_cores']}", flush=True)
    args.out.mkdir(parents=True)
    (args.out / "logs").mkdir()
    record = {"status": "preflight", "arm": args.arm}
    try:
        provenance = run_preflight(args, spec)
        posthoc.reuse_upstream(source, args.out, reused)
        config = {"arm": args.arm, "model": args.model, "seed": old["seed"], "cores": args.cores,
                  "deadline": time.time() + budget["hours"] * 3600, "budget_hours": budget["hours"],
                  "train_cells_per_site": old["train_cells_per_site"], "inference_chunk": old["inference_chunk"],
                  "images": old["images"], "smoke": False, "assets_root": str(args.assets_root)}
        if spec is not CPJUMP1:
            config["dataset"] = spec.name
        from .pipeline import SSL_EXPERT_CONFIG, ssl_epoch_budget
        config["ssl_epoch_budget"] = ssl_epoch_budget(rows, config["train_cells_per_site"], spec.usages)
        if args.arm == "shape":
            config["shape_policy"] = shape_policy(args.cores)
            config["ssl_expert_config"] = [dict(r) for r in SSL_EXPERT_CONFIG]    # the verifier's expert start
        config["reasoning"] = reasoning_record(args.arm)          # pipeline.REASONING, per stage, as run() records
        config["resources"] = resources
        config["rerun"] = {
            "source_run": str(args.source), "source_commit": source["commit"],
            "source_branch": source["provenance"].get("branch"), "source_config_sha256": source["config_sha256"],
            "source_delivery_sha256": source["delivery_sha256"], "reused_stages": list(reused),
            "rerun_stages": list(stages), "reason": args.reason, "commit": provenance.get("commit"),
            "branch": provenance.get("branch"), "budget": budget, "source_cores": old["cores"],
            "source_reasoning": old.get("reasoning"), "source_ssl_epoch_budget": old.get("ssl_epoch_budget"),
            "record": RERUN_RECORD, "record_sha256": digest(args.out / RERUN_RECORD)}
        write_json(args.out / "CONFIG.json", config)
        # The reused stages' records stay the source's (its commit); the rerun stages' are this run's.
        record.update(status="running", provenance=provenance, config=config, rerun=config["rerun"],
                      prepare_sha256=digest(args.prep / "PREPARE.json"),
                      stages={s: read_json(args.out / f"{s}_COMPLETE.json") for s in reused},
                      stage_code={**{s: {"run": "source", "commit": source["commit"]} for s in reused},
                                  **{s: {"run": "rerun", "commit": provenance.get("commit")} for s in stages}})
        run_stages(args, spec, config, record, stages)
        deliver(args, spec, record, provenance, RERUN_MARKER, control=args.control)
    except BaseException as error:
        record.update(status="failed", error=f"{type(error).__name__}: {error}")
        write_json(args.out / "RUN_META.json", record)
        raise


def stage(args) -> None:
    config = read_json(args.run / "CONFIG.json")
    asset_environment(Path(config["assets_root"]))
    rows = [r for r in dataset(config).loader(load_prepared)(args.prep) if r["image_id"] in set(config["images"])]
    from .pipeline import STAGES as functions
    from shape.engine import usage
    from .pipeline import check_stage_env
    # The settings this stage's reasoner runs with: those its CONFIG.json records, carried by this worker's env.
    reasoning = stage_reasoning(config["arm"], args.stage)
    if "reasoning" in config and config["reasoning"]["stages"].get(args.stage) != reasoning:
        raise RuntimeError(f"stage {args.stage}: CONFIG.json reasoning differs from pipeline.REASONING")
    check_stage_env(config["arm"], args.stage)
    recorded = {} if reasoning is None or config["arm"] in AGENT_ARMS else {"reasoning": reasoning}
    if reasoning is not None:
        print(f"STAGE_REASONING {args.stage} {json.dumps(reasoning)}", flush=True)
    started = time.time()
    try:
        if config["arm"] in AGENT_ARMS and args.stage != "cluster":
            from .codex_arm import run_stage
            run_stage(args.stage, args.prep, args.run, config)
        else:
            functions[args.stage](args.prep, args.run, config, rows)
    finally:
        write_json(args.run / f"{args.stage}_USAGE.json", {**usage.snapshot(), **recorded})
    if config["arm"] == "shape" and args.stage != "cluster" and usage.total_tokens() <= 0:
        raise RuntimeError("SHAPE stage reported zero reasoner tokens")
    tokens = {**usage.snapshot(), **recorded}
    if config["arm"] in AGENT_ARMS and args.stage != "cluster":
        tokens = read_json(args.run / "agent" / args.stage / "SELECTED.json")
        if tokens["total_tokens"] <= 0:
            raise RuntimeError(f"{config['arm']} stage reported zero tokens")
        write_json(args.run / f"{args.stage}_USAGE.json", tokens)
    write_json(args.run / f"{args.stage}_COMPLETE.json", {"status": "complete", "stage": args.stage,
               "seconds": time.time() - started, "usage": tokens})


def native_trial(args):
    from .pipeline import STAGES as functions, codex_calls
    config = read_json(args.config or args.run / "CONFIG.json")
    # Every agent arm's worker takes pipeline.py's agent-arm path (arm "codex"); agent_arm names the real arm.
    if config["arm"] != "codex" or config.get("agent_arm", "codex") not in AGENT_ARMS:
        raise ValueError("native-trial is internal to the agent arms (the Codex native-call path)")
    asset_environment(Path(config["assets_root"]))
    rows = [r for r in dataset(config).loader(load_prepared)(args.prep) if r["image_id"] in set(config["images"])]
    if args.stage in ("segment", "crop"):
        # Image stages: only the named per-image calls of one request wave.
        codex_calls(args.stage, args.prep, args.run, config, rows)
    else:
        functions[args.stage](args.prep, args.run, config, rows)


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description="CPJUMP1 five-channel CRISPR full pipeline")
    sub = p.add_subparsers(dest="command", required=True)
    prep = sub.add_parser("prepare")
    prep.add_argument("--data", type=Path, required=True)
    prep.add_argument("--out", type=Path, required=True)
    prep.add_argument("--merged-dir", type=Path, help="reuse verified fields.csv/COMPLETE five-channel stacks")
    assets = sub.add_parser("assets")
    assets.add_argument("--root", type=Path, required=True)
    assets.add_argument("--dinov3-repo", type=Path, required=True)
    assets.add_argument("--weights", type=Path, required=True)
    assets.add_argument("--cpsam", type=Path)
    assets.add_argument("--prep", type=Path, help="verify and reuse completed preparation before building assets")
    verify = sub.add_parser("verify-prepared", help="verify transferred prepared data without rebuilding it")
    verify.add_argument("--prep", type=Path, required=True)
    verify.add_argument("--out", type=Path, required=True)
    for name in ("run", "preflight"):
        cmd = sub.add_parser(name)
        cmd.add_argument("--arm", choices=ARMS, required=True)
        # The reasoner model, validated against the arm's allowed set in preflight (ARM_ALLOWED). cc/claude
        # passes claude-opus-5; shape/codex/biomni default to gpt-6-sol, with claude-opus-5 also allowed for
        # shape and biomni (Anthropic-direct). choices is the union so --model accepts every wired id.
        cmd.add_argument("--model", default=DEFAULT_MODEL,
                         choices=sorted({m for models in ARM_ALLOWED.values() for m in models}))
        cmd.add_argument("--expected-commit", required=True)
        cmd.add_argument("--expected-branch", default=DEFAULT_BRANCH)
        cmd.add_argument("--assets-root", type=Path, required=True)
        cmd.add_argument("--out", type=Path, required=True)
        if name == "run":
            cmd.add_argument("--prep", type=Path, required=True)
            cmd.add_argument("--budget-hours", type=float, default=11)
            cmd.add_argument("--seed", type=int, default=0)
            cmd.add_argument("--cores", type=int, default=8)
            cmd.add_argument("--train-cells-per-site", type=int, default=128)
            cmd.add_argument("--inference-chunk", type=int, default=2048)
            cmd.add_argument("--smoke", action="store_true")
    st = sub.add_parser("stage", help="internal stage worker")
    st.add_argument("stage", choices=STAGES)
    st.add_argument("--prep", type=Path, required=True)
    st.add_argument("--run", type=Path, required=True)
    native = sub.add_parser("native-trial", help="trusted Codex native-tool worker")
    native.add_argument("stage", choices=STAGES[:-1])
    native.add_argument("--prep", type=Path, required=True)
    native.add_argument("--run", type=Path, required=True)
    native.add_argument("--config", type=Path)
    compare = sub.add_parser("compare", help="verify and compare the complete arms (claude and biomni optional)")
    compare.add_argument("--prep", type=Path, required=True)
    for arm in ARMS:
        compare.add_argument(f"--{arm}", type=Path, required=arm in ("control", "shape", "codex"))
    compare.add_argument("--out", type=Path, required=True)
    compare.add_argument("--expected-commit", required=True)
    compare.add_argument("--expected-branch", default=DEFAULT_BRANCH)
    redo = sub.add_parser("rerun", help="post-hoc partial rerun of a completed arm from --from-stage (not a full run)")
    redo.add_argument("--source", type=Path, required=True,
                      help="the completed full run whose upstream stages are reused")
    redo.add_argument("--from-stage", choices=RERUN_FROM, required=True)
    redo.add_argument("--arm", choices=("shape",), required=True)
    redo.add_argument("--model", default=DEFAULT_MODEL, choices=sorted(set(ARM_MODELS.values())))
    redo.add_argument("--prep", type=Path, required=True)
    redo.add_argument("--assets-root", type=Path, required=True)
    redo.add_argument("--out", type=Path, required=True)
    redo.add_argument("--expected-commit", required=True)
    redo.add_argument("--expected-branch", default=DEFAULT_BRANCH)
    redo.add_argument("--budget-hours", type=float,
                      help="default: the source's budget_hours less its reused stages' recorded seconds")
    redo.add_argument("--cores", type=int, help="default: the source run's cores (at most this machine's)")
    redo.add_argument("--control", type=Path, help="a complete control run of the preparation: the verdicts against it")
    redo.add_argument("--reason", default=RERUN_REASON)
    posthoc = sub.add_parser("compare-rerun", help="post-hoc comparison of a rerun with complete arms (not compare)")
    posthoc.add_argument("--prep", type=Path, required=True)
    posthoc.add_argument("--rerun", type=Path, required=True)
    posthoc.add_argument("--control", type=Path, required=True)
    posthoc.add_argument("--arm", action="append", default=[], metavar="NAME=DIR",
                         help=f"a complete agent arm ({', '.join(AGENT_ARMS)}); repeatable")
    posthoc.add_argument("--source", type=Path,
                         help="the rerun's source run (default: the path its CONFIG.json records)")
    posthoc.add_argument("--out", type=Path, required=True)
    scoring = sub.add_parser("score")
    scoring.add_argument("--prep", type=Path, required=True)
    scoring.add_argument("--run", type=Path, required=True)
    return p


def main(argv=None) -> None:
    args = parser().parse_args(argv)
    for name, value in vars(args).items():
        if isinstance(value, Path):
            setattr(args, name, value.resolve())
    if args.command == "prepare":
        prepare(args.data, args.out, merged_dir=args.merged_dir)
    elif args.command == "assets":
        from benchmarks.full_HCMA.harness.assets import build
        if args.prep is not None:
            rows = load_prepared(args.prep, verify_pixels=True)
            record = read_json(args.prep / "PREPARE.json")
            for name, key in (("labels.json", "labels_sha256"), ("raw_hashes.json", "raw_hashes_sha256")):
                if digest(args.prep / "private" / name) != record[key]:
                    raise ValueError(f"prepared private metadata changed: {name}")
            print(f"PREPARED_REUSED views={len(rows)}", flush=True)
        build(args.root, dinov3_repo=args.dinov3_repo, weights=args.weights, cpsam=args.cpsam)
    elif args.command == "preflight":
        write_json(args.out, preflight(args))
        print("PREFLIGHT_OK")
    elif args.command == "verify-prepared":
        verify_prepared(args.prep, args.out)
    elif args.command == "run":
        run(args)
    elif args.command == "stage":
        stage(args)
    elif args.command == "native-trial":
        native_trial(args)
    elif args.command == "compare":
        from .compare import compare
        compare(args.prep, {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm) is not None},
                args.out, args.expected_commit, args.expected_branch)
    elif args.command == "rerun":
        rerun(args)
    elif args.command == "compare-rerun":
        from .rerun import compare_rerun, named_runs
        compare_rerun(args.prep, args.rerun, args.control, named_runs(args.arm), args.out, source=args.source)
    else:
        from .score import score
        print(score(args.prep, args.run))


if __name__ == "__main__":
    main()

"""SHAPE run on the prepared `250429_96well_CO_DEFINE candidates` (all images; train/val 8:2, no evaluation set): runs the
unmodified harness stages segment, crop, embed of the pinned source snapshot, each stage in its own process (as the CO-FC
runner does; this driver never opens CUDA itself).

Channels: segment and crop read the phase-contrast image only; the embed (SSL) stage reads two-channel (PhaseContrast, GFP)
crops cut at the crop boxes chosen on the phase image (`ssl_two_channel` variant). The final embedding covers every retained
crop of every image (train and val).

usage (env SMOKE_ROOT = the run folder, CO_SRC = the pinned source snapshot):
    python full_driver.py run NAME ARM MODE        MODE = validate (10 plate-wells, full-run search settings) | full (all images)
    python full_driver.py stage NAME ARM STAGE     (called by `run`)
"""
import importlib
import json
import os
import random
import subprocess
import sys
import time
from pathlib import Path

ROOT, SRC = Path(os.environ["SMOKE_ROOT"]), Path(os.environ["CO_SRC"])
sys.path.insert(0, str(SRC))
os.chdir(SRC)

SEED = 20261024                      # the harness seed (fixed for every call of the CO-FC study code)
SUBSET_SEED = 20261025
MODEL = "gpt-5.6-luna"
ORGANOID_MODEL = os.environ.get("CO_ORGANOID_MODEL", "/scratch/groups/your-group/your-user/shape_benchmark/CO_cpsam_finetune/shape_seg_co")
ASSETS = os.environ.get("CO_ASSETS", "/scratch/users/your-user/co_fc_timecourse/assets")
PREP = ROOT / "prepared"
STUDY = "case studies.cardiac organoids.fc_timecourse.runners"        # e2e, snapshot, spec of the CO-FC study
COHORT = "case studies.cardiac organoids.runners.cohort"                # the organoid segmentation and crop cohort
STAGES = ("segment", "crop", "embed")


def modules():
    from benchmarks.full_CPJUMP1.harness import pipeline
    from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json
    return pipeline, digest, read_json, write_json, importlib.import_module(STUDY + ".e2e"), importlib.import_module(STUDY + ".snapshot")


def select_rows(rows, mode):
    """validate: all time points of 3 train and 2 val plate-wells per plate (seeded; the SSL search needs two val wells on a
    plate to be measurable); full: every image."""
    if mode == "full":
        return rows
    rng = random.Random(SUBSET_SEED)
    chosen = set()
    for plate in sorted({r["plate"] for r in rows}):
        for usage, n in (("development_train", 3), ("development_val", 2)):
            wells = sorted({r["well"] for r in rows if r["plate"] == plate and r["usage"] == usage})
            chosen |= set(rng.sample(wells, n))
    return [r for r in rows if r["well"] in chosen]


def build_config(arm, mode, pipeline, digest, read_json, e2e, expected):
    rows = select_rows(read_json(PREP / "public" / "images.json"), mode)
    prepared = read_json(PREP / "PREPARE.json")
    protocol = e2e.reasoning_protocol()
    cores = int(os.environ.get("SLURM_CPUS_PER_TASK", "8"))
    checkpoint = Path(ORGANOID_MODEL).resolve()
    config = dict(dataset="co_fc_timecourse", variant=prepared["variant"], ssl_channels=prepared["ssl_channels"],
                  stage_channels=prepared["stage_channels"], arm=arm, model=MODEL, seed=SEED, source_sha256=expected,
                  deadline=None, budget_hours=None, cores=cores, assets_root=str(Path(ASSETS).resolve()),
                  organoid_model=str(checkpoint), organoid_model_sha256=digest(checkpoint),
                  images=[r["image_id"] for r in rows],
                  # `smoke` only widens the allowed images per plate-well from 8 to 1..8 (this plate has up to 6 time
                  # points per well)
                  smoke=True, train_cells_per_site=128, inference_chunk=2048,
                  reasoning={"protocol": protocol, "stages": {s: pipeline.stage_reasoning(arm, s, protocol) for s in STAGES}},
                  shape_policy={"crop_vrr_workers": cores}, ssl_epoch_budget=pipeline.ssl_epoch_budget(rows, 128),
                  prediction_channels=sorted(set(prepared["stage_channels"]["segment"]) | set(prepared["ssl_channels"])),
                  evaluation_channels=["PhaseContrast", "GFP"],
                  # shape arm: SHAPE's segmentation cohort (organoid_cohort.py, study layer); control has no reasoner: per-image
                  segmentation_regime="cohort" if arm == "shape" else "per-image",
                  crop_mask_check=False,        # the crop VRR's mask-error check rejected a clean organoid (validation run 1)
                  agent_context_policy=e2e.AGENT_CONTEXT_POLICY, agent_context_stages=list(e2e.TOOL_STAGES),
                  agent_decision_rounds=e2e.AGENT_DECISION_ROUNDS,
                  readout="none (segment, crop and embed only; train/val 8:2 by plate-well, no evaluation set)",
                  prepare_sha256=digest(PREP / "PREPARE.json"), mode=mode, n_images=len(rows))
    return config, protocol


def preflight_shape(config, protocol, pipeline):
    from benchmarks.full_CPJUMP1.harness.gateway_route import validate
    from shape.engine import usage
    out = {"gateway_route": validate(os.environ.get("LLM_BASE_URL", ""))}
    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("missing LLM_API_KEY")
    os.environ.update(pipeline.stage_env("shape", "segment", protocol))
    before = usage.total_tokens()
    pipeline.engine_for(dict(arm="shape", model=config["model"], reasoning={"protocol": protocol}), "segment")
    out["total_tokens"] = usage.total_tokens() - before
    if out["total_tokens"] <= 0:
        raise RuntimeError("SHAPE model preflight has no observed token usage")
    return out


def run(name, arm, mode):
    pipeline, digest, read_json, write_json, e2e, snapshot = modules()
    expected = json.loads((SRC / "SOURCE_SNAPSHOT.json").read_text(encoding="utf-8"))["source_sha256"]
    source_record = snapshot.verify_source(expected)
    out = ROOT / "runs" / name
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f"run folder is not empty: {out}")
    (out / "logs").mkdir(parents=True)
    config, protocol = build_config(arm, mode, pipeline, digest, read_json, e2e, expected)
    write_json(out / "CONFIG.json", config)
    write_json(out / "SOURCE_VERIFIED.json", source_record)
    if arm == "shape":
        write_json(out / "PREFLIGHT.json", preflight_shape(config, protocol, pipeline))
    e2e.environment_record(out)
    meta = dict(status="running", arm=arm, mode=mode, stages={})
    try:
        for stage in STAGES:
            env = dict(os.environ, SHAPE_ORGANOID_MODEL_PATH=ORGANOID_MODEL)
            env.update(pipeline.stage_env(arm, stage, protocol))
            print("STAGE_START", stage, time.strftime("%H:%M:%S"), flush=True)
            with open(out / "logs" / f"{stage}.log", "w") as log:
                code = subprocess.run([sys.executable, __file__, "stage", name, arm, stage], env=env, stdout=log, stderr=subprocess.STDOUT).returncode
            if code:
                raise RuntimeError(f"stage {stage} exited with {code}; see {out / 'logs' / (stage + '.log')}")
            meta["stages"][stage] = read_json(out / f"{stage}_COMPLETE.json")
            print("STAGE_COMPLETE", stage, time.strftime("%H:%M:%S"), flush=True)
        meta["status"] = "complete"
    except BaseException as exc:
        meta.update(status="failed", error=type(exc).__name__ + ": " + str(exc))
        raise
    finally:
        write_json(out / "RUN_META.json", meta)
    (out / "RUN_COMPLETE").write_text(expected + "\n")


def stage(name, arm, stage_name):
    pipeline, digest, read_json, write_json, e2e, snapshot = modules()
    from benchmarks.full_CPJUMP1.harness.cli import asset_environment
    from shape.engine import usage
    out = ROOT / "runs" / name
    config = read_json(out / "CONFIG.json")
    snapshot.verify_source(config["source_sha256"])
    wanted = set(config["images"])
    rows = [r for r in read_json(PREP / "public" / "images.json") if r["image_id"] in wanted]
    if len(rows) != len(wanted):
        raise RuntimeError("config images are not all in the prepared table")
    protocol = config["reasoning"]["protocol"]
    if config.get("segmentation_regime") == "cohort":      # the organoid segmentation cohort replaces the harness's per-image stage
        importlib.import_module(COHORT + ".organoid_cohort").install()
    started = time.time()
    usage.reset()
    asset_environment(Path(config["assets_root"]))
    e2e.organoid_environment(config)
    pipeline.check_stage_env(arm, stage_name, protocol)
    try:
        pipeline.STAGES[stage_name](PREP, out, config, rows)
    finally:
        write_json(out / f"{stage_name}_USAGE.json", usage.snapshot())
    tokens = usage.snapshot()
    if arm != "control" and tokens.get("total_tokens", 0) <= 0:
        raise RuntimeError("reasoner stage has no observed token usage")
    write_json(out / f"{stage_name}_COMPLETE.json", dict(status="complete", stage=stage_name, seconds=time.time() - started, usage=tokens))


if __name__ == "__main__":
    if sys.argv[1] == "run":
        run(*sys.argv[2:5])
    elif sys.argv[1] == "stage":
        stage(*sys.argv[2:5])
    else:
        raise SystemExit(__doc__)

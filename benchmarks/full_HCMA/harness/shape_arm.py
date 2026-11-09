"""The SHAPE arm: SHAPE's own VRR loops choose the parameters of the fixed chain; every delivered call runs through the
workspace tool CLIs (control._call, logged in logs/control_commands.jsonl), so `score --run`, the candidates, the
process metrics and `attribute` read it as they read the control.

    1 setup_run on the prep and metadata level of the other arms (deadline BUDGET_HOURS)
    2 segment --name defaults                                                           S001, the control's call
      per image (SEGMENT_WORKERS images at a time, their segmenter calls taking turns on one model), in one child
      process that holds the GPU: SegmentationAgenticVerifier (trial 0 the image; its call
      at the defaults, c01, is S001's result for the image, every other call the segmenter in that process); no new
      search after the segment stage's deadline (those images stay at the defaults)
      segment --name shape_seg --images <changed> --per-image FILE                     S002, when an image's
      delivered levers differ from the defaults; then the reproduction check (mask sha256 and cell count of the
      CLI's mask against the in-process one)
    3 per image (CROP_WORKERS images at a time): CropAgenticVerifier on the mask the crop call reads (S001 or S002), its
      base kwargs the CLI's (min_area = the segmentation's min_size, the CARD defaults: margin by the tool's rule), arm C's vision
      triage with the engine; no new search after the crop stage's deadline
      crop --name shape_crop --masks-map FILE [--per-image FILE] [--cell-overrides FILE]  C001, then the
      reproduction check (every crop's box against the in-process delivered call's)
    4 embed --name shape_zero_shot --crops C001 --set seed=0 adaptation_depth=zero_shot [--exclude-cells FILE]
      E001 (arm C's rejects excluded: misses for the scorer) = trial 0 of RepresentationAgenticVerifier, whose
      gate reads quant.tier_of on site = image_id, well = parent_id, plate = source: SKIP delivers E001; RUN makes
      every trial an embed call (levers the embed CLI does not have fail the trial)
      batch_correct --name shape_batch --embedding E --batch-key source                 B001 (the calibration rule)
      MorphologicalStateVerifier (stage_map's morphological_state, the linear loop) on B's embedding, each trial
      re-clustered here with the analyzer's clustering code (cluster.py's)
      cluster --name shape_states --embedding B --set resolution=R n_neighbors=N         K001; submission/choice.json

SHAPE reads only workspace/data (images.csv, the images) and its own tool outputs: `source` is the batch, parent_id /
image_modality / magnification / source reach the loops in their request or brief text (what the Claude arm sees at
that metadata level). The reasoner is SHAPE_LLM_MODEL (no model: the arm refuses to start), built once by SHAPE's
vision-engine builder and checked by two smoke calls (a plain one and one structured question,
`vision_engine.structured_smoke`). Only a fatal LLM error (``shape.engine.errors``: no engine, authentication /
permission, the model not found, the endpoint unreachable; by exception type) stops the SHAPE run: the images not
yet started are not started, SHAPE_RUN.json records `stopped` (the stage, the image, the question label, the cause)
and the run's status "failed", the log prints it, and the driver exits with SHAPE_FAILED. Every other failure is
recorded and the run goes on: a question that failed inside a loop is the loop's (`execution_errors`: a failed
round question ends the image's search, a failed final question delivers no call -- the image keeps the stage's
input, the defaults --, a failed admission question leaves the image unsearched), a failed vision call is a failed
call the agent sees, and a search that ends in any other error is the image's outcome "failed" (the image at the
defaults).
Records (no AGENT_RUN.json, no traces/: this is not a Claude run): RUN_META.json (arm "shape", the model, the GPU,
the commit), SHAPE_RUN.json (every stage's decision, the per-image outcomes, the reproduction checks, the reasoner
usage from shape/engine/usage.py, the wall time), logs/shape/ (each loop's log and selection details); the loops'
own files are in cache/shape_arm/ (not exported).
"""
from __future__ import annotations

import contextlib
import json
import os
import shutil
import socket
import subprocess
import sys
import threading
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from typing import Callable

import numpy as np
import pandas as pd

from .control import Runner, _call
from .ledger import CONTROL_COMMANDS, calls

ARM = "shape"
RUN_RECORD = "SHAPE_RUN.json"
LOGS = "shape"                       # <run>/logs/shape/
SCRATCH = Path("cache") / "shape_arm"
LAUNCHER_DIRS = {"logs", "cache"}           # what a launcher creates in the run dir first
MODEL_ENV = "SHAPE_LLM_MODEL"
BUDGET_DEFAULT = 4.0
SEGMENT_SHARE, CROP_SHARE = 0.35, 0.75   # the stage deadlines: these shares of the budget from the start
# the searches end early enough for the calls after them: the tools refuse a call that starts less than 10 min
# before the deadline, and the crop / embed / batch / cluster calls take ~10-15 min on pairs
TAIL_MINUTES = 30
# images searched at the same time (the reasoner's latency dominates; the segmenter's GPU calls take turns, the
# cropper's run on the CPU)
SEGMENT_WORKERS, CROP_WORKERS = 4, 8
NAMES = {"segment": "shape_seg", "crop": "shape_crop", "embed": "shape_zero_shot", "batch_correct": "shape_batch",
         "cluster": "shape_states"}
FIXED_SEGMENT = {"model_type": "cpsam", "bsize": 256}      # segment.py's FIXED
CROP_GROUP = "default"                                     # crop.py's GROUP
SMOKE_PROMPT = "Reply with exactly the word OK."
# never in a tool's environment (agents._SCRUB, and the reasoner's credentials)
_TOOL_SCRUB = ("ANTHROPIC_API_KEY", "ANTHROPIC_AUTH_TOKEN", "OPENAI_API_KEY", "OPENAI_BASE_URL")
_ORIGINAL_CUDA = "FULL_HCMA_SHAPE_CUDA_VISIBLE_DEVICES"     # the node's value, kept while the driver hides the GPU
NOT_STARTED = {"status": "not_started", "levers": {}, "note": "not started: the run stopped on a fatal LLM error"}


# ---------------------------------------------------------------- shared helpers

def tools_common():
    """The repository template's tools/_common.py (the lever registry, value checks, the tool classes), loaded
    once under its own module name (coverage._registry)."""
    from .coverage import _TEMPLATE_MODULE, _registry

    _registry()
    return sys.modules[_TEMPLATE_MODULE]


def _json(path: Path):
    try:
        return json.loads(Path(path).read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def _trial_label(verifier, selected) -> str | None:
    if selected is None:
        return None
    try:
        return verifier.label(int(selected))
    except Exception:  # noqa: BLE001 -- older details may already store a label
        return str(selected)


def _write(path: Path, obj) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(f".{path.name}.tmp")
    tmp.write_text(json.dumps(obj, indent=1, default=str), encoding="utf-8")
    for attempt in range(5):     # Windows refuses to replace a file another process briefly holds open
        try:
            os.replace(tmp, path)
            return
        except PermissionError:
            if attempt == 4:
                raise
            time.sleep(0.05)


def _link(source: Path, target: Path) -> Path:
    """`target` as a hard link to `source` (a copy where the filesystem has none)."""
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.exists():
        target.unlink()
    try:
        os.link(source, target)
    except OSError:
        shutil.copyfile(source, target)
    return target


def _literal(value) -> str:
    return json.dumps(value, separators=(",", ":"))


def _cuda_env(env: dict) -> dict:
    """`env` with the node's CUDA_VISIBLE_DEVICES (the driver hides the GPU from itself)."""
    env = dict(env)
    original = env.pop(_ORIGINAL_CUDA, None)
    if original is None:
        return env
    if original == "<unset>":
        env.pop("CUDA_VISIBLE_DEVICES", None)
    else:
        env["CUDA_VISIBLE_DEVICES"] = original
    return env


def tool_runner(repo: Path) -> Runner:
    """control.subprocess_runner with the reasoner's credentials scrubbed too and the node's GPU visible."""
    from .agents import _SCRUB

    def run(workspace: Path, tool: str, args: list, log: Path) -> int:
        env = _cuda_env({k: v for k, v in os.environ.items() if k not in _SCRUB + _TOOL_SCRUB})
        env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo), os.environ.get("PYTHONPATH", "")) if p)
        with open(log, "w", encoding="utf-8") as handle:
            return subprocess.run([sys.executable, f"tools/{tool}.py", *args], cwd=workspace, env=env,
                                  stdout=handle, stderr=subprocess.STDOUT).returncode
    return run


def _usage_delta(before: dict, after: dict) -> dict:
    return {k: v - before.get(k, 0) for k, v in after.items()
            if isinstance(v, int) and not isinstance(v, bool)}


def _add_usage(total: dict, part: dict) -> dict:
    for k, v in (part or {}).items():
        if isinstance(v, int) and not isinstance(v, bool):
            total[k] = total.get(k, 0) + v
    return total


def stage_deadlines(started: float, budget_hours: float) -> dict:
    """The stage deadlines: the segment and crop searches stop at SEGMENT_SHARE / CROP_SHARE of the budget, but
    never later than TAIL_MINUTES before the run's deadline (the crop search) and the crop search's own stop (the
    segment search), so the calls after them can still start."""
    run = started + budget_hours * 3600
    crop = min(started + CROP_SHARE * budget_hours * 3600, run - TAIL_MINUTES * 60)
    return {"segment": min(started + SEGMENT_SHARE * budget_hours * 3600, crop), "crop": crop, "run": run}


def stop_record(stage: str, image: str | None, error: BaseException) -> dict:
    """What stopped the run: the stage, the image (None: a stage-level step), the reasoner question's label (the
    one `AgenticVerifier._ask` tags a fatal error with; None: not a reasoner question) and the cause."""
    return {"stage": stage, "image": image, "question": getattr(error, "shape_question", None),
            "cause": f"{type(error).__name__}: {error}"[-1000:]}


def search_failure(stage: str, image: str, error: BaseException, stopped) -> dict:
    """An image's search that raised: a fatal LLM error sets `stopped` (no further image starts) and carries the
    stop record; any other error is the image's outcome "failed" (the run goes on). Printed to the log."""
    from shape.engine.errors import is_fatal_llm_error

    outcome = {"status": "failed", "levers": {}, "error": f"{type(error).__name__}: {error}"[-1000:]}
    if is_fatal_llm_error(error):
        stopped.set()
        outcome["stop"] = stop_record(stage, image, error)
        print(stop_message(outcome["stop"]), flush=True)
    else:
        print(f"{stage} search {image} failed (the run goes on): {outcome['error']}", flush=True)
    return outcome


def stop_message(stop: dict) -> str:
    where = f" on image {stop['image']}" if stop.get("image") else ""
    question = f", the {stop['question']} question" if stop.get("question") else ""
    return f"SHAPE_FAILED: the {stop['stage']} stage stopped{where}{question}: {stop['cause']}"


def _log_to(path: Path) -> Callable[..., None]:
    """A verifier's log: one JSON line per event in `path`."""
    lock = threading.Lock()
    path.parent.mkdir(parents=True, exist_ok=True)

    def log(kind: str, **fields) -> None:
        with lock, open(path, "a", encoding="utf-8") as handle:
            handle.write(json.dumps({"time": time.time(), "kind": kind, **fields}, default=str) + "\n")
    return log


def image_facts(row: dict) -> str:
    """The technical metadata of one image as the arm's text gives it (the images.csv values; empty = not given)."""
    given = [f"{label}: {row.get(column)}" for column, label in (
        ("image_modality", "image modality"), ("magnification", "magnification"), ("source", "source (vendor or "
         "dataset)"), ("parent_id", "micrograph id (panels of one micrograph share it)")) if str(row.get(column) or "")]
    return "; ".join(given) if given else "no technical metadata is given for this image"


def segment_request(row: dict) -> str:
    """The request the segmentation verifier states its target from (its "## Target" section)."""
    return ("Single-cell morphology of cultured cells: segment the cells of this raw microscopy image.\n\n"
            "## Target\n"
            "Every cell of the culture: each cell is one instance, over its full extent. "
            f"Image facts: {image_facts(row)}.")


def build_engine(model: str | None, *, engine=None):
    """The reasoner: `engine` (tests), else SHAPE's vision engine for `model` (shape.agent.verification.vision_engine),
    checked by two smoke calls: a plain one, then one structured question (`vision_engine.structured_smoke`: the
    gateway must do structured output for the model). SystemExit without a model, without an engine or when a smoke
    call fails."""
    if not model:
        raise SystemExit(f"NOT_READY: set {MODEL_ENV} to the reasoner's gateway model id; the SHAPE arm does not start "
                         "without a reasoner")
    if engine is None:
        from shape.agent.verification.vision_engine import build_vision_engine

        engine = build_vision_engine(model)
    if engine is None:
        raise SystemExit(f"NOT_READY: no engine for {model} (see the vision engine's warning: key, base URL, provider)")
    try:
        answer = engine.generate([SMOKE_PROMPT])
    except Exception as error:  # noqa: BLE001 -- the smoke call's failure is the refusal's reason
        raise SystemExit(f"NOT_READY: the smoke call to {model} failed: {type(error).__name__}: {error}") from None
    text = answer.get("content") if isinstance(answer, dict) else answer
    if "OK" not in str(text or "").upper():
        raise SystemExit(f"NOT_READY: the smoke call to {model} answered {str(text)[:200]!r}, not OK")
    from shape.agent.verification.vision_engine import structured_smoke

    try:
        structured_smoke(engine)
    except Exception as error:  # noqa: BLE001 -- the gateway must do structured output for the model
        raise SystemExit(f"NOT_READY: the structured smoke call to {model} failed: {type(error).__name__}: "
                         f"{error}") from None
    return engine


# ---------------------------------------------------------------- 2. segmentation

def _same_levers(C, levers: list[str], params: dict, defaults: dict) -> bool:
    return all(C._same(params.get(k), defaults.get(k)) for k in levers)


def _s001_result(item: dict, scratch: Path) -> dict:
    """S001's result for one image as the segmenter returns it: its tool record, the mask (S001's .tif as the .npy
    the tool writes) and the overlay (copied)."""
    import tifffile

    record = dict(item["s001_record"] or {})
    out = {**record, "image_id": item["image_id"]}
    if item.get("s001_mask"):
        mask = np.asarray(tifffile.imread(item["s001_mask"])).astype(np.int32)
        np.save(scratch / f"cell_mask_{item['image_id']}_s001.npy", mask)
        out["mask_path"] = str(scratch / f"cell_mask_{item['image_id']}_s001.npy")
    else:
        out.setdefault("error", item.get("s001_error") or "S001 has no mask for this image")
    if item.get("s001_overlay") and Path(item["s001_overlay"]).is_file():
        overlay = shutil.copyfile(item["s001_overlay"], scratch / f"cell_overlay_{item['image_id']}_s001.png")
        out["overlay_path"] = out["output_path"] = str(overlay)
    return out


def segment_one(C, tool, item: dict, levers: list[str], defaults: dict, engine, scratch: Path, log_dir: Path,
                lock=None, budget: int | None = None, verifier_options: dict | None = None,
                tool_name: str = "Cell_Segmenter_Tool", fixed: dict | None = None) -> dict:
    """`segment_resolution`'s outcome."""
    return segment_resolution(C, tool, item, levers, defaults, engine, scratch, log_dir, lock, budget, verifier_options,
                              tool_name=tool_name, fixed=fixed)[0]


def segment_resolution(C, tool, item: dict, levers: list[str], defaults: dict, engine, scratch: Path, log_dir: Path,
                       lock=None, budget: int | None = None, verifier_options: dict | None = None,
                       tool_name: str = "Cell_Segmenter_Tool", fixed: dict | None = None):
    """One image's agentic segmentation (as benchmarks/cropper_NeurIPS22-CellSeg/run_shape_crop.py drives its
    verifier): SegmentationAgenticVerifier through StepVerifier's wave mode, trial 0 the image (`input_record`),
    the call at the defaults answered by S001's result, every other call `stage_map.invoke_managed_tool` on the
    segmenter (under `lock`: one model, one call at a time). Returns the image's outcome: the delivered call, its
    levers that differ from the defaults, its mask (.npy) and the loop's record; returns (the outcome, the stage's
    StageResolution). `verifier_options` ({attribute: value}) are set on the verifier as
    `Verifier.resolve_managed_stage` sets them (the segmentation cohort's `cohort_levers` / `start_params`)."""
    from shape.agent.verification.stage_map import invoke_managed_tool, stage_by_name, verifier_cls_for
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.agent.verification.verifiers.segmentation import input_record

    # `tool_name` selects the segmenter and its verifier (`stage_map.verifier_cls_for`: Cellpose-SAM's
    # SegmentationAgenticVerifier, the organoid segmenter's OrganoidSegmentationAgenticVerifier); `fixed` are the
    # tool kwargs no arm sets (None: Cellpose-SAM's FIXED_SEGMENT).
    image_id = item["image_id"]
    fixed = dict(FIXED_SEGMENT if fixed is None else fixed)
    scratch.mkdir(parents=True, exist_ok=True)
    base = {"image": item["path"], "image_id": image_id}
    reused = {"s001": None}

    def runner(params: dict) -> dict:
        if reused["s001"] is None and _same_levers(C, levers, params, defaults):
            reused["s001"] = _s001_result(item, scratch)
            return dict(reused["s001"])
        with lock or contextlib.nullcontext():
            return invoke_managed_tool(tool, tool_name, {**base, **fixed}, params, str(scratch))

    options = dict(verifier_options or {})
    construction = {k: options.pop(k) for k in ('gain_policy', 'native_gain_noise') if k in options}
    Verifier = verifier_cls_for(stage_by_name("segmentation"), tool_name)
    verifier = Verifier(log=_log_to(log_dir / f"{image_id}.jsonl"), **construction)
    for lever, value in defaults.items():
        if lever in verifier.lever_fields:
            verifier.lever_fields[lever]["Default"] = json.dumps(value)
    verifier.llm_engine = engine
    if budget is not None:
        verifier.budget = int(budget)
    verifier.request = item["request"]
    for name, value in options.items():
        if not hasattr(verifier, name):
            raise ValueError(f"{type(verifier).__name__} has no option {name!r}")
        setattr(verifier, name, value)
    started = time.time()
    res = StepVerifier(max_iters=1 + verifier.budget).resolve_stage(
        verifier, verifier.make_adapter(base), runner, {}, input_record(base), upstream_ctx={})
    selected = res.details.get("selected")
    delivered = res.result if selected else None
    config = {k: res.params.get(k) for k in levers} if delivered is not None else dict(defaults)
    changed = {k: config[k] for k in levers if not C._same(config[k], defaults.get(k))}
    outcome = {"status": "searched", "delivered": _trial_label(verifier, selected),
               "defaults_call_reused": reused["s001"] is not None, "levers": changed,
               "mask_npy": (delivered or {}).get("mask_path") if changed else None,
               "cell_count": (delivered or {}).get("cell_count"), "resolution": {"status": res.status,
               "reason": res.reason, "fault": res.fault}, "admission": verifier.admission,
               "stop_reason": verifier.stop_reason, "n_calls": len(res.trials) - 1,
               "reasoner_failures": verifier.reasoner_failures,
               "fallback_rounds": verifier.fallback_rounds,
               "execution_errors": verifier.execution_errors,
               "seconds": round(time.time() - started, 1)}
    if delivered is None:
        outcome["note"] = f"no call delivered ({res.reason}): the image stays at the defaults (S001)"
    _write(log_dir / f"{image_id}.details.json", {"outcome": outcome, "params": res.params, "details": res.details})
    return outcome, res


def segment_search(job: dict, *, engine=None) -> dict:
    """The per-image segmentation searches of one run (in the child process `python -m
    benchmarks.full_HCMA.harness.shape_arm segment-search JOB.json`, which holds the GPU; tests call it directly).
    `job`: {"config": the workspace's tools/_config.json, "items": [{image_id, path, request, s001_record, s001_mask,
    s001_overlay, s001_error}], "levers", "defaults" (S001's params), "stop_at" (epoch: no new search after it),
    "model", "scratch", "log_dir", "workers" (images at a time, one segmenter instance)}. Returns {"images":
    {image_id: outcome}, "usage": the reasoner usage}; a search that raised is an outcome "failed" (with its `stop`
    record, `stop_record`, when the error is fatal: the images not started after it are "not_started")."""
    from shape.engine import usage

    C = tools_common()
    cfg = json.loads(Path(job["config"]).read_text(encoding="utf-8"))
    C.setup(cfg, gpu=True)
    if engine is None and job.get("model"):
        from shape.agent.verification.vision_engine import build_vision_engine

        engine = build_vision_engine(job["model"])
        if engine is None:
            raise SystemExit(f"NOT_READY: no engine for {job['model']} in the segmentation child")
    tool, lock = C.tool_class("Cell_Segmenter_Tool", cfg)(), threading.Lock()
    before = usage.snapshot()
    stopped = threading.Event()               # a failed search: no further image starts

    def one(item: dict) -> tuple[str, dict]:
        image_id = item["image_id"]
        if stopped.is_set():
            return image_id, dict(NOT_STARTED)
        if time.time() >= float(job["stop_at"]):
            return image_id, {"status": "left_at_defaults", "levers": {},
                              "note": "the segment stage's deadline passed before this image's search"}
        try:
            outcome = segment_one(C, tool, item, job["levers"], job["defaults"], engine,
                                  Path(job["scratch"]) / image_id, Path(job["log_dir"]), lock)
        except Exception as error:  # noqa: BLE001 -- fatal: the run stops (Arm.segment); else the image's outcome
            outcome = search_failure("segment", image_id, error, stopped)
        print(f"segment search {image_id}: {outcome['status']} {outcome.get('levers')}", flush=True)
        return image_id, outcome

    with ThreadPoolExecutor(max_workers=max(1, int(job.get("workers") or 1))) as pool:
        out = dict(pool.map(one, job["items"]))
    return {"images": out, "usage": _usage_delta(before, usage.snapshot())}


def _segment_items(ws: Path, s001: Path, rows: list[dict], images: list[str]) -> list[dict]:
    per_image = {r["image_id"]: r for r in pd.read_csv(s001 / "per_image.csv", dtype=str, keep_default_na=False)
                 .to_dict("records")}
    records = {}
    path = s001 / "tool_results.jsonl"
    for line in path.read_text(encoding="utf-8").splitlines() if path.exists() else []:
        record = json.loads(line)
        records[record.pop("image_id")] = record
    by_id = {r["image_id"]: r for r in rows}
    items = []
    for image_id in images:
        row, seg = by_id[image_id], per_image.get(image_id, {})
        mask = s001 / "masks" / f"{image_id}.tif"
        items.append({"image_id": image_id, "path": str(ws / "data" / "images" / Path(row["file"]).name),
                      "request": segment_request(row), "s001_record": records.get(image_id),
                      "s001_mask": str(mask) if seg.get("status") == "done" and mask.is_file() else None,
                      "s001_overlay": str(s001 / "overlays" / f"{image_id}.png"),
                      "s001_error": seg.get("error") or None})
    return items


def _run_child_search(job_path: Path, out_path: Path, repo: Path, log: Path) -> dict:
    env = _cuda_env(dict(os.environ))
    env.pop("SHAPE_SAM_SERVER_URL", None)          # the segmenter in process, as in the segment CLI (agents._SCRUB)
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(repo), os.environ.get("PYTHONPATH", "")) if p)
    with open(log, "w", encoding="utf-8") as handle:
        code = subprocess.run([sys.executable, "-m", "benchmarks.full_HCMA.harness.shape_arm", "segment-search",
                               str(job_path), str(out_path)], cwd=repo, env=env, stdout=handle,
                              stderr=subprocess.STDOUT).returncode
    if code != 0 or not out_path.exists():
        tail = "\n".join(log.read_text(encoding="utf-8", errors="replace").splitlines()[-20:])
        raise SystemExit(f"SHAPE_FAILED: the segmentation child exited {code}; log {log}:\n{tail}")
    return json.loads(out_path.read_text(encoding="utf-8"))


def _mask_check(cli_mask: Path, npy: str | None) -> dict:
    """The CLI's mask (.tif, uint16) against the in-process delivered mask (.npy, stored as uint16 by segment.py)."""
    import hashlib

    import tifffile

    cli = np.asarray(tifffile.imread(str(cli_mask))).astype(np.uint16)
    out = {"cli_cells": int(len(np.unique(cli)) - (1 if (cli == 0).any() else 0)),
           "cli_sha256": hashlib.sha256(np.ascontiguousarray(cli).tobytes()).hexdigest()}
    if not npy or not Path(npy).is_file():
        return {**out, "identical": None, "note": "no in-process mask to compare"}
    mine = np.asarray(np.load(npy)).astype(np.uint16)
    out.update(inprocess_cells=int(len(np.unique(mine)) - (1 if (mine == 0).any() else 0)),
               inprocess_sha256=hashlib.sha256(np.ascontiguousarray(mine).tobytes()).hexdigest())
    out["identical"] = out["cli_sha256"] == out["inprocess_sha256"] and cli.shape == mine.shape
    return out


# ---------------------------------------------------------------- 3. crop

def crop_brief(row: dict) -> str:
    return (f"Image {row['image_id']} of cultured cells; its square single-cell crops feed a DINOv3 embedding of "
            f"each cell's morphology. Image facts: {image_facts(row)}.")


def crop_one(C, tool_class, item: dict, engine, deadline: float, cores: int, scratch: Path, log_dir: Path,
             budget: int | None = None, *, mask_check: bool = True) -> dict:
    """One image's agentic crop (run_shape_crop.run_shape_agentic_arm's flow): CropAgenticVerifier through
    StepVerifier's wave mode on the mask the crop call will read, its base kwargs the crop CLI's (min_area = the
    segmentation's min_size; no margin: the tool's rule, the CLI's default), trial 0 the input record, every call the cropper. Returns the delivered call's
    levers that differ from the CLI's defaults, its cell_overrides, arm C's rejected cells and its crop boxes."""
    from shape.agent.verification.step_verifier import StepVerifier
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier, CropTask, input_record

    image_id = item["image_id"]
    scratch.mkdir(parents=True, exist_ok=True)
    alias = _link(Path(item["mask"]), scratch / "masks" / f"cell_mask_{image_id}.tif")    # crop.py's mask name
    base = {**dict(item["defaults"]), "original_image": item["path"], "nuclei_mask": str(alias),
            "min_area": item["min_area"], "source_image_id": image_id, "group": CROP_GROUP}
    tool = tool_class()

    def runner(params: dict) -> dict:
        result = tool.execute(original_image=item["path"], nuclei_mask=str(alias), source_image_id=image_id,
                              group=CROP_GROUP, output_format="png", query_cache_dir=str(scratch), **params)
        if result.get("error") or not result.get("cell_count"):
            raise RuntimeError(str(result.get("error") or "the tool produced no crops"))
        return result

    verifier = CropAgenticVerifier(log=_log_to(log_dir / f"{image_id}.jsonl"))
    for lever, value in item["defaults"].items():
        if lever in verifier.lever_fields:
            verifier.lever_fields[lever]["Default"] = json.dumps(value)
    verifier.llm_engine = engine
    if budget is not None:
        verifier.budget = int(budget)
    started = time.time()
    res = StepVerifier(max_iters=1 + verifier.budget).resolve_stage(
        verifier, verifier.make_adapter(base), runner, {}, input_record(base),
        upstream_ctx={"crop_task": CropTask(cores=cores, deadline=deadline, brief=item["brief"], enable_mask_check=mask_check)})
    selected = res.details.get("selected")
    defaults = item["defaults"]
    outcome = {"status": "searched", "delivered": _trial_label(verifier, selected), "levers": {},
               "cell_overrides": {}, "rejected_tool_cells": [], "boxes": {},
               "resolution": {"status": res.status, "reason": res.reason, "fault": res.fault},
               "admission": verifier.admission, "stop_reason": verifier.stop_reason, "n_calls": len(res.trials) - 1,
               "reasoner_failures": verifier.reasoner_failures,
               "fallback_rounds": verifier.fallback_rounds,
               "execution_errors": verifier.execution_errors,
               "seconds": round(time.time() - started, 1)}
    if selected and res.result is not None:
        config = verifier.trial_config(res.params)
        outcome["levers"] = {k: config.get(k) for k in verifier.levers if not C._same(config.get(k), defaults.get(k))}
        outcome["cell_overrides"] = {cid: dict(v) for cid, v in (config.get("cell_overrides") or {}).items() if v}
        outcome["rejected_tool_cells"] = list((verifier._reports.get(selected) or {}).get("mask_check_rejected_cells") or [])
        metadata = _json(Path(res.result.get("cell_crops_metadata_path") or "")) or {}
        outcome["boxes"] = {m.get("cell_id"): m.get("crop_bbox") for m in metadata.get("cell_metadata") or []}
    else:
        outcome["note"] = f"no call delivered ({res.reason}): the image is cropped at the CLI's defaults"
    _write(log_dir / f"{image_id}.details.json", {"outcome": outcome, "params": res.params, "details": res.details})
    return outcome


# ---------------------------------------------------------------- 4. representation and states

def representation_meta(cells: pd.DataFrame, rows: list[dict]) -> tuple[list[dict], dict]:
    """The crops' metadata the representation verifier reads, per embedded cell (cells.csv order): cell_id,
    image_name, and quant's identities site = image_id, well = parent_id, plate = source (blank = absent), with a
    label-free split (quant.assign_val_wells over the plates' wells, else every cell "train"). Returns (meta, the
    split rule)."""
    from shape.tools.cell_state_analyzer import quant

    by_id = {r["image_id"]: r for r in rows}

    def value(row, column):
        text = str(row.get(column) or "").strip()
        return text or None

    meta = [{"cell_id": c, "image_name": i, "site": i, "well": value(by_id[i], "parent_id"),
             "plate": value(by_id[i], "source")} for c, i in zip(cells["cell_id"], cells["image_id"])]
    try:
        split = quant.assign_val_wells(meta)
        rule = "quant.assign_val_wells (val_fraction 0.1, seed 0) over plate = source, well = parent_id"
    except ValueError as error:
        split, rule = ["train"] * len(meta), f"every cell train: {error}"
    for m, part in zip(meta, split):
        m["split"] = part
    return meta, {"rule": rule, "n_val": int(sum(s == "val" for s in split))}


def representation_brief(meta: list[dict]) -> str:
    plates = {m["plate"] for m in meta if m["plate"]}
    wells = {m["well"] for m in meta if m["well"]}
    return (f"{len(meta)} single-cell crops (square, cut from raw micrographs of cultured cells) from "
            f"{len({m['site'] for m in meta})} images, {len(wells)} micrographs and {len(plates)} sources; a crop's "
            "site is its image, its well its micrograph, its plate its source (blank when not given).")


def embed_set_args(C, params: dict, defaults: dict) -> list[str]:
    """A representation trial's params as embed CLI `--set` items: the CLI's levers (EMBED_LEVERS) that differ from
    the verifier's defaults; a changed lever the CLI does not have, or a resume state, raises ValueError (the trial
    fails: the CLI cannot make that call)."""
    items, missing = [], []
    for key, value in params.items():
        if key in ("output_subdir", "split_key", "val_fraction", "max_minutes", "init_from") or value is None:
            continue
        if key == "init_checkpoint":
            missing.append("init_from=parent (a resume state)")
        elif key in C.EMBED_LEVERS:
            if not C._same(value, C.normal_default(C.EMBED_LEVERS[key])):
                items.append(f"{key}={_literal(value)}")
        elif not C._same(value, defaults.get(key)):
            missing.append(f"{key}={value!r}")
    if missing:
        raise ValueError(f"the embed CLI has no lever for {', '.join(missing)}")
    return items


def _embed_result(run: Path, call_id: str, scratch: Path, subdir: str) -> dict:
    """An embed call as the analyzer's result: its tool_result.json, its seconds, and its h5ad linked into
    `scratch/<subdir>/` (the verifier's delivery links the delivered trial's h5ad up its output_subdir, so it never
    writes into the call directory)."""
    record = calls(run)[call_id]
    result = dict(_json(record["dir"] / "tool_result.json") or {})
    result["adata_path"] = str(_link(record["dir"] / "cell_state_analyzed.h5ad",
                                     scratch / subdir / "cell_state_analyzed.h5ad"))
    result["seconds"] = (record["end"] or {}).get("seconds")
    result["embed_call"] = call_id
    return result


def morphology_runner(C, cfg: dict, embedding: np.ndarray, cells: pd.DataFrame, rows: list[dict], scratch: Path):
    """The morphological_state stage's runner: each trial's clustering of the delivered embedding with the
    analyzer's code (cluster.py's `_compute_umap_and_clustering`) in an h5ad whose obs carry image_name, group
    "default" and the analyzer's batch (`resolve_technical_metadata` with batch = source)."""
    import anndata as ad

    from shape.tools.cell_state_analyzer.tool import resolve_technical_metadata

    by_id = {r["image_id"]: r for r in rows}
    meta = [{"cell_id": c, **({"batch": by_id[i]["source"]} if str(by_id[i].get("source") or "").strip() else {})}
            for c, i in zip(cells["cell_id"], cells["image_id"])]
    _, batches, _ = resolve_technical_metadata(meta, list(cells["image_id"]))
    analyzer = C.tool_class("Cell_State_Analyzer_Tool", cfg)
    scratch.mkdir(parents=True, exist_ok=True)

    def runner(params: dict) -> dict:
        resolution, neighbors = float(params["leiden_resolution"]), int(round(float(params["n_neighbors"])))
        adata = ad.AnnData(X=np.asarray(embedding, dtype=np.float32))
        adata.obs_names = list(cells["cell_id"])
        adata.obs["image_name"] = list(cells["image_id"])
        adata.obs["group"] = "default"
        adata.obs["batch"] = [str(b) for b in batches]
        key = analyzer()._compute_umap_and_clustering(adata, resolution, None, neighbors, n_pcs=None)
        path = scratch / f"states_r{resolution:g}_n{neighbors}.h5ad"
        adata.write_h5ad(path)
        return {"adata_path": str(path), "cluster_resolution": resolution, "n_neighbors": neighbors,
                "cluster_key": key}
    return runner


# ---------------------------------------------------------------- the driver

class Arm:
    """One SHAPE-arm run: the workspace, the CLI calls and the record (SHAPE_RUN.json)."""

    def __init__(self, run: Path, repo: Path, runner: Runner, engine, model: str, budget_hours: float,
                 started: float):
        self.run, self.repo, self.runner, self.engine, self.model = run, repo, runner, engine, model
        self.ws = run / "workspace"
        self.scratch, self.logs = run / SCRATCH, run / "logs" / LOGS
        self.scratch.mkdir(parents=True, exist_ok=True)
        self.logs.mkdir(parents=True, exist_ok=True)
        self.cfg = json.loads((self.ws / "tools" / "_config.json").read_text(encoding="utf-8"))
        self.C = tools_common()
        self.rows = pd.read_csv(self.ws / "data" / "images.csv", dtype=str, keep_default_na=False).to_dict("records")
        self.by_id = {r["image_id"]: r for r in self.rows}
        self.started, self.budget_hours = started, budget_hours
        self.segment_workers = SEGMENT_WORKERS
        self.deadline = started + budget_hours * 3600
        self.record = {"arm": ARM, "model": model, "budget_hours": budget_hours, "started": started,
                       "stage_deadlines": stage_deadlines(started, budget_hours),
                       "stages": {}, "usage": {}, "calls": []}
        self.stage = None                        # the step running (run_shape), for a stage-level failure

    def save(self) -> None:
        self.record["wall_hours"] = round((time.time() - self.started) / 3600, 4)
        _write(self.run / RUN_RECORD, self.record)

    def stop_on_failure(self, stage: dict, order: list[str]) -> None:
        """A stage with a search stopped by a fatal LLM error stops the run: its `stopped` (the first such image in
        the stage's order) in the stage and the record, saved, then SystemExit SHAPE_FAILED."""
        failed = [i for i in order if "stop" in (stage["per_image"].get(i) or {})]
        if not failed:
            return
        stop = stage["per_image"][failed[0]]["stop"]
        stage["stopped"] = self.record["stopped"] = stop
        self.save()
        raise SystemExit(stop_message(stop))

    def call(self, tool: str, name: str, args: list, statuses=None) -> str:
        call_id = _call(self.run, self.runner, tool, name, args, *(() if statuses is None else (statuses,)))
        self.record["calls"].append({"call": call_id, "tool": tool, "name": name})
        self.save()
        return call_id

    def input_file(self, name: str, obj) -> Path:
        path = self.scratch / "inputs" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(obj if isinstance(obj, str) else json.dumps(obj, indent=1, sort_keys=True), encoding="utf-8")
        return path

    def call_dir(self, call_id: str) -> Path:
        return calls(self.run)[call_id]["dir"]

    def ledger_params(self, call_id: str) -> dict:
        return dict((calls(self.run)[call_id]["start"] or {}).get("params") or {})

    # -- 2
    def segment(self, images: list[str] | None, seg_workers: int, search: Callable | None) -> dict[str, str]:
        """S001, the per-image searches, S002; returns {image_id: the segment call whose mask the crop reads}."""
        args = (["--images", ",".join(images)] if images else []) + ["--workers", str(seg_workers)]
        s001 = self.call("segment", "defaults", args)
        s001_dir = self.call_dir(s001)
        per_image = pd.read_csv(s001_dir / "per_image.csv", dtype=str, keep_default_na=False)
        order = list(per_image["image_id"])
        levers = list(self.C.tool_levers("segment", self.cfg["repo"]))
        defaults = {k: v for k, v in self.ledger_params(s001).items() if k in levers}
        job = {"config": str(self.ws / "tools" / "_config.json"), "items": _segment_items(self.ws, s001_dir, self.rows,
               order), "levers": levers, "defaults": defaults, "stop_at": self.record["stage_deadlines"]["segment"],
               "model": self.model, "scratch": str(self.scratch / "segment"), "log_dir": str(self.logs / "segment"),
               "workers": self.segment_workers}
        job_path = self.input_file("segment_search_job.json", job)
        if search is None:
            result = _run_child_search(job_path, self.scratch / "segment_search_result.json", self.repo,
                                       self.logs / "segment_search.log")
        else:
            result = search(job)
        outcomes = result["images"]
        changed = {i: o["levers"] for i, o in outcomes.items() if o.get("levers")}
        stage = {"defaults_call": s001, "n_images": len(order), "per_image": outcomes, "usage": result.get("usage"),
                 "n_searched": sum(o["status"] == "searched" for o in outcomes.values()),
                 "left_at_defaults": sorted(i for i, o in outcomes.items() if o["status"] == "left_at_defaults"),
                 "failed": sorted(i for i, o in outcomes.items() if o["status"] == "failed"),
                 "not_started": sorted(i for i, o in outcomes.items() if o["status"] == "not_started"),
                 "changed": sorted(changed), "per_image_call": None, "reproduction": {}}
        self.record["stages"]["segment"] = stage
        _add_usage(self.record["usage"].setdefault("total", {}), result.get("usage"))
        self.record["usage"]["segment"] = result.get("usage")
        self.save()
        self.stop_on_failure(stage, order)
        mask_of = {i: s001 for i in order if (per_image.set_index("image_id").at[i, "status"] == "done")}
        if changed:
            s002 = self.call("segment", NAMES["segment"], [
                "--images", ",".join(i for i in order if i in changed),
                "--per-image", str(self.input_file("segment_per_image.json", changed)), "--workers", str(seg_workers),
                "--note", "SHAPE: the levers SegmentationAgenticVerifier delivered per image (logs/shape/segment/)"],
                statuses=("done", "partial"))
            stage["per_image_call"] = s002
            s002_dir = self.call_dir(s002)
            done = set(pd.read_csv(s002_dir / "per_image.csv", dtype=str).query("status == 'done'")["image_id"])
            for image_id in sorted(changed):
                if image_id in done:
                    mask_of[image_id] = s002
                    stage["reproduction"][image_id] = _mask_check(s002_dir / "masks" / f"{image_id}.tif",
                                                                  outcomes[image_id].get("mask_npy"))
                else:
                    stage["reproduction"][image_id] = {"identical": None, "note": f"{s002} has no mask for it: the "
                                                       f"crop reads {mask_of.get(image_id) or 'no mask'}"}
            stage["mismatches"] = sorted(i for i, r in stage["reproduction"].items() if r.get("identical") is False)
        self.save()
        print(f"SHAPE segment: {stage['n_searched']} searched, {len(changed)} with own levers, "
              f"{len(stage['left_at_defaults'])} left at the defaults, {len(stage['failed'])} failed; reproduction "
              f"mismatches {stage.get('mismatches', [])}", flush=True)
        return mask_of

    # -- 3
    def crop(self, mask_of: dict[str, str], crop_workers: int) -> tuple[str, list[str]]:
        """The per-image crop searches and C001; returns (C001, the cells to exclude: arm C's rejects)."""
        from shape.agent.verification.base import available_cores
        from shape.engine import usage

        levers = self.C.tool_levers("crop", self.cfg["repo"])
        defaults = {k: self.C.normal_default(v) for k, v in levers.items() if v.kind != "per_cell"}
        min_area = {}
        for s_id in sorted(set(mask_of.values())):
            table = pd.read_csv(self.call_dir(s_id) / "per_image.csv", dtype=str, keep_default_na=False)
            min_area.update({(s_id, r["image_id"]): int(float(r["min_size"])) for r in table.to_dict("records")
                             if r.get("min_size")})
        items = [{"image_id": i, "path": str(self.ws / "data" / "images" / Path(self.by_id[i]["file"]).name),
                  "mask": str(self.call_dir(s) / "masks" / f"{i}.tif"), "min_area": min_area.get((s, i)),
                  "defaults": defaults, "brief": crop_brief(self.by_id[i])} for i, s in mask_of.items()]
        stop_at = self.record["stage_deadlines"]["crop"]
        workers = max(1, int(crop_workers))
        cores = max(1, available_cores() // workers)
        tool_class = self.C.tool_class("Single_Cell_Cropper_Tool", self.cfg)
        before = usage.snapshot()

        def one(item: dict) -> tuple[str, dict]:
            if time.time() >= stop_at:
                return item["image_id"], {"status": "left_at_defaults", "levers": {}, "cell_overrides": {},
                                          "rejected_tool_cells": [], "boxes": {},
                                          "note": "the crop stage's deadline passed before this image's search"}
            if stopped.is_set():
                return item["image_id"], {**NOT_STARTED, "cell_overrides": {}, "rejected_tool_cells": [], "boxes": {}}
            try:
                return item["image_id"], crop_one(self.C, tool_class, item, self.engine, stop_at, cores,
                                                  self.scratch / "crop" / item["image_id"], self.logs / "crop")
            except Exception as error:  # noqa: BLE001 -- fatal: the run stops (stop_on_failure); else recorded
                return item["image_id"], {"cell_overrides": {}, "rejected_tool_cells": [], "boxes": {},
                                          **search_failure("crop", item["image_id"], error, stopped)}

        stopped = threading.Event()               # a failed search: no further image starts
        with ThreadPoolExecutor(max_workers=workers) as pool:
            outcomes = dict(pool.map(one, items))
        crop_usage = _usage_delta(before, usage.snapshot())
        if any("stop" in o for o in outcomes.values()):     # a fatal LLM error: the run stops
            for outcome in outcomes.values():
                outcome.pop("boxes", None)
            stage = {"per_image": outcomes, "usage": crop_usage, "workers": workers, "cores_per_image": cores,
                     "failed": sorted(i for i, o in outcomes.items() if o["status"] == "failed"),
                     "not_started": sorted(i for i, o in outcomes.items() if o["status"] == "not_started")}
            self.record["stages"]["crop"] = stage
            self.record["usage"]["crop"] = crop_usage
            _add_usage(self.record["usage"].setdefault("total", {}), crop_usage)
            self.stop_on_failure(stage, [i["image_id"] for i in items])
        per_image = {i: o["levers"] for i, o in outcomes.items() if o["levers"]}
        overrides = {i: o["cell_overrides"] for i, o in outcomes.items() if o["cell_overrides"]}
        args = ["--masks-map", str(self.input_file("crop_masks_map.json", mask_of))]
        if per_image:
            args += ["--per-image", str(self.input_file("crop_per_image.json", per_image))]
        if overrides:
            args += ["--cell-overrides", str(self.input_file("crop_cell_overrides.json", overrides))]
        c001 = self.call("crop", NAMES["crop"], args + [
            "--workers", str(max(1, min(16, os.cpu_count() or 1))),
            "--note", "SHAPE: the levers and cell rules CropAgenticVerifier delivered per image (logs/shape/crop/)"],
            statuses=("done", "partial"))
        crops = pd.read_csv(self.call_dir(c001) / "crops.csv", dtype=str, keep_default_na=False)
        cell_of = {(r["image_id"], r["tool_cell_id"]): r["cell_id"] for r in crops.to_dict("records")}
        excluded, reproduction = [], {}
        for image_id, outcome in sorted(outcomes.items()):
            excluded += [cell_of[(image_id, t)] for t in outcome["rejected_tool_cells"] if (image_id, t) in cell_of]
            if outcome.get("boxes"):
                mine = crops[crops["image_id"] == image_id]
                cli = {r["tool_cell_id"]: json.loads(r["crop_bbox"]) if r["crop_bbox"] else None
                       for r in mine.to_dict("records")}
                theirs = {k: list(v) if v is not None else None for k, v in outcome["boxes"].items()}
                differ = sorted(k for k in set(cli) | set(theirs) if cli.get(k) != theirs.get(k))
                reproduction[image_id] = {"n_cli": len(cli), "n_inprocess": len(theirs), "identical": not differ,
                                          "differing_cells": differ[:20]}
        for outcome in outcomes.values():
            outcome.pop("boxes", None)
        stage = {"call": c001, "per_image": outcomes, "usage": crop_usage, "workers": workers, "cores_per_image": cores,
                 "n_searched": sum(o["status"] == "searched" for o in outcomes.values()),
                 "left_at_defaults": sorted(i for i, o in outcomes.items() if o["status"] == "left_at_defaults"),
                 "failed": sorted(i for i, o in outcomes.items() if o["status"] == "failed"),
                 "images_with_own_levers": sorted(per_image), "images_with_cell_overrides": sorted(overrides),
                 "n_excluded_cells": len(excluded), "reproduction": reproduction,
                 "mismatches": sorted(i for i, r in reproduction.items() if not r["identical"])}
        self.record["stages"]["crop"] = stage
        self.record["usage"]["crop"] = crop_usage
        _add_usage(self.record["usage"].setdefault("total", {}), crop_usage)
        self.save()
        print(f"SHAPE crop: {stage['n_searched']} searched, {len(per_image)} with own levers, {len(overrides)} with "
              f"cell rules, {len(excluded)} cells excluded, {len(stage['left_at_defaults'])} left at the defaults, "
              f"{len(stage['failed'])} failed; reproduction mismatches {stage['mismatches']}", flush=True)
        return c001, excluded

    # -- 4
    def represent(self, c001: str, excluded: list[str]) -> str:
        """E001 (trial 0) and RepresentationAgenticVerifier's decision; returns the delivered embed call."""
        from shape.agent.verification.base import available_cores
        from shape.agent.verification.step_verifier import StepVerifier
        from shape.agent.verification.verifiers.representation import (RepresentationAgenticVerifier,
                                                                         RepresentationTask, frozen_params)
        from shape.engine import usage

        args = ["--crops", c001, "--set", "seed=0", "adaptation_depth=zero_shot"]
        if excluded:
            args += ["--exclude-cells", str(self.input_file("embed_exclude_cells.txt", "\n".join(excluded) + "\n"))]
        e001 = self.call("embed", NAMES["embed"], args + [
            "--note", "SHAPE: trial 0 of RepresentationAgenticVerifier (the frozen backbone); arm C's rejected crops "
                      "excluded"], statuses=("done",))
        cells = pd.read_csv(self.call_dir(e001) / "cells.csv", dtype=str, keep_default_na=False)
        meta, split = representation_meta(cells, self.rows)
        scratch = self.scratch / "representation"
        task = RepresentationTask(cores=available_cores(), gpus=1, deadline=self.deadline - 1800,
                                  brief=representation_brief(meta), split_key="split")
        made = {}                               # output_subdir -> embed call
        initial = frozen_params(task)
        made[initial["output_subdir"]] = e001
        verifier = RepresentationAgenticVerifier(log=_log_to(self.logs / "representation.jsonl"))
        verifier.llm_engine = self.engine
        defaults = verifier.defaults()

        def runner(params: dict) -> dict:
            set_args = embed_set_args(self.C, params, defaults)
            name = f"shape_rep_{len(made):02d}"
            exclude = ["--exclude-cells", str(self.scratch / "inputs" / "embed_exclude_cells.txt")] if excluded else []
            try:
                call_id = self.call("embed", name, ["--crops", c001, "--set", "seed=0", *set_args, *exclude,
                                                    "--note", "SHAPE: a RepresentationAgenticVerifier trial"],
                                    statuses=("done",))
            except SystemExit as error:          # a failed call is the loop's failed trial, not the end of the run
                raise RuntimeError(str(error)) from None
            made[params["output_subdir"]] = call_id
            return _embed_result(self.run, call_id, scratch, params["output_subdir"])

        before = usage.snapshot()
        res = StepVerifier(max_iters=1 + verifier.budget).resolve_stage(
            verifier, verifier.make_adapter({"cell_metadata": meta}), runner, initial,
            _embed_result(self.run, e001, scratch, initial["output_subdir"]),
            upstream_ctx={"representation_task": task})
        rep_usage = _usage_delta(before, usage.snapshot())
        selected = res.details.get("selected")
        delivered = made.get((res.params or {}).get("output_subdir")) if selected else e001
        stage = {"trial_0": e001, "delivered": delivered or e001, "tier": verifier.tier,
                 "execution_errors": verifier.execution_errors,
                 "reasoner_failures": verifier.reasoner_failures, "fallback_rounds": verifier.fallback_rounds,
                 "adopted_proxy": verifier.proxy,
                 "gate": verifier.admission, "split": split, "resolution": {"status": res.status, "reason": res.reason,
                 "fault": res.fault}, "decision": res.details.get("decision"), "trial_calls": made,
                 "identities": "site = image_id, well = parent_id, plate = source", "usage": rep_usage}
        _write(self.logs / "representation.details.json", {"stage": stage, "details": res.details})
        self.record["stages"]["representation"] = stage
        self.record["usage"]["representation"] = rep_usage
        _add_usage(self.record["usage"].setdefault("total", {}), rep_usage)
        self.save()
        print(f"SHAPE representation: tier {verifier.tier}, gate {(verifier.admission or {}).get('decision')}, "
              f"delivered {stage['delivered']} ({res.status}, {res.reason})", flush=True)
        return delivered or e001

    def states(self, embed_call: str) -> dict:
        """B001 (the calibration rule), the morphological_state decision, K001 and choice.json."""
        from shape.agent.verifier import Verifier
        from shape.agent.verification.stage_map import stage_by_name

        b001 = self.call("batch_correct", NAMES["batch_correct"], [
            "--embedding", embed_call, "--batch-key", "source",
            "--note", "SHAPE: the batch step after the analyzer, batch = source (the calibration rule decides)"],
            statuses=("done",))
        b_dir = self.call_dir(b001)
        cells = pd.read_csv(b_dir / "cells.csv", dtype=str, keep_default_na=False)
        spec = stage_by_name("morphological_state")
        runner = morphology_runner(self.C, self.cfg, np.load(b_dir / "embedding.npy"), cells, self.rows,
                                   self.scratch / "morphology")
        initial = dict(spec.default_params)
        res = Verifier().resolve_managed_stage(spec, runner, initial, runner(initial),
                                               tool_name="Cell_State_Analyzer_Tool")
        params = dict(res.params or initial)
        resolution = round(float(params["leiden_resolution"]), 4)
        neighbors = int(round(float(params["n_neighbors"])))
        k001 = self.call("cluster", NAMES["cluster"], [
            "--embedding", b001, "--set", f"resolution={_literal(resolution)}", f"n_neighbors={neighbors}",
            "--note", f"SHAPE: MorphologicalStateVerifier's delivered clustering ({res.status}, {res.reason})"],
            statuses=("done",))
        self.record["stages"]["batch_correct"] = {"call": b001, "summary": (calls(self.run)[b001]["end"] or {})
                                                  .get("summary")}
        self.record["stages"]["morphology"] = {
            "call": k001, "resolution": resolution, "n_neighbors": neighbors,
            "status": res.status, "reason": res.reason, "fault": res.fault, "upstream_target": res.upstream_target,
            "verdict": {"decision": res.verdict.decision, "action": res.verdict.action, "notes": res.verdict.notes},
            "trials": [{"params": p, "decision": v.decision, "action": v.action, "score": v.scalar_score,
                        "metrics": v.metrics} for p, v, _ in res.trials]}
        choice = {"cluster_call": k001, "resolution": float(resolution), "skipped": {},
                  "reason": (f"SHAPE arm ({self.model}): the chain its VRR loops delivered; states at "
                             f"MorphologicalStateVerifier's resolution ({res.status}, {res.reason})")}
        (self.ws / "submission" / "choice.json").write_text(json.dumps(choice, indent=1), encoding="utf-8")
        self.save()
        print(f"SHAPE states: {b001} {self.record['stages']['batch_correct']['summary']}; {k001} at resolution "
              f"{resolution}, n_neighbors {neighbors} ({res.status}, {res.reason})", flush=True)
        return choice


def write_run_meta(run: Path, *, model: str, budget_hours: float, prep_name: str, prep: Path, metadata: str,
                   root: Path, repo: Path, n_images: int | None) -> dict:
    """RUN_META.json of a SHAPE run: arm "shape", the reasoner model, the GPU (name and slug), the commit."""
    from .hardware import gpu_name, gpu_slug
    from .score import control_dir

    slug = gpu_slug()
    control = control_dir(root, prep_name, metadata, slug) / "RESULT.json"
    meta = {"arm": ARM, "model": model, "repo_commit": os.environ.get("FULL_HCMA_EXPECTED_COMMIT"),
            "branch": os.environ.get("FULL_HCMA_BRANCH"), "slurm_job": os.environ.get("SLURM_JOB_ID"),
            "host": socket.gethostname(),
            "gpu": gpu_name(), "gpu_slug": slug, "budget_hours": budget_hours, "prep_name": prep_name,
            "prep": str(prep), "metadata": metadata, "root": str(root), "repo": str(repo),
            "control_result": str(control) if control.exists() else None, "smoke_images": n_images,
            "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    _write(run / "RUN_META.json", meta)
    return meta


def run_shape(root: Path, repo: Path, out: Path, *, model: str | None, budget_hours: float = BUDGET_DEFAULT,
              prep_name: str = "pairs", metadata: str = "full", n_images: int | None = None,
              runner: Runner | None = None, engine=None, segment_search_fn: Callable | None = None,
              crop_workers: int | None = None) -> Path:
    """One SHAPE-arm run in `out` (the module docstring). `n_images`: a smoke on the prep's first whole type pairs
    (control.smoke_selection), scored here over its images only. Tests: `runner` replaces the tool subprocesses,
    `engine` the reasoner (still smoke-called), `segment_search_fn` the segmentation child process."""
    from .control import smoke_selection
    from .hardware import gpu_slots, node_facts
    from .score import NO_CLASS_MAP, class_map_check
    from .subset import prep_dir
    from .workspace import setup_run

    from shape.engine import usage

    root, repo, run = Path(root), Path(repo), Path(out)
    # the launcher creates logs/ and cache/ (TMPDIR and other caches) before it starts the driver
    if run.exists() and not set(p.name for p in run.iterdir()) <= LAUNCHER_DIRS:
        raise SystemExit(f"{run} holds a run already; use a new run directory")
    # the harness's checks (they read the labels, as the control's do), before any SHAPE code runs
    check = class_map_check(root, prep_name)
    if not check["ok"]:
        raise SystemExit(f"SHAPE_FAILED: {NO_CLASS_MAP}: " + "; ".join(check["problems"]))
    smoke = smoke_selection(root, prep_name, n_images, check["map"]) if n_images is not None else None
    before = usage.snapshot()
    engine = build_engine(model, engine=engine)
    smoke_usage = _usage_delta(before, usage.snapshot())
    started = time.time()
    workspace = setup_run(run, root, repo, budget_hours=budget_hours, gpu_slots=gpu_slots(),
                          prep=prep_dir(root, prep_name), metadata=metadata)
    write_run_meta(run, model=model, budget_hours=budget_hours, prep_name=prep_name, prep=prep_dir(root, prep_name),
                   metadata=metadata, root=root, repo=repo, n_images=n_images)
    (run / "logs" / CONTROL_COMMANDS).touch()
    os.environ.setdefault(_ORIGINAL_CUDA, os.environ.get("CUDA_VISIBLE_DEVICES", "<unset>"))
    os.environ["CUDA_VISIBLE_DEVICES"] = ""            # the driver never opens a CUDA context (exclusive GPUs)
    facts = node_facts()
    tool_config = json.loads((workspace / "tools" / "_config.json").read_text(encoding="utf-8"))
    seg_workers = 1 if tool_config.get("gpu_exclusive") else max(1, min(4, int((facts.get("gpu_mem_gb") or 0) // 6)))
    arm = Arm(run, repo, runner or tool_runner(repo), engine, model, budget_hours, started)
    arm.record.update(prep_name=prep_name, metadata=metadata, smoke=smoke, segment_workers=seg_workers)
    arm.record["usage"] = {"smoke_call": smoke_usage, "total": dict(smoke_usage)}
    try:
        arm.stage = "segment"
        mask_of = arm.segment(smoke["images"] if smoke else None, seg_workers, segment_search_fn)
        arm.stage = "crop"
        c001, excluded = arm.crop(mask_of, crop_workers or CROP_WORKERS)
        arm.stage = "representation"
        embed_call = arm.represent(c001, excluded)
        arm.stage = "states"
        arm.record["choice"] = arm.states(embed_call)
        arm.record["status"] = "delivered"
    except BaseException as error:
        arm.record.update(status="failed", error=f"{type(error).__name__}: {error}"[-2000:])
        if "stopped" not in arm.record:          # a stage-level step's failure (a per-image one is recorded)
            arm.record["stopped"] = stop_record(arm.stage, None, error)
            print(stop_message(arm.record["stopped"]), flush=True)
        raise
    finally:
        arm.record["ended"] = time.time()
        arm.save()
    if smoke:
        from .score import score_choice, write_outputs

        result, table, _ = score_choice(run, root, arm.record["choice"], forbidden=[], control=None,
                                        lever_coverage=False, prep_name=prep_name, metadata=metadata,
                                        smoke_images=smoke["images"])
        result.update(arm=ARM, smoke=smoke)
        write_outputs(run, result, table, None)
    return run


def main(argv: list[str] | None = None) -> None:
    """`python -m benchmarks.full_HCMA.harness.shape_arm segment-search JOB.json OUT.json` (the segmentation child)."""
    argv = list(sys.argv[1:] if argv is None else argv)
    if len(argv) != 3 or argv[0] != "segment-search":
        raise SystemExit("usage: python -m benchmarks.full_HCMA.harness.shape_arm segment-search JOB.json OUT.json")
    job = json.loads(Path(argv[1]).read_text(encoding="utf-8"))
    _write(Path(argv[2]), segment_search(job))


if __name__ == "__main__":
    main()

#!/usr/bin/env python3
"""Formal five-arm cropper benchmark on NeurIPS22-CellSeg.

Arms: maskcrop, cellprofiler, gpt, react, shape. Every arm crops the ground-truth instance mask
(`labels/<image>_label.tiff`), and the crops are scored on that same mask.
Registered readouts:
neighbor_contamination, target_retention, runtime_per_image, tool_calls_per_image, tokens_per_image.

Typical pilot:
    python run_benchmark.py prepare --run-dir R --data-dir DATA --n-images 30
    python run_benchmark.py run --run-dir R --arm maskcrop
    python run_benchmark.py run --run-dir R --arm cellprofiler
    python run_benchmark.py run --run-dir R --arm gpt
    python run_benchmark.py run --run-dir R --arm react
    python run_benchmark.py run --run-dir R --arm shape
    python run_benchmark.py compare --run-dir R
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.util
import inspect
import json
import os
import platform
import re
import shutil
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
_REPO = _HERE.parents[1]
sys.path.insert(0, str(_REPO))
sys.path.insert(0, str(_REPO / "benchmarks" / "baselines" / "metrics"))
sys.path.insert(0, str(_HERE / "runners"))      # the arms and the runners they reuse

import cropper_arms as arm_lib  # noqa: E402
import run_shape_crop as rsc  # noqa: E402
import crop_qc  # noqa: E402
from cropper_config import PIXEL_SUFFIXES  # noqa: E402
from pairing import _find_image_mask_pairs  # noqa: E402
from shape.agent.verification import base as vrr_base  # noqa: E402
from shape.engine import usage as token_usage  # noqa: E402
from shape.tools.single_cell_cropper.tool import SEGMENTER_DEFAULT_MIN_SIZE  # noqa: E402

# AP50 is not a readout: the input is the ground-truth mask (AP50 = 1 by definition) and no arm changes it.
METRICS = (
    "neighbor_contamination",
    "target_retention",
    "runtime_per_image",
    "tool_calls_per_image",
    "tokens_per_image",
)
DOMAIN_TOOL_ARMS = ("maskcrop", "cellprofiler")
CONTROL_ARMS = ("cellprofiler",)   # checked by the domain-tool pairing; never gates the other arms' comparison
GATED_ARMS = tuple(arm for arm in arm_lib.ARMS if arm not in CONTROL_ARMS)
MODEL_USAGE_FIELDS = ("model_calls_per_image", "input_tokens_per_image", "output_tokens_per_image",
                      "cached_tokens_per_image", "reasoning_tokens_per_image")
SHAPE_USAGE_FIELDS = ("reasoner_calls_per_image", "vision_calls_per_image",
                      "reasoner_tokens_per_image", "vision_tokens_per_image")
PER_IMAGE_FIELDS = (
    "selection_index", "record_index", "image_id", "image_path", "mask_path", "gt_path", "arm",
    "status", "error", *METRICS, *MODEL_USAGE_FIELDS, *SHAPE_USAGE_FIELDS,
    "n_triage_rejected", "delivery", "rounds_per_image", "n_crops", "n_expected_crops", "crop_coverage",
)
PER_CROP_FIELDS = (
    "selection_index", "record_index", "image_id", "image_path", "mask_path", "gt_path", "arm",
    "crop_index", "cell_label", "source", "crop_image_path", "native_crop_image_path",
    "bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc",
    "native_bbox_minr", "native_bbox_minc", "native_bbox_maxr", "native_bbox_maxc",
    "neighbor_contamination", "target_retention", "occupancy", "target_dominance", "valid",
    "triage_rejected",
)
DOMAIN_PAIR_FIELDS = (
    "selection_index", "image_id", "cell_label",
    "maskcrop_present", "cellprofiler_present",
    "maskcrop_is_square", "cellprofiler_is_square", "same_bbox",
    "maskcrop_bbox_minr", "maskcrop_bbox_minc", "maskcrop_bbox_maxr", "maskcrop_bbox_maxc",
    "cellprofiler_bbox_minr", "cellprofiler_bbox_minc", "cellprofiler_bbox_maxr", "cellprofiler_bbox_maxc",
    "maskcrop_neighbor_contamination", "cellprofiler_neighbor_contamination",
    "neighbor_contamination_delta_cellprofiler_minus_maskcrop",
    "maskcrop_target_retention", "cellprofiler_target_retention",
    "target_retention_delta_cellprofiler_minus_maskcrop",
)


def _git_commit() -> str:
    import subprocess
    try:
        return subprocess.check_output(
            ["git", "-C", str(_REPO), "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return "unknown"


# the packages of the run_meta `packages` block: name -> (import name, distributions); absent when not installed
PACKAGES = {"numpy": ("numpy", ("numpy",)), "scipy": ("scipy", ("scipy",)),
            "scikit-image": ("skimage", ("scikit-image",)),
            "opencv": ("cv2", ("opencv-python", "opencv-python-headless", "opencv-contrib-python",
                               "opencv-contrib-python-headless")),
            "tifffile": ("tifffile", ("tifffile",)), "pydantic": ("pydantic", ("pydantic",)),
            "openai": ("openai", ("openai",)), "cellpose": ("cellpose", ("cellpose",))}


def _packages() -> dict:
    from importlib import metadata
    out = {}
    for name, (module, dists) in PACKAGES.items():
        if importlib.util.find_spec(module) is None:
            continue
        for dist in dists:
            try:
                out[name] = metadata.version(dist)
                break
            except metadata.PackageNotFoundError:
                continue
    return out


def _llm_meta(arm_name: str, model: str, engine, shape_loop: str = "agentic") -> dict:
    """The model configuration an LLM arm's requests carry, as the engine sends it (`ChatOpenAI`): the provider,
    the gateway's host (never the key or the full URL), reasoning_effort (`_apply_backend_quirks`: SHAPE_REASONING_EFFORT,
    default "medium"; None when not sent), temperature / top_p (None when the model takes only its default and they
    are not sent), max_tokens and the response schema per question kind, the arm's tries and retry wait, and
    whether the engine's response cache is on. No hidden reasoning is requested: the provider's reasoning is
    recorded only as a token count (the usage)."""
    from urllib.parse import urlsplit

    from shape.engine import openai as openai_engine
    from shape.engine.factory import provider_of
    from shape.agent.verification.crop_triage import MaskCheckVerdict
    from shape.agent.verification.verifiers.crop import CropAgenticVerifier

    defaults = inspect.signature(openai_engine.ChatOpenAI._generate_multimodal).parameters
    sent = not openai_engine._requires_temperature_1(model)
    effort = os.environ.get("SHAPE_REASONING_EFFORT", "medium")
    base_url = getattr(getattr(engine, "client", None), "base_url", None) or os.environ.get("LLM_BASE_URL")
    engine_max = defaults["max_tokens"].default
    if arm_name == "gpt":
        kinds = {"crop_boxes": (arm_lib.GPT_MAX_COMPLETION_TOKENS, None)}
    elif arm_name == "react":
        kinds = {"action": (engine_max, arm_lib.react_arm.ReactAction.__name__)}
    elif shape_loop == "agentic":
        verifier = CropAgenticVerifier
        kinds = {"admission": (engine_max, verifier.admission_schema.__name__),
                 "round": (engine_max, verifier.plan_schema.__name__),
                 "repair": (engine_max, verifier.plan_schema.__name__),
                 "final": (engine_max, verifier.choice_schema.__name__),
                 "final_retry": (engine_max, verifier.choice_schema.__name__),
                 "mask_check": (engine_max, MaskCheckVerdict.__name__)}
    else:
        kinds = {}
    shape_arm = arm_name == "shape"
    return {
        "model": model,
        "provider": provider_of(model),
        "base_url_host": urlsplit(str(base_url)).hostname if base_url else None,
        "reasoning_effort": (effort.lower() if openai_engine._is_reasoning_or_next_gen_model(model) and effort
                             and effort.lower() != "default" else None),
        "temperature": defaults["temperature"].default if sent else None,
        "top_p": defaults["top_p"].default if sent else None,
        "max_tokens": {kind: value[0] for kind, value in kinds.items()},
        "response_schema": {kind: value[1] for kind, value in kinds.items()},
        # the arm's own tries per question kind (the mask check asks each montage once; the shared ReAct runner
        # asks each round once: a failed reply is an observation that uses its round)
        "llm_attempts": {kind: 1 if kind == "mask_check" or arm_name == "react" else
                         (vrr_base.REASONER_ATTEMPTS if shape_arm else arm_lib.LLM_ATTEMPTS) for kind in kinds},
        "llm_retry_seconds": (None if arm_name == "react" else
                              vrr_base.REASONER_RETRY_SECONDS if shape_arm else arm_lib.LLM_RETRY_SECONDS),
        "response_cache": bool(getattr(engine, "enable_cache", False)),
        "system_prompt": getattr(engine, "system_prompt", None),        # sent as the system message of every request
        # below the arm: the SDK's own retries / timeout per request, and the engine's one plain re-ask when a
        # structured parse fails (inside the same generate call)
        "sdk_max_retries": getattr(getattr(engine, "client", None), "max_retries", None),
        "sdk_timeout_seconds": getattr(getattr(engine, "client", None), "timeout", None),
        "structured_parse_fallback": "one plain request when the structured parse fails",
    }


def _read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def _write_json(path: Path, payload) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, default=str), encoding="utf-8")


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _write_rows(path_stem: Path, rows: list[dict], fieldnames) -> None:
    fieldnames = list(fieldnames)
    with path_stem.with_suffix(".jsonl").open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, default=str) + "\n")
    with path_stem.with_suffix(".csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _select_random_images(pairs: list[tuple[Path, Path]], n: int,
                          rng: np.random.RandomState) -> list[tuple[Path, Path]]:
    """A random draw of `n` image / mask pairs without replacement; every pair when there are fewer than `n`."""
    order = rng.permutation(len(pairs))
    return [pairs[i] for i in order[:min(n, len(pairs))]]


def prepare(args) -> None:
    """The seeded image set. The input mask of every arm is the image's ground-truth mask (mask_path =
    gt_path): the arms differ only in how they crop, never in segmentation."""
    available = _find_image_mask_pairs(args.data_dir, args.label_suffix, args.label_ext)
    if not available:
        raise SystemExit(f"no image / ground-truth mask pairs under {args.data_dir}")
    rng = np.random.RandomState(args.seed)
    selected = _select_random_images(available, args.n_images, rng)
    payload = {
        "schema": "shape.cropper.formal.selection.v2",
        "created_utc": datetime.now(timezone.utc).isoformat(),
        "data_dir": str(args.data_dir),
        "mask_source": "ground_truth",
        "seed": args.seed,
        "n_images_requested": args.n_images,
        "selected_images": [
            {"image_id": img.stem, "image_path": str(img), "gt_path": str(gt), "mask_path": str(gt)}
            for img, gt in selected
        ],
    }
    _write_json(args.run_dir / "selection.json", payload)
    print(f"prepared {len(selected)} images -> {args.run_dir / 'selection.json'}")


def _score_crops(mask, crops: list[dict]) -> list[dict]:
    rows = []
    for crop_index, crop in enumerate(crops):
        if "bbox" not in crop or "cell_label" not in crop:
            continue
        bbox = [int(round(float(value))) for value in crop["bbox"]]
        if len(bbox) != 4:
            continue
        qc = crop_qc.crop_qc(mask, bbox, int(crop["cell_label"]))
        native_bbox = list(crop.get("native_bbox") or [None, None, None, None])
        if len(native_bbox) != 4:
            native_bbox = [None, None, None, None]
        rows.append({
            "crop_index": crop_index,
            "cell_label": int(crop["cell_label"]),
            "source": crop.get("source"),
            "crop_image_path": crop.get("crop_image_path"),
            "native_crop_image_path": crop.get("native_crop_image_path"),
            "bbox_minr": bbox[0],
            "bbox_minc": bbox[1],
            "bbox_maxr": bbox[2],
            "bbox_maxc": bbox[3],
            "native_bbox_minr": native_bbox[0],
            "native_bbox_minc": native_bbox[1],
            "native_bbox_maxr": native_bbox[2],
            "native_bbox_maxc": native_bbox[3],
            **qc,
            "triage_rejected": crop.get("triage_rejected"),
        })
    return rows


def react_tool_budget(cap: int) -> int:
    """The agentic arms' tool calls per image at most: the single defaults call (round 1) and then at most
    `cap` calls in each of the other MAX_ROUNDS - 1 rounds (SHAPE's budget, and ReAct's on the shared runner)."""
    return 1 + (arm_lib.MAX_ROUNDS - 1) * int(cap)


def _validate_crops(
        arm: str, crops: list[dict], expected_labels: list[int], mask_shape: tuple[int, int]) -> None:
    arm_lib.validate_crops(arm, crops, expected_labels, mask_shape)


def _usage_delta(before: dict, after: dict) -> dict:
    calls = after["calls"] - before["calls"]
    observed_before = before.get("usage_field_observed_calls", {})
    observed_after = after.get("usage_field_observed_calls", {})

    def field(name: str):
        observed = observed_after.get(name, 0) - observed_before.get(name, 0)
        return after.get(name, 0) - before.get(name, 0) if calls and observed == calls else None

    return {"model_calls_per_image": calls,
            "input_tokens_per_image": field("input_tokens"),
            "output_tokens_per_image": field("output_tokens"),
            "cached_tokens_per_image": field("cached_tokens"),
            "reasoning_tokens_per_image": field("reasoning_tokens")}


def _score_one(mask, crops: list[dict], seconds: float, tool_calls: int,
               tokens: int | None, crop_results: list[dict] | None = None) -> dict:
    """The registered readouts of one image. Every delivered crop is scored, SHAPE's triage-rejected
    crops included; their count is the separate `n_triage_rejected` (None for arms without triage)."""
    qcs = crop_results if crop_results is not None else _score_crops(mask, crops)

    def mean(key: str):
        values = [q[key] for q in qcs if q.get(key) is not None]
        return float(np.mean(values)) if values else None

    flags = [crop["triage_rejected"] for crop in crops if crop.get("triage_rejected") is not None]
    return {
        "neighbor_contamination": mean("neighbor_contamination"),
        "target_retention": mean("target_retention"),
        "runtime_per_image": float(seconds),
        "tool_calls_per_image": int(tool_calls),
        "tokens_per_image": tokens,
        "n_triage_rejected": int(sum(flags)) if flags else None,
        "n_crops": len(crops),
    }


def _work_dir(arm: str, image_id: str) -> Path | None:
    """The node-local dir an image's arm runs in, `<CROPPER_WORK_ROOT>/<arm>/<image>`; None when the env is unset
    (the arm runs in `<run dir>/<arm>/images/<image>`)."""
    root = os.environ.get("CROPPER_WORK_ROOT")
    return Path(os.path.abspath(root)) / arm / image_id if root else None


def _keep_non_pixel_files(work: Path, image_dir: Path) -> None:
    """Copies every non-pixel file of `work` (logs, metadata, manifests) to the same relative path under
    `image_dir`, then removes `work`: the crop pixels never reach the run dir. A `.jsonl` log already there (an
    earlier, resumed attempt of the image) is appended to, as the arm itself appends, so no attempt is lost."""
    for path in sorted(work.rglob("*")) if work.is_dir() else []:
        if path.is_file() and path.suffix.lower() not in PIXEL_SUFFIXES:
            target = image_dir / path.relative_to(work)
            target.parent.mkdir(parents=True, exist_ok=True)
            if path.suffix.lower() == ".jsonl" and target.is_file():
                with target.open("ab") as out, path.open("rb") as src:
                    shutil.copyfileobj(src, out)
            else:
                shutil.copy2(path, target)
    shutil.rmtree(work, ignore_errors=True)


def _repoint(value, work: Path, image_dir: Path):
    """A JSON-able `value` with every path under `work` (either separator, or its resolved form) renamed to the
    same path under `image_dir`."""
    prefixes = {str(work): str(image_dir), work.as_posix(): image_dir.as_posix(),
                str(work.resolve()): str(image_dir), work.resolve().as_posix(): image_dir.as_posix()}
    patterns = [(re.compile(re.escape(src) + r"(?![\w.-])"), dst) for src, dst in prefixes.items()]

    def repoint(item):
        if isinstance(item, str):
            for pattern, dst in patterns:
                item = pattern.sub(lambda _m, dst=dst: dst, item)
            return item
        if isinstance(item, dict):
            return {key: repoint(sub) for key, sub in item.items()}
        if isinstance(item, list):
            return [repoint(sub) for sub in item]
        return item

    return repoint(value)


def run_arm(args) -> None:
    selection_path = args.run_dir / "selection.json"
    selection = _read_json(selection_path)
    out = args.run_dir / args.arm
    out.mkdir(parents=True, exist_ok=True)
    cores = int(args.cores) if args.cores else vrr_base.available_cores()
    cap = vrr_base.concurrency_cap(cores, arm_lib.THREADS_PER_CALL)
    arm = arm_lib.make_arm(
        args.arm,
        model=args.model,
        min_area=args.min_area,
        margin=args.margin,
        margin_frac=args.maskcrop_margin_frac,
        batch_size=args.gpt_batch_size,
        cores=cores,
        deadline=args.deadline,
        shape_loop=args.shape_loop,
        cellprofiler_python=args.cellprofiler_python,
    )
    if args.arm == "react":
        cap = getattr(arm, "cap", cap)              # SHAPE's crop verifier's own per-round cap on these cores
    meta = {
        "schema": "shape.cropper.formal.arm_run.v1",
        "arm": args.arm,
        "model": args.model,
        "git_commit": _git_commit(),
        "started_utc": datetime.now(timezone.utc).isoformat(),
        "metrics": METRICS,
        "env": sys.prefix,
        "cellprofiler_python": args.cellprofiler_python if args.arm == "cellprofiler" else None,
        "selection_sha256": _sha256(selection_path),
        "n_selected_images": len(selection["selected_images"]),
        "min_area": int(args.min_area),
        "margin": int(args.margin),
        "maskcrop_margin_frac": float(args.maskcrop_margin_frac),
        "mask_source": selection.get("mask_source"),
        # the agentic arms' budget, the same for both: MAX_ROUNDS rounds, round 1 the single defaults call, then at
        # most `concurrency_cap` calls per round (at most 1 + (MAX_ROUNDS - 1) x concurrency_cap calls)
        "cores": cores if args.arm in ("react", "shape") else None,
        "concurrency_cap": cap if args.arm in ("react", "shape") else None,
        "max_rounds": arm_lib.MAX_ROUNDS if args.arm in ("react", "shape") else None,
        "react_tool_budget": react_tool_budget(cap) if args.arm == "react" else None,
        "react_runner": "benchmarks/_shared/react_arm.py" if args.arm == "react" else None,
        "deadline": args.deadline if args.arm in ("react", "shape") else None,
        "gpt_batch_size": int(args.gpt_batch_size) if args.arm == "gpt" else None,
        "domain_tool_contract": (
            "same selection, ground-truth mask, min_area, target cell labels, and Maskcrop square geometry"
            if args.arm in DOMAIN_TOOL_ARMS else None
        ),
        "llm": (_llm_meta(args.arm, args.model, getattr(arm, "engine", None), args.shape_loop)
                if args.arm in arm_lib.LLM_ARMS else None),
        "host": socket.gethostname(),
        "python": platform.python_version(),
        "packages": _packages(),
        "seed": selection.get("seed"),
    }
    records_path = out / "records.jsonl"
    kept: list[dict] = []
    if getattr(args, "resume", False) and (out / "run_meta.json").is_file():
        # a resubmitted arm (e.g. after its job's time limit): the same code, selection and settings; the
        # finished images are kept, the failed and missing ones run again
        previous = _read_json(out / "run_meta.json")
        keys = [key for key in meta if key not in ("started_utc", "env", "deadline", "cellprofiler_python", "host")]
        changed = [key for key in keys if previous.get(key) != json.loads(json.dumps(meta[key], default=str))]
        if changed:
            raise SystemExit(f"--resume: the arm's settings changed since its first run: {changed}")
        kept = [record for record in _records(args.run_dir, args.arm) if record.get("status") == "ok"]
        meta["started_utc"] = previous.get("started_utc")
        meta["resumed_utc"] = [*previous.get("resumed_utc", []), datetime.now(timezone.utc).isoformat()]
    _write_json(out / "run_meta.json", meta)
    done = {record["image_id"] for record in kept}
    with records_path.open("w", encoding="utf-8") as handle:
        for record in kept:
            handle.write(json.dumps(record, default=str) + "\n")
        handle.flush()
        for item in selection["selected_images"]:
            image_id = item["image_id"]
            if image_id in done:
                print(f"{args.arm} {image_id}: kept (resume)")
                continue
            image_dir = out / "images" / image_id
            work = _work_dir(args.arm, image_id)          # CROPPER_WORK_ROOT: the crop pixels stay node-local
            t0 = time.perf_counter()
            usage_before = token_usage.snapshot()
            try:
                mask = arm_lib.load_mask(Path(item["mask_path"]))
                expected_labels = arm_lib.labels_in_tool_order(mask, min_area=args.min_area)
                result = arm(Path(item["image_path"]), Path(item["mask_path"]), mask, work or image_dir)
                _validate_crops(args.arm, result.crops, expected_labels, mask.shape[:2])
                seconds = time.perf_counter() - t0
                usage_after = token_usage.snapshot()
                tokens = (int(usage_after["total_tokens"] - usage_before["total_tokens"])
                          if args.arm in arm_lib.LLM_ARMS else 0)
                crop_results = _score_crops(mask, result.crops)
                metrics = _score_one(mask, result.crops, seconds, result.tool_calls, tokens,
                                     crop_results=crop_results)
                metrics.update(_usage_delta(usage_before, usage_after) if args.arm in arm_lib.LLM_ARMS
                               else {key: 0 for key in MODEL_USAGE_FIELDS})
                metrics.update({key: result.trace.get("model_usage", {}).get(key)
                                for key in SHAPE_USAGE_FIELDS})
                actual_labels = {int(crop["cell_label"]) for crop in result.crops
                                 if "cell_label" in crop}
                metrics["delivery"] = result.trace.get("delivery")
                metrics["rounds_per_image"] = result.trace.get("rounds")      # the ReAct session's rounds
                metrics["n_expected_crops"] = len(expected_labels)
                metrics["crop_coverage"] = (
                    len(actual_labels & set(expected_labels)) / len(expected_labels)
                    if expected_labels else 1.0
                )
                record = {"image_id": image_id, "status": "ok", "arm": args.arm,
                          "metrics": metrics, "crops": result.crops,
                          "crop_results": crop_results, "trace": result.trace}
            except Exception as exc:
                usage_after = token_usage.snapshot()
                failure_metrics = {
                    "runtime_per_image": float(time.perf_counter() - t0),
                    "tokens_per_image": (int(usage_after["total_tokens"] - usage_before["total_tokens"])
                                         if args.arm in arm_lib.LLM_ARMS else 0),
                }
                failure_metrics.update(_usage_delta(usage_before, usage_after)
                                       if args.arm in arm_lib.LLM_ARMS
                                       else {key: 0 for key in MODEL_USAGE_FIELDS})
                record = {"image_id": image_id, "status": "error", "arm": args.arm,
                          "error": str(exc), "metrics": failure_metrics}
                if getattr(exc, "trace", None) is not None:      # the requests of a failed image (GPT)
                    record["trace"] = exc.trace
            finally:
                if work:                    # ok or error: the non-pixel files to the run dir, the work dir removed
                    _keep_non_pixel_files(work, image_dir)
            if work:                        # every recorded path names where its file is (or would be) in the run dir
                record = _repoint(json.loads(json.dumps(record, default=str)), work, image_dir)
            handle.write(json.dumps(record, default=str) + "\n")
            handle.flush()
            print(f"{args.arm} {image_id}: {record['status']}")
    print(f"wrote {records_path}")


def _records(run_dir: Path, arm: str) -> list[dict]:
    path = run_dir / arm / "records.jsonl"
    if not path.exists():
        return []
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def _selection_maps(selection: dict) -> tuple[dict[str, dict], dict[str, int]]:
    items = selection["selected_images"]
    return ({item["image_id"]: item for item in items},
            {item["image_id"]: index for index, item in enumerate(items)})


def _group_records(records: list[dict]) -> dict[str, list[dict]]:
    grouped: dict[str, list[dict]] = {}
    for record in records:
        grouped.setdefault(str(record.get("image_id")), []).append(record)
    return grouped


def _crop_results_for_record(record: dict, item: dict | None) -> list[dict]:
    stored = record.get("crop_results")
    if isinstance(stored, list):
        return stored
    crops = list(record.get("crops") or [])
    if item:
        mask_path = Path(item["mask_path"])
        if mask_path.is_file():
            return _score_crops(arm_lib.load_mask(mask_path), crops)
    rows = []
    for crop_index, crop in enumerate(crops):
        bbox = list(crop.get("bbox") or [])
        if len(bbox) != 4 or "cell_label" not in crop:
            continue
        native_bbox = list(crop.get("native_bbox") or [None, None, None, None])
        if len(native_bbox) != 4:
            native_bbox = [None, None, None, None]
        rows.append({
            "crop_index": crop_index,
            "cell_label": int(crop["cell_label"]),
            "source": crop.get("source"),
            "crop_image_path": crop.get("crop_image_path"),
            "native_crop_image_path": crop.get("native_crop_image_path"),
            "bbox_minr": int(bbox[0]), "bbox_minc": int(bbox[1]),
            "bbox_maxr": int(bbox[2]), "bbox_maxc": int(bbox[3]),
            "native_bbox_minr": native_bbox[0],
            "native_bbox_minc": native_bbox[1],
            "native_bbox_maxr": native_bbox[2],
            "native_bbox_maxc": native_bbox[3],
            "neighbor_contamination": None, "target_retention": None,
            "occupancy": None, "target_dominance": None, "valid": None,
        })
    return rows


def _detail_rows(selection: dict, records_by_arm: dict[str, list[dict]]) -> tuple[list[dict], list[dict]]:
    item_by_id, index_by_id = _selection_maps(selection)
    image_rows = []
    crop_rows = []
    for arm in [name for name in arm_lib.ARMS if name in records_by_arm]:
        grouped = _group_records(records_by_arm[arm])
        ordered_ids = list(index_by_id)
        ordered_ids.extend(sorted(set(grouped) - set(index_by_id)))
        for image_id in ordered_ids:
            item = item_by_id.get(image_id)
            records = grouped.get(image_id) or []
            if not records:
                records = [{"image_id": image_id, "arm": arm, "status": "missing",
                            "error": "no record for selected image", "metrics": {}}]
            for record_index, record in enumerate(records):
                metrics = record.get("metrics") or {}
                base = {
                    "selection_index": index_by_id.get(image_id),
                    "record_index": record_index,
                    "image_id": image_id,
                    "image_path": item.get("image_path") if item else None,
                    "mask_path": item.get("mask_path") if item else None,
                    "gt_path": item.get("gt_path") if item else None,
                    "arm": arm,
                }
                image_rows.append({
                    **base,
                    "status": record.get("status", "missing"),
                    "error": record.get("error"),
                    **{key: metrics.get(key) for key in METRICS},
                    **{key: metrics.get(key) for key in MODEL_USAGE_FIELDS},
                    **{key: metrics.get(key) for key in SHAPE_USAGE_FIELDS},
                    "n_triage_rejected": metrics.get("n_triage_rejected"),
                    "delivery": metrics.get("delivery"),
                    "rounds_per_image": metrics.get("rounds_per_image"),
                    "n_crops": metrics.get("n_crops", len(record.get("crops") or [])),
                    "n_expected_crops": metrics.get("n_expected_crops"),
                    "crop_coverage": metrics.get("crop_coverage"),
                })
                if record.get("status") != "ok":
                    continue
                for crop in _crop_results_for_record(record, item):
                    crop_rows.append({**base, **crop})
    return image_rows, crop_rows


def _domain_pair_rows(crop_rows: list[dict]) -> list[dict]:
    by_arm: dict[str, dict[tuple[str, int], dict]] = {arm: {} for arm in DOMAIN_TOOL_ARMS}
    for row in crop_rows:
        arm = row["arm"]
        if arm in by_arm:
            by_arm[arm][(row["image_id"], int(row["cell_label"]))] = row
    keys = set().union(*(set(rows) for rows in by_arm.values()))
    selection_index = {row["image_id"]: row.get("selection_index") for row in crop_rows}
    paired_rows = []

    def sort_index(image_id: str):
        value = selection_index.get(image_id)
        return value if value is not None else float("inf")

    for image_id, cell_label in sorted(
            keys, key=lambda key: (sort_index(key[0]), key[0], key[1])):
        maskcrop = by_arm["maskcrop"].get((image_id, cell_label))
        cellprofiler = by_arm["cellprofiler"].get((image_id, cell_label))

        def value(row, key):
            return row.get(key) if row else None

        mc_contam = value(maskcrop, "neighbor_contamination")
        cp_contam = value(cellprofiler, "neighbor_contamination")
        mc_retention = value(maskcrop, "target_retention")
        cp_retention = value(cellprofiler, "target_retention")
        bbox_keys = ("bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc")

        def is_square(row):
            return bool(row and row["bbox_maxr"] - row["bbox_minr"] ==
                        row["bbox_maxc"] - row["bbox_minc"])

        paired_rows.append({
            "selection_index": selection_index.get(image_id),
            "image_id": image_id,
            "cell_label": cell_label,
            "maskcrop_present": maskcrop is not None,
            "cellprofiler_present": cellprofiler is not None,
            "maskcrop_is_square": is_square(maskcrop),
            "cellprofiler_is_square": is_square(cellprofiler),
            "same_bbox": bool(maskcrop and cellprofiler and
                              all(maskcrop[key] == cellprofiler[key] for key in bbox_keys)),
            **{f"maskcrop_{key}": value(maskcrop, key) for key in
               ("bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc")},
            **{f"cellprofiler_{key}": value(cellprofiler, key) for key in
               ("bbox_minr", "bbox_minc", "bbox_maxr", "bbox_maxc")},
            "maskcrop_neighbor_contamination": mc_contam,
            "cellprofiler_neighbor_contamination": cp_contam,
            "neighbor_contamination_delta_cellprofiler_minus_maskcrop": (
                float(cp_contam) - float(mc_contam)
                if cp_contam is not None and mc_contam is not None else None
            ),
            "maskcrop_target_retention": mc_retention,
            "cellprofiler_target_retention": cp_retention,
            "target_retention_delta_cellprofiler_minus_maskcrop": (
                float(cp_retention) - float(mc_retention)
                if cp_retention is not None and mc_retention is not None else None
            ),
        })
    return paired_rows


def _domain_tool_comparability(
        run_dir: Path, selection: dict, records_by_arm: dict[str, list[dict]],
        domain_pair_rows: list[dict]) -> dict:
    expected_ids = [item["image_id"] for item in selection["selected_images"]]
    expected_set = set(expected_ids)
    selection_hash = _sha256(run_dir / "selection.json")
    reasons = []
    arm_details = {}
    min_areas = {}
    geometry_settings = {}
    for arm in DOMAIN_TOOL_ARMS:
        records = records_by_arm.get(arm, [])
        grouped = _group_records(records)
        record_ids = set(grouped)
        missing = sorted(expected_set - record_ids)
        extra = sorted(record_ids - expected_set)
        duplicates = sorted(image_id for image_id, values in grouped.items() if len(values) != 1)
        error_ids = sorted(str(record.get("image_id")) for record in records
                           if record.get("status") != "ok")
        meta_path = run_dir / arm / "run_meta.json"
        meta = _read_json(meta_path) if meta_path.is_file() else {}
        min_areas[arm] = meta.get("min_area")
        geometry_settings[arm] = (meta.get("margin"), meta.get("maskcrop_margin_frac"))
        hash_matches = meta.get("selection_sha256") == selection_hash
        arm_details[arm] = {
            "n_records": len(records),
            "n_ok": sum(record.get("status") == "ok" for record in records),
            "missing_image_ids": missing,
            "extra_image_ids": extra,
            "duplicate_image_ids": duplicates,
            "error_image_ids": error_ids,
            "selection_sha256_matches": hash_matches,
            "min_area": meta.get("min_area"),
            "margin": meta.get("margin"),
            "maskcrop_margin_frac": meta.get("maskcrop_margin_frac"),
        }
        if missing or extra or duplicates or error_ids:
            reasons.append(f"{arm} does not have exactly one successful record per selected image")
        if not hash_matches:
            reasons.append(f"{arm} run metadata does not match selection.json")
    if min_areas["maskcrop"] is None or min_areas["cellprofiler"] is None:
        reasons.append("domain-tool run metadata is missing min_area")
    elif min_areas["maskcrop"] != min_areas["cellprofiler"]:
        reasons.append("domain tools used different min_area values")
    if any(value is None for settings in geometry_settings.values() for value in settings):
        reasons.append("domain-tool run metadata is missing square-geometry settings")
    elif geometry_settings["maskcrop"] != geometry_settings["cellprofiler"]:
        reasons.append("domain tools used different margin or maskcrop_margin_frac values")
    maskcrop_keys = {(row["image_id"], row["cell_label"]) for row in domain_pair_rows
                     if row["maskcrop_present"]}
    cellprofiler_keys = {(row["image_id"], row["cell_label"]) for row in domain_pair_rows
                         if row["cellprofiler_present"]}
    same_crop_keys = maskcrop_keys == cellprofiler_keys
    if not same_crop_keys:
        reasons.append("domain tools did not return the same (image_id, cell_label) targets")
    maskcrop_non_square = sum(row["maskcrop_present"] and not row["maskcrop_is_square"]
                              for row in domain_pair_rows)
    cellprofiler_non_square = sum(row["cellprofiler_present"] and not row["cellprofiler_is_square"]
                                  for row in domain_pair_rows)
    bbox_mismatches = sum(row["maskcrop_present"] and row["cellprofiler_present"] and
                          not row["same_bbox"] for row in domain_pair_rows)
    if maskcrop_non_square or cellprofiler_non_square:
        reasons.append("domain tools emitted non-square final crops")
    if bbox_mismatches:
        reasons.append("domain tools did not use identical final square bboxes")
    return {
        "contract": (
            "same selection, ground-truth mask, min_area, target cell labels, and "
            "Maskcrop fraction-mode square geometry"
        ),
        "selection_sha256": selection_hash,
        "n_expected_images": len(expected_ids),
        "arms": arm_details,
        "same_crop_keys": same_crop_keys,
        "n_maskcrop_crops": len(maskcrop_keys),
        "n_cellprofiler_crops": len(cellprofiler_keys),
        "n_paired_crops": len(maskcrop_keys & cellprofiler_keys),
        "n_maskcrop_non_square": maskcrop_non_square,
        "n_cellprofiler_non_square": cellprofiler_non_square,
        "n_bbox_mismatches": bbox_mismatches,
        "comparable": not reasons,
        "reasons": reasons,
    }


def _all_arm_comparability(run_dir: Path, selection: dict,
                           records_by_arm: dict[str, list[dict]]) -> dict:
    expected_ids = {item["image_id"] for item in selection["selected_images"]}
    selection_hash = _sha256(run_dir / "selection.json")
    reference = {record["image_id"]: {int(crop["cell_label"]) for crop in record.get("crops", [])}
                 for record in records_by_arm.get("maskcrop", []) if record.get("status") == "ok"}
    reasons = []
    details = {}
    min_areas = set()
    models = set()
    for arm in GATED_ARMS:
        records = records_by_arm.get(arm, [])
        meta_path = run_dir / arm / "run_meta.json"
        meta = _read_json(meta_path) if meta_path.exists() else {}
        if meta.get("min_area") is not None:
            min_areas.add(meta["min_area"])
        if arm in arm_lib.LLM_ARMS and meta.get("model"):
            models.add(meta["model"])
        grouped = _group_records(records)
        bad_ids = sorted(image_id for image_id in expected_ids
                         if len(grouped.get(image_id, [])) != 1 or
                         grouped[image_id][0].get("status") != "ok")
        wrong_targets = []
        bad_geometry = []
        for image_id in expected_ids - set(bad_ids):
            record = grouped[image_id][0]
            crops = record.get("crops") or []
            labels = [int(crop["cell_label"]) for crop in crops]
            metrics = record.get("metrics") or {}
            if (len(labels) != len(set(labels)) or
                    set(labels) != reference.get(image_id) or
                    metrics.get("crop_coverage") != 1.0 or
                    metrics.get("n_expected_crops") != len(reference.get(image_id, set()))):
                wrong_targets.append(image_id)
            for crop in crops:
                bbox = crop.get("bbox") or []
                if len(bbox) != 4 or bbox[2] - bbox[0] != bbox[3] - bbox[1]:
                    bad_geometry.append(image_id)
                    break
        hash_matches = meta.get("selection_sha256") == selection_hash
        details[arm] = {"n_records": len(records), "bad_image_ids": bad_ids,
                        "wrong_target_image_ids": sorted(wrong_targets),
                        "non_square_image_ids": sorted(bad_geometry),
                        "selection_sha256_matches": hash_matches}
        if not records:
            reasons.append(f"{arm} has no records (the arm is missing)")
        elif (bad_ids or wrong_targets or bad_geometry or set(grouped) != expected_ids or
                not hash_matches):
            reasons.append(f"{arm} failed image, target, geometry, or selection contract")
    if not reference or set(reference) != expected_ids:
        reasons.append("MaskCrop reference is incomplete")
    if len(min_areas) != 1 or any(
            not (run_dir / arm / "run_meta.json").exists() or
            _read_json(run_dir / arm / "run_meta.json").get("min_area") is None
            for arm in GATED_ARMS):
        reasons.append("arms did not record the same min_area")
    if len(models) != 1 or any(
            not (run_dir / arm / "run_meta.json").exists() or
            not _read_json(run_dir / arm / "run_meta.json").get("model")
            for arm in arm_lib.LLM_ARMS):
        reasons.append("LLM arms did not use the same model")
    react_meta_path, shape_meta_path = run_dir / "react" / "run_meta.json", run_dir / "shape" / "run_meta.json"
    react_meta = _read_json(react_meta_path) if react_meta_path.exists() else {}
    shape_meta = _read_json(shape_meta_path) if shape_meta_path.exists() else {}
    cap = shape_meta.get("concurrency_cap")
    if (not cap or react_meta.get("concurrency_cap") != cap or react_meta.get("cores") != shape_meta.get("cores")
            or react_meta.get("react_tool_budget") != react_tool_budget(cap)):
        reasons.append(f"ReAct did not use SHAPE's budget (1 + {arm_lib.MAX_ROUNDS - 1} x SHAPE's concurrency cap "
                       "tool calls on the same cores)")
    return {"gated_arms": list(GATED_ARMS), "control_arms": list(CONTROL_ARMS),
            "selection_sha256": selection_hash, "arms": details,
            "n_expected_images": len(expected_ids), "n_reference_targets": sum(map(len, reference.values())),
            "comparable": not reasons, "reasons": reasons}


def compare(args) -> None:
    selection = _read_json(args.run_dir / "selection.json")
    expected_ids = [item["image_id"] for item in selection["selected_images"]]
    arms_present = [a for a in arm_lib.ARMS if (args.run_dir / a / "records.jsonl").exists()]
    records_by_arm = {arm: _records(args.run_dir, arm) for arm in arms_present}
    image_rows, crop_rows = _detail_rows(selection, records_by_arm)
    domain_pair_rows = _domain_pair_rows(crop_rows)
    comparability = _domain_tool_comparability(
        args.run_dir, selection, records_by_arm, domain_pair_rows)
    all_comparability = _all_arm_comparability(args.run_dir, selection, records_by_arm)

    rows = []
    summary = {
        "schema": "shape.cropper.formal.compare.v4",
        "metrics": METRICS,
        "selection_sha256": _sha256(args.run_dir / "selection.json"),
        "n_expected_images": len(expected_ids),
        "comparability": {"domain_tools": comparability, "all_arms": all_comparability},
        "notes": {
            "input": "every arm crops the ground-truth mask; contamination and retention are measured on it",
            "cellprofiler": ("a control (standard-pipeline reference): its scored crops share MaskCrop's square "
                             "geometry, so its quality readouts equal MaskCrop's by construction; it reports "
                             "implementation parity and runtime"),
            "triage": ("every target is delivered once and scored, the crops SHAPE's mask check rejects included; "
                       "n_triage_rejected reports them separately"),
            "means": ("every arm mean is over comparable_image_ids: the images every present arm but the "
                      "CellProfiler control finished; the control's means are over those it also finished"),
        },
        "artifacts": {
            "per_image": ["per_image.jsonl", "per_image.csv"],
            "per_crop": ["per_crop.jsonl", "per_crop.csv"],
            "domain_tool_pairs": ["domain_tool_pairs.jsonl", "domain_tool_pairs.csv"],
        },
        "arms": {},
    }
    # the comparable set: the selected images every present arm finished (exactly one ok record), the
    # CellProfiler control aside (its gaps are reported by the domain-tool check); a missing arm is reported
    # by the comparable flag and its reasons, not by blanking the other arms' means
    def finished(arm: str) -> set[str]:
        return {image_id for image_id, values in _group_records(records_by_arm.get(arm, [])).items()
                if len(values) == 1 and values[0].get("status") == "ok"}

    comparable_ids = set(expected_ids)
    for arm in arms_present:
        if arm not in CONTROL_ARMS:
            comparable_ids &= finished(arm)
    summary["comparable_image_ids"] = [image_id for image_id in expected_ids if image_id in comparable_ids]
    summary["n_comparable_images"] = len(comparable_ids)
    summary["missing_arms"] = [arm for arm in arm_lib.ARMS if arm not in arms_present]
    for arm in arms_present:
        records = records_by_arm[arm]
        ok = [record for record in records if record.get("status") == "ok"]
        arm_ids = comparable_ids & finished(arm)
        metrics = [record.get("metrics") or {} for record in ok if record.get("image_id") in arm_ids]

        def mean(key: str):
            vals = [metric.get(key) for metric in metrics if metric.get(key) is not None]
            return float(np.mean(vals)) if vals else None

        block = {key: mean(key) for key in (*METRICS, *MODEL_USAGE_FIELDS, *SHAPE_USAGE_FIELDS,
                                            "n_triage_rejected", "rounds_per_image")}
        grouped = _group_records(records)
        block.update({
            "comparable": bool(all_comparability["comparable"]),
            "n_comparable_images": len(arm_ids),
            "n_images": len(ok),
            "n_errors": len(records) - len(ok),
            "n_records": len(records),
            "n_expected_images": len(expected_ids),
            "n_missing_images": len(set(expected_ids) - set(grouped)),
            "n_duplicate_images": sum(len(values) != 1 for values in grouped.values()),
            "n_crops": sum(len(record.get("crops") or []) for record in ok),
        })
        summary["arms"][arm] = block
        rows.append({"arm": arm, **block})

    out = args.run_dir / "compare"
    out.mkdir(exist_ok=True)
    _write_rows(out / "per_image", image_rows, PER_IMAGE_FIELDS)
    _write_rows(out / "per_crop", crop_rows, PER_CROP_FIELDS)
    _write_rows(out / "domain_tool_pairs", domain_pair_rows, DOMAIN_PAIR_FIELDS)
    _write_json(out / "summary.json", summary)
    with (out / "summary.csv").open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "arm", *METRICS, *MODEL_USAGE_FIELDS, *SHAPE_USAGE_FIELDS, "n_triage_rejected", "rounds_per_image",
            "comparable", "n_comparable_images", "n_images", "n_errors", "n_records", "n_expected_images",
            "n_missing_images", "n_duplicate_images", "n_crops",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)
    print(
        f"wrote {out / 'summary.json'}, per-image rows={len(image_rows)}, "
        f"per-crop rows={len(crop_rows)}, domain-tool pairs={len(domain_pair_rows)}"
    )


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p = sub.add_parser("prepare", help="randomly select the pilot image set")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--data-dir", type=Path, required=True)
    p.add_argument("--n-images", type=int, default=30)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--label-suffix", default="_label")
    p.add_argument("--label-ext", default=".tiff")

    p = sub.add_parser("run", help="run one formal arm")
    p.add_argument("--run-dir", type=Path, required=True)
    p.add_argument("--arm", choices=arm_lib.ARMS, required=True)
    p.add_argument("--model", default=arm_lib.DEFAULT_MODEL)
    p.add_argument("--min-area", type=int, default=SEGMENTER_DEFAULT_MIN_SIZE)
    p.add_argument("--margin", type=int, default=rsc.DEFAULT_MARGIN)
    p.add_argument("--maskcrop-margin-frac", type=float, default=rsc.DEFAULT_MASKCROP_MARGIN_FRAC)
    p.add_argument("--gpt-batch-size", type=int, default=arm_lib.GPT_BATCH_SIZE,
                   help="targets per GPT request; every eligible target is processed")
    p.add_argument("--cores", type=int, default=None,
                   help="cores for the react and shape arms (default: the cores this process may use); the "
                        "concurrency cap is cores // THREADS_PER_CALL, SHAPE's per-round cap and ReAct's budget "
                        "factor")
    p.add_argument("--deadline", type=float, default=None,
                   help="epoch seconds by which the run must end (shown to SHAPE and ReAct as the time left)")
    p.add_argument("--resume", action="store_true",
                   help="keep the arm's finished images and run the failed or missing ones again (the same code, "
                        "selection and settings; e.g. after the job's time limit)")
    p.add_argument("--shape-loop", choices=("agentic", "linear"), default="agentic")
    p.add_argument(
        "--cellprofiler-python",
        default=os.environ.get("CELLPROFILER_PYTHON"),
        help="Python executable/wrapper with CellProfiler (or set CELLPROFILER_PYTHON)",
    )

    p = sub.add_parser("compare", help="aggregate the registered readouts")
    p.add_argument("--run-dir", type=Path, required=True)
    args = ap.parse_args()
    if args.cmd == "prepare":
        prepare(args)
    elif args.cmd == "run":
        run_arm(args)
    elif args.cmd == "compare":
        compare(args)


if __name__ == "__main__":
    main()

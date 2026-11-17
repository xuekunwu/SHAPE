#!/usr/bin/env python3
"""NeurIPS22-CellSeg segmentation benchmark: seven arms on the same difficulty-banded images.

Arms (see runners/cellseg_arms.py): cellpose_sam, cellpose3, microsam, stardist (domain tools); gpt (gpt-5.6-luna alone),
react (gpt-5.6-luna + Cellpose-SAM on the shared ReAct runner, `run_react_arm`), shape (the SHAPE agent).

Bands are the committed Cellpose-SAM-alone per-image AP50 (data/NeurIPS22-CellSeg-cell_image_metrics.csv):
hard < 0.30 <= medium <= 0.60 < easy. `lists` draws n images per band (seed 0) once per run dir;
every arm then runs exactly those images.

Readouts, per arm x band and pooled (`compare`):
    AP50, mAP@0.50:0.95, runtime per image (s), tool calls per image,
    AP50 gain per tool call = sum(AP50 - Cellpose-SAM AP50) / sum(tool calls)   (the definition
        of benchmarks/_shared/segmentation_metrics.py; the reference is this run's cellpose_sam arm),
    LLM tokens per image.
An image an arm attempted without delivering a mask scores 0. Only images every compared arm
attempted enter the readouts (an arm without any record is left out and reported); plus SHAPE minus each arm, paired per image: mean delta AP50,
bootstrap 95% CI and a two-sided sign test.

Layout of a run dir (linked as this folder's results/<name>):
    lists/<band>.txt                     the drawn stems
    <arm>/run_meta.json                  commit, host, env, package versions, model, times
    <arm>/<band>/records.jsonl           one scored record per image (the resume point)
    <arm>/<band>/<image>/mask.tif        the delivered mask, + trace.json and the arm's own files
    <arm>/<band>/run_summary.json        the band's macro readouts
    compare/                             readouts.csv, per_image.csv, paired_vs_shape.csv,
                                         summary.md, readouts.png

Usage:
    python run_cellseg_benchmark.py lists   --run-dir R [--n-per-band 20] [--seed 0]
    python run_cellseg_benchmark.py run     --run-dir R --arm stardist [--bands ...] [--limit N] [--workers W (react)]
    python run_cellseg_benchmark.py run-set --run-dir R --arm shape_cohort [--workers W] [--limit N]
    python run_cellseg_benchmark.py compare --run-dir R [--arms cellpose_sam shape ...]
    python run_cellseg_benchmark.py smoke   --arm microsam --image path/to/image.png

(The module names carry `cellseg_`: the cropper benchmark has its own `arms` and `run_benchmark`.)
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import platform
import random
import shutil
import subprocess
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

_HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(_HERE / "runners"))     # the arms, the config and the runners it reuses
import cellseg_arms as arm_lib         # noqa: E402
import run_cellpose_sam_only as score  # noqa: E402  -- shape-free scoring (same AP definition)

BANDS = ("hard", "medium", "easy")
HARD_MAX, MEDIUM_MAX = 0.30, 0.60
METRICS_CSV_NAME = "NeurIPS22-CellSeg-cell_image_metrics.csv"
METRICS_CSV = _HERE / "results" / METRICS_CSV_NAME
IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
READOUTS = (("ap50", "AP50"), ("map_50_95", "mAP@0.50:0.95"),
            ("seconds_per_image", "Runtime (s / image)"), ("tool_calls_per_image", "Tool calls / image"),
            ("ap50_gain_per_tool_call", "AP50 gain / tool call"), ("tokens_per_image", "Tokens / image"))
_SCORE_KEYS = ("ap50", "ap75", "map_50_95", "precision_at_50", "recall_at_50", "f1_at_50",
               "ground_truth_cell_count", "predicted_cell_count", "cell_count_error_fraction")
IMAGE_WORKERS = 8          # images at a time: run-set's image_workers, the react arm's concurrent sessions
REACT_DESIGN = "benchmarks/_shared/react_arm.py"   # the react arm's run_meta.json marker (REACT_ARM_CONTRACT.md)
ARM_COLORS = {"cellpose_sam": "#9e9e9e", "cellpose3": "#d9d9d9", "microsam": "#80b1d3", "stardist": "#b3de69",
              "gpt": "#fdb462", "react": "#bc80bd", "shape": "#3b6fb0",
              "shape_cohort": "#e41a1c"}


# --------------------------------------------------------------------------- #
# Bands and lists
# --------------------------------------------------------------------------- #
def read_band_metric(csv_path: Path = METRICS_CSV) -> list[tuple[str, float]]:
    """(stem, Cellpose-SAM AP50) for every scored row of the committed CSV (it has a BOM)."""
    with open(csv_path, newline="", encoding="utf-8-sig") as handle:
        rows = []
        for row in csv.DictReader(handle):
            name = (row.get("cell_image_name") or row.get("image_name") or "").strip()
            try:
                rows.append((Path(name).stem, float(row["ap50"])))
            except (KeyError, TypeError, ValueError):
                continue
    return [r for r in rows if r[0]]


def band_metric_csv(data_root: Path | None) -> Path:
    """Where the per-image Cellpose-SAM AP50 table is: the dataset's own copy, else this folder's."""
    if data_root is not None and (data_root / METRICS_CSV_NAME).exists():
        return data_root / METRICS_CSV_NAME
    return METRICS_CSV


def partition_bands(rows, hard_max: float = HARD_MAX, medium_max: float = MEDIUM_MAX) -> dict:
    """hard: ap50 < hard_max; medium: hard_max <= ap50 <= medium_max; easy: ap50 > medium_max."""
    return {"hard": sorted(r for r in rows if r[1] < hard_max),
            "medium": sorted(r for r in rows if hard_max <= r[1] <= medium_max),
            "easy": sorted(r for r in rows if r[1] > medium_max)}


def sample_band(band_rows, n: int, seed: int) -> list[tuple[str, float]]:
    rows = list(band_rows)
    return rows if len(rows) <= n else sorted(random.Random(seed).sample(rows, n))


def image_pixels(path: Path) -> int:
    """Height x width from the file header (no full decode)."""
    if path.suffix.lower() in (".tif", ".tiff"):
        import tifffile
        with tifffile.TiffFile(str(path)) as tf:
            shape = tf.series[0].shape
        hw = sorted(shape, reverse=True)[:2] if len(shape) > 2 else shape
        return int(hw[0]) * int(hw[1])
    from PIL import Image
    with Image.open(path) as im:
        return int(im.size[0]) * int(im.size[1])


def largest_images(data_root: Path, n: int) -> dict[str, int]:
    """stem -> pixel count of the n largest images under data_root/images."""
    sizes = {stem: image_pixels(p) for stem, p in _image_index(data_root / "images").items()}
    return dict(sorted(sizes.items(), key=lambda kv: -kv[1])[:n])


def write_lists(run_dir: Path, n_per_band: int, seed: int, csv_path: Path = METRICS_CSV,
                largest: dict[str, int] | None = None, exclude: set[str] | None = None) -> dict:
    """lists/<band>.txt, drawn once: existing lists are kept (a run dir has one image set).
    With `largest` (stem -> pixels, a stress test) a band holds only those images, each in its own band.
    `exclude`: images no band may draw (e.g. every image earlier runs used), left out before the draw."""
    out = {}
    bands = partition_bands(read_band_metric(csv_path))
    if exclude:
        bands = {b: [(s, a) for s, a in rows if s not in exclude] for b, rows in bands.items()}
    for band in BANDS:
        path = run_dir / "lists" / f"{band}.txt"
        if not path.exists():
            path.parent.mkdir(parents=True, exist_ok=True)
            if largest is not None:
                rows = [(s, a) for s, a in bands[band] if s in largest]
                text = "".join(f"{s}  # cellpose_sam_ap50={a:.3f} pixels={largest[s]}\n" for s, a in rows)
            else:
                text = "".join(f"{s}  # cellpose_sam_ap50={a:.3f}\n"
                               for s, a in sample_band(bands[band], n_per_band, seed))
            path.write_text(text, encoding="utf-8")
        out[band] = read_list(path)
        print(f"[lists] {band}: {len(out[band])} of {len(bands[band])} images -> {path}")
    return out


def read_list(path: Path) -> list[str]:
    names = (ln.split("#", 1)[0].strip() for ln in path.read_text(encoding="utf-8").splitlines())
    return [n for n in names if n]


# --------------------------------------------------------------------------- #
# Running one arm
# --------------------------------------------------------------------------- #
def _token_counter():
    try:
        from shape.engine import usage
        return usage.total_tokens
    except Exception:          # an env without shape (the microsam arm): no LLM tokens
        return None


def _write_mask(path: Path, mask: np.ndarray) -> None:
    import tifffile
    tifffile.imwrite(str(path), np.asarray(mask).astype(np.int32), compression="zlib")


def run_image(arm, arm_name: str, image_path: Path, gt_path: Path, out_dir: Path) -> dict:
    """Run one arm on one image, keep its mask and trace, and score the mask against GT.
    The image's dir starts empty, so a killed run's files (e.g. the segmenter's attempt masks,
    which the shape arm's tool-call count reads) cannot leak into this one."""
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir(parents=True)
    tokens = _token_counter()
    tok0 = tokens() if tokens else None
    t0 = time.perf_counter()
    rec = {"image": image_path.stem, "arm": arm_name, "error": None}
    try:
        res = arm(image_path, out_dir)
    except Exception as exc:
        res = {"mask": None, "tool_calls": None, "trace": {}}
        rec["error"] = f"{type(exc).__name__}: {exc}"
    rec["seconds"] = round(time.perf_counter() - t0, 2)
    rec["tool_calls"] = res.get("tool_calls")
    rec["tokens"] = (tokens() - tok0) if tokens else (None if arm_name in arm_lib.LLM_ARMS else 0)
    (out_dir / "trace.json").write_text(json.dumps(res.get("trace") or {}, indent=1, default=str),
                                        encoding="utf-8")
    mask = res.get("mask")
    if mask is None:
        rec["error"] = rec["error"] or "no mask delivered"
    else:
        try:
            _write_mask(out_dir / "mask.tif", mask)
            metrics = score.evaluate_instance_segmentation(np.asarray(mask),
                                                           score.load_label_mask(gt_path))
            rec.update({k: metrics[k] for k in _SCORE_KEYS})
        except Exception as exc:
            rec["error"] = f"scoring: {type(exc).__name__}: {exc}"
    rec["delivered"] = "ap50" in rec
    if not rec["delivered"]:          # attempted, nothing scorable delivered -> 0
        rec.update(ap50=0.0, ap75=0.0, map_50_95=0.0)
    return rec


def _package_versions() -> dict:
    from importlib import metadata
    out = {}
    for pkg in ("cellpose", "micro_sam", "stardist", "tensorflow", "tensorflow-cpu", "torch",
                "numpy", "openai"):
        try:
            out[pkg] = metadata.version(pkg)
        except metadata.PackageNotFoundError:
            pass
    return out


def _commit() -> str | None:
    try:
        return subprocess.run(["git", "-C", str(_HERE), "rev-parse", "HEAD"], capture_output=True,
                              text=True, check=True).stdout.strip()
    except Exception:
        return None


def _image_index(image_dir: Path) -> dict[str, Path]:
    return {p.stem: p for p in sorted(image_dir.iterdir())
            if p.is_file() and p.suffix.lower() in IMAGE_EXTS}


def run_arm(run_dir: Path, arm_name: str, data_root: Path, *, bands=BANDS, limit: int | None = None,
            model: str, max_steps: int, label_suffix: str = "_label", label_ext: str = ".tiff",
            arm=None, workers: int = IMAGE_WORKERS) -> None:
    if arm_name == "react" and arm is None:
        return run_react_arm(run_dir, data_root, bands=bands, limit=limit, model=model, workers=workers,
                             label_suffix=label_suffix, label_ext=label_ext)
    images = _image_index(data_root / "images")
    meta_path = run_dir / arm_name / "run_meta.json"
    meta_path.parent.mkdir(parents=True, exist_ok=True)
    meta = {"arm": arm_name, "commit": _commit(), "host": platform.node(), "python": sys.version,
            "env": sys.prefix, "packages": _package_versions(), "model": model,
            "max_steps": max_steps, "data_root": str(data_root), "bands": list(bands),
            "limit": limit, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    arm = arm or arm_lib.make_arm(arm_name, model=model, max_steps=max_steps)
    for band in bands:
        stems = read_list(run_dir / "lists" / f"{band}.txt")[:limit]
        band_dir = run_dir / arm_name / band
        band_dir.mkdir(parents=True, exist_ok=True)
        records_path = band_dir / "records.jsonl"
        done = {r["image"] for r in load_records(records_path)}
        for i, stem in enumerate(stems, 1):
            if stem in done:
                continue
            if stem not in images:
                print(f"[{arm_name}/{band}] {stem}: image not found in {data_root / 'images'}")
                continue
            rec = run_image(arm, arm_name, images[stem],
                            data_root / "labels" / f"{stem}{label_suffix}{label_ext}",
                            band_dir / stem)
            rec["band"] = band
            with open(records_path, "a", encoding="utf-8") as handle:
                handle.write(json.dumps(rec, default=str) + "\n")
            print(f"[{arm_name}/{band} {i}/{len(stems)}] {stem}: AP50={rec['ap50']:.3f} "
                  f"mAP={rec['map_50_95']:.3f} calls={rec['tool_calls']} tokens={rec['tokens']} "
                  f"t={rec['seconds']:.1f}s" + (f"  ERROR {rec['error']}" if rec["error"] else ""),
                  flush=True)
        recs = [r for r in load_records(records_path) if r["image"] in set(stems)]
        summary = {"arm": arm_name, "band": band, "n_listed": len(stems), "n_attempted": len(recs),
                   "n_errors": sum(1 for r in recs if r["error"]),
                   "readouts": readouts(recs, baseline_ap50={})}
        (band_dir / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    meta["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


REACT_INPUT = "One microscopy image; this session segments only this image."


def run_react_arm(run_dir: Path, data_root: Path, *, bands=BANDS, limit: int | None = None, model: str,
                  workers: int = IMAGE_WORKERS, label_suffix: str = "_label", label_ext: str = ".tiff",
                  engine=None, tool=None, verifier=None) -> None:
    """The react arm: an LLM plus Cell_Segmenter_Tool on the shared ReAct runner (`benchmarks/_shared/react_arm.py`,
    REACT_ARM_CONTRACT.md; the adapter `cellseg_arms.SegmenterAdapter`). Per band, one `run_react` over the band's
    listed images without a record, `workers` sessions at a time; the task text is SHAPE's in this benchmark (the
    shape arm's question, `Prompt.txt`), the raw input image on each session's first question
    (`SegmenterAdapter.input_images`), the engine SHAPE's verifier engine (`build_vision_engine`), no
    deadline. Each image's delivered call (the agent's finish, else its default call: `delivery`) is scored as
    `run_image` scores a mask, into the same record layout, plus `rounds`, `delivered_call`, `delivery`, `stop`,
    `delivered_call_error` and `wall_seconds_per_image` (the band's wall time over its images; `seconds` is the
    image's own session, which shares the GPU with the other sessions). Per image dir: the runner's TRACE.jsonl,
    CALLS.json, SELECTED.json, the tool's files, trace.json and mask.tif; per band: REACT_SUMMARY.json. A react dir
    with records of the earlier ReAct design (its run_meta.json without `design`) is refused."""
    if str(_HERE.parents[1]) not in sys.path:              # the repo root: benchmarks._shared
        sys.path.insert(0, str(_HERE.parents[1]))
    from benchmarks._shared import react_arm
    from shape.agent.verification.base import BUDGET
    arm_dir = run_dir / "react"
    meta_path = arm_dir / "run_meta.json"
    old = json.loads(meta_path.read_text(encoding="utf-8")) if meta_path.is_file() else {}
    if old.get("design") != REACT_DESIGN and any(load_records(arm_dir / b / "records.jsonl") for b in BANDS):
        raise SystemExit(f"{arm_dir} holds records of the earlier ReAct design (superseded): move it aside first")
    import run_shape_neurips_cellseg as bench
    task_text = bench._default_question()
    if engine is None:
        from shape.agent.verification.vision_engine import build_vision_engine
        engine = build_vision_engine(model)                # None: run_react raises NoEngineError
    if verifier is None:
        from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier
        verifier = SegmentationAgenticVerifier()
    if tool is None:
        from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
        tool = Cell_Segmenter_Tool()
    images = _image_index(data_root / "images")
    probe = arm_lib.SegmenterAdapter({}, {}, tool=tool, verifier=verifier)
    arm_dir.mkdir(parents=True, exist_ok=True)
    meta = {"arm": "react", "design": REACT_DESIGN, "commit": _commit(), "host": platform.node(),
            "python": sys.version, "env": sys.prefix, "packages": _package_versions(), "model": model,
            "budget": BUDGET, "round_cap": probe.round_cap(), "default_counts_as_round": probe.default_counts_as_round(),
            "workers": workers, "parameters": [vars(p) for p in probe.parameters()], "data_root": str(data_root),
            "bands": list(bands), "limit": limit, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")
    for band in bands:
        stems = read_list(run_dir / "lists" / f"{band}.txt")[:limit]
        band_dir = arm_dir / band
        band_dir.mkdir(parents=True, exist_ok=True)
        records_path = band_dir / "records.jsonl"
        done = {r["image"] for r in load_records(records_path)}
        for stem in stems:
            if stem not in done and stem not in images:
                print(f"[react/{band}] {stem}: image not found in {data_root / 'images'}")
        todo = [s for s in stems if s not in done and s in images]
        if todo:
            for stem in todo:                              # each image starts from an empty dir, as in run_image
                if (band_dir / stem).exists():
                    shutil.rmtree(band_dir / stem)
                (band_dir / stem).mkdir(parents=True)
            adapter = arm_lib.SegmenterAdapter({s: images[s] for s in todo}, {s: band_dir / s for s in todo},
                                               tool=tool, verifier=verifier, private_roots=[data_root / "labels"])
            out = react_arm.run_react(adapter, task_text=task_text, input_text=REACT_INPUT, engine=engine,
                                      out_dir=band_dir, workers=workers, deadline=None,
                                      unit_input=lambda s: arm_lib.input_facts(images[s]),
                                      unit_images=adapter.input_images)
            wall = out["totals"]["wall_seconds"] / len(todo)
            for i, stem in enumerate(todo, 1):
                rec = _react_record(stem, band, out["selected"][stem], adapter.delivered_mask_path(stem),
                                    data_root / "labels" / f"{stem}{label_suffix}{label_ext}", band_dir / stem)
                rec["wall_seconds_per_image"] = round(wall, 2)
                with open(records_path, "a", encoding="utf-8") as handle:
                    handle.write(json.dumps(rec, default=str) + "\n")
                print(f"[react/{band} {i}/{len(todo)}] {stem}: AP50={rec['ap50']:.3f} mAP={rec['map_50_95']:.3f} "
                      f"calls={rec['tool_calls']} rounds={rec['rounds']} tokens={rec['tokens']} "
                      f"t={rec['seconds']:.1f}s delivered={rec['delivered_call']} ({rec['delivery']})"
                      + (f"  ERROR {rec['error']}" if rec["error"] else ""), flush=True)
        recs = [r for r in load_records(records_path) if r["image"] in set(stems)]
        summary = {"arm": "react", "band": band, "n_listed": len(stems), "n_attempted": len(recs),
                   "n_errors": sum(1 for r in recs if r["error"]), "readouts": readouts(recs, baseline_ap50={})}
        (band_dir / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    meta["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_path.write_text(json.dumps(meta, indent=1, default=str), encoding="utf-8")


def _react_record(stem: str, band: str, selected: dict, mask_path, gt_path: Path, out_dir: Path) -> dict:
    """One react image's record (`run_image`'s fields) from its session's SELECTED and the delivered call's mask."""
    rec = {"image": stem, "arm": "react", "error": None, "seconds": selected["seconds"],
           "tool_calls": selected["calls"], "tokens": selected["tokens"]["total_tokens"],
           "rounds": selected["rounds"], "delivered_call": selected["call_id"], "delivery": selected["source"],
           "stop": selected["stop"], "delivered_call_error": selected.get("call_error")}
    (out_dir / "trace.json").write_text(json.dumps(
        {"design": REACT_DESIGN, "selected": selected, "mask_path": mask_path,
         "records": ["TRACE.jsonl", "CALLS.json", "SELECTED.json"]}, indent=1, default=str), encoding="utf-8")
    if not mask_path or not Path(mask_path).is_file():
        rec["error"] = selected.get("call_error") or "no mask delivered"
    else:
        try:
            mask = arm_lib.load_mask(mask_path)
            _write_mask(out_dir / "mask.tif", mask)
            metrics = score.evaluate_instance_segmentation(np.asarray(mask), score.load_label_mask(gt_path))
            rec.update({k: metrics[k] for k in _SCORE_KEYS})
        except Exception as exc:
            rec["error"] = f"scoring: {type(exc).__name__}: {exc}"
    rec["delivered"] = "ap50" in rec
    if not rec["delivered"]:
        rec.update(ap50=0.0, ap75=0.0, map_50_95=0.0)
    rec["band"] = band
    return rec


# --------------------------------------------------------------------------- #
# A set arm: one solve() over a set of images (the segmentation cohort on / off)
# --------------------------------------------------------------------------- #
def delivered_problems(stage_record: dict | None) -> list[str]:
    """The admission standard on an image's delivered segmentation, from its stage record (the solver's per-image
    verification record): the delivered call's measured problems; nothing delivered or a conceded stage is a
    problem (as `segmentation_cohort.problems_of` reads a resolution)."""
    stage = next((s for s in (stage_record or {}).get("stages") or [] if s.get("stage") == "segmentation"), None)
    if stage is None or stage.get("no_output"):
        return ["no segmentation was delivered"]
    details = stage.get("details") or {}
    found = list((details.get("measured_problems") or {}).get(details.get("call")) or [])
    if stage.get("status", "satisfied") != "satisfied":
        found.append(f"the stage {stage['status']} ({stage.get('reason')})")
    return found


def _set_images(memory) -> tuple[dict, dict, dict | None]:
    """image_id -> (its result entry, its stage record) of the segmentation step of one solve(), and the step's
    cohort record (None when it ran without one)."""
    results, records, cohort = {}, {}, None
    for action in memory.get_actions(llm_safe=False):
        result, ver = action.get("result"), action.get("verification") or {}
        if not (isinstance(result, dict) and isinstance(result.get("per_image"), list)):
            continue
        for entry, record in zip(result["per_image"], ver.get("per_image") or []):
            if isinstance(entry, dict) and entry.get("image_id"):
                results[entry["image_id"]], records[entry["image_id"]] = entry, record
        cohort = ver.get("cohort", cohort)
    return results, records, cohort


def run_set(run_dir: Path, arm_name: str, data_root: Path, *, bands=BANDS, limit: int | None = None,
            model: str, max_steps: int, workers: int = 8, label_suffix: str = "_label", label_ext: str = ".tiff",
            preflight_cache: Path | None = None, solve=None) -> None:
    """A set arm (`arm_lib.SET_ARMS`: `shape_cohort`) on the run dir's lists, every band together: one solve() over
    all listed images with the segmentation cohort on, `workers` images at a time (`image_workers`). Its
    calibration and groups span the whole set, so a partly recorded run is refused (remove <arm>/ to rerun it).
    Per image: the record of `run_arm` (scored against GT) plus its `parameter_source` and the admission-standard
    `problems` of its delivered call; `seconds` and `tokens` are the solve()'s wall time and tokens divided by
    its images. <arm>/set/: the solve()'s cache, set_meta.json (images, wall time, tokens, workers, stop reason,
    the cohort record), verification_summary.json and preflight_cache.json (every preflight read of the run).
    `preflight_cache`: an earlier run's preflight_cache.json, loaded into the preflight's process cache first, so
    those reads are not made again (a read is reused only for the same file: path, mtime and size; recorded as
    `preflight_reused`, and the run's wall time and tokens then exclude them)."""
    if arm_name not in arm_lib.SET_ARMS:
        raise ValueError(f"{arm_name!r} is not a set arm ({', '.join(arm_lib.SET_ARMS)})")
    if solve is None:
        sys.path.insert(0, str(_HERE / "runners"))
        import run_shape_neurips_cellseg as bench
        from shape.engine.factory import resolve_api_key
        api_key = resolve_api_key(model)
        question = bench._default_question()

        def solve(**kwargs):
            return bench.solve(question=question, llm_engine_name=model, api_key=api_key, max_steps=max_steps,
                               enabled_tools=bench._SHAPE_MAIN_TOOLS, **kwargs)
    images = _image_index(data_root / "images")
    arm_dir = run_dir / arm_name
    meta_path = arm_dir / "run_meta.json"
    arm_dir.mkdir(parents=True, exist_ok=True)
    listed = [(band, stem) for band in bands for stem in read_list(run_dir / "lists" / f"{band}.txt")[:limit]]
    for band, stem in listed:
        if stem not in images:
            print(f"[{arm_name}/{band}] {stem}: image not found in {data_root / 'images'}")
    items = [(b, s) for b, s in listed if s in images]
    done = {(b, r["image"]) for b in bands for r in load_records(arm_dir / b / "records.jsonl")}
    if done:
        if len(done) < len(items):
            raise SystemExit(f"{arm_dir}: {len(done)} of {len(items)} images are recorded; the cohort runs over the "
                             f"whole set: remove {arm_dir} to rerun it")
        return
    meta = {"arm": arm_name, "commit": _commit(), "host": platform.node(), "python": sys.version,
            "env": sys.prefix, "packages": _package_versions(), "model": model, "max_steps": max_steps,
            "data_root": str(data_root), "bands": list(bands), "limit": limit, "image_workers": workers,
            "segmentation_cohort": True, "started_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())}
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    set_dir = arm_dir / "set"
    if set_dir.exists():
        shutil.rmtree(set_dir)
    set_dir.mkdir(parents=True)
    specs = [{"path": str(images[s]), "image_id": s, "group": "default"} for _, s in items]
    reused = _load_preflight(preflight_cache) if preflight_cache else 0
    meta.update(preflight_cache=str(preflight_cache) if preflight_cache else None, preflight_reused=reused)
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")
    tokens = _token_counter()
    tok0 = tokens() if tokens else None
    t0 = time.perf_counter()
    print(f"[{arm_name}] {len(specs)} images, {workers} at a time", flush=True)
    result = solve(images=specs, query_cache_dir=str(set_dir), image_workers=workers, segmentation_cohort=True)
    wall = time.perf_counter() - t0
    used = (tokens() - tok0) if tokens else None
    entries, stage_records, cohort = _set_images(result["memory"])
    if not any((entries.get(s) or {}).get("mask_path") for _, s in items):
        raise RuntimeError(f"{arm_name}: no image got a mask (stop_reason {result.get('stop_reason')!r}); "
                           f"nothing recorded")
    (set_dir / "set_meta.json").write_text(json.dumps(
        {"images": [s for _, s in items], "seconds": round(wall, 2), "tokens": used, "image_workers": workers,
         "stop_reason": result.get("stop_reason"), "preflight_reused": reused, "cohort": cohort},
        indent=1, default=str), encoding="utf-8")
    _save_preflight(set_dir / "preflight_cache.json")
    (set_dir / "verification_summary.json").write_text(
        json.dumps(result.get("verification_summary") or [], indent=1, default=str), encoding="utf-8")
    for band, stem in items:
        out_dir = arm_dir / band / stem
        if out_dir.exists():
            shutil.rmtree(out_dir)
        out_dir.mkdir(parents=True)
        entry, record = entries.get(stem) or {}, stage_records.get(stem)
        prediction = entry.get("mask_path")
        rec = {"image": stem, "arm": arm_name, "band": band, "error": None,
               "seconds": round(wall / len(items), 2),
               "tool_calls": len(_attempt_masks(prediction)) if prediction else None,
               "tokens": None if used is None else used / len(items),
               "parameter_source": entry.get("parameter_source", "per_image_vrr"),
               "problems": delivered_problems(record)}
        (out_dir / "trace.json").write_text(json.dumps(
            {"parameter_source": rec["parameter_source"], "mask_path": prediction,
             "stage_record": record}, indent=1, default=str), encoding="utf-8")
        if not prediction:
            rec["error"] = entry.get("error") or "no mask delivered"
        else:
            try:
                mask = arm_lib.load_mask(prediction)
                _write_mask(out_dir / "mask.tif", mask)
                metrics = score.evaluate_instance_segmentation(
                    np.asarray(mask), score.load_label_mask(data_root / "labels" / f"{stem}{label_suffix}{label_ext}"))
                rec.update({key: metrics[key] for key in _SCORE_KEYS})
            except Exception as exc:
                rec["error"] = f"scoring: {type(exc).__name__}: {exc}"
        rec["delivered"] = "ap50" in rec
        if not rec["delivered"]:
            rec.update(ap50=0.0, ap75=0.0, map_50_95=0.0)
        (arm_dir / band).mkdir(parents=True, exist_ok=True)
        with open(arm_dir / band / "records.jsonl", "a", encoding="utf-8") as handle:
            handle.write(json.dumps(rec, default=str) + "\n")
    print(f"[{arm_name}] done in {wall:.0f} s, tokens {used}", flush=True)
    for band in bands:
        stems = {s for b, s in listed if b == band}
        recs = [r for r in load_records(arm_dir / band / "records.jsonl") if r["image"] in stems]
        summary = {"arm": arm_name, "band": band, "n_listed": len(stems), "n_attempted": len(recs),
                   "n_errors": sum(1 for r in recs if r["error"]), "readouts": readouts(recs, baseline_ap50={})}
        (arm_dir / band).mkdir(parents=True, exist_ok=True)
        (arm_dir / band / "run_summary.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    meta["ended_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    meta_path.write_text(json.dumps(meta, indent=1), encoding="utf-8")


def _load_preflight(path: Path) -> int:
    """An earlier run's preflight reads (`_save_preflight`) into the preflight's process cache; the number loaded."""
    from shape.agent.verification import image_context
    items = json.loads(Path(path).read_text(encoding="utf-8"))
    for key, profile in items:
        image_context._PROFILE_CACHE[tuple(key)] = image_context.ImageContextProfile(**profile)
    print(f"[preflight] {len(items)} reads loaded from {path}", flush=True)
    return len(items)


def _save_preflight(path: Path) -> None:
    """This process's preflight reads: [[key (path, mtime, size), profile], ...]."""
    try:
        from shape.agent.verification import image_context
    except ImportError:
        return
    items = [[list(k), v.model_dump(mode="json")] for k, v in list(image_context._PROFILE_CACHE.items())]
    path.write_text(json.dumps(items), encoding="utf-8")


def _attempt_masks(mask_path) -> list:
    """The segmenter's per-call masks of one image (its `_a<N>` siblings): the tool calls the image cost."""
    if str(_HERE.parent / "_shared") not in sys.path:
        sys.path.insert(0, str(_HERE.parent / "_shared"))
    import segmentation_metrics as sm
    return sm.attempt_masks_for(mask_path)


def load_records(path: Path) -> list[dict]:
    if not path.exists():
        return []
    return [json.loads(ln) for ln in path.read_text(encoding="utf-8").splitlines() if ln.strip()]


# --------------------------------------------------------------------------- #
# Readouts and comparison
# --------------------------------------------------------------------------- #
def _mean(values):
    vals = [v for v in values if v is not None]
    return float(np.mean(vals)) if vals else None


def readouts(records: list[dict], baseline_ap50: dict[str, float]) -> dict:
    """The six readouts over records (the gain needs the Cellpose-SAM AP50 of each image)."""
    paired = [r for r in records if r["image"] in baseline_ap50 and r.get("tool_calls") is not None]
    calls = sum(r["tool_calls"] for r in paired)
    gain = sum(r["ap50"] - baseline_ap50[r["image"]] for r in paired)
    tokens = [r.get("tokens") for r in records]
    return {"n": len(records), "n_errors": sum(1 for r in records if r.get("error")),
            "ap50": _mean(r["ap50"] for r in records),
            "map_50_95": _mean(r["map_50_95"] for r in records),
            "seconds_per_image": _mean(r.get("seconds") for r in records),
            "tool_calls_per_image": _mean(r.get("tool_calls") for r in records),
            "ap50_gain_per_tool_call": (gain / calls) if calls else None,
            "tokens_per_image": None if any(t is None for t in tokens) else _mean(tokens)}


def bootstrap_ci(deltas, n_boot: int = 10000, seed: int = 0) -> tuple[float, float]:
    rng = np.random.default_rng(seed)
    d = np.asarray(deltas, dtype=float)
    means = d[rng.integers(0, len(d), size=(n_boot, len(d)))].mean(axis=1)
    return float(np.percentile(means, 2.5)), float(np.percentile(means, 97.5))


def sign_test_p(deltas, eps: float = 1e-9) -> float:
    """Two-sided exact sign test; ties (|delta| <= eps) are dropped."""
    pos = sum(1 for d in deltas if d > eps)
    neg = sum(1 for d in deltas if d < -eps)
    n = pos + neg
    if n == 0:
        return 1.0
    tail = sum(math.comb(n, k) for k in range(0, min(pos, neg) + 1)) / 2 ** n
    return min(1.0, 2 * tail)


def compare(run_dir: Path, arms=None, reference: str = "shape") -> dict:
    """Readouts of `arms` (default: every arm with at least one record, a set arm only when it has a dir) on their
    common images; the paired AP50 deltas are `reference` minus each other arm."""
    arms = arms or arm_lib.ARMS + tuple(a for a in arm_lib.SET_ARMS if (run_dir / a).is_dir())
    by_arm = {a: {b: {r["image"]: r for r in load_records(run_dir / a / b / "records.jsonl")}
                  for b in BANDS} for a in arms}
    arms_present = [a for a in by_arm if any(by_arm[a][b] for b in BANDS)]
    result = {"arms": arms_present, "arms_without_records": [a for a in by_arm if a not in arms_present],
              "bands": {}, "reference": reference, "paired_vs_shape": {}, "excluded": {}}
    by_arm = {a: by_arm[a] for a in arms_present}
    scopes = list(BANDS) + ["all"]
    common = {}
    for band in BANDS:
        sets = [set(by_arm[a][band]) for a in arms_present]
        common[band] = sorted(set.intersection(*sets)) if sets else []
        union = set.union(*sets) if sets else set()
        result["excluded"][band] = sorted(union - set(common[band]))
    common["all"] = [(b, i) for b in BANDS for i in common[b]]

    def recs(arm, scope):
        if scope == "all":
            return [by_arm[arm][b][i] for b, i in common["all"]]
        return [by_arm[arm][scope][i] for i in common[scope]]

    for scope in scopes:
        base = ({r["image"]: r["ap50"] for r in recs("cellpose_sam", scope)}
                if "cellpose_sam" in arms_present else {})
        result["bands"][scope] = {a: readouts(recs(a, scope), base) for a in arms_present}
        if reference in arms_present:
            shape_ap = [r["ap50"] for r in recs(reference, scope)]
            for a in arms_present:
                if a == reference:
                    continue
                deltas = [s - r["ap50"] for s, r in zip(shape_ap, recs(a, scope))]
                entry = {"n": len(deltas), "mean_delta_ap50": _mean(deltas),
                         "ci_95": bootstrap_ci(deltas) if len(deltas) >= 2 else None,
                         "sign_test_p": sign_test_p(deltas) if deltas else None}
                result["paired_vs_shape"].setdefault(scope, {})[a] = entry
    result["_records"] = {a: {s: recs(a, s) for s in scopes} for a in arms_present}
    return result


def write_compare(run_dir: Path, result: dict) -> Path:
    out = run_dir / "compare"
    out.mkdir(parents=True, exist_ok=True)
    keys = [k for k, _ in READOUTS]
    with open(out / "readouts.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["band", "arm", "n", "n_errors"] + keys)
        for scope, per_arm in result["bands"].items():
            for a, ro in per_arm.items():
                w.writerow([scope, a, ro["n"], ro["n_errors"]] + [ro[k] for k in keys])
    with open(out / "per_image.csv", "w", newline="", encoding="utf-8") as h:
        cols = ["band", "image", "arm", "ap50", "map_50_95", "seconds", "tool_calls", "tokens",
                "predicted_cell_count", "ground_truth_cell_count", "error"]
        w = csv.DictWriter(h, fieldnames=cols, extrasaction="ignore")
        w.writeheader()
        for a, by_scope in result["_records"].items():
            for r in by_scope["all"]:
                w.writerow({**r, "arm": a})
    with open(out / "paired_vs_shape.csv", "w", newline="", encoding="utf-8") as h:
        w = csv.writer(h)
        w.writerow(["band", "arm", "n", f"mean_delta_ap50 ({result['reference']} - arm)", "ci_low", "ci_high",
                    "sign_test_p"])
        for scope, per_arm in result["paired_vs_shape"].items():
            for a, e in per_arm.items():
                ci = e["ci_95"] or (None, None)
                w.writerow([scope, a, e["n"], e["mean_delta_ap50"], ci[0], ci[1], e["sign_test_p"]])
    (out / "summary.md").write_text(_markdown(result), encoding="utf-8")
    dump = {k: v for k, v in result.items() if not k.startswith("_")}
    (out / "compare.json").write_text(json.dumps(dump, indent=1), encoding="utf-8")
    try:
        make_figure(result, out / "readouts.png")
    except Exception as exc:
        print(f"[compare] figure failed: {exc}")
    return out


def _fmt(v, nd=3):
    return "n/a" if v is None else f"{v:.{nd}f}"


def _markdown(result: dict) -> str:
    lines = ["# NeurIPS22-CellSeg: readouts per arm", ""]
    for scope, per_arm in result["bands"].items():
        n = next(iter(per_arm.values()))["n"] if per_arm else 0
        lines += [f"## {scope} (n = {n} images every arm attempted)", "",
                  "| arm | " + " | ".join(lbl for _, lbl in READOUTS) + " | errors |",
                  "|---|" + "---|" * (len(READOUTS) + 1)]
        for a, ro in per_arm.items():
            lines.append(f"| {a} | " + " | ".join(
                _fmt(ro[k], 0 if k == "tokens_per_image" else 1 if k == "seconds_per_image" else 3)
                for k, _ in READOUTS) + f" | {ro['n_errors']} |")
        paired = result["paired_vs_shape"].get(scope)
        if paired:
            lines += ["", f"{result['reference']} minus arm, AP50, paired per image (bootstrap 95% CI; sign test):",
                      ""]
            for a, e in paired.items():
                ci = e["ci_95"]
                ci_txt = f"[{ci[0]:+.3f}, {ci[1]:+.3f}]" if ci else "n/a"
                lines.append(f"- vs {a}: {_fmt(e['mean_delta_ap50'])} {ci_txt}, "
                             f"p = {_fmt(e['sign_test_p'])} (n = {e['n']})")
        lines.append("")
    if result["arms_without_records"]:
        lines += ["Arms without any record (left out): " + ", ".join(result["arms_without_records"]), ""]
    excluded = {b: v for b, v in result["excluded"].items() if v}
    if excluded:
        lines += ["Images not attempted by every arm (left out of the readouts): "
                  + "; ".join(f"{b}: {len(v)}" for b, v in excluded.items()), ""]
    return "\n".join(lines)


def _nature_style() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({
        "font.family": "sans-serif", "font.sans-serif": ["Arial", "Helvetica", "DejaVu Sans"],
        "font.size": 7, "axes.titlesize": 8, "axes.labelsize": 7, "xtick.labelsize": 6,
        "ytick.labelsize": 6, "legend.fontsize": 6, "axes.linewidth": 0.5,
        "xtick.major.width": 0.5, "ytick.major.width": 0.5, "lines.linewidth": 1.0,
        "axes.spines.top": False, "axes.spines.right": False, "savefig.dpi": 300, "figure.dpi": 300})


def make_figure(result: dict, out_png: Path) -> None:
    """One panel per readout: bars per arm, grouped by band (+ all); dots = images."""
    _nature_style()
    import matplotlib.pyplot as plt
    scopes = list(result["bands"])
    arms_present = result["arms"]
    per_image = {"ap50": "ap50", "map_50_95": "map_50_95", "seconds_per_image": "seconds",
                 "tool_calls_per_image": "tool_calls", "tokens_per_image": "tokens"}
    fig, axes = plt.subplots(2, 3, figsize=(7.2, 4.6))
    width = 0.8 / max(1, len(arms_present))
    rng = np.random.default_rng(0)
    for ax, (key, label) in zip(axes.ravel(), READOUTS):
        for j, a in enumerate(arms_present):
            xs = np.arange(len(scopes)) + (j - (len(arms_present) - 1) / 2) * width
            vals = [result["bands"][s][a][key] for s in scopes]
            ax.bar(xs, [0 if v is None else v for v in vals], width * 0.9, color=ARM_COLORS[a],
                   label=a, linewidth=0)
            if key in per_image:
                for x, s in zip(xs, scopes):
                    ys = [r.get(per_image[key]) for r in result["_records"][a][s]]
                    ys = [y for y in ys if y is not None]
                    ax.scatter(x + (rng.random(len(ys)) - 0.5) * width * 0.6, ys, s=1.5,
                               color="black", alpha=0.35, linewidths=0, zorder=3)
        ax.set_xticks(np.arange(len(scopes)))
        ax.set_xticklabels(scopes)
        ax.set_title(label)
        if key == "seconds_per_image" and all(
                (result["bands"][s][a][key] or 0) > 0 for s in scopes for a in arms_present):
            ax.set_yscale("log")
        if key in ("ap50", "map_50_95"):
            ax.set_ylim(0, 1.05)
        if key == "ap50_gain_per_tool_call":
            ax.axhline(0, color="black", linewidth=0.5)
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, frameon=False, ncol=len(labels), loc="upper center",
               bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    fig.savefig(out_png, dpi=300, bbox_inches="tight")
    plt.close(fig)


# --------------------------------------------------------------------------- #
# CLI
# --------------------------------------------------------------------------- #
def main() -> None:
    from cellseg_config import DEFAULT as cfg
    default_data = (Path(os.environ["SHAPE_BENCH_DATA_ROOT"]) / "NeurIPS22-CellSeg"
                    if os.environ.get("SHAPE_BENCH_DATA_ROOT") else None)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_lists = sub.add_parser("lists", help="draw the per-band image lists (once per run dir)")
    p_lists.add_argument("--run-dir", type=Path, required=True)
    p_lists.add_argument("--n-per-band", type=int, default=cfg.n_per_band)
    p_lists.add_argument("--seed", type=int, default=cfg.seed)
    p_lists.add_argument("--largest", type=int,
                         help="a stress test: only the N largest images (pixels), each in its own band")
    p_lists.add_argument("--data-root", type=Path, default=default_data,
                         help="the images to measure for --largest")
    p_lists.add_argument("--exclude", type=Path, nargs="+", default=[],
                         help="list files (lists/<band>.txt of earlier runs) whose images are not drawn")
    p_run = sub.add_parser("run", help="run one arm on the run dir's lists")
    p_run.add_argument("--run-dir", type=Path, required=True)
    p_run.add_argument("--arm", choices=arm_lib.ARMS, required=True)
    p_run.add_argument("--data-root", type=Path, default=default_data, required=default_data is None)
    p_run.add_argument("--bands", nargs="+", choices=BANDS, default=list(BANDS))
    p_run.add_argument("--limit", type=int, help="first N images of each band list (a smoke run)")
    p_run.add_argument("--model", default=cfg.llm_model)
    p_run.add_argument("--max-steps", type=int, default=cfg.agent_max_steps)
    p_run.add_argument("--workers", type=int, default=IMAGE_WORKERS,
                       help="react: images (ReAct sessions) at a time")
    p_set = sub.add_parser("run-set", help="run a set arm (one solve() over the lists' images)")
    p_set.add_argument("--run-dir", type=Path, required=True)
    p_set.add_argument("--arm", choices=arm_lib.SET_ARMS, required=True)
    p_set.add_argument("--data-root", type=Path, default=default_data, required=default_data is None)
    p_set.add_argument("--bands", nargs="+", choices=BANDS, default=list(BANDS))
    p_set.add_argument("--limit", type=int, help="first N images of each band list (a smoke run)")
    p_set.add_argument("--model", default=cfg.llm_model)
    p_set.add_argument("--max-steps", type=int, default=cfg.agent_max_steps)
    p_set.add_argument("--workers", type=int, default=IMAGE_WORKERS,
                       help="images at a time (solve()'s image_workers)")
    p_set.add_argument("--preflight-cache", type=Path,
                       help="an earlier run's set/preflight_cache.json: its preflight reads are reused")
    p_cmp = sub.add_parser("compare", help="the six readouts across the arms of a run dir")
    p_cmp.add_argument("--run-dir", type=Path, required=True)
    p_cmp.add_argument("--arms", nargs="+", choices=arm_lib.ARMS + arm_lib.SET_ARMS,
                       help="compare only these arms (default: every arm with a record)")
    p_cmp.add_argument("--reference", default="shape", choices=arm_lib.ARMS + arm_lib.SET_ARMS,
                       help="the arm the paired AP50 deltas are taken from (reference minus arm)")
    p_smoke = sub.add_parser("smoke", help="build one arm and run it on one image (no GT)")
    p_smoke.add_argument("--arm", choices=arm_lib.ARMS, required=True)
    p_smoke.add_argument("--image", type=Path, required=True)
    p_smoke.add_argument("--model", default=cfg.llm_model)
    p_smoke.add_argument("--max-steps", type=int, default=cfg.agent_max_steps)
    args = ap.parse_args()

    if args.cmd == "lists":
        largest = None
        if args.largest:
            if args.data_root is None:
                ap.error("--largest needs --data-root (or SHAPE_BENCH_DATA_ROOT)")
            largest = largest_images(args.data_root, args.largest)
            for stem, px in largest.items():
                print(f"[lists] largest: {stem} {px / 1e6:.1f} Mpx")
        exclude = {s for f in args.exclude for s in read_list(f)}
        if args.exclude:
            print(f"[lists] excluded: {len(exclude)} images from {len(args.exclude)} list file(s)")
        write_lists(args.run_dir, args.n_per_band, args.seed, csv_path=band_metric_csv(args.data_root),
                    largest=largest, exclude=exclude)
    elif args.cmd == "run":
        run_arm(args.run_dir, args.arm, args.data_root, bands=args.bands, limit=args.limit,
                model=args.model, max_steps=args.max_steps, workers=args.workers)
    elif args.cmd == "run-set":
        run_set(args.run_dir, args.arm, args.data_root, bands=args.bands, limit=args.limit, model=args.model,
                max_steps=args.max_steps, workers=args.workers, preflight_cache=args.preflight_cache)
    elif args.cmd == "compare":
        result = compare(args.run_dir, args.arms, args.reference)
        out = write_compare(args.run_dir, result)
        print(_markdown(result))
        print(f"[compare] -> {out}")
    elif args.cmd == "smoke":
        arm = arm_lib.make_arm(args.arm, model=args.model, max_steps=args.max_steps)
        with tempfile.TemporaryDirectory() as tmp:
            t0 = time.perf_counter()
            res = arm(args.image, Path(tmp))
            mask = res.get("mask")
            cells = int(len(np.unique(mask)) - (0 in mask)) if mask is not None else None
            status = "SMOKE_OK" if mask is not None else "SMOKE_FAIL"
            print(f"{status} {args.arm} cells={cells} tool_calls={res.get('tool_calls')} "
                  f"seconds={time.perf_counter() - t0:.1f}")
        sys.exit(0 if mask is not None else 1)


if __name__ == "__main__":
    main()

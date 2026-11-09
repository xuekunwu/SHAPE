"""Formal arms for the NeurIPS22-CellSeg cropper benchmark.

The benchmark has five public arms:

1. maskcrop      deterministic MaskCrop domain tool
2. cellprofiler  native CellProfiler SaveCroppedObjects module (a control: its scored crops share
                 MaskCrop's square geometry)
3. gpt           gpt-5.6-luna alone, producing crop boxes directly (one shot)
4. react         gpt-5.6-luna plus MaskCrop, on the shared ReAct runner (benchmarks/_shared/react_arm.py)
5. shape         SHAPE crop verifier loop with MaskCrop

Each arm returns crop boxes in the same schema so the runner can compute the registered
readouts without arm-specific scoring code.
"""

from __future__ import annotations

import inspect
import json
import os
import re
import shlex
import subprocess
import time
from dataclasses import dataclass
from pathlib import Path
from threading import Lock
from typing import Any

import numpy as np
import tifffile
from skimage.measure import regionprops

import cellprofiler_native as cp_native
import run_shape_crop as rsc
from benchmarks._shared import react_arm
from shape.agent.verification import base as vrr_base
from shape.agent.verification.verifiers.crop import (
    PROMPT_IMAGE_PX,
    THREADS_PER_CALL,
    CELL_KEYS,
    CropAgenticVerifier,
    CropTask,
    _input_overlay,
    input_record,
)
from shape.agent.verification.vision_engine import build_vision_engine
from shape.tools.single_cell_cropper.tool import (
    Single_Cell_Cropper_Tool,
    compute_square_crop_containing_bbox,
    effective_margin,
)

ARMS = ("maskcrop", "cellprofiler", "gpt", "react", "shape")
LLM_ARMS = ("gpt", "react", "shape")
DEFAULT_MODEL = "gpt-5.6-luna"
GPT_BATCH_SIZE = 64
GPT_MAX_COMPLETION_TOKENS = 8000
MAX_ROUNDS = vrr_base.BUDGET           # SHAPE's round budget (8), ReAct's too: at most 1 + (MAX_ROUNDS - 1) x cap calls
LLM_ATTEMPTS = 3                       # tries of one GPT question before the image fails
LLM_RETRY_SECONDS = 30                 # wait after a gateway error before the next try (SHAPE's retry wait)
CELLPROFILER_HELPER = Path(__file__).with_name("cellprofiler_native.py")
CELLPROFILER_NORMALIZED_SOURCE = (
    "cellprofiler.SaveCroppedObjects+maskcrop_square_normalization"
)


@dataclass
class ArmResult:
    crops: list[dict[str, Any]]
    tool_calls: int
    trace: dict[str, Any]


def labels_in_tool_order(mask: np.ndarray, min_area: int = 0) -> list[int]:
    """Labels in the same regionprops order used by Single_Cell_Cropper_Tool."""
    labels = []
    for region in regionprops(mask.astype(np.int64)):
        if int(region.area) >= int(min_area):
            labels.append(int(region.label))
    return labels


def maskcrop_square_bboxes(
        mask: np.ndarray, min_area: int, margin: int, margin_frac: float) -> dict[int, list[int]]:
    """The formal Maskcrop fraction-mode geometry, keyed by target label."""
    image_height, image_width = mask.shape[:2]
    bboxes = {}
    for region in regionprops(mask.astype(np.int64)):
        if int(region.area) < int(min_area):
            continue
        minr, minc, maxr, maxc = (int(value) for value in region.bbox)
        height, width = maxr - minr, maxc - minc
        pad = effective_margin(
            margin, region.area, margin_frac=margin_frac, margin_mode="fraction", tight_buffer_px=2)
        half_side = int(np.ceil(max(height, width) / 2.0)) + int(pad)
        center_row = (minr + maxr) // 2
        center_col = (minc + maxc) // 2
        bboxes[int(region.label)] = list(compute_square_crop_containing_bbox(
            minr, minc, maxr, maxc, center_row, center_col, half_side,
            image_height, image_width,
        ))
    return bboxes


def normalize_cellprofiler_crops(
        image_path: Path, mask: np.ndarray, native_crops: list[dict], out_dir: Path,
        min_area: int, margin: int, margin_frac: float) -> list[dict]:
    """Convert native tight rectangles to the formal Maskcrop square geometry. `image_path` may be the
    host-decoded `.npy` the native helper read."""
    native_by_label = {int(crop["cell_label"]): crop for crop in native_crops}
    square_bboxes = maskcrop_square_bboxes(mask, min_area, margin, margin_frac)
    if set(native_by_label) != set(square_bboxes):
        missing = sorted(set(square_bboxes) - set(native_by_label))
        extra = sorted(set(native_by_label) - set(square_bboxes))
        raise RuntimeError(
            f"CellProfiler native target mismatch before square normalization: "
            f"missing={missing[:20]} extra={extra[:20]}"
        )

    source_image = cp_native._orient_source_image(
        cp_native._read_source_image(image_path), mask.shape[:2])
    square_dir = out_dir / "square_crops"
    square_dir.mkdir(parents=True, exist_ok=True)
    normalized = []
    for label, bbox in square_bboxes.items():
        minr, minc, maxr, maxc = bbox
        if maxr - minr != maxc - minc:
            raise RuntimeError(f"square normalization produced non-square bbox for label {label}: {bbox}")
        crop_path = square_dir / f"Cells_{label}.tiff"
        tifffile.imwrite(str(crop_path), source_image[minr:maxr, minc:maxc])
        written_shape = tifffile.imread(str(crop_path)).shape
        expected_shape = (maxr - minr, maxc - minc)
        if tuple(written_shape[:2]) != expected_shape:
            raise RuntimeError(
                f"normalized CellProfiler crop has shape {written_shape}, expected {expected_shape}"
            )
        native = native_by_label[label]
        normalized.append({
            "cell_label": label,
            "bbox": bbox,
            "source": CELLPROFILER_NORMALIZED_SOURCE,
            "crop_image_path": str(crop_path),
            "native_bbox": list(native["bbox"]),
            "native_crop_image_path": native.get("crop_image_path"),
        })
    return normalized


def _load_metadata(result: dict) -> list[dict[str, Any]]:
    path = result.get("cell_crops_metadata_path")
    if not path or not Path(path).exists():
        return []
    payload = json.loads(Path(path).read_text(encoding="utf-8"))
    return list(payload.get("cell_metadata", []) or [])


def _match_metadata(result: dict, mask: np.ndarray, min_area: int) -> tuple[dict[int, dict], list[dict]]:
    """The tool's crop metadata keyed by target label, matched by `original_bbox` (the target's own
    regionprops bbox), and the entries that match no target (or whose bbox two targets share)."""
    by_bbox: dict[tuple[int, ...], list[int]] = {}
    for region in regionprops(mask.astype(np.int64)):
        if int(region.area) >= int(min_area):
            by_bbox.setdefault(tuple(int(v) for v in region.bbox), []).append(int(region.label))
    matched: dict[int, dict] = {}
    unmatched: list[dict] = []
    for meta in _load_metadata(result):
        labels = by_bbox.get(tuple(int(v) for v in meta.get("original_bbox") or []), [])
        if len(labels) != 1 or labels[0] in matched or len(meta.get("crop_bbox") or []) != 4:
            unmatched.append(meta)
            continue
        matched[labels[0]] = meta
    return matched, unmatched


def _metadata_to_crops(result: dict, mask: np.ndarray, min_area: int) -> list[dict[str, Any]]:
    """The delivered crops, one per target label (matched by `original_bbox`); raises when the metadata
    and the eligible targets do not match one to one."""
    expected = labels_in_tool_order(mask, min_area=min_area)
    matched, unmatched = _match_metadata(result, mask, min_area)
    if unmatched or len(matched) != len(expected):
        missing = sorted(set(expected) - set(matched))
        raise ValueError(
            f"crop metadata does not match the targets one to one: {len(matched)} matched of "
            f"{len(expected)} targets, missing={missing[:20]}, unmatched_entries={len(unmatched)}")
    return [{"cell_label": label,
             "bbox": [int(v) for v in matched[label]["crop_bbox"]],
             "source": "maskcrop_metadata",
             "crop_image_path": matched[label].get("crop_path"),
             "metadata": matched[label]} for label in expected]


def _response_content(reply: Any) -> Any:
    """Unwrap SHAPE engine responses while preserving direct stub responses."""
    if not isinstance(reply, dict):
        return reply
    if "content" in reply:
        return reply["content"]
    if reply.get("error"):
        raise RuntimeError(
            f"model generation failed: {reply.get('error')}: {reply.get('message', '')}".strip()
        )
    return reply


def _parse_json_list(reply: Any) -> list[dict[str, Any]]:
    text = _response_content(reply)
    if isinstance(text, list):
        obj = text
    elif isinstance(text, dict):
        obj = text
    else:
        text = str(text or "")
        try:
            obj = json.loads(text)
        except Exception:
            match = re.search(r"\[[\s\S]*\]", text)
            obj = json.loads(match.group(0)) if match else []
    if isinstance(obj, dict):
        obj = obj.get("crops", [])
    if not isinstance(obj, list):
        return []
    crops = []
    for item in obj:
        if not isinstance(item, dict) or "cell_label" not in item or "bbox" not in item:
            continue
        bbox = item["bbox"]
        if isinstance(bbox, list) and len(bbox) == 4:
            crops.append({"cell_label": int(item["cell_label"]),
                          "bbox": [int(round(float(v))) for v in bbox],
                          "source": "llm"})
    return crops


def _with_images(text: str, images: list) -> Any:
    """A text prompt, or the text followed by its labelled images when there are any."""
    return [text, *images] if images else text


def _overlay_images(image_path: Path, mask_path: Path) -> list:
    """SHAPE's input image (the mask's object boundaries in green, longest side PROMPT_IMAGE_PX), as SHAPE's
    prompts label it; [] when it cannot be drawn."""
    png = _input_overlay(str(image_path), str(mask_path), PROMPT_IMAGE_PX)
    return ["INPUT IMAGE (the mask's object boundaries in green):", png] if png else []


def _mask_summary(mask: np.ndarray, min_area: int = 0) -> str:
    rows = []
    for region in regionprops(mask.astype(np.int64)):
        if int(region.area) < min_area:
            continue
        minr, minc, maxr, maxc = region.bbox
        rows.append({"cell_label": int(region.label), "bbox": [minr, minc, maxr, maxc],
                     "area": int(region.area)})
    return json.dumps(rows, separators=(",", ":"))


def validate_crops(arm: str, crops: list[dict], expected_labels: list[int],
                   mask_shape: tuple[int, int]) -> None:
    """All arms must return one in-bounds square crop for every eligible target."""
    actual_labels = []
    height, width = mask_shape
    for crop in crops:
        if "cell_label" not in crop or "bbox" not in crop:
            raise ValueError(f"{arm} emitted a crop without cell_label/bbox: {crop}")
        label = int(crop["cell_label"])
        bbox = [int(round(float(value))) for value in crop["bbox"]]
        if len(bbox) != 4:
            raise ValueError(f"{arm} emitted invalid bbox for label {label}: {bbox}")
        minr, minc, maxr, maxc = bbox
        if not (0 <= minr < maxr <= height and 0 <= minc < maxc <= width):
            raise ValueError(f"{arm} emitted out-of-bounds bbox for label {label}: {bbox}, mask_shape={mask_shape}")
        if maxr - minr != maxc - minc:
            raise ValueError(f"{arm} emitted non-square bbox for label {label}: {bbox}")
        actual_labels.append(label)
    if len(actual_labels) != len(set(actual_labels)):
        raise ValueError(f"{arm} emitted duplicate cell labels")
    if set(actual_labels) != set(expected_labels):
        missing = sorted(set(expected_labels) - set(actual_labels))
        extra = sorted(set(actual_labels) - set(expected_labels))
        raise ValueError(f"{arm} target-cell contract failed: missing={missing[:20]} extra={extra[:20]}")


def _engine(model: str):
    engine = build_vision_engine(model)
    if engine is None:
        raise RuntimeError(f"no vision engine for {model}; source LLM_API_KEY/LLM_BASE_URL")
    return engine


class MaskCrop:
    def __init__(self, min_area: int, margin: int, margin_frac: float, **_: Any):
        self.min_area = min_area
        self.margin = margin
        self.margin_frac = margin_frac
        self.tool = Single_Cell_Cropper_Tool()

    def __call__(self, image_path: Path, mask_path: Path, mask: np.ndarray, out_dir: Path) -> ArmResult:
        result = rsc.run_maskcrop_arm(
            self.tool, image_path, mask_path, out_dir, self.min_area, self.margin, self.margin_frac)
        rsc._organize_kept_rejected_crops(out_dir, result)
        return ArmResult(_metadata_to_crops(result, mask, self.min_area), 1, {"result": result})


class CellProfiler:
    """Native SaveCroppedObjects plus the formal Maskcrop square-output adapter."""

    def __init__(self, min_area: int, margin: int = 1, margin_frac: float = 0.20,
                 cellprofiler_python: str | None = None, **_: Any):
        self.min_area = min_area
        self.margin = margin
        self.margin_frac = margin_frac
        self.cellprofiler_python = cellprofiler_python or os.environ.get("CELLPROFILER_PYTHON")

    def __call__(self, image_path: Path, mask_path: Path, mask: np.ndarray, out_dir: Path) -> ArmResult:
        if not self.cellprofiler_python:
            raise RuntimeError(
                "cellprofiler arm requires --cellprofiler-python or CELLPROFILER_PYTHON; "
                "point it to a Python executable/wrapper with CellProfiler installed")

        out_dir.mkdir(parents=True, exist_ok=True)
        result_path = out_dir / "cellprofiler_result.json"
        if result_path.exists():
            result_path.unlink()
        # the source image is decoded here, in the benchmark's environment (the reader every other arm's
        # image passes through), and handed to the CellProfiler image uncompressed: its tifffile has no
        # `imagecodecs` and fails on e.g. LZW-compressed TIFFs
        decoded_path = out_dir / "source_image.npy"
        np.save(decoded_path, cp_native._read_source_image(image_path), allow_pickle=False)
        try:
            command = [
                *shlex.split(self.cellprofiler_python),
                str(CELLPROFILER_HELPER),
                "--image-path", str(decoded_path),
                "--mask-path", str(mask_path),
                "--out-dir", str(out_dir),
                "--min-area", str(self.min_area),
                "--result-json", str(result_path),
            ]
            completed = subprocess.run(command, capture_output=True, text=True, check=False)
            if completed.returncode != 0:
                detail = (completed.stderr or completed.stdout or "no subprocess output").strip()
                raise RuntimeError(
                    f"native CellProfiler failed with exit code {completed.returncode}: {detail[-2000:]}")
            if not result_path.is_file():
                raise RuntimeError(f"native CellProfiler did not write {result_path}")

            payload = json.loads(result_path.read_text(encoding="utf-8"))
            if payload.get("backend") != "CellProfiler" or payload.get("module") != "SaveCroppedObjects":
                raise RuntimeError(f"unexpected CellProfiler result metadata: {payload}")
            native_crops = list(payload.get("crops") or [])
            crops = normalize_cellprofiler_crops(
                decoded_path, mask, native_crops, out_dir,
                self.min_area, self.margin, self.margin_frac)
        finally:
            decoded_path.unlink(missing_ok=True)
        return ArmResult(crops, 1, {
            "backend": payload["backend"],
            "module": payload["module"],
            "cellprofiler_version": payload.get("cellprofiler_version"),
            "cellprofiler_core_version": payload.get("cellprofiler_core_version"),
            "native_crop_count": len(native_crops),
            "square_normalization": {
                "geometry": "maskcrop_fraction_square",
                "margin": self.margin,
                "margin_frac": self.margin_frac,
                "recenter": False,
            },
            "stdout": completed.stdout.strip(),
        })


class GPT:
    """The model alone, one shot (no feedback from any tool): it sees the input image with the mask's
    object boundaries (SHAPE's input overlay) and every object of the mask with its bbox and area, and
    returns one square box per target. Targets are requested in batches of `batch_size` (the reply
    length), each prompt carrying all the objects; an invalid batch is asked again with the error, up to
    LLM_ATTEMPTS times. Every request is recorded in `trace["calls"]` (its prompt, image refs, parameters,
    reply as returned, usage, error and seconds; on a failed image the error carries the trace)."""

    def __init__(self, model: str, min_area: int = 0, batch_size: int = GPT_BATCH_SIZE, **_: Any):
        self.model = model
        self.engine = _engine(model)
        self.min_area = min_area
        if batch_size < 1:
            raise ValueError("GPT batch_size must be positive")
        self.batch_size = batch_size

    def __call__(self, image_path: Path, mask_path: Path, mask: np.ndarray, out_dir: Path) -> ArmResult:
        targets = labels_in_tool_order(mask, min_area=self.min_area)
        objects = _mask_summary(mask)
        images = _overlay_images(image_path, mask_path)
        crops = []
        batches = []
        calls = []
        params = {"model": self.model, "max_tokens": GPT_MAX_COMPLETION_TOKENS, "response_format": None}
        for offset in range(0, len(targets), self.batch_size):
            expected = targets[offset:offset + self.batch_size]
            prompt = (
                "Return only a JSON list with exactly one {cell_label:int,bbox:[minr,minc,maxr,maxc]} "
                "for EACH target cell_label below. Use half-open pixel coordinates and square boxes within "
                "the image. Preserve the entire target while minimizing neighboring cells. "
                f"Image shape (height,width): {mask.shape[:2]}. "
                f"All objects of the mask (cell_label, bbox, area): {objects}\n"
                f"Targets: {json.dumps(expected, separators=(',', ':'))}")
            for attempt in range(1, LLM_ATTEMPTS + 1):
                content = _with_images(prompt, images)
                t0 = time.perf_counter()
                reply = self.engine.generate(content, max_tokens=GPT_MAX_COMPLETION_TOKENS)
                calls.append({"batch_index": offset // self.batch_size, "attempt": attempt, "targets": expected,
                              "prompt": prompt, "images": vrr_base.image_refs(content), "params": params,
                              "reply": vrr_base.reply_record(reply.get("content", reply) if isinstance(reply, dict)
                                                             else reply),
                              "usage": reply.get("usage") if isinstance(reply, dict) else None, "error": None,
                              "seconds": round(time.perf_counter() - t0, 2)})
                try:
                    batch_crops = _parse_json_list(reply)
                    validate_crops("gpt", batch_crops, expected, mask.shape[:2])
                    break
                except (ValueError, TypeError, KeyError, RuntimeError) as exc:
                    calls[-1]["error"] = str(exc)
                    if attempt == LLM_ATTEMPTS:
                        failure = ValueError(
                            f"GPT batch {offset // self.batch_size + 1} failed after {LLM_ATTEMPTS} "
                            f"attempts: {exc}")
                        failure.trace = {"model": self.model, "batches": batches, "calls": calls,
                                         "image_attached": bool(images)}
                        raise failure from exc
                    if isinstance(exc, RuntimeError):       # a gateway error: wait, then ask the same again
                        time.sleep(LLM_RETRY_SECONDS)
                        continue
                    prompt += f"\nYour previous answer was invalid: {exc}. Return the complete corrected list."
            crops.extend(batch_crops)
            batches.append({"index": offset // self.batch_size, "n_targets": len(expected),
                            "attempts": attempt})
        return ArmResult(crops, 0, {"model": self.model, "batches": batches, "calls": calls,
                                    "image_attached": bool(images)})


# ---- react: an LLM plus the tool, on the shared runner (benchmarks/_shared/REACT_ARM_CONTRACT.md) ----------

REACT_TOOL_NAME = "Single_Cell_Cropper_Tool"
CELL_OVERRIDES = "cell_overrides"
# the per-cell keys of the tool's `cell_overrides` that SHAPE's cell rules set (`crop.CELL_KEYS`; its "reset" is a
# rule directive that removes a cell's entry, not a key the tool reads)
CELL_OVERRIDE_KEYS = {key: spec for key, spec in CELL_KEYS.items() if key != "reset"}


def _type_text(kind: str, bounds: Any = None, nullable: bool = False) -> str:
    """A parameter's type as the ReAct table shows it: the kind; a choice with its choices (no ranges)."""
    text = kind
    if kind == "choice" and bounds:
        text = "choice of " + ", ".join(json.dumps(choice) for choice in bounds if choice is not None)
    return text + (" or null" if nullable else "")


def react_input_text(image_path: Path, mask_path: Path, mask: np.ndarray, min_area: int) -> str:
    """The input as SHAPE is given it (the image, the given mask, the fixed min_area), with nothing SHAPE computes
    from them."""
    height, width = (int(v) for v in mask.shape[:2])
    return (f"Image: {Path(image_path).name}, {height} x {width} pixels (height x width).\n"
            f"Given segmentation mask: {Path(mask_path).name}, a label image with one integer label per object.\n"
            f"min_area: {int(min_area)} (fixed, not a parameter).")


class CropToolAdapter:
    """The cropper on one image for the shared ReAct runner (`react_arm.ToolAdapter`), called exactly as SHAPE's
    arm calls it (`run_shape_crop.run_shape_agentic_arm`: the image, the given mask, `query_cache_dir` = the
    image's dir; `trial_params`: the levers, `cell_overrides`, the fixed `min_area` and the call's own
    `output_subdir`). A call fails as SHAPE's tool runner fails it (a tool error, or no crops).

    Parameters: SHAPE's crop verifier levers (`CropAgenticVerifier.levers`: name, type and the default of a move
    from "input", resolved for this mask as SHAPE resolves it) and, as one parameter, `cell_overrides` with the
    per-cell keys SHAPE's cell rules compile into. The observation is the tool's returned dict without its crop
    objects, and the crop tiles the tool wrote (the first `react_arm.MAX_IMAGES` in the tool's order).
    `python_inputs`: the raw image, the given mask and each finished call's metadata json and crop files; never a
    file under `private_roots` (the scoring ground truth beyond the given mask)."""

    tool_name = REACT_TOOL_NAME
    unit_kind = "image"

    def __init__(self, tool, image_path: Path, mask_path: Path, out_dir: Path, verifier: CropAgenticVerifier,
                 private_roots=()):
        self.tool, self.image_path, self.mask_path, self.out_dir = tool, Path(image_path), Path(mask_path), out_dir
        self.verifier = verifier
        self.private_roots = [Path(r) for r in private_roots]
        self.unit = self.image_path.stem
        self.min_area = int(verifier.input.get("min_area") or 0)
        self.results: dict[str, dict] = {}
        self.delivered: str | None = None
        self.lock = Lock()

    def parameters(self) -> list:
        defaults = self.verifier.defaults()
        params = []
        for lever in self.verifier.levers:
            params.append(react_arm.Param(lever, _type_text(*self.verifier.lever_spec(lever)), defaults[lever]))
        keys = ", ".join(f"{key}: {_type_text(kind, bounds)}" for key, (kind, bounds) in CELL_OVERRIDE_KEYS.items())
        cell_id = f"default_{self.unit}_cell_<NNNN>"     # the tool's cell_id: your-group_<image stem>_cell_<index>
        params.append(react_arm.Param(
            CELL_OVERRIDES, f'object: cell id -> {{{keys}}}; cell id "{cell_id}", NNNN as in the crop file names',
            {}))
        return params

    def units(self) -> list[str]:
        return [self.unit]

    def round_cap(self) -> int:
        return self.verifier.concurrency()            # SHAPE's per-round cap: cores // THREADS_PER_CALL

    def default_counts_as_round(self) -> bool:
        return bool(self.verifier.admission_call)     # crop: round 1 is the tool at its defaults

    def call(self, unit: str, params: dict, call_id: str) -> dict:
        result = self.tool.execute(original_image=str(self.image_path), nuclei_mask=str(self.mask_path),
                                   query_cache_dir=str(self.out_dir), **params, min_area=self.min_area,
                                   output_subdir=f"crop_calls/{call_id}")
        error = None
        if result.get("error") or not result.get("cell_count"):
            error = str(result.get("error") or "the tool produced no crops")
        with self.lock:
            self.results[call_id] = result
        output = {key: value for key, value in result.items() if key != "cell_crop_objects"}
        return {"output": output, "images": self._crop_tiles(result), "error": error}

    @staticmethod
    def _crop_tiles(result: dict) -> list[str]:
        """The crop files the call wrote, as the tool lists them in its metadata file."""
        path = result.get("cell_crops_metadata_path")
        if not path or not Path(path).is_file():
            return []
        listed = json.loads(Path(path).read_text(encoding="utf-8")).get("cell_crops_paths") or []
        return [str(p) for p in listed if Path(p).is_file()][:react_arm.MAX_IMAGES]

    def deliver(self, unit: str, call_id: str) -> None:
        self.delivered = call_id

    def python_inputs(self, unit: str) -> dict:
        """The files a python step may read: inputs/<the raw image>, inputs/<the given mask>, calls/<call id>/<the
        call's metadata json and each crop file it lists>."""
        paths = {f"inputs/{self.image_path.name}": self.image_path, f"inputs/{self.mask_path.name}": self.mask_path}
        with self.lock:
            results = dict(self.results)
        for call_id, result in results.items():
            meta = result.get("cell_crops_metadata_path")
            if result.get("error") or not meta or not Path(meta).is_file():
                continue
            paths[f"calls/{call_id}/{Path(meta).name}"] = Path(meta)
            listed = json.loads(Path(meta).read_text(encoding="utf-8")).get("cell_crops_paths") or []
            for crop in listed:
                if Path(crop).is_file():
                    paths[f"calls/{call_id}/{Path(crop).name}"] = Path(crop)
        react_arm.assert_no_private(paths.values(), self.private_roots)
        return paths


class ReAct:
    """An LLM plus the tool (benchmarks/_shared/REACT_ARM_CONTRACT.md): the shared runner
    (`benchmarks/_shared/react_arm.py`) with the cropper as its only tool (`CropToolAdapter`), one session per
    image. The prompt holds SHAPE's task text for the image (`CropAgenticVerifier.task_text`, on the input record
    SHAPE's arm builds), the input (`react_input_text`), the parameter table (name, type, default), the budget and
    the time left; each observation is the tool's own returned dict and its crop tiles. Budget: SHAPE's `BUDGET`
    rounds, round 1 the tool at its defaults (SHAPE's `admission_call`), at most SHAPE's per-round cap of calls
    (`CropAgenticVerifier.concurrency()` on `cores`). The agent's finish names the delivered call; without one the
    default call is delivered (`default_session_unfinished`). Records: TRACE.jsonl, CALLS.json and SELECTED.json in
    `<out_dir>/<image>/`, REACT_SUMMARY.json in `out_dir`; the arm's trace carries them too."""

    def __init__(self, model: str, min_area: int, margin: int, margin_frac: float,
                 cores: int | None = None, deadline: float | None = None, **_: Any):
        self.model = model
        self.engine = _engine(model)
        self.min_area = min_area
        self.margin = margin
        self.margin_frac = margin_frac
        self.cores = int(cores) if cores else vrr_base.available_cores()
        self.deadline = deadline
        self.cap = self._verifier(None).concurrency()
        self.tool = Single_Cell_Cropper_Tool()

    def _verifier(self, base_kwargs: dict | None) -> CropAgenticVerifier:
        """SHAPE's crop verifier without an engine, begun as SHAPE's arm begins it (`CropTask`: the cores, the
        deadline, no brief), on SHAPE's input record of `base_kwargs`."""
        verifier = CropAgenticVerifier(log=lambda *_args, **_kwargs: None)
        verifier.begin(CropTask(cores=self.cores, deadline=self.deadline))
        verifier.input = input_record(base_kwargs) if base_kwargs else {}
        return verifier

    def __call__(self, image_path: Path, mask_path: Path, mask: np.ndarray, out_dir: Path) -> ArmResult:
        verifier = self._verifier({"original_image": str(image_path), "nuclei_mask": str(mask_path),
                                   "min_area": self.min_area, "margin": self.margin, "margin_frac": self.margin_frac})
        adapter = CropToolAdapter(self.tool, image_path, mask_path, out_dir, verifier)
        extra = {}
        if "unit_images" in inspect.signature(react_arm.run_react).parameters:
            # the raw input image (the file SHAPE's cropper gets), attached to the session's first question
            extra["unit_images"] = lambda _unit: [str(image_path)]
        summary = react_arm.run_react(
            adapter, task_text=verifier.task_text(),
            input_text=react_input_text(image_path, mask_path, mask, adapter.min_area),
            engine=self.engine, out_dir=out_dir, workers=1, deadline=self.deadline, image_px=PROMPT_IMAGE_PX,
            **extra)
        selected = summary["selected"][adapter.unit]
        session_dir = react_arm._unit_dir(Path(out_dir), adapter.unit)
        rounds_trace = [json.loads(line) for line in
                        (session_dir / "TRACE.jsonl").read_text(encoding="utf-8").splitlines() if line.strip()]
        trace = {"model": self.model, "runner": "benchmarks/_shared/react_arm.py", "cores": self.cores,
                 "round_cap": summary["round_cap"], "budget": summary["budget"],
                 "default_counts_as_round": summary["default_counts_as_round"],
                 "parameters": [vars(p) for p in adapter.parameters()],
                 "delivered_call": selected["call_id"], "delivery": selected["source"], "stop": selected["stop"],
                 "rounds": selected["rounds"], "calls": selected["calls"], "tokens": selected["tokens"],
                 "seconds": selected["seconds"], "session_dir": session_dir.name, "rounds_trace": rounds_trace,
                 "tool_calls": json.loads((session_dir / "CALLS.json").read_text(encoding="utf-8"))}
        try:
            result = adapter.results.get(selected["call_id"])
            if result is None or selected["call_error"]:
                raise RuntimeError(f"the delivered call {selected['call_id']} failed: {selected['call_error']}")
            crops = _metadata_to_crops(result, mask, adapter.min_area)
        except (RuntimeError, ValueError) as exc:
            exc.trace = trace                 # the runner records the trace of a failed image
            raise
        return ArmResult(crops, selected["calls"], trace)


class _CountingTool:
    def __init__(self, tool: Single_Cell_Cropper_Tool):
        self.tool = tool
        self.calls = 0
        self.lock = Lock()

    def execute(self, *args: Any, **kwargs: Any):
        with self.lock:
            self.calls += 1
        return self.tool.execute(*args, **kwargs)


class _MeteredEngine:
    def __init__(self, engine: Any):
        self.engine = engine
        self.lock = Lock()
        self.calls = {"reasoner": 0, "vision": 0}
        self.reported_calls = {"reasoner": 0, "vision": 0}
        self.tokens = {"reasoner": 0, "vision": 0}

    def __getattr__(self, name: str):
        return getattr(self.engine, name)

    def generate(self, *args: Any, **kwargs: Any):
        schema = kwargs.get("response_format")
        category = "vision" if getattr(schema, "__name__", "").endswith("Verdict") else "reasoner"
        reply = None
        try:
            reply = self.engine.generate(*args, **kwargs)
            return reply
        finally:
            usage = reply.get("usage") if isinstance(reply, dict) else None
            try:
                reported_tokens = int(usage["total_tokens"]) if isinstance(usage, dict) else None
            except (KeyError, TypeError, ValueError, OverflowError):
                reported_tokens = None
            with self.lock:
                self.calls[category] += 1
                if reported_tokens is not None:
                    self.reported_calls[category] += 1
                    self.tokens[category] += reported_tokens

    def summary(self) -> dict[str, int | None]:
        with self.lock:
            return {
                f"{category}_calls_per_image": self.calls[category]
                for category in ("reasoner", "vision")
            } | {
                f"{category}_tokens_per_image": (
                    self.tokens[category] if self.calls[category] == self.reported_calls[category] else None)
                for category in ("reasoner", "vision")
            }


class Shape:
    """SHAPE's crop loop on MaskCrop. The scored crops are SHAPE's delivered call (the agentic loop's
    `resolution.result`, which `deliver` points at the copy in `tool_cache/`; the linear loop's converged
    result). Every target is scored; the crops the mask check rejected stay in the scored set
    and are flagged `triage_rejected` (reported separately as `n_triage_rejected`). The model usage is
    metered per image."""

    def __init__(self, model: str, min_area: int, margin: int, margin_frac: float,
                 cores: int | None = None, deadline: float | None = None, shape_loop: str = "agentic", **_: Any):
        self.model = model
        self.engine = _engine(model)
        self.min_area = min_area
        self.margin = margin
        self.margin_frac = margin_frac
        self.cores = int(cores) if cores else vrr_base.available_cores()
        self.cap = vrr_base.concurrency_cap(self.cores, THREADS_PER_CALL)
        self.deadline = deadline
        self.shape_loop = shape_loop
        self.tool = Single_Cell_Cropper_Tool()

    def __call__(self, image_path: Path, mask_path: Path, mask: np.ndarray, out_dir: Path) -> ArmResult:
        counted_tool = _CountingTool(self.tool)
        engine = _MeteredEngine(self.engine)
        if self.shape_loop == "linear":
            resolution = rsc.run_shape_arm(
                counted_tool, image_path, mask_path, out_dir, self.min_area, self.margin,
                self.margin_frac, engine)
            result = rsc._converged_result(resolution)
        else:
            resolution = rsc.run_shape_agentic_arm(
                counted_tool, image_path, mask_path, out_dir, self.min_area, self.margin,
                self.margin_frac, engine, cores=self.cores, deadline=self.deadline)
            result = resolution.result
            if result is None:
                raise RuntimeError(f"SHAPE delivered no call: {resolution.status} ({resolution.reason})")
        rsc._organize_kept_rejected_crops(out_dir, result)
        crops = _metadata_to_crops(result, mask, self.min_area)
        metadata_path = result.get("cell_crops_metadata_path")
        kept = rsc._read_kept_paths(metadata_path) if metadata_path else None
        for crop in crops:
            crop["triage_rejected"] = kept is not None and crop.get("crop_image_path") not in kept
        details = getattr(resolution, "details", None) or {}
        return ArmResult(crops, counted_tool.calls, {
            "model": self.model, "shape_loop": self.shape_loop, "cores": self.cores,
            "concurrency_cap": self.cap, "status": getattr(resolution, "status", None),
            "reason": getattr(resolution, "reason", None), "delivered_call": details.get("call"),
            "delivery": details.get("choice"),
            "scored_metadata_path": metadata_path,
            "delivered_metadata_path": details.get("delivered_metadata_path"),
            "n_triage_rejected": sum(crop["triage_rejected"] for crop in crops),
            "result": result, "model_usage": engine.summary()})


def make_arm(name: str, **kwargs: Any):
    table = {
        "maskcrop": MaskCrop,
        "cellprofiler": CellProfiler,
        "gpt": GPT,
        "react": ReAct,
        "shape": Shape,
    }
    if name not in table:
        raise ValueError(f"unknown arm {name!r}; choices: {', '.join(ARMS)}")
    return table[name](**kwargs)


def load_mask(path: Path) -> np.ndarray:
    return tifffile.imread(str(path)).astype(np.int64)

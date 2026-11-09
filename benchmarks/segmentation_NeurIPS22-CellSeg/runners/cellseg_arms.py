"""The seven arms of the NeurIPS22-CellSeg benchmark, one callable per arm.

Domain tools (no LLM):
    cellpose_sam  Cell_Segmenter_Tool (Cellpose-SAM) once, at the tool's default parameters.
    cellpose3     Cellpose 3 `cyto3` (its own env: cellpose 3 and 4 cannot share one), the
                  diameter estimated by its size model, the highest-variance channel.
    microsam     micro-sam automatic instance segmentation, `vit_b_lm`, default settings
                  (tiled when the long side exceeds 2048 px).
    stardist      StarDist2D: `2D_versatile_he` for an RGB image with a bright background,
                  else `2D_versatile_fluo` on the highest-variance channel (inverted when the
                  background is bright); percentile normalization 1 / 99.8.
LLM arms (gpt-5.6-luna through the gateway, the engine SHAPE's vision judge uses):
    gpt           the LLM alone: it sees the image and returns one polygon per cell (JSON);
                  the polygons are rasterized into the instance mask. No tool.
    react         an LLM plus Cell_Segmenter_Tool on the shared ReAct runner
                  (benchmarks/_shared/react_arm.py; `SegmenterAdapter`): SHAPE's task text and
                  lever set (name, type, default), SHAPE's budget, round cap and default-round
                  rule; one session per image, run concurrently (`run_cellseg_benchmark.run_react_arm`).
    shape         the full SHAPE agent (`shape.solver.solve`, Prompt.txt, Cell_Segmenter_Tool
                  only), as `run_shape_neurips_cellseg.run_batch` runs it.

Every arm but react is `arm(image_path, out_dir) -> {"mask", "tool_calls", "trace"}`. Heavy imports
are inside the arms, so the cellpose3, microsam and stardist arms run in envs without `shape`.

Set arm (SET_ARMS: one `solve()` over a set of images, `run_cellseg_benchmark.py run-set`, not a callable here):
    shape_cohort  the SHAPE agent over the whole set with the segmentation cohort on (`segmentation_cohort=True`).
"""

from __future__ import annotations

import io
import json
import os
import re
from pathlib import Path

import numpy as np

ARMS = ("cellpose_sam", "cellpose3", "microsam", "stardist", "gpt", "react", "shape")
LLM_ARMS = ("gpt", "react", "shape")
SET_ARMS = ("shape_cohort",)

_BENCH = Path(__file__).resolve().parents[1]          # benchmarks/segmentation_NeurIPS22-CellSeg
_PROMPT_FILE = _BENCH / "results" / "Prompt.txt"


# --------------------------------------------------------------------------- #
# Image / mask I/O
# --------------------------------------------------------------------------- #
def read_image(path) -> np.ndarray:
    """The raw image as H x W or H x W x C (a channels-first stack is moved last)."""
    try:
        from skimage.io import imread
        arr = np.asarray(imread(str(path)))
    except Exception:
        from PIL import Image
        with Image.open(path) as image:
            arr = np.asarray(image)
    arr = arr.squeeze()
    if arr.ndim == 3 and arr.shape[0] <= 4 < arr.shape[-1]:
        arr = np.moveaxis(arr, 0, -1)
    if arr.ndim == 3 and arr.shape[-1] == 4:            # drop alpha
        arr = arr[..., :3]
    return arr


def load_mask(path) -> np.ndarray:
    """An instance-label map from .npy (the segmenter's, maybe a dict payload) or an image."""
    path = str(path)
    if path.lower().endswith(".npy"):
        loaded = np.load(path, allow_pickle=True)
        if loaded.shape == () and loaded.dtype == object and isinstance(loaded.item(), dict):
            payload = loaded.item()
            loaded = next(payload[k] for k in ("masks", "mask", "labels") if k in payload)
        return np.asarray(loaded).squeeze()
    arr = np.asarray(read_image(path)).squeeze()
    return arr[..., 0] if arr.ndim == 3 else arr


def _gray(img: np.ndarray) -> np.ndarray:
    """One channel: the image itself, or its highest-variance channel."""
    if img.ndim == 2:
        return img.astype(np.float32)
    stds = [float(img[..., c].std()) for c in range(img.shape[-1])]
    return img[..., int(np.argmax(stds))].astype(np.float32)


def _unit(img: np.ndarray, lo: float = 1.0, hi: float = 99.8) -> np.ndarray:
    a, b = np.percentile(img, (lo, hi))
    return np.clip((img.astype(np.float32) - a) / max(float(b - a), 1e-6), 0.0, 1.0)


def _bright_background(img: np.ndarray) -> bool:
    return float(np.median(_unit(_gray(img) if img.ndim == 3 else img))) > 0.5


def display_png(img: np.ndarray, max_px: int) -> tuple[bytes, float]:
    """An 8-bit PNG of the image (RGB kept, otherwise one channel), long side <= max_px.
    Returns (png bytes, scale = shown size / original size)."""
    from PIL import Image
    if img.ndim == 3 and img.shape[-1] == 3:
        shown = np.stack([_unit(img[..., c]) for c in range(3)], axis=-1)
    else:
        shown = _unit(_gray(img))
    pil = Image.fromarray((shown * 255).astype(np.uint8))
    scale = min(1.0, max_px / max(pil.size))
    if scale < 1.0:
        pil = pil.resize((max(1, round(pil.width * scale)), max(1, round(pil.height * scale))),
                         Image.LANCZOS)
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    return buf.getvalue(), pil.width / img.shape[1]


def rasterize_polygons(polygons, shape: tuple[int, int], scale: float) -> tuple[np.ndarray, int]:
    """Instance mask from polygons given as [[x, y], ...] in the shown (scaled) frame, and the
    number of polygons skipped (fewer than 3 vertices, a vertex that is not [x, y], or empty)."""
    from skimage.draw import polygon as fill
    mask = np.zeros(shape, dtype=np.int32)
    label = skipped = 0
    for poly in polygons:
        try:
            pts = np.asarray(poly, dtype=np.float64)
            if pts.ndim != 2 or pts.shape[1] != 2 or len(pts) < 3:
                raise ValueError("not a polygon")
        except (TypeError, ValueError):
            skipped += 1
            continue
        pts = pts / scale
        rr, cc = fill(pts[:, 1], pts[:, 0], shape=shape)
        if not rr.size:
            skipped += 1
            continue
        label += 1
        mask[rr, cc] = label
    return mask, skipped


_POLYGON = re.compile(r'"polygon"\s*:\s*(\[\s*\[.*?\]\s*\])', flags=re.S)


def salvage_polygons(text: str) -> list:
    """The complete {"polygon": [[x, y], ...]} entries of a (possibly truncated) JSON reply."""
    out = []
    for m in _POLYGON.finditer(text):
        try:
            out.append(json.loads(m.group(1)))
        except ValueError:
            continue
    return out


def target_text() -> str:
    """The '## Target' section of Prompt.txt, so every LLM arm is told the same target."""
    text = _PROMPT_FILE.read_text(encoding="utf-8")
    m = re.search(r"^## Target\s*\n(.*?)(?=^---|^## )", text, flags=re.S | re.M)
    return m.group(1).strip() if m else ""


def _parse(reply, schema):
    """The engine's reply as a `schema` instance, or None: a parsed object, or JSON text."""
    content = reply.get("content") if isinstance(reply, dict) else reply
    if isinstance(content, schema):
        return content
    if isinstance(content, str):
        m = re.search(r"\{.*\}", content, flags=re.S)
        if m:
            try:
                return schema.model_validate_json(m.group(0))
            except Exception:
                return None
    return None


def _vision_engine(model: str, timeout_s: float):
    from shape.agent.verification.vision_engine import build_vision_engine
    engine = build_vision_engine(model)
    if engine is None:
        raise RuntimeError(f"no vision engine for {model} (LLM_BASE_URL / LLM_API_KEY?)")
    if hasattr(getattr(engine, "client", None), "with_options"):
        engine.client = engine.client.with_options(timeout=timeout_s)
    return engine


# --------------------------------------------------------------------------- #
# Domain tools
# --------------------------------------------------------------------------- #
class CellposeSAM:
    def __init__(self, **_):
        from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
        self.tool = Cell_Segmenter_Tool()

    def __call__(self, image_path: Path, out_dir: Path) -> dict:
        r = self.tool.execute(image=str(image_path), query_cache_dir=str(out_dir),
                              image_id=image_path.stem)
        mask_path = r.get("mask_path") if isinstance(r, dict) else None
        trace = {"parameters": r.get("parameters"), "summary": r.get("summary")} \
            if isinstance(r, dict) else {"result": str(r)}
        return {"mask": load_mask(mask_path) if mask_path else None, "tool_calls": 1,
                "trace": trace}


class Cellpose3:
    def __init__(self, model_type: str = "cyto3", **_):
        from cellpose import core, models
        from importlib import metadata
        version = metadata.version("cellpose")
        if not version.startswith("3."):
            raise RuntimeError(f"the cellpose3 arm needs cellpose 3.x, this env has {version}")
        self.version, self.model_type = version, model_type
        self.gpu = bool(core.use_gpu())
        self.model = models.Cellpose(gpu=self.gpu, model_type=model_type)

    def __call__(self, image_path: Path, out_dir: Path) -> dict:
        img = _gray(read_image(image_path))
        masks, _flows, _styles, diams = self.model.eval(img, diameter=None, channels=[0, 0])
        return {"mask": np.asarray(masks), "tool_calls": 1,
                "trace": {"cellpose": self.version, "model_type": self.model_type, "gpu": self.gpu,
                          "estimated_diameter": float(np.asarray(diams).ravel()[0])}}


class MicroSAM:
    TILE, HALO, TILE_ABOVE = (1024, 1024), (256, 256), 2048

    def __init__(self, model_type: str = "vit_b_lm", **_):
        self.model_type = model_type
        self._by_tiling: dict = {}
        self._get(False)

    def _get(self, tiled: bool):
        if tiled not in self._by_tiling:
            from micro_sam.automatic_segmentation import get_predictor_and_segmenter
            self._by_tiling[tiled] = get_predictor_and_segmenter(model_type=self.model_type,
                                                                 is_tiled=tiled)
        return self._by_tiling[tiled]

    def __call__(self, image_path: Path, out_dir: Path) -> dict:
        from micro_sam.automatic_segmentation import automatic_instance_segmentation
        img = read_image(image_path)
        if img.ndim == 3 and img.shape[-1] != 3:
            img = _gray(img)
        tiled = max(img.shape[:2]) > self.TILE_ABOVE
        predictor, segmenter = self._get(tiled)
        kw = {"tile_shape": self.TILE, "halo": self.HALO} if tiled else {}
        mask = automatic_instance_segmentation(predictor=predictor, segmenter=segmenter,
                                               input_path=img, ndim=2, verbose=False, **kw)
        return {"mask": np.asarray(mask), "tool_calls": 1,
                "trace": {"model_type": self.model_type, "tiled": tiled}}


class StarDist:
    def __init__(self, **_):
        from stardist.models import StarDist2D
        self.models = {name: StarDist2D.from_pretrained(name)
                       for name in ("2D_versatile_he", "2D_versatile_fluo")}

    def __call__(self, image_path: Path, out_dir: Path) -> dict:
        from csbdeep.utils import normalize
        img = read_image(image_path)
        bright = _bright_background(img)
        if img.ndim == 3 and img.shape[-1] == 3 and bright:
            name, x = "2D_versatile_he", normalize(img.astype(np.float32), 1, 99.8, axis=(0, 1))
            tiles = (*self._tiles(img.shape), 1)
        else:
            g = _gray(img)
            name, x = "2D_versatile_fluo", normalize(g.max() - g if bright else g, 1, 99.8)
            tiles = self._tiles(img.shape)
        labels, _ = self.models[name].predict_instances(x, n_tiles=tiles, show_tile_progress=False)
        return {"mask": np.asarray(labels), "tool_calls": 1,
                "trace": {"model": name, "inverted": name.endswith("fluo") and bright,
                          "n_tiles": list(tiles)}}

    @staticmethod
    def _tiles(shape) -> tuple[int, int]:
        return tuple(max(1, int(np.ceil(s / 1024))) for s in shape[:2])


# --------------------------------------------------------------------------- #
# LLM arms
# --------------------------------------------------------------------------- #
_GPT_PROMPT = """You are given ONE microscopy image (NeurIPS 2022 Cell Segmentation Challenge).
Segment every individual cell: return one polygon per cell outlining that cell's full extent.

Target: {target}

The image is {width} x {height} pixels. Give each polygon as a list of [x, y] vertices in this
image's pixel coordinates (x = column, y = row, origin top-left), 4-24 vertices per cell,
in order around the cell. Every cell gets its own polygon; touching cells get separate polygons.
Return JSON only: {{"cells": [{{"polygon": [[x, y], ...]}}, ...]}}"""


class GPTPolygons:
    def __init__(self, model: str, max_px: int = 1024, max_tokens: int = 32000, engine=None, **_):
        from pydantic import BaseModel

        class Cell(BaseModel):
            polygon: list[list[float]]

        class Cells(BaseModel):
            cells: list[Cell]

        self.schema = Cells
        self.engine = engine or _vision_engine(model, timeout_s=900)
        self.max_px, self.max_tokens = max_px, max_tokens

    def __call__(self, image_path: Path, out_dir: Path) -> dict:
        img = read_image(image_path)
        png, scale = display_png(img, self.max_px)
        h, w = round(img.shape[0] * scale), round(img.shape[1] * scale)
        prompt = _GPT_PROMPT.format(target=target_text(), width=w, height=h)
        reply = self.engine.generate([prompt, png], response_format=self.schema,
                                     max_tokens=self.max_tokens)
        parsed = _parse(reply, self.schema)
        trace = {"shown_size": [w, h], "scale": scale, "parsed": parsed is not None}
        if parsed is not None:
            polygons = [c.polygon for c in parsed.cells]
        else:            # e.g. a reply cut at max_tokens: keep its complete polygons
            content = reply.get("content") if isinstance(reply, dict) else reply
            polygons = salvage_polygons(content) if isinstance(content, str) else []
            trace["reply_head"] = str(reply)[:2000]
        trace["n_polygons"] = len(polygons)
        # the full exchange for the audit record (export_run_records.py): prompt, reply, usage
        content = reply.get("content") if isinstance(reply, dict) else reply
        (out_dir / "gpt_reply.json").write_text(json.dumps({
            "prompt": prompt, "image_shown": {"size": [w, h], "scale_to_original": scale},
            "reply": parsed.model_dump() if parsed is not None else str(content),
            "usage": reply.get("usage") if isinstance(reply, dict) else None}, default=str), encoding="utf-8")
        if not polygons:
            return {"mask": None, "tool_calls": 0, "trace": trace}
        mask, trace["n_skipped"] = rasterize_polygons(polygons, img.shape[:2], scale)
        return {"mask": mask, "tool_calls": 0, "trace": trace}


# SHAPE's own segmentation levers, not offered to ReAct (user decision 2026-10-12): the tool runs them at its
# own defaults in every ReAct call.
SHAPE_ONLY_LEVERS = ("color_class_filter", "resample")


class SegmenterAdapter:
    """The react arm's tool adapter on the shared ReAct runner (`benchmarks/_shared/react_arm.py`,
    REACT_ARM_CONTRACT.md): an LLM plus Cell_Segmenter_Tool, one session per image.

    - The tool is called as SHAPE's arm calls it: `execute(image=, query_cache_dir=, image_id=, **params)`
      (model cpsam, the tool's own bsize / model defaults), one tool for every session.
    - The parameters are SHAPE's segmentation verifier lever set (`SegmentationAgenticVerifier`: its CARD
      `## Levers`, through the verifier's `levers`, `lever_spec` and `defaults`) minus SHAPE's own levers
      (`SHAPE_ONLY_LEVERS`: color_class_filter, resample - the tool runs them at its defaults), each as name, type
      and default only; a choice's type names its values (the type of the value), nothing else of the CARD.
    - The round cap is the verifier's own `concurrency()` (`sam_concurrency`: 1 for the in-process model); the
      default call is round 1 as the verifier's `admission_call` counts it.
    - An observation is the tool's returned dict and the overlay PNG the tool wrote for that call; a call is failed
      when the tool returns an `error` or no mask. The raw input image (`input_images`, the file the segmenter gets)
      is attached to each session's first question.
    - The in-process Cellpose model is one GPU: its calls are serialized by one lock (a SAM server takes the
      concurrent calls itself).
    - `python_inputs`: the raw input image and each finished call's mask and overlay files; never a file under
      `private_roots` (the labels)."""

    tool_name = "Cell_Segmenter_Tool"
    unit_kind = "image"

    def __init__(self, images: dict, out_dirs: dict, tool=None, verifier=None, private_roots=()):
        import threading
        from contextlib import nullcontext
        if verifier is None:
            from shape.agent.verification.verifiers.segmentation import SegmentationAgenticVerifier
            verifier = SegmentationAgenticVerifier()
        if tool is None:
            from shape.tools.cell_segmenter.tool import Cell_Segmenter_Tool
            tool = Cell_Segmenter_Tool()
        self.images = {u: Path(p) for u, p in images.items()}
        self.out_dirs = {u: Path(d) for u, d in out_dirs.items()}
        self.tool, self.verifier = tool, verifier
        self.private_roots = [Path(r) for r in private_roots]
        self.lock = nullcontext() if getattr(tool, "_sam_server_url", None) else threading.Lock()
        self.results: dict = {}          # (unit, call id) -> the tool's returned dict
        self.delivered: dict = {}        # unit -> the call id delivered

    def parameters(self) -> list:
        from benchmarks._shared.react_arm import Param
        defaults = self.verifier.defaults()
        return [Param(lever, lever_type(*self.verifier.lever_spec(lever)), defaults.get(lever))
                for lever in self.verifier.levers if lever not in SHAPE_ONLY_LEVERS]

    def units(self) -> list:
        return list(self.images)

    def round_cap(self) -> int:
        return int(self.verifier.concurrency())

    def default_counts_as_round(self) -> bool:
        return bool(self.verifier.admission_call)

    def call(self, unit: str, params: dict, call_id: str) -> dict:
        with self.lock:
            result = self.tool.execute(image=str(self.images[unit]), query_cache_dir=str(self.out_dirs[unit]),
                                       image_id=unit, **params)
        if not isinstance(result, dict):
            return {"output": None, "images": [], "error": f"the tool returned {type(result).__name__}"}
        self.results[(unit, call_id)] = result
        overlay = result.get("overlay_path")
        error = result.get("error") or (None if result.get("mask_path") else "the tool returned no mask")
        return {"output": result, "images": [overlay] if overlay and os.path.isfile(overlay) else [],
                "error": error}

    def deliver(self, unit: str, call_id: str) -> None:
        self.delivered[unit] = call_id

    def input_images(self, unit: str) -> list:
        """The raw input image as the agent sees it: the file the segmenter gets when it renders directly (a TIFF,
        rendered by the runner as SHAPE's vision reads render it; an 8-bit PNG / JPEG / BMP), else that rendering of
        it written to the image's dir as input_image.png (e.g. a 16-bit PNG)."""
        from PIL import Image
        from benchmarks._shared.react_arm import shape_display_image
        path = self.images[unit]
        if path.suffix.lower() in (".tif", ".tiff"):
            return [str(path)]
        with Image.open(path) as img:
            if img.mode in ("L", "RGB", "RGBA", "P"):
                return [str(path)]
        out = self.out_dirs[unit] / "input_image.png"
        shape_display_image(str(path)).save(out)
        return [str(out)]

    def python_inputs(self, unit: str) -> dict:
        """The files a python step may read: inputs/<the raw image>, calls/<call id>/<the call's mask and overlay>."""
        from benchmarks._shared.react_arm import assert_no_private
        paths = {f"inputs/{self.images[unit].name}": self.images[unit]}
        for (owner, call_id), result in list(self.results.items()):
            if owner != unit or result.get("error"):
                continue
            for key in ("mask_path", "overlay_path"):
                path = result.get(key)
                if path and os.path.isfile(path):
                    paths[f"calls/{call_id}/{Path(path).name}"] = Path(path)
        assert_no_private(paths.values(), self.private_roots)
        return paths

    def delivered_mask_path(self, unit: str):
        """The mask file of the call delivered for `unit` (None: nothing delivered, or the call wrote no mask)."""
        return (self.results.get((unit, self.delivered.get(unit))) or {}).get("mask_path")


def lever_type(kind: str, bounds, nullable: bool) -> str:
    """A lever's type as the ReAct prompt states it: the kind ("number", "integer", "bool"; "... or null" when
    null is allowed), a choice with its values (no range, no description)."""
    if kind == "choice":
        return "choice of " + ", ".join(json.dumps(v) for v in (bounds or ()))
    return f"{kind} or null" if nullable else kind


def input_facts(image_path: Path) -> str:
    """The image's own facts, as SHAPE's input record states them (file, size, channels, dtype); nothing computed."""
    img = np.asarray(read_image(image_path))
    channels = img.shape[2] if img.ndim == 3 else 1
    return (f"Image: {image_path.name} ({img.shape[1]} x {img.shape[0]} pixels, {channels} channel(s), "
            f"dtype {img.dtype}).")


class Shape:
    def __init__(self, model: str, max_steps: int = 10, **_):
        import sys
        if str(_BENCH / "runners") not in sys.path:
            sys.path.insert(0, str(_BENCH / "runners"))
        import run_shape_neurips_cellseg as bench   # also puts benchmarks/_shared on sys.path
        import segmentation_metrics as sm
        from shape.engine.factory import resolve_api_key
        self.bench, self.sm, self.model, self.max_steps = bench, sm, model, max_steps
        self.api_key = resolve_api_key(model)
        self.question = bench._default_question()

    def __call__(self, image_path: Path, out_dir: Path) -> dict:
        result = self.bench.solve(question=self.question,
                       images=[{"path": str(image_path), "group": "default"}],
                       llm_engine_name=self.model, api_key=self.api_key,
                       max_steps=self.max_steps, query_cache_dir=str(out_dir),
                       enabled_tools=self.bench._SHAPE_MAIN_TOOLS)
        ver = result.get("verification_summary") or []
        (out_dir / "verification_summary.json").write_text(
            json.dumps(ver, indent=1, default=str), encoding="utf-8")
        prediction = self.bench._collect_masks_by_image(result["memory"]).get(image_path.stem)
        calls = len(self.sm.attempt_masks_for(prediction)) if prediction else \
            sum(1 for _ in out_dir.rglob("*_a[0-9]*.npy"))
        return {"mask": load_mask(prediction) if prediction else None, "tool_calls": calls,
                "trace": {"stop_reason": result.get("stop_reason"), "mask_path": prediction,
                          "agent_trace": self.bench._image_trace(ver)["stages"]}}


_FACTORIES = {"cellpose_sam": CellposeSAM, "cellpose3": Cellpose3, "microsam": MicroSAM, "stardist": StarDist,
              "gpt": GPTPolygons, "shape": Shape}


def make_arm(name: str, **kwargs):
    """The arm `name`, built once (models loaded) and then called per image. The react arm runs its images as
    concurrent sessions (`run_cellseg_benchmark.run_react_arm`), not as a per-image callable."""
    if name == "react":
        raise ValueError("the react arm runs through run_cellseg_benchmark.run_react_arm (the shared ReAct runner), "
                         "not as a per-image callable")
    if name not in _FACTORIES:
        raise ValueError(f"unknown arm {name!r}; arms: {', '.join(ARMS)}")
    return _FACTORIES[name](**kwargs)

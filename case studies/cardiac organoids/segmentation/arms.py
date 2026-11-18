"""Study adapters; inference, ReAct and VRR stay in their existing shared implementations."""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np

from common import HERE, MODEL, REPO, cellseg, dump, sha256

TASK = (HERE / "Prompt.txt").read_text(encoding="utf-8")


def probe_dependencies(name):
    """Check the selected interpreter's real imports/API before any paid calls or image loop."""
    import importlib
    checks = {
        "cellpose_sam": [("shape.tools.cell_segmenter.tool", "Cell_Segmenter_Tool")],
        "shape_seg_co": [("shape.tools.organoid_segmenter.tool", "Organoid_Segmenter_Tool")],
        "microsam": [("micro_sam.automatic_segmentation", "get_predictor_and_segmenter"),
                     ("micro_sam.automatic_segmentation", "automatic_instance_segmentation")],
        "stardist": [("stardist.models", "StarDist2D"), ("csbdeep.utils", "normalize")],
        "llm": [("shape.engine.openai", "ChatOpenAI")],
        "react": [("shape.tools.organoid_segmenter.tool", "Organoid_Segmenter_Tool"),
                  ("benchmarks._shared.react_arm", "run_react")],
        "shape": [("shape.solver", "solve"),
                  ("shape.agent.verification.verifiers.organoid_segmentation", "OrganoidSegmentationAgenticVerifier")],
        "organoid_id": [("Core.Model", "LoadLiteModel"), ("Core.Identification", "SeparateContours")],
    }
    if name == "organoid_id":
        import sys
        repo = str(Path(os.environ["ORGANOID_ID_REPO"]).resolve())
        if repo not in sys.path:
            sys.path.insert(0, repo)
    origins = {}
    for module_name, symbol in checks[name]:
        module = importlib.import_module(module_name)
        if not callable(getattr(module, symbol)):
            raise RuntimeError(f"Missing callable {module_name}.{symbol}")
        origin = Path(module.__file__).resolve()
        if module_name.startswith(("shape.", "benchmarks.")) and REPO not in origin.parents:
            raise RuntimeError(f"Imported stale checkout: {origin}")
        origins[module_name] = str(origin)
    if name in ("cellpose_sam", "shape_seg_co", "react", "shape"):
        from importlib.metadata import version
        if int(version("cellpose").split(".")[0]) != 4:
            raise RuntimeError("CO/CPSAM arms require the repository's Cellpose 4 environment")
    return origins


def gateway_preflight(out_dir):
    from shape.agent.verification.vision_engine import build_vision_engine, structured_smoke
    from pydantic import BaseModel
    from typing import Literal

    if not os.environ.get("LLM_API_KEY"):
        raise RuntimeError("LLM_API_KEY is missing; use the SHAPE gateway configuration")
    from benchmarks.full_CPJUMP1.harness.gateway_route import validate
    route = validate(os.environ.get("LLM_BASE_URL", ""))      # an https gateway (or the verified loopback proxy)
    engine = build_vision_engine(MODEL)
    if engine is None:
        raise RuntimeError("No gateway engine; no model/provider fallback is permitted")
    structured_smoke(engine)

    class VisionSmoke(BaseModel):
        answer: Literal["OK"]

    png, _ = cellseg.display_png(np.arange(64, dtype=np.uint8).reshape(8, 8), 1024)
    response = engine.generate(["An image is attached. Return answer OK.", png], response_format=VisionSmoke)
    if cellseg._parse(response, VisionSmoke) is None:
        raise RuntimeError("Structured multimodal preflight failed")
    dump(Path(out_dir) / "PREFLIGHT.json", {"model": MODEL,
         "reasoning_effort": os.environ.get("SHAPE_REASONING_EFFORT", "high"),
         "endpoint": route["base_url"], "text_and_vision_structured": True})
    return engine


class OrganoidDirect:
    def __init__(self, model_path):
        from shape.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
        self.tool = Organoid_Segmenter_Tool(model_path=str(model_path))

    def __call__(self, image, out):
        result = self.tool.execute(image=str(image), query_cache_dir=str(out), image_id=image.stem,
                                   adaptive_rescue=False, attempt_id=1)
        # Native QC warnings are recorded, but a baseline mask remains evaluable.
        path = result.get("mask_path")
        return {"mask": cellseg.load_mask(path) if path else None, "tool_calls": 1, "trace": result}


ORGANOID_ID_MODEL_SHA256 = "d21c7975b321102acb2003edc4dac9ca730046b4829a93bc441e4d8a6f017c3d"


class OrganoID:
    """Pretrained OrganoID (Matthews et al., PLOS Comput Biol 2022; jono-m/OrganoID @ deb7aeaa) through its
    own Python pipeline: model-size input with per-image min-max scaling, TFLite belief map, edge-guided
    watershed separation and cleanup at its published defaults (threshold 0.5, edge sigma 2, hysteresis
    0.005/0.05, minimum area 100, holes filled). Its CLI cannot be used: Run.py reads a misspelled
    argument. One task alignment: border-touching organoids are kept, as the task and GT require."""

    THRESHOLD, EDGE_SIGMA, EDGE_MIN, EDGE_MAX, MIN_AREA = 0.5, 2.0, 0.005, 0.05, 100

    def __init__(self, repo=None):
        import sys
        repo = Path(repo or os.environ["ORGANOID_ID_REPO"]).resolve()
        model = repo / "OptimizedModel"
        if sha256(model) != ORGANOID_ID_MODEL_SHA256:
            raise RuntimeError("OrganoID model differs from the pinned OptimizedModel")
        if str(repo) not in sys.path:
            sys.path.insert(0, str(repo))
        from Core.Model import LoadLiteModel
        self.repo, self.model = repo, LoadLiteModel(model)

    def __call__(self, image, out):
        from PIL import Image
        from Core.Model import Detect, PrepareImagesForModel
        from Core.Identification import Cleanup, DetectEdges, SeparateContours
        with Image.open(image) as pil:
            size = pil.size
            prepared = PrepareImagesForModel([pil], self.model, verbose=False)
        belief = Detect(self.model, prepared, 1)
        edges = DetectEdges(belief, self.EDGE_SIGMA, self.EDGE_MIN, self.EDGE_MAX, self.THRESHOLD)
        labels = SeparateContours(belief, edges, self.THRESHOLD, self.EDGE_SIGMA)
        cleaned = Cleanup(labels, self.MIN_AREA, False, True)[0]          # removeBorders=False
        mask = np.asarray(Image.fromarray(cleaned).resize(size, resample=Image.Resampling.NEAREST))
        return {"mask": mask.astype(np.int32), "tool_calls": 1,
                "trace": {"model_sha256": ORGANOID_ID_MODEL_SHA256, "model_input": list(prepared.shape[1:]),
                          "threshold": self.THRESHOLD, "edge_sigma": self.EDGE_SIGMA, "edge_min": self.EDGE_MIN,
                          "edge_max": self.EDGE_MAX, "minimum_area": self.MIN_AREA, "remove_border": False,
                          "fill_holes": True, "organoids": int(len(np.unique(mask)) - 1)}}


class LLMPolygons:
    def __init__(self, engine):
        from pydantic import BaseModel

        class Organoid(BaseModel):
            polygon: list[list[float]]

        class Organoids(BaseModel):
            organoids: list[Organoid]

        self.engine, self.schema = engine, Organoids

    def __call__(self, image, out):
        raw = cellseg.read_image(image)
        png, _ = cellseg.display_png(raw, 1024)
        from PIL import Image
        import io
        with Image.open(io.BytesIO(png)) as shown:
            w, h = shown.size
        prompt = (TASK + f"\nThis displayed image is {w} x {h} pixels. Return one polygon per whole organoid, "
                  "4-48 [x,y] vertices each, origin top-left, x=column and y=row in this displayed image. "
                  'JSON: {"organoids": [{"polygon": [[x,y], ...]}, ...]}. No tools are available.')
        reply = self.engine.generate([prompt, png], response_format=self.schema, max_tokens=32000)
        parsed = cellseg._parse(reply, self.schema)
        if parsed is None:
            raise ValueError("Malformed polygon response; no partial JSON salvage")
        polygons = []
        for organoid in parsed.organoids:
            pts = np.asarray(organoid.polygon, dtype=float)
            if (pts.ndim != 2 or pts.shape[1:] != (2,) or len(pts) < 3 or not np.isfinite(pts).all()
                    or (pts < 0).any() or (pts[:, 0] > w).any() or (pts[:, 1] > h).any()):
                raise ValueError("Invalid organoid polygon coordinates")
            # Separate x/y factors avoid rounding distortion for non-square images.
            pts *= np.array([raw.shape[1] / w, raw.shape[0] / h])
            polygons.append(pts)
        mask, skipped = cellseg.rasterize_polygons(polygons, raw.shape[:2], 1.0)
        if skipped:
            raise ValueError(f"{skipped} polygon(s) rasterized empty")
        return {"mask": mask, "tool_calls": 0, "trace": {"shown_size": [w, h], "polygons": parsed.model_dump()}}


class OrganoidAdapter(cellseg.SegmenterAdapter):
    tool_name = "Organoid_Segmenter_Tool"

    def call(self, unit, params, call_id):
        # Fixed model and disabled nested rescue; only the shared ReAct runner spends rounds.
        if set(params) - set(self.verifier.levers):
            return {"output": None, "images": [], "error": "Unknown organoid parameter"}
        with self.lock:
            result = self.tool.execute(image=str(self.images[unit]), query_cache_dir=str(self.out_dirs[unit]),
                                       image_id=unit, attempt_id=call_id, adaptive_rescue=False, **params)
        self.results[(unit, call_id)] = result
        path = result.get("overlay_path")
        return {"output": result, "images": [path] if path and Path(path).is_file() else [],
                "error": result.get("error") or (None if result.get("mask_path") else "No mask")}


class React:
    def __init__(self, model_path, engine):
        from shape.agent.verification.verifiers.organoid_segmentation import OrganoidSegmentationAgenticVerifier
        self.tool = OrganoidDirect(model_path).tool
        self.verifier = OrganoidSegmentationAgenticVerifier()
        self.engine = engine

    def __call__(self, image, out):
        from benchmarks._shared.react_arm import run_react
        # One image session at a time: exact per-image usage and no GPU concurrency differences.
        adapter = OrganoidAdapter({image.stem: image}, {image.stem: out}, tool=self.tool,
                                  verifier=self.verifier)
        result = run_react(adapter, task_text=TASK, input_text="One microscopy image; deliver its instance mask.",
                           engine=self.engine, out_dir=out / "react", workers=1, deadline=None,
                           unit_input=lambda _: cellseg.input_facts(image), unit_images=adapter.input_images)
        selected = result["selected"][image.stem]
        path = adapter.delivered_mask_path(image.stem)
        return {"mask": cellseg.load_mask(path) if path and not selected.get("call_error") else None,
                "tool_calls": selected["calls"], "trace": result}


class Shape:
    def __init__(self, max_steps):
        self.max_steps = max_steps

    def __call__(self, image, out):
        from shape.solver import solve
        result = solve(question=TASK, images=[{"path": str(image), "group": "cardiac_organoid"}],
                       llm_engine_name=MODEL, api_key=os.environ["LLM_API_KEY"],
                       enabled_tools=["Organoid_Segmenter_Tool"], max_steps=self.max_steps,
                       query_cache_dir=str(out), enable_verification=True, enable_llm_verifier=True,
                       segmentation_cohort=False, image_workers=1)
        verification = result.get("verification_summary") or []
        from audit import safe
        dump(out / "verification_summary.json", safe(verification))
        actions = result["memory"].get_actions(llm_safe=False)
        dump(out / "actions.json", safe(actions))
        chosen = None
        for action in actions:
            data = action.get("result")
            if not isinstance(data, dict):
                continue
            for entry in data.get("per_image", [data]):
                if isinstance(entry, dict) and entry.get("mask_path"):
                    chosen = entry
        if chosen and (chosen.get("error") or chosen.get("status") == "failed"):
            chosen = None
        return {"mask": cellseg.load_mask(chosen["mask_path"]) if chosen else None,
                "trace": {"stop_reason": result.get("stop_reason"), "selected": chosen,
                          "verification_summary": verification}}


def make_arm(name, model_path, engine, max_steps):
    if name == "shape_seg_co":
        return OrganoidDirect(model_path)
    if name == "llm":
        return LLMPolygons(engine)
    if name == "react":
        return React(model_path, engine)
    if name == "shape":
        return Shape(max_steps)
    if name == "organoid_id":
        return OrganoID()
    return cellseg.make_arm(name)

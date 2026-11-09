"""The segmentation cohort of the organoid segmenter (`Organoid_Segmenter_Tool`, the fixed custom CO Cellpose model).

The shared harness runs SHAPE's segment stage for this segmenter image by image (`pipeline.segmentation`: "the cohort and
shared paths are Cellpose-SAM's"), although `shape.agent.verification.segmentation_cohort.run` accepts the organoid tool.
This module is the cohort path for it, `pipeline._segmentation_cohort` with the Cellpose-SAM tool replaced by the
dataset's segmenter (`pipeline.segmenter`), nothing else changed: `segmentation_cohort.run` at its defaults (no organoid
selective review) groups the images by their preflight read, runs the per-image VRR on a seeded calibration sample of each
group (`calibration_size`: 29 images for 1141), applies the consensus of the levers the calibration agrees on to the other
images in the verifier's cohort mode (the call at the defaults, the admission and one reference call, no agent round),
and gives the images that the admission standard still flags their own VRR from the consensus. A small group, an unstable
consensus and an image without a confident read take the per-image VRR.

An arm without a reasoner (control) has no cohort: it takes the harness's own per-image path. `install()` swaps this
function into `pipeline.STAGES['segment']` of the current process (a stage worker); no harness file is edited.

Records: `segment/<id>/SEARCH.json` (every pass that ran the image's own VRR), `segment/<id>/COHORT_SEARCH.json` (a
cohort-mode pass), `segment_COHORT.json` (the cohort's record: groups, calibration samples, consensus, routing) and each
image's `parameter_source` in `segmentation.json`.
"""
from __future__ import annotations

from pathlib import Path
import threading

import numpy as np
import tifffile

from benchmarks.full_CPJUMP1.harness import pipeline
from benchmarks.full_CPJUMP1.harness.data import digest, write_json, within

REGIME = "cohort"                 # CONFIG.json `segmentation_regime` that selects this path (the harness's own: "per-image")
IMPLEMENTATION = "shape.agent.verification.segmentation_cohort.run"


def segmentation(prep: Path, run: Path, config: dict, images: list[dict]) -> None:
    engine = pipeline.engine_for(config, "segment")
    if engine is None or config.get("segmentation_regime") != REGIME:
        return pipeline.segmentation(prep, run, config, images)
    from benchmarks.full_HCMA.harness.shape_arm import segment_resolution, tools_common
    from shape.agent.verification import segmentation_cohort

    spec = pipeline.dataset(config)
    seg = pipeline.segmenter(spec, config.get("tool_params", {}))
    tool, registry, defaults, fixed, name = seg["tool"], seg["registry"], seg["defaults"], seg["fixed"], seg["name"]
    if name != segmentation_cohort.TOOLS[1]:
        raise ValueError(f"the organoid cohort needs the organoid segmenter, not {name}")
    C = tools_common()
    levers = list(registry)
    model_lock = threading.Lock()
    workers = max(1, int(config["cores"]))
    rows = {r["image_id"]: r for r in images}
    s001, s001_lock = {}, threading.Lock()

    def default_call(ident: str) -> tuple[dict, Path, str | None]:
        """The image's call at the defaults (S001), once per image: a routed image's second pass reuses it. A failed visual
        QC (zero organoids, implausible coverage) comes back as `error` with the mask delivered; the mask stays the
        baseline and the complaint is recorded (the per-image path's convention)."""
        with s001_lock:
            if ident in s001:
                return s001[ident]
        folder = run / "segment" / ident / "defaults"
        folder.mkdir(parents=True, exist_ok=True)
        image = within(prep / "public", rows[ident]["segment_file"])
        with model_lock:
            initial = tool.execute(image=str(image), image_id=ident, **fixed, query_cache_dir=str(folder), **defaults)
        qc_error = None
        if isinstance(initial, dict) and initial.get("mask_path") and initial.get("error"):
            qc_error = str(initial["error"])
        else:
            initial = pipeline.check_result(initial, "mask_path")
        write_json(run / "segment" / ident / "defaults.json", initial)
        target = folder / f"cell_mask_{ident}.tif"
        tifffile.imwrite(target, np.load(initial["mask_path"]).astype(np.uint32))
        with s001_lock:
            s001[ident] = (initial, target, qc_error)
        return s001[ident]

    def run_one(spec_: dict, options: dict | None) -> tuple:
        pipeline.remaining(config)
        ident = spec_["image_id"]
        mode = "cohort" if (options or {}).get("cohort_levers") is not None else (
            "routed" if (options or {}).get("start_params") is not None else "search")
        initial, s001_mask, _ = default_call(ident)
        item = {"image_id": ident, "path": spec_["path"], "s001_record": initial, "s001_mask": str(s001_mask),
                "s001_overlay": initial.get("overlay_path"), "request": spec.tasks["segment"]}
        folder = run / "segment" / ident
        record, res = segment_resolution(C, tool, item, levers, defaults, engine, folder / mode,
                                         run / "logs" / "segment" / mode, model_lock, verifier_options=options,
                                         tool_name=name, fixed=fixed)
        record = {**record, "mode": mode, "verifier_options": options}
        write_json(folder / ("COHORT_SEARCH.json" if mode == "cohort" else "SEARCH.json"), record)
        pipeline.record_search(run, "segment", ident, record)
        print(f"SEGMENT_{mode.upper()} {ident} calls={record['n_calls']} levers={record['levers']}", flush=True)
        return [res], record, res.status, res.fault

    specs = [{"image_id": r["image_id"], "path": str(within(prep / "public", r["segment_file"]))} for r in images]
    runs, cohort = segmentation_cohort.run(
        specs, run_one, lambda items, fn: pipeline._parallel(items, fn, min(workers, len(items) or 1)),
        lambda s: segmentation_cohort.preflight_read(s["path"], engine, s["image_id"]),
        defaults, levers, workers=workers, tool_name=name)
    write_json(run / pipeline.COHORT_RECORD, {**cohort, "implementation": IMPLEMENTATION, "workers": workers,
                                              "defaults": defaults, "organoid_selective_review": False})
    outputs = []
    for s, (_, record, _, _) in zip(specs, runs):
        ident = s["image_id"]
        target = run / "segment" / ident / f"cell_mask_{ident}.tif"
        _, s001_mask, qc_error = s001[ident]
        if record["levers"]:
            if not record.get("mask_npy"):
                raise RuntimeError(f"selected segmentation has no mask: {ident}")
            mask = np.load(record["mask_npy"])
        else:
            mask = tifffile.imread(s001_mask)
        tifffile.imwrite(target, mask.astype(np.uint32))
        if mask.shape != spec.image_shape or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0):
            raise ValueError(f"invalid segmentation mask: {ident}")
        n_cells = int(np.count_nonzero(np.unique(mask) > 0))
        if not n_cells and not spec.allow_empty_masks:
            raise RuntimeError(f"empty segmentation: {ident}; recorded failure, not silently omitted")
        params = {**defaults, **record["levers"]}
        outputs.append({"image_id": ident, "mask": str(target.resolve()), "sha256": digest(target),
                        "min_area": params["min_size"], "n_cells": n_cells,
                        "parameter_source": cohort["parameter_source"][ident],
                        **({"defaults_qc_error": qc_error} if qc_error else {})})
    write_json(run / "segmentation.json", outputs)


def install() -> None:
    """Make `pipeline.STAGES['segment']` this cohort path in the current process (a stage worker)."""
    pipeline.STAGES["segment"] = segmentation

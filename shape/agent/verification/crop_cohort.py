"""One crop VRR over a cohort of images -- the cropper's counterpart of `segmentation_cohort`.

`CropAgenticVerifier` (CARD.md "Agentic VRR") runs once on the cohort's input (`verifiers.crop.input_record` with
`images`): its image-level levers are shared by every image and its cell rules act per cell, in any image. A call
measures every image with that image's cells' `cell_overrides`, each image in its own directory
(`<output_subdir>/<image_key>`), writing only the panel cells' files (`write_crops`, after crop v10's mask-only
measurement), and one merged metadata file is the call's result for the verifier. Passing crops (crop v10's
standard: target_retention at least the floor plus the VoI margin, `pass_retention`) are held (`keep_passing`), each crop is delivered from its best observation over every finished call (`delivery_policy`,
`crop_trajectory.CropTrajectory`), and the delivery writes the delivered crops with the tool (`materialize`).
"""
from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import json
import hashlib
import threading
from pathlib import Path
import time
from typing import Any, Callable, Mapping, Optional

TOOL = "Single_Cell_Cropper_Tool"
# crop_trajectory's policy under the cropper's own retention floor (0.90, the guard's): each crop's best observation,
# so never worse than the chosen call's crop of it
DELIVERY_POLICY = "quality_only"


def _image_name(image: Mapping[str, Any]) -> str:
    """The image name the tool puts in its cell ids (`source_image_id`, else the image file's stem)."""
    return image.get("source_image_id") or Path(image["original_image"]).stem


def tool_runner(run_image: Callable[[dict, dict], dict], images: list[Mapping[str, Any]], shared: Mapping[str, Any],
                workers: int = 1) -> Callable[[dict], dict]:
    """A cohort call: `run_image(image_kwargs, params)` (the tool on one image; the image's kwargs over `shared`,
    without the keys the call's params set) for every image, in parallel (`workers`), with that image's cells'
    cell_overrides and its own output_subdir; the results merged into one
    metadata file next to the images' directories (cell_metadata concatenated, statistics summed, parameters with
    the whole cell_overrides). A failed image fails the call (`error`)."""
    from .verifiers.crop import _safe_name

    group = shared.get("group") or "default"
    cache, fingerprints, cache_lock = {}, {}, threading.Lock()

    def fingerprint(path):
        path = Path(path)
        info = path.stat()
        version=(str(path.resolve()),info.st_size,info.st_mtime_ns)
        with cache_lock:
            digest=fingerprints.get(version)
        if digest is None:
            digest=hashlib.sha256(path.read_bytes()).hexdigest()
            after=path.stat()
            if (after.st_size,after.st_mtime_ns)!=(info.st_size,info.st_mtime_ns):
                raise ValueError('crop input changed while fingerprinting')
            with cache_lock:fingerprints[version]=digest
        return [*version,digest]

    def runner(params: Mapping[str, Any]) -> dict:
        started = time.time()
        overrides = params.get("cell_overrides") or {}
        base = {k: v for k, v in params.items() if k not in ("cell_overrides", "output_subdir")}
        # Index once, avoiding an O(images*objects) prefix scan per cohort call.
        indexed={}
        for cid,setting in overrides.items():
            indexed.setdefault(cid.rsplit('_cell_',1)[0],{})[cid]=setting

        def one(image):
            name = _image_name(image)
            prefix = f"{image.get('group') or group}_{name}_cell_"
            own = indexed.get(prefix.removesuffix('_cell_'),{})
            sub = f"{params['output_subdir']}/{_safe_name(name)}" if params.get("output_subdir") else _safe_name(name)
            call = {**base, "cell_overrides": own, "output_subdir": sub}
            # the call's params (levers, min_area, ...) override the image's and the shared kwargs
            inputs = {k:v for k,v in {**shared,**dict(image)}.items() if k not in call}
            key = json.dumps(dict(inputs=inputs,params={k:v for k,v in call.items() if k!='output_subdir'},
                files=[fingerprint(image[k]) for k in ('original_image','nuclei_mask')]),sort_keys=True,default=str)
            with cache_lock:
                cached = cache.get(key)
            if cached:
                old, digest = cached
                path = Path(old['cell_crops_metadata_path'])
                if path.exists() and hashlib.sha256(path.read_bytes()).hexdigest()==digest:
                    return _safe_name(name), {**old,'reused_unchanged_image':True}
            result = run_image(inputs,call)
            if result.get('cell_crops_metadata_path') and not result.get('error'):
                path = Path(result['cell_crops_metadata_path'])
                with cache_lock:
                    cache[key] = dict(result),hashlib.sha256(path.read_bytes()).hexdigest()
            return _safe_name(name),result

        from shape.utils.concurrency import ordered_map
        parts = ordered_map(one,images,max(1,int(workers)))
        errors = [f"{key}: {r.get('error') or 'no metadata'}" for key, r in parts
                  if not isinstance(r, Mapping) or r.get("error") or not r.get("cell_crops_metadata_path")]
        if errors:
            return {"error": "; ".join(errors)[-2000:], "execution_status": "error"}
        cell_metadata, statistics, parameters = [], {}, None
        for _, r in parts:
            with open(r["cell_crops_metadata_path"], "r") as fh:
                meta = json.load(fh)
            cell_metadata += meta.get("cell_metadata") or []
            for k, v in (meta.get("statistics") or r.get("processing_statistics") or {}).items():
                if isinstance(v, (int, float)) and not isinstance(v, bool):
                    statistics[k] = statistics.get(k, 0) + v
            parameters = parameters or dict(meta.get("parameters") or {})
        folder = Path(parts[0][1]["cell_crops_metadata_path"]).parent.parent
        # A reused first image may point into an earlier call: never overwrite
        # that candidate's merged metadata when only another image changed.
        call_id=hashlib.sha256(json.dumps(dict(params),sort_keys=True,default=str).encode()).hexdigest()[:16]
        path = folder / f"cohort_crops_metadata_{call_id}.json"
        path.write_text(json.dumps({"cell_metadata": cell_metadata, "statistics": statistics,
                                    "parameters": {**(parameters or {}), "cell_overrides": overrides},
                                    "images": {key: r["cell_crops_metadata_path"] for key, r in parts}}, default=str))
        return {"cell_crops_metadata_path": str(path), "processing_statistics": statistics,
                "cell_count": len(cell_metadata), "seconds": time.time() - started,
                "reused_images": sum(bool(r.get('reused_unchanged_image')) for _,r in parts),
                "image_results": [{"image_key": key, "cell_crops_metadata_path": r["cell_crops_metadata_path"]}
                                  for key, r in parts]}
    return runner


def run(run_image: Callable[[dict, dict], dict], images: list[Mapping[str, Any]], *, llm_engine=None,
        request: Optional[str] = None, task=None, workers: int = 1, keep_passing: bool = True,
        delivery_policy: Optional[str] = DELIVERY_POLICY, rule_first: bool = False,
        delivery_objective: str = 'pass_then_dominance', progress_path: str | None = None,
        max_rounds: int = 8, **shared):
    """The cohort's crop stage: `Verifier.resolve_managed_stage` on the crop stage with trial 0 the cohort's input
    (`images`: one tool-kwargs dict per image -- original_image, nuclei_mask, optionally source_image_id --;
    `shared`: kwargs every image takes, e.g. min_area, margin, group), the cohort tool runner (`workers` images in
    parallel per call; the verifier runs cores // workers calls at a time, `threads_per_call`), `task` the crop
    stage's `CropTask` (cores, deadline, brief). The search strategy is the crop stage's own switch
    (`SHAPE_SEARCH_STRATEGY_CROP`, as for one image). Returns the StageResolution."""
    from shape.agent.verifier import Verifier

    from .stage_map import stage_by_name
    from .verifiers.crop import CropVerifier, _safe_name, input_record

    by_key = {_safe_name(_image_name(image)): image for image in images}
    if not 2<=max_rounds<=8:raise ValueError('crop cohort budget must be in [2,8]')
    if rule_first and max_rounds<3:raise ValueError('rule-first crop needs baseline, rule and decision rounds')

    def materialize(key: str, params: dict) -> dict:
        """The tool on one image of the cohort (by image key) with `params` -- the delivery's writing run."""
        return run_image({k: v for k, v in {**shared, **dict(by_key[key])}.items() if k not in params}, dict(params))

    base_kwargs = {**shared, "images": [dict(image) for image in images]}
    # crop v10's pass standard: a passing crop keeps the floor plus the VoI margin of its target (0.95)
    options = {"keep_passing": keep_passing, "delivery_policy": delivery_policy, "threads_per_call": max(1, int(workers)),
               "rule_first":rule_first,"delivery_objective":delivery_objective,
               "progress_path":progress_path,
               "budget":max_rounds,
               "materialize": materialize,
               "pass_retention": CropVerifier._TARGET_RETENTION_FLOOR + CropVerifier._VOI_RETENTION_MARGIN}
    if delivery_objective=='balanced_v2':
        options.update(score_key='mean_balanced_quality',
            objectives=('mean_balanced_quality','mean_target_retention','mean_centered'),
            key_readouts=('mean_balanced_quality','mean_target_dominance','mean_neighbor_contamination',
                          'mean_target_retention','mean_centroid_offset'))
    return Verifier().resolve_managed_stage(
        stage_by_name("crop"), tool_runner(run_image, images, shared, workers), {}, input_record(base_kwargs),
        base_kwargs=base_kwargs, llm_engine=llm_engine, tool_name=TOOL, request=request,
        upstream_ctx={"crop_task": task} if task is not None else {}, verifier_options=options)

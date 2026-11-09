"""Deliver the merged SHAPE crop cohort into the case's raw scientific crop boundary."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

from ..runtime import artifacts as data


def run(prep, out, config, rows):
    from benchmarks.full_CPJUMP1.harness import pipeline
    from shape.agent.verification import crop_cohort
    from shape.agent.verification.verifiers.crop import CropTask, _safe_name
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool

    masks = {r["image_id"]: r for r in data.read_json(out / "segmentation.json")}
    if set(masks) != {r["image_id"] for r in rows}:
        raise ValueError("segmentation does not match scheduled crop images")
    spec = pipeline.dataset(config)
    active_rows = [row for row in rows if not (spec.allow_empty_masks and masks[row["image_id"]]["n_cells"] == 0)]
    empty_rows = [row for row in rows if row not in active_rows]
    if not active_rows:
        raise RuntimeError("all masks have zero objects; no crop cohort or model input is available")
    images = []
    for row in active_rows:
        mask = masks[row["image_id"]]
        if data.digest(Path(mask["mask"])) != mask["sha256"]:
            raise ValueError("segmentation changed before cohort cropping")
        images.append({"original_image": str(data.within(prep / "public", row["file"])),
                       "nuclei_mask": mask["mask"], "source_image_id": row["image_id"],
                       "min_area": mask["min_area"], "group": "default"})
    cores = max(1, int(config["cores"]))
    workers = min(16, cores, len(images))
    tool = Single_Cell_Cropper_Tool()
    from shape.tools.single_cell_cropper.geometry_cache import configure
    geometry_cache=configure(tool)

    def run_image(image, params):
        pipeline.remaining(config)
        return tool.execute(query_cache_dir=str(out / "crop_cohort"), build_archive=False, **image, **params)

    resolution = crop_cohort.run(run_image, images, llm_engine=pipeline.engine_for(config, "crop"),
                                workers=workers,rule_first=True,delivery_objective='balanced_v2',
                                progress_path=str(out/'CROP_VRR_PROGRESS.json'),
                                max_rounds=max(3,int(config['smoke_policy']['shape_search_rounds'])) if config.get('smoke_policy') else 8,
                                task=CropTask(cores=cores, deadline=config["deadline"],
                                brief=pipeline.dataset(config).tasks["crop"]))
    details = resolution.details or {}
    details['native_geometry_cache']=dict(geometry_cache)
    data.write_json(out / "crop_COHORT.json", {"status": resolution.status, "reason": resolution.reason,
                                               "params": resolution.params, "details": details})
    paths = details.get("delivered_image_metadata") or {}
    if set(paths) != {_safe_name(row["image_id"]) for row in active_rows}:
        raise RuntimeError("cohort crop did not deliver every scheduled image")
    rejected = set((details.get("mask_check") or {}).get("rejected_cells") or [])
    spec = pipeline.dataset(config)

    def materialize(row):
        import tifffile
        ident = row["image_id"]
        metadata_path = Path(paths[_safe_name(ident)])
        if not metadata_path.resolve().is_relative_to(out.resolve()):
            raise ValueError("crop metadata outside run")
        crops = data.read_json(metadata_path)["cell_metadata"]
        if any(not Path(c["crop_path"]).is_file() for c in crops):
            raise ValueError("cohort delivered missing crop files")
        own_rejected = rejected & {c["cell_id"] for c in crops}
        folder = out / "crop" / ident
        data.write_json(folder / "result.json", {"cell_crops_metadata_path": str(metadata_path)})
        kept, excluded = pipeline.retained_crops(
            pipeline.raw_stack(data.within(prep / "public", row["file"]), spec), row,
            crops, own_rejected, folder, spec,
            tifffile.imread(masks[ident]["mask"]) if spec.mask_label else None)
        print(f"CROP {ident} retained={len(kept)} excluded={len(excluded)}", flush=True)
        return kept, excluded

    with ThreadPoolExecutor(max_workers=workers) as pool:
        parts = list(pool.map(materialize, active_rows))
    for row in empty_rows:
        ident = row["image_id"]
        data.write_json(out/"crop"/ident/"result.json", dict(status="no_objects", cell_count=0,
            image_id=ident, reason="delivered mask has zero detections; no biological absence claim"))
    data.write_json(out/"CROP_NO_DETECTIONS.json", [row["image_id"] for row in empty_rows])
    data.write_json(out / "cells.json", [c for kept, _ in parts for c in kept])
    data.write_json(out / "excluded_cells.json", [c for _, excluded in parts for c in excluded])

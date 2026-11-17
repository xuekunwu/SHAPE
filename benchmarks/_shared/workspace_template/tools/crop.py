"""Single-cell crops with MaskCrop: SHAPE's Single_Cell_Cropper_Tool on the masks of segment calls.

    python tools/crop.py --describe
    python tools/crop.py --name NAME (--masks S<NNN> | --masks-map FILE.json)
                         [--images all | --images ID,ID,... | --images-file FILE] [--set lever=value ...]
                         [--per-image FILE.json] [--cell-overrides FILE.json] [--config FILE.json]
                         [--workers N] [--note "why"] [--dry-run]

margin. This CLI's default is the CARD's null, the cropper's own rule (fraction mode, margin_frac 0.20, the
margin floor by mask file name), as SHAPE's own chain (shape/agent/executor.py) and its crop verifier use it:
the control and SHAPE start from the same crop. The agent may set null or any value in the CARD's range.

Mask file names. With `margin=null` (and `min_area=null`) the cropper resolves them by the mask FILE
NAME ("cell_mask" -> margin 1 px; "organoid_mask" -> 10 px; any other name -> 25 px). In SHAPE's chain
it receives the segmenter's `mask_path`, `cell_mask_<image_id>_a<k>.npy`. A segment call stores
`masks/<image_id>.tif` (the layout the chain verification reads), so a crop call hands the cropper a
hard link (a copy where the filesystem has none) named `cell_mask_<image_id>.tif`: a null margin
resolves as it would for the segmenter's own file, and the label array is the same (the cropper reads
a .npy or a .tif mask into uint16 either way).

min_area. What counts as a cell is defined once, at segmentation: SHAPE's chain passes the delivering
segmentation's `min_size` as the cropper's `min_area` (shape/agent/executor.py; the cropper CARD), so a
crop call passes each image's `min_size` from its segment call's per_image.csv.
"""
from __future__ import annotations

import argparse
import json
import os
import re
import shutil
import sys
import time
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TOOL = "crop"
GROUP = "default"
TOOL_CELL_RE = re.compile(r"^default_(?P<image>.+)_cell_(?P<index>\d+)$")
# result / metadata keys that are paths into the tool's cache or objects (removed after the outputs are moved)
RESULT_DROP = ("cell_crop_objects", "cell_crops_metadata_path", "crops_zip_path", "visual_outputs",
               "cell_crops", "cell_metadata")
CROP_PATH_KEYS = ("crop_path", "mask_path", "baseline_crop_path", "baseline_mask_path")
CROP_COLUMNS = ["cell_id", "image_id", "tool_cell_id", "mask_label", "crop_path", "sha256", "area",
                "original_bbox", "crop_bbox", "centroid", "margin_mode", "recenter", "crop_h", "crop_w"]
STAT_KEYS = ("total_objects", "filtered_by_area", "filtered_by_border", "invalid_crop_data")

USAGE = """\
## Usage

    python tools/crop.py --name NAME (--masks S<NNN> | --masks-map FILE.json)
                         [--images all | --images ID,ID,... | --images-file FILE] [--set lever=value ...]
                         [--per-image FILE.json] [--cell-overrides FILE.json] [--config FILE.json]
                         [--workers N] [--note "why"] [--dry-run]

One call runs `Single_Cell_Cropper_Tool.execute(original_image=..., nuclei_mask=<the mask>,
source_image_id=<image_id>, group="default", <levers>, output_format="png", cell_overrides=...)` per image.
- `--masks S<NNN>`: every image that segment call wrote a mask for; `--masks-map FILE.json`:
  `{"img_0001": "S001", "img_0002": "S004", ...}` (one segment call per image). `--images` /
  `--images-file` keep a subset of those images.
- `--set lever=value`: a lever for every image (JSON literal values); `--per-image FILE.json`:
  `{"img_0001": {"margin_mode": "tight"}, ...}` on top of the global levers.
- `--cell-overrides FILE.json`: `{"img_0001": {"default_img_0001_cell_0003": {"margin_mode": "mask_bbox",
  "auto_tighten": true}, ...}, ...}` -- the per-cell keys of the table above, keyed by the cropper's own
  cell id (crops.csv `tool_cell_id`; it depends only on the mask: the object's index in label order).
  Values: margin_mode fraction | tight | mask_bbox, recenter true | false, auto_tighten true | false
  (only for a cell in mask_bbox mode), min_containment [0.9, 1]; `{"reset": true}` alone = no keys.
- `--config FILE.json`: `{"set": {...}, "per_image": {...}, "cell_overrides": {...}}` instead of the flags.
- `--workers N`: N CPU processes, each with its own tool instance (default 1).
- `--dry-run`: validate and print the resolved call; nothing is run or written.
- `margin`: this CLI's default is the CARD's null, as in SHAPE's own chain (shape/agent/executor.py): the
  tool's rule by mask file name (the masks are named as in SHAPE's chain, `cell_mask_...`: a 1 px floor).
  Any value in the Range is valid.
- `min_area` is not a lever: each image's `min_size` in the segment call that delivers its mask (what
  counts as a cell is the segmentation's), as SHAPE's chain passes it. per_image.csv shows both.

Writes calls/C<NNN>_<NAME>/: `crops/<image_id>/<cell_id>.png` (a multi-channel image gives .tiff crops),
`crops.csv` (cell_id = <image_id>_<mask call>_<mask label, 5 digits>, image_id, tool_cell_id, mask_label,
crop_path relative to the call directory, sha256, area, original_bbox, crop_bbox, centroid, margin_mode,
recenter, crop_h, crop_w, and the tool's other per-crop values: its qc readouts as qc_<name>, ...),
`per_image.csv` (image_id, mask_call, status, n_crops, seconds, the levers used, the min_area passed, the
margin and min_area the tool resolved, the tool's object counts, error), `tool_results.jsonl` (the tool's
returned values per image), meta.json, result.json, tool.log. An embed call takes the crops with
--crops C<NNN> or --crops-map."""


def describe(repo: Path) -> str:
    levers = C.tool_levers(TOOL, repo)
    note = ("This CLI's default `margin` is the CARD's null, as in SHAPE's own chain (shape/agent/executor.py): "
            "the tool's rule by mask file name (a 1 px floor for these cell_mask masks).")
    return "\n\n".join([C.card_section(repo, C.CARD_DIRS[TOOL], "## Levers"), note,
                        "## Lever registry (validated by this CLI)\n\n" + C.registry_table(levers), USAGE])


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="crop.py", description="MaskCrop single-cell crops (SHAPE Single_Cell_Cropper_Tool)")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--masks")
    ap.add_argument("--masks-map")
    ap.add_argument("--images")
    ap.add_argument("--images-file")
    ap.add_argument("--set", nargs="+", action="extend", default=[])
    ap.add_argument("--per-image")
    ap.add_argument("--cell-overrides")
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def mask_labels(path: Path) -> list[int]:
    """The mask's object labels in the cropper's enumeration order: it reads the mask as uint16 and
    labels connected components only when the mask is binary (at most one object value), then takes
    regionprops in label order; its cell_<NNNN> is the index in this list."""
    import numpy as np
    import tifffile
    from skimage.measure import label

    mask = np.asarray(tifffile.imread(str(path)))
    if mask.ndim > 2:
        mask = mask.squeeze()
    mask = mask.astype(np.uint16)
    labeled = mask if len(np.unique(mask)) > 2 else label(mask)
    return [int(v) for v in np.unique(labeled) if v != 0]


def _from_file_or_config(flag_path: str | None, config: dict, key: str, config_bytes: bytes | None,
                         files: dict, rel: str, what: str):
    if flag_path and key in config:
        raise C.Invalid(f"give {key} once: {what} or --config's {key}")
    if flag_path:
        data = C.read_input_file(flag_path, what)
        files[rel] = data
        return C.parse_json(data, f"{what} {flag_path}"), C.sha256_bytes(data)
    if key in config:
        return config[key], C.sha256_bytes(config_bytes)
    return None, None


def check_cell_overrides(raw, known: set[str], image_modes: dict[str, str]) -> dict[str, dict]:
    """{image_id: {tool_cell_id: {key: value}}} validated (the cropper's per-cell keys, the values the
    crop verifier accepts); a `reset` entry leaves the cell without keys. Raises C.Invalid."""
    if not isinstance(raw, dict):
        raise C.Invalid("cell_overrides: expected a JSON object {image_id: {tool_cell_id: {key: value}}}")
    out = {}
    for image_id, cells in raw.items():
        if image_id not in known:
            raise C.Invalid(f"cell_overrides: unknown image {image_id!r}")
        if not isinstance(cells, dict) or not cells:
            raise C.Invalid(f"cell_overrides[{image_id}]: expected a non-empty object {{tool_cell_id: {{key: value}}}}")
        for cell_id, keys in cells.items():
            match = TOOL_CELL_RE.match(str(cell_id))
            if not match or match.group("image") != image_id:
                raise C.Invalid(f"cell_overrides[{image_id}]: {cell_id!r} is not a tool_cell_id of this image "
                                f"(default_{image_id}_cell_NNNN, crops.csv tool_cell_id)")
            if not isinstance(keys, dict) or not keys:
                raise C.Invalid(f"cell_overrides[{image_id}][{cell_id}] sets no per-cell key")
            unknown = sorted(set(keys) - set(C.CELL_KEYS))
            if unknown:
                raise C.Invalid(f"cell_overrides[{image_id}][{cell_id}]: unknown key(s) {unknown}; "
                                f"the per-cell keys are {list(C.CELL_KEYS)}")
            values = {k: C.check_value(C.CELL_KEYS[k], v) for k, v in keys.items()}
            if "reset" in values:
                if len(values) > 1:
                    raise C.Invalid(f"cell_overrides[{image_id}][{cell_id}]: reset is true and alone")
                continue
            mode = values.get("margin_mode", image_modes.get(image_id))
            if values.get("auto_tighten") and mode != "mask_bbox":
                raise C.Invalid(f"cell_overrides[{image_id}][{cell_id}]: auto_tighten acts only in mask_bbox mode "
                                f"(the cell resolves to {mode})")
            out.setdefault(image_id, {})[cell_id] = values
    return out


def plan(args, cfg: dict) -> dict:
    """Validate everything and verify the masks; raises C.Invalid or C.Mismatch."""
    name = C.validate_name(args.name)
    if args.workers < 1:
        raise C.Invalid("--workers must be at least 1")
    config, config_bytes = C.read_config_file(args.config, ("set", "per_image", "cell_overrides"))
    levers = C.tool_levers(TOOL, cfg["repo"])
    params, changed = C.resolve_levers(levers, C.given_levers(config, args.set))
    rows = C.read_images(cfg)
    by_id = {r["image_id"]: r for r in rows}
    files = {}
    if config_bytes is not None:
        files["inputs/config.json"] = config_bytes
    per_image_raw, per_image_sha = _from_file_or_config(args.per_image, config, "per_image", config_bytes, files,
                                                        "inputs/per_image.json", "--per-image")
    per_image, _ = C.resolve_per_image(levers, per_image_raw or {}, set(by_id))
    overrides_raw, overrides_sha = _from_file_or_config(args.cell_overrides, config, "cell_overrides",
                                                        config_bytes, files, "inputs/cell_overrides.json",
                                                        "--cell-overrides")

    # the masks: the segment calls, their per_image.csv (verified) and the images with a mask
    if bool(args.masks) == bool(args.masks_map):
        raise C.Invalid("give --masks S<NNN> or --masks-map FILE.json")
    if args.masks:
        wanted = None
        call_ids = [args.masks]
    else:
        data = C.read_input_file(args.masks_map, "--masks-map")
        wanted = C.parse_json(data, f"--masks-map {args.masks_map}")
        if not isinstance(wanted, dict) or not wanted or not all(isinstance(v, str) for v in wanted.values()):
            raise C.Invalid("--masks-map: expected a JSON object {image_id: \"S<NNN>\"}")
        unknown = sorted(set(wanted) - set(by_id))
        if unknown:
            raise C.Invalid(f"--masks-map: unknown image(s) {unknown[:5]}")
        files["inputs/masks_map.json"] = data
        call_ids = sorted(set(wanted.values()))
    producers = {cid: C.producer(cfg, cid, ("segment",)) for cid in call_ids}
    with_mask, min_size_of = {}, {}
    for cid, prod in producers.items():
        seg_rows = [r for r in C.read_csv(prod.verify("per_image.csv")) if r["status"] == "done"]
        with_mask[cid] = {r["image_id"] for r in seg_rows}
        # the segmentation's object definition, as the segmenter read it (int); see the module docstring
        min_size_of[cid] = {r["image_id"]: int(float(r["min_size"])) for r in seg_rows if r.get("min_size")}
    if wanted is None:
        mask_of = {i: args.masks for i in with_mask[args.masks]}
    else:
        missing = sorted(i for i, cid in wanted.items() if i not in with_mask[cid])
        if missing:
            raise C.Invalid(f"--masks-map: no mask for {missing[:5]} in the call named for it")
        mask_of = dict(wanted)
    candidates = [r["image_id"] for r in rows if r["image_id"] in mask_of]
    selected, image_files = C.select_images([r["image_id"] for r in rows], args.images, args.images_file)
    files.update(image_files)
    if args.images or args.images_file:
        no_mask = [i for i in selected if i not in mask_of]
        if no_mask:
            raise C.Invalid(f"no mask for {no_mask[:5]} in {', '.join(call_ids)}")
    images = [i for i in candidates if i in set(selected)]
    if not images:
        raise C.Invalid("no image with a mask selected")

    levers_of = {i: {**params, **per_image.get(i, {})} for i in images}
    modes = {i: per_image.get(i, {}).get("margin_mode", params["margin_mode"]) for i in by_id}
    overrides = check_cell_overrides(overrides_raw, set(by_id), modes) if overrides_raw is not None else {}
    overrides = {i: v for i, v in overrides.items() if i in levers_of}
    masks = {i: producers[mask_of[i]].verify(f"masks/{i}.tif") for i in images}
    for image_id, cells in overrides.items():
        n_objects = len(mask_labels(masks[image_id]))
        for cell_id in cells:
            if int(TOOL_CELL_RE.match(cell_id).group("index")) >= n_objects:
                raise C.Invalid(f"cell_overrides[{image_id}]: {cell_id} does not exist (the mask of {image_id} in "
                                f"{mask_of[image_id]} has {n_objects} objects)")

    applied_per_image = {i: v for i, v in per_image.items() if i in levers_of}
    changed_per_image = {k for v in applied_per_image.values() for k, x in v.items()
                         if not C._same(x, C.normal_default(levers[k]))}
    set_levers = set(changed) | changed_per_image | ({"cell_overrides"} if overrides else set())
    tasks = [(i, str(C.image_file(cfg, by_id[i])), str(masks[i]), mask_of[i], min_size_of[mask_of[i]].get(i),
              levers_of[i], overrides.get(i)) for i in images]
    return {"name": name, "params": params, "set": changed, "set_levers": sorted(set_levers), "images": images,
            "mask_of": {i: mask_of[i] for i in images}, "per_image": applied_per_image, "per_image_sha256": per_image_sha,
            "overrides": overrides, "overrides_sha256": overrides_sha, "tasks": tasks, "files": files,
            "min_area": {t[0]: t[4] for t in tasks}}


# ---------------------------------------------------------------- the work (in this process or a worker)

_STATE: dict = {}


def init_worker(cfg: dict, call_dir: str, log: str | None) -> None:
    if log:
        C.redirect_process_output(Path(log))
    C.setup(cfg, gpu=False)
    _STATE["tool"] = C.tool_class("Single_Cell_Cropper_Tool", cfg)()
    _STATE["dir"] = Path(call_dir)


def _crop_row(meta: dict, image_id: str, mask_call: str, labels: list[int], call_dir: Path) -> dict:
    index = int(TOOL_CELL_RE.match(meta["cell_id"]).group("index"))
    label = labels[index]
    cell_id = f"{image_id}_{mask_call}_{label:05d}"
    source = Path(meta["crop_path"])
    rel = f"crops/{image_id}/{cell_id}{source.suffix}"
    os.replace(source, call_dir / rel)
    size = list(meta.get("crop_size") or [None, None])
    row = {"cell_id": cell_id, "image_id": image_id, "tool_cell_id": meta["cell_id"], "mask_label": label,
           "crop_path": rel, "sha256": C.sha256_file(call_dir / rel), "area": meta.get("area"),
           "original_bbox": meta.get("original_bbox"), "crop_bbox": meta.get("crop_bbox"),
           "centroid": meta.get("centroid"), "margin_mode": meta.get("margin_mode"), "recenter": meta.get("recenter"),
           "crop_h": size[0], "crop_w": size[1]}
    for key, value in meta.items():          # the tool's other per-crop values
        if key in row or key in CROP_PATH_KEYS or key in ("cell_id", "crop_size"):
            continue
        if key == "qc" and isinstance(value, dict):
            row.update({f"qc_{k}": v for k, v in value.items()})
        else:
            row[key] = value
    return row


def failed(task, error: str | None) -> tuple:
    """The result of an image without crops: (per_image row, crop rows, tool record)."""
    image_id, _, _, mask_call, min_area, levers, overrides = task
    return ({"image_id": image_id, "mask_call": mask_call, "status": "failed", "n_crops": 0, "seconds": None,
             **levers, "min_area_passed": min_area, "margin_resolved": None, "min_area_resolved": None,
             "n_cell_overrides": len(overrides or {}), **{k: None for k in STAT_KEYS}, "error": error}, [], {})


def work(item):
    index, task = item
    image_id, image_path, mask_path, mask_call, min_area, levers, overrides = task
    call_dir, tool = _STATE["dir"], _STATE["tool"]
    cache = call_dir / "cache"
    alias = cache / "masks" / f"cell_mask_{image_id}.tif"       # see the module docstring
    started = time.time()
    row = failed(task, None)[0]
    crops, record = [], {}
    try:
        alias.parent.mkdir(parents=True, exist_ok=True)
        try:
            os.link(mask_path, alias)
        except OSError:
            shutil.copyfile(mask_path, alias)
        result = tool.execute(original_image=image_path, nuclei_mask=str(alias), source_image_id=image_id,
                              group=GROUP, min_area=min_area, **levers, output_format="png",
                              query_cache_dir=str(cache), cell_overrides=overrides or None,
                              output_subdir=image_id) or {}
        record = C.json_safe({k: v for k, v in result.items() if k not in RESULT_DROP})
        metadata_path = result.get("cell_crops_metadata_path")
        if result.get("error"):
            row["error"] = str(result["error"])
        elif not metadata_path or not os.path.isfile(metadata_path):
            row["error"] = "the tool wrote no crop metadata"
        else:
            metadata = json.loads(Path(metadata_path).read_text(encoding="utf-8"))
            parameters = metadata.get("parameters") or {}
            record["parameters"] = C.json_safe(parameters)
            stats = metadata.get("statistics") or result.get("processing_statistics") or result.get("statistics") or {}
            row.update(margin_resolved=parameters.get("margin"), min_area_resolved=parameters.get("min_area"),
                       **{k: stats.get(k) for k in STAT_KEYS})
            cell_metadata = metadata.get("cell_metadata") or []
            if cell_metadata:
                labels = mask_labels(alias)
                (call_dir / "crops" / image_id).mkdir(parents=True, exist_ok=True)
                crops = [_crop_row(m, image_id, mask_call, labels, call_dir) for m in cell_metadata]
            row.update(status="done", n_crops=len(crops))
            row["seconds"] = result.get("seconds")
    except Exception as exc:          # noqa: BLE001 -- this image fails, the others go on
        row.update(status="failed", n_crops=0, error=f"{type(exc).__name__}: {exc}")
        crops = []
        C.remove_tree(call_dir / "crops" / image_id)
    finally:
        C.remove_tree(cache / "tool_cache" / image_id)
        if alias.exists():
            alias.unlink()
    if row["seconds"] is None:
        row["seconds"] = round(time.time() - started, 1)
    return index, (row, crops, record)


def run(call: C.Call, p: dict, workers: int) -> tuple[str, dict, dict]:
    (call.dir / "crops").mkdir()
    (call.dir / "cache").mkdir()
    results = C.run_items(call, init_worker, (call.cfg, str(call.dir)), work, p["tasks"], workers, "images",
                          failed)
    per_image, crops, records = [], [], []
    for row, image_crops, record in results:
        per_image.append(row)
        records.append({"image_id": row["image_id"], **record})
        if row["status"] == "done":
            crops += image_crops
    extra = sorted({k for c in crops for k in c} - set(CROP_COLUMNS))
    C.write_csv(call.dir / "crops.csv", crops, CROP_COLUMNS + extra)
    columns = ["image_id", "mask_call", "status", "n_crops", "seconds", *p["params"], "min_area_passed",
               "margin_resolved", "min_area_resolved", "n_cell_overrides", *STAT_KEYS, "error"]
    C.write_csv(call.dir / "per_image.csv", per_image, columns)
    C.write_atomic(call.dir / "tool_results.jsonl", "".join(json.dumps(r) + "\n" for r in C.json_safe(records)))
    C.remove_tree(call.dir / "cache")
    done = [r for r in per_image if r["status"] == "done"]
    summary = {"n_images": len(per_image), "n_done": len(done), "n_failed": len(per_image) - len(done),
               "failed": [r["image_id"] for r in per_image if r["status"] != "done"], "n_crops": len(crops),
               "crops_per_image": C.count_stats([r["n_crops"] for r in done]),
               "n_cell_overrides": sum(len(v) for v in p["overrides"].values())}
    outputs = call.outputs([call.dir / "crops.csv", call.dir / "per_image.csv"])
    return C.call_status(len(done), len(per_image)), summary, outputs


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    args = parser().parse_args(argv)
    if args.describe:
        cfg = C.load_config(optional=True)
        if cfg:
            C.setup(cfg, gpu=False)
        print(describe(C.repo_root(cfg)))
        return 0
    cfg = C.load_config()
    C.setup(cfg, gpu=False)
    try:
        p = plan(args, cfg)
    except (C.Invalid, C.Mismatch) as error:
        return C.finish(error)
    if args.dry_run:
        return C.dry_run(TOOL, p["name"], set=p["set"], set_levers=p["set_levers"], params=p["params"],
                         workers=args.workers, masks=p["mask_of"], min_area=p["min_area"], per_image=p["per_image"],
                         cell_overrides=p["overrides"])
    inputs = {"from_calls": sorted(set(p["mask_of"].values())), "masks": p["mask_of"]}
    meta = {"workers": args.workers,
            "per_image": {"n_images": len(p["per_image"]), "levers": sorted({k for v in p["per_image"].values() for k in v}),
                          "sha256": p["per_image_sha256"], "values": p["per_image"]} if p["per_image"] else None,
            "cell_overrides": {"n_images": len(p["overrides"]), "n_cells": sum(len(v) for v in p["overrides"].values()),
                               "sha256": p["overrides_sha256"], "values": p["overrides"]} if p["overrides"] else None}
    call = C.Call(cfg, TOOL, p["name"], note=args.note, argv=argv, set_=p["set"], set_levers=p["set_levers"],
                  params=p["params"], inputs=inputs, files=p["files"], meta=meta,
                  ledger_extra={"per_image_sha256": p["per_image_sha256"],
                                "cell_overrides_sha256": p["overrides_sha256"]})
    return C.run_call(call, lambda c: run(c, p, args.workers), gpu=False)


if __name__ == "__main__":
    sys.exit(main())

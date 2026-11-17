"""Cell segmentation with Cellpose-SAM: SHAPE's Cell_Segmenter_Tool over a set of images.

    python tools/segment.py --describe
    python tools/segment.py --name NAME [--images all | --images ID,ID,... | --images-file FILE]
                            [--set lever=value ...] [--per-image FILE.json] [--config FILE.json]
                            [--workers N] [--note "why"] [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
from pathlib import Path

if str(Path(__file__).resolve().parent) not in sys.path:
    sys.path.insert(0, str(Path(__file__).resolve().parent))
import _common as C  # noqa: E402

TOOL = "segment"
FIXED = {"model_type": "cpsam", "bsize": 256}      # not levers (the CARD: bsize is fixed at 256)
# result keys that are paths into the tool's cache (removed after the outputs are copied)
PATH_KEYS = ("mask_path", "overlay_path", "output_path", "visual_outputs", "deliverables")

USAGE = """\
## Usage

    python tools/segment.py --name NAME [--images all | --images ID,ID,... | --images-file FILE]
                            [--set lever=value ...] [--per-image FILE.json] [--config FILE.json]
                            [--workers N] [--note "why"] [--dry-run]

One call runs `Cell_Segmenter_Tool.execute(image=..., <levers>, model_type="cpsam", bsize=256)` on
every selected image (default: all images of data/images.csv).
- `--set lever=value`: a lever for every image; the value is a JSON literal (null, true, 45, 0.6).
- `--per-image FILE.json`: `{"img_0001": {"diameter": 60}, ...}`, applied on top of the global levers.
- `--config FILE.json`: `{"set": {...}, "per_image": {...}}` instead of the flags (--set wins per lever).
- `--workers N`: N processes, each with its own tool instance on the GPU (default 1). On a GPU in
  exclusive-process mode (one process per GPU; `tools/_config.json: gpu_exclusive`) only 1 is accepted.
- `--dry-run`: validate and print the resolved call; nothing is run or written.

Writes calls/S<NNN>_<NAME>/: `masks/<image_id>.tif` (uint16 label image, 0 = background) for every
image the tool segmented, `overlays/<image_id>.png` (the tool's overlay), `per_image.csv` (image_id,
status, cell_count, flow_error, segmented_channel, seconds, device = the torch device the tool's model
ran on, the levers used, error), `tool_results.jsonl` (the tool's returned values per image),
meta.json, result.json, tool.log. The result summary lists the devices used and `cpu_fallback` (true
when a tool instance ran on the CPU while CUDA is available on the node; the tool falls back to the CPU
when it cannot use the GPU).
An image is `failed` when the tool reports an error (e.g. no cells or an unusable overlay); it then
has no mask. A crop call takes the masks of a segment call with --masks S<NNN> or --masks-map."""


def describe(repo: Path) -> str:
    levers = C.tool_levers(TOOL, repo)
    return "\n\n".join([C.card_section(repo, C.CARD_DIRS[TOOL], "## Levers"),
                        "## Lever registry (validated by this CLI)\n\n" + C.registry_table(levers), USAGE])


def parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="segment.py", description="Cellpose-SAM segmentation (SHAPE Cell_Segmenter_Tool)")
    ap.add_argument("--describe", action="store_true")
    ap.add_argument("--name")
    ap.add_argument("--images")
    ap.add_argument("--images-file")
    ap.add_argument("--set", nargs="+", action="extend", default=[])
    ap.add_argument("--per-image")
    ap.add_argument("--config")
    ap.add_argument("--workers", type=int, default=1)
    ap.add_argument("--note", default="")
    ap.add_argument("--dry-run", action="store_true")
    return ap


def plan(args, cfg: dict) -> dict:
    """Validate everything; raises C.Invalid."""
    name = C.validate_name(args.name)
    if args.workers < 1:
        raise C.Invalid("--workers must be at least 1")
    if cfg.get("gpu_exclusive") and args.workers > 1:
        raise C.Invalid("this GPU is in exclusive-process mode (one process per GPU): --workers must be 1")
    config, config_bytes = C.read_config_file(args.config, ("set", "per_image"))
    levers = C.tool_levers(TOOL, cfg["repo"])
    params, changed = C.resolve_levers(levers, C.given_levers(config, args.set))
    rows = C.read_images(cfg)
    by_id = {r["image_id"]: r for r in rows}
    images, files = C.select_images([r["image_id"] for r in rows], args.images, args.images_file)
    if args.per_image and "per_image" in config:
        raise C.Invalid("give per_image once: --per-image or --config's per_image")
    per_image_raw, per_image_sha = None, None
    if args.per_image:
        data = C.read_input_file(args.per_image, "--per-image")
        per_image_raw, per_image_sha = C.parse_json(data, f"--per-image {args.per_image}"), C.sha256_bytes(data)
        files["inputs/per_image.json"] = data
    elif "per_image" in config:
        per_image_raw, per_image_sha = config["per_image"], C.sha256_bytes(config_bytes)
    if config_bytes is not None:
        files["inputs/config.json"] = config_bytes
    per_image, _ = C.resolve_per_image(levers, per_image_raw or {}, set(by_id))
    applied = {i: v for i, v in per_image.items() if i in set(images)}
    changed_per_image = sorted({k for v in applied.values() for k, x in v.items()
                                if not C._same(x, C.normal_default(levers[k]))})
    tasks = [(image_id, str(C.image_file(cfg, by_id[image_id])), {**params, **applied.get(image_id, {})})
             for image_id in images]
    return {"name": name, "params": params, "set": changed, "set_levers": sorted(set(changed) | set(changed_per_image)),
            "per_image": applied, "per_image_sha256": per_image_sha, "images": images, "tasks": tasks,
            "files": files}


# ---------------------------------------------------------------- the work (in this process or a worker)

_STATE: dict = {}


def init_worker(cfg: dict, call_dir: str, log: str | None) -> None:
    if log:
        C.redirect_process_output(Path(log))
    C.setup(cfg, gpu=True)
    _STATE["tool"] = C.tool_class("Cell_Segmenter_Tool", cfg)()
    _STATE["dir"] = Path(call_dir)
    try:
        import torch

        _STATE["cuda"] = bool(torch.cuda.is_available())
    except ImportError:
        _STATE["cuda"] = False


def _device(tool) -> str:
    """The torch device the tool's model runs on (the tool's own device without a local model)."""
    device = getattr(getattr(tool, "model", None), "device", None)
    return str(device if device is not None else getattr(tool, "device", "unknown"))


def failed(task, error: str | None) -> tuple:
    """The result of an image without a mask: (per_image row, tool record, mask sha256, CUDA available)."""
    image_id, _, levers = task
    return ({"image_id": image_id, "status": "failed", "cell_count": None, "flow_error": None,
             "segmented_channel": None, "seconds": None, "device": None, **levers, "error": error}, {}, None, None)


def work(item):
    index, task = item
    image_id, image_path, levers = task
    import numpy as np
    import tifffile

    call_dir, tool = _STATE["dir"], _STATE["tool"]
    started = time.time()
    row = failed(task, None)[0]
    row["device"] = _device(tool)
    record, mask_sha, result = {}, None, {}
    try:
        result = tool.execute(image=image_path, **levers, **FIXED, query_cache_dir=str(call_dir / "cache"),
                              image_id=image_id) or {}
        row["device"] = _device(tool)            # execute() may rebuild the model
        record = C.json_safe({k: v for k, v in result.items() if k not in PATH_KEYS})
        row.update(cell_count=result.get("cell_count"), flow_error=result.get("flow_error"),
                   segmented_channel=result.get("segmented_channel"), seconds=result.get("seconds"))
        overlay = result.get("overlay_path")
        if overlay and os.path.isfile(overlay):
            shutil.copyfile(overlay, call_dir / "overlays" / f"{image_id}.png")
        mask_path = result.get("mask_path")
        if result.get("error"):
            row["error"] = str(result["error"])
        elif not mask_path or not os.path.isfile(mask_path):
            row["error"] = "the tool returned no mask file"
        else:
            mask = np.load(mask_path)
            if mask.size and int(mask.max()) > 65535:
                row["error"] = f"{int(mask.max())} labels do not fit a uint16 mask"
            else:
                # The tool's mask_path is the .npy label array; the cropper reads a uint16 .tif the same
                # way (tifffile, then uint16), so the stored mask is that array as uint16.
                target = call_dir / "masks" / f"{image_id}.tif"
                tifffile.imwrite(target, mask.astype(np.uint16), compression="zlib")
                mask_sha = C.sha256_file(target)
                row["status"] = "done"
    except Exception as exc:          # noqa: BLE001 -- this image fails, the others go on
        row["error"] = f"{type(exc).__name__}: {exc}"
    finally:
        for key in ("mask_path", "overlay_path", "output_path"):
            if isinstance(result.get(key), str) and os.path.isfile(result[key]):
                os.remove(result[key])
        for path in result.get("visual_outputs") or []:
            if isinstance(path, str) and os.path.isfile(path):
                os.remove(path)
    if row["seconds"] is None:
        row["seconds"] = round(time.time() - started, 1)
    return index, (row, record, mask_sha, _STATE.get("cuda"))


def run(call: C.Call, p: dict, workers: int) -> tuple[str, dict, dict]:
    for sub in ("masks", "overlays", "cache"):
        (call.dir / sub).mkdir()
    results = C.run_items(call, init_worker, (call.cfg, str(call.dir)), work, p["tasks"], workers, "images",
                          failed)
    levers = list(p["params"])
    rows, outputs, records, cpu_fallback = [], {}, [], False
    for row, record, mask_sha, cuda in results:
        rows.append(row)
        cpu_fallback |= bool(cuda) and str(row["device"]).startswith("cpu")
        records.append({"image_id": row["image_id"], **record})
        if mask_sha:
            outputs[f"masks/{row['image_id']}.tif"] = mask_sha
    columns = ["image_id", "status", "cell_count", "flow_error", "segmented_channel", "seconds", "device", *levers,
               "error"]
    C.write_csv(call.dir / "per_image.csv", rows, columns)
    C.write_atomic(call.dir / "tool_results.jsonl", "".join(json.dumps(r) + "\n" for r in C.json_safe(records)))
    C.remove_tree(call.dir / "cache")
    outputs = {**call.outputs([call.dir / "per_image.csv"]), **outputs}
    done = [r for r in rows if r["status"] == "done"]
    summary = {"n_images": len(rows), "n_done": len(done), "n_failed": len(rows) - len(done),
               "failed": [r["image_id"] for r in rows if r["status"] != "done"],
               "n_cells": sum(int(r["cell_count"] or 0) for r in done),
               "cells_per_image": C.count_stats([r["cell_count"] for r in done]),
               "devices": sorted({str(r["device"]) for r in rows if r["device"]}), "cpu_fallback": cpu_fallback}
    return C.call_status(len(done), len(rows)), summary, outputs


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
    C.setup(cfg, gpu=True)
    try:
        p = plan(args, cfg)
    except C.Invalid as error:
        return C.finish(error)
    if args.dry_run:
        return C.dry_run(TOOL, p["name"], set=p["set"], set_levers=p["set_levers"], params=p["params"],
                         fixed=FIXED, workers=args.workers, images=p["images"], per_image=p["per_image"])
    call = C.Call(cfg, TOOL, p["name"], note=args.note, argv=argv, set_=p["set"], set_levers=p["set_levers"],
                  params=p["params"], inputs={"from_calls": [], "images": p["images"]}, files=p["files"],
                  meta={"fixed": FIXED, "workers": args.workers,
                        "per_image": {"n_images": len(p["per_image"]), "levers": sorted(
                            {k for v in p["per_image"].values() for k in v}), "sha256": p["per_image_sha256"],
                            "values": p["per_image"]} if p["per_image"] else None},
                  ledger_extra={"per_image_sha256": p["per_image_sha256"]})
    return C.run_call(call, lambda c: run(c, p, args.workers), gpu=True)


if __name__ == "__main__":
    sys.exit(main())

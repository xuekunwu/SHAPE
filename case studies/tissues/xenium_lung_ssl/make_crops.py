"""Single-cell morphology crops of a Xenium section: one four-channel uint16 TIFF per segmented cell, `<cell_id>.tiff`
(the `--crops` input of `e2e.py prepare`).

The crop of a cell is the square around its pixels in the Xenium cell label mask (`cells.zarr.zip`, `masks/1`, value =
the cell's index in `cell_id` + 1): the side is the longer side of the label's bounding box (pixels, both ends
included) and the shorter side is centred (the extra pixels split floor before, ceil after). The pixels are the four
`morphology_focus` channels at full resolution, zero outside the image and outside the cell's label. This reproduces
the crops of the published runs pixel for pixel.

    python "case studies/tissues/xenium_lung_ssl/make_crops.py" XENIUM_OUTS CROPS [--workers 16]
"""
from __future__ import annotations

import argparse
from collections import deque
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy import ndimage
import tifffile
import zarr

STRIP, MARGIN = 2048, 256      # rows read per pass; a cell is less than MARGIN pixels tall


def open_zip(path: Path):
    store = getattr(zarr, "ZipStore", None) or zarr.storage.ZipStore     # zarr 2 or zarr 3
    return zarr.open(store(str(path), mode="r"), mode="r")


def cell_ids(cells) -> list[str]:
    """The Xenium cell ID strings ('aaaaadnb-1'): the 8 hex digits of the prefix shifted to a-p, then the suffix."""
    return ["".join(chr(ord("a") + int(h, 16)) for h in f"{int(p):08x}") + f"-{int(s)}" for p, s in np.asarray(cells["cell_id"])]


def iter_crops(outs: Path):
    """(cell_id, crop) for every cell with label pixels, strip by strip from the top of the section."""
    cells = open_zip(Path(outs) / "cells.zarr.zip")
    ids, mask = cell_ids(cells), cells["masks/1"]
    focus = tifffile.TiffFile(Path(outs) / "morphology_focus" / "morphology_focus_0000.ome.tif").series[0]
    # the four morphology_focus files as one CYX array; level=0, as newer tifffile opens the series as its pyramid
    image = zarr.open(focus.aszarr(level=0), mode="r")
    if image.ndim != 3 or image.shape[0] != 4 or tuple(image.shape[1:]) != tuple(mask.shape):
        raise ValueError(f"morphology_focus {image.shape} does not match the cell mask {mask.shape}")
    height, width = mask.shape
    for top in range(0, height, STRIP):
        a, b = max(top - MARGIN, 0), min(top + STRIP + MARGIN, height)
        labels, pixels = np.asarray(mask[a:b]), np.asarray(image[:, a:b])
        for index, box in enumerate(ndimage.find_objects(labels)):
            if box is None or not top <= box[0].start + a < top + STRIP:      # each cell once: the strip of its top row
                continue
            y0, y1, x0, x1 = box[0].start + a, box[0].stop + a, box[1].start, box[1].stop
            ny, nx = y1 - y0, x1 - x0
            side = max(nx, ny)
            oy, ox = y0 - (side - ny) // 2, x0 - (side - nx) // 2
            crop = np.zeros((4, side, side), np.uint16)
            ys, xs = slice(max(oy, 0), min(oy + side, height)), slice(max(ox, 0), min(ox + side, width))
            target = (slice(None), slice(ys.start - oy, ys.stop - oy), slice(xs.start - ox, xs.stop - ox))
            crop[target] = pixels[:, ys.start - a:ys.stop - a, xs]
            own = np.zeros((side, side), bool)
            own[target[1:]] = labels[ys.start - a:ys.stop - a, xs] == index + 1
            crop[:, ~own] = 0
            yield ids[index], crop


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("outs", type=Path, help="the section's Xenium output folder (cells.zarr.zip, morphology_focus/)")
    p.add_argument("crops", type=Path, help="a new or empty output folder")
    p.add_argument("--workers", type=int, default=16, help="parallel TIFF writers")
    args = p.parse_args(argv)
    if args.crops.exists() and any(args.crops.iterdir()):
        raise FileExistsError(f"output folder is not empty: {args.crops}")
    args.crops.mkdir(parents=True, exist_ok=True)
    expected = len(cell_ids(open_zip(args.outs / "cells.zarr.zip")))
    written, pending = 0, deque()
    with ThreadPoolExecutor(args.workers) as pool:
        for cell, crop in iter_crops(args.outs):
            pending.append(pool.submit(tifffile.imwrite, args.crops / f"{cell}.tiff", crop))
            while len(pending) > 64 * args.workers:      # bounded: crops wait in memory only briefly
                pending.popleft().result()
            written += 1
            if written % 20000 == 0:
                print(f"CROPS {written}/{expected}", flush=True)
        for future in pending:
            future.result()
    if written != expected:
        raise RuntimeError(f"{written} crops for {expected} cells: some cells have no label pixels")
    print(f"CROPS_DONE {written} -> {args.crops}")


if __name__ == "__main__":
    main()

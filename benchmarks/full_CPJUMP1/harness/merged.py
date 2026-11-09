"""Import the downloader's verified CYX stacks without merging channels again."""
from __future__ import annotations

import ast
import configparser
import csv
from pathlib import Path
import shutil

import numpy as np
import tifffile

from .data import digest, within


def source_path(root: Path, value: str) -> Path:
    if not value or not isinstance(value, str):
        raise ValueError("missing source path in fields.csv")
    path = Path(value)
    if not path.is_absolute():
        path = within(root, value)
    path = path.resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError(f"source path escapes declared root: {path}")
    if not path.is_file():
        raise FileNotFoundError(path)
    return path


def load_index(root: Path, data: Path, manifest: dict) -> tuple[dict[str, Path], dict]:
    complete = root / "COMPLETE"
    fields = root / "fields.csv"
    provenance = {"mode": "import_merged", "fields_sha256": digest(fields),
                  "complete_sha256": digest(complete), "storage": "independent_copy"}
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_string("[complete]\n" + complete.read_text(encoding="utf-8"))
    required = {"fields": str(len(manifest["sites"])), "axes": "CYX",
                "channels": "1,2,3,4,5", "pixels": "verified"}
    if any(parser["complete"].get(k) != v for k, v in required.items()):
        raise ValueError("merged COMPLETE does not describe the verified five-channel subset")
    by_site = {site["id"]: {} for site in manifest["sites"]}
    for item in manifest["files"]:
        by_site[item["site_id"]][int(item["channel"])] = within(data / "raw", item["relative_path"])
    expected = {tuple(channels[c] for c in range(1, 6)): site for site, channels in by_site.items()}
    result, outputs = {}, set()
    with fields.open(newline="", encoding="utf-8-sig") as handle:
        reader = csv.DictReader(handle)
        columns = ["output", "shape", "dtype", *[f"ch{c}" for c in range(1, 6)]]
        if reader.fieldnames != columns:
            raise ValueError(f"unexpected fields.csv columns; expected {columns}")
        for row in reader:
            if None in row or any(value is None for value in row.values()):
                raise ValueError("malformed fields.csv row")
            if ast.literal_eval(row["shape"]) != (5, 1080, 1080) or row["dtype"] != "uint16":
                raise ValueError("fields.csv must declare (5, 1080, 1080), uint16")
            channels = tuple(source_path(data / "raw", row[f"ch{c}"]) for c in range(1, 6))
            site = expected.get(channels)
            if site is None:
                raise ValueError("fields.csv channels do not match a manifest site in channel order")
            output = source_path(root, row["output"])
            if site in result or output in outputs:
                raise ValueError("duplicate site or merged output in fields.csv")
            result[site] = output
            outputs.add(output)
    if set(result) != set(by_site):
        raise ValueError("fields.csv must cover every manifest site exactly once")
    if digest(fields) != provenance["fields_sha256"] or digest(complete) != provenance["complete_sha256"]:
        raise ValueError("merged index changed while reading")
    return result, provenance


def import_view(source: Path, channels: list[Path], output: Path, segment: Path) -> dict:
    if output.exists() or segment.exists():
        raise FileExistsError("refusing to replace a prepared view")
    before = digest(source)
    with tifffile.TiffFile(source) as image:
        if len(image.series) != 1 or image.series[0].axes != "CYX":
            raise ValueError("merged TIFF must contain exactly one CYX series")
        if image.series[0].shape != (5, 1080, 1080) or image.series[0].dtype != np.uint16:
            raise ValueError("merged TIFF must be five-channel 1080x1080 uint16")
        stack = image.series[0].asarray()
    if stack.shape != (5, 1080, 1080) or stack.dtype != np.uint16:
        raise ValueError("merged TIFF must be five-channel 1080x1080 uint16")
    if len(channels) != 5:
        raise ValueError("exactly five source channels are required")
    for index, path in enumerate(channels):
        raw = tifffile.imread(path)
        if raw.dtype != np.uint16 or not np.array_equal(stack[index], raw):
            raise ValueError(f"merged pixels differ from source channel {index + 1}: {source}")
    output.parent.mkdir(parents=True, exist_ok=True)
    segment.parent.mkdir(parents=True, exist_ok=True)
    # Copy instead of hardlink: later writes in either tree cannot mutate the other's source data.
    shutil.copyfile(source, output)
    if digest(output) != before or digest(source) != before:
        raise ValueError("merged TIFF changed during import")
    tifffile.imwrite(segment, stack[[4, 1]], ome=True, photometric="minisblack",
                     metadata={"axes": "CYX", "Channel": {"Name": ["DNA", "AGP"]}})
    if not np.array_equal(tifffile.imread(segment), stack[[4, 1]]):
        raise ValueError("segmentation view round-trip differs")
    return {"shape": list(stack.shape), "dtype": "uint16", "sha256": before,
            "segment_sha256": digest(segment)}

"""Pinned, authenticated metadata access; never downloads complete image archives."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import io
import json
import math
from pathlib import Path, PurePosixPath, PureWindowsPath
import tarfile
import zlib

REPO = Path(__file__).resolve().parents[2]
PREFIX_BYTES = 2 * 1024 * 1024
DECOMPRESSED_LIMIT = 16 * 1024 * 1024


def sha256(raw: bytes) -> str:
    return hashlib.sha256(raw).hexdigest()


def archive_metadata(prefix: bytes, member_name: str) -> bytes:
    """Read a single pinned member from a bounded prefix; never extract paths."""
    payload = zlib.decompressobj(16 + zlib.MAX_WBITS).decompress(prefix, DECOMPRESSED_LIMIT)
    try:
        with tarfile.open(fileobj=io.BytesIO(payload), mode="r:") as archive:
            for member in archive:
                if member.name == member_name:
                    if not member.isfile() or member.size > PREFIX_BYTES:
                        raise ValueError("metadata member must be a small regular file")
                    return archive.extractfile(member).read()
    except tarfile.ReadError as error:
        raise ValueError("metadata is not fully available in the bounded archive prefix") from error
    raise ValueError("expected metadata member absent from archive prefix")


def fetch_metadata(spec: dict) -> bytes:
    from huggingface_hub import get_token, hf_hub_download, hf_hub_url

    token = get_token()
    if not token:
        raise RuntimeError("Hugging Face login required; use hf auth login or HF_TOKEN")
    if not spec["metadata_in_archive"]:
        path = hf_hub_download(repo_id=spec["repo_id"], repo_type="dataset",
                               revision=spec["revision"], filename=spec["metadata_path"], token=token)
        return Path(path).read_bytes()
    import requests

    url = hf_hub_url(spec["repo_id"], spec["archive"], repo_type="dataset", revision=spec["revision"])
    with requests.get(url, headers={"Authorization": f"Bearer {token}",
                      "Range": f"bytes=0-{PREFIX_BYTES - 1}"}, stream=True, timeout=30) as response:
        if response.status_code not in (200, 206):
            raise RuntimeError(f"Hugging Face archive access failed (HTTP {response.status_code})")
        # Even if the server ignores Range, stop after the bounded prefix.
        prefix = response.raw.read(PREFIX_BYTES)
    return archive_metadata(prefix, spec["metadata_path"])


def inventory(spec: dict) -> dict:
    from huggingface_hub import HfApi

    info = HfApi(token=True).dataset_info(spec["repo_id"], revision=spec["revision"], files_metadata=True)
    if info.sha != spec["revision"]:
        raise ValueError("dataset revision differs from pinned contract")
    files = {f.rfilename: f.size for f in info.siblings}
    if files.get(spec["archive"]) != spec["archive_bytes"]:
        raise ValueError("image archive differs from pinned contract")
    if not spec["metadata_in_archive"] and spec["metadata_path"] not in files:
        raise ValueError("standalone metadata absent from pinned revision")
    return {"status": "access_verified", "repo_id": info.id, "revision": info.sha,
            "private": info.private, "files": files, "image_pixels_verified": False}


def audit(raw: bytes, spec: dict) -> dict:
    digest = sha256(raw)
    if digest != spec["metadata_sha256"]:
        raise ValueError("metadata SHA256 mismatch; use the pinned source, not the older archive CSV")
    reader = csv.DictReader(io.StringIO(raw.decode("utf-8-sig")))
    fields = reader.fieldnames or []
    required = {"image_id", "plate_id", "well_id", "site_id", "relative_path", "metadata_status"}
    flow = spec["study"] == "ctnt_flowcytometry"
    if flow:
        required.update(("flow_sample_id", "ctnt_positive_percent", "donor_id"))
    if len(fields) != len(set(fields)) or not required.issubset(fields):
        raise ValueError("missing or duplicate metadata columns")
    rows = list(reader)
    if len(rows) != spec["expected_images"]:
        raise ValueError("unexpected image count in metadata")
    seen_ids, seen_paths, seen_sites = set(), set(), set()
    wells, assays = defaultdict(list), {}
    for index, row in enumerate(rows, 2):
        if set(row) != set(fields) or any(v is None for v in row.values()):
            raise ValueError(f"malformed CSV row {index}")
        if any(not row[k].strip() for k in required):
            raise ValueError(f"missing required value at CSV row {index}")
        path = row["relative_path"]
        if (PurePosixPath(path).is_absolute() or PureWindowsPath(path).drive
                or "\\" in path or any(p in ("", ".", "..") for p in path.split("/"))):
            raise ValueError(f"unsafe relative image path at CSV row {index}")
        well = (row["plate_id"], row["well_id"])
        site = (*well, row["site_id"])
        if row["image_id"] in seen_ids or path in seen_paths or site in seen_sites:
            raise ValueError(f"duplicate image/path/site at CSV row {index}")
        seen_ids.add(row["image_id"])
        seen_paths.add(path)
        seen_sites.add(site)
        wells[well].append(row)
        if flow:
            value = float(row["ctnt_positive_percent"])
            if not math.isfinite(value) or not 0 <= value <= 100:
                raise ValueError(f"invalid cTnT percentage at CSV row {index}")
            assay = row["flow_sample_id"]
            if assay in assays and assays[assay] != well:
                raise ValueError("one flow assay maps to multiple imaging wells")
            assays[assay] = well
    if len(wells) != spec["expected_wells"]:
        raise ValueError("unexpected plate-qualified well count")
    if flow:
        for records in wells.values():
            for key in ("flow_sample_id", "donor_id", "ctnt_positive_percent"):
                values = {float(r[key]) if key == "ctnt_positive_percent" else r[key] for r in records}
                if len(values) != 1:
                    raise ValueError(f"conflicting {key} within one well")
    return {"status": "metadata_verified", "study": spec["study"],
            "repo_id": spec["repo_id"], "revision": spec["revision"],
            "metadata_path": spec["metadata_path"], "metadata_sha256": digest,
            "images": len(rows), "wells": len(wells), "plates": len({w[0] for w in wells}),
            "images_per_well": dict(sorted(Counter(len(v) for v in wells.values()).items())),
            "metadata_status_counts": dict(Counter(r["metadata_status"] for r in rows)),
            "missing_values": {k: sum(not r[k].strip() for r in rows) for k in fields},
            "label_unit": "well" if flow else "cell",
            "labels_available_in_metadata": flow, "image_pixels_verified": False,
            "end_to_end_ready": False}


def write_json(path: Path, value: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("x", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, indent=2, sort_keys=True)
        handle.write("\n")


def main(spec_path: Path) -> None:
    spec = json.loads(spec_path.read_text(encoding="utf-8"))
    parser = argparse.ArgumentParser(description=spec["study"] + " data admission (not an E2E runner)")
    sub = parser.add_subparsers(dest="command", required=True)
    inspect = sub.add_parser("inspect", help="verify authenticated access at the pinned HF revision")
    inspect.add_argument("--out", type=Path, help="fresh aggregate JSON report")
    fetch = sub.add_parser("fetch-metadata", help="fetch and verify metadata only; no full image download")
    fetch.add_argument("--out", type=Path, required=True, help="fresh private directory outside the repository")
    check = sub.add_parser("audit-metadata", help="verify a local CSV against the pinned contract")
    check.add_argument("--csv", type=Path, required=True)
    check.add_argument("--out", type=Path, help="fresh aggregate JSON report")
    args = parser.parse_args()
    if args.out is not None and (args.out.exists() or args.out.is_symlink()):
        raise FileExistsError("use a fresh output path")
    if args.command == "inspect":
        report = inventory(spec)
    elif args.command == "audit-metadata":
        report = audit(args.csv.read_bytes(), spec)
    else:
        if args.out.resolve().is_relative_to(REPO):
            raise ValueError("private metadata must be stored outside the code repository")
        raw = fetch_metadata(spec)
        report = audit(raw, spec)
        args.out.mkdir(parents=True, exist_ok=False)
        with (args.out / "metadata.csv").open("xb") as handle:
            handle.write(raw)
        write_json(args.out / "METADATA_AUDIT.json", report)
    if args.command != "fetch-metadata" and args.out is not None:
        write_json(args.out, report)
    print(json.dumps(report, indent=2, sort_keys=True))

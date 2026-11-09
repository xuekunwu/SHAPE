"""Expand only the 600 pinned TIFFs from a read-only HF mount into a fresh raw tree."""
from __future__ import annotations

# Internal worker CLI; the supported public entrypoint is run.py.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[5]))
    __package__ = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners.dataset'


import csv
from ..runtime.artifacts import digest
import json
from pathlib import Path, PurePosixPath
import shutil
import sys
import tarfile
import time

ARCHIVE = "iPSC-CM-cTnT_flowcytometry.tar.gz"
PREFIX = "iPSC-CM-cTnT_flowcytometry"
METADATA = "flow_cytometry_metadata.csv"
METADATA_SHA = "90204cc6e3894e0f7f2c9ee024438850bd5d15fc717810cabb51cec236110ff0"




def main(mount: Path, output: Path) -> None:
    if output.exists():
        raise FileExistsError(output)
    if (mount / ARCHIVE).stat().st_size != 4325280145:
        raise RuntimeError("pinned archive size differs")
    if digest(mount / METADATA) != METADATA_SHA:
        raise RuntimeError("pinned standalone metadata differs")
    with (mount / METADATA).open(newline="", encoding="utf-8-sig") as source:
        rows = list(csv.DictReader(source))
    wanted = {row["relative_path"] for row in rows}
    if len(rows) != 600 or len(wanted) != 600:
        raise RuntimeError("metadata must name 600 distinct TIFFs")
    for name in wanted:
        path = PurePosixPath(name)
        if path.is_absolute() or ".." in path.parts or path.suffix.lower() not in {".tif", ".tiff"}:
            raise RuntimeError(f"unsafe source path: {name}")
    output.mkdir(parents=True)
    shutil.copyfile(mount / METADATA, output / METADATA)
    seen = set()
    started = time.time()
    with tarfile.open(mount / ARCHIVE, "r|gz") as archive:
        for member in archive:
            if member.isdir():
                continue
            if not member.name.startswith(PREFIX + "/"):
                raise RuntimeError(f"unexpected archive member: {member.name}")
            relative = member.name[len(PREFIX) + 1:]
            if relative not in wanted:
                continue  # ignore the archive's older metadata and unselected files
            if not member.isfile() or relative in seen:
                raise RuntimeError(f"invalid or duplicate image: {relative}")
            target = output / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            temporary = target.with_name(target.name + ".part")
            with archive.extractfile(member) as source, temporary.open("wb") as destination:
                shutil.copyfileobj(source, destination, 8 * 1024 * 1024)
            if temporary.stat().st_size != member.size:
                raise RuntimeError(f"size mismatch: {relative}")
            temporary.replace(target)
            seen.add(relative)
            if len(seen) % 20 == 0:
                print(f"RAW_PROGRESS {len(seen)}/600 elapsed_min={(time.time()-started)/60:.1f}", flush=True)
    if seen != wanted:
        raise RuntimeError(f"missing {len(wanted - seen)} images")
    record = {"status": "complete", "images": len(seen), "metadata_sha256": METADATA_SHA,
              "archive_bytes": 4325280145, "source": str(mount), "seconds": time.time()-started}
    (output / "RAW_COMPLETE.json").write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record), flush=True)


if __name__ == "__main__":
    main(Path(sys.argv[1]).resolve(), Path(sys.argv[2]).resolve())

"""Content-addressed source snapshots for cluster transfer (the cardiomyocyte studies' scheme, with this study's
folder among the roots): a run is bound to the exact source files, committed or not."""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from benchmarks.full_CPJUMP1.harness.data import digest

REPO = Path(__file__).resolve().parents[3]
ROOTS = ("shape", "benchmarks", "case studies/_shared", "case studies/cardiomyocytes",
         "case studies/cardiac organoids/timecourse", "case studies/cardiac organoids/cohort",
         "case studies/cardiac organoids/drug_screen")
SUFFIXES = {".py", ".md", ".json", ".toml", ".lock", ".txt", ".yaml", ".yml", ".sh", ".csv"}
SKIP = {"results", "__pycache__", ".git"}


def source_manifest() -> dict:
    files = {}
    for folder in ROOTS:
        root = REPO / folder
        if not root.is_dir():
            continue
        for path in sorted(root.rglob("*")):
            relative = path.relative_to(REPO)
            if path.is_file() and path.suffix in SUFFIXES and not SKIP & set(relative.parts):
                files[relative.as_posix()] = digest(path)
    for name in ("pyproject.toml", "uv.lock", "requirements.txt"):
        if (REPO / name).is_file():
            files[name] = digest(REPO / name)
    canonical = json.dumps(files, sort_keys=True, separators=(",", ":")).encode()
    return {"source_sha256": hashlib.sha256(canonical).hexdigest(), "files": files}


def verify_source(expected: str) -> dict:
    record = source_manifest()
    if len(expected) != 64 or record["source_sha256"] != expected:
        raise RuntimeError("INVALID_SOURCE_SNAPSHOT: runner or SHAPE code differs from the intended snapshot")
    snapshot = REPO / "SOURCE_SNAPSHOT.json"
    if snapshot.exists():
        origin = json.loads(snapshot.read_text(encoding="utf-8"))
        if origin["source_sha256"] != expected or origin["files"] != record["files"]:
            raise RuntimeError("source snapshot contents changed after transfer")
        record.update({k: origin[k] for k in ("base_commit", "base_branch") if k in origin})
    return {k: v for k, v in record.items() if k != "files"}


def pack(out: Path) -> dict:
    out = Path(out)
    if out.exists():
        raise FileExistsError(f"snapshot archive exists: {out}")
    record = source_manifest()
    for key, cmd in (("base_commit", ["rev-parse", "HEAD"]), ("base_branch", ["branch", "--show-current"])):
        record[key] = subprocess.check_output(["git", *cmd], cwd=REPO, text=True).strip()
    record["identity_policy"] = "base commit plus exact source-file hashes; includes uncommitted edits"
    record["roots"] = list(ROOTS)
    out.parent.mkdir(parents=True, exist_ok=True)
    with tarfile.open(out, "w:gz") as archive:
        for relative in record["files"]:
            archive.add(REPO / relative, arcname=relative, recursive=False)
        raw = (json.dumps(record, indent=2) + "\n").encode()
        member = tarfile.TarInfo("SOURCE_SNAPSHOT.json")
        member.size = len(raw)
        archive.addfile(member, io.BytesIO(raw))
    if source_manifest()["files"] != record["files"]:
        raise RuntimeError("source changed while packing; discard this snapshot")
    return {k: v for k, v in record.items() if k != "files"} | {"archive_sha256": digest(out), "archive": str(out)}

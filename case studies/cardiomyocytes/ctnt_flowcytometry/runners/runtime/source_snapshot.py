"""Content-addressed code snapshots for VM testing without committing or pushing local edits."""
from __future__ import annotations


import hashlib
import io
import json
from pathlib import Path
import subprocess
import tarfile

from .artifacts import REPO
from .artifacts import digest
from .artifacts import fresh_output

ROOTS = ("shape", "benchmarks", "case studies/_shared", "case studies/cardiomyocytes", "environments/benchmark")
SUFFIXES = {".py", ".md", ".json", ".toml", ".lock", ".txt", ".yaml", ".yml", ".sh", ".csv"}


def source_manifest() -> dict:
    files = {}
    for folder in ROOTS:
        for path in sorted((REPO / folder).rglob("*")):
            relative = path.relative_to(REPO)
            if path.is_file() and path.suffix in SUFFIXES and not {"results", "__pycache__", ".git"} & set(relative.parts):
                files[relative.as_posix()] = digest(path)
    for name in ("pyproject.toml", "uv.lock", "requirements.txt"):
        path = REPO / name
        if path.is_file():
            files[name] = digest(path)
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
        record.update({k: origin[k] for k in ("base_commit", "base_branch")})
    return {k: v for k, v in record.items() if k != "files"}


def pack(out: Path) -> dict:
    fresh_output(out)
    record = source_manifest()
    for key, cmd in (("base_commit", ["rev-parse", "HEAD"]), ("base_branch", ["branch", "--show-current"])):
        record[key] = subprocess.check_output(["git", "-C", str(REPO), *cmd], text=True).strip()
    record["identity_policy"] = "base commit plus exact source-file hashes; includes uncommitted edits"
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

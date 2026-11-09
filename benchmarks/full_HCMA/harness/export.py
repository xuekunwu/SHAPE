"""EXPORT_<run>.tar.gz: everything needed to audit a run, without the bulk outputs.

Kept: the run's top-level reports (trace, RESULT, AGENT_RUN, RUN_META, TOOLS_SHA256, per_image),
traces/, logs/ (including .out/.err) and the workspace's text files (credential-shaped strings masked).
Left behind (listed in EXPORT_MANIFEST.json): arrays (.npy, .h5ad), masks, crops, overlays and
per-image figures of the tool calls, caches, the venv, the images, and large files.
"""
from __future__ import annotations

import io
import json
import os
import tarfile
import time
from pathlib import Path

from .trace_md import redact, run_secrets

EXPORT_SUFFIXES = {".md", ".py", ".sh", ".json", ".jsonl", ".times", ".log", ".csv", ".txt", ".yaml",
                   ".toml", ".out", ".err"}
EXPORT_MAX_BYTES = 20_000_000
SKIP_TOP = {"venv", "cache", "claude_config"}     # claude_config: the CLI's state (sessions = traces/)
SKIP_DIRS = {"crops", "masks", "overlays", "cache", "__pycache__"}
SKIP_CALL_DIRS = {"figures"}           # one figure per image inside a tool call
WHOLE = ("traces", "logs", "trace.md", "trace.json")   # never capped


def _files(run: Path):
    for dirpath, dirnames, filenames in os.walk(run, followlinks=False):
        rel_dir = Path(dirpath).relative_to(run)
        parts = rel_dir.parts
        keep = []
        for name in sorted(dirnames):
            if (not parts and name in SKIP_TOP) or name in SKIP_DIRS:
                continue
            if len(parts) == 3 and parts[:2] == ("workspace", "calls") and name in SKIP_CALL_DIRS:
                continue
            keep.append(name)
        dirnames[:] = keep          # os.walk does not descend into symlinked dirs (data/images)
        for name in sorted(filenames):
            yield Path(dirpath) / name


def export(run: Path) -> Path:
    """Tarball of the run for download; data arrays, crops, masks and checkpoints stay behind."""
    run = Path(run)
    out = run / f"EXPORT_{run.name}.tar.gz"
    redactions, skipped = {}, []
    secrets = run_secrets(run)            # the exact gateway key, in memory only

    def add(tar, rel: str, data: bytes, mtime: float):
        info = tarfile.TarInfo(f"{run.name}/{rel}")
        info.size, info.mtime = len(data), int(mtime)
        tar.addfile(info, io.BytesIO(data))

    with tarfile.open(out, "w:gz") as tar:
        for path in _files(run):
            if path == out or path.is_symlink() or not path.is_file():
                continue
            rel = path.relative_to(run).as_posix()
            size = path.stat().st_size
            capped = not rel.startswith(tuple(f"{w}/" for w in WHOLE)) and rel not in WHOLE
            if path.suffix.lower() not in EXPORT_SUFFIXES or (capped and size > EXPORT_MAX_BYTES):
                skipped.append({"path": rel, "bytes": size})
                continue
            text, n = redact(path.read_text(encoding="utf-8", errors="replace"), secrets)
            if n:
                redactions[rel] = n
            add(tar, rel, text.encode("utf-8"), path.stat().st_mtime)
        manifest = {"redactions": redactions, "skipped_not_exported": skipped,
                    "skipped_dirs": sorted(SKIP_TOP | SKIP_DIRS) + [f"workspace/calls/*/{d}" for d in SKIP_CALL_DIRS]}
        add(tar, "EXPORT_MANIFEST.json", json.dumps(manifest, indent=1).encode(), time.time())
    print(f"EXPORT {out} ({out.stat().st_size / 1e6:.1f} MB); redacted {sum(redactions.values())} "
          f"credential-shaped strings in {len(redactions)} files; {len(skipped)} files not exported "
          f"(arrays, large files; listed in EXPORT_MANIFEST.json)")
    return out

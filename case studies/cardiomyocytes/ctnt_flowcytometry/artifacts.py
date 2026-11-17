"""Standard-library artifact hashing, atomic JSON and confined data paths."""
from pathlib import Path
import hashlib
import json


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as f:
        for block in iter(lambda: f.read(8 * 1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path: Path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    tmp.replace(path)


def within(root: Path, relative: str) -> Path:
    if "\\" in relative or ":" in relative or ".." in Path(relative).parts:
        raise ValueError("unsafe relative path")
    path = (root / relative).resolve()
    if not path.is_relative_to(root.resolve()):
        raise ValueError("path escapes data root")
    return path


REPO = Path(__file__).resolve().parents[3]


def fresh_output(out: Path, source: Path | None = None):
    out = out.resolve()
    if out.exists():
        raise FileExistsError(f"use a fresh output: {out}")
    if out.is_relative_to(REPO) or (source is not None and out.is_relative_to(source.resolve())):
        raise ValueError("outputs must be outside the code and source data trees")

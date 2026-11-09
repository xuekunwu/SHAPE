"""Small I/O helpers shared by the publication commands (stdlib only)."""
import csv
import hashlib
import json
from pathlib import Path

ARMS = ("control", "shape", "codex")
NAMES = {"control": "CONTROL", "shape": "SHAPE", "codex": "CODEX-FULL"}
SPLITS = ("development_train", "development_val", "evaluation_only")


def read(path):
    return json.loads(Path(path).read_text(encoding="utf-8"))


def sha(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def write(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2, ensure_ascii=False, allow_nan=False) + "\n", encoding="utf-8", newline="\n")


def csv_rows(path):
    with Path(path).open(encoding="utf-8", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    rows = list(rows)
    if not rows:
        raise ValueError("Cannot infer CSV fields from an empty table")
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]), lineterminator="\n")
        writer.writeheader()
        writer.writerows(rows)


def fraction(numerator, denominator):
    """Missing measurement or zero denominator is unavailable, never zero."""
    if numerator is None or denominator in (None, 0):
        return None
    if numerator < 0 or denominator < 0 or numerator > denominator:
        raise ValueError("Invalid rate counts")
    return numerator / denominator

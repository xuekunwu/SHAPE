"""Named preps: the full prep ($ROOT/prep, name `full`) and subsets of it ($ROOT/prep_<name>).

`pairs` (subset with --per-class 2): for every cell_state other than "unknown" that occurs in >= 2
distinct parent micrographs, 2 images from 2 different micrographs. Seeded and deterministic
(SELECTION_RULE): classes sorted, a class's parents sorted; the rng picks the first parent, then each
next one among the remaining parents of a source not picked yet for that class (any remaining parent
when there is none), so a class whose micrographs span >= 2 sources gets a cross-source pair; then
one image of each parent (its image_ids sorted). The subset is laid out exactly like a prep (same
image_ids, real copies of the anonymized files, the full prep's images.csv rows, a counts-only
PREPARE.json) and is read-only; the selection, the settings and every file's sha256 are recorded in
$ROOT/private/SUBSET_<name>.json. A subset is made once: a request with other settings is refused.
"""
from __future__ import annotations

import json
import shutil
import stat
import time
from pathlib import Path

import numpy as np
import pandas as pd

from .prepare import _most_frequent, make_read_only, sha256

DESIGN = "{per_class} images per cell type, each from a different micrograph"
SELECTION_RULE = ("types sorted; parents sorted; rng: first parent, then each next parent among the remaining "
                  "parents of a source not yet picked for the type (else any remaining); rng: one image per parent")


def prep_dir(root: Path, name: str) -> Path:
    """`full` -> $ROOT/prep; any other name -> $ROOT/prep_<name>."""
    return Path(root) / ("prep" if name == "full" else f"prep_{name}")


def prep_name(prep: Path) -> str:
    """The inverse of prep_dir: $ROOT/prep -> `full`, $ROOT/prep_<name> -> <name>."""
    name = Path(prep).name
    if name == "prep":
        return "full"
    if name.startswith("prep_") and len(name) > 5:
        return name[5:]
    raise ValueError(f"not a prep directory name: {name!r} (expected prep or prep_<name>)")


def all_preps(root: Path) -> list[Path]:
    """Every prep directory under $ROOT (the full prep and every subset)."""
    return sorted(p for p in Path(root).glob("prep*") if p.is_dir() and not p.name.endswith(".partial")
                  and (p.name == "prep" or p.name.startswith("prep_")))


def _files(prep: Path) -> dict[str, str]:
    return {p.relative_to(prep).as_posix(): sha256(p) for p in sorted(prep.rglob("*")) if p.is_file()}


def prep_problems(root: Path, name: str) -> list[str]:
    """A prep against its record (PREPARE_PRIVATE.json prep_sha256 for `full`, SUBSET_<name>.json
    prep_sha256 for a subset): every file present, unchanged, none added."""
    root = Path(root)
    prep = prep_dir(root, name)
    record = root / "private" / ("PREPARE_PRIVATE.json" if name == "full" else f"SUBSET_{name}.json")
    try:
        recorded = json.loads(record.read_text(encoding="utf-8"))["prep_sha256"]
    except (OSError, ValueError, KeyError) as error:
        return [f"{record}: no readable prep_sha256 ({type(error).__name__})"]
    if not prep.is_dir():
        return [f"{prep} is missing"]
    found = _files(prep)
    problems = [f"{prep / rel}: missing or changed" for rel, digest in recorded.items() if found.get(rel) != digest]
    problems += [f"{prep / rel}: added after the record" for rel in sorted(set(found) - set(recorded))]
    return problems


def select(labels: pd.DataFrame, per_class: int, seed: int) -> tuple[pd.DataFrame, list[dict]]:
    """(image_id, cell_state, parent_id, source of the selection in selection order; per type: its
    parents' sources and whether the chosen parents span more than one source). `labels` needs
    image_id, cell_state, parent_id, source (a parent's source: its images' most frequent one)."""
    rng = np.random.default_rng(seed)
    chosen, types = [], []
    for state in sorted(set(labels["cell_state"]) - {"unknown"}):
        rows = labels[labels["cell_state"] == state]
        source_of = rows.groupby("parent_id")["source"].agg(_most_frequent).to_dict()
        remaining = sorted(source_of)
        if len(remaining) < per_class:
            continue
        picked = []
        for _ in range(per_class):
            used = {source_of[p] for p in picked}
            pool = [p for p in remaining if source_of[p] not in used] or remaining
            parent = str(rng.choice(pool))
            picked.append(parent)
            remaining.remove(parent)
        for parent in picked:
            ids = sorted(rows.loc[rows["parent_id"] == parent, "image_id"])
            chosen.append({"image_id": str(rng.choice(ids)), "cell_state": state, "parent_id": parent,
                           "source": source_of[parent]})
        sources = sorted(set(source_of.values()))
        types.append({"cell_state": state, "parents": picked, "sources": [source_of[p] for p in picked],
                      "n_parents_available": len(source_of), "sources_available": sources,
                      "cross_source": len({source_of[p] for p in picked}) > 1})
    return pd.DataFrame(chosen, columns=["image_id", "cell_state", "parent_id", "source"]), types


def subset(root: Path, name: str = "pairs", per_class: int = 2, seed: int = 0) -> Path:
    """Write $ROOT/prep_<name> and $ROOT/private/SUBSET_<name>.json. An existing subset that matches
    its record is kept; a partial or changed one is an error."""
    root = Path(root).absolute()
    if name == "full":
        raise SystemExit("NOT_READY: 'full' is the name of the whole prep, not of a subset")
    full, private = root / "prep", root / "private"
    out, record_path = prep_dir(root, name), private / f"SUBSET_{name}.json"
    if out.exists() or record_path.exists():
        try:
            recorded = json.loads(record_path.read_text(encoding="utf-8"))
        except (OSError, ValueError):
            recorded = {}
        asked = {"per_class": per_class, "seed": seed, "rule": SELECTION_RULE}
        differs = {k: (recorded.get(k), v) for k, v in asked.items() if recorded.get(k) != v}
        if differs:
            raise SystemExit(f"NOT_READY: subset {name} exists with other settings (recorded, requested): "
                             f"{differs}; a subset is made once -- use another --name")
        problems = prep_problems(root, name)
        if problems:
            raise SystemExit(f"NOT_READY: subset {name} exists but does not verify: {problems[:5]}")
        print(f"SUBSET {name} exists and verifies -> {out}")
        return out
    problems = prep_problems(root, "full")         # copy only from an unchanged full prep
    if problems:
        raise SystemExit(f"NOT_READY: the full prep does not verify, no subset made: {problems[:5]}")
    images = pd.read_csv(full / "images.csv", dtype=str, keep_default_na=False)
    labels = pd.read_csv(private / "labels.csv", dtype=str, keep_default_na=False)
    chosen, types = select(labels[["image_id", "cell_state", "parent_id", "source"]], per_class, seed)
    if chosen.empty:
        raise SystemExit(f"NOT_READY: no cell_state occurs in >= {per_class} micrographs")
    rows = images[images["image_id"].isin(set(chosen["image_id"]))].sort_values("image_id")

    partial = root / f"prep_{name}.partial"
    if partial.exists():
        shutil.rmtree(partial)
    (partial / "images").mkdir(parents=True)
    for file in rows["file"]:
        shutil.copyfile(full / file, partial / file)
    rows.to_csv(partial / "images.csv", index=False)
    summary = {"n_images": len(rows), "n_parents": int(rows["parent_id"].nunique()),
               "n_types": int(chosen["cell_state"].nunique()), "per_class": per_class,
               "n_types_cross_source": sum(t["cross_source"] for t in types),
               "design": DESIGN.format(per_class=per_class),
               "sources": rows["source"].value_counts().to_dict()}
    (partial / "PREPARE.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")
    hashes = _files(partial)
    partial.rename(out)
    make_read_only(out)

    record = {"name": name, "per_class": per_class, "seed": seed, "rule": SELECTION_RULE,
              "design": summary["design"],
              "source_prep": str(full), "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
              "n_images": len(rows), "n_types": summary["n_types"],
              "n_types_cross_source": summary["n_types_cross_source"], "types": types,
              "selection": chosen.to_dict(orient="records"), "prep_sha256": hashes}
    mode = private.stat().st_mode
    private.chmod(mode | stat.S_IWUSR)          # private is read-only after prepare: open it for this write
    try:
        record_path.write_text(json.dumps(record, indent=1), encoding="utf-8")
        record_path.chmod(record_path.stat().st_mode & ~0o222)
    finally:
        private.chmod(mode)
    print(f"SUBSET {name}: {len(rows)} images, {summary['n_types']} cell types x {per_class} micrographs "
          f"({summary['n_types_cross_source']} types cross-source) -> {out}")
    return out

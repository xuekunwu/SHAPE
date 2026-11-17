"""Anonymize HCMA into an agent-visible prep and a private label store.

$ROOT/prep (the agent sees it through its workspace):
  images/<image_id><ext>   real files under random ids (img_0001 ...); file metadata removed
                           losslessly, decoded pixels checked identical to the source. When the
                           images' channel counts differ, every image is delivered with 3 channels
                           (gray replicated, a constant 4th channel dropped; written as PNG / TIFF)
  images.csv               image_id, file, parent_id, source, image_modality, magnification,
                           height, width, channels
  PREPARE.json             counts only
$ROOT/private (scoring only):
  labels.csv, folds.csv (5 folds over parent micrographs), scorable.csv, PREPARE_PRIVATE.json

The original file names, paths and ids carry the vendor and the cell line, so none of them
reaches the prep: images are copied (a symlink would reveal the source name) and parent ids are
renamed too.
"""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
import time
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

from .hcma_labels import normalize_cell_state

REQUIRED = ("image_id", "image_filename", "relative_path", "parent_image_id", "source", "cell_type",
            "organ_category", "image_modality", "magnification")
VISIBLE = ("source", "image_modality", "magnification")       # technical metadata the agent gets
IMAGES_COLUMNS = ("image_id", "file", "parent_id", *VISIBLE, "height", "width", "channels")
LABEL_COLUMNS = ("image_id", "orig_image_id", "image_filename", "parent_id", "orig_parent_image_id",
                 "source", "cell_type", "cell_state", "organ_category")
JPEG, PNG, TIFF = (".jpg", ".jpeg"), (".png",), (".tif", ".tiff")
N_FOLDS = 5
REPO = Path(__file__).resolve().parents[3]


def make_read_only(top: Path) -> None:
    """chmod -R a-w: no write bit on `top` or anything below it (directories keep x)."""
    for path in [*sorted(top.rglob("*"), reverse=True), top]:
        if not path.is_symlink():
            path.chmod(path.stat().st_mode & ~0o222)


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


# ---------------------------------------------------------------- lossless metadata stripping
_STANDALONE = {0x01, *range(0xD0, 0xD8)}       # TEM, RST0..RST7: no length field


def _is_adobe(data: bytes, j: int) -> bool:
    return data[j] == 0xEE and data[j + 3:j + 8] == b"Adobe"


def strip_jpeg(data: bytes) -> tuple[bytes, dict]:
    """The JPEG with every APP1..APP15 and COM segment removed and any bytes after EOI dropped.
    APP0 (JFIF) is kept, and so is the 12-byte Adobe APP14 segment: it carries no metadata but sets
    the decoder's colour transform, so removing it could change the decoded pixels. Every other
    segment and the entropy-coded scan data are copied byte for byte."""
    if data[:2] != b"\xff\xd8":
        raise ValueError("not a JPEG (no SOI marker)")
    out, removed, i, n = bytearray(data[:2]), Counter(), 2, len(data)
    trailing = 0
    while i < n:
        if data[i] != 0xFF:
            raise ValueError(f"JPEG marker expected at byte {i}")
        j = i
        while j < n and data[j] == 0xFF:      # optional fill bytes before a marker
            j += 1
        if j >= n:
            break
        marker = data[j]
        if marker == 0xD9:                    # EOI
            out += data[i:j + 1]
            trailing = n - (j + 1)
            break
        if marker in _STANDALONE:
            out += data[i:j + 1]
            i = j + 1
            continue
        if j + 3 > n:
            raise ValueError("truncated JPEG segment header")
        end = j + 1 + int.from_bytes(data[j + 1:j + 3], "big")
        if end > n:
            raise ValueError(f"truncated JPEG segment 0xFF{marker:02X}")
        if (0xE1 <= marker <= 0xEF or marker == 0xFE) and not _is_adobe(data, j):
            removed["COM" if marker == 0xFE else f"APP{marker - 0xE0}"] += 1
        else:
            out += data[i:end]
        i = end
        if marker == 0xDA:                    # SOS: entropy-coded data up to the next real marker
            k = i
            while True:
                k = data.find(b"\xff", k)
                if k < 0 or k + 1 >= n:
                    k = n
                    break
                following = data[k + 1]
                if following == 0x00 or 0xD0 <= following <= 0xD7:
                    k += 2                    # stuffed byte or restart marker: still scan data
                elif following == 0xFF:
                    k += 1                    # fill byte
                else:
                    break
            out += data[i:k]
            i = k
    return bytes(out), {"removed": dict(removed), "trailing_bytes_dropped": trailing}


def _pil_pixels(source) -> tuple[np.ndarray, str]:
    from PIL import Image

    with Image.open(source) as image:
        image.load()
        return np.asarray(image), image.mode


def _dims(shape: tuple) -> tuple[int, int, int]:
    """(height, width, channels) of a decoded image shape (HW, HWC or CHW)."""
    if len(shape) == 2:
        return shape[0], shape[1], 1
    if len(shape) == 3:
        if shape[-1] <= 4:
            return shape[0], shape[1], shape[2]
        return shape[1], shape[2], shape[0]
    raise ValueError(f"unsupported image shape {shape}")


def channel_count(src: Path) -> int:
    """Channels of the decoded image, read from the header (PIL bands, the TIFF series shape)."""
    if src.suffix.lower() in TIFF:
        import tifffile

        with tifffile.TiffFile(src) as tif:
            return _dims(tuple(tif.series[0].shape))[2]
    from PIL import Image

    with Image.open(src) as image:
        return len(image.getbands())


def copy_clean(src: Path, dst: Path) -> dict:
    """Copy one image without its file metadata; raise unless the decoded pixels are identical.
    Returns {height, width, channels, removed}."""
    ext = src.suffix.lower()
    if ext in JPEG:
        data, info = strip_jpeg(src.read_bytes())
        dst.write_bytes(data)
        (a, mode_a), (b, mode_b) = _pil_pixels(src), _pil_pixels(dst)
    elif ext in PNG:
        from PIL import Image

        with Image.open(src) as image:
            image.load()
            clean = image.copy()
            keep = {"transparency": image.info["transparency"]} if "transparency" in image.info else {}
            info = {"removed": {f"PNG {k}": 1 for k in image.info if k not in keep}}
        clean.info = keep                     # no text, exif, icc, dpi, time chunks
        clean.save(dst, format="PNG")
        (a, mode_a), (b, mode_b) = _pil_pixels(src), _pil_pixels(dst)
    elif ext in TIFF:
        import tifffile

        a = tifffile.imread(src)
        photometric = "rgb" if a.ndim == 3 and a.shape[-1] in (3, 4) else "minisblack"
        tifffile.imwrite(dst, a, photometric=photometric, metadata=None, software=False)
        b = tifffile.imread(dst)
        mode_a, mode_b = str(a.dtype), str(b.dtype)
        info = {"removed": {"TIFF rewritten without description/software tags": 1}}
    else:
        raise ValueError(f"unsupported image extension {src.suffix!r}: {src.name}")
    if mode_a != mode_b or a.shape != b.shape or not np.array_equal(a, b):
        raise RuntimeError(f"decoded pixels differ after metadata removal: {src}")
    height, width, channels = _dims(a.shape)
    return {"height": int(height), "width": int(width), "channels": int(channels), **info}


def copy_as_rgb(src: Path, dst_stem: Path) -> dict:
    """Write one image with 3 channels, values and dtype unchanged: a 1-channel image replicated
    into 3 identical channels, a 4-channel image without its constant 4th channel (e.g. opaque
    alpha). Lossless: PNG for uint8, TIFF otherwise (Pillow writes no 16-bit RGB PNG). Raises
    unless every output channel equals its source channel. Returns {name, height, width, channels}."""
    import tifffile
    from PIL import Image

    a = tifffile.imread(src) if src.suffix.lower() in TIFF else _pil_pixels(src)[0]
    if a.ndim == 2:
        a = a[..., None]
    elif a.ndim == 3 and a.shape[-1] > 4:            # CHW TIFF
        a = np.moveaxis(a, 0, -1)
    if a.shape[-1] == 1:
        rgb, source_channel = np.repeat(a, 3, axis=-1), (0, 0, 0)
    elif a.shape[-1] == 4 and (a[..., 3] == a[..., 3].flat[0]).all():
        rgb, source_channel = a[..., :3], (0, 1, 2)
    else:
        raise SystemExit(f"NOT_READY: cannot bring {src.name} ({a.shape[-1]} channels"
                         f"{', non-constant 4th channel' if a.shape[-1] == 4 else ''}) to 3 channels")
    rgb = np.ascontiguousarray(rgb)
    if rgb.dtype == np.uint8:
        dst = dst_stem.with_name(dst_stem.name + ".png")
        Image.fromarray(rgb, "RGB").save(dst, format="PNG")
        b = _pil_pixels(dst)[0]
    else:
        dst = dst_stem.with_name(dst_stem.name + ".tif")
        tifffile.imwrite(dst, rgb, photometric="rgb", metadata=None, software=False)
        b = tifffile.imread(dst)
    if (b.dtype != a.dtype or b.shape != rgb.shape
            or not all(np.array_equal(b[..., c], a[..., s]) for c, s in enumerate(source_channel))):
        raise RuntimeError(f"pixels differ after the 3-channel conversion: {src}")
    return {"name": dst.name, "height": int(b.shape[0]), "width": int(b.shape[1]), "channels": 3}


# ---------------------------------------------------------------- ids, folds, scorable set
def _ids(prefix: str, n: int) -> list[str]:
    width = max(4, len(str(n)))
    return [f"{prefix}_{i:0{width}d}" for i in range(1, n + 1)]


def _most_frequent(values) -> str:
    counts = Counter(values)
    return sorted(counts.items(), key=lambda kv: (-kv[1], kv[0]))[0][0]


def assign_folds(labels: pd.DataFrame, seed: int) -> pd.DataFrame:
    """parent_id -> fold: StratifiedGroupKFold over images grouped by parent micrograph,
    stratified by each parent's (most frequent) cell_state."""
    from sklearn.model_selection import StratifiedGroupKFold

    state_of = labels.groupby("parent_id")["cell_state"].agg(_most_frequent)
    y = labels["parent_id"].map(state_of).to_numpy()
    groups = labels["parent_id"].to_numpy()
    folds = {}
    splitter = StratifiedGroupKFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (_, test) in enumerate(splitter.split(np.zeros(len(labels)), y, groups)):
        for parent in np.unique(groups[test]):
            folds[parent] = fold
    return pd.DataFrame({"parent_id": sorted(folds), "fold": [folds[p] for p in sorted(folds)]})


def scorable_images(labels: pd.DataFrame, folds: pd.DataFrame) -> pd.DataFrame:
    """Images whose cell_state is known and occurs in >= 2 distinct parent micrographs."""
    known = labels[labels["cell_state"] != "unknown"]
    parents = known.groupby("cell_state")["parent_id"].nunique()
    keep = known[known["cell_state"].map(parents) >= 2]
    out = keep[["image_id", "cell_state", "parent_id"]].merge(folds, on="parent_id", how="left")
    return out.sort_values("image_id").reset_index(drop=True)


def _repo_commit() -> str | None:
    try:
        out = subprocess.run(["git", "-C", str(REPO), "rev-parse", "HEAD"], capture_output=True,
                             text=True, timeout=60)
    except (OSError, subprocess.SubprocessError):
        return None
    return out.stdout.strip() or None


# ---------------------------------------------------------------- the prepare stage
def prepare(hcma_root: Path, root: Path, seed: int = 0) -> tuple[Path, Path]:
    """Write $root/prep and $root/private from the HCMA download at hcma_root."""
    import sklearn

    hcma_root, root = Path(hcma_root).absolute(), Path(root).absolute()
    prep, private = root / "prep", root / "private"
    for path in (prep, private):
        if path.exists():
            raise SystemExit(f"NOT_READY: {path} exists; earlier evidence is preserved")
    meta_path = hcma_root / "HCMA_metadata.csv"
    meta = pd.read_csv(meta_path, encoding="latin-1", dtype=str)
    absent = [c for c in REQUIRED if c not in meta.columns]
    if absent:
        raise SystemExit(f"NOT_READY: {meta_path} lacks the required columns {absent}")
    for column in ("image_id", "relative_path", "parent_image_id"):
        if meta[column].isna().any():
            raise SystemExit(f"NOT_READY: {meta_path}: {int(meta[column].isna().sum())} rows without {column}")
    if meta["image_id"].duplicated().any():
        raise SystemExit(f"NOT_READY: {meta_path}: duplicated image_id values")

    # Shuffle (from a fixed base order, so the CSV row order does not matter) and rename.
    meta = meta.sort_values("image_id", kind="stable").reset_index(drop=True)
    meta = meta.iloc[np.random.default_rng(seed).permutation(len(meta))].reset_index(drop=True)
    meta.insert(0, "new_image_id", _ids("img", len(meta)))
    parents = list(dict.fromkeys(meta["parent_image_id"]))
    parent_of = dict(zip(parents, _ids("par", len(parents))))
    meta["new_parent_id"] = meta["parent_image_id"].map(parent_of)

    partial_prep, partial_private = root / "prep.partial", root / "private.partial"
    for path in (partial_prep, partial_private):
        if path.exists():
            shutil.rmtree(path)
    (partial_prep / "images").mkdir(parents=True)
    partial_private.mkdir(parents=True)

    sources = []
    for record in meta.itertuples(index=False):
        relative = Path(record.relative_path)
        if relative.is_absolute() or ".." in relative.parts:
            raise SystemExit(f"NOT_READY: relative_path leaves the HCMA root: {record.relative_path}")
        src = hcma_root / relative
        if not src.is_file():
            raise SystemExit(f"NOT_READY: missing image {src}")
        if src.suffix.lower() not in JPEG + PNG + TIFF:
            raise SystemExit(f"NOT_READY: unsupported image extension {src.suffix!r} ({src.name})")
        sources.append(src)
    # Cell_State_Analyzer_Tool refuses mixed channel counts in one call: when the images differ,
    # every image is delivered with 3 channels (copy_as_rgb); when they agree, nothing changes.
    counts = [channel_count(src) for src in sources]
    mixed = len(set(counts)) > 1
    if mixed and not set(counts) <= {1, 3, 4}:
        raise SystemExit(f"NOT_READY: mixed channel counts {dict(sorted(Counter(counts).items()))}; "
                         "only 1-, 3- and 4-channel images can be brought to 3 channels")

    rows, stripped, originals = [], Counter(), {}
    for record, src, channels in zip(meta.itertuples(index=False), sources, counts):
        stem = partial_prep / "images" / record.new_image_id
        if mixed and channels != 3:
            facts = copy_as_rgb(src, stem)
            name = facts.pop("name")
        else:
            name = f"{record.new_image_id}{src.suffix.lower()}"
            facts = copy_clean(src, stem.with_name(name))
            stripped.update(facts.pop("removed"))
            if facts.pop("trailing_bytes_dropped", 0):
                stripped["bytes_after_EOI"] += 1
        originals[record.new_image_id] = {"channels": channels, "format": src.suffix.lower(),
                                          "delivered": Path(name).suffix}
        rows.append({"image_id": record.new_image_id, "file": f"images/{name}",
                     "parent_id": record.new_parent_id,
                     **{c: getattr(record, c) for c in VISIBLE}, **facts})
    images = pd.DataFrame(rows, columns=list(IMAGES_COLUMNS)).fillna("")
    images.to_csv(partial_prep / "images.csv", index=False)

    labels = meta.rename(columns={"image_id": "orig_image_id", "parent_image_id": "orig_parent_image_id"})
    labels = labels.rename(columns={"new_image_id": "image_id", "new_parent_id": "parent_id"})
    labels["cell_state"] = labels["cell_type"].map(normalize_cell_state)
    labels = labels[list(LABEL_COLUMNS) + [c for c in labels.columns if c not in LABEL_COLUMNS]]
    folds = assign_folds(labels, seed)
    scorable = scorable_images(labels, folds)
    labels.to_csv(partial_private / "labels.csv", index=False)
    folds.to_csv(partial_private / "folds.csv", index=False)
    scorable.to_csv(partial_private / "scorable.csv", index=False)

    present = {c: int((images[c].astype(str) != "").sum()) for c in ("parent_id", *VISIBLE)}
    summary = {"n_images": len(images), "n_parents": len(parents),
               "sources": images["source"].value_counts().to_dict(),
               "extensions": dict(Counter(Path(f).suffix for f in images["file"])),
               "images_with_value": present,
               "height": [int(images["height"].min()), int(images["height"].max())],
               "width": [int(images["width"].min()), int(images["width"].max())],
               "channels": {str(k): int(v) for k, v in images["channels"].value_counts().items()},
               "channel_conversion": {
                   "to_channels": 3 if mixed else None, "n_converted": sum(1 for c in counts if mixed and c != 3),
                   "from_channels": {str(c): n for c, n in sorted(Counter(counts).items()) if mixed and c != 3}}}
    (partial_prep / "PREPARE.json").write_text(json.dumps(summary, indent=1), encoding="utf-8")

    hashes = {p.relative_to(partial_prep).as_posix(): sha256(p)
              for p in sorted(partial_prep.rglob("*")) if p.is_file()}
    private_summary = {
        "hcma_root": str(hcma_root), "hcma_root_resolved": str(hcma_root.resolve()),
        "metadata": str(meta_path), "seed": seed,
        "repo": str(REPO), "repo_commit": _repo_commit(), "sklearn": sklearn.__version__,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "n_images": len(images), "n_parents": len(parents),
        "n_scorable_images": len(scorable), "n_scorable_classes": int(scorable["cell_state"].nunique()),
        "n_scorable_parents": int(scorable["parent_id"].nunique()),
        "stripped_segments": dict(stripped),
        "original_images": originals,       # image_id -> source channels and format, delivered format
        "prep_sha256": hashes,
        "private_sha256": {name: sha256(partial_private / name)
                           for name in ("labels.csv", "folds.csv", "scorable.csv")}}
    (partial_private / "PREPARE_PRIVATE.json").write_text(json.dumps(private_summary, indent=1),
                                                         encoding="utf-8")
    partial_private.rename(private)
    partial_prep.rename(prep)
    for path in (prep, private):        # every run and the control read them; nothing may change them
        make_read_only(path)
    print(f"PREPARE {len(images)} images, {len(parents)} parents -> {prep}; "
          f"{len(scorable)} scorable images in {private_summary['n_scorable_classes']} classes -> {private}; "
          f"removed metadata segments {dict(stripped)}")
    return prep, private

"""Data/provenance helpers. No model, SHAPE or gateway imports at module load."""
from __future__ import annotations

import hashlib
import importlib.util
import json
import os
import random
import re
import subprocess
import sys
from pathlib import Path

import numpy as np

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
CELLSEG = REPO / "benchmarks/segmentation_NeurIPS22-CellSeg/runners"
ARMS = ("cellpose_sam", "shape_seg_co", "microsam", "stardist", "llm", "react", "shape")
EXTRA_ARMS = ("organoid_id",)   # opt-in arms outside the default seven-arm comparison
LLM_ARMS = ("llm", "react", "shape")
CO_ARMS = ("shape_seg_co", "react", "shape")
MODEL = "gpt-5.6-luna"
DATA_ROOT = "/scratch/groups/your-group/your-user/shape_benchmark/Inhouse_Organoid_Seg"
LABEL_ROOT = "/scratch/groups/your-group/your-user/shape_benchmark/Inhouse_Organoid_Seg_GT"
WEIGHTS = "/scratch/groups/your-group/your-user/shape_benchmark/CO_cpsam_finetune/shape_seg_co"
IMAGE_EXTS = {".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg"}
LABEL_EXTS = IMAGE_EXTS - {".jpg", ".jpeg", ".bmp"} | {".npy"}


def load_shared(name, filename):
    spec = importlib.util.spec_from_file_location(name, CELLSEG / filename)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


cellseg = load_shared("organoid_shared_cellseg_arms", "cellseg_arms.py")
scoring = load_shared("organoid_shared_instance_scoring", "run_cellpose_sam_only.py")


def dump(path, obj):
    Path(path).write_text(json.dumps(obj, indent=2, default=str, allow_nan=False) + "\n", encoding="utf-8")


def append(path, obj):
    with Path(path).open("a", encoding="utf-8") as fh:
        fh.write(json.dumps(obj, default=str, allow_nan=False) + "\n")
        fh.flush()
        os.fsync(fh.fileno())


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def digest(obj):
    return hashlib.sha256(json.dumps(obj, sort_keys=True).encode()).hexdigest()


def labels(path):
    path = Path(path)
    if path.suffix.lower() == ".npy":
        arr = np.load(path, allow_pickle=True)  # trusted local Cellpose *_seg.npy dictionaries
        if arr.shape == () and arr.dtype == object and isinstance(arr.item(), dict):
            payload = arr.item()
            arr = next(payload[k] for k in ("masks", "mask", "labels") if k in payload)
    else:
        from skimage.io import imread
        arr = imread(str(path))
    arr = np.asarray(arr).squeeze()
    if arr.ndim != 2 or not np.issubdtype(arr.dtype, np.integer) or np.any(arr < 0):
        raise ValueError(f"Expected a nonnegative 2D integer instance mask: {path}")
    return arr


def discover(data_root, image_root=None, label_root=None, label_suffix="_seg"):
    """Strict recursive pairing; missing/ambiguous GT blocks preparation, never scores zero."""
    root = Path(data_root).resolve()
    images = Path(image_root).resolve() if image_root else root / "images" if (root / "images").is_dir() else root
    if label_root:
        label_dirs = [Path(label_root).resolve()]
    else:
        label_dirs = [root / n for n in ("labels", "masks", "ground_truth", "gt") if (root / n).is_dir()]
    pairs = []
    for image in sorted(images.rglob("*")):
        if (not image.is_file() or image.suffix.lower() not in IMAGE_EXTS
                or any(d == image or d in image.parents for d in label_dirs)
                or image.stem.endswith((label_suffix, "_flows", "_masks", "_label"))):
            continue
        relative = image.relative_to(images)
        candidates = set()
        # In a separate labels tree, both same-stem and suffix labels are accepted.
        # Flat data accepts only the suffix: a second image format is not a GT label.
        for folder, stems in [(d / relative.parent, (image.stem, image.stem + label_suffix)) for d in label_dirs] + [
                (image.parent, (image.stem + label_suffix,))]:
            for stem in stems:
                for ext in LABEL_EXTS:
                    candidate = folder / (stem + ext)
                    if candidate.is_file() and candidate.resolve() != image.resolve():
                        candidates.add(candidate.resolve())
        if len(candidates) != 1:
            raise ValueError(f"Expected exactly one GT for {image}; found {sorted(map(str, candidates))}")
        gt_path = candidates.pop()
        gt = labels(gt_path)
        raw = cellseg.read_image(image)
        if raw.ndim not in (2, 3) or raw.shape[:2] != gt.shape:
            raise ValueError(f"Image/GT geometry mismatch: {image}: {raw.shape}, {gt.shape}")
        unit = re.sub(r"[^A-Za-z0-9_.-]+", "__", relative.with_suffix("").as_posix()).strip("_.-")
        if not unit:
            raise ValueError(f"Empty image ID: {image}")
        pairs.append({"image_id": unit, "image": str(image.resolve()), "label": str(gt_path),
                      "shape": list(gt.shape), "image_sha256": sha256(image), "label_sha256": sha256(gt_path)})
    if not pairs or len({p["image_id"] for p in pairs}) != len(pairs):
        raise ValueError("No paired images, or image IDs collide after sanitization")
    return pairs


def prepare(args):
    pairs = discover(args.data_root, args.image_root, args.label_root, args.label_suffix)
    if args.sample_size is not None:
        if not 0 < args.sample_size <= len(pairs):
            raise ValueError("sample-size must be positive and no larger than the paired test set")
        pairs = sorted(random.Random(args.seed).sample(pairs, args.sample_size), key=lambda p: p["image_id"])
    manifest = {"schema": 1, "split": "user-provided-evaluation", "seed": args.seed, "pairs": pairs}
    manifest["fingerprint"] = digest(manifest)
    out = Path(args.manifest).resolve()
    protect_output(out.parent, pairs)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x", encoding="utf-8") as fh:
        fh.write(json.dumps(manifest, indent=2) + "\n")
    return manifest


def read_manifest(path, verify=True):
    manifest = json.loads(Path(path).read_text(encoding="utf-8"))
    payload = {k: v for k, v in manifest.items() if k != "fingerprint"}
    if manifest.get("schema") != 1 or digest(payload) != manifest.get("fingerprint"):
        raise ValueError("Invalid manifest schema or fingerprint")
    pairs = manifest["pairs"]
    ids = [p["image_id"] for p in pairs]
    if not ids or len(set(ids)) != len(ids) or any(not re.fullmatch(r"[A-Za-z0-9][A-Za-z0-9_.-]*", s) for s in ids):
        raise ValueError("Invalid/duplicate image IDs")
    if verify:
        for pair in pairs:
            for name in ("image", "label"):
                if sha256(pair[name]) != pair[name + "_sha256"]:
                    raise ValueError(f"Input changed since preparation: {pair[name]}")
            if list(labels(pair["label"]).shape) != pair["shape"]:
                raise ValueError("Ground-truth geometry changed")
    return manifest


def protect_output(out, pairs, weights=None):
    out = Path(out).resolve()
    # Protect entire input directories, including flat image/GT trees.
    protected = {Path(p[k]).resolve().parent for p in pairs for k in ("image", "label")}
    if weights:
        protected.add(Path(weights).resolve().parent)
    if any(out == p or p in out.parents for p in protected):
        raise ValueError("Output must be outside source image/label/model directories")
    if out == REPO or REPO in out.parents:
        raise ValueError("Write run outputs outside the source checkout")


def source_identity():
    def git(*args):
        return subprocess.check_output(["git", *args], cwd=REPO, text=True, timeout=15).strip()
    paths = list(HERE.glob("*.py")) + list(HERE.glob("*.sh")) + [HERE / "Prompt.txt"]
    # Dirty changes, including the uncommitted VRR integration, must affect resume identity.
    for folder in ("shape", "benchmarks/_shared", "benchmarks/segmentation_NeurIPS22-CellSeg/runners"):
        paths.extend((REPO / folder).rglob("*.py"))
    paths.extend((REPO / "shape/tools").rglob("CARD.md"))
    hashes = {p.relative_to(REPO).as_posix(): sha256(p) for p in sorted(set(paths))}
    return {"commit": git("rev-parse", "HEAD"), "branch": git("branch", "--show-current"),
            "dirty": bool(git("status", "--porcelain")), "source_sha256": digest(hashes), "files": hashes}


def configure(search, vrr_policy="legacy", max_completion_tokens=None, reasoning_effort="high"):
    # Pin the experiment rather than inheriting stale shell settings from another benchmark.
    if reasoning_effort not in ("low", "medium", "high"):
        raise ValueError(f"unsupported reasoning effort: {reasoning_effort!r}")
    settings = {"SHAPE_REASONING_EFFORT": reasoning_effort, "ORGANOID_PINNED_REASONING_EFFORT": reasoning_effort,
                "LLM_MODEL": MODEL,
                "SHAPE_ORGANOID_ADAPTIVE_RESCUE": "0", "SHAPE_SEARCH_STRATEGY_SEGMENTATION": search,
                "SHAPE_SEGMENTATION_EARLY_STOP": "1", "SHAPE_SEGMENTATION_CONFIRM_STOP": "valid_choice",
                "SHAPE_SEGMENTATION_COMPACT_PROMPTS": "0", "SHAPE_SEGMENTATION_CALIBRATION": "full",
                "SHAPE_VISION_MAX_PX": "1024", "SHAPE_SAM_SERVER_URL": "", "SHAPE_VRR_POLICY": vrr_policy}
    if max_completion_tokens:
        settings["SHAPE_MAX_COMPLETION_TOKENS"] = str(int(max_completion_tokens))
    else:
        os.environ.pop("SHAPE_MAX_COMPLETION_TOKENS", None)   # the engine's per-call default applies
    os.environ.update(settings)
    return settings


def versions():
    from importlib.metadata import PackageNotFoundError, version
    found = {}
    for name in ("cellpose", "torch", "micro-sam", "stardist", "tensorflow", "numpy", "scipy", "openai"):
        try:
            found[name] = version(name)
        except PackageNotFoundError:
            pass
    return found

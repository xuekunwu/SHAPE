"""External comparison arms (contract amendments A2, A3): label-free embeddings of every FULL cell, made without the
analyzer and without training, stored RAW; score.py post-processes every arm the same way (`score.post_process`: the
native and the matched table) and scores them exactly like the analyzer's arms.

  openphenom    OpenPhenom-S/16 zero-shot (Hugging Face recursionpharma/OpenPhenom at OPENPHENOM_REVISION:
                Recursion's channel-agnostic MAE, ViT-S/16, trained on RxRx3 and JUMP cpg0016, no RxRx1). The 6 RxRx1
                channels in their native order (CHANNELS, the same Recursion stain set) enter at 256 px only (the
                model's positional embeddings are fixed to a 16 x 16 patch grid per channel, so the 64 px crops are
                resized, `fit="resize"`, or centred on a zero canvas, `fit="pad"`); the stored embedding is the model
                card's `MAEModel.predict` output as is (return_channelwise_embeddings=False: the mean of the patch
                tokens, 384-d). The card's PCA-CenterScale is not applied (score.py: no PCA in either table).
  cellprofiler  CellProfiler-style hand-crafted features per crop with cp_measure (the Python port of CellProfiler's
                measurements; not the full CellProfiler pipeline on whole sites). Masks from the crop itself
                (`crop_masks`): nuclei = Otsu on the smoothed DNA channel (the task-116 prep's nucleus rule), the
                central one = the component at or nearest the crop centre; cells = a seeded watershed of the cytoplasm
                foreground (Otsu on the smoothed sum of the other channels) from all nuclei of the crop, the central
                nucleus's cell kept (close to CellProfiler's IdentifySecondaryObjects / Propagation; fallback: the
                whole crop, counted). Per compartment (Nuclei, Cells): shape (sizeshape) and, per channel, intensity,
                texture, granularity and radial distribution (CP_PARAMS). Position, constant and mostly non-finite
                features are dropped (`filter_features`); the stored matrix is the raw values of the kept features
                (float32, a remaining non-finite value as NaN).

`stage_openphenom` (the `assets` stage, a login node: it downloads) puts the pinned snapshot under
<output root>/assets/openphenom/ with the sha256 of every file in its ASSETS.json; `check_openphenom` verifies it and
`load_openphenom` loads it offline (HF_HUB_OFFLINE=1) from that path. `run` (the `external` stage) writes
<root>/external/<arm>/: embeddings.npy (float32, one row per FULL cell in prep/cells.csv order) and EMBED.json (the
settings, the package versions, the feature list, the sha256 of the prep table and of images.npy, the commit); with
`limit` (a pilot) it embeds the first `limit` cells into <root>/external_pilot/<arm>[_<fit>]_n<limit>/ (one directory
per pilot setting, never overwritten; never scored) and records the label-free sibling-site proxy on the native and
the matched post-processing. Nothing here reads a label: the inputs are prep/cells.csv (no label column) and the
prep's images.npy.
"""
from __future__ import annotations

import json
import os
import shutil
import socket
import sys
import time
from pathlib import Path

import numpy as np

from benchmarks.representation_RxRx1.runners.control import fs_retry, git_commit, write_json

EXTERNAL_ARMS = ("openphenom", "cellprofiler")
EXTERNAL_DIR, PILOT_DIR = "external", "external_pilot"
CHANNELS = ("hoechst", "concanavalin_a", "phalloidin", "syto14", "mitotracker", "wga")   # RxRx1 w1..w6 (images.npy)

# ---- OpenPhenom ----------------------------------------------------------------------------------------------
OPENPHENOM_REPO = "recursionpharma/OpenPhenom"
OPENPHENOM_REVISION = "0f92333685f6e9f031b804c70fe246f9b05ae90d"          # main on 2026-10-04
OPENPHENOM_WEIGHTS = "model.safetensors"         # despite its name a torch zip pickle: {"state_dict": ...}
OPENPHENOM_WEIGHTS_SHA256 = "f6e5f1c97101331b1574c9e4b99623260191c55eea5d98e40460849c0e4c4d47"   # the Hub's LFS oid
OPENPHENOM_FILES = ("__init__.py", "huggingface_mae.py", "mae_modules.py", "mae_utils.py", "masking.py", "loss.py",
                    "normalizer.py", "vit.py", "vit_encoder.py", "config.json", "config.yaml", "pyproject.toml",
                    "README.md", "LICENSE", OPENPHENOM_WEIGHTS)
OPENPHENOM_LICENSE = "Recursion Pharmaceuticals Non-Commercial End User License Agreement (LICENSE in the snapshot)"
OPENPHENOM_LICENSE_URL = f"https://huggingface.co/{OPENPHENOM_REPO}/blob/{OPENPHENOM_REVISION}/LICENSE"
# the EULA's section 4 attribution form for an AI model (verbatim, the bracket filled in), and its modification notice
OPENPHENOM_ATTRIBUTION = (
    "We used the Phenom-Beta AI model, available from Recursion Pharmaceuticals, with software documentation at "
    "www.rxrx.ai, and from NVIDIA Corporation at https://developer.nvidia.com/docs/bionemo-service/phenom-beta.html, "
    f"pursuant to Recursion Pharmaceuticals' licensing terms at {OPENPHENOM_LICENSE_URL}. Under this license, "
    "Recursion Pharmaceuticals disclaims all representations and warranties with respect to such AI model. "
    f"(Model used: {OPENPHENOM_REPO} at revision {OPENPHENOM_REVISION}, unmodified, zero-shot; its embeddings of "
    "RxRx1 crops were computed for a non-commercial benchmark comparison.)")
OPENPHENOM_INPUT_SIZE, OPENPHENOM_FIT, OPENPHENOM_BATCH = 256, "resize", 64   # 256 px only: its fixed patch grid
OPENPHENOM_PACKAGES = ("torch", "timm", "transformers")

# ---- CellProfiler-style ----------------------------------------------------------------------------------------
COMPARTMENTS = ("Nuclei", "Cells")
CP_SHAPE = "sizeshape"
CP_GROUPS = ("intensity", "texture", "granularity", "radial_distribution")   # per channel and compartment
CP_PARAMS = {
    "sizeshape": {},
    "intensity": {},
    "texture": {"scale": 3, "gray_levels": 256},
    # the 64 px crops are not subsampled (the 0.25 default is meant for whole sites: it would leave 16 px)
    "granularity": {"subsample_size": 1.0, "image_sample_size": 1.0, "element_size": 10,
                    "granular_spectrum_length": 16},
    "radial_distribution": {"bin_count": 4, "scaled": True, "maximum_radius": 100},
}
CP_SMOOTH_SIGMA, CP_OPENING_RADIUS = 2.0, 2        # the task-116 prep's nucleus detection (harness/data.py)
CP_MAX_NONFINITE = 0.05                            # dropped when non-finite in more cells (pycytominer drop_na_columns)
CP_CHUNK = 64                                      # crops per worker task
CP_MASK_COUNTS = ("nucleus_fallback", "cell_fallback", "cell_touches_border")
# absolute coordinates in the crop (object centre, bounding box, intensity-weighted / maximum location): not morphology
CP_POSITION = ("_Center_", "_BoundingBoxMinimum_", "_BoundingBoxMaximum_", "_Location_")
CP_PACKAGES = ("cp-measure", "scikit-image", "scipy", "numpy", "mahotas", "centrosome")


def sha256_file(path: Path) -> str:
    from benchmarks.representation_RxRx1.runners.prepare import sha256_file as digest

    return digest(path)


def versions(names) -> dict:
    from importlib.metadata import PackageNotFoundError, version

    out = {}
    for name in names:
        try:
            out[name] = version(name)
        except PackageNotFoundError:
            out[name] = None
    return out


# ---- OpenPhenom: the offline snapshot --------------------------------------------------------------------------

def stage_openphenom(output_root: Path, *, source: Path | None = None,
                     expected_weights_sha256: str = OPENPHENOM_WEIGHTS_SHA256) -> Path:
    """<output_root>/assets/openphenom/: the OpenPhenom snapshot at OPENPHENOM_REVISION (downloaded from the Hub, or
    copied from `source`, a local copy of it) with ASSETS.json (repo, revision, license, the sha256 of every file).
    Kept when it exists; needs the DINOv3 assets (prepare) first; written into a .partial directory and renamed."""
    assets = Path(output_root).absolute() / "assets"
    final = assets / "openphenom"
    if (final / "ASSETS.json").exists():
        print(f"OpenPhenom assets exist, kept: {final}")
        return final / "ASSETS.json"
    if not (assets / "ASSETS.json").exists():
        raise SystemExit(f"NOT_READY: {assets}/ASSETS.json is missing (run the prepare stage first)")
    if final.exists():
        raise SystemExit(f"NOT_READY: {final} exists without ASSETS.json; move it aside")
    partial = assets / "openphenom.partial"
    if partial.exists():
        fs_retry(shutil.rmtree, partial)
    snapshot = partial / "snapshot"
    if source is not None:
        fs_retry(snapshot.mkdir, parents=True)
        for name in OPENPHENOM_FILES:
            if not (Path(source) / name).exists():
                raise SystemExit(f"NOT_READY: {source} lacks {name}")
            fs_retry(shutil.copyfile, Path(source) / name, snapshot / name)
        origin = f"copy of {Path(source).absolute()}"
    else:
        from huggingface_hub import snapshot_download

        fs_retry(snapshot_download, repo_id=OPENPHENOM_REPO, revision=OPENPHENOM_REVISION, local_dir=str(snapshot),
                 allow_patterns=list(OPENPHENOM_FILES))
        origin = f"https://huggingface.co/{OPENPHENOM_REPO} at {OPENPHENOM_REVISION}"
    missing = [name for name in OPENPHENOM_FILES if not (snapshot / name).is_file()]
    if missing:
        raise SystemExit(f"NOT_READY: the OpenPhenom snapshot lacks {missing}")
    files = {name: sha256_file(snapshot / name) for name in OPENPHENOM_FILES}
    if files[OPENPHENOM_WEIGHTS] != expected_weights_sha256:
        raise SystemExit(f"NOT_READY: {OPENPHENOM_WEIGHTS} sha256 {files[OPENPHENOM_WEIGHTS]} is not the pinned "
                         f"{expected_weights_sha256}")
    manifest = {"created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "repo": OPENPHENOM_REPO,
                "revision": OPENPHENOM_REVISION, "origin": origin, "license": OPENPHENOM_LICENSE,
                "snapshot": str(final / "snapshot"), "files": files}
    fs_retry((partial / "ASSETS.json").write_text, json.dumps(manifest, indent=1), encoding="utf-8")
    fs_retry(partial.rename, final)
    print(f"ASSETS {final / 'ASSETS.json'}: OpenPhenom {OPENPHENOM_REVISION[:8]}, "
          f"weights {files[OPENPHENOM_WEIGHTS][:12]}")
    return final / "ASSETS.json"


def check_openphenom(output_root: Path) -> dict:
    """The OpenPhenom ASSETS.json, after checking every staged file against its sha256."""
    path = Path(output_root) / "assets" / "openphenom" / "ASSETS.json"
    if not path.exists():
        raise SystemExit(f"NOT_READY: {path} is missing (run the assets stage on a login node)")
    manifest = json.loads(path.read_text(encoding="utf-8"))
    snapshot = Path(manifest["snapshot"])
    changed = [name for name, digest in manifest["files"].items()
               if not (snapshot / name).is_file() or sha256_file(snapshot / name) != digest]
    if changed:
        raise SystemExit(f"NOT_READY: the OpenPhenom snapshot {snapshot} changed since staging: {changed}")
    return manifest


def load_openphenom(snapshot: Path, device: str):
    """The OpenPhenom MAEModel from the staged snapshot, offline: its code is imported from the snapshot directory as a
    package (the model card's `huggingface_mae.MAEModel`), its config from config.json and its weights with
    torch.load(weights_only=True) -- what MAEModel.from_pretrained does, without any Hub lookup. Readout as the card:
    return_channelwise_embeddings=False, eval mode."""
    import importlib
    import importlib.util

    os.environ["HF_HUB_OFFLINE"] = os.environ["TRANSFORMERS_OFFLINE"] = "1"
    import torch

    snapshot = Path(snapshot)
    package = "openphenom_snapshot"
    for name in [m for m in sys.modules if m == package or m.startswith(package + ".")]:
        del sys.modules[name]
    spec = importlib.util.spec_from_file_location(package, snapshot / "__init__.py",
                                                  submodule_search_locations=[str(snapshot)])
    module = importlib.util.module_from_spec(spec)
    sys.modules[package] = module
    spec.loader.exec_module(module)
    mae = importlib.import_module(f"{package}.huggingface_mae")
    model = mae.MAEModel(mae.MAEConfig.from_pretrained(str(snapshot)))
    # through a file object: a newer torch picks its loader by the `.safetensors` extension of a path and fails on
    # these weights, which are a torch zip pickle despite the name (OPENPHENOM_WEIGHTS)
    with open(snapshot / OPENPHENOM_WEIGHTS, "rb") as handle:
        state = torch.load(handle, map_location="cpu", weights_only=True)
    model.load_state_dict(state["state_dict"])
    model.return_channelwise_embeddings = False
    return model.eval().to(device)


def check_input_size(size) -> int:
    """OPENPHENOM_INPUT_SIZE (256), else SystemExit: the model's positional embeddings are a fixed 16 x 16 grid of
    16 px patches per channel, so it takes 256 px only."""
    if int(size) != OPENPHENOM_INPUT_SIZE:
        raise SystemExit(f"NOT_READY: OpenPhenom takes {OPENPHENOM_INPUT_SIZE} px only (its positional grid is fixed), "
                         f"not {size}")
    return int(size)


def fit_input(batch, size: int, fit: str):
    """A (B, 6, h, w) uint8 tensor as the model's float32 input (B, 6, size, size) on the uint8 scale (predict divides
    by 255, then normalises each channel of each image): bilinear resize, or the crop centred on a zero canvas."""
    import torch
    import torch.nn.functional as F

    x = batch.float()
    height, width = x.shape[-2:]
    if (height, width) == (size, size):
        return x
    if fit == "resize":
        return F.interpolate(x, size=(size, size), mode="bilinear", align_corners=False)
    if fit == "pad":
        if height > size or width > size:
            raise ValueError(f"fit='pad' needs crops of at most {size} px, got {height} x {width}")
        canvas = torch.zeros((*x.shape[:2], size, size), dtype=x.dtype, device=x.device)
        top, left = (size - height) // 2, (size - width) // 2
        canvas[..., top:top + height, left:left + width] = x
        return canvas
    raise ValueError(f"fit must be 'resize' or 'pad', not {fit!r}")


def openphenom_vectors(model, images, rows, *, input_size: int = OPENPHENOM_INPUT_SIZE, fit: str = OPENPHENOM_FIT,
                       batch_size: int = OPENPHENOM_BATCH, device: str = "cpu", progress: bool = True) -> np.ndarray:
    """model.predict of every crop images[rows[i]] (uint8, 6 channels in their stored order), row i of the result."""
    import contextlib

    import torch

    input_size = check_input_size(input_size)
    rows = np.asarray(rows, dtype=np.int64)
    out, started, step = None, time.time(), max(1, (len(rows) // batch_size) // 20)
    for index, start in enumerate(range(0, len(rows), batch_size)):
        batch = torch.from_numpy(np.ascontiguousarray(images[rows[start:start + batch_size]])).to(device)
        x = fit_input(batch, input_size, fit)
        cuda = str(device).startswith("cuda")
        with torch.no_grad(), (torch.autocast("cuda") if cuda else contextlib.nullcontext()):
            latent = model.predict(x).float().cpu().numpy()
        if out is None:
            out = np.empty((len(rows), latent.shape[1]), dtype=np.float32)
        out[start:start + len(latent)] = latent
        if progress and (index % step == 0 or start + batch_size >= len(rows)):
            done = min(start + batch_size, len(rows))
            print(f"openphenom: {done}/{len(rows)} cells, {done / max(time.time() - started, 1e-9):.0f} cells/s",
                  flush=True)
    return out if out is not None else np.empty((0, 0), dtype=np.float32)


# ---- CellProfiler-style: masks and features --------------------------------------------------------------------

def _foreground(image: np.ndarray) -> np.ndarray | None:
    """Otsu foreground of the Gaussian-smoothed image, opened and hole-filled (the prep's nucleus rule); None when the
    image is flat or nothing remains."""
    from scipy import ndimage
    from skimage.filters import threshold_otsu
    from skimage.morphology import binary_opening, disk

    smooth = ndimage.gaussian_filter(np.asarray(image, dtype=np.float32), CP_SMOOTH_SIGMA)
    if smooth.max() <= smooth.min():
        return None
    mask = ndimage.binary_fill_holes(binary_opening(smooth > threshold_otsu(smooth), disk(CP_OPENING_RADIUS)))
    return mask if mask.any() else None


def _central_label(labels: np.ndarray) -> int:
    """The label at the crop centre, else the label of the labelled pixel nearest to it."""
    r, c = labels.shape[0] // 2, labels.shape[1] // 2
    if labels[r, c]:
        return int(labels[r, c])
    rr, cc = np.nonzero(labels)
    nearest = int(np.argmin((rr - r) ** 2 + (cc - c) ** 2))       # ties: the first in C order
    return int(labels[rr[nearest], cc[nearest]])


def crop_masks(crop: np.ndarray) -> tuple[np.ndarray, np.ndarray, dict]:
    """(nucleus, cell, flags) boolean masks of one (6, h, w) crop. Nuclei: the DNA channel's foreground components;
    the central one is the nucleus. Cells: a seeded watershed of the cytoplasm foreground (the sum of the other
    channels, with the nuclei) on the inverted smoothed cytoplasm image, one seed per nucleus of the crop; the central
    nucleus's cell is the cell (close to CellProfiler's IdentifySecondaryObjects / Propagation). The whole crop when a
    channel has no foreground; a nucleus fallback is a cell fallback too (no seed). Flags: the fallbacks and whether
    the cell touches the crop border."""
    from scipy import ndimage
    from skimage.measure import label
    from skimage.segmentation import watershed

    whole = np.ones(crop.shape[1:], dtype=bool)
    flags = dict.fromkeys(CP_MASK_COUNTS, False)
    dna = _foreground(crop[0])
    if dna is None:
        flags["nucleus_fallback"] = flags["cell_fallback"] = True
        return whole, whole, flags
    nuclei = label(dna, connectivity=2)
    central = _central_label(nuclei)
    nucleus = nuclei == central
    cytoplasm = np.asarray(crop[1:], dtype=np.float32).sum(axis=0)
    other = _foreground(cytoplasm)
    if other is None:
        flags["cell_fallback"] = True
        return nucleus, whole, flags
    smooth = ndimage.gaussian_filter(cytoplasm, CP_SMOOTH_SIGMA)
    cell = watershed(-smooth, markers=nuclei, mask=other | dna) == central
    flags["cell_touches_border"] = bool(cell[0].any() or cell[-1].any() or cell[:, 0].any() or cell[:, -1].any())
    return nucleus, cell, flags


def _measurements() -> dict:
    from cp_measure.bulk import get_core_measurements

    core = get_core_measurements()
    return {group: core[group] for group in (CP_SHAPE, *CP_GROUPS)}


def _jobs():
    """(compartment, group, channel index or None): shape per compartment, the CP_GROUPS per channel."""
    for compartment in COMPARTMENTS:
        yield compartment, CP_SHAPE, None
        for channel in range(len(CHANNELS)):
            for group in CP_GROUPS:
                yield compartment, group, channel


def measure_crop(crop: np.ndarray, measurements: dict) -> tuple[dict, dict, list]:
    """({feature: value}, mask flags, errors) of one uint8 crop: every cp_measure call of `_jobs` on the crop's masks,
    the pixels scaled to [0, 1] (cp_measure's CellProfiler convention); a call that raises leaves its features out."""
    nucleus, cell, flags = crop_masks(crop)
    pixels = np.asarray(crop, dtype=np.float64) / 255.0
    labels = {"Nuclei": nucleus.astype(np.int32), "Cells": cell.astype(np.int32)}
    values, errors = {}, []
    for compartment, group, channel in _jobs():
        suffix = "" if channel is None else f"_{CHANNELS[channel]}"
        try:
            result = measurements[group](labels[compartment], None if channel is None else pixels[channel],
                                         **CP_PARAMS[group])
        except Exception as error:  # noqa: BLE001 -- one failing measurement must not lose the cell
            errors.append(f"{compartment}/{group}{suffix}: {type(error).__name__}")
            continue
        for key, value in result.items():
            value = np.asarray(value, dtype=np.float64).ravel()
            values[f"{compartment}_{key}{suffix}"] = float(value[0]) if value.size else float("nan")
    return values, flags, errors


def probe_crop(size: int = 64) -> np.ndarray:
    """A synthetic crop (a disk nucleus inside a disk cell, seeded texture) that fixes the feature list."""
    rng = np.random.default_rng(0)
    yy, xx = np.mgrid[:size, :size]
    radius = np.hypot(yy - size // 2, xx - size // 2)
    crop = rng.normal(20, 4, size=(len(CHANNELS), size, size))
    crop[0] += 160 * (radius <= size / 8)
    crop[1:] += 90 * (radius <= size / 3) * (1 + 0.3 * np.sin(xx / 3.0))[None]
    return np.clip(crop, 0, 255).astype(np.uint8)


_WORKER: dict = {}


def feature_names() -> list[str]:
    """The cellprofiler arm's raw feature list (the probe crop's, in `_jobs` order)."""
    if "names" not in _WORKER:
        _WORKER.setdefault("measurements", _measurements())
        values, _, errors = measure_crop(probe_crop(), _WORKER["measurements"])
        if errors:
            raise SystemExit(f"CELLPROFILER_FAILED: cp_measure cannot measure the probe crop: {errors}")
        _WORKER["names"] = list(values)
    return _WORKER["names"]


def featurise_chunk(crops: np.ndarray) -> tuple[np.ndarray, dict, dict]:
    """(matrix, flag counts, error counts) of a chunk of crops; columns in `feature_names` order (NaN when missing)."""
    names = feature_names()
    matrix = np.full((len(crops), len(names)), np.nan)
    flags, errors = dict.fromkeys(CP_MASK_COUNTS, 0), {}
    for i, crop in enumerate(crops):
        values, crop_flags, crop_errors = measure_crop(crop, _WORKER["measurements"])
        matrix[i] = [values.get(name, np.nan) for name in names]
        for key, flag in crop_flags.items():
            flags[key] += int(flag)
        for error in crop_errors:
            errors[error] = errors.get(error, 0) + 1
    return matrix, flags, errors


def cellprofiler_features(images, rows, *, workers: int = 1, progress: bool = True) -> tuple[np.ndarray, list, dict]:
    """(raw feature matrix, feature names, mask / error counts) of the crops images[rows[i]], row i of the matrix;
    `workers` > 1 spreads chunks of CP_CHUNK crops over a spawn pool (ordered)."""
    rows = np.asarray(rows, dtype=np.int64)
    names = feature_names()
    starts = list(range(0, len(rows), CP_CHUNK))
    chunks = (np.ascontiguousarray(images[rows[s:s + CP_CHUNK]]) for s in starts)
    matrix = np.empty((len(rows), len(names)))
    counts = {**dict.fromkeys(CP_MASK_COUNTS, 0), "errors": {}}
    started, step = time.time(), max(1, len(starts) // 20)

    def collect(index, result):
        block, flags, errors = result
        matrix[starts[index]:starts[index] + len(block)] = block
        for key, n in flags.items():
            counts[key] += n
        for key, n in errors.items():
            counts["errors"][key] = counts["errors"].get(key, 0) + n
        if progress and (index % step == 0 or index == len(starts) - 1):
            done = starts[index] + len(block)
            rate = done / max(time.time() - started, 1e-9)
            print(f"cellprofiler: {done}/{len(rows)} cells, {rate:.1f} cells/s, "
                  f"eta {(len(rows) - done) / rate:.0f} s", flush=True)

    if workers > 1:
        import multiprocessing

        with multiprocessing.get_context("spawn").Pool(workers) as pool:
            for index, result in enumerate(pool.imap(featurise_chunk, chunks)):
                collect(index, result)
    else:
        for index, chunk in enumerate(chunks):
            collect(index, featurise_chunk(chunk))
    return matrix, names, counts


def filter_features(matrix: np.ndarray, names: list, *,
                    max_nonfinite: float = CP_MAX_NONFINITE) -> tuple[np.ndarray, list, dict]:
    """(raw matrix of the kept features, float32, a non-finite value as NaN; kept names; dropped {name: reason}):
    position features (CP_POSITION), features non-finite in more than `max_nonfinite` of the cells and features
    constant over all cells are dropped. No scaling here: score.post_process scales every arm (label-free)."""
    import warnings

    x = np.asarray(matrix, dtype=np.float64)
    finite = np.isfinite(x)
    share = 1.0 - finite.mean(axis=0)
    with np.errstate(all="ignore"), warnings.catch_warnings():       # an all-NaN feature: dropped below
        warnings.simplefilter("ignore", RuntimeWarning)
        spread = np.nanstd(np.where(finite, x, np.nan), axis=0)
    dropped = {}
    for j, name in enumerate(names):
        if any(key in name for key in CP_POSITION):
            dropped[name] = "absolute position in the crop"
        elif share[j] > max_nonfinite:
            dropped[name] = f"non-finite in {share[j]:.1%} of the cells"
        elif not spread[j] > 0:
            dropped[name] = "constant"
    keep = np.array([name not in dropped for name in names], dtype=bool)
    kept = [name for name, k in zip(names, keep) if k]
    return np.where(finite, x, np.nan)[:, keep].astype(np.float32), kept, dropped


# ---- the stage -------------------------------------------------------------------------------------------------

def openphenom_settings(input_size: int, fit: str, batch_size: int) -> dict:
    return {"model": OPENPHENOM_REPO, "revision": OPENPHENOM_REVISION, "weights_sha256": OPENPHENOM_WEIGHTS_SHA256,
            "license": OPENPHENOM_LICENSE, "channels": list(CHANNELS), "input_size": int(input_size), "fit": fit,
            "interpolation": "bilinear, align_corners=False" if fit == "resize" else None,
            "input": "float32 on the uint8 scale (MAEModel.predict: /255, then InstanceNorm2d per image and channel)",
            "readout": "MAEModel.predict, return_channelwise_embeddings=False: the mean of the patch tokens (no CLS)",
            "autocast": "float16 on CUDA (the card's RxRx3-core notebook)", "batch_size": int(batch_size),
            "stored": "the raw predict output; score.post_process scales it (no PCA in either table)"}


def cellprofiler_settings() -> dict:
    return {"package": "cp_measure (bulk.get_core_measurements)", "params": CP_PARAMS, "channels": list(CHANNELS),
            "compartments": list(COMPARTMENTS), "pixels": "uint8 / 255",
            "masks": f"nuclei: Otsu on the DNA channel after a Gaussian (sigma {CP_SMOOTH_SIGMA}), opening "
                     f"(disk {CP_OPENING_RADIUS}), holes filled; the nucleus: the component at or nearest the centre; "
                     "cells: seeded watershed of the cytoplasm foreground (the same rule on the sum of the other "
                     "channels, with the nuclei) on the inverted smoothed cytoplasm image, one seed per nucleus, the "
                     "central nucleus's cell kept; the whole crop when a channel has no foreground (a nucleus "
                     "fallback is a cell fallback too); counted, with the cells touching the crop border",
            "dropped": f"position features (names with {', '.join(CP_POSITION)}), features non-finite in more than "
                       f"{CP_MAX_NONFINITE:.0%} of the cells, constant features",
            "stored": "the raw values of the kept features (float32, a remaining non-finite value as NaN); "
                      "score.post_process scales them",
            "scope": "CellProfiler-style features per crop, not the full CellProfiler pipeline on whole sites"}


def pilot_name(arm: str, fit: str, limit: int) -> str:
    """The directory of one pilot setting under <root>/external_pilot/: the arm, openphenom's fit, the cell count."""
    return f"{arm}_{fit}_n{int(limit)}" if arm == "openphenom" else f"{arm}_n{int(limit)}"


def run(root: Path, output_root: Path, arms=EXTERNAL_ARMS, *, workers: int = 1,
        input_size: int = OPENPHENOM_INPUT_SIZE, fit: str = OPENPHENOM_FIT, batch_size: int = OPENPHENOM_BATCH,
        limit: int | None = None) -> dict:
    """Each arm's raw embedding of every FULL cell (rows in prep/cells.csv order) into <root>/external/<arm>/; with
    `limit`, a pilot on the first `limit` cells into <root>/external_pilot/<pilot_name>/. Never overwritten: an
    existing one is kept when it has the same settings, code, prep table and images.npy, else refused. Returns
    {arm: EMBED.json}."""
    from benchmarks.representation_RxRx1.runners.prepare import load

    unknown = sorted(set(arms) - set(EXTERNAL_ARMS))
    if unknown:
        raise SystemExit(f"unknown external arms {unknown}; expected {list(EXTERNAL_ARMS)}")
    if "openphenom" in arms:
        input_size = check_input_size(input_size)
    pilot = limit is not None
    if pilot and int(limit) < 1:
        raise SystemExit(f"NOT_READY: a pilot needs a positive number of cells, not {limit}")
    summary, cells = load(root)
    if pilot:
        cells = cells.iloc[:int(limit)]
    code = sha256_file(Path(__file__))
    binding = {"cells_sha256": summary["cells_sha256"], "crop_array_sha256": summary["crop_array"]["sha256"],
               "n_cells": int(len(cells))}
    plan = {}
    for arm in arms:
        settings = (openphenom_settings(input_size, fit, batch_size) if arm == "openphenom"
                    else cellprofiler_settings())
        out = root / PILOT_DIR / pilot_name(arm, fit, limit) if pilot else root / EXTERNAL_DIR / arm
        if (out / "EMBED.json").exists():
            record = json.loads((out / "EMBED.json").read_text(encoding="utf-8"))
            same = (record["settings"] == json.loads(json.dumps(settings)) and record["external_sha256"] == code
                    and all(record.get(k) == v for k, v in binding.items()))
            if not same:
                raise SystemExit(f"EXTERNAL_STALE: {out} holds another embedding (settings, code, prep or "
                                 "images.npy); move it aside")
            print(f"external {arm}: kept ({out})", flush=True)
            plan[arm] = record
            continue
        plan[arm] = (settings, out)
    todo = {arm: p for arm, p in plan.items() if isinstance(p, tuple)}
    if not todo:
        return plan
    manifest = check_openphenom(output_root) if "openphenom" in todo else None
    images = fs_retry(np.load, summary["crop_array"]["path"])        # plain reads into memory, no memory map
    rows = cells.array_row.astype(int).to_numpy()
    # cellprofiler first: its spawn pool starts before CUDA is initialised in this process
    for arm in sorted(todo, key=lambda a: a != "cellprofiler"):
        settings, out = todo[arm]
        started = time.time()
        if arm == "cellprofiler":
            raw, names, counts = cellprofiler_features(images, rows, workers=workers)
            vectors, kept, dropped = filter_features(raw, names)
            extra = {"packages": versions(CP_PACKAGES), "n_raw_features": len(names), "features": kept,
                     "dropped_features": dropped, "masks": {k: counts[k] for k in CP_MASK_COUNTS},
                     "nonfinite_values": int((~np.isfinite(vectors)).sum()),
                     "measurement_errors": counts["errors"], "workers": int(workers)}
        else:
            import torch

            device = "cuda" if torch.cuda.is_available() else "cpu"
            model = load_openphenom(Path(manifest["snapshot"]), device)
            vectors = openphenom_vectors(model, images, rows, input_size=input_size, fit=fit, batch_size=batch_size,
                                         device=device)
            if not np.isfinite(vectors).all():
                raise SystemExit(f"EXTERNAL_FAILED: {arm} produced non-finite values")
            extra = {"packages": versions(OPENPHENOM_PACKAGES), "device": device,
                     "assets": {k: manifest[k] for k in ("snapshot", "revision", "origin")}}
        partial = out.with_name(out.name + ".partial")
        if partial.exists():
            fs_retry(shutil.rmtree, partial)
        fs_retry(partial.mkdir, parents=True)
        fs_retry(np.save, partial / "embeddings.npy", vectors.astype(np.float32))
        record = {"arm": arm, "label_free": True, "pilot": pilot, "settings": settings, **binding,
                  "dim": int(vectors.shape[1]),
                  "rows": "prep/cells.csv order" + (f", its first {len(vectors)} rows" if pilot else ""),
                  "crop_array": summary["crop_array"]["path"],
                  "embeddings_sha256": sha256_file(partial / "embeddings.npy"), "external_sha256": code,
                  "commit": git_commit(), "seconds": round(time.time() - started, 1), "host": socket.gethostname(),
                  "slurm_job": os.environ.get("SLURM_JOB_ID"), **extra}
        if pilot:            # label-free: the analyzer's sibling-site proxy on the pilot cells, as score would scale
            from shape.tools.cell_state_analyzer import quant

            from benchmarks.representation_RxRx1.runners.score import TABLES, post_process

            meta = cells[["plate", "experiment", "well", "site"]].to_dict("records")
            record["pilot_proxy"] = {}
            for table in TABLES:
                scaled = post_process(arm, vectors, cells.plate.to_numpy(), table)
                proxy = quant.sibling_site_map(scaled, meta, seed=quant.FRESH_SEED)
                record["pilot_proxy"][table] = {k: proxy.get(k) for k in ("value", "chance", "n_queries")}
        write_json(partial / "EMBED.json", record)
        fs_retry(partial.rename, out)
        print(f"external {arm}: {record['n_cells']} cells x {record['dim']}, {record['seconds']} s -> {out}"
              + (f"; pilot proxy {record['pilot_proxy']}" if pilot else ""), flush=True)
        plan[arm] = record
    return plan


def load_embeddings(root: Path, summary: dict, n_cells: int) -> tuple[dict, dict]:
    """({arm: raw vectors}, {arm: status}) of the external arms under <root>/external/ for score.py: a missing arm is
    skipped with a note; one made for another prep table, another images.npy, another cell count or with other code,
    or a changed array, is refused."""
    vectors, status = {}, {}
    code = sha256_file(Path(__file__))
    for arm in EXTERNAL_ARMS:
        out = root / EXTERNAL_DIR / arm
        if not (out / "EMBED.json").exists():
            status[arm] = {"status": "skipped", "note": f"no {out / 'EMBED.json'} (run the external stage)"}
            continue
        record = json.loads((out / "EMBED.json").read_text(encoding="utf-8"))
        if record["cells_sha256"] != summary["cells_sha256"] or record["n_cells"] != n_cells:
            raise SystemExit(f"EXTERNAL_STALE: {out} was made for another prep; move it aside")
        if record.get("crop_array_sha256") != summary["crop_array"]["sha256"]:
            raise SystemExit(f"EXTERNAL_STALE: {out} was made from another images.npy; move it aside")
        if record["external_sha256"] != code:
            raise SystemExit(f"EXTERNAL_STALE: {out} was made with another external.py; move it aside")
        if sha256_file(out / "embeddings.npy") != record["embeddings_sha256"]:
            raise SystemExit(f"INTEGRITY_FAILED: {out / 'embeddings.npy'} changed since the external stage")
        vectors[arm] = np.load(out / "embeddings.npy")
        status[arm] = {"status": "scored", "dim": record["dim"], "settings": record["settings"],
                       "packages": record.get("packages"), "commit": record.get("commit"),
                       "seconds": record.get("seconds"), "embeddings_sha256": record["embeddings_sha256"],
                       "external_sha256": record["external_sha256"],
                       "crop_array_sha256": record["crop_array_sha256"], "masks": record.get("masks")}
    return vectors, status

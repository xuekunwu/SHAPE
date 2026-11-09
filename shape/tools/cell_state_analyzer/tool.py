#!/usr/bin/env python3
"""
Unified Cell State Analyzer - auto-channel self-supervised cell-state representation.
Performs symmetric contrastive learning on cell crops and generates UMAP/clustering outputs.
"""

import copy
import errno
from contextlib import contextmanager
import math
import multiprocessing
import os
import random
import sys
import threading
import time
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, Sampler
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import numpy as np
import json
import logging
from io import BytesIO
from pathlib import Path
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download
except ImportError:
    raise ImportError("huggingface_hub is required for model weight download")

try:
    import anndata as ad
    import scanpy as sc
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("anndata/scanpy not available. Install them for visualizations.")

try:
    from skimage import io
except ImportError:
    raise ImportError("skimage is required for image loading")

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from shape.tools.base import BaseTool
from shape.tools.cell_state_analyzer import quant
from shape.utils.agent_io import VisualizationConfig
from shape.utils.image_data import ImageData
from shape.utils.image_processor import ImageProcessor
from shape.utils.segmentation_qc import to_display_uint8
from shape.utils.adaptive_training import (
    AdaptiveEarlyStopper,
    is_cuda_out_of_memory,
    next_epoch_budget,
    resolve_batch_size,
    resolve_clustering_representation,
    resolve_training_config,
)


def _identity_value(values, idx):
    """None-safe per-sample lookup for an optional identity-companion list.

    Returns `""` (the established "absent" sentinel throughout the stable-identity
    contract -- see `_validate_identity_metadata`) when the whole list was never supplied
    (`values is None`) OR when THIS element is `None` -- e.g. HPA's `mask_id`, which is
    always `None` per record because no real mask manifest exists (`discover_organelles`),
    so `values` itself is a non-empty (truthy) list of `None`s. A prior revision checked
    only `values` truthiness (`values[idx] if values else ""`), which missed exactly this
    case and let a raw `None` reach `torch.utils.data.dataloader.default_collate`, which
    raises `TypeError` on `NoneType` -- crashing the real 13-unit HPA regeneration (no real
    masks) the first time a DataLoader batched more than one sample. Never fabricates a
    value: absence stays absence, just represented in the collatable/string form.
    """
    if values is None:
        return ""
    value = values[idx]
    return value if value is not None else ""


def _channel_first(raw, path="<array>"):
    """Return an image as CxHxW without silently guessing an ambiguous layout."""
    raw = np.asarray(raw)
    if raw.ndim == 2:
        return raw[None, ...], "HW"
    if raw.ndim != 3:
        raise ValueError(f"Unsupported crop shape {raw.shape} for {path}; expected HW, CHW, or HWC")
    # A small cell cropped tightly can be square down to C == H == W (e.g. a 5-channel 5x5 crop),
    # where "channel axis strictly smaller than both spatial axes" can decide neither end. These
    # crops are always written channel-first (materialize_raw_crop: C=spec.channels leading), so on
    # a tie prefer CHW: the first axis is the channel axis when it is <= 16 and no larger than either
    # spatial axis (`<=`), while HWC still needs the last axis strictly smaller (`<`) than both.
    first_is_channel = (
        raw.shape[0] <= 16
        and raw.shape[0] <= raw.shape[1]
        and raw.shape[0] <= raw.shape[2]
    )
    last_is_channel = (
        raw.shape[2] <= 16
        and raw.shape[2] < raw.shape[0]
        and raw.shape[2] < raw.shape[1]
    )
    if first_is_channel == last_is_channel:
        raise ValueError(
            f"Ambiguous channel layout {raw.shape} for {path}; provide an unambiguous "
            "CHW/HWC crop (channel axis must be no larger than both spatial axes)"
        )
    return (raw, "CHW") if first_is_channel else (np.moveaxis(raw, -1, 0), "HWC")


# A transient read failure of a packed array: the node's Lustre client was evicted (errno 108, ESHUTDOWN on Linux)
# or an I/O error on Lustre (errno 5, EIO). `ArrayCrops` reads the rows a call needs once, with plain file reads (a
# failure is an OSError, never the SIGBUS of a memory-mapped page), tried again after each of READ_RETRY_DELAYS.
READ_TRANSIENT_ERRNOS = frozenset({errno.EIO, 108, getattr(errno, "ESHUTDOWN", 108)})
READ_RETRY_DELAYS = (5, 10, 30, 75)      # seconds between the 5 tries of a read
CROP_ARRAY_MAX_CHANNELS, CROP_ARRAY_MIN_SIDE = 16, 8
_READ_CHUNK = 64 << 20                   # bytes per read call


def _read_array_rows(path, rows):
    """The distinct ascending `rows` of the .npy at `path`, read into process memory with plain file reads (one
    pass per contiguous run of rows): {"shape", "dtype", "bytes" (file size), "position" (row -> index),
    "pixels" (len(rows), C, H, W), "rows_sha256" (of the pixels in this order)}."""
    import hashlib

    with open(path, "rb") as handle:
        version = np.lib.format.read_magic(handle)
        if version not in ((1, 0), (2, 0)):
            raise ValueError(f"crop_array {path!r}: .npy format version {version} is not read")
        read_header = (np.lib.format.read_array_header_1_0 if version == (1, 0)
                       else np.lib.format.read_array_header_2_0)
        shape, fortran_order, dtype = read_header(handle)
        if (dtype not in (np.uint8, np.uint16) or len(shape) != 4 or fortran_order
                or not 1 <= shape[1] <= CROP_ARRAY_MAX_CHANNELS or min(shape[2:]) < CROP_ARRAY_MIN_SIDE):
            raise ValueError(f"crop_array {path!r} is {dtype} {shape}{' (Fortran order)' if fortran_order else ''}; "
                             f"expected uint8 or uint16 of shape (N, C, H, W) in C order, C <= "
                             f"{CROP_ARRAY_MAX_CHANNELS}, H and W >= {CROP_ARRAY_MIN_SIDE}")
        outside = [row for row in rows if not 0 <= row < shape[0]]
        if outside:
            raise ValueError(f"array_row outside the {shape[0]} rows of {path!r}: {outside[:5]}")
        offset, size = handle.tell(), os.fstat(handle.fileno()).st_size
        row_bytes = int(np.prod(shape[1:])) * dtype.itemsize
        if offset + shape[0] * row_bytes > size:
            raise ValueError(f"crop_array {path!r} is truncated ({size} bytes for {shape[0]} rows)")
        pixels = np.empty((len(rows), *shape[1:]), dtype=dtype)
        flat = memoryview(pixels.reshape(-1).view(np.uint8))
        digest, start = hashlib.sha256(), 0
        while start < len(rows):
            end = start + 1
            while end < len(rows) and rows[end] == rows[end - 1] + 1:
                end += 1
            handle.seek(offset + rows[start] * row_bytes)
            for lo in range(start * row_bytes, end * row_bytes, _READ_CHUNK):
                part = flat[lo:min(lo + _READ_CHUNK, end * row_bytes)]
                if handle.readinto(part) != len(part):
                    raise ValueError(f"crop_array {path!r}: short read at row {rows[start]}")
                digest.update(part)
            start = end
    return {"shape": tuple(int(n) for n in shape), "dtype": dtype, "bytes": int(size),
            "position": {row: i for i, row in enumerate(rows)}, "pixels": pixels, "rows": tuple(rows),
            "rows_sha256": digest.hexdigest()}


def _read_array_rows_retrying(path, rows):
    """`_read_array_rows`, tried again (READ_RETRY_DELAYS) while it raises an OSError with a READ_TRANSIENT_ERRNOS
    errno and re-raised after the last try; any other error is raised at once."""
    for delay in (*READ_RETRY_DELAYS, None):
        try:
            return _read_array_rows(path, rows)
        except OSError as error:
            if error.errno not in READ_TRANSIENT_ERRNOS or delay is None:
                raise
            logger.warning("reading %s: %s; trying again in %d s", path, error, delay)
            time.sleep(delay)


class ArrayCrops:
    """The crops of one packed array (`execute(crop_array=...)`): crop i is row `rows[i]` of a .npy of shape
    (N, C, H, W), uint8 or uint16, C <= CROP_ARRAY_MAX_CHANNELS, H and W >= CROP_ARRAY_MIN_SIDE. The distinct rows
    are read once into process memory (`_read_array_rows_retrying`; memory = distinct rows x C x H x W x itemsize)
    and every read is served from there. Indexing and iteration give each crop's stable key
    "<array file name>#<row>", which stands wherever a file crop's path stands; `subset` shares the rows in memory,
    `raw` copies a crop's pixels (the same array a file read returns for the same crop). `metadata` (one row per
    crop, or None) enters the embedding fingerprint with the rows. A pickle holds the path, rows and metadata; the
    rows are read again on unpickling."""

    def __init__(self, path, rows, metadata=None, loaded=None):
        self.path = str(path)
        self.rows = [int(row) for row in rows]
        self.name = os.path.basename(self.path)
        self.metadata = None if metadata is None else list(metadata)
        self.loaded = _read_array_rows_retrying(self.path, sorted(set(self.rows))) if loaded is None else loaded

    def __getstate__(self):
        return {"path": self.path, "rows": self.rows, "metadata": self.metadata}

    def __setstate__(self, state):
        self.__init__(state["path"], state["rows"], state["metadata"])

    def __len__(self):
        return len(self.rows)

    def __getitem__(self, idx):
        return f"{self.name}#{self.rows[idx]}"

    def __iter__(self):
        return (self[i] for i in range(len(self)))

    def subset(self, idx):
        return ArrayCrops(self.path, [self.rows[i] for i in idx],
                          None if self.metadata is None else [self.metadata[i] for i in idx], self.loaded)

    def raw(self, idx):
        return np.array(self.loaded["pixels"][self.loaded["position"][self.rows[idx]]])

    def content_record(self):
        """What the embedding reads from the array, independent of its path: shape, dtype, file size and the sha256
        of the rows it reads (each distinct row once, ascending); the read's own digest when they are the rows
        read, else computed from memory."""
        import hashlib

        rows, loaded = tuple(sorted(set(self.rows))), self.loaded
        if rows == loaded["rows"]:
            rows_sha256 = loaded["rows_sha256"]
        else:
            digest = hashlib.sha256()
            for row in rows:
                digest.update(loaded["pixels"][loaded["position"][row]])
            rows_sha256 = digest.hexdigest()
        return {"shape": list(loaded["shape"]), "dtype": str(loaded["dtype"]), "bytes": loaded["bytes"],
                "rows_sha256": rows_sha256}


def array_crops(crop_array, cell_metadata):
    """The `ArrayCrops` of `crop_array`, one crop per `cell_metadata` row at its integer `array_row`."""
    rows = [m.get("array_row") if isinstance(m, dict) else None for m in cell_metadata]
    bad = [i for i, row in enumerate(rows) if isinstance(row, bool) or not isinstance(row, (int, np.integer))]
    if bad:
        raise ValueError(f"crop_array needs an integer array_row in every cell_metadata row; rows {bad[:5]} lack one")
    return ArrayCrops(crop_array, rows, metadata=cell_metadata)


def _crop_pixels(crops, idx):
    """The stored pixels of crop `idx`: its row of the packed array, else its file."""
    return crops.raw(idx) if isinstance(crops, ArrayCrops) else io.imread(crops[idx])


def _crop_subset(crops, rows):
    """The crops (or any per-crop list) at `rows`; a packed array stays one."""
    return crops.subset(rows) if isinstance(crops, ArrayCrops) else [crops[i] for i in rows]


def _crop_name(crops, idx):
    """The image-name fallback of crop `idx`: an array crop's key, a file's name without its extension."""
    path = crops[idx]
    return path if isinstance(crops, ArrayCrops) else os.path.splitext(os.path.basename(path))[0]


def infer_channel_spec(image_paths):
    """Inspect every crop and fail closed when channel count/layout is inconsistent."""
    if not image_paths:
        raise ValueError("Cannot infer channels from an empty crop list")
    if isinstance(image_paths, ArrayCrops):     # one (C, H, W) for every row: its first crop is every crop's spec
        chw, layout = _channel_first(image_paths.raw(0), image_paths[0])
        return int(chw.shape[0]), [layout]
    counts, layouts = [], []
    for path in image_paths:
        chw, layout = _channel_first(io.imread(path), path)
        counts.append(int(chw.shape[0]))
        layouts.append(layout)
    unique_counts = sorted(set(counts))
    if len(unique_counts) != 1:
        examples = [(str(path), count) for path, count in zip(image_paths, counts)][:8]
        raise ValueError(
            f"Mixed channel counts are not supported in one analyzer run: {unique_counts}; "
            f"examples={examples}"
        )
    return unique_counts[0], sorted(set(layouts))


def _normalized_channel_tensor(raw, path):
    chw, _ = _channel_first(raw, path)
    channels = [to_display_uint8(channel).astype(np.float32) / 255.0 for channel in chw]
    return torch.from_numpy(np.stack(channels, axis=0))


class ChannelIntensityJitter(nn.Module):
    """Label-free intensity augmentation that works for any channel count."""
    def __init__(self, strength=0.2):
        super().__init__()
        self.strength = float(strength)

    def forward(self, image):
        scale = torch.empty((image.shape[0], 1, 1), device=image.device).uniform_(
            1.0 - self.strength, 1.0 + self.strength)
        return torch.clamp(image * scale, 0.0, 1.0)


def _ssl116_blur_kernel(sigma, radius=4):
    """The normalised (2r+1) x (2r+1) Gaussian kernel of SSL116View: the task 116 tool's 5 x 5 kernel
    (radius 2) at 112 px, with radius and sigma scaled by 224 / 112."""
    r = torch.arange(-radius, radius + 1, dtype=torch.float32)
    k1 = torch.exp(-(r ** 2) / (2 * float(sigma) ** 2))
    k1 = k1 / k1.sum()
    return k1[:, None] * k1[None, :]


class SSL116View(nn.Module):
    """augmentation=light (named ssl116 before 2026-10-11): the training view of the task 116 SSL hub (agentic_rxrx1 @ f667f94,
    ssl_tool/augment.py `resized_crop` + `photometric` at its defaults, params.py), ported to one
    C x H x W crop in [0, 1] at 224 px, with torch's global RNG:
    1. geometry, one draw shared by all channels: area fraction ~ U[scale_min, 1], log aspect ~
       U[log 3/4, log 4/3], a k x 90-degree rotation and with p 0.5 a mirror of the first axis; one
       affine grid_sample to 224 x 224 (bilinear, reflection padding);
    2. per channel: brightness ~ U[-0.3 j, 0.3 j], contrast ~ U[1 - 0.5 j, 1 + 0.5 j] around the
       channel mean, clamp, then gamma exp(U[-0.5 j, 0.5 j]) (jitter strength j = 0.4);
    3. per channel with p `blur_prob` a Gaussian blur, sigma ~ U[0.6, 3.0] px, 9 x 9 kernel, reflect
       padding (116's U[0.3, 1.5] px and 5 x 5 at 112 px, scaled by 224 / 112);
    4. additive Gaussian noise of std `noise_std`, then clamp to [0, 1].
    `ratio`, `jitter`, `blur_prob` and `noise_std` keep the 116 defaults in the tool."""

    def __init__(self, scale_min=0.3, size=224, ratio=(3 / 4, 4 / 3), jitter=0.4, blur_prob=0.2,
                 noise_std=0.02):
        super().__init__()
        self.scale_min, self.size, self.ratio = float(scale_min), int(size), ratio
        self.jitter, self.blur_prob, self.noise_std = float(jitter), float(blur_prob), float(noise_std)

    @staticmethod
    def _uniform(n, lo, hi):
        return lo + (hi - lo) * torch.rand(n)

    def geometry(self, x):
        area = float(self._uniform(1, self.scale_min, 1.0))
        ratio = math.exp(float(self._uniform(1, math.log(self.ratio[0]), math.log(self.ratio[1]))))
        w, h = min(math.sqrt(area * ratio), 1.0), min(math.sqrt(area / ratio), 1.0)
        cx = (float(torch.rand(1)) * 2 - 1) * (1 - w)
        cy = (float(torch.rand(1)) * 2 - 1) * (1 - h)
        angle = int(torch.randint(0, 4, (1,))) * (math.pi / 2)
        c, s = round(math.cos(angle)), round(math.sin(angle))
        rot = torch.tensor([[c, -s], [s, c]], dtype=torch.float32)
        if float(torch.rand(1)) < 0.5:
            rot[:, 0] = -rot[:, 0]
        theta = torch.zeros(1, 2, 3)
        theta[0, :, :2] = torch.diag(torch.tensor([w, h], dtype=torch.float32)) @ rot
        theta[0, 0, 2], theta[0, 1, 2] = cx, cy
        grid = F.affine_grid(theta, (1, x.shape[0], self.size, self.size), align_corners=False)
        return F.grid_sample(x[None].float(), grid, mode="bilinear", padding_mode="reflection",
                             align_corners=False)[0]

    def photometric(self, x):
        n, j = x.shape[0], self.jitter
        if j > 0:
            brightness = self._uniform(n, -0.3 * j, 0.3 * j).view(n, 1, 1)
            contrast = self._uniform(n, 1 - 0.5 * j, 1 + 0.5 * j).view(n, 1, 1)
            gamma = torch.exp(self._uniform(n, -0.5 * j, 0.5 * j)).view(n, 1, 1)
            mean = x.mean(dim=(1, 2), keepdim=True)
            x = ((x - mean) * contrast + mean + brightness).clamp(0, 1)
            x = x.clamp_min(1e-6) ** gamma
        if self.blur_prob > 0:
            chosen = torch.rand(n) < self.blur_prob
            if chosen.any():
                sigma = self._uniform(int(chosen.sum()), 0.6, 3.0)
                kernels = torch.stack([_ssl116_blur_kernel(s) for s in sigma]).unsqueeze(1)
                pad = kernels.shape[-1] // 2
                sub = x[chosen][None]                                    # (1, m, H, W)
                blurred = F.conv2d(F.pad(sub, (pad, pad, pad, pad), mode="reflect"), kernels,
                                   groups=kernels.shape[0])[0]
                x = x.clone()
                x[chosen] = blurred
        if self.noise_std > 0:
            x = (x + self.noise_std * torch.randn(x.shape)).clamp(0, 1)
        return x

    def forward(self, x):
        return self.photometric(self.geometry(x))


# When set, the environment variable LOADER_WORKERS_ENV caps the workers further (the repository's conftest.py
# sets it to 0, so test DataLoaders load in the test process).
LOADER_WORKERS_ENV = "SHAPE_DATALOADER_WORKERS"


def resolve_num_workers():
    """The DataLoader worker processes of a call, from the hardware: the usable CPUs minus one (the main
    process trains), at least 0, where the start method fork exists (Linux); 0 (every sample is made in the
    main process) where it does not (Windows) or is unsafe (macOS). Usable CPUs: the process's CPU affinity
    (os.sched_getaffinity) where available, else os.cpu_count(), and SLURM_CPUS_PER_TASK when it is set
    and smaller. LOADER_WORKERS_ENV, when set, caps the workers. Returns {"num_workers", "usable_cpus",
    "start_method"}."""
    if "fork" not in multiprocessing.get_all_start_methods() or sys.platform == "darwin":
        return {"num_workers": 0, "usable_cpus": None, "start_method": None}
    try:
        cpus = len(os.sched_getaffinity(0))
    except AttributeError:
        cpus = os.cpu_count() or 1
    try:
        slurm = int(os.environ.get("SLURM_CPUS_PER_TASK", ""))
    except ValueError:
        slurm = 0
    if 0 < slurm < cpus:
        cpus = slurm
    workers = max(0, cpus - 1)
    cap = os.environ.get(LOADER_WORKERS_ENV, "").strip()
    if cap:
        workers = min(workers, max(0, int(cap)))
    return {"num_workers": workers, "usable_cpus": cpus, "start_method": "fork"}


def loader_kwargs(num_workers, pin_memory=False):
    """The DataLoader kwargs of `num_workers` worker processes: forked explicitly (a CUDA-initialised
    parent stays usable; the workers only make CPU tensors), not persistent (each epoch's iterator forks
    them, so they see the main process's state at that point), with pinned batches on CUDA."""
    kwargs = {"num_workers": int(num_workers), "pin_memory": bool(pin_memory)}
    if num_workers:
        short_multiprocessing_tempdir()
        kwargs["multiprocessing_context"] = multiprocessing.get_context("fork")
    return kwargs


def short_multiprocessing_tempdir(limit=60):
    """Workers pass batches through AF_UNIX sockets in multiprocessing's temp dir, under TMPDIR; a long
    TMPDIR (the benchmark drivers' cache tmp) overflows the 108-byte socket path. Then the temp dir is
    a fresh one under /tmp, set before the workers fork so they inherit it, removed at exit."""
    import shutil
    import tempfile
    from multiprocessing import util

    config = multiprocessing.current_process()._config
    if len(config.get("tempdir") or tempfile.gettempdir()) <= limit:
        return
    path = tempfile.mkdtemp(prefix="pymp-", dir="/tmp")
    util.Finalize(None, shutil.rmtree, args=[path], kwargs={"ignore_errors": True}, exitpriority=-100)
    config["tempdir"] = path


def nn_refresh_steps(nn_refresh_epochs, steps_per_epoch):
    """The steps between two rebuilds of the nn_other_image table (0 = never)."""
    return max(1, int(round(nn_refresh_epochs * steps_per_epoch))) if nn_refresh_epochs > 0 else 0


@contextmanager
def one_thread():
    """Run a sample's transform at one CPU thread, as a DataLoader worker does: some kernels
    (grid_sample, conv2d) round differently at other thread counts, so the views would depend on it."""
    threads = torch.get_num_threads()
    if threads == 1:
        yield
        return
    torch.set_num_threads(1)
    try:
        yield
    finally:
        torch.set_num_threads(threads)


# The per-sample random streams of a seeded `CellCropDataset`: one per (seed, epoch, crop, stream).
VIEW_STREAMS, PARTNER_STREAM = (0, 1), 2


class CellCropDataset(Dataset):
    """Unified crop dataset preserving every channel until model-side C-to-3 adaptation.

    seed=None: the transforms and the partner draw use torch's global generator and `units.rng`.
    seed=int: every random draw of a sample comes from its own generators, seeded by (seed, epoch, crop
    index, stream): the two views (torch's CPU generator, forked so the caller's stream is untouched) and
    the partner (numpy). A sample is then a pure function of (seed, epoch, index), whichever process
    makes it and in whichever order, so the batches do not depend on the DataLoader's worker count. The
    main process sets the epoch (`set_epoch`) before it creates the epoch's iterator."""
    def __init__(self, image_paths, groups=None, image_names=None, transform=None,
                 cell_ids=None, mask_ids=None, rel_paths=None, plate_ids=None,
                 units=None, view_pairing="same_crop", affine=None, seed=None):
        self.image_paths = image_paths
        self.groups = groups if groups else ["default"] * len(image_paths)
        self.image_names = image_names
        self.transform = transform
        # Stable per-cell identity (task 087 contract): when supplied, cell_id is the
        # reviewed join key and travels WITH each sample through every DataLoader
        # (train/eval/alignment) -- it is never reattached afterwards by assuming output
        # row order. None (not requested) -> "" per sample, same as a legacy caller.
        self.cell_ids = cell_ids
        self.mask_ids = mask_ids
        self.rel_paths = rel_paths
        self.plate_ids = plate_ids
        # Metadata units of these crops (`CropUnits`, training only): the second view comes from
        # the crop `view_pairing` names, and the rows of both views travel with the sample (the
        # contrastive mask compares their units).
        self.units = units
        self.view_pairing = view_pairing
        # intensity_norm=per_plate: (scale, shift), one (C,) row per crop (`norm_affine`), applied
        # right after the [0, 1] conversion, before any transform; None = the crop as stored.
        self.affine = affine
        self.seed = None if seed is None else int(seed)
        self.epoch = 0

    def set_epoch(self, epoch):
        self.epoch = int(epoch)

    def _stream_seed(self, idx, stream):
        """The seed of stream `stream` of sample `idx` at this epoch (a 64-bit integer)."""
        entropy = [self.seed & 0xFFFFFFFFFFFFFFFF, self.epoch, int(idx), stream]
        return int(np.random.SeedSequence(entropy).generate_state(1, np.uint64)[0])

    def _view(self, img, idx, view):
        with one_thread():
            if self.seed is None:
                return self.transform(img)
            with torch.random.fork_rng(devices=[]):
                torch.random.default_generator.manual_seed(self._stream_seed(idx, VIEW_STREAMS[view]))
                return self.transform(img)

    def __len__(self):
        return len(self.image_paths)

    def _tensor(self, idx):
        path = self.image_paths[idx]
        img = _normalized_channel_tensor(_crop_pixels(self.image_paths, idx), path)
        if self.affine is None:
            return img
        scale, shift = (torch.from_numpy(part[idx]).view(-1, 1, 1) for part in self.affine)
        return torch.clamp(img * scale + shift, 0.0, 1.0)

    def __getitem__(self, idx):
        group = self.groups[idx]

        # Preserve every fluorescence channel; the learnable C->3 adapter is in the model.
        img = self._tensor(idx)
        partner = idx
        if self.units is not None and self.view_pairing != "same_crop":
            rng = None if self.seed is None else np.random.default_rng(self._stream_seed(idx, PARTNER_STREAM))
            partner = int(self.units.partners(np.array([idx]), self.view_pairing, rng)[0])
        other = img if partner == idx else self._tensor(partner)

        # Apply transforms (two augmented views for contrastive learning)
        if self.transform:
            v1 = self._view(img, idx, 0)
            v2 = self._view(other, idx, 1)
        else:
            v1, v2 = img, other

        image_name = (
            self.image_names[idx]
            if self.image_names and idx < len(self.image_names)
            else _crop_name(self.image_paths, idx)
        )
        identity = {
            "cell_id": _identity_value(self.cell_ids, idx),
            "mask_id": _identity_value(self.mask_ids, idx),
            "rel_path": _identity_value(self.rel_paths, idx),
        }
        if self.plate_ids is not None:
            identity["plate_id"] = _identity_value(self.plate_ids, idx)
        if self.units is not None:
            identity["row"] = idx
            identity["partner"] = partner
        return v1, v2, image_name, group, identity


def _chunks(n, batch):
    """Bounds of ceil(n / batch) near-equal chunks of range(n), without single-item chunks."""
    k = max(1, math.ceil(n / max(batch, 1)))
    edges = np.linspace(0, n, k + 1).round().astype(int)
    return [(int(a), int(b)) for a, b in zip(edges[:-1], edges[1:]) if b - a >= 2]


class CropUnits:
    """The plate-qualified site / well and the plate / experiment of one set of crops (the training
    crops, or the val crops of the eval loss; `quant.identities`) for view pairing, contrastive masks
    and per-plate / per-experiment batches. Port of `Cells` (partners, epoch_batches) of the task 116
    SSL hub (agentic_rxrx1 @ f667f94, ssl_tool/data.py); one seeded numpy generator draws the partners
    and the batch order. `nn` is the view_pairing=nn_other_image partner table (`neighbour_table`)."""

    def __init__(self, identities, seed):
        self.columns = {key: np.array([row[key] for row in identities], dtype=object)
                        for key in quant.IDENTITY_KEYS}
        self.rng = np.random.default_rng(seed)
        self._groups = {}
        self.nn = None

    def groups(self, column):
        """{unit: the rows of its crops} of a column every crop has."""
        if column not in self._groups:
            values = self.columns[column].astype(str)
            order = np.argsort(values, kind="stable")
            keys, starts = np.unique(values[order], return_index=True)
            bounds = list(starts[1:]) + [len(values)]
            self._groups[column] = {k: order[s:e] for k, s, e in zip(keys, starts, bounds)}
        return self._groups[column]

    def codes(self, column):
        """One integer per crop: equal codes = the same unit of `column`."""
        return np.unique(self.columns[column].astype(str), return_inverse=True)[1]

    def partners(self, rows, pairing, rng=None):
        """The crop whose augmentation gives each row's second view: another crop of the same site /
        well, drawn at random (the row itself when it is alone there), or (nn_other_image) a random
        one of the row's candidates in the neighbour table (the row itself when it has none). The draws
        come from `rng` (a numpy Generator), default the units' own generator."""
        rng = self.rng if rng is None else rng
        if pairing == "nn_other_image":
            table = self.nn[rows]
            count = (table >= 0).sum(1)
            pick = table[np.arange(len(rows)), rng.integers(np.maximum(count, 1))]
            return np.where(count > 0, pick, rows)
        column = {"same_site": "site", "same_well": "well"}[pairing]
        out = np.array(rows, copy=True)
        for i, row in enumerate(rows):
            members = self.groups(column)[str(self.columns[column][row])]
            if len(members) > 1:
                pick = members[rng.integers(len(members) - 1)]
                out[i] = pick if pick != row else members[-1]
        return out

    def _pools(self, sampling):
        return list(self.groups({"per_plate": "plate", "per_experiment": "experiment"}[sampling]).values())

    def batches_per_epoch(self, batch, sampling):
        return sum(len(_chunks(len(pool), batch)) for pool in self._pools(sampling))

    def epoch_batches(self, batch, sampling):
        """One epoch: every crop once, in batches of <= `batch` drawn within one plate / experiment each,
        in a shuffled order; a pool is split into near-equal chunks and chunks of one crop are dropped
        (nothing to contrast)."""
        out = []
        for pool in self._pools(sampling):
            perm = self.rng.permutation(pool)
            out += [perm[a:b] for a, b in _chunks(len(perm), batch)]
        return [out[i] for i in self.rng.permutation(len(out))]


NN_CHUNK_ENTRIES = 1 << 26      # similarities neighbour_table holds at a time (256 MB in float32)


def neighbour_table(vectors, groups, k, mutual=False, device="cpu", chunk=None):
    """Every row's k nearest rows by cosine similarity among the rows of OTHER groups (plate-qualified
    sites), nearest first: an (N, k) int64 index, -1 where there is none. mutual=True keeps j as a
    neighbour of i only if i is one of j's (the kept ones move to the front, in order); a row left
    without one pairs with itself (`CropUnits.partners`). Computed `chunk` rows at a time (default: at
    most NN_CHUNK_ENTRIES similarities), never as the full N x N matrix. Port of `neighbour_table` of
    the task 116 SSL hub (agentic_rxrx1 @ f667f94, ssl_tool/data.py), in float32 on every device."""
    n = len(vectors)
    x = torch.from_numpy(np.asarray(vectors, dtype=np.float32))
    x = (x / x.norm(dim=1, keepdim=True).clamp_min(1e-12)).to(device)
    codes = torch.as_tensor(np.unique(np.asarray(groups, dtype=str), return_inverse=True)[1], device=device)
    chunk = chunk or max(1, NN_CHUNK_ENTRIES // max(n, 1))
    top = min(k, n)
    index = np.full((n, k), -1, dtype=np.int64)
    for start in range(0, n, chunk):
        sim = x[start:start + chunk] @ x.t()
        sim.masked_fill_(codes[start:start + chunk, None] == codes[None, :], -math.inf)   # own site, itself too
        values, columns = (t.cpu().numpy() for t in sim.topk(top, dim=1))
        index[start:start + chunk, :top] = np.where(np.isfinite(values), columns, -1)
    if mutual:
        anchors = np.broadcast_to(np.arange(n)[:, None], index.shape)
        found = index >= 0
        kept = np.zeros_like(found)
        kept[found] = np.isin(index[found] * n + anchors[found], anchors[found] * n + index[found])
        order = np.argsort(~kept, axis=1, kind="stable")
        index = np.take_along_axis(np.where(kept, index, -1), order, 1)
    return index


class UnitBatchSampler(Sampler):
    """batch_sampling=per_plate / per_experiment: the DataLoader's batches (`CropUnits.epoch_batches`).
    It runs in the main process and draws an epoch at its first batch, not at iter(): a multi-worker
    DataLoader iterator calls iter() on it twice before it reads a batch."""

    def __init__(self, units, batch_size, sampling):
        self.units, self.batch_size, self.sampling = units, int(batch_size), sampling

    def __iter__(self):
        for rows in self.units.epoch_batches(self.batch_size, self.sampling):
            yield [int(row) for row in rows]

    def __len__(self):
        return self.units.batches_per_epoch(self.batch_size, self.sampling)


class ChannelAdapter(nn.Module):
    """Learnable C->3 fluorescence adapter with deterministic initialization."""
    def __init__(self, input_channels):
        super().__init__()
        self.input_channels = int(input_channels)
        self.proj = nn.Conv2d(self.input_channels, 3, kernel_size=1, bias=True)
        self.init_name = self._initialize()

    def _initialize(self):
        with torch.no_grad():
            self.proj.weight.zero_()
            self.proj.bias.zero_()
            if self.input_channels == 1:
                self.proj.weight[:, 0, 0, 0] = 1.0
                return "replicate_1_to_rgb"
            if self.input_channels == 3:
                for channel in range(3):
                    self.proj.weight[channel, channel, 0, 0] = 1.0
                return "identity_rgb"
            if self.input_channels == 6:
                # V6.3 biological initialization.
                for out_channel, in_channels in enumerate(((2, 4), (1, 5), (0, 3))):
                    for in_channel in in_channels:
                        self.proj.weight[out_channel, in_channel, 0, 0] = 0.5
                return "v63_biological_6_to_rgb"
            for in_channel in range(self.input_channels):
                count = len(range(in_channel % 3, self.input_channels, 3))
                self.proj.weight[in_channel % 3, in_channel, 0, 0] = 1.0 / count
            return "deterministic_balanced_to_rgb"

    def forward(self, image):
        return torch.clamp(self.proj(image), 0.0, 1.0)


class DinoV3Projector(nn.Module):
    """DINOv3 plus one unified, learnable input-channel adapter."""
    def __init__(self, backbone_name="dinov3_vitb16", proj_dim=256, freeze_backbone=False,
                 input_channels=1):
        super().__init__()
        
        # Checkpoint per architecture. The filename used to be hard-coded to the vitb16 weights
        # regardless of `backbone_name`, and the load below used strict=False, so ANY other
        # choice silently produced a largely random backbone with no error and a correct-looking
        # feature dim -- a wrong answer that no test could see. Unavailable architectures now
        # fail here instead.
        custom_repo_id = "5xuekun/dinov3_vits16"
        checkpoints = {
            "dinov3_vitb16": "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
        }
        if backbone_name not in checkpoints:
            raise ValueError(
                f"no checkpoint available for backbone_name={backbone_name!r}; "
                f"obtainable architectures: {sorted(checkpoints)}. Add its weights to "
                f"{custom_repo_id} and register the filename here before selecting it."
            )
        model_filename = checkpoints[backbone_name]
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        logger.info(f"Downloading weights from Hugging Face Hub: {custom_repo_id}/{model_filename}")
        ckpt_path = hf_hub_download(
            repo_id=custom_repo_id,
            filename=model_filename,
            token=hf_token
        )
        
        # Prefer a pinned local checkout when the caller requires offline reproducibility.
        local_hub_repo = os.environ.get("SHAPE_DINOV3_REPO")
        if local_hub_repo:
            hub_repo = Path(local_hub_repo).expanduser().resolve()
            if not (hub_repo / "hubconf.py").is_file():
                raise FileNotFoundError(
                    f"SHAPE_DINOV3_REPO does not contain hubconf.py: {hub_repo}"
                )
            logger.info(f"Loading model architecture from local repository: {hub_repo}")
            self.backbone = torch.hub.load(
                str(hub_repo), backbone_name, pretrained=False, source="local"
            )
        else:
            hub_repo = "facebookresearch/dinov3"
            logger.info(f"Loading model architecture from GitHub: {hub_repo}")
            self.backbone = torch.hub.load(
                hub_repo, backbone_name, pretrained=False, source="github"
            )
        
        # Load weights
        logger.info(f"Loading weights from local path: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu", weights_only=True)
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        # strict=False is kept (checkpoints may carry extra bookkeeping keys) but the result is
        # now CHECKED: a correct checkpoint loads this architecture with zero missing keys, so
        # any missing key means the weights do not belong to this backbone. Silently proceeding
        # would leave those tensors at their random init.
        incompatible = self.backbone.load_state_dict(state_dict, strict=False)
        if incompatible.missing_keys:
            missing = list(incompatible.missing_keys)
            raise RuntimeError(
                f"checkpoint {model_filename!r} does not fit backbone {backbone_name!r}: "
                f"{len(missing)} parameter(s) were left at random initialisation "
                f"(e.g. {missing[:3]}). Refusing to run a partly-random backbone."
            )
        if incompatible.unexpected_keys:
            logger.info("Checkpoint carried %d unused key(s); backbone fully loaded.",
                        len(incompatible.unexpected_keys))

        self.channel_adapter = ChannelAdapter(input_channels)
        self.register_buffer(
            "imagenet_mean", torch.tensor((0.485, 0.456, 0.406)).view(1, 3, 1, 1))
        self.register_buffer(
            "imagenet_std", torch.tensor((0.229, 0.224, 0.225)).view(1, 3, 1, 1))

        # Freeze the pretrained backbone when requested: adapter + projection head train.
        # In this mode features MUST be read from the trained projector (see
        # _extract_features), otherwise training would not move the embedding at all.
        if freeze_backbone:
            self.backbone.requires_grad_(False)

        # Projection head - use feat_dim_map like multi-channel version
        feat_dim_map = {
            "dinov3_vits16": 384,
            "dinov3_vits16plus": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
            "dinov3_vith16plus": 1280,
            "dinov3_vit7b16": 4096,
        }
        feat_dim = feat_dim_map[backbone_name]
        
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
    
    def backbone_input(self, x):
        """The adapter output, ImageNet-normalised: what the backbone reads."""
        rgb = self.channel_adapter(x)
        return (rgb - self.imagenet_mean) / self.imagenet_std

    def encode_backbone(self, x):
        return self.backbone(self.backbone_input(x))

    def forward(self, x):
        feats = self.encode_backbone(x)  # CLS token
        z = self.projector(feats)
        return F.normalize(z, dim=-1)


def contrastive_loss(z1, z2, temperature=0.1, source_ids=None, symmetric=True, same_unit=None):
    """Compute contrastive loss for self-supervised learning.

    `source_ids` (int tensor, one per sample) enables FIELD-AWARE negatives: off-diagonal
    pairs from the SAME source field are masked out of the negative set, so two crops of the
    same field (likely the same cell line) are no longer pushed apart as false negatives.
    The diagonal positive (the other augmented view of the same crop) is always kept. This
    is label-free -- it uses the source-image id, never the cell-line label.
    `same_unit` (bool, n x n; the contrastive mask) drops further negatives: column j's second
    view comes from a crop of anchor i's own site / well."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temperature
    drop = None
    if source_ids is not None:
        source_ids = source_ids.to(logits.device)
        drop = source_ids[:, None] == source_ids[None, :]
    if same_unit is not None:
        same_unit = same_unit.to(logits.device)
        drop = same_unit if drop is None else drop | same_unit
    if drop is not None:
        eye = torch.eye(logits.size(0), dtype=torch.bool, device=logits.device)
        logits = logits.masked_fill(drop & ~eye, float("-inf"))
    labels = torch.arange(z1.size(0)).to(z1.device)
    forward_loss = nn.CrossEntropyLoss()(logits, labels)
    if not symmetric:
        return forward_loss
    return 0.5 * (forward_loss + nn.CrossEntropyLoss()(logits.t(), labels))


def gram_loss(patches, target):
    """gram_weight: the MSE between the per-image Gram matrices of the L2-normalised patch tokens
    (B, N, D) of the trained backbone and of the frozen anchor (port of dinov3 `GramLoss(apply_norm=
    True)` at img_level, without its negative-removal options, as the task 116 tool used it)."""
    student = F.normalize(patches.float(), dim=-1)
    anchor = F.normalize(target.float(), dim=-1)
    return F.mse_loss(student @ student.transpose(-1, -2), anchor @ anchor.transpose(-1, -2))


def koleo_loss(x, eps=1e-8):
    """koleo_weight: the KoLeo spreading term of one view's rows (port of dinov3 `KoLeoLoss`): after
    L2 normalisation, -mean(log(distance to the nearest other row + eps)); 0 for fewer than 2 rows."""
    if x.shape[0] < 2:
        return x.float().sum() * 0.0
    x = F.normalize(x.float(), eps=eps, p=2, dim=-1)
    dots = x @ x.t()
    dots.view(-1)[:: x.shape[0] + 1].fill_(-1)
    nearest = dots.argmax(dim=1)
    return -torch.log(nn.PairwiseDistance(2, eps=eps)(x, x[nearest]) + eps).mean()


def set_drop_path(backbone, drop_path):
    """drop_path > 0: block i of n gets sample_drop_ratio = drop_path * i / (n - 1) (port of the task
    116 SSL hub's ssl_tool/model.py `load_backbone`); in training a dinov3 block then lets a random subset
    of max(int(b * (1 - ratio)), 1) of a batch's b crops through each residual branch, scaled by b / subset,
    so at least one crop per batch skips it. A block without `sample_drop_ratio` raises; at 0 nothing is
    set or checked."""
    if drop_path <= 0:
        return
    blocks = list(backbone.blocks)
    for i, block in enumerate(blocks):
        if not hasattr(block, "sample_drop_ratio"):
            raise ValueError(f"drop_path {drop_path}: backbone block {i} has no sample_drop_ratio attribute")
        block.sample_drop_ratio = float(drop_path) * i / max(len(blocks) - 1, 1)


class GradientReversal(torch.autograd.Function):
    @staticmethod
    def forward(ctx, value, coefficient):
        ctx.coefficient = coefficient
        return value.view_as(value)

    @staticmethod
    def backward(ctx, gradient):
        return -ctx.coefficient * gradient, None


class PlateDiscriminator(nn.Module):
    """Disposable nuisance head; it is never used to export representations."""
    def __init__(self, dimensions, n_plates):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(dimensions, 128),
            nn.GELU(),
            nn.Linear(128, n_plates),
        )

    def forward(self, embedding):
        return self.network(GradientReversal.apply(embedding.float(), 1.0))


def plate_mean_alignment_loss(first, second, plate_ids):
    """Penalize differences among anonymous plate means inside a train batch."""
    combined = 0.5 * (first.float() + second.float())
    codes = {value: index for index, value in enumerate(sorted(set(plate_ids)))}
    if len(codes) < 2:
        return combined.sum() * 0.0
    encoded = torch.tensor(
        [codes[value] for value in plate_ids],
        device=combined.device,
        dtype=torch.long,
    )
    means = torch.stack([
        combined[encoded.eq(index)].mean(dim=0)
        for index in range(len(codes))
    ])
    centered = means - means.mean(dim=0, keepdim=True)
    return centered.square().sum(dim=1).mean()


def resolve_plate_nuisance(cell_metadata, mean_weight, adversarial_weight):
    """Enable plate losses only for complete, genuine technical metadata."""
    mean_weight = float(mean_weight)
    adversarial_weight = float(adversarial_weight)
    if mean_weight < 0.0 or adversarial_weight < 0.0:
        raise ValueError("plate nuisance weights must be non-negative")

    plates, sites = [], []
    for metadata in cell_metadata:
        metadata = metadata if isinstance(metadata, dict) else {}
        plate = _technical_value(metadata, "plate", "experiment", "batch")
        site = _technical_value(metadata, "site", "technical_group", "well")
        plates.append(None if plate is None else str(plate))
        sites.append(None if site is None else str(site))

    eligible = all(value is not None for value in plates + sites)
    unique_plates = sorted(set(plates)) if eligible else []
    eligible = eligible and len(unique_plates) >= 2
    per_plate = {}
    if eligible:
        for plate in unique_plates:
            indices = [index for index, value in enumerate(plates) if value == plate]
            per_plate[plate] = {
                "cells": len(indices),
                "sites": len({sites[index] for index in indices}),
            }
        eligible = all(
            values["cells"] >= 2 and values["sites"] >= 2
            for values in per_plate.values()
        )

    return {
        "rule": "complete_metadata_and_at_least_2_plates_2_cells_2_sites_each",
        "eligible": bool(eligible),
        "n_plates": len(unique_plates),
        "per_plate": per_plate,
        "requested_plate_mean_weight": mean_weight,
        "requested_plate_adversarial_weight": adversarial_weight,
        "plate_mean_weight": mean_weight if eligible else 0.0,
        "plate_adversarial_weight": adversarial_weight if eligible else 0.0,
        "plate_ids": plates if eligible else [""] * len(cell_metadata),
    }


def _technical_value(metadata, *names):
    """First non-empty technical identifier, accepting canonical and ``*_id`` keys."""
    metadata = metadata if isinstance(metadata, dict) else {}
    for name in names:
        for key in (name, f"{name}_id"):
            value = metadata.get(key)
            if value is not None and str(value).strip():
                return value
    return None


def resolve_technical_metadata(cell_metadata, image_names):
    """Normalize per-cell technical identities and build the verifier reliability report.
    A well-derived technical group uses the plate-qualified well and site (`quant.identities`),
    so equal well names on two plates are two groups."""
    technical_groups, technical_batches = [], []
    explicit_plate, explicit_well = [], []
    for image_name, metadata, ids in zip(image_names, cell_metadata, quant.identities(cell_metadata)):
        metadata = metadata if isinstance(metadata, dict) else {}
        plate = _technical_value(metadata, "plate", "experiment", "batch")
        well = _technical_value(metadata, "well")
        site = _technical_value(metadata, "site")
        technical_group = _technical_value(metadata, "technical_group")
        if technical_group is None and ids["well"] is not None:
            technical_group = f"{ids['well']}:{ids['site'] if ids['site'] is not None else 'all'}"
        elif technical_group is None and site is not None:
            technical_group = f"{plate}:{site}" if plate is not None else site
        technical_groups.append(str(technical_group or image_name))
        technical_batches.append(str(plate or image_name))
        explicit_plate.append(plate is not None)
        explicit_well.append(well is not None)

    counts = {}
    for source in technical_groups:
        counts[source] = counts.get(source, 0) + 1
    median_crops = float(np.median(list(counts.values()))) if counts else 0.0
    availability = {
        "plate_id": "available" if explicit_plate and all(explicit_plate) else "unavailable",
        "well_id": "available" if explicit_well and all(explicit_well) else "unavailable",
    }
    report = {
        "scale_and_balance": {
            "metadata_availability": availability,
            "median_crops_per_source": median_crops,
        }
    }
    return technical_groups, technical_batches, report


# Adaptation depth -> (zero_shot, freeze_backbone). A single categorical makes the illegal
# combination (zero_shot AND freeze_backbone) unreachable by construction, and gives the agent
# ONE name to decide instead of two booleans whose product includes a meaningless corner.
# This is a PRE-FLIGHT decision: every alternative costs a full fine-tune to evaluate, so it is
# deliberately absent from the mid-loop retune vocabulary (see stage_map).
_ADAPTATION_DEPTH = {
    "zero_shot": (True, False),    # no training at all; pretrained backbone CLS
    "frozen_head": (False, True),  # train only the projector, read it
    "trained": (False, False),     # full fine-tune (tool default)
}


def admissible_adaptation_depths(n_channels=None):
    """All depths are valid because every C->3 initialization is deterministic."""
    return tuple(_ADAPTATION_DEPTH)


def resolve_adaptation_depth(depth, n_channels=None):
    """Pure: map an `adaptation_depth` name to its (zero_shot, freeze_backbone) pair."""
    if depth not in _ADAPTATION_DEPTH:
        raise ValueError(
            f"unknown adaptation_depth {depth!r}; expected one of {sorted(_ADAPTATION_DEPTH)}"
        )
    allowed = admissible_adaptation_depths(n_channels)
    if depth not in allowed:
        raise ValueError(
            f"adaptation_depth {depth!r} is not admissible for {n_channels} channels; expected one of "
            f"{sorted(allowed)}"
        )
    return _ADAPTATION_DEPTH[depth]


# trainable_pct: the share of the backbone's transformer blocks that trains, counted from the last
# block. Ports of `depth_of`, `n_trainable_blocks` and `trainable_names` of the task 116 SSL hub
# (agentic_rxrx1 @ f667f94, ssl_tool/model.py).
def backbone_depth(names):
    """Number of transformer blocks of a backbone, from its parameter names (blocks.<i>.*)."""
    return len({name.split(".")[1] for name in names if name.startswith("blocks.")})


def resolve_trainable_blocks(trainable_pct, depth):
    """Blocks trained at `trainable_pct`: round(pct / 100 * depth), at least 1 when the share is > 0."""
    pct = float(trainable_pct)
    return 0 if pct <= 0 else min(depth, max(1, int(round(pct / 100 * depth))))


def trainable_backbone_names(names, k):
    """Backbone parameter names trained when the last k blocks train: those blocks, the final norms when
    k > 0, and with every block also the patch embedding and the tokens."""
    depth = backbone_depth(names)
    out = []
    for name in names:
        if name.startswith("blocks."):
            if int(name.split(".")[1]) >= depth - k:
                out.append(name)
        elif name.startswith(("norm.", "local_cls_norm.", "cls_norm.")):
            if k > 0:
                out.append(name)
        elif k == depth and not name.startswith("head."):
            out.append(name)
    return out


# Per-cell morphology the cropper may record alongside `area`. Whatever is present is carried
# into `adata.obs`; whatever is absent is simply not materialised, so this degrades gracefully
# against older crop metadata (the HPA crops predate several of these).
_MORPHOLOGY_KEYS = (
    "perimeter", "solidity", "eccentricity", "extent", "orientation",
    "major_axis_length", "minor_axis_length", "equivalent_diameter", "convex_area",
    "filled_area", "euler_number", "feret_diameter_max",
)


# Everything that changes the EMBEDDING. Clustering knobs (cluster_resolution, n_neighbors)
# are deliberately absent: they are applied after the embedding exists, so two runs that differ
# only in those may share one embedding. That is the whole point of the fingerprint -- it is the
# safety condition for reusing a cached embedding instead of paying another fine-tune.
# Anything that alters the feature space MUST be listed here; omitting a parameter would let a
# stale embedding be reused silently, which is far worse than paying for the retrain.
_EMBEDDING_PARAMS = (
    "adaptation_depth", "zero_shot", "freeze_backbone", "learning_rate", "max_epochs",
    "early_stop_loss", "patience", "temperature", "augment_color", "field_aware",
    "backbone_name", "proj_dim", "batch_size", "seed", "input_channels",
    "channel_layout", "channel_adapter_init", "symmetric_info_nce",
    "plate_mean_weight", "plate_adversarial_weight",
    "view_pairing", "contrastive_mask", "batch_sampling", "trainable_pct", "init_checkpoint",
    "split_key", "val_fraction", "output_subdir", "max_minutes", "eval_every_epochs", "embed_only",
    "weight_decay", "head_learning_rate", "patch_embed_lr_mult", "clip_grad_norm", "warmup_fraction",
    "head_warmup_epochs", "gram_weight", "koleo_weight", "intensity_norm", "augmentation",
    "crop_scale_min", "drop_path", "nn_k", "nn_space", "nn_mutual", "nn_refresh_epochs",
)


def embedding_fingerprint(params, crop_paths=()):
    """Stable digest of (embedding-affecting params + the exact crop set).

    Two runs share an embedding only if this matches exactly. The crop set is included
    because the same settings over different cells are a different feature space.
    A packed array (`ArrayCrops`) enters by its content, never its path, so the digest is the
    same in every job and on every node: `ArrayCrops.content_record` plus the (row, metadata)
    of every crop, sorted like the file crops.
    """
    import hashlib

    payload = {k: params.get(k) for k in _EMBEDDING_PARAMS}
    if params.get("min_epochs") is not None:     # only when set: the digests of every other call stay as they were
        payload["min_epochs"] = params["min_epochs"]
    if params.get("early_stop") is not None:     # likewise: the call's early-stopping rule (`resolve_early_stop`)
        payload["early_stop"] = params["early_stop"]
    if isinstance(crop_paths, ArrayCrops):
        meta = crop_paths.metadata if crop_paths.metadata is not None else [None] * len(crop_paths)
        payload["_crops"] = {"array": crop_paths.content_record(), "crops": sorted(
            json.dumps([row, m], sort_keys=True, default=str) for row, m in zip(crop_paths.rows, meta))}
        blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
        return hashlib.sha256(blob.encode("utf-8")).hexdigest()
    crop_records = []
    for raw_path in sorted(str(p) for p in crop_paths):
        digest = hashlib.sha256()
        try:
            with open(raw_path, "rb") as handle:
                for chunk in iter(lambda: handle.read(1024 * 1024), b""):
                    digest.update(chunk)
            content_digest = digest.hexdigest()
        except OSError:
            # Retain deterministic legacy behavior for synthetic/nonexistent paths in
            # tests, while real crops are always content-addressed.
            content_digest = None
        crop_records.append({"path": raw_path, "sha256": content_digest})
    payload["_crops"] = crop_records
    blob = json.dumps(payload, sort_keys=True, default=str, separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


def _identity_field_present(value):
    """Is `value` a real, non-blank identity string?

    `None`, `""`, a whitespace-only string, and any non-string value (a caller bug -- an
    int/list/dict accidentally passed as an id) are all treated as ABSENT. Applied
    identically to `cell_id` and `crop_id` so neither field can smuggle a not-really-an-
    identity value past the fail-closed gate below.
    """
    return isinstance(value, str) and value.strip() != ""


def _validate_identity_metadata(cell_metadata):
    """Fail-closed stable-identity gate (task 087 contract SS1-3).

    Returns the ordered list of `cell_id` (one per row, in `cell_metadata` order) when
    stable identity was requested and is valid; `None` when it was not requested for ANY
    row (legacy behavior: callers keep synthetic `cell_i` row names).

    The TRIGGER for "identity was requested at all" is a present `crop_id` on ANY row --
    not the joint presence of both fields (a prior revision's bug). `crop_id` is the NEW
    stable-identity-contract key: `single_cell_cropper`'s own (unrelated) per-crop metadata
    already carries a `cell_id` key for every existing non-HPA caller of this analyzer (its
    own override-lookup key, format `{group}_{image_name}_cell_{idx:04d}`) but NEVER a
    `crop_id` key, so `crop_id` presence is the only reliable signal that the NEW contract
    is actually in play; gating the trigger on "both fields together" let a `crop_id`-only
    row (crop_id present, cell_id absent -- a genuinely incomplete identity request) get
    silently swallowed into "no identity requested" instead of aborting.

    Once triggered, EVERY row must independently satisfy all of:
      1. a present, non-blank `cell_id`;
      2. a present, non-blank `crop_id`;
      3. `cell_id == crop_id` (the contract's join key is one string, not two independently
         supplied fields that happen to usually agree);
      4. `cell_id` unique across the whole batch.
    Any violation raises `ValueError` (hard-abort, BEFORE any write) -- a `crop_id`-only row,
    a `cell_id`-only row once ANY other row has a real `crop_id`, a mismatched pair, a
    duplicate id, a blank/whitespace-only value, or a non-string value are all rejected the
    same way: fail closed, never silently dropped/normalized/preferred over the other field.
    """
    n = len(cell_metadata)
    ids = [m.get("cell_id") if isinstance(m, dict) else None for m in cell_metadata]
    crop_ids = [m.get("crop_id") if isinstance(m, dict) else None for m in cell_metadata]

    if not any(_identity_field_present(k) for k in crop_ids):
        return None  # legacy: no row carries the new-contract crop_id key at all

    valid = [_identity_field_present(c) and _identity_field_present(k)
             for c, k in zip(ids, crop_ids)]
    n_valid = sum(valid)
    if n_valid != n:
        raise ValueError(
            f"stable cell identity requested for {n_valid}/{n} rows (a present, non-blank "
            "cell_id AND crop_id string is required on every row); partial identity is not "
            "allowed (fail-closed, stable identity contract SS3)"
        )
    if len(set(ids)) != len(ids):
        dupes = sorted({v for v in ids if ids.count(v) > 1})
        raise ValueError(
            f"cell_id must be unique across the batch; duplicate id(s) (showing up to 5): "
            f"{dupes[:5]} (stable identity contract SS3)"
        )
    mismatched = sorted({(c, k) for c, k in zip(ids, crop_ids) if c != k})
    if mismatched:
        raise ValueError(
            f"cell_id must equal crop_id (the stable identity contract defines the join key "
            f"as cell_id == crop_id); mismatched (cell_id, crop_id) pair(s) (showing up to 5): "
            f"{mismatched[:5]} (stable identity contract, 'Stable identity contract' SS1)"
        )
    return ids


def _identity_sequence_digest(ordered_ids):
    """sha256 of the exact ORDERED cell_id sequence (task 087 contract SS6).

    Binds cache reuse to the ordered crop/identity association: the same SET of ids in a
    DIFFERENT order must not reuse a row-associated cache, so order (not just membership)
    is part of the digest -- unlike `embedding_fingerprint`, whose crop-set component is
    deliberately sorted (crop order never affects the embedding itself).
    """
    import hashlib

    blob = json.dumps(list(ordered_ids), separators=(",", ":"))
    return hashlib.sha256(blob.encode("utf-8")).hexdigest()


# Linear LR warmup, as a FRACTION of the planned steps rather than an absolute count, so it
# scales with dataset size instead of asserting one. Unlike a data threshold, this constant
# claims nothing about the data -- it is a property of AdamW + ViT fine-tuning dynamics, where
# the first steps on a pretrained backbone are the ones large enough to damage it.
# There was previously NO warmup and no schedule at all: a flat LR from step 0.
WARMUP_FRACTION = 0.05


EARLY_STOP_DEFAULTS = {"stop_by": "val_proxy", "patience": 2, "overfit_gap": True, "gap_evals": 2, "train_wells": 264}


def resolve_early_stop(early_stop):
    """The call's within-call early-stopping rule; None (the default) = `main`'s: the training-loss stopper.

    With a dict (keys of EARLY_STOP_DEFAULTS), a call with a measurable val split stops on its val proxy instead:
    once the proxy has set no new best for `patience` evaluations in a row (`val_proxy_plateau`), or, with
    `overfit_gap`, once the gap proxy(train wells) - proxy(val wells) has grown at `gap_evals` evaluations in a row
    while the last evaluation set no new val best (`overfit_gap`); never before the stopper's `min_epochs`. The
    proxy on the training wells is measured on a fixed seeded sample of `train_wells` training wells (all their
    cells). Label-free: both proxies read only plate, well and site."""
    if early_stop is None:
        return None
    unknown = set(early_stop) - set(EARLY_STOP_DEFAULTS)
    if unknown:
        raise ValueError(f"early_stop: unknown keys {sorted(unknown)}; allowed {sorted(EARLY_STOP_DEFAULTS)}")
    rule = {**EARLY_STOP_DEFAULTS, **early_stop}
    if rule["stop_by"] != "val_proxy":
        raise ValueError(f"early_stop stop_by must be 'val_proxy', got {rule['stop_by']!r}")
    for key in ("patience", "gap_evals", "train_wells"):
        if int(rule[key]) < 1:
            raise ValueError(f"early_stop {key} must be >= 1, got {rule[key]!r}")
        rule[key] = int(rule[key])
    rule["overfit_gap"] = bool(rule["overfit_gap"])
    return rule


def _train_probe_rows(train_meta, train_rows, n_wells, seed):
    """All the rows of a seeded sample of `n_wells` training wells (plate-qualified), in input order: the
    overfitting monitor's training wells. [] when the metadata names no plate and well."""
    if train_meta is None:
        return []
    keys = [r["well"] if r["plate"] and r["well"] and r["site"] else None for r in quant.identities(train_meta)]
    wells = sorted({k for k in keys if k is not None})
    if not wells:
        return []
    rng = np.random.default_rng(seed)
    chosen = set(rng.choice(wells, size=min(n_wells, len(wells)), replace=False).tolist())
    return [row for row, k in zip(train_rows, keys) if k in chosen]


class ProxyStopper:
    """The val-proxy stop of `resolve_early_stop`: `observe` one evaluation -> (stop, reason or None)."""

    def __init__(self, rule, min_epochs):
        self.rule, self.min_epochs = rule, int(min_epochs)
        self.best, self.since, self.gap_rises, self.last_gap = -math.inf, 0, 0, None

    def observe(self, epoch, proxy, gap=None):
        new_best = proxy is not None and proxy > self.best
        if new_best:
            self.best, self.since = proxy, 0
        else:
            self.since += 1
        if gap is not None and self.last_gap is not None:
            self.gap_rises = self.gap_rises + 1 if gap > self.last_gap else 0
        self.last_gap = gap if gap is not None else self.last_gap
        if epoch < self.min_epochs:
            return False, None
        if self.since >= self.rule["patience"]:
            return True, "val_proxy_plateau"
        if self.rule["overfit_gap"] and not new_best and self.gap_rises >= self.rule["gap_evals"]:
            return True, "overfit_gap"
        return False, None


def warmup_lr(step, warmup_steps, base_lr):
    """LR for `step` (0-based) under linear warmup; `base_lr` once warmup is over."""
    if warmup_steps <= 0 or step >= warmup_steps:
        return base_lr
    # Compute the fraction FIRST and clamp it. Scaling before dividing overshoots the target by
    # one ULP on the final warmup step (`3e-5 * 10 / 10` != `3e-5`), which would hand the
    # optimizer a learning rate fractionally above the one that was calibrated.
    return base_lr * min(1.0, (step + 1) / warmup_steps)


def optimizer_groups(model, learning_rate, head_learning_rate=None, patch_embed_lr_mult=1.0,
                     plate_discriminator=None):
    """The AdamW param groups of the trainable parameters, each {"kind", "lr", "names", "params"}, in
    this order (empty groups are left out): body = the adapter (channel_adapter.*) and the backbone
    (backbone.*) at learning_rate, except patch_embed = backbone.patch_embed.* at learning_rate *
    patch_embed_lr_mult; head = the projector (projector.*) and the plate discriminator
    (plate_discriminator.*) at head_learning_rate (None = learning_rate). The names in group order
    are the optimizer's parameter order, by which the resume state maps the moments."""
    head_lr = learning_rate if head_learning_rate is None else head_learning_rate
    spec = (("body", learning_rate), ("patch_embed", learning_rate * patch_embed_lr_mult), ("head", head_lr))
    members = {kind: ([], []) for kind, _ in spec}
    named = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if plate_discriminator is not None:
        named += [(f"plate_discriminator.{name}", p) for name, p in plate_discriminator.named_parameters()]
    for name, param in named:
        kind = ("patch_embed" if name.startswith("backbone.patch_embed.")
                else "head" if name.startswith(("projector.", "plate_discriminator.")) else "body")
        members[kind][0].append(name)
        members[kind][1].append(param)
    return [{"kind": kind, "lr": lr, "names": members[kind][0], "params": members[kind][1]}
            for kind, lr in spec if members[kind][1]]


# The training levers beyond the historical flags (their first value is the default) and the metadata
# column a value needs on every crop.
_LEVER_CHOICES = {
    "view_pairing": ("same_crop", "same_site", "same_well", "nn_other_image"),
    "contrastive_mask": ("off", "same_site", "same_well"),
    "batch_sampling": ("random", "per_plate", "per_experiment"),
    "intensity_norm": ("off", "per_plate"),
    "augmentation": ("strong", "light"),
    "nn_space": ("frozen", "current"),
}
# the augmentation values before 2026-10-11 (named by origin, not by what they do): accepted, run as the new names
AUGMENTATION_ALIASES = {"shape": "strong", "ssl116": "light"}

_LEVER_COLUMNS = {"same_site": "site", "same_well": "well", "per_plate": "plate", "per_experiment": "experiment",
                  "nn_other_image": "site"}
CROP_SCALE_RANGE = (0.05, 1.0)
# The val evaluation of a call with a split: besides the proxy, a label-free eval loss -- the call's
# InfoNCE on the first EVAL_CELLS val crops with fixed augmentations and a fixed view pairing within
# val (seed EVAL_SEED), in batches of EVAL_BATCH, comparable across epochs and calls (port of the eval
# losses of the task 116 SSL hub, agentic_rxrx1 @ f667f94, ssl_tool/train.py).
EVAL_CELLS = 512
EVAL_BATCH = 128
EVAL_SEED = 12345
# The full training state a call with a split writes (`init_checkpoint` resumes from either).
RESUME_BEST, RESUME_END = "resume_best.pt", "resume_end.pt"
_RESUME_KEYS = ("model", "optimizer", "scaler", "rng", "counters", "config")
# A transient filesystem failure: the node's Lustre client was evicted (errno 108, ESHUTDOWN on Linux); the mount
# works again after seconds to minutes. Every checkpoint write retries it (`save_retrying`).
TRANSIENT_ERRNOS = frozenset({108, getattr(errno, "ESHUTDOWN", 108)})
SAVE_RETRY_DELAYS = (5, 10, 30, 75)      # seconds between the 5 tries of a write


def save_retrying(obj, path):
    """torch.save(obj) to `path` through a temporary file; the write is tried again (SAVE_RETRY_DELAYS) while it
    raises an OSError with a TRANSIENT_ERRNOS errno and re-raised after the last try. The object is serialised in
    memory first, because torch.save's own file writer turns an OSError into a RuntimeError without its errno."""
    buffer = BytesIO()
    torch.save(obj, buffer)
    for delay in (*SAVE_RETRY_DELAYS, None):
        try:
            with open(path + ".tmp", "wb") as handle:
                handle.write(buffer.getbuffer())
            os.replace(path + ".tmp", path)
            return
        except OSError as error:
            if error.errno not in TRANSIENT_ERRNOS or delay is None:
                raise
            logger.warning("writing %s: %s; trying again in %d s", path, error, delay)
            time.sleep(delay)


def resolve_split(cell_metadata, split_key=None, val_fraction=None):
    """"train" / "val" per crop, or None (no split): the `split_key` metadata values (any other value
    raises), else the val wells `quant.assign_val_wells` assigns per plate at `val_fraction`."""
    if split_key is not None:
        values = [m.get(split_key) if isinstance(m, dict) else None for m in cell_metadata]
        bad = sorted({repr(v) for v in values if v not in ("train", "val")})
        if bad:
            raise ValueError(f"split_key {split_key!r}: every crop needs 'train' or 'val', got {bad[:5]}")
        return values
    if val_fraction is not None:
        return quant.assign_val_wells(cell_metadata, val_fraction=val_fraction)
    return None


# intensity_norm=per_plate (port of `channel_stats`, `plate_reference` and `norm_affine` of the task
# 116 SSL hub, agentic_rxrx1 @ f667f94, ssl_tool/data.py).
def crop_channel_stats(paths):
    """Per crop and channel mean / std of the [0, 1] crop tensor: two float64 arrays (N, C)."""
    stats = [(x.mean(dim=(1, 2)).double().numpy(), x.std(dim=(1, 2), unbiased=False).double().numpy())
             for x in (_normalized_channel_tensor(_crop_pixels(paths, i), paths[i]) for i in range(len(paths)))]
    return np.stack([m for m, _ in stats]), np.stack([s for _, s in stats])


def plate_reference(paths, plates):
    """The intensity reference of a training set: the global channel mean / std = the means over its
    crops, and per plate the means over that plate's crops. Plain lists, as the resume state stores it."""
    mean, std = crop_channel_stats(paths)
    plates = np.asarray(plates, dtype=str)
    return {"global_mean": mean.mean(0).tolist(), "global_std": std.mean(0).tolist(),
            "plates": {str(plate): [mean[plates == plate].mean(0).tolist(), std[plates == plate].mean(0).tolist()]
                       for plate in sorted(set(plates))}}


def norm_affine(paths, plates, reference):
    """Per crop (scale, shift), float32 (N, C) each, so that x * scale + shift maps the channel mean /
    std of the crop's plate onto the reference's global ones; a plate absent from `reference` uses the
    statistics of its own crops among `paths`."""
    plates = np.asarray(plates, dtype=str)
    stats = {plate: tuple(np.asarray(v, dtype=np.float64) for v in pair)
             for plate, pair in reference["plates"].items()}
    for plate in sorted(set(plates) - set(stats)):
        mean, std = crop_channel_stats(_crop_subset(paths, np.flatnonzero(plates == plate)))
        stats[plate] = (mean.mean(0), std.mean(0))
    g_mean = np.asarray(reference["global_mean"], dtype=np.float64)
    g_std = np.asarray(reference["global_std"], dtype=np.float64)
    scale = np.empty((len(plates), len(g_mean)), dtype=np.float32)
    shift = np.empty_like(scale)
    for i, plate in enumerate(plates):
        p_mean, p_std = stats[plate]
        s = g_std / np.maximum(p_std, 1e-6)
        scale[i], shift[i] = s, g_mean - p_mean * s
    return scale, shift


def analysis_dir(query_cache_dir, output_subdir=None):
    """`<query_cache_dir>/cell_state_analysis`, or `<...>/cell_state_analysis/<output_subdir>` when a caller
    keeps one call's files apart from the other calls on the same query (every call writes the same
    file names). `output_subdir` must be a relative path inside cell_state_analysis."""
    base = os.path.join(query_cache_dir or "solver_cache/temp", "cell_state_analysis")
    if not output_subdir:
        return base
    sub = os.path.normpath(str(output_subdir))
    if os.path.isabs(sub) or sub == ".." or sub.startswith(".." + os.sep):
        raise ValueError(f"output_subdir must be a relative path inside cell_state_analysis, got {output_subdir!r}")
    return os.path.join(base, sub)


def _rng_state(loader_generator=None, units=None):
    """The random streams of a training state -- torch (CPU, CUDA), numpy, python, the DataLoader's
    shuffle generator and the metadata units' generator -- in types torch.load(weights_only=True) reads."""
    name, keys, pos, has_gauss, cached = np.random.get_state()
    return {
        "torch": torch.get_rng_state(),
        "cuda": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
        "numpy": [name, keys.tolist(), int(pos), int(has_gauss), float(cached)],
        "python": random.getstate(),
        "loader": None if loader_generator is None else loader_generator.get_state(),
        "units": None if units is None else json.dumps(units.rng.bit_generator.state),
    }


def _restore_rng(state, loader_generator=None, units=None):
    torch.set_rng_state(state["torch"])
    if state["cuda"] and len(state["cuda"]) == torch.cuda.device_count():
        torch.cuda.set_rng_state_all(state["cuda"])
    name, keys, pos, has_gauss, cached = state["numpy"]
    np.random.set_state((name, np.asarray(keys, dtype=np.uint32), pos, has_gauss, cached))
    random.setstate(state["python"])
    if loader_generator is not None and state["loader"] is not None:
        loader_generator.set_state(state["loader"])
    if units is not None and state["units"] is not None:
        units.rng.bit_generator.state = json.loads(state["units"])


def save_resume_state(path, model, optimizer, names, scaler, counters, config,
                      loader_generator=None, units=None, intensity_reference=None):
    """Write the full training state (resume_best.pt / resume_end.pt; port of `resume_state` of the task
    116 SSL hub's ssl_tool/train.py): the model's weights (backbone, adapter, projector), the AdamW
    moments by parameter name (`names`: the optimizer's parameters in its order; the plate
    discriminator's are skipped, it is rebuilt by every call), the GradScaler, the random streams, the
    epoch / step counters, the call's configuration and, with intensity_norm=per_plate, the intensity
    reference (`plate_reference`). Written with `save_retrying`."""
    moments = optimizer.state_dict()["state"]
    state = {
        "model": model.state_dict(),
        "optimizer": {name: moments[i] for i, name in enumerate(names)
                      if i in moments and not name.startswith("plate_discriminator.")},
        "scaler": scaler.state_dict(),
        "rng": _rng_state(loader_generator, units),
        "counters": dict(counters),
        "config": json.loads(json.dumps(config, default=str)),
        "intensity_reference": intensity_reference,
    }
    save_retrying(state, path)


def load_resume_state(path):
    """A resume state an earlier call wrote (`save_resume_state`); ValueError for any other file."""
    state = torch.load(path, map_location="cpu", weights_only=True)
    missing = [key for key in _RESUME_KEYS if not isinstance(state, dict) or key not in state]
    if missing:
        raise ValueError(f"init_checkpoint {path!r} is not a resume state of this tool (missing {missing})")
    return state


def _load_moments(optimizer, names, saved):
    """Load the saved AdamW moments of the parameters that are still trainable (`names`, in the
    optimizer's order); a parameter that newly trains starts at zero. Returns (kept, new). The saved
    tensors are cloned, so a failed attempt (OOM retry) cannot change what the next one loads."""
    state_dict = optimizer.state_dict()
    for i, name in enumerate(names):
        if name in saved:
            state_dict["state"][i] = {key: value.clone() if torch.is_tensor(value) else value
                                      for key, value in saved[name].items()}
    optimizer.load_state_dict(state_dict)
    kept = sum(name in saved for name in names)
    return kept, len(names) - kept


def _save_loss_curve_png(history, path, size=(1000, 600), margin=60):
    """The training-loss curve as a plain PNG drawn with PIL: matplotlib is not thread-safe, so a call
    in a worker thread (wave mode) draws it this way."""
    from PIL import ImageDraw

    width, height = size
    image = Image.new("RGB", size, "white")
    draw = ImageDraw.Draw(image)
    draw.line([(margin, margin), (margin, height - margin), (width - margin, height - margin)], fill="black")
    finite = [(i, float(v)) for i, v in enumerate(history) if np.isfinite(v)]
    if finite:
        low, high = min(v for _, v in finite), max(v for _, v in finite)
        step = (width - 2 * margin) / max(len(history) - 1, 1)
        points = [(margin + i * step, height - margin - (height - 2 * margin) * (v - low) / ((high - low) or 1.0))
                  for i, v in finite]
        draw.line(points + points[-1:], fill=(31, 119, 180), width=2)
        draw.text((4, margin - 6), f"{high:.4f}", fill="black")
        draw.text((4, height - margin - 6), f"{low:.4f}", fill="black")
    draw.text((width // 2 - 40, margin // 3), "Training Loss", fill="black")
    draw.text((width // 2 - 15, height - margin // 2), "Epoch", fill="black")
    image.save(path)


def _per_cell_scalars(cell_metadata, key, n_obs):
    """One float per cell for `key`, NaN where absent/non-numeric; all-NaN on a length mismatch."""
    vals = [m.get(key, np.nan) if isinstance(m, dict) else np.nan for m in cell_metadata]
    if len(vals) != n_obs:
        return np.full(n_obs, np.nan, dtype=float)
    out = np.empty(n_obs, dtype=float)
    for i, v in enumerate(vals):
        out[i] = float(v) if isinstance(v, (int, float)) and not isinstance(v, bool) else np.nan
    return out


def _load_kept_crops_manifest(metadata_file_path):
    """Task 038 C4: the kept-crops manifest sidecar for ONE metadata file, written by
    `llm_recommend.write_kept_manifest` (task 038 M-B; the shared population-triage
    lander the cropper's `vision_recommend_crop` adapter calls) when the crop-stage vision
    arbitration ran (see that function's docstring for the wiring gap this closes --
    `vision_reject`/`is_crop_valid` previously had no downstream teeth; this tool globbed
    every crop PNG in the metadata file regardless). Returns the set of KEPT crop paths, or
    None when no manifest exists (no vision engine ran, or pre-038 metadata) -- callers then
    apply NO filtering, today's unfiltered set."""
    if not metadata_file_path:
        return None
    basename = os.path.basename(metadata_file_path)
    if 'cell_crops_metadata_' not in basename:
        return None
    manifest_path = os.path.join(
        os.path.dirname(metadata_file_path),
        basename.replace('cell_crops_metadata_', 'cell_crops_kept_manifest_'))
    if not os.path.exists(manifest_path):
        return None
    try:
        with open(manifest_path, 'r') as fh:
            return set(json.load(fh).get('kept_crop_paths', []))
    except Exception as e:
        logger.warning(f"Error reading kept-crops manifest {manifest_path}: {e}")
        return None


class Cell_State_Analyzer_Tool(BaseTool):
    """
    Analyzes cell states using self-supervised learning (contrastive learning).
    Performs training on single-cell crops and generates UMAP visualizations with clustering.
    Automatically detects 1..N channel crops and adapts them to DINOv3.
    """
    
    def __init__(self):
        super().__init__(
            tool_name="Cell_State_Analyzer_Tool",
            tool_description="Unified self-supervised cell-state analyzer. Automatically detects crop channel count, applies a learnable C-to-3 adapter, extracts DINOv3 representations, and performs UMAP/clustering. REQUIRES: Single_Cell_Cropper_Tool output.",
            tool_version="1.0.0",
            input_types={
                "cell_crops": "List[str] - List of cell crop image paths from Single_Cell_Cropper_Tool output.",
                "cell_metadata": "List[dict] - List of metadata dictionaries for each cell (must include 'group' field for multi-group analysis).",
                "max_epochs": "Optional int - Safety cap. By default it is derived from dataset size and optimizer steps.",
                "early_stop_loss": "Optional float - Explicit absolute loss target. By default convergence-based stopping is used.",
                "batch_size": "Optional int - Upper bound; selected automatically from live memory and transformed crop size by default.",
                "learning_rate": "float - Learning rate for optimizer (default: 3e-5).",
                "seed": "int - Random seed recorded in the embedding fingerprint (default: 0).",
                "plate_mean_weight": "float - Optional plate-mean nuisance penalty; zero unless eligible metadata exists.",
                "plate_adversarial_weight": "float - Optional gradient-reversal plate penalty; zero unless eligible metadata exists.",
                "cluster_resolution": "float - Resolution for Leiden clustering (default: 0.5).",
                "n_pcs": "Optional int or 'auto' - Clustering representation. Default None keeps the raw embedding (no reduction, unchanged). 'auto' derives PCA dimensionality from the embedding's own eigenvalue spectrum (broken-stick rule); an explicit int is honored exactly.",
                "query_cache_dir": "str - Directory for caching results.",
            },
            output_type="dict - Analysis results with training history, UMAP visualizations, and cluster analysis.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(cell_crops=cell_crops, cell_metadata=cell_metadata)',
                    "description": "Train model and generate UMAP visualizations with default parameters."
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for training. All crops in one run must have the same channel count. Requires anndata and scanpy for advanced visualizations.",
                "best_practice": "MUST use output from Single_Cell_Cropper_Tool. Input should be individual crop images, not original images or masks. Include 'group' field in cell_metadata for multi-group cluster composition analysis."
            },
            output_dir="output_visualizations"
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Cell_State_Analyzer_Tool: Using device: {self.device}")
    
    def _get_augmentation_transform(self, augment_color=True, augmentation="strong", crop_scale_min=0.3):
        """Get augmentation transform for training.

        `augment_color=False` drops the channel-wise intensity jitter; geometry,
        blur, and erasing are kept. The tensor transform supports any channel count.
        `augmentation="light"` replaces the whole pipeline by `SSL116View` (augment_color is inert
        there); `crop_scale_min` is the lower area bound of the resized crop under either.
        """
        augmentation = AUGMENTATION_ALIASES.get(augmentation, augmentation)
        if augmentation not in _LEVER_CHOICES["augmentation"]:
            raise ValueError(f"augmentation must be one of {list(_LEVER_CHOICES['augmentation'])}, "
                             f"got {augmentation!r}")
        if not CROP_SCALE_RANGE[0] <= crop_scale_min <= CROP_SCALE_RANGE[1]:
            raise ValueError(f"crop_scale_min must be in {list(CROP_SCALE_RANGE)}, got {crop_scale_min!r}")
        if augmentation == "light":
            return SSL116View(crop_scale_min)
        color_ops = [transforms.RandomApply([ChannelIntensityJitter(0.2)], p=0.8)] if augment_color else []
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(crop_scale_min, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(60),
            *color_ops,
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.8),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
    
    def _get_eval_transform(self):
        """Get evaluation transform (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
        ])
    
    def _extract_features(self, model, dataloader, use_projector=False):
        """Extract features from the model.

        Default (`use_projector=False`): the backbone CLS token, with the projector
        discarded -- the historical behavior, used when the backbone itself is
        fine-tuned. When `use_projector=True` (freeze-backbone mode) the projector is
        the ONLY part that learned, so its (normalized) output is returned instead;
        otherwise training would not affect the embedding at all.

        Also collects the per-sample identity dict the dataset yields (task 087): the
        stable cell_id (and optional mask_id/rel_path) travels WITH each sample through
        this same DataLoader, in the exact order features are produced -- it is never
        reattached afterwards by indexing back into the caller's original crop list.
        """
        model.eval()
        feats, names, groups, identities = [], [], [], []
        proj_backup = model.projector
        if not use_projector:
            model.projector = nn.Identity()

        with torch.no_grad():
            for v1, _, n, g, ident in tqdm(dataloader, desc="Extracting features"):
                v1 = v1.to(self.device)

                # Frozen-backbone mode: backbone -> trained projector (model.forward).
                # Otherwise: the backbone CLS token directly.
                out = model(v1) if use_projector else model.encode_backbone(v1)
                feats.append(out.cpu())
                names.extend(n)
                groups.extend(g)
                # `ident` is a dict-of-lists after default_collate (one list per key,
                # aligned with this batch); re-assemble into one dict per sample.
                identities.extend(
                    {key: ident[key][i] for key in ident} for i in range(len(n))
                )

        model.projector = proj_backup
        return torch.cat(feats, dim=0).numpy(), names, groups, identities

    def _alignment_consistency(self, model, dataset_aug, sample=64):
        """Mean cosine between two AUGMENTED views of the same crop (alignment): high =
        the embedding is augmentation-consistent. A label-free training-health signal (the
        'neighbour consistency' the pre-flight lr/epoch calibration uses)."""
        model.eval()
        n = len(dataset_aug)
        if n == 0:
            return None
        sims = []
        with torch.no_grad():
            for i in range(min(sample, n)):
                v1, v2, _, _, _ = dataset_aug[i]   # two augmented views of crop i
                f1 = F.normalize(model.encode_backbone(v1.unsqueeze(0).to(self.device)), dim=1)
                f2 = F.normalize(model.encode_backbone(v2.unsqueeze(0).to(self.device)), dim=1)
                sims.append(float((f1 * f2).sum().item()))
        return float(np.mean(sims)) if sims else None

    def _batch_loss(self, model, v1, v2, names, identity, temperature, field_aware, mask_units,
                    patches=False):
        """(z1, z2, InfoNCE, views) of one batch, with the field-aware negatives and the contrastive
        mask (`mask_units`: the mask unit of every crop of the loader's dataset, compared between the
        rows of the anchors and those of the second views). views: per view {"cls": the backbone CLS
        (the exported embedding; z is the projector of it, as in model.forward)}, with patches=True
        from one forward_features pass that also gives "tokens" (the patch tokens) and "input" (the
        backbone input) for the Gram anchor."""
        # Field-aware negatives: integer-code the per-crop source id (image_name)
        # within the batch, so contrastive_loss can mask same-field off-diagonals.
        source_ids = None
        if field_aware:
            order = {s: i for i, s in enumerate(dict.fromkeys(names))}
            source_ids = torch.tensor([order[s] for s in names], device=self.device)
        same_unit = None
        if mask_units is not None:
            rows, partners = identity["row"].numpy(), identity["partner"].numpy()
            same_unit = torch.from_numpy(mask_units[rows][:, None] == mask_units[partners][None, :])
        views, z = [], []
        for v in (v1, v2):
            if patches:
                inputs = model.backbone_input(v)
                out = model.backbone.forward_features(inputs)
                view = {"cls": out["x_norm_clstoken"], "tokens": out["x_norm_patchtokens"], "input": inputs}
            else:
                view = {"cls": model.encode_backbone(v)}
            views.append(view)
            z.append(F.normalize(model.projector(view["cls"]), dim=-1))
        z1, z2 = z
        return z1, z2, contrastive_loss(z1, z2, temperature=temperature, source_ids=source_ids,
                                        same_unit=same_unit), views

    @staticmethod
    def _val_proxy(vectors, meta):
        """The val proxy of the val crops' embedding `vectors` (rows aligned with `meta`):
        `quant.sibling_site_map` at k = 2, 20 draws, on the selection draws `quant.SELECTION_SEED`."""
        return quant.sibling_site_map(vectors, meta, k=2, draws=20, seed=quant.SELECTION_SEED)["value"]

    def _evaluate_val(self, model, validation, temperature, field_aware):
        """One val evaluation of a call with a split: the val crops' embedding (the call's feature
        source) -> the proxy (`_val_proxy`), and the eval loss (InfoNCE only)."""
        vectors, _, _, _ = self._extract_features(
            model, validation["loader"], use_projector=validation["use_projector"])
        proxy = self._val_proxy(vectors, validation["meta"])
        train_proxy = None
        if validation.get("train_loader") is not None:      # the overfitting monitor (`resolve_early_stop`)
            train_vectors, _, _, _ = self._extract_features(
                model, validation["train_loader"], use_projector=validation["use_projector"])
            train_proxy = self._val_proxy(train_vectors, validation["train_meta"])
        # The same view pairing and augmentations at every evaluation: a seeded loss dataset uses the
        # per-sample streams of seed EVAL_SEED at epoch 0; an unseeded one the streams reset here.
        if validation["loss_units"] is not None:
            validation["loss_units"].rng = np.random.default_rng(EVAL_SEED)
        total, count = 0.0, 0
        model.eval()
        with torch.no_grad(), torch.random.fork_rng(devices=[]):
            torch.manual_seed(EVAL_SEED)
            for v1, v2, names, _, identity in validation["loss_loader"]:
                if len(names) < 2:
                    continue
                with autocast():
                    _, _, loss, _ = self._batch_loss(
                        model, v1.to(self.device), v2.to(self.device), names, identity,
                        temperature, field_aware, validation["mask_units"])
                total += float(loss) * len(names)
                count += len(names)
        model.train()
        record = {"sibling_site_map": proxy, "eval_loss": total / count if count else None}
        if validation.get("train_loader") is not None:
            record["train_sibling_site_map"] = train_proxy
            record["overfit_gap"] = (None if train_proxy is None or proxy is None else train_proxy - proxy)
        return record

    def _train_model(self, model, train_loader, max_epochs, early_stop_loss, learning_rate,
                     output_dir, training_logs=None, patience=None, min_epochs=None,
                     weight_decay=1e-4, temperature=0.1, field_aware=False,
                     plate_ids=None, plate_mean_weight=0.0,
                     plate_adversarial_weight=0.0,
                     warmup_fraction=None, batch_size=None, units=None,
                     mask_units=None, validation=None, save_state=False, deadline=None,
                     resume=None, call_config=None, head_learning_rate=None,
                     patch_embed_lr_mult=1.0, clip_grad_norm=0.0, head_warmup_epochs=0.0,
                     gram_weight=0.0, gram_anchor=None, koleo_weight=0.0,
                     refresh_neighbours=None, nn_refresh_epochs=0.0, intensity_reference=None,
                     early_stop=None):
        """Train the model with contrastive learning.

        field_aware: mask same-source-field crops out of each other's negatives (label-free,
        via the per-crop source id carried as the dataset's image_name).
        units / mask_units: the training crops' `CropUnits` (their generator draws the partners and
        the per-plate batches) and the contrastive mask's unit of every training crop.
        validation: the val crops of a split; they are evaluated (`_evaluate_val`) every
        `validation["every"]` epochs and after the last one. The best point is the evaluation with
        the highest proxy when the proxy has a query unit (best_by "val_proxy"), else the epoch
        with the lowest training loss (best_by "train_loss").
        save_state: write the full training state at the best point and at the end
        (resume_best.pt / resume_end.pt, `save_resume_state`, with `call_config` and
        `intensity_reference`).
        deadline: the time.time() at which training stops (max_minutes), checked after each step.
        resume: the resume state the model's weights were loaded from (`init_checkpoint`): the
        AdamW moments of the parameters that are still trainable, the GradScaler and the counters
        continue, the random streams too when the seed is the same.
        head_learning_rate / patch_embed_lr_mult: the param groups (`optimizer_groups`).
        warmup_fraction: a linear warmup over that fraction of this call's initial step budget;
        None = 5 % (WARMUP_FRACTION) on a fresh call, none on a resumed one. Steps are counted from
        the call's own first step (the counter continues from the parent on resume).
        head_warmup_epochs: for that many epochs' steps only the head trains (the adapter and the
        backbone get no gradient, their moments stay); their warmup starts after it.
        clip_grad_norm > 0: the global gradient norm of every trainable parameter is clipped to it
        (after unscaling); the pre-clip norms give grad_norm_median and clipped_fraction.
        gram_weight / gram_anchor: the Gram anchoring term against the frozen anchor backbone.
        koleo_weight: the KoLeo term on the backbone CLS of both views.
        refresh_neighbours / nn_refresh_epochs: rebuild the nn_other_image table from the model
        every round(nn_refresh_epochs x steps per epoch) steps.
        min_epochs: the early stopper's floor (None = `resolve_training_config`'s rule).
        early_stop: the resolved within-call rule (`resolve_early_stop`); None = the training-loss stopper. With
        a rule and a measurable val split, the call stops on its val proxy (`ProxyStopper`), the training-loss
        plateau no longer stops it, and every val evaluation is recorded with its overfitting gap."""
        config = resolve_training_config(
            len(train_loader.dataset), batch_size or train_loader.batch_size, max_epochs, patience, min_epochs
        )
        max_epochs = config["max_epochs"]
        hard_cap = config["max_epochs_hard_cap"]
        stopper = AdaptiveEarlyStopper(
            min_epochs=config["min_epochs"],
            patience=config["patience"],
            relative_min_delta=config["relative_min_delta"],
            absolute_loss_target=early_stop_loss,
        )
        plate_codebook = {
            value: index for index, value in enumerate(sorted(set(plate_ids or [])))
            if value != ""
        }
        plate_discriminator = (
            PlateDiscriminator(256, len(plate_codebook)).to(self.device)
            if plate_adversarial_weight > 0.0 and len(plate_codebook) > 1
            else None
        )
        groups = optimizer_groups(model, learning_rate, head_learning_rate, patch_embed_lr_mult,
                                  plate_discriminator)
        param_names = [name for group in groups for name in group["names"]]
        params = [param for group in groups for param in group["params"]]
        body = [param for group in groups if group["kind"] != "head" for param in group["params"]]
        optimizer = optim.AdamW(
            [{"params": group["params"], "lr": group["lr"], "base_lr": group["lr"], "kind": group["kind"]}
             for group in groups],
            lr=learning_rate, weight_decay=weight_decay)
        scaler = GradScaler(enabled=self.device.type == "cuda")
        steps_per_epoch = max(1, len(train_loader))
        # Warmup is sized from the INITIAL budget; it only covers the opening steps, so a later
        # budget extension does not change it.
        if warmup_fraction is None:
            warmup_fraction = 0.0 if resume is not None else WARMUP_FRACTION
        warmup_steps = int(max(0.0, warmup_fraction) * max_epochs * steps_per_epoch)
        head_warmup_steps = int(round(float(head_warmup_epochs) * steps_per_epoch))
        refresh_steps = (nn_refresh_steps(nn_refresh_epochs, steps_per_epoch)
                         if refresh_neighbours is not None else 0)
        global_step = 0
        start_epoch = 0
        if resume is not None:
            # Continue the parent's trajectory under this call's configuration: a fresh optimizer
            # over this call's trainable set, with the moments of the parameters that stay trainable.
            kept, new = _load_moments(optimizer, param_names, resume["optimizer"])
            if resume["scaler"]:
                scaler.load_state_dict(resume["scaler"])
            global_step, start_epoch = int(resume["counters"]["step"]), int(resume["counters"]["epoch"])
            continued = resume["config"].get("seed") == (call_config or {}).get("seed")
            if continued:
                _restore_rng(resume["rng"], train_loader.generator, units)
            config["resume"] = {"epoch": start_epoch, "step": global_step, "moments_kept": kept,
                                "moments_new": new, "random_streams": "continued" if continued else "reseeded"}
        start_step = global_step
        config.update(warmup_steps=warmup_steps, head_warmup_steps=head_warmup_steps,
                      nn_refresh_steps=refresh_steps)
        if warmup_steps:
            logger.info("Linear LR warmup over %d step(s) to %.2e", warmup_steps, learning_rate)

        history = []
        plate_history = []
        val_history = []
        grad_norms = []
        best_loss = float("inf")
        best_model_path = os.path.join(output_dir, "best_model.pth")
        stop_reason = "max_epochs_reached"
        by_proxy = validation is not None and validation["measurable"]
        best = {"epoch": None, "proxy": -math.inf}
        proxy_stopper = ProxyStopper(early_stop, config["min_epochs"]) if early_stop and by_proxy else None

        def keep_best(epochs_done):
            best["epoch"] = epochs_done
            best["step"] = global_step
            save_retrying(model.state_dict(), best_model_path)
            if save_state:
                save_resume_state(os.path.join(output_dir, RESUME_BEST), model, optimizer, param_names,
                                  scaler, {"epoch": start_epoch + epochs_done, "step": global_step},
                                  call_config, train_loader.generator, units, intensity_reference)

        if training_logs is None:
            training_logs = []

        # While-loop so the budget can grow: reaching max_epochs while the loss is
        # still improving extends the ceiling toward hard_cap instead of stopping.
        epoch = 0
        budget_extensions = 0
        timed_out = False
        body_frozen = False
        try:
            while epoch < max_epochs:
                model.train()
                if plate_discriminator is not None:
                    plate_discriminator.train()
                total_loss = 0
                total_plate_mean = 0.0
                total_plate_adversarial = 0.0
                total_plate_accuracy = 0.0
                steps = 0

                # the epoch of the per-sample streams, set before this epoch's iterator (and its workers) exists
                train_loader.dataset.set_epoch(start_epoch + epoch)
                for v1, v2, names, _, identity in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
                    v1, v2 = v1.to(self.device), v2.to(self.device)
                    local_step = global_step - start_step
                    if (local_step < head_warmup_steps) != body_frozen:
                        body_frozen = not body_frozen      # heads-only warm-in: no grad for the body
                        for param in body:
                            param.requires_grad_(not body_frozen)
                    if local_step < head_warmup_steps + warmup_steps:
                        # each group warms up from its own base lr; the body's after the warm-in
                        for group in optimizer.param_groups:
                            offset = local_step - (0 if group["kind"] == "head" else head_warmup_steps)
                            if offset >= 0:
                                group["lr"] = warmup_lr(offset, warmup_steps, group["base_lr"])
                    global_step += 1
                    optimizer.zero_grad()

                    with autocast():
                        z1, z2, contrastive, views = self._batch_loss(
                            model, v1, v2, names, identity, temperature, field_aware, mask_units,
                            patches=gram_anchor is not None)
                        batch_plate_ids = list(identity["plate_id"])
                        plate_mean = plate_mean_alignment_loss(z1, z2, batch_plate_ids)
                        if plate_discriminator is None:
                            plate_adversarial = contrastive.new_zeros(())
                            plate_accuracy = contrastive.new_zeros(())
                        else:
                            plate_labels = torch.tensor(
                                [plate_codebook[value] for value in batch_plate_ids],
                                device=self.device,
                                dtype=torch.long,
                            )
                            plate_logits = plate_discriminator(0.5 * (z1 + z2))
                            plate_adversarial = F.cross_entropy(plate_logits, plate_labels)
                            plate_accuracy = plate_logits.argmax(dim=1).eq(
                                plate_labels).float().mean()
                        loss = (
                            contrastive
                            + plate_mean_weight * plate_mean
                            + plate_adversarial_weight * plate_adversarial
                        )
                        if gram_anchor is not None:
                            with torch.no_grad():
                                targets = [gram_anchor.forward_features(view["input"].detach())["x_norm_patchtokens"]
                                           for view in views]
                    with autocast(enabled=False):      # the regularisers in fp32
                        if gram_anchor is not None:
                            gram = 0.5 * sum(gram_loss(view["tokens"], target)
                                             for view, target in zip(views, targets))
                            loss = loss + gram_weight * gram
                        if koleo_weight > 0:
                            koleo = koleo_loss(views[0]["cls"]) + koleo_loss(views[1]["cls"])
                            loss = loss + koleo_weight * koleo

                    scaler.scale(loss).backward()
                    if clip_grad_norm > 0:
                        scaler.unscale_(optimizer)
                        grad_norms.append(float(torch.nn.utils.clip_grad_norm_(params, clip_grad_norm)))
                    scaler.step(optimizer)
                    scaler.update()

                    total_loss += loss.item()
                    total_plate_mean += float(plate_mean.detach())
                    total_plate_adversarial += float(plate_adversarial.detach())
                    total_plate_accuracy += float(plate_accuracy.detach())
                    steps += 1
                    if refresh_steps and (global_step - start_step) % refresh_steps == 0:
                        refresh_neighbours(model)      # nn_space=current: the table from this model
                        model.train()
                    if deadline is not None and time.time() >= deadline:
                        timed_out = True       # max_minutes: this epoch ends here, after >= 1 step
                        break

                avg_loss = total_loss / steps
                history.append(avg_loss)
                plate_history.append({
                    "plate_mean_penalty": total_plate_mean / steps,
                    "plate_adversarial_loss": total_plate_adversarial / steps,
                    "plate_adversarial_accuracy": total_plate_accuracy / steps,
                })
                logger.info(f"Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}")

                is_best, should_stop, reason = stopper.observe(avg_loss, epoch + 1)
                if proxy_stopper is not None and should_stop and reason != "non_finite_loss":
                    should_stop, reason = False, None        # the val proxy decides when the call stops
                epoch += 1
                if is_best:
                    best_loss = avg_loss
                    if not by_proxy:
                        keep_best(epoch)
                        logger.info(f"New best model saved (Loss = {best_loss:.4f})")
                if timed_out and not should_stop:
                    should_stop, reason = True, "max_minutes_reached"

                if not should_stop and epoch >= max_epochs and max_epochs < hard_cap:
                    new_max = next_epoch_budget(max_epochs, hard_cap)
                    if new_max > max_epochs:
                        budget_extensions += 1
                        logger.info("Loss still improving at epoch %d; extending budget %d -> %d",
                                    epoch, max_epochs, new_max)
                        max_epochs = new_max
                        stop_reason = "hard_cap_reached"

                if validation is not None and (should_stop or epoch >= max_epochs
                                               or epoch % validation["every"] == 0):
                    record = self._evaluate_val(model, validation, temperature, field_aware)
                    val_history.append({"epoch": epoch, **record})
                    if by_proxy and record["sibling_site_map"] is not None and record["sibling_site_map"] > best["proxy"]:
                        best["proxy"] = record["sibling_site_map"]
                        keep_best(epoch)
                        logger.info("New best model saved (val sibling_site_map = %.4f)", best["proxy"])
                    if proxy_stopper is not None and not should_stop:
                        proxy_stop, proxy_reason = proxy_stopper.observe(
                            epoch, record["sibling_site_map"], record.get("overfit_gap"))
                        if proxy_stop and epoch < max_epochs:
                            should_stop, reason = True, proxy_reason

                if should_stop:
                    stop_reason = reason or "converged"
                    logger.info("Early stopping after epoch %d: %s", epoch, stop_reason)
                    break
        finally:
            if body_frozen:                    # the call's trainable set, exactly as it was
                for param in body:
                    param.requires_grad_(True)

        if save_state:
            save_resume_state(os.path.join(output_dir, RESUME_END), model, optimizer, param_names, scaler,
                              {"epoch": start_epoch + epoch, "step": global_step}, call_config,
                              train_loader.generator, units, intensity_reference)
        config["stop_reason"] = stop_reason
        config["epochs_trained"] = len(history)
        config.update(start_step=start_step, end_step=global_step, steps_trained=global_step - start_step,
                      best_step=best.get("step"))
        config["budget_extensions"] = budget_extensions
        config["final_max_epochs"] = max_epochs
        config["plate_mean_weight"] = float(plate_mean_weight)
        config["plate_adversarial_weight"] = float(plate_adversarial_weight)
        config["plate_history"] = plate_history
        config["best_epoch"] = best["epoch"]
        config["best_by"] = "val_proxy" if by_proxy else "train_loss"
        config["early_stop"] = early_stop if proxy_stopper is not None else None
        config["epochs_saved"] = (max(0, max_epochs - len(history))
                                  if proxy_stopper is not None and stop_reason in ("val_proxy_plateau", "overfit_gap")
                                  else 0)
        config["val_history"] = val_history
        if clip_grad_norm > 0:
            finite = [norm for norm in grad_norms if math.isfinite(norm)]
            config["grad_norm_median"] = float(np.median(finite)) if finite else None
            config["clipped_fraction"] = (sum(norm > clip_grad_norm for norm in finite) / len(finite)
                                          if finite else None)
        return history, best_model_path, training_logs, config
    
    def _reuse_cached_embedding(self, cached_path, fingerprint, cluster_resolution, n_neighbors,
                                n_pcs=None, expected_identity_digest=None):
        """Re-cluster a cached embedding, or return None to fall through to a full run.

        Returns None (never raises) whenever reuse is not provably safe: unreadable file,
        missing or mismatched fingerprint. Falling through costs a retrain; reusing wrongly
        costs a wrong answer, so every ambiguous case falls through.

        `expected_identity_digest` (task 087 contract SS6): when the CURRENT call requests
        stable per-cell identity, this is the sha256 of its ordered cell_id sequence. The
        cache must carry the SAME digest (`adata.uns["identity_sequence_digest"]`) to be
        reused -- a legacy cache (no identity schema) or one built from a differently-
        ordered/different id sequence is rejected even on a fingerprint match, because the
        fingerprint sorts the crop set and is blind to per-cell identity/order.
        """
        try:
            adata = ad.read_h5ad(cached_path)
        except Exception as exc:                     # unreadable/corrupt cache -> full run
            logger.warning("Cached embedding unreadable (%s); retraining.", exc)
            return None
        cached_fp = adata.uns.get("embedding_fingerprint")
        if not cached_fp or cached_fp != fingerprint:
            logger.info("Cached embedding fingerprint differs; retraining.")
            return None
        if expected_identity_digest is not None:
            cached_digest = adata.uns.get("identity_sequence_digest")
            if not cached_digest or cached_digest != expected_identity_digest:
                logger.info(
                    "Stable identity requested but the cached embedding lacks the "
                    "matching identity schema/sequence; retraining."
                )
                return None

        logger.info("Reusing cached embedding (%d cells); re-clustering only.", adata.n_obs)
        groups = (adata.obs["group"].astype(str).tolist()
                  if "group" in adata.obs else None)
        cluster_key = self._compute_umap_and_clustering(
            adata, cluster_resolution, groups, n_neighbors, n_pcs=n_pcs)
        clustering_representation = adata.uns.get("clustering_representation")
        adata.write(cached_path)

        n_clusters = (int(adata.obs[cluster_key].nunique())
                      if cluster_key and cluster_key in adata.obs else 0)
        return {
            "summary": (f"Re-clustered {adata.n_obs} cells from the cached embedding at "
                        f"resolution {cluster_resolution} (n_neighbors={n_neighbors}) -> "
                        f"{n_clusters} clusters. No training: the embedding-affecting "
                        f"configuration is unchanged."),
            "cell_count": int(adata.n_obs),
            "mode": "reused-embedding",
            "epochs_trained": 0,
            "final_loss": None,
            "loss_history": [],
            "alignment": None,
            "training_config": {"stop_reason": "reused_cached_embedding", "epochs_trained": 0},
            "analyzer_kwargs": dict(adata.uns.get("analyzer_kwargs", {})),
            "source_reliability_report": dict(
                adata.uns.get("source_reliability_report", {}) or {}),
            "adata_path": cached_path,
            "deliverables": [cached_path],
            "cluster_key": cluster_key,
            "cluster_resolution": cluster_resolution,
            "n_neighbors": n_neighbors,
            "clustering_representation": clustering_representation,
            "embedding_fingerprint": fingerprint,
        }

    def recluster(self, adata_path, fingerprint, cluster_resolution, n_neighbors, n_pcs=None):
        """Re-cluster the embedding in `adata_path` (its X), fail-closed: the result of
        `_reuse_cached_embedding`, or None when the file is unreadable or its fingerprint is not
        `fingerprint`. Never trains."""
        return self._reuse_cached_embedding(adata_path, fingerprint, cluster_resolution, n_neighbors,
                                            n_pcs=n_pcs)

    def _resolve_clustering_representation(self, adata, n_pcs):
        """Task 075 M1: choose raw-768d vs PCA-reduced clustering representation, from
        the embedding's OWN data properties (eigenvalue spectrum / dimensionality /
        sample count) only -- NEVER from `scalar_score` or any verifier quality metric
        (069/072 established that score is degenerate).

        `n_pcs=None` (the default, i.e. this parameter not exercised): reproduces the
        historical raw-768d behaviour EXACTLY -- existing callers see unchanged output.
        `n_pcs="auto"`: derive the PCA dimension from the embedding's broken-stick
        spectrum (see `resolve_clustering_representation`). `n_pcs=<int>`: explicit
        override, honored exactly (clamped only to the data's own usable rank).
        """
        if n_pcs is None:
            return {
                "representation": "raw", "n_pcs": 0, "method": "default_unchanged",
                "n_obs": int(adata.n_obs), "n_features": int(adata.n_vars),
            }
        requested = None if n_pcs == "auto" else n_pcs
        return resolve_clustering_representation(adata.X, requested_n_pcs=requested)

    def _compute_umap_and_clustering(self, adata, resolution, groups=None, n_neighbors=20,
                                     n_pcs=None):
        """Compute UMAP coordinates and perform clustering (no visualization).

        `n_pcs`: see `_resolve_clustering_representation`. The resolved choice is
        recorded into `adata.uns["clustering_representation"]` unconditionally (069/072:
        an unrecorded representation makes every downstream number uninterpretable).
        """
        if not ANNDATA_AVAILABLE:
            logger.warning("anndata/scanpy not available. Cannot compute UMAP and clustering.")
            return None

        # Compute neighbors and UMAP
        try:
            n_neighbors_int = max(2, int(n_neighbors))
        except (TypeError, ValueError):
            n_neighbors_int = 20
        # cap neighbours below the number of cells
        n_neighbors_int = min(n_neighbors_int, max(2, adata.n_obs - 1))

        representation = self._resolve_clustering_representation(adata, n_pcs)
        representation["metric"] = "cosine"   # frozen, not this task's scope -- recorded for context
        adata.uns["clustering_representation"] = representation
        if representation["representation"] == "pca":
            sc.pp.pca(adata, n_comps=representation["n_pcs"])
            use_rep = "X_pca"
        else:
            use_rep = "X"
        logger.info("Clustering representation: %s", representation)

        sc.pp.neighbors(adata, use_rep=use_rep, n_neighbors=n_neighbors_int, metric="cosine")
        sc.tl.umap(adata, min_dist=0.05, spread=1, random_state=42)

        # Perform clustering
        cluster_key = f"leiden_{resolution}"
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)

        logger.info(f"Computed UMAP and Leiden clustering (resolution={resolution})")
        logger.info(f"Found {len(adata.obs[cluster_key].unique())} clusters")

        return cluster_key

    def _load_cell_data_from_metadata(self, query_cache_dir):
        """Load cell crops and metadata from metadata files."""
        tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
        metadata_files = glob.glob(os.path.join(tool_cache_dir, 'cell_crops_metadata_*.json'))
        
        if not metadata_files:
            raise ValueError(f"No metadata files found in {tool_cache_dir}")
        
        # Deduplicate by source_image_id
        metadata_by_image = {}
        metadata_path_by_image = {}   # task 038 C4: locate each image's kept-crops manifest
        for f in sorted(metadata_files, key=os.path.getmtime):
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                img_id = data.get('source_image_id', f"unknown_{len(metadata_by_image)}")
                existing = metadata_by_image.get(img_id)
                if existing is None or data.get("execution_status") == "success":
                    metadata_by_image[img_id] = data
                    metadata_path_by_image[img_id] = f
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")

        # Merge all crops
        # Filter to exclude overlay and visualization files (only include actual crop files)
        all_crops, all_metadata, skipped = [], [], []
        for img_id, data in metadata_by_image.items():
            if data.get('execution_status') != 'success':
                skipped.append({'image': img_id, 'reason': data.get('execution_status', 'invalid')})
                continue

            crops = data.get('cell_crops_paths', [])
            metadata = data.get('cell_metadata', [])
            group = data.get('group', 'default')
            parent_technical_metadata = data.get('technical_metadata', {})
            if not isinstance(parent_technical_metadata, dict):
                parent_technical_metadata = {}

            # task 038 C4: exclude a crop the crop-stage arbitration vision_reject'ed (or
            # that failed is_crop_valid) via the sidecar kept-crops manifest, instead of
            # ingesting every crop unconditionally. None (no manifest -> no vision engine
            # ran, or pre-038 metadata) -> no filtering, byte-identical to before this task.
            kept_paths = _load_kept_crops_manifest(metadata_path_by_image.get(img_id))

            # Filter crops to exclude overlay and visualization files
            # Crop files should contain 'crop' in filename and not contain 'overlay', 'viz', 'mask_viz'
            filtered_crops = []
            filtered_indices = []
            for idx, crop_path in enumerate(crops):
                if isinstance(crop_path, str):
                    crop_filename_lower = os.path.basename(crop_path).lower()
                    # Exclude overlay, visualization, and mask files
                    if ('overlay' in crop_filename_lower or
                        'viz' in crop_filename_lower or
                        'mask_viz' in crop_filename_lower or
                        'summary' in crop_filename_lower):
                        continue
                    # Include files that contain 'crop' in filename (actual crop files)
                    if 'crop' in crop_filename_lower and os.path.isfile(crop_path):
                        if kept_paths is not None and crop_path not in kept_paths:
                            continue   # task 038 C4: arbitration-rejected / gate-invalid crop
                        filtered_crops.append(crop_path)
                        filtered_indices.append(idx)
            
            if not filtered_crops:
                logger.warning(f"No valid crop files found for image {img_id} (all files appear to be overlays/visualizations). Skipping.")
                skipped.append({'image': img_id, 'reason': 'no_valid_crops'})
                continue
            
            all_crops.extend(filtered_crops)
            # Match metadata to filtered crops
            filtered_metadata = []
            for idx in filtered_indices:
                if idx < len(metadata) and isinstance(metadata[idx], dict):
                    m = metadata[idx].copy()
                    m.setdefault('group', group)
                    m.setdefault('image_name', img_id)
                    for key, value in parent_technical_metadata.items():
                        m.setdefault(key, value)
                    filtered_metadata.append(m)
                else:
                    filtered_metadata.append({
                        'group': group, 'image_name': img_id,
                        **parent_technical_metadata,
                    })
            
            all_metadata.extend(filtered_metadata)
        
        if not all_crops:
            raise ValueError("No valid crop files found in metadata; Cell_State_Analyzer_Tool requires actual crops from Single_Cell_Cropper_Tool.")
        
        return all_crops, all_metadata, skipped
    
    
    def execute(self, cell_crops=None, cell_metadata=None, max_epochs=None, early_stop_loss=None,
                batch_size=None, learning_rate=3e-5, weight_decay=1e-4, temperature=0.1,
                cluster_resolution=0.5, n_neighbors=15, n_pcs=None,
                input_strategy="full_crop", query_cache_dir=None, patience=None,
                zero_shot=False, augment_color=True, freeze_backbone=False,
                adaptation_depth=None,
                field_aware=False, compute_alignment=False, reuse_embedding=False, seed=0,
                plate_mean_weight=0.0, plate_adversarial_weight=0.0,
                view_pairing="same_crop", contrastive_mask="off", batch_sampling="random",
                trainable_pct=None, init_checkpoint=None, split_key=None, val_fraction=None,
                output_subdir=None, max_minutes=None, eval_every_epochs=1, embed_only=False,
                head_learning_rate=None, patch_embed_lr_mult=1.0, clip_grad_norm=0.0,
                warmup_fraction=None, head_warmup_epochs=0.0, gram_weight=0.0, koleo_weight=0.0,
                intensity_norm="off", augmentation="strong", crop_scale_min=0.3, drop_path=0.0,
                nn_k=5, nn_space="frozen", nn_mutual=False, nn_refresh_epochs=0.0, crop_array=None,
                early_stop=None,
                min_epochs=None):
        """Execute self-supervised learning training and analysis.

        Pixels: one file per crop (`cell_crops`, the default), or with `crop_array` the path of one
        packed .npy of shape (N, C, H, W), uint8 or uint16, C <= 16, H and W >= 8, with an integer
        `array_row` in every `cell_metadata` row and no `cell_crops`: each crop is `array[array_row]`,
        normalized like a file read, and its key "<array file name>#<array_row>" stands where a file
        crop's path stands (the image_name fallback); the embedding fingerprint hashes the rows it
        reads, never the path. The call reads its distinct rows into memory once, with plain file
        reads retried on a transient error (`ArrayCrops`); a read that still fails returns {"error"}.

        Embedding-variant flags (all default to the historical behavior):
          - zero_shot: skip contrastive training, extract the pretrained backbone CLS.
          - augment_color: keep ColorJitter/Grayscale augmentation (False for 1-channel).
          - freeze_backbone: train only the projector and read features from it.
          - field_aware: mask same-source-field crops out of contrastive negatives
            (label-free; uses the per-crop source id, never the cell-line label).
          - n_pcs: clustering representation (task 075 M1). Default None reproduces the
            historical raw-768d neighbour graph EXACTLY (unchanged). "auto" derives the
            PCA dimension from the embedding's own broken-stick spectrum; an explicit int
            is honored exactly (clamped to the data's usable rank). Never derived from
            scalar_score or any verifier quality metric.
          - view_pairing: the second view of a training crop is another augmentation of the
            crop itself (same_crop), or of another training crop of the same plate-qualified site
            / well (same_site / same_well; the crop itself when it is alone there), or
            (nn_other_image) of a random one of the crop's nn_k nearest training crops of other
            sites by the cosine of their CLS (eval transform) in the pretrained (nn_space frozen)
            or the loaded model (current), rebuilt from the model being trained every
            nn_refresh_epochs epochs under current (0 = never); nn_mutual keeps mutual neighbours
            only. The val eval loss pairs val crops the same way among the val crops.
          - contrastive_mask: drop the in-batch negatives whose second view comes from the
            anchor's own site / well (same_site / same_well); the positive pair stays.
          - batch_sampling: shuffled batches over all training crops (random), or every batch
            from one plate / experiment (per_plate / per_experiment).
          - trainable_pct: None keeps adaptation_depth / freeze_backbone; 0..100 = the share of
            the backbone's transformer blocks that trains, counted from the last block (resolved
            to a block count, logged); the adapter and the projector always train unless
            zero-shot, and the backbone CLS is read.
          - head_learning_rate (None = learning_rate) / patch_embed_lr_mult: the AdamW step of
            the projector (and plate discriminator) and of the patch embedding (`optimizer_groups`).
          - clip_grad_norm > 0: clip the global gradient norm (0 = no clip at all).
          - warmup_fraction: None = 5 % warmup on a fresh call, none on a resumed call; a number
            warms up over that fraction of this call's steps either way. head_warmup_epochs:
            heads-only epochs at the start of the call.
          - gram_weight > 0: Gram anchoring of the patch tokens to a frozen copy of the
            pretrained backbone; koleo_weight > 0: KoLeo on the backbone CLS of both views.
          - intensity_norm=per_plate: every crop's channels are mapped from its plate's statistics
            onto those of all training crops (`plate_reference`, stored with the resume state).
          - augmentation: strong (this tool's pipeline) or light (`SSL116View`; the old names shape /
            ssl116 are accepted, `AUGMENTATION_ALIASES`); crop_scale_min:
            the lower area bound of the training crop under either; drop_path: stochastic depth.
          A lever value that needs a metadata column (site, well, plate, experiment) the crops
          lack raises ValueError naming the column.

        The label-free train / val split: `split_key` names a metadata key holding "train" /
        "val" per crop; else `val_fraction` assigns val wells per plate
        (`quant.assign_val_wells`); with neither there is no split. Only the train crops train,
        every crop is embedded (obs `split`, `trained`), the val crops are evaluated every
        `eval_every_epochs` epochs and after the last one (`val_history`), and the best point is
        the evaluation with the highest val proxy (`quant.sibling_site_map`); without a split,
        the epoch with the lowest training loss. With a split the full training state is saved
        at the best point and at the end (`resume_best`, `resume_end`). A call that does not train
        evaluates the val crops' proxy once, as epoch 0 (no eval loss).
          - init_checkpoint: such a resume state; training continues from it under this call's
            configuration. With embed_only=True every crop is embedded with it (eval transform,
            the resume state's intensity_norm and stored reference; no split is read) and nothing
            trains.
          - output_subdir: the call's files go to <query_cache_dir>/cell_state_analysis/<output_subdir>/.
          - max_minutes: wall-clock cap of the training (at least one step trains).
          - min_epochs: the early stopper's floor, no stop on the loss plateau before it; None = the
            resolved rule (`resolve_training_config`), every caller's default. A harness setting, not a
            CARD lever (the benchmarks' plain fine-tune control passes 1: no floor).
        Every result carries `seconds` (the call's wall time); a failed call returns {"error": msg}.
        """
        started = time.time()
        logger.info("Starting cell state analysis...")

        # Load data
        if crop_array is not None:
            if cell_crops:
                raise ValueError("pass the crops as cell_crops (files) or as crop_array (one packed array), not both")
            if cell_metadata is None:
                raise ValueError("crop_array needs cell_metadata rows that carry array_row")
            try:
                cell_crops = array_crops(crop_array, cell_metadata)
            except OSError as error:
                return {"error": f"crop_array {crop_array!r} unreadable: {type(error).__name__}: {error}",
                        "seconds": round(time.time() - started, 1)}
        if cell_crops is None or cell_metadata is None:
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            cell_crops, cell_metadata, skipped = self._load_cell_data_from_metadata(query_cache_dir)

        if not cell_crops:
            return {"error": "No cell crops found"}
        if cell_metadata is None or len(cell_metadata) != len(cell_crops):
            raise ValueError(
                f"cell_metadata must align one-to-one with cell_crops "
                f"({0 if cell_metadata is None else len(cell_metadata)} != {len(cell_crops)})"
            )
        augmentation = AUGMENTATION_ALIASES.get(augmentation, augmentation)     # the names before 2026-10-11
        for name, value in (("view_pairing", view_pairing), ("contrastive_mask", contrastive_mask),
                            ("batch_sampling", batch_sampling), ("intensity_norm", intensity_norm),
                            ("augmentation", augmentation), ("nn_space", nn_space)):
            if value not in _LEVER_CHOICES[name]:
                raise ValueError(f"{name} must be one of {list(_LEVER_CHOICES[name])}, got {value!r}")
        if not CROP_SCALE_RANGE[0] <= crop_scale_min <= CROP_SCALE_RANGE[1]:
            raise ValueError(f"crop_scale_min must be in {list(CROP_SCALE_RANGE)}, got {crop_scale_min!r}")
        if trainable_pct is not None and not 0 <= trainable_pct <= 100:
            raise ValueError(f"trainable_pct must be in [0, 100], got {trainable_pct!r}")
        if embed_only and not init_checkpoint:
            raise ValueError("embed_only needs init_checkpoint (the training state to embed with)")
        if int(eval_every_epochs) < 1:
            raise ValueError(f"eval_every_epochs must be >= 1, got {eval_every_epochs!r}")
        early_stop = resolve_early_stop(early_stop)
        resume = load_resume_state(init_checkpoint) if init_checkpoint else None
        # embed_only embeds under the resume state's intensity_norm (the kwarg cannot be told from
        # an omitted one) and reads no split value.
        intensity_used = resume["config"].get("intensity_norm", "off") if embed_only else intensity_norm
        if intensity_used == "per_plate" and embed_only and not resume.get("intensity_reference"):
            raise ValueError(f"init_checkpoint {init_checkpoint!r}: intensity_norm per_plate without a stored "
                             "intensity_reference")
        unit_ids = quant.identities(cell_metadata)     # plate-qualified site / well, plate, experiment
        needs = (intensity_used,) if embed_only else (view_pairing, contrastive_mask, batch_sampling, intensity_used)
        for value in needs:
            column = _LEVER_COLUMNS.get(value)
            if column is not None and any(row[column] is None for row in unit_ids):
                raise ValueError(f"{value} needs the metadata column {column!r} on every crop")
        split = None if embed_only else resolve_split(cell_metadata, split_key, val_fraction)
        train_rows = [i for i in range(len(cell_crops)) if split is None or split[i] == "train"]
        val_rows = [] if split is None else [i for i, value in enumerate(split) if value == "val"]

        # One recorded seed controls model initialization, augmentation and DataLoader order.
        seed = int(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
        # Seeded runs should be replayable. ``warn_only`` avoids converting an otherwise
        # valid workflow into a hard failure when a backend lacks a deterministic kernel.
        torch.use_deterministic_algorithms(True, warn_only=True)
        if hasattr(torch.backends, "cudnn"):
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False

        input_channels, channel_layouts = infer_channel_spec(cell_crops)
        logger.info(
            "Detected %d input channel(s); source layout(s)=%s",
            input_channels, channel_layouts)
        plate_route = resolve_plate_nuisance(
            cell_metadata, plate_mean_weight, plate_adversarial_weight)
        effective_plate_mean_weight = plate_route["plate_mean_weight"]
        effective_plate_adversarial_weight = plate_route["plate_adversarial_weight"]
        if (plate_mean_weight or plate_adversarial_weight) and not plate_route["eligible"]:
            logger.warning(
                "Plate losses requested but disabled by eligibility rule: %s",
                plate_route["rule"])

        # Stable per-cell identity (task 087): fail-closed BEFORE any GPU/dataset work.
        # `requested_cell_ids` is None (legacy: synthetic cell_i row names) unless every
        # row carries both cell_id and crop_id; partial/duplicate identity raises here.
        requested_cell_ids = _validate_identity_metadata(cell_metadata)

        num_crops = len(cell_crops)
        logger.info(f"Processing {num_crops} cell crops")

        # A contrastive batch needs at least two distinct samples. This is a
        # mathematical constraint, not a dataset-size quality heuristic. An explicit
        # zero_shot request also forces the no-training path (pretrained baseline).
        # A supplied adaptation_depth is authoritative over the two legacy booleans; omitting it
        # leaves their behaviour byte-identical to before this parameter existed.
        if adaptation_depth is not None:
            zero_shot, freeze_backbone = resolve_adaptation_depth(
                adaptation_depth, input_channels)
        if trainable_pct is not None:
            freeze_backbone = False     # the share of trained blocks replaces the frozen-backbone choice
        # embed_only embeds with a saved training state and never trains.
        n_train = len(train_rows)
        use_zero_shot = zero_shot or embed_only or n_train < 2
        output_dir = analysis_dir(query_cache_dir, output_subdir)
        os.makedirs(output_dir, exist_ok=True)

        # --- cached-embedding reuse -------------------------------------------------------
        # Clustering knobs are applied AFTER the embedding exists, so a run differing ONLY in
        # cluster_resolution/n_neighbors can re-cluster the cached feature space instead of
        # paying another DINOv3 fine-tune. That is what makes a clustering sweep affordable:
        # seconds instead of a GPU-hour per candidate.
        #
        # Guarded by the fingerprint, which covers every embedding-affecting parameter AND the
        # crop set. Reuse happens only on an exact match, because re-clustering a stale
        # embedding is a silently wrong answer -- far worse than the retrain it would save.
        #
        # OPT-IN on purpose: reuse is a caller decision even though seeded reruns are reproducible.
        embedding_params = {
            "adaptation_depth": adaptation_depth, "zero_shot": zero_shot,
            "freeze_backbone": freeze_backbone, "learning_rate": learning_rate,
            "max_epochs": max_epochs, "early_stop_loss": early_stop_loss,
            "patience": patience, "temperature": temperature,
            "augment_color": augment_color, "field_aware": field_aware,
            "seed": seed, "input_channels": input_channels,
            "channel_layout": channel_layouts,
            "channel_adapter_init": (
                "replicate_1_to_rgb" if input_channels == 1 else
                "identity_rgb" if input_channels == 3 else
                "v63_biological_6_to_rgb" if input_channels == 6 else
                "deterministic_balanced_to_rgb"),
            "symmetric_info_nce": True,
            "plate_mean_weight": effective_plate_mean_weight,
            "plate_adversarial_weight": effective_plate_adversarial_weight,
            "view_pairing": view_pairing, "contrastive_mask": contrastive_mask,
            "batch_sampling": batch_sampling, "trainable_pct": trainable_pct,
            "init_checkpoint": init_checkpoint, "split_key": split_key,
            "val_fraction": val_fraction, "output_subdir": output_subdir,
            "max_minutes": max_minutes, "eval_every_epochs": eval_every_epochs,
            "embed_only": embed_only, "weight_decay": weight_decay,
            "head_learning_rate": head_learning_rate, "patch_embed_lr_mult": patch_embed_lr_mult,
            "clip_grad_norm": clip_grad_norm, "warmup_fraction": warmup_fraction,
            "head_warmup_epochs": head_warmup_epochs, "gram_weight": gram_weight,
            "koleo_weight": koleo_weight, "intensity_norm": intensity_used,
            "augmentation": augmentation, "crop_scale_min": crop_scale_min, "drop_path": drop_path,
            "nn_k": nn_k, "nn_space": nn_space, "nn_mutual": nn_mutual,
            "nn_refresh_epochs": nn_refresh_epochs, "min_epochs": min_epochs, "early_stop": early_stop}
        cached_fingerprint = embedding_fingerprint(embedding_params, cell_crops or [])
        cached_path = os.path.join(output_dir, "cell_state_analyzed.h5ad")
        expected_identity_digest = (
            _identity_sequence_digest(requested_cell_ids)
            if requested_cell_ids is not None else None)
        if reuse_embedding and ANNDATA_AVAILABLE and os.path.exists(cached_path):
            reused = self._reuse_cached_embedding(
                cached_path, cached_fingerprint, cluster_resolution, n_neighbors,
                n_pcs=n_pcs, expected_identity_digest=expected_identity_digest)
            if reused is not None:
                reused["seconds"] = round(time.time() - started, 1)
                return reused

        # Prepare groups
        groups = [m.get('group', 'default') if isinstance(m, dict) else 'default' for m in cell_metadata]
        image_names = [
            m.get('image_name', _crop_name(cell_crops, i))
            if isinstance(m, dict)
            else _crop_name(cell_crops, i)
            for i, m in enumerate(cell_metadata)
        ]
        technical_groups, technical_batches, source_reliability_report = (
            resolve_technical_metadata(cell_metadata, image_names)
        )
        # Optional identity companions (task 087): only meaningful when identity was
        # requested; None otherwise, so the dataset falls back to its "" sentinel.
        mask_ids = ([m.get('mask_id') for m in cell_metadata]
                    if requested_cell_ids is not None else None)
        rel_paths = ([m.get('rel_path') for m in cell_metadata]
                     if requested_cell_ids is not None else None)
        # intensity_norm=per_plate: the reference of the training crops (embed_only: the stored one),
        # applied to every crop, a plate absent from it with its own crops' statistics.
        plates = [row["plate"] for row in unit_ids]
        intensity_reference = None
        if intensity_used == "per_plate":
            intensity_reference = (resume["intensity_reference"] if embed_only else plate_reference(
                _crop_subset(cell_crops, train_rows), [plates[i] for i in train_rows]))
        affine = None if intensity_reference is None else norm_affine(cell_crops, plates, intensity_reference)

        def rows_of(values, rows):
            return None if values is None else _crop_subset(values, rows)

        def affine_of(rows):
            return None if affine is None else (affine[0][rows], affine[1][rows])

        # Create datasets
        eval_transform = self._get_eval_transform()
        eval_dataset = CellCropDataset(
            cell_crops, groups, image_names=image_names, transform=eval_transform,
            cell_ids=requested_cell_ids, mask_ids=mask_ids, rel_paths=rel_paths, affine=affine,
        )
        # Initialize model - uses dinov3_vitb16 (768 dim) to match weights
        model = DinoV3Projector(
            backbone_name="dinov3_vitb16", proj_dim=256,
            freeze_backbone=freeze_backbone,
            input_channels=input_channels).to(self.device)
        set_drop_path(model.backbone, drop_path)
        if trainable_pct is not None:
            backbone_names = [name for name, _ in model.backbone.named_parameters()]
            trained_names = set(trainable_backbone_names(
                backbone_names, resolve_trainable_blocks(trainable_pct, backbone_depth(backbone_names))))
            for name, param in model.backbone.named_parameters():
                param.requires_grad_(name in trained_names)
        sample_tensor = eval_dataset[0][0]
        # gram_weight: the anchor is the PRETRAINED backbone, copied before any resume state is
        # loaded, kept outside the model (no optimizer, state_dict or resume state holds it).
        gram_anchor = None
        if gram_weight > 0 and not use_zero_shot:
            gram_anchor = copy.deepcopy(model.backbone).eval().requires_grad_(False)
            model.eval()
            with torch.no_grad():
                inputs = model.backbone_input(sample_tensor[None].to(self.device))
                if not torch.allclose(model.backbone.forward_features(inputs)["x_norm_clstoken"],
                                      model.backbone(inputs), rtol=1e-5, atol=1e-6):
                    raise ValueError("gram_weight: the backbone's forward_features CLS differs from its forward output")
        model_parameter_bytes = sum(
            parameter.numel() * parameter.element_size() for parameter in model.parameters()
        )
        anchor_bytes = 0 if gram_anchor is None else sum(
            parameter.numel() * parameter.element_size() for parameter in gram_anchor.parameters())
        if self.device.type == "cuda":
            # mem_get_info counts the caching allocator's pool as used: release what earlier calls of this
            # process left cached, so every call sizes its batch against the same free memory.
            import gc
            gc.collect()
            torch.cuda.empty_cache()
        batch_config = resolve_batch_size(
            num_samples=num_crops if use_zero_shot else n_train,
            sample_bytes=sample_tensor.numel() * sample_tensor.element_size(),
            # the anchor needs no optimizer state; the estimator reserves 3x the bytes it gets
            model_parameter_bytes=model_parameter_bytes + anchor_bytes // 3,
            device=self.device,
            requested_batch_size=batch_size,
        )
        resolved_batch_size = batch_config["batch_size"]
        if split is not None and not use_zero_shot and resolved_batch_size < 2:
            # a training call with a split must train and leave its resume state, never become zero-shot
            return {"error": f"insufficient memory for a contrastive batch (batch {resolved_batch_size}, "
                             f"{batch_config['available_memory_bytes']} bytes free)"}
        use_zero_shot = use_zero_shot or resolved_batch_size < 2
        logger.info("Adaptive batch size: %d (%s)", resolved_batch_size, batch_config)
        # Data loading workers from the hardware (`resolve_num_workers`); the results do not depend on them.
        workers = resolve_num_workers()
        num_workers = workers["num_workers"]
        pin_memory = self.device.type == "cuda"
        logger.info("Adaptive DataLoader workers: %d (%s)", num_workers, workers)
        if embed_only:
            use_projector = resume["config"].get("feature_source") == "projector"
        else:
            use_projector = freeze_backbone and not use_zero_shot
        call_config = {**embedding_params, "feature_source": "projector" if use_projector else "backbone_cls"}
        trainable_blocks = 0 if use_zero_shot else len(
            {name.split(".")[1] for name, param in model.backbone.named_parameters()
             if param.requires_grad and name.startswith("blocks.")})
        val_history, resume_best, resume_end = [], None, None
        train_seconds = 0.0
        # Train or zero-shot
        if use_zero_shot:
            logger.info("Using zero-shot inference (no training)")
            history, loss_path = [], None
            if embed_only:
                model.load_state_dict(resume["model"])
            training_config = {
                "stop_reason": (
                    "embed_only"
                    if embed_only
                    else "zero_shot_requested"
                    if zero_shot
                    else "insufficient_contrastive_samples"
                    if n_train < 2
                    else "insufficient_memory_for_contrastive_batch"
                ),
                "epochs_trained": 0,
            }
        else:
            logger.info("Starting training...")
            train_transform = self._get_augmentation_transform(
                augment_color=augment_color, augmentation=augmentation, crop_scale_min=crop_scale_min)
            train_paths = rows_of(cell_crops, train_rows)
            units = CropUnits(rows_of(unit_ids, train_rows), seed)
            mask_column = {"same_site": "site", "same_well": "well"}.get(contrastive_mask)
            train_dataset = CellCropDataset(
                train_paths, rows_of(groups, train_rows), image_names=rows_of(technical_groups, train_rows),
                transform=train_transform, plate_ids=rows_of(plate_route["plate_ids"], train_rows),
                units=units, view_pairing=view_pairing, affine=affine_of(train_rows), seed=seed,
            )
            validation = None
            if val_rows:
                # The val crops in input order (the order the verifier reads them in); the eval loss
                # on the first EVAL_CELLS of them, paired among themselves.
                loss_rows = val_rows[:EVAL_CELLS]
                loss_units = CropUnits(rows_of(unit_ids, loss_rows), EVAL_SEED)
                val_meta = rows_of(cell_metadata, val_rows)
                val_dataset = CellCropDataset(
                    rows_of(cell_crops, val_rows), rows_of(groups, val_rows),
                    image_names=rows_of(image_names, val_rows), transform=eval_transform,
                    affine=affine_of(val_rows))
                loss_dataset = CellCropDataset(
                    rows_of(cell_crops, loss_rows), rows_of(groups, loss_rows),
                    image_names=rows_of(technical_groups, loss_rows), transform=train_transform,
                    units=loss_units, view_pairing=view_pairing, affine=affine_of(loss_rows), seed=EVAL_SEED)
                validation = {
                    "meta": val_meta, "use_projector": use_projector, "every": int(eval_every_epochs),
                    "loss_loader": DataLoader(loss_dataset, batch_size=EVAL_BATCH, shuffle=False,
                                              **loader_kwargs(num_workers, pin_memory)),
                    "loss_units": loss_units,
                    "mask_units": loss_units.codes(mask_column) if mask_column else None,
                    "measurable": quant.sibling_site_map(
                        np.ones((len(val_rows), 1)), val_meta, draws=1)["n_queries"] > 0,
                }
                if early_stop and early_stop["overfit_gap"]:
                    # the overfitting monitor: the proxy on a fixed seeded sample of the training wells
                    probe_rows = _train_probe_rows(rows_of(cell_metadata, train_rows), train_rows,
                                                   early_stop["train_wells"], EVAL_SEED)
                    if probe_rows:
                        validation["train_meta"] = rows_of(cell_metadata, probe_rows)
                        validation["train_loader"] = DataLoader(
                            CellCropDataset(rows_of(cell_crops, probe_rows), rows_of(groups, probe_rows),
                                            image_names=rows_of(image_names, probe_rows),
                                            transform=eval_transform, affine=affine_of(probe_rows)),
                            batch_size=EVAL_BATCH, shuffle=False, **loader_kwargs(num_workers, pin_memory))

            def embed_crops(rows, network):
                """The CLS of the crops `rows` (eval transform) under `network`: the neighbour space."""
                dataset = CellCropDataset(rows_of(cell_crops, rows), transform=eval_transform,
                                          affine=affine_of(rows))
                loader = DataLoader(dataset, batch_size=resolved_batch_size, shuffle=False,
                                    **loader_kwargs(num_workers, pin_memory))
                return self._extract_features(network, loader, use_projector=False)[0]

            def train_neighbours(network):
                units.nn = neighbour_table(embed_crops(train_rows, network), units.columns["site"], nn_k,
                                           mutual=nn_mutual, device=self.device)

            def neighbours(network):
                train_neighbours(network)
                if validation is not None:
                    validation["loss_units"].nn = neighbour_table(
                        embed_crops(loss_rows, network), validation["loss_units"].columns["site"], nn_k,
                        mutual=nn_mutual, device=self.device)

            nn_pairing = view_pairing == "nn_other_image"
            if nn_pairing and nn_space == "frozen":
                neighbours(model)                       # the pretrained state, before any resume load
            if resume is not None:
                # The resumed weights come BEFORE initial_model_state: an OOM retry restarts from them.
                model.load_state_dict(resume["model"])
            if nn_pairing and nn_space == "current":
                neighbours(model)
            initial_table = units.nn
            initial_model_state = {
                key: value.detach().cpu().clone()
                for key, value in model.state_dict().items()
            }
            deadline = None if max_minutes is None else started + 60.0 * float(max_minutes)
            batch_attempts = []
            train_started = time.time()
            while True:
                # Recreate every stochastic component from the recorded seed. This matters
                # after CUDA OOM: the failed attempt has already advanced augmentation and
                # discriminator RNG state before the smaller-batch retry starts.
                np.random.seed(seed)
                torch.manual_seed(seed)
                if torch.cuda.is_available():
                    torch.cuda.manual_seed_all(seed)
                units.rng = np.random.default_rng(seed)
                units.nn = initial_table

                def make_train_loader(workers):
                    if batch_sampling == "random":
                        return DataLoader(
                            train_dataset,
                            batch_size=resolved_batch_size,
                            shuffle=True,
                            drop_last=n_train % resolved_batch_size == 1,
                            generator=torch.Generator().manual_seed(seed),
                            **loader_kwargs(workers, pin_memory),
                        )
                    return DataLoader(
                        train_dataset,
                        batch_sampler=UnitBatchSampler(units, resolved_batch_size, batch_sampling),
                        generator=torch.Generator().manual_seed(seed),
                        **loader_kwargs(workers, pin_memory),
                    )

                train_workers = num_workers
                train_loader = make_train_loader(train_workers)
                refresh_every = (nn_refresh_steps(nn_refresh_epochs, max(1, len(train_loader)))
                                 if nn_pairing and nn_space == "current" else 0)
                if train_workers and refresh_every % max(1, len(train_loader)):
                    # The table is rebuilt mid-epoch: workers forked at the epoch's start (and the batches
                    # they prefetch) would pair with the old one, so this loader makes its samples in the
                    # main process, where each sample reads the table at the point it is made.
                    logger.info("nn_other_image table rebuilt every %d step(s), %d per epoch: the training "
                                "DataLoader runs with 0 workers", refresh_every, len(train_loader))
                    train_workers = 0
                    train_loader = make_train_loader(train_workers)
                if validation is not None:
                    validation["loader"] = DataLoader(
                        val_dataset, batch_size=resolved_batch_size, shuffle=False,
                        **loader_kwargs(num_workers, pin_memory))
                try:
                    history, best_path, _, training_config = self._train_model(
                        model, train_loader, max_epochs, early_stop_loss,
                        learning_rate, output_dir, [], patience=patience, min_epochs=min_epochs,
                        weight_decay=weight_decay, temperature=temperature,
                        field_aware=field_aware,
                        plate_ids=rows_of(plate_route["plate_ids"], train_rows),
                        plate_mean_weight=effective_plate_mean_weight,
                        plate_adversarial_weight=effective_plate_adversarial_weight,
                        warmup_fraction=warmup_fraction, batch_size=resolved_batch_size, units=units,
                        mask_units=units.codes(mask_column) if mask_column else None,
                        validation=validation, save_state=split is not None, deadline=deadline,
                        resume=resume, call_config=call_config, head_learning_rate=head_learning_rate,
                        patch_embed_lr_mult=patch_embed_lr_mult, clip_grad_norm=clip_grad_norm,
                        head_warmup_epochs=head_warmup_epochs, gram_weight=gram_weight,
                        gram_anchor=gram_anchor, koleo_weight=koleo_weight,
                        refresh_neighbours=train_neighbours if nn_pairing and nn_space == "current" else None,
                        nn_refresh_epochs=nn_refresh_epochs, intensity_reference=intensity_reference,
                        early_stop=early_stop,
                    )
                    batch_attempts.append({
                        "batch_size": resolved_batch_size,
                        "status": "success",
                    })
                    break
                except RuntimeError as exc:
                    if (
                        self.device.type != "cuda"
                        or not is_cuda_out_of_memory(exc)
                        or resolved_batch_size <= 2
                    ):
                        raise
                    batch_attempts.append({
                        "batch_size": resolved_batch_size,
                        "status": "cuda_oom",
                    })
                    resolved_batch_size = max(2, resolved_batch_size // 2)
                    logger.warning(
                        "CUDA OOM; retrying SSL with batch_size=%d",
                        resolved_batch_size,
                    )
                    del train_loader
                    import gc
                    model.zero_grad(set_to_none=True)
                    gc.collect()
                    torch.cuda.empty_cache()
                    model.load_state_dict(initial_model_state)
            train_seconds = time.time() - train_started
            model.load_state_dict(
                torch.load(best_path, map_location=self.device, weights_only=True)
            )
            batch_config["attempts"] = batch_attempts
            training_config["train_num_workers"] = train_workers
            val_history = training_config["val_history"]
            if split is not None:
                resume_best, resume_end = (path if os.path.exists(path) else None for path in (
                    os.path.join(output_dir, RESUME_BEST), os.path.join(output_dir, RESUME_END)))

            # Save loss curve (matplotlib is not thread-safe: a call in a worker thread draws it with PIL)
            if history:
                loss_path = os.path.join(output_dir, "loss_curve.png")
                if threading.current_thread() is not threading.main_thread():
                    _save_loss_curve_png(history, loss_path)
                else:
                    fig, ax = plt.subplots(figsize=(10, 6))
                    ax.plot(range(1, len(history) + 1), history)
                    ax.set_xlabel("Epoch")
                    ax.set_ylabel("Loss")
                    ax.set_title("Training Loss")
                    plt.savefig(loss_path)
                    plt.close()
        batch_config["batch_size"] = resolved_batch_size
        training_config["batch_size"] = resolved_batch_size
        training_config.update({
            "seed": seed,
            "input_channels": input_channels,
            "channel_layouts": channel_layouts,
            "channel_adapter_init": model.channel_adapter.init_name,
            "symmetric_info_nce": True,
            "field_aware": bool(field_aware),
            "augment_color": bool(augment_color),
            "trainable_blocks": trainable_blocks,
            "num_workers": num_workers,
            "plate_nuisance_route": {
                key: value for key, value in plate_route.items()
                if key != "plate_ids"
            },
        })
        training = {
            "epochs_trained": len(history),
            **{k: training_config.get(k) for k in ("start_step", "end_step", "steps_trained", "best_step")},
            "best_epoch": training_config.get("best_epoch"),
            "best_by": training_config.get("best_by"),
            "batch_size": resolved_batch_size,
            "trainable_blocks": trainable_blocks,
            "train_seconds": round(train_seconds, 1),
        }
        if training_config.get("early_stop"):           # only with early_stop: why the call stopped, what it saved
            gaps = [r.get("overfit_gap") for r in training_config.get("val_history") or []]
            best = next((r for r in training_config.get("val_history") or []
                         if r.get("epoch") == training_config.get("best_epoch")), {})
            training.update(stop_reason=training_config.get("stop_reason"),
                            epochs_saved=training_config.get("epochs_saved"),
                            overfit_gap_best=best.get("overfit_gap"),
                            overfit_gap_last=next((g for g in reversed(gaps) if g is not None), None))
        if clip_grad_norm > 0 and not use_zero_shot:    # logged only
            training.update(grad_norm_median=training_config.get("grad_norm_median"),
                            clipped_fraction=training_config.get("clipped_fraction"))

        eval_loader = DataLoader(
            eval_dataset,
            batch_size=resolved_batch_size,
            shuffle=False,
            **loader_kwargs(num_workers, pin_memory),
        )

        # Extract features. Frozen backbone => read the TRAINED projector (the only part
        # that learned); fine-tuned/zero-shot => the backbone CLS token.
        feats, names, groups_extracted, identities = self._extract_features(
            model, eval_loader, use_projector=use_projector)
        if use_zero_shot and val_rows:
            # The untrained call's val evaluation (epoch 0): the proxy of the val rows of this
            # embedding, as a training call evaluates val; no eval loss (it needs the untrained head).
            val_history = [{"epoch": 0, "sibling_site_map": self._val_proxy(
                feats[val_rows], rows_of(cell_metadata, val_rows))}]

        # Optional label-free alignment (augmentation consistency) for the lr/epoch
        # calibration's training-health signal. Off by default (extra forward passes).
        alignment = None
        if compute_alignment:
            aug_dataset = CellCropDataset(
                cell_crops, groups, image_names=image_names,
                transform=self._get_augmentation_transform(
                    augment_color=augment_color, augmentation=augmentation, crop_scale_min=crop_scale_min),
                affine=affine)
            alignment = self._alignment_consistency(
                model, aug_dataset, sample=min(64, num_crops))

        # Create AnnData
        if not ANNDATA_AVAILABLE:
            return {"error": "anndata/scanpy not available"}
        
        adata = ad.AnnData(X=feats)
        adata.obs["group"] = groups_extracted if len(groups_extracted) == adata.n_obs else groups[:adata.n_obs]
        adata.obs["image_name"] = names if len(names) == adata.n_obs else ['unknown'] * adata.n_obs
        adata.obs["technical_group"] = technical_groups[:adata.n_obs]

        # Stable per-cell identity (task 087 contract SS4-5): `identities` traveled WITH
        # each sample through the eval DataLoader above, in the exact order features were
        # produced -- never reattached by assuming row order. Legacy callers (no identity
        # requested) keep the historical synthetic `cell_i` row names, byte-identical to
        # before this contract existed.
        if requested_cell_ids is not None:
            if len(identities) != adata.n_obs:
                raise RuntimeError(
                    f"stable cell identity was requested but the DataLoader produced "
                    f"{len(identities)} identities for {adata.n_obs} cells; refusing to "
                    "write (stable identity contract SS4)"
                )
            cell_ids_out = [d["cell_id"] for d in identities]
            if any(not c for c in cell_ids_out) or len(set(cell_ids_out)) != len(cell_ids_out):
                raise RuntimeError(
                    "stable cell identity was requested but the extracted cell_id "
                    "sequence is incomplete or non-unique; refusing to write (stable "
                    "identity contract SS2-3)"
                )
            adata.obs_names = cell_ids_out
            adata.obs["cell_id"] = cell_ids_out
            adata.obs["crop_id"] = cell_ids_out
            mask_ids_out = [d["mask_id"] for d in identities]
            if any(mask_ids_out):
                adata.obs["mask_id"] = mask_ids_out
            rel_paths_out = [d["rel_path"] for d in identities]
            if any(rel_paths_out):
                adata.obs["crop_rel_path"] = rel_paths_out
        else:
            adata.obs.index = [f"cell_{i}" for i in range(adata.n_obs)]

        # VLRR: per-cell area (from cropper metadata) feeds the representation
        # verifier's PC1_area_correlation; technical batch defaults to source image.
        # Carry EVERY per-cell morphological scalar the cropper recorded, not just `area`.
        # These are measured from the segmentation mask, INDEPENDENTLY of the embedding, which
        # is what makes them usable as an external anchor when judging whether a clustering is
        # reasonable: an objective computed only from the partition's own geometry can be gamed
        # by making the partition trivial, one anchored to independently-measured morphology
        # cannot. (They are evidence, not ground truth -- masks err too.)
        adata.obs["area"] = _per_cell_scalars(cell_metadata, "area", adata.n_obs)
        for _key in _MORPHOLOGY_KEYS:
            _vals = _per_cell_scalars(cell_metadata, _key, adata.n_obs)
            if np.isfinite(_vals).any():          # only materialise what this cropper emitted
                adata.obs[_key] = _vals
        _cent = [m.get("centroid") if isinstance(m, dict) else None for m in cell_metadata]
        if len(_cent) == adata.n_obs and any(isinstance(c, (list, tuple)) and len(c) >= 2
                                             for c in _cent):
            adata.obs["centroid_row"] = np.array(
                [c[0] if isinstance(c, (list, tuple)) and len(c) >= 2 else np.nan for c in _cent],
                dtype=float)
            adata.obs["centroid_col"] = np.array(
                [c[1] if isinstance(c, (list, tuple)) and len(c) >= 2 else np.nan for c in _cent],
                dtype=float)
        adata.obs["batch"] = technical_batches[:adata.n_obs]
        if split is not None:
            adata.obs["split"] = list(split)
            adata.obs["trained"] = [value == "train" and not use_zero_shot for value in split]
        adata.uns["ssl_input_contract"] = {
            "input_channels": input_channels,
            "source_layouts": channel_layouts,
            "internal_layout": "CHW",
            "channel_adapter_init": model.channel_adapter.init_name,
            "seed": seed,
            "symmetric_info_nce": True,
        }
        recorded_analyzer_kwargs = {
            key: value for key, value in {
                "learning_rate": learning_rate,
                "max_epochs": max_epochs,
                "patience": patience,
                "min_epochs": min_epochs,
                "early_stop": json.dumps(early_stop, sort_keys=True) if early_stop else None,
                "augment_color": augment_color,
                "field_aware": field_aware,
                "adaptation_depth": adaptation_depth,
                "seed": seed,
                "plate_mean_weight": plate_mean_weight,
                "plate_adversarial_weight": plate_adversarial_weight,
            }.items()
            if value is not None
        }
        adata.uns["analyzer_kwargs"] = recorded_analyzer_kwargs
        # Safety condition for reusing this embedding instead of re-training: a later run may
        # skip the fine-tune and re-cluster from this file ONLY on an exact fingerprint match.
        adata.uns["embedding_fingerprint"] = cached_fingerprint
        adata.uns["source_reliability_report"] = source_reliability_report
        # Stable-identity cache gate (task 087 contract SS6): recorded ONLY when identity was
        # requested, so a legacy cache (this key absent) is never mistaken for an identity-
        # carrying one on a later reuse attempt.
        if expected_identity_digest is not None:
            adata.uns["identity_sequence_digest"] = expected_identity_digest

        # Compute UMAP and clustering
        cluster_key = self._compute_umap_and_clustering(
            adata, cluster_resolution,
            groups if len(groups) == adata.n_obs else groups_extracted,
            n_neighbors, n_pcs=n_pcs)
        clustering_representation = adata.uns.get("clustering_representation")

        # Save
        adata_path = os.path.join(output_dir, "cell_state_analyzed.h5ad")
        adata.write(adata_path)
        
        # Check if results are sufficient for group comparison queries
        unique_groups = set(groups_extracted if len(groups_extracted) == adata.n_obs else groups[:adata.n_obs])
        is_single_group = len(unique_groups) == 1
        insufficient_for_comparison = False
        comparison_limitations = []
        termination_recommended = False
        termination_reason = None
        
        if is_single_group:
            insufficient_for_comparison = True
            comparison_limitations.append("All cells are labeled as 'default' - no group labels available for comparison")
            # If query requires group comparison, recommend termination
            termination_recommended = True
            termination_reason = (
                "**Cannot perform group comparison analysis:** All cells are in a single group ('default'). "
                "Group comparison queries cannot be answered with the current data.\n\n"
                "**Recommendation:** Terminate analysis and inform user that group labels are required for comparison queries.\n\n"
                "**Solution:** Provide treatment/group labels when uploading images to enable group comparison analysis."
            )
        
        if num_crops < 10:
            insufficient_for_comparison = True
            comparison_limitations.append(f"Very small sample size ({num_crops} cells) - may be insufficient for robust statistical comparisons")
        
        if use_zero_shot and not history:
            comparison_limitations.append("Zero-shot inference mode (no training) - features may not be optimal for comparison")
        
        # Generate summary
        if is_single_group:
            summary = f"Single-group analysis completed. {num_crops} cells analyzed. "
            summary += f"{len(history)} epochs trained" if history else "Zero-shot inference (no training)"
            summary += "\n\n**Limitations for Group Comparison Queries:**\n"
            summary += "- All cells are in a single group ('default') - group comparison analysis cannot be performed\n"
            summary += "- To enable group comparison, provide treatment/group labels when uploading images\n"
            summary += "- Current analysis provides single-group clustering and UMAP visualization only"
            if termination_recommended:
                summary += f"\n\n**Termination Recommendation:**\n{termination_reason}"
        else:
            summary = f"Multi-group analysis completed. {num_crops} cells across {len(unique_groups)} groups. "
            summary += f"{len(history)} epochs trained" if history else "Zero-shot inference (no training)"
            summary += f". Groups: {', '.join(sorted(unique_groups))}"
            if comparison_limitations:
                summary += "\n\n**Note:** " + "; ".join(comparison_limitations)
        
        return {
            "summary": summary,
            "cell_count": num_crops,
            "mode": "training" if not use_zero_shot else "zero-shot",
            "epochs_trained": len(history),
            "final_loss": history[-1] if history else None,
            # full per-epoch loss curve for the label-free training-health signal (lr/epoch
            # pre-flight calibration); empty in zero-shot.
            "loss_history": list(history),
            "alignment": alignment,   # augmentation consistency (None unless compute_alignment)
            "training_config": training_config,
            "analyzer_kwargs": recorded_analyzer_kwargs,
            "batch_config": batch_config,
            "seconds": round(time.time() - started, 1),
            "split": None if split is None else {
                "n_train": n_train, "n_val": len(val_rows),
                "n_val_wells": len({unit_ids[i]["well"] for i in val_rows})},
            "training": training,
            "val_history": val_history,
            "resume_best": resume_best,
            "resume_end": resume_end,
            "intensity_norm_used": intensity_used,
            "adata_path": adata_path,
            "deliverables": [adata_path] + ([loss_path] if history and loss_path else []),
            "cluster_key": cluster_key,
            "cluster_resolution": cluster_resolution,
            "n_neighbors": n_neighbors,
            # task 075 M1: the resolved clustering representation (raw vs PCA, dimension,
            # metric) -- recorded unconditionally so downstream analysis can condition on
            # it; also written to adata.uns["clustering_representation"] (persists in the
            # saved h5ad). Derived from data properties only, never scalar_score.
            "clustering_representation": clustering_representation,
            "input_strategy": input_strategy,
            "embedding_variant_config": {
                "zero_shot": zero_shot,
                "augment_color": augment_color,
                "freeze_backbone": freeze_backbone,
                "field_aware": field_aware,
                "temperature": temperature,
                "learning_rate": learning_rate,
                "feature_source": "projector" if use_projector else "backbone_cls",
                "input_channels": input_channels,
                "channel_layouts": channel_layouts,
                "channel_adapter_init": model.channel_adapter.init_name,
                "symmetric_info_nce": True,
            },
            "ssl_diagnostics": {
                "input_channels": input_channels,
                "channel_layouts": channel_layouts,
                "channel_adapter_init": model.channel_adapter.init_name,
                "technical_metadata_available": any(
                    isinstance(m, dict) and any(
                        m.get(key) is not None
                        for key in (
                            "technical_group", "well", "well_id", "site", "site_id",
                            "plate", "plate_id", "experiment", "experiment_id",
                            "batch", "batch_id",
                        )
                    )
                    for m in cell_metadata
                ),
                "biological_labels_used_for_ssl_selection": False,
                "plate_nuisance_route": {
                    key: value for key, value in plate_route.items()
                    if key != "plate_ids"
                },
            },
            "source_reliability_report": source_reliability_report,
            "analysis_type": "cell_state_analysis",
            "num_groups": len(unique_groups),
            "groups": sorted(unique_groups) if not is_single_group else ["default"],
            "insufficient_for_comparison": insufficient_for_comparison,
            "comparison_limitations": comparison_limitations if comparison_limitations else None,
            "can_terminate_after_chain": termination_recommended,  # Can terminate after tool chain completes
            "termination_reason": termination_reason,
            "required_next_tools": ["Analysis_Visualizer_Tool"]  # This tool requires Analysis_Visualizer_Tool to complete the workflow
        }


if __name__ == "__main__":
    # Test script
    tool = Cell_State_Analyzer_Tool()
    print("Tool initialized successfully")

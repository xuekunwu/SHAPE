"""Attest and seed local native segmentation without resolving any model assets."""
from contextlib import contextmanager
import hashlib
from importlib.metadata import version
from pathlib import Path
import platform
import random
import threading

import numpy as np

from .trial_cache import SegmentationTrialCache, attest_model_identity, canonical, digest, REQUIRED_PARAMETERS


_RNG_LOCK = threading.RLock()


def configure(tool, root, *, seed: int, required_parameters=REQUIRED_PARAMETERS):
    """Enable exact-trial reuse for a loaded local model with pinned RNG inputs.

    Actual weight files and all SHAPE Python sources are hashed. No network or
    checkpoint resolver is invoked. Server-backed models need their own remote
    attestation and are rejected here instead of assuming the local identity.
    """
    import torch
    if isinstance(seed, bool) or not isinstance(seed, int) or not 0 <= seed < 2**32:
        raise ValueError('native cache seed must be an explicit uint32 integer')
    if getattr(tool, '_sam_server_url', None) or getattr(tool, 'model', None) is None:
        raise ValueError('native cache requires an attested loaded local model')
    package = Path(__file__).resolve().parents[2]
    sources = {p.relative_to(package).as_posix(): digest(p) for p in sorted(package.rglob('*.py'))}
    source_sha = hashlib.sha256(canonical(sources).encode()).hexdigest()
    # Version metadata failures are intentional: unknown computational runtimes
    # cannot share a cache namespace with known runtimes.
    versions = {p: version(p) for p in ('numpy', 'torch', 'cellpose', 'Pillow', 'tifffile',
                'scipy', 'scikit-image', 'matplotlib')}
    device = torch.device(tool.device)
    runtime = dict(tool=type(tool).__name__, python=platform.python_version(), platform=platform.platform(), packages=versions,
        cv2=__import__('cv2').__version__, device=str(device), seed=seed, rng_policy='scoped_native_v1',
        cuda=torch.version.cuda, deterministic_algorithms=torch.are_deterministic_algorithms_enabled(),
        cudnn_deterministic=torch.backends.cudnn.deterministic, cudnn_benchmark=torch.backends.cudnn.benchmark,
        cudnn_allow_tf32=torch.backends.cudnn.allow_tf32, matmul_allow_tf32=torch.backends.cuda.matmul.allow_tf32)
    net = getattr(tool.model, 'net', None)
    if net is None:
        raise ValueError('loaded model does not expose its inference network')
    runtime['parameter_dtypes'] = sorted({str(p.dtype) for p in net.parameters()})
    if device.type == 'cuda':
        index = device.index if device.index is not None else torch.cuda.current_device()
        runtime.update(device_index=index, device_name=torch.cuda.get_device_name(index),
                       compute_capability=list(torch.cuda.get_device_capability(index)))
    identity = attest_model_identity(tool.model, source_sha, runtime)
    tool.set_trial_cache(SegmentationTrialCache(Path(root), identity, required_parameters=required_parameters))
    tool._trial_cache_seed = seed
    return identity


@contextmanager
def native_rng(seed, device):
    """Restore process RNG streams after each seeded native call; serialize scopes.

    Different tool instances in this process share the lock. The caller still
    owns model scheduling. This pins stochastic native inputs, while recorded
    deterministic/CUDA flags state the limits of bitwise reproducibility.
    """
    import torch
    device = torch.device(device)
    devices = [] if device.type != 'cuda' else [
        device.index if device.index is not None else torch.cuda.current_device()]
    with _RNG_LOCK:
        python_state, numpy_state = random.getstate(), np.random.get_state()
        try:
            with torch.random.fork_rng(devices=devices):
                random.seed(seed)
                np.random.seed(seed)
                torch.random.default_generator.manual_seed(seed)
                if devices:
                    with torch.cuda.device(devices[0]):
                        torch.cuda.manual_seed(seed)
                yield
        finally:
            random.setstate(python_state)
            np.random.set_state(numpy_state)

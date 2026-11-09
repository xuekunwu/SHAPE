"""Durable, verified native segmentation trials shared by successive VRR passes.

This caches tool outputs, never reasoning/selection decisions. A hit still goes
through the caller's normal verification and delivery guards. Runtime/model/code
identity is supplied by the integration layer; incomplete identities fail closed.
"""
from __future__ import annotations

from contextlib import contextmanager
import copy
import hashlib
import inspect
import json
import math
import os
from pathlib import Path
import re
import threading
import time
import uuid


FORMAT = 'segmentation-native-trial-v1'
FILE_FIELDS = ('mask_path', 'overlay_path', 'output_path', 'prefilter_mask_path', 'mask_tif_path', 'organoid_mask_path',
               'latest_mask_path', 'latest_mask_tif_path', 'latest_overlay_path')
LIST_FILE_FIELDS = ('visual_outputs', 'deliverables')
REQUIRED_PARAMETERS = frozenset(('diameter', 'flow_threshold', 'cellprob_threshold',
    'min_size', 'niter', 'bsize', 'augment', 'resample', 'tile_overlap', 'model_type', 'color_class_filter'))


def effective_parameters(execute, supplied: dict) -> dict:
    """Expand the native execute signature, including fixed/default computational kwargs."""
    signature = inspect.signature(execute)
    bound = signature.bind_partial(**supplied)
    bound.apply_defaults()
    parameters = {}
    for name, parameter in signature.parameters.items():
        if name in ('self', 'image', 'image_id', 'query_cache_dir'):
            continue
        if parameter.kind in (parameter.VAR_POSITIONAL, parameter.VAR_KEYWORD):
            raise ValueError('unbounded native kwargs cannot be attested by this cache')
        if name not in bound.arguments:
            raise ValueError('missing native argument: ' + name)
        parameters[name] = bound.arguments[name]
    return parameters


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            value.update(block)
    return value.hexdigest()


def canonical(value) -> str:
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def attest_model_identity(model, source_sha256: str, runtime: dict) -> dict:
    """Hash the loaded model's resolved weight files; never resolve/download a model.

    Cellpose exposes its resolved checkpoint path as ``pretrained_model``. An
    unresolved alias or unavailable checkpoint is insufficient for safe reuse.
    Paths themselves do not identify weights; relocated byte-identical weights
    produce the same identity. An ensemble retains checkpoint order.
    """
    weights = getattr(model, 'pretrained_model', None)
    if isinstance(weights, (str, os.PathLike)):
        weights = [weights]
    if not isinstance(weights, (list, tuple)) or not weights:
        raise ValueError('loaded model does not expose resolved checkpoint files')
    hashes = []
    for weight in weights:
        path = Path(weight)
        if not path.is_file():
            raise ValueError('resolved model checkpoint is unavailable; cannot attest cache identity')
        hashes.append(digest(path))
    combined = hashes[0] if len(hashes) == 1 else hashlib.sha256(canonical(hashes).encode()).hexdigest()
    return dict(source_sha256=source_sha256, weights_sha256=combined,
                checkpoint_hashes=hashes, runtime=copy.deepcopy(runtime))


def atomic_json(path: Path, value) -> None:
    temporary = path.with_name(path.name + '.' + uuid.uuid4().hex + '.tmp')
    with temporary.open('x', encoding='utf-8') as stream:
        stream.write(canonical(value) + '\n')
        stream.flush()
        os.fsync(stream.fileno())
    os.replace(temporary, path)


@contextmanager
def file_lock(path: Path):
    """Cross-process lock released by the OS if a native worker terminates."""
    with path.open('a+b') as stream:
        if os.name == 'nt':
            import msvcrt
            if path.stat().st_size == 0:
                stream.write(b'\0')
                stream.flush()
            stream.seek(0)
            while True:
                try:
                    msvcrt.locking(stream.fileno(), msvcrt.LK_NBLCK, 1)
                    break
                except OSError:
                    time.sleep(0.05)
            try:
                yield
            finally:
                stream.seek(0)
                msvcrt.locking(stream.fileno(), msvcrt.LK_UNLCK, 1)
        else:
            import fcntl
            fcntl.flock(stream.fileno(), fcntl.LOCK_EX)
            try:
                yield
            finally:
                fcntl.flock(stream.fileno(), fcntl.LOCK_UN)


class SegmentationTrialCache:
    """One immutable output directory per attempted execution, plus a verified index.

    ``compute(folder)`` must write its outputs below that folder. Interrupted,
    failed or corrupted attempts are retained for audit and never accepted as a
    hit. Every hit hashes the current image and all recorded result artifacts.
    No cache is keyed only by image name, parameter subset or file timestamp.
    """

    def __init__(self, root: Path, identity: dict, *, required_parameters=REQUIRED_PARAMETERS):
        required = ('source_sha256', 'weights_sha256')
        if any(not re.fullmatch('[0-9a-f]{64}', str(identity.get(k, ''))) for k in required):
            raise ValueError('cache requires attested code and actual model-weight SHA256 identities')
        if not isinstance(identity.get('runtime'), dict) or not identity['runtime']:
            raise ValueError('cache requires an explicit runtime/device identity')
        canonical(identity)
        self.root = Path(root).resolve()
        # A full fingerprint plus retained attempt and atomic-write suffixes
        # routinely exceed Windows' legacy 260-character limit. Use Python's
        # extended absolute paths, without changing machine registry settings.
        if os.name == 'nt' and not str(self.root).startswith('\\\\?\\'):
            absolute = str(self.root)
            self.root = Path('\\\\?\\UNC\\' + absolute[2:] if absolute.startswith('\\\\')
                             else '\\\\?\\' + absolute)
        self.root.mkdir(parents=True, exist_ok=True)
        (self.root / 'events').mkdir(exist_ok=True)
        self.identity = copy.deepcopy(identity)
        self.required_parameters = frozenset(required_parameters)
        if not self.required_parameters:
            raise ValueError("a bounded native parameter contract is required")
        self._mutex = threading.Lock()
        self._locks = {}

    def _thread_lock(self, key):
        with self._mutex:
            return self._locks.setdefault(key, threading.Lock())

    @staticmethod
    def _artifacts(result: dict, attempt: Path) -> dict[str, str]:
        files = [result[k] for k in FILE_FIELDS if result.get(k)]
        for key in LIST_FILE_FIELDS:
            files.extend(result.get(key) or [])
        if not result.get('mask_path') or not result.get('overlay_path'):
            raise ValueError('a cacheable segmentation requires its mask and overlay')
        hashes = {}
        for name in files:
            path = Path(name).resolve()
            if not path.is_relative_to(attempt.resolve()) or not path.is_file():
                raise ValueError('native output must be an existing file inside its owned cache attempt')
            hashes[str(path)] = digest(path)
        return hashes

    def _load(self, index: Path, request: dict, key: str):
        try:
            value = json.loads(index.read_text(encoding='utf-8'))
            if value.get('format') != FORMAT or value.get('fingerprint') != key:
                return None, 'index_identity_mismatch'
            if value.get('request') != request or not value.get('cacheable'):
                return None, 'request_mismatch_or_failed_attempt'
            sealed = {k: v for k, v in value.items() if k != 'record_sha256'}
            if hashlib.sha256(canonical(sealed).encode()).hexdigest() != value.get('record_sha256'):
                return None, 'record_digest_mismatch'
            if not math.isfinite(float(value['native_seconds'])) or value['native_seconds'] < 0:
                return None, 'invalid_native_cost'
            result = value['result']
            attempt = Path(value['attempt']).resolve()
            if not attempt.is_relative_to(index.parent.resolve()):
                return None, 'attempt_outside_entry'
            actual = self._artifacts(result, attempt)
            if actual != value['artifacts']:
                return None, 'artifact_digest_mismatch'
            if result.get('status') != 'success' or int(result.get('cell_count', 0)) <= 0:
                return None, 'failed_or_empty_segmentation'
            return value, 'verified'
        except (OSError, ValueError, KeyError, TypeError):
            return None, 'missing_or_invalid_index_or_artifact'

    def get_or_compute(self, image: Path, image_id: str, effective_parameters: dict,
                       compute, *, context: dict | None = None) -> dict:
        """Return a native result; log cache provenance separately from stage/model cost.

        Supply the complete computational kwargs, including fixed parameters,
        normalization, channel and random-seed settings when present. Output
        directory kwargs belong to ``compute`` and are not computational inputs.
        """
        started = time.monotonic()
        image = Path(image).resolve()
        if not image_id or not isinstance(effective_parameters, dict) or not effective_parameters:
            raise ValueError('image identity and complete effective parameters are required')
        if self.required_parameters - effective_parameters.keys():
            raise ValueError('incomplete native recipe: ' + ', '.join(sorted(self.required_parameters - effective_parameters.keys())))
        if any(k in effective_parameters for k in ('query_cache_dir', 'image', 'image_id')):
            raise ValueError('image identity and output location have dedicated cache arguments')
        request = dict(format=FORMAT, image_id=str(image_id), image_sha256=digest(image),
                       effective_parameters=copy.deepcopy(effective_parameters), identity=self.identity)
        key = hashlib.sha256(canonical(request).encode()).hexdigest()
        entry = self.root / key
        entry.mkdir(exist_ok=True)
        index = entry / 'INDEX.json'
        with self._thread_lock(key), file_lock(entry / 'LOCK'):
            cached, reason = self._load(index, request, key)
            if cached is not None:
                result = copy.deepcopy(cached['result'])
                native_seconds = cached['native_seconds']
                hit = True
                attempt = Path(cached['attempt'])
            else:
                # Never overwrite partial attempt files or an earlier invalid index.
                if index.exists():
                    (entry / ('INDEX.previous.' + uuid.uuid4().hex + '.json')).write_bytes(index.read_bytes())
                attempt = entry / ('attempt_' + uuid.uuid4().hex)
                attempt.mkdir()
                atomic_json(attempt / 'ATTEMPT.json', dict(request=request, context=context or {},
                                                        created_at=time.time()))
                before = time.monotonic()
                try:
                    result = compute(attempt)
                except BaseException as exc:
                    atomic_json(attempt / 'ERROR.json', dict(fingerprint=key, request=request,
                        error_type=type(exc).__name__, error=str(exc), created_at=time.time()))
                    raise
                native_seconds = time.monotonic() - before
                if not isinstance(result, dict):
                    raise TypeError('native segmentation must return a result dictionary')
                cacheable = result.get('status') == 'success' and int(result.get('cell_count', 0)) > 0
                artifacts = self._artifacts(result, attempt) if cacheable else {}
                if digest(image) != request['image_sha256']:
                    raise ValueError('input image changed during native execution; no cache entry published')
                record = dict(format=FORMAT, fingerprint=key, request=request, attempt=str(attempt),
                              cacheable=cacheable, artifacts=artifacts, result=result,
                              native_seconds=native_seconds, created_at=time.time())
                record['record_sha256'] = hashlib.sha256(canonical(record).encode()).hexdigest()
                atomic_json(attempt / 'CALL.json', record)
                if cacheable:
                    atomic_json(index, record)
                hit = False
            if digest(image) != request['image_sha256']:
                raise ValueError('input image changed during cache validation')
            event = dict(format=FORMAT, fingerprint=key, hit=hit, reason=reason,
                         attempt=str(attempt), original_native_seconds=native_seconds,
                         original_reported_seconds=result.get('seconds'),
                         actual_seconds=time.monotonic() - started, context=context or {},
                         image_id=str(image_id), created_at=time.time())
            atomic_json(self.root / 'events' / (uuid.uuid4().hex + '.json'), event)
        result = copy.deepcopy(result)
        # The planner sees actual lookup time; original compute cost remains explicit.
        if hit:
            result['seconds'] = event['actual_seconds']
        result['native_cache'] = event
        return result

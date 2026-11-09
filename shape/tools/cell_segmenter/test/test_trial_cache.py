"""Safety and resume behavior of persistent native segmentation trials."""
from concurrent.futures import ThreadPoolExecutor
import copy
import importlib.util
import json
import os
from pathlib import Path
import time
from types import SimpleNamespace

import pytest

# Keep these artifact-cache tests independent of CUDA/agent initialization.
SPEC = importlib.util.spec_from_file_location('segmentation_trial_cache',
                                             Path(__file__).parents[1] / 'trial_cache.py')
MODULE = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(MODULE)
Cache = MODULE.SegmentationTrialCache


IDENTITY = dict(source_sha256='a' * 64, weights_sha256='b' * 64,
                runtime=dict(python='3.12', cellpose='test', device='cpu', seed=0))
PARAMS = dict(diameter=30., flow_threshold=.4, cellprob_threshold=0., min_size=15,
              niter=400, bsize=256, augment=False, resample=True, tile_overlap=.1,
              model_type='cpsam', color_class_filter=None)


@pytest.fixture
def setup(tmp_path):
    image = tmp_path / 'image.tif'
    image.write_bytes(b'original pixels')
    calls = []

    def compute(folder):
        calls.append(folder)
        (folder / 'mask.npy').write_bytes(b'valid-mask-payload')
        (folder / 'overlay.png').write_bytes(b'overlay-payload')
        return dict(status='success', cell_count=5, mask_path=str(folder / 'mask.npy'),
                    overlay_path=str(folder / 'overlay.png'), parameters=copy.deepcopy(PARAMS),
                    visual_outputs=[str(folder / 'overlay.png')], seconds=12.)
    return image, Cache(tmp_path / 'cache', IDENTITY), calls, compute


def test_resume_reuses_verified_files_and_preserves_compute_cost(setup):
    image, cache, calls, compute = setup
    first = cache.get_or_compute(image, 'i1', PARAMS, compute, context={'pass': 'cohort'})
    resumed = Cache(cache.root, IDENTITY)
    second = resumed.get_or_compute(image, 'i1', PARAMS, compute, context={'pass': 'per_image'})
    assert len(calls) == 1
    assert second['mask_path'] == first['mask_path']
    assert not first['native_cache']['hit'] and second['native_cache']['hit']
    assert second['native_cache']['original_native_seconds'] == first['native_cache']['original_native_seconds']
    assert second['seconds'] == second['native_cache']['actual_seconds']
    events = [json.loads(p.read_text()) for p in (cache.root / 'events').glob('*.json')]
    assert {e['context']['pass'] for e in events} == {'cohort', 'per_image'}


@pytest.mark.parametrize('change', [dict(diameter=45.), dict(augment=True), dict(bsize=512), dict(seed=1)])
def test_changed_effective_recipe_gets_a_separate_trial(setup, change):
    image, cache, calls, compute = setup
    first = cache.get_or_compute(image, 'i1', PARAMS, compute)
    second = cache.get_or_compute(image, 'i1', {**PARAMS, **change}, compute)
    assert len(calls) == 2
    assert first['mask_path'] != second['mask_path']


@pytest.mark.parametrize('field', ['weights_sha256', 'source_sha256', 'runtime'])
def test_changed_model_code_or_runtime_never_reuses_results(setup, field):
    image, cache, calls, compute = setup
    cache.get_or_compute(image, 'i1', PARAMS, compute)
    identity = copy.deepcopy(IDENTITY)
    identity[field] = 'c' * 64 if field.endswith('sha256') else dict(device='different-device')
    Cache(cache.root, identity).get_or_compute(image, 'i1', PARAMS, compute)
    assert len(calls) == 2


def test_changed_pixels_with_restored_timestamp_do_not_hit(setup):
    image, cache, calls, compute = setup
    first = cache.get_or_compute(image, 'i1', PARAMS, compute)
    previous = image.stat()
    image.write_bytes(b'different pixels')
    os.utime(image, ns=(previous.st_atime_ns, previous.st_mtime_ns))
    second = cache.get_or_compute(image, 'i1', PARAMS, compute)
    assert len(calls) == 2 and first['mask_path'] != second['mask_path']


@pytest.mark.parametrize('corruption', ['mask', 'result_metadata', 'partial_index'])
def test_corrupted_cache_is_recomputed_without_overwriting_old_attempt(setup, corruption):
    image, cache, calls, compute = setup
    first = cache.get_or_compute(image, 'i1', PARAMS, compute)
    entry = Path(first['native_cache']['attempt']).parent
    if corruption == 'mask':
        Path(first['mask_path']).write_bytes(b'corrupted')
    elif corruption == 'result_metadata':
        record = json.loads((entry / 'INDEX.json').read_text())
        record['result']['cell_count'] = 999
        (entry / 'INDEX.json').write_text(json.dumps(record))
    else:
        (entry / 'INDEX.json').write_text('{partial')
    second = cache.get_or_compute(image, 'i1', PARAMS, compute)
    assert len(calls) == 2 and second['mask_path'] != first['mask_path']
    assert Path(first['mask_path']).exists()
    assert list(entry.glob('INDEX.previous.*.json'))


def test_interrupted_compute_leaves_a_trace_and_can_resume(setup):
    image, cache, calls, compute = setup
    def interrupted(folder):
        (folder / 'partial-mask.npy').write_bytes(b'partial')
        raise RuntimeError('simulated worker interruption')
    with pytest.raises(RuntimeError, match='interruption'):
        cache.get_or_compute(image, 'i1', PARAMS, interrupted)
    assert list(cache.root.glob('*/attempt_*/ERROR.json'))
    assert not list(cache.root.glob('*/INDEX.json'))
    cache.get_or_compute(image, 'i1', PARAMS, compute)
    assert len(calls) == 1
    assert list(cache.root.glob('*/attempt_*/partial-mask.npy'))


def test_concurrent_cache_instances_compute_the_same_trial_once(setup):
    image, cache, calls, compute = setup
    other = Cache(cache.root, IDENTITY)
    def slow(folder):
        time.sleep(.05)
        return compute(folder)
    with ThreadPoolExecutor(2) as pool:
        results = list(pool.map(lambda c: c.get_or_compute(image, 'i1', PARAMS, slow), [cache, other]))
    assert len(calls) == 1
    assert sum(r['native_cache']['hit'] for r in results) == 1


def test_failed_native_outputs_are_retained_but_not_reused(setup):
    image, cache, calls, compute = setup
    def failed(folder):
        result = compute(folder)
        result['status'] = 'failed'
        return result
    cache.get_or_compute(image, 'i1', PARAMS, failed)
    cache.get_or_compute(image, 'i1', PARAMS, compute)
    assert len(calls) == 2
    assert len(list(cache.root.glob('*/attempt_*/CALL.json'))) == 2


def test_incomplete_recipe_fails_before_native_execution(setup):
    image, cache, calls, compute = setup
    with pytest.raises(ValueError, match='incomplete native recipe'):
        cache.get_or_compute(image, 'i1', {'diameter': 30}, compute)
    assert not calls


def test_attestation_uses_actual_weight_bytes_and_never_resolves_an_alias(tmp_path):
    weights = tmp_path / 'checkpoint'
    weights.write_bytes(b'actual checkpoint contents')
    model = SimpleNamespace(pretrained_model=str(weights))
    first = MODULE.attest_model_identity(model, 'a' * 64, dict(device='cpu'))
    assert first['weights_sha256'] == MODULE.digest(weights)
    weights.write_bytes(b'changed checkpoint contents')
    assert first != MODULE.attest_model_identity(model, 'a' * 64, dict(device='cpu'))
    with pytest.raises(ValueError, match='unavailable'):
        MODULE.attest_model_identity(SimpleNamespace(pretrained_model='unresolved-model-alias'), 'a' * 64, {})


def test_effective_parameters_expand_signature_defaults_without_dropping_fixed_kwargs():
    def execute(image, diameter=30, bsize=256, resample=True, query_cache_dir=None, image_id=None):
        pass
    parameters = MODULE.effective_parameters(execute, dict(image='x', diameter=45, image_id='i1'))
    assert parameters == dict(diameter=45, bsize=256, resample=True)

"""The cohort crop runner's input checks and accounting (no model, no tool call)."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'run_crop_cohort', Path(__file__).resolve().parents[1] / 'workflows/crop_from_fixed_masks.py')
runner = importlib.util.module_from_spec(spec)
spec.loader.exec_module(runner)


def make_bundle(root: Path, margins=(25, 25)) -> Path:
    inputs = root / 'inputs'
    for sub in ('images', 'masks', 'traces'):
        (inputs / sub).mkdir(parents=True)
    entries = []
    for ident, margin in zip(('i0001', 'i0002'), margins):
        for sub, data in (('images', b'image ' + ident.encode()), ('masks', b'mask ' + ident.encode())):
            path = inputs / sub / f'{ident}.tif'
            path.write_bytes(data)
            entries.append(dict(relative=f'{sub}/{ident}.tif', bytes=len(data), sha256=runner.sha256(path),
                                source=f'/archive/{ident}/cell_mask_{ident}.tif' if sub == 'masks' else str(path)))
        runner.write(inputs / 'traces' / f'{ident}.json', dict(selected_parameters=dict(min_area=15, margin=margin)))
    runner.write(inputs / 'PILOT.json', dict(images=['i0001', 'i0002']))
    runner.write(inputs / 'MANIFEST.json', dict(entries=entries, pilot_sha256=runner.sha256(inputs / 'PILOT.json')))
    return inputs


def test_the_bundle_is_checked_masks_keep_their_source_names_and_shared_kwargs_are_common(tmp_path):
    inputs = make_bundle(tmp_path)
    root = tmp_path / 'run'
    root.mkdir()
    images, shared = runner.bundle(inputs, root)
    assert [Path(i['nuclei_mask']).name for i in images] == ['cell_mask_i0001.tif', 'cell_mask_i0002.tif']
    assert all(i['min_area']==15 and i['margin']==25 for i in images)
    assert [i['source_image_id'] for i in images] == ['i0001', 'i0002'] and shared == dict(min_area=15, margin=25)
    mixed = make_bundle(tmp_path / 'mixed', margins=(25, 30))
    (tmp_path / 'mixed' / 'run').mkdir()
    assert runner.bundle(mixed, tmp_path / 'mixed' / 'run')[1] == dict(min_area=15)
    (inputs / 'images' / 'i0002.tif').write_bytes(b'changed')
    (tmp_path / 'again').mkdir()
    with pytest.raises(ValueError, match='differs from the manifest'):
        runner.bundle(inputs, tmp_path / 'again')


def test_usage_and_delivered_readouts(tmp_path):
    calls = [dict(usage=dict(prompt_tokens=10, completion_tokens=3, total_tokens=13)),
             dict(usage=dict(input_tokens=5, output_tokens=2, reasoning_tokens=1, total_tokens=7)), dict(usage=None)]
    assert runner.usage(calls) == dict(input=15, output=5, reasoning=1, total=20)
    meta = tmp_path / 'm.json'
    meta.write_text(json.dumps(dict(cell_metadata=[
        dict(qc=dict(neighbor_contamination=0.2, target_dominance=0.8, target_retention=1.0, centroid_offset=0.1)),
        dict(qc=dict(neighbor_contamination=0.4, target_dominance=0.6, target_retention=0.9, centroid_offset=0.3))])))
    out = runner.delivered_readouts({'a': str(meta)})
    assert out['n_crops'] == 2 and out['mean_target_dominance'] == pytest.approx(0.7)
    assert out['min_target_retention'] == 0.9

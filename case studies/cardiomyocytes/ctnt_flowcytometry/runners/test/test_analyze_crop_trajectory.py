"""Offline source preparation does not need live search or model dependencies."""
import importlib.util
import json
from pathlib import Path

import pytest

spec = importlib.util.spec_from_file_location(
    'crop_trajectory_analysis', Path(__file__).resolve().parents[1] / 'evaluation/historical/crop_trajectory_v10.py')
analysis = importlib.util.module_from_spec(spec)
spec.loader.exec_module(analysis)


def completed_run(tmp_path):
    run = tmp_path / 'run'
    source = run / 'source_v10'
    source.mkdir(parents=True)
    (source / 'scientific.py').write_text('ORIGINAL = True\n')
    analysis.write(run / 'SOURCE_MANIFEST.json', dict(branch='cs-cm', base_commit='test-base',
        files={'scientific.py': analysis.sha(source / 'scientific.py')}))
    digest = analysis.sha(run / 'SOURCE_MANIFEST.json')
    for name in ('FINAL_SUMMARY.json', 'LAUNCH_PROVENANCE.json'):
        analysis.write(run / name, dict(source_manifest_sha256=digest))
    analysis.write(run / 'RUN_META.json', dict(status='COMPLETED'))
    return run


def test_changed_parent_source_fails_before_output_creation(tmp_path):
    run = completed_run(tmp_path)
    (run / 'source_v10/scientific.py').write_text('CHANGED = True\n')
    out = tmp_path / 'analysis'
    with pytest.raises(AssertionError):
        analysis.prepare_output(run, out)
    assert not out.exists()


def test_duplicate_identity_cannot_silently_replace_a_row():
    with pytest.raises(ValueError, match='Duplicate'):
        analysis.decode_rows([['image', 'cell', {}], ['image', 'cell', {}]])


def test_payload_digest_is_canonical_and_rejects_nonfinite_values():
    assert analysis.payload_sha({'b': 1, 'a': 2}) == analysis.payload_sha({'a': 2, 'b': 1})
    with pytest.raises(ValueError):
        analysis.payload_sha({'value': float('nan')})

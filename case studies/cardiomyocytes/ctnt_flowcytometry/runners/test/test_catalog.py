"""Single public entrypoint, implementation inventory and safe frozen-result replay."""
import importlib
from pathlib import Path
import subprocess
import sys

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[5]))
PKG = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners'
inventory = importlib.import_module(PKG + '.runtime.inventory')
catalog = importlib.import_module(PKG + '.evaluation.result_catalog')


def test_root_has_one_public_entry_and_no_obsolete_compatibility_modules():
    assert {p.name for p in catalog.ROOT.glob('*.py')} == {'run.py','__init__.py'}
    assert not (catalog.ROOT/'operations').exists()
    assert not (catalog.ROOT/'pipeline').exists()
    with pytest.raises(ModuleNotFoundError):
        importlib.import_module(PKG+'.data')


def test_catalog_does_not_misclassify_direct_bootstrap_as_library_cli():
    records = {r['path']: r for r in inventory.scripts()}
    assert records['dataset/preparation.py']['kind'] == 'library'
    assert records['workflows/experiment_runner.py']['help_supported']
    assert not records['dataset/import_archive.py']['help_supported']


def test_public_and_internal_worker_cli_work_from_another_directory(tmp_path):
    for relative in ('run.py', 'workflows/experiment_runner.py'):
        result = subprocess.run([sys.executable, str(catalog.ROOT / relative), '--help'],
                                cwd=tmp_path, capture_output=True, text=True, timeout=30)
        assert result.returncode == 0, result.stderr
        assert 'compare-pilot' in result.stdout


@pytest.fixture
def export(tmp_path, monkeypatch):
    case = tmp_path / 'case'
    source = case / 'results/frozen'
    source.mkdir(parents=True)
    (source / 'replay.py').write_text(
        'from pathlib import Path\nPath("rebuilt.txt").write_text("replayed")\n')
    (source / 'evidence.txt').write_text('frozen')
    monkeypatch.setattr(catalog, 'CASE', case)
    monkeypatch.setattr(catalog, 'result_sets', lambda: [dict(id='fixture', path='results/frozen', replay_script='replay.py')])
    return case, source, tmp_path / 'fresh'


def test_replay_executes_only_a_copy_preserving_original_evidence(export):
    _, source, out = export
    before = {p.name: p.read_bytes() for p in source.iterdir()}
    catalog.replay('fixture', out)
    assert (out / 'rebuilt.txt').read_text() == 'replayed'
    assert {p.name: p.read_bytes() for p in source.iterdir()} == before


def test_replay_refuses_existing_or_published_output_before_copy(export):
    case, _, out = export
    out.mkdir()
    for destination in (out, case / 'results/new'):
        with pytest.raises(ValueError, match='fresh output'):
            catalog.replay('fixture', destination)
    assert not (case / 'results/new').exists()


def test_replay_refuses_unregistered_and_escaping_sources(export, monkeypatch):
    _, _, out = export
    with pytest.raises(ValueError, match='unknown result'):
        catalog.replay('unknown', out)
    monkeypatch.setattr(catalog, 'result_sets', lambda: [dict(id='escape', path='../outside', replay_script='replay.py')])
    with pytest.raises(ValueError, match='outside results'):
        catalog.replay('escape', out)
    assert not out.exists()

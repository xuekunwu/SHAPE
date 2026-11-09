"""Prevent orchestration, inference and result replay from leaking into lower layers."""
import ast
import importlib
import importlib.util
from pathlib import Path

PKG = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners'
ROOT = Path(__file__).resolve().parents[1]


def local_dependencies(path):
    relative = path.relative_to(ROOT).with_suffix('')
    package = PKG + '.' + '.'.join(relative.parts[:-1])
    for node in ast.walk(ast.parse(path.read_text(encoding='utf-8'))):
        if isinstance(node, ast.ImportFrom):
            base = importlib.util.resolve_name('.' * node.level + (node.module or ''), package) if node.level else node.module
            if base and base.startswith(PKG + '.'):
                yield base.removeprefix(PKG + '.')
        elif isinstance(node, ast.Constant) and isinstance(node.value, str) and node.value.startswith(PKG + '.'):
            yield node.value.removeprefix(PKG + '.')


def test_layer_dependency_boundaries():
    allowed = {
        'runtime': {'runtime'},
        'dataset': {'runtime', 'dataset'},
        'prediction': {'runtime', 'dataset', 'prediction', 'evaluation'},
        'evaluation': {'runtime', 'dataset', 'evaluation'},
    }
    violations = []
    for layer, destinations in allowed.items():
        for path in (ROOT / layer).rglob('*.py'):
            for dependency in local_dependencies(path):
                if dependency.split('.')[0] not in destinations:
                    violations.append((str(path.relative_to(ROOT)), dependency))
                # Predictor may share pure metric arithmetic, never evaluation IO.
                if layer == 'prediction' and dependency.startswith('evaluation.') and dependency != 'evaluation.metrics':
                    violations.append((str(path.relative_to(ROOT)), dependency))
    assert not violations, violations


def test_runtime_does_not_need_dataset_import_or_scientific_stage_sequence():
    launcher = importlib.import_module(PKG + '.runtime.stage_runner')
    assert hasattr(launcher, 'invoke') and not hasattr(launcher, 'run_stages')
    sequence = importlib.import_module(PKG + '.workflows.stage_sequence')
    assert hasattr(sequence, 'run_stages')
    artifacts = importlib.import_module(PKG + '.runtime.artifacts')
    preparation = importlib.import_module(PKG + '.dataset.preparation')
    assert preparation.digest is artifacts.digest
    assert preparation.fresh_output is artifacts.fresh_output


def test_result_replay_does_not_own_implementation_inventory():
    catalog = importlib.import_module(PKG + '.evaluation.result_catalog')
    inventory = importlib.import_module(PKG + '.runtime.inventory')
    assert not hasattr(catalog, 'scripts')
    assert not hasattr(inventory, 'replay')
    assert set(inventory.GROUPS) == {'dataset', 'runtime', 'workflows', 'prediction', 'evaluation'}
    for old in ('execution', 'experiments', 'reports'):
        assert not (ROOT / old).exists()

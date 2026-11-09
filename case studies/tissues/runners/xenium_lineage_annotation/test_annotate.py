"""End-to-end tests of annotate.main on a synthetic section with planted lineages, a tumor and low-count cells.

The CellTypist opinion comes from a real CellTypist model trained here on a separate synthetic reference.
"""
from pathlib import Path
import sys

import numpy as np
import pandas as pd
import pytest
from scipy import sparse

celltypist = pytest.importorskip('celltypist')
import anndata as ad
import scanpy as sc

sys.path.insert(0, str(Path(__file__).resolve().parent))
import annotate
import markers

TUMOR_GENES = [f'TUM{i}' for i in range(6)]
NORMAL_GENES = [f'NOR{i}' for i in range(6)]
GROUPS = {'Epithelial': 300, 'Tumor': 300, 'Endothelial': 200, 'Stromal': 200, 'Immune': 300}
GENES = list(dict.fromkeys([g for v in markers.LINEAGE_GENES.values() for g in v] + TUMOR_GENES + NORMAL_GENES
                           + [f'FILL{i}' for i in range(300)]))


def counts(group, n, rng):
    rate = np.full(len(GENES), 0.02)
    rate[[GENES.index(g) for g in GENES if g.startswith('FILL')]] = 0.3
    high = {'Tumor': markers.LINEAGE_GENES['Epithelial'] + TUMOR_GENES,
            'Epithelial': markers.LINEAGE_GENES['Epithelial'] + NORMAL_GENES}.get(group, markers.LINEAGE_GENES.get(group, []))
    rate[[GENES.index(g) for g in high]] = 4.0
    return rng.poisson(rate, size=(n, len(GENES))).astype(np.float32)


def section(tmp, rng):
    blocks = [counts(g, n, rng) for g, n in GROUPS.items()]
    low = np.zeros((30, len(GENES)), np.float32)
    low[:, :3] = 1
    x = np.vstack(blocks + [low])
    truth = np.concatenate([[g] * n for g, n in GROUPS.items()] + [['Others'] * 30])
    ids = [f'cell{i:05d}-1' for i in range(len(x))]
    a = ad.AnnData(X=sparse.csr_matrix(x), obs=pd.DataFrame({'cell_id': ids}, index=ids),
                   var=pd.DataFrame(index=GENES), obsm={'spatial': rng.uniform(0, 1000, (len(x), 2))})
    a.write_h5ad(tmp / 'section.h5ad')
    pd.DataFrame({'cell_id': ids}).to_parquet(tmp / 'cells.parquet')
    return truth


def train_model(path, rng, groups):
    ref = ad.AnnData(X=np.vstack([counts(g, 150, rng) for g in groups]), var=pd.DataFrame(index=GENES),
                     obs=pd.DataFrame({'label': np.repeat([f'ref_{g}' for g in groups], 150)}))
    sc.pp.normalize_total(ref, target_sum=1e4)
    sc.pp.log1p(ref)
    celltypist.train(ref, labels='label', feature_selection=False, n_jobs=1, max_iter=200).write(str(path))


@pytest.fixture
def organ(tmp_path, monkeypatch):
    monkeypatch.setitem(markers.TUMOR, 'TestOrgan', dict(parent='Epithelial', tumor=TUMOR_GENES, normal=NORMAL_GENES))
    monkeypatch.setitem(markers.CELLTYPIST_MODEL, 'TestOrgan', 'test.pkl')
    monkeypatch.setitem(markers.METHOD, 'TestOrgan', 'v3')
    return tmp_path


def run(tmp, groups, monkeypatch):
    rng = np.random.default_rng(0)
    truth = section(tmp, rng)
    train_model(tmp / 'test.pkl', rng, groups)
    monkeypatch.setitem(markers.CELLTYPIST_LINEAGE, 'test.pkl', {f'ref_{g}': g for g in groups})
    annotate.main('TestOrgan', tmp / 'section.h5ad', tmp / 'cells.parquet', tmp, tmp / 'out')
    cells = pd.read_csv(tmp / 'out' / 'cells.csv.gz')
    return truth, cells


def test_end_to_end_recovers_planted_classes(organ, monkeypatch):
    truth, cells = run(organ, ['Epithelial', 'Endothelial', 'Stromal', 'Immune'], monkeypatch)
    lab = pd.read_csv(organ / 'out' / 'class.csv.gz')
    assert list(lab.columns) == ['cell_id', 'class'] and set(lab['class']) <= set(markers.CLASSES)
    for g in list(GROUPS) + ['Others']:
        acc = (cells['class'][truth == g] == g).mean()
        assert acc >= 0.9, (g, acc)
    assert (cells['evidence'][truth == 'Others'] == 'low_counts').all()
    assert (cells['evidence'][truth != 'Others'] == 'agree').mean() >= 0.9
    sub = pd.read_csv(organ / 'out' / 'tumor_subclusters.csv')
    assert sub['tumor'].any() and not sub['tumor'].all()


def test_marker_call_kept_when_the_model_lacks_the_lineage(organ, monkeypatch):
    truth, cells = run(organ, ['Epithelial', 'Stromal', 'Immune'], monkeypatch)
    endo = cells[truth == 'Endothelial']
    assert (endo['class'] == 'Endothelial').mean() >= 0.9
    assert (endo['evidence'] == 'marker_only').mean() >= 0.9


def test_absent_lineage_is_never_called(organ, monkeypatch):
    monkeypatch.setitem(markers.ABSENT, 'TestOrgan', ['Epithelial'])
    truth, cells = run(organ, ['Epithelial', 'Endothelial', 'Stromal', 'Immune'], monkeypatch)
    assert not cells['class'].isin(['Epithelial', 'Tumor']).any()
    for g in ('Endothelial', 'Stromal', 'Immune'):
        assert (cells['class'][truth == g] == g).mean() >= 0.9, g


def test_cell_id_mismatch_is_refused(organ, monkeypatch):
    rng = np.random.default_rng(0)
    section(organ, rng)
    ids = pd.read_parquet(organ / 'cells.parquet')
    ids.loc[0, 'cell_id'] = 'other-1'
    ids.to_parquet(organ / 'cells.parquet')
    with pytest.raises(ValueError, match='does not match'):
        annotate.main('TestOrgan', organ / 'section.h5ad', organ / 'cells.parquet', organ, organ / 'out')
    assert not (organ / 'out' / 'class.csv.gz').exists()


def test_combination_rules():
    # clusters: a agree, b no positive marker score, c strong marker conflict, d weak conflict, e lineage the model
    # lacks, f no CellTypist vote, g weak conflict where CellTypist has no majority (25% plurality)
    cluster = np.repeat(list('abcdefg'), 4)
    cols = ['Epithelial', 'Endothelial', 'Stromal', 'Immune', 'Melanocytic']
    rows = {'a': [1.0, 0, 0, 0, 0], 'b': [-0.1] * 5, 'c': [0, 0, 0.8, 0, 0], 'd': [0, 0, 0.15, 0.05, 0],
            'e': [0, 0, 0, 0, 0.5], 'f': [0, 0.5, 0, 0, 0], 'g': [0, 0, 0.15, 0.05, 0]}
    scores = pd.DataFrame([rows[k] for k in cluster], columns=cols)
    ct = np.array(['Epithelial'] * 4 + ['Immune'] * 4 + ['Epithelial'] * 4 + ['Epithelial'] * 3 + ['Stromal']
                  + ['Immune'] * 4 + [None] * 4 + ['Epithelial', 'Immune', 'Endothelial', 'Stromal'], dtype=object)
    main = {'Epithelial', 'Endothelial', 'Stromal', 'Immune'}
    by = lambda k: (list(lineage[cluster == k]), list(evidence[cluster == k]))
    for min_majority in (0.0, annotate.MIN_MAJORITY):
        lineage, evidence, _, _ = annotate.combine(cluster, scores, ct, main, min_majority)
        assert by('a') == (['Epithelial'] * 4, ['agree'] * 4)
        assert by('b') == (['Immune'] * 4, ['celltypist_only'] * 4)
        assert by('c') == (['Stromal'] * 4, ['marker_strong'] * 4)
        assert by('d') == (['Others'] * 3 + ['Stromal'], ['unresolved'] * 3 + ['conflict_cell_agree'])
        assert by('e') == (['Melanocytic'] * 4, ['marker_only'] * 4)
        assert by('f') == (['Endothelial'] * 4, ['marker_only'] * 4)
    assert by('g') == (['Stromal'] * 4, ['ct_no_majority'] * 4)
    lineage, evidence, _, _ = annotate.combine(cluster, scores, ct, main)
    assert by('g') == (['Others'] * 3 + ['Stromal'], ['unresolved'] * 3 + ['conflict_cell_agree'])
    lineage, evidence, _, _ = annotate.combine(cluster, scores, None, set())
    assert by('b') == (['Others'] * 4, ['no_evidence'] * 4) and by('c') == (['Stromal'] * 4, ['marker_only'] * 4)


def test_v3_drops_genes_detected_in_under_one_percent_of_cells():
    x = np.zeros((1000, 3), np.float32)
    x[:4, 0] = 1    # rare: 0.4% of cells
    x[:500, 1] = 1  # common
    a = ad.AnnData(X=sparse.csr_matrix(x), var=pd.DataFrame(index=['rare', 'common', 'absent']))
    assert annotate.usable_genes(a, v3=True) == {'common'}
    assert annotate.usable_genes(a, v3=False) == {'rare', 'common', 'absent'}


def test_v3_tumor_background_includes_cluster_means(monkeypatch):
    # a sparse tumor program: almost every background cell scores 0, so the cell-level background is ~0 and a weak
    # parent subcluster passes it; a small background cluster that carries more signal must raise the background
    rng = np.random.default_rng(1)
    groups = {'tumor': (150, 2.0), 'weak': (150, 0.5), 'bg_signal': (120, 1.0), 'bg': (2400, 0.0)}
    t, lineage, cluster, pca = [], [], [], []
    for i, (g, (n, value)) in enumerate(groups.items()):
        t.append(value + rng.normal(0, 0.05, n))
        lineage += ['Epithelial' if g in ('tumor', 'weak') else 'Stromal'] * n
        cluster += [g] * n
        pca.append(rng.normal(10.0 * i, 1.0, (n, 10)))
    t = np.concatenate(t)
    monkeypatch.setattr(annotate, 'score', lambda a, genes, name: t)
    a = ad.AnnData(X=sparse.csr_matrix((len(t), 1), dtype=np.float32), obsm={'X_pca': np.vstack(pca)})
    lineage, cluster = np.array(lineage, dtype=object), np.array(cluster)
    cfg = dict(parent='Epithelial', tumor=TUMOR_GENES, normal=[])
    for cl, weak_called in ((None, True), (cluster, False)):
        tumor, _, _, record, _, _ = annotate.call_tumor(a, lineage, cfg, set(TUMOR_GENES), cl)
        assert tumor[cluster == 'tumor'].mean() >= 0.9
        assert (tumor[cluster == 'weak'].mean() >= 0.9) == weak_called, record
    assert record['tau_cell'] < 0.5 < record['tau']

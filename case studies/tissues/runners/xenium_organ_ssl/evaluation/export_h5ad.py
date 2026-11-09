"""One h5ad per arm with the cells of all three draws of an organ (the SSL embedding, the cell records, the split, the
cross-tissue class label, the held-out probe predictions) and a UMAP of the merged embedding.

UMAP protocol of Fig4b / Fig2e: PCA (50) and UMAP (n_neighbors 30, min_dist 0.3, 300 epochs, seed 20261017) fitted on
the development_train rows of the three draws, every other row transformed. The three draws of a trained arm come from
three checkpoints (one per draw), so their embeddings are separate spaces merged only for display.
Input: a copy of the VM run root (ROOT/<organ>/draw<d>/{prepared,runs}, ROOT/eval/...).
usage: export_h5ad.py ROOT ORGAN LABELS_NAME PROBE [PROBE ...] OUT_DIR
  e.g. export_h5ad.py vm lung HumanLungCancer_class_v2 probe_class6_v2 probe_v1 hf/figure4b_tissues_celltypes/lung"""
import hashlib
import json
from pathlib import Path
import sys

import anndata as ad
import numpy as np
import pandas as pd
from sklearn.decomposition import PCA
import umap

ARMS = ('control', 'shape', 'codex', 'scportrait')
DRAWS = (1, 2, 3)
SPLIT = {'development_train': 'Train', 'development_val': 'Validation', 'evaluation_only': 'Held-out'}
SEED, N_PCS, CHUNK = 20261017, 50, 8192
UMAP_PARAMS = dict(n_neighbors=30, min_dist=0.3, n_epochs=300, n_components=2, random_state=SEED)
CELL_FIELDS = ('x_centroid', 'y_centroid', 'tile', 'well', 'cell_area', 'nucleus_area', 'transcript_counts')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def require(ok, message):
    if not ok:
        raise ValueError(message)


def project(x, train):
    pcs = PCA(n_components=N_PCS, random_state=SEED).fit(x[train]).transform(x)
    reducer = umap.UMAP(**UMAP_PARAMS).fit(pcs[train])
    emb = np.zeros((len(x), 2), dtype=np.float32)
    emb[train] = reducer.embedding_
    rest = np.flatnonzero(~train)
    for start in range(0, len(rest), CHUNK):
        emb[rest[start:start + CHUNK]] = reducer.transform(pcs[rest[start:start + CHUNK]])
    require(np.isfinite(emb).all(), 'non-finite UMAP coordinates')
    return emb


def main(root, organ, labels_name, probes, out):
    root, out = Path(root), Path(out)
    out.mkdir(parents=True, exist_ok=True)
    labels_path = root / 'eval' / 'labels' / f'{labels_name}.csv.gz'
    label = pd.read_csv(labels_path, dtype=str).set_index('cell_id')['class']
    extra = {}
    if organ == 'lung':          # the HLCA-level CellTypist labels of the 10-class probe (probe_v1)
        extra['celltype_final'] = pd.read_csv(root / 'eval' / 'labels' / 'lung_celltype_final.csv.gz',
                                              dtype=str).set_index('cell_id')['celltype_final']
    frames, ids_by_draw = [], {}
    for d in DRAWS:
        cells = json.loads((root / organ / f'draw{d}' / 'prepared' / 'private' / 'cells.json').read_text())
        order = [c['cell_id'] for c in json.loads((root / organ / f'draw{d}' / 'runs' / 'control' / 'cells.json').read_text())]
        require(set(order) == set(cells), f'draw {d}: run cells differ from the preparation')
        ids_by_draw[d] = order
        f = pd.DataFrame([{k: cells[c][k] for k in CELL_FIELDS} for c in order], index=pd.Index(order, name='cell_id'))
        f.insert(0, 'usage', [cells[c]['usage'] for c in order])
        f.insert(0, 'draw', d)
        frames.append(f)
    obs = pd.concat(frames)
    require(obs.index.is_unique, 'draws are not disjoint')
    obs['split'] = pd.Categorical(obs['usage'].map(SPLIT), categories=list(SPLIT.values()))
    obs['draw'] = pd.Categorical(obs['draw'], categories=list(DRAWS))
    require(obs.index.isin(label.index).all(), 'cells without a class label')
    obs['class'] = pd.Categorical(label.reindex(obs.index).to_numpy())
    for name, series in extra.items():
        require(obs.index.isin(series.index).all(), f'cells without {name}')
        obs[name] = pd.Categorical(series.reindex(obs.index).to_numpy())
    probe_records = {}
    for probe in probes:
        preds = pd.concat([pd.read_csv(root / 'eval' / organ / f'draw{d}' / probe / 'predictions.csv', dtype=str).set_index('cell_id')
                           for d in DRAWS])
        metrics = {d: json.loads((root / 'eval' / organ / f'draw{d}' / probe / 'metrics.json').read_text()) for d in DRAWS}
        probe_records[probe] = (preds, metrics)
    train = (obs['usage'] == 'development_train').to_numpy()
    for arm in ARMS:
        xs, sources = [], {}
        for d in DRAWS:
            run = root / organ / f'draw{d}' / 'runs' / arm
            order = [c['cell_id'] for c in json.loads((run / 'cells.json').read_text())] if (run / 'cells.json').exists() else None
            x = np.load(run / 'embedding.npy')
            require(x.shape == (len(ids_by_draw[d]), 768) and np.isfinite(x).all(), f'{arm} draw {d}: bad embedding')
            if order is not None:
                require(order == ids_by_draw[d], f'{arm} draw {d}: cell order differs from control')
            sel = json.loads((run / 'SSL_SELECTION.json').read_text())
            comp = json.loads((run / 'embed_COMPLETE.json').read_text())
            sources[f'draw{d}'] = dict(run=f'{organ}/draw{d}/runs/{arm}', embedding_sha256=digest(run / 'embedding.npy'),
                                       selection=sel.get('mode'), checkpoint_sha256=sel.get('checkpoint_sha256', ''),
                                       stage_seconds=comp.get('seconds'), tokens=comp.get('usage', {}).get('total_tokens', 0))
            xs.append(x.astype(np.float32))
        a = ad.AnnData(X=np.concatenate(xs), obs=obs.copy())
        probes_uns = {}
        for probe, (preds, metrics) in probe_records.items():
            column = preds.columns[1]                       # the label column of that probe (class or celltype_final)
            pred = preds['pred_' + arm].reindex(a.obs.index)
            a.obs[f'{probe}_pred'] = pd.Categorical(pred.to_numpy())
            # held-out cells of the probe's classes only (a probe drops classes with too few training cells)
            scored = sum(sum(metrics[d]['counts']['evaluation_only'].values()) for d in DRAWS)
            require(int(pred.notna().sum()) == scored == len(preds) and
                    (a.obs.loc[pred.notna(), 'usage'] == 'evaluation_only').all(), f'{probe}: predictions not on held-out')
            probes_uns[probe] = dict(label_column=column, labels_sha256=metrics[1]['labels_sha256'],
                                     classes=metrics[1]['classes'], excluded_classes=metrics[1].get('excluded_classes', []),
                                     **{f'draw{d}': dict(C=metrics[d]['arms'][arm]['C'], **metrics[d]['arms'][arm]['test'])
                                        for d in DRAWS})
        print(organ, arm, a.shape, 'UMAP ...', flush=True)
        a.obsm['X_umap'] = project(np.asarray(a.X), train)
        a.uns['provenance'] = dict(organ=organ, arm=arm, draws=list(DRAWS), sources=sources,
                                   labels=dict(file=labels_path.name, sha256=digest(labels_path)), probes=probes_uns,
                                   umap=dict(fit_on='development_train rows of all draws', n_pcs=N_PCS, **UMAP_PARAMS),
                                   note='embeddings of trained arms come from one checkpoint per draw')
        a.uns['provenance'] = json.loads(json.dumps(a.uns['provenance'], default=str))
        path = out / f'{organ}_{arm}.h5ad'
        a.write_h5ad(path, compression='gzip')
        print(path, digest(path), flush=True)


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], sys.argv[4:-1], sys.argv[-1])

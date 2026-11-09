"""Cross-tissue class annotation of one Xenium Prime section: Epithelial / Endothelial / Stromal / Immune / Tumor / Others.

Protocol (raw counts in the h5ad X, gene sets and model maps in markers.py):
1. Cells with fewer than MIN_COUNTS transcripts -> Others (evidence low_counts). The rest: normalize_total 1e4 and
   log1p (the CellTypist input convention of the lung notebook), PCA (50, all panel genes, no scaling), 15-NN graph,
   Leiden at RESOLUTION -> clusters.
2. Marker opinion: sc.tl.score_genes per lineage set (panel-intersected); cluster call = the lineage with the highest
   cluster-mean score, cell call = the highest cell score.
3. CellTypist opinion: the organ's model, per cell (no majority voting), label -> lineage; cluster call = the majority
   lineage of the cluster (labels mapped to None do not vote).
4. Combination per cluster, in this order: no marker set has a positive cluster-mean score -> the CellTypist call
   (celltypist_only), or Others without a model (no_evidence); the marker call is a lineage the model cannot output,
   or the organ has no model -> the marker call (marker_only); the calls agree -> that lineage (agree); they disagree
   and the marker margin (best minus second cluster-mean score) is at least MARGIN -> the marker call (marker_strong);
   otherwise each cell keeps the lineage that both opinions give it (conflict_cell_agree), else Others (unresolved).
5. Tumor: cells of the organ's parent lineage are re-clustered (Leiden TUMOR_RESOLUTION on the same PCA). A subcluster
   is Tumor when its mean tumor score exceeds the background (TAU_Q quantile of the tumor score in main-lineage cells
   outside the parent) and its mean normal-counterpart score.
6. Melanocytic, Adipocyte and Others lineages end in Others.

Method v3 (markers.METHOD; lung stays v2) adds three rules: gene-set genes detected in fewer than MIN_DETECT of the
section's cells are dropped (steps 2 and 5); a CellTypist cluster call with a plurality below MIN_MAJORITY does not vote,
so the marker call is used (ct_no_majority); the tumor background is at least the highest mean tumor score of a
main-lineage cluster outside the parent with MIN_BG_CLUSTER or more cells.

usage: python annotate.py ORGAN H5AD CELLS_PARQUET MODEL_DIR OUT
"""
import hashlib
import json
import logging
from pathlib import Path
import sys
import time

import anndata as ad
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import scanpy as sc

import markers

MIN_COUNTS = 10
N_PCS, N_NEIGHBORS = 50, 15
RESOLUTION, TUMOR_RESOLUTION = 2.0, 1.0
TAU_Q = 0.95
MARGIN = 0.2
MIN_SET_GENES = 2
MIN_DETECT, MIN_MAJORITY, MIN_BG_CLUSTER = 0.01, 0.5, 100  # v3 rules
CHUNK = 100_000
SEED = 20261027
GZIP = {'method': 'gzip', 'mtime': 0}
_CLASS_COLORS = {'Epithelial': '#E69F00', 'Endothelial': '#D55E00', 'Stromal': '#009E73', 'Immune': '#0072B2',
                 'Tumor': '#CC79A7', 'Others': '#BBBBBB'}
T0 = time.time()


def log(msg):
    print(f'[{time.time() - T0:7.0f} s] {msg}', flush=True)


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def usable_genes(a, v3):
    """Panel genes a gene set may use: all of them (v2), or those detected in at least MIN_DETECT of the cells (v3)."""
    if not v3:
        return set(a.var_names)
    detected = np.asarray((a.X > 0).sum(axis=0)).ravel() / a.n_obs
    return set(a.var_names[detected >= MIN_DETECT])


def gene_sets(present, sets):
    """(sets restricted to the usable genes, dropping those with fewer than MIN_SET_GENES genes; genes left out)."""
    present = set(present)
    used = {k: [g for g in v if g in present] for k, v in sets.items()}
    return ({k: v for k, v in used.items() if len(v) >= MIN_SET_GENES},
            {k: [g for g in v if g not in present] for k, v in sets.items()})


def score(a, genes, name):
    sc.tl.score_genes(a, genes, score_name=name, ctrl_size=50, n_bins=25, random_state=SEED, use_raw=False)
    return a.obs.pop(name).to_numpy()


def leiden(a, resolution, key):
    sc.pp.neighbors(a, n_neighbors=N_NEIGHBORS, use_rep='X_pca', random_state=SEED)
    sc.tl.leiden(a, resolution=resolution, key_added=key, flavor='igraph', n_iterations=2, directed=False,
                 random_state=SEED)
    return a.obs[key].astype(str).to_numpy()


def celltypist_lineage(a, model_path, mapping):
    """(label, confidence, lineage) per cell; the model is applied in chunks of CHUNK cells."""
    import celltypist
    model = celltypist.models.Model.load(str(model_path))
    unmapped = sorted(set(model.cell_types) - set(mapping))
    if unmapped:
        raise ValueError(f'{model_path.name}: labels without a lineage: {unmapped}')
    labels, conf = [], []
    for start in range(0, a.n_obs, CHUNK):
        chunk = ad.AnnData(X=a.X[start:start + CHUNK], obs=pd.DataFrame(index=a.obs_names[start:start + CHUNK]),
                           var=pd.DataFrame(index=a.var_names))
        res = celltypist.annotate(chunk, model=model, majority_voting=False)
        labels.append(res.predicted_labels['predicted_labels'].astype(str).to_numpy())
        conf.append(res.probability_matrix.max(axis=1).to_numpy())
        log(f'celltypist {min(start + CHUNK, a.n_obs)}/{a.n_obs}')
    labels = np.concatenate(labels)
    return labels, np.concatenate(conf), np.array([mapping[x] for x in labels], dtype=object)


def combine(cluster, cell_scores, ct_lineage, model_lineages, min_majority=0.0):
    """(lineage per cell, evidence per cell, per-cluster table) from the marker and CellTypist opinions."""
    cell_call = cell_scores.idxmax(axis=1).to_numpy(dtype=object)
    means = cell_scores.groupby(cluster).mean()
    table = means.add_prefix('score_')
    table.insert(0, 'n', pd.Series(cluster).value_counts())
    table['marker_call'] = means.idxmax(axis=1)
    ranked = np.sort(means.to_numpy(), axis=1)
    table['marker_top'], table['marker_margin'] = ranked[:, -1], ranked[:, -1] - ranked[:, -2]
    table['celltypist_call'], table['celltypist_frac'] = None, np.nan
    if ct_lineage is not None:
        ok = np.array([x is not None for x in ct_lineage])
        votes = pd.crosstab(pd.Series(cluster[ok], name='cluster'), pd.Series(ct_lineage[ok], name='lineage'))
        table['celltypist_call'] = votes.idxmax(axis=1)
        table['celltypist_frac'] = votes.max(axis=1) / votes.sum(axis=1)

    def decide(r):
        has_ct = ct_lineage is not None and not pd.isna(r.celltypist_call)
        if r.marker_top <= 0:
            return ('celltypist_only', r.celltypist_call) if has_ct else ('no_evidence', 'Others')
        if not has_ct or r.marker_call not in model_lineages:
            return 'marker_only', r.marker_call
        if r.celltypist_frac < min_majority:
            return 'ct_no_majority', r.marker_call
        if r.celltypist_call == r.marker_call:
            return 'agree', r.marker_call
        return ('marker_strong', r.marker_call) if r.marker_margin >= MARGIN else ('conflict', r.marker_call)

    table['evidence'], table['call'] = zip(*[decide(r) for r in table.itertuples()])
    lineage = table['call'].reindex(cluster).to_numpy(dtype=object)
    evidence = table['evidence'].reindex(cluster).to_numpy(dtype=object)
    conflict = evidence == 'conflict'
    both = conflict & (cell_call == ct_lineage)
    lineage[conflict] = np.where(both[conflict], cell_call[conflict], 'Others')
    evidence[conflict] = np.where(both[conflict], 'conflict_cell_agree', 'unresolved')
    return lineage, evidence, cell_call, table


def call_tumor(a, lineage, cfg, present, cluster=None):
    """(tumor mask, subcluster per cell, subcluster table, record, tumor score, normal score).

    With `cluster` (v3) the background is at least the highest mean tumor score of a background cluster."""
    parent = lineage == cfg['parent']
    t_genes = [g for g in cfg['tumor'] if g in present]
    n_genes = [g for g in cfg['normal'] if g in present]
    t = score(a, t_genes, 'tumor_score')
    n = score(a, n_genes, 'normal_score') if len(n_genes) >= MIN_SET_GENES else None
    background = np.isin(lineage, markers.MAIN) & ~parent
    tau_cell = tau = float(np.quantile(t[background], TAU_Q))
    if cluster is not None:
        means = pd.Series(t[background]).groupby(cluster[background]).agg(['mean', 'size'])
        means = means[means['size'] >= MIN_BG_CLUSTER]
        if len(means):
            tau = max(tau, float(means['mean'].max()))
    tumor, sub, rows = np.zeros(a.n_obs, bool), np.full(a.n_obs, '', dtype=object), []
    if parent.sum() > 2 * N_NEIGHBORS:
        p = ad.AnnData(obs=pd.DataFrame(index=a.obs_names[parent]), obsm={'X_pca': a.obsm['X_pca'][parent]})
        s = leiden(p, TUMOR_RESOLUTION, 'sub')
        sub[parent] = s
        idx = np.flatnonzero(parent)
        for k in sorted(set(s), key=int):
            m = s == k
            mt = float(t[idx[m]].mean())
            mn = None if n is None else float(n[idx[m]].mean())
            hit = mt > tau and (mn is None or mt > mn)
            tumor[idx[m]] = hit
            rows.append(dict(subcluster=k, n=int(m.sum()), tumor_score=mt, normal_score=mn, tumor=hit))
    record = dict(parent=cfg['parent'], tumor_genes=t_genes, normal_genes=n_genes,
                  missing=[g for g in cfg['tumor'] + cfg['normal'] if g not in present], tau=tau, tau_cell=tau_cell,
                  n_parent=int(parent.sum()), n_tumor=int(tumor.sum()))
    return tumor, sub, pd.DataFrame(rows), record, t, n


def _apply_nature_style():
    logging.getLogger('matplotlib.font_manager').setLevel(logging.ERROR)
    plt.rcParams.update({
        'font.family': 'sans-serif', 'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'], 'font.size': 7,
        'axes.titlesize': 8, 'axes.labelsize': 7, 'xtick.labelsize': 6, 'ytick.labelsize': 6, 'legend.fontsize': 6,
        'axes.spines.top': False, 'axes.spines.right': False, 'axes.linewidth': 0.5, 'xtick.major.width': 0.5,
        'ytick.major.width': 0.5, 'lines.linewidth': 1, 'savefig.dpi': 300})


def plot_spatial(xy, cls, title, path):
    _apply_nature_style()
    span = np.ptp(xy, axis=0)
    fig, ax = plt.subplots(figsize=(7.2, min(9.0, 7.2 * span[1] / span[0] + 0.4)))
    for c in ('Others', 'Stromal', 'Endothelial', 'Immune', 'Epithelial', 'Tumor'):
        m = cls == c
        ax.scatter(xy[m, 0], xy[m, 1], s=0.02, c=_CLASS_COLORS[c], linewidths=0, rasterized=True,
                   label=f'{c} ({m.sum():,})')
    ax.set_aspect('equal')
    ax.invert_yaxis()
    ax.set_axis_off()
    ax.set_title(title)
    ax.legend(loc='upper left', bbox_to_anchor=(1.0, 1.0), frameon=False, markerscale=40)
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def plot_scores(table, title, path):
    """Mean score per final class (rows), each column z-scored across classes."""
    _apply_nature_style()
    z = (table - table.mean()) / table.std(ddof=0).replace(0, 1)
    fig, ax = plt.subplots(figsize=(3.5, 0.3 * len(table) + 1.2))
    im = ax.imshow(z.to_numpy(), cmap='RdBu_r', vmin=-2, vmax=2, aspect='auto')
    ax.set_xticks(range(z.shape[1]), z.columns, rotation=45, ha='right')
    ax.set_yticks(range(z.shape[0]), [f'{c} ({n:,})' for c, n in zip(z.index, table.attrs['n'])])
    ax.tick_params(length=0)
    ax.set_title(title)
    fig.colorbar(im, ax=ax, shrink=0.8, label='column z-score')
    fig.savefig(path, dpi=300, bbox_inches='tight')
    plt.close(fig)


def main(organ, h5ad, parquet, model_dir, out):
    out = Path(out)
    out.mkdir(parents=True, exist_ok=False)
    adata = ad.read_h5ad(h5ad)
    cell_id = adata.obs['cell_id'].astype(str).to_numpy()
    ref = pd.read_parquet(parquet, columns=['cell_id'])['cell_id']
    ref = {v.decode() if isinstance(v, bytes) else str(v) for v in ref}
    if len(set(cell_id)) != len(cell_id) or set(cell_id) != ref:
        raise ValueError(f'{organ}: h5ad cell_id does not match {parquet} ({len(cell_id)} vs {len(ref)} cells)')
    counts = np.asarray(adata.X.sum(axis=1)).ravel()
    if not np.allclose(counts, np.round(counts)):
        raise ValueError(f'{organ}: X is not raw counts')
    keep = counts >= MIN_COUNTS
    xy = np.asarray(adata.obsm['spatial'], dtype=float)
    a = ad.AnnData(X=adata.X[keep].astype(np.float32), obs=pd.DataFrame(index=cell_id[keep]),
                   var=pd.DataFrame(index=adata.var_names.astype(str)))
    del adata
    log(f'{organ}: {len(cell_id)} cells, {int((~keep).sum())} below {MIN_COUNTS} counts, {a.n_vars} genes')

    sc.pp.normalize_total(a, target_sum=1e4)
    sc.pp.log1p(a)
    sc.pp.pca(a, n_comps=N_PCS, svd_solver='arpack', random_state=SEED)
    cluster = leiden(a, RESOLUTION, 'cluster')
    log(f'leiden: {len(set(cluster))} clusters')

    v3 = markers.METHOD[organ] == 'v3'
    present = usable_genes(a, v3)
    sets, missing = gene_sets(present, markers.LINEAGE_GENES)
    sets = {k: v for k, v in sets.items() if k not in markers.ABSENT.get(organ, ())}
    cell_scores = pd.DataFrame({k: score(a, v, f'lineage_{k}') for k, v in sets.items()}, index=a.obs_names)
    model_name = markers.CELLTYPIST_MODEL[organ]
    labels = conf = ct_lineage = None
    model_lineages = set()
    if model_name:
        mapping = markers.CELLTYPIST_LINEAGE[model_name]
        labels, conf, ct_lineage = celltypist_lineage(a, Path(model_dir) / model_name, mapping)
        model_lineages = {v for v in mapping.values() if v}
    lineage, evidence, cell_call, clusters = combine(cluster, cell_scores, ct_lineage, model_lineages,
                                                     MIN_MAJORITY if v3 else 0.0)
    log('lineage: ' + json.dumps(pd.Series(lineage).value_counts().to_dict()))

    cfg = markers.TUMOR[organ]
    tumor, sub, subtable = np.zeros(a.n_obs, bool), np.full(a.n_obs, '', dtype=object), pd.DataFrame()
    t_score = n_score = None
    tumor_record = None
    if cfg:
        tumor, sub, subtable, tumor_record, t_score, n_score = call_tumor(a, lineage, cfg, present,
                                                                          cluster if v3 else None)
        log('tumor: ' + json.dumps({k: v for k, v in tumor_record.items() if k not in ('tumor_genes', 'normal_genes')}))
    cls = np.where(np.isin(lineage, markers.MAIN), lineage, 'Others').astype(object)
    cls[tumor] = 'Tumor'

    def full(values, fill):
        col = np.full(len(cell_id), fill, dtype=object if isinstance(fill, str) else float)
        if values is not None:
            col[keep] = values
        return col

    cells = pd.DataFrame({
        'cell_id': cell_id, 'class': full(cls, 'Others'), 'lineage': full(lineage, 'low_counts'),
        'evidence': full(evidence, 'low_counts'), 'cluster': full(cluster, ''), 'marker_cell_call': full(cell_call, ''),
        'celltypist_label': full(labels, ''), 'celltypist_conf': full(conf, np.nan),
        'celltypist_lineage': full(None if ct_lineage is None else np.array(
            ['' if x is None else x for x in ct_lineage], dtype=object), ''),
        'tumor_subcluster': full(sub, ''), 'tumor_score': full(t_score, np.nan), 'normal_score': full(n_score, np.nan),
        'total_counts': counts})
    if set(cells['class']) - set(markers.CLASSES):
        raise AssertionError(f'unexpected classes {set(cells["class"]) - set(markers.CLASSES)}')
    cells[['cell_id', 'class']].to_csv(out / 'class.csv.gz', index=False, compression=GZIP)
    cells.to_csv(out / 'cells.csv.gz', index=False, compression=GZIP, float_format='%.4f')
    clusters.to_csv(out / 'clusters.csv', float_format='%.4f')
    subtable.to_csv(out / 'tumor_subclusters.csv', index=False, float_format='%.4f')

    kept = cells[keep]
    score_table = pd.concat([cell_scores.set_axis(kept.index), kept[['tumor_score', 'normal_score']].dropna(axis=1, how='all')],
                            axis=1).groupby(kept['class'].to_numpy()).mean()
    score_table = score_table.reindex([c for c in markers.CLASSES if c in score_table.index])
    score_table.attrs['n'] = [int((kept['class'] == c).sum()) for c in score_table.index]
    plot_spatial(xy, cells['class'].to_numpy(), organ, out / 'spatial_class.png')
    plot_scores(score_table, organ, out / 'class_scores.png')

    import celltypist
    summary = dict(
        organ=organ, method=markers.METHOD[organ], n_cells=len(cell_id), n_low_counts=int((~keep).sum()),
        classes=cells['class'].value_counts().to_dict(), lineage=cells['lineage'].value_counts().to_dict(),
        evidence=cells['evidence'].value_counts().to_dict(),
        cluster_evidence=clusters['evidence'].value_counts().to_dict(), n_clusters=len(clusters),
        inputs=dict(h5ad=str(h5ad), h5ad_sha256=digest(h5ad), cells_parquet=str(parquet),
                    cells_parquet_sha256=digest(parquet), celltypist_model=model_name,
                    celltypist_model_sha256=digest(Path(model_dir) / model_name) if model_name else None),
        lineage_genes=sets, absent_lineages=markers.ABSENT.get(organ, []), lineage_genes_missing={k: v for k, v in missing.items() if v}, tumor=tumor_record,
        params=dict(min_counts=MIN_COUNTS, n_pcs=N_PCS, n_neighbors=N_NEIGHBORS, resolution=RESOLUTION,
                    tumor_resolution=TUMOR_RESOLUTION, tau_quantile=TAU_Q, margin=MARGIN, min_set_genes=MIN_SET_GENES, seed=SEED,
                    **(dict(min_detect=MIN_DETECT, min_majority=MIN_MAJORITY, min_bg_cluster=MIN_BG_CLUSTER) if v3 else {})),
        versions=dict(python=sys.version.split()[0], scanpy=sc.__version__, anndata=ad.__version__,
                      numpy=np.__version__, pandas=pd.__version__, celltypist=celltypist.__version__),
        script_sha256=digest(__file__), markers_sha256=digest(markers.__file__),
        class_csv_sha256=digest(out / 'class.csv.gz'), seconds=round(time.time() - T0))
    (out / 'summary.json').write_text(json.dumps(summary, indent=1))
    log('classes: ' + json.dumps(summary['classes']))


if __name__ == '__main__':
    if len(sys.argv) != 6:
        sys.exit(__doc__)
    main(*sys.argv[1:])

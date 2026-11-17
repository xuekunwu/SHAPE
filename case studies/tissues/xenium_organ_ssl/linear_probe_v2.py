"""Linear probe of the arms' SSL embeddings on a cell-label column (PROBE_LABEL, default celltype_final) of a Xenium section.

v2 of the frozen lung probe (sha256 9252001f...): the only change is the label column, read from the environment
variable PROBE_LABEL and written as the label column of predictions.csv.

Protocol: rows L2-normalised; StandardScaler fitted on the development_train cells; multinomial logistic
regression fitted on development_train; C from {0.01, 0.1, 1, 10} by balanced accuracy on development_val (ties:
smaller C), no refit on validation; scored once on the held-out tiles (evaluation_only), which no SSL call saw.
Classes with fewer than MIN_TRAIN training cells are dropped from every split. Intervals: 1000 bootstrap draws
over held-out tiles (seed 20261026), the same draws for every arm, so arm differences are paired. A chance group
(random classes from the training-class frequencies) is scored on the same draws and compared with every arm.
Sanity checks: the training-majority class, and the first arm refitted on shuffled training labels.

usage: python linear_probe.py PREP LABELS_CSV_GZ OUT NAME=RUN [NAME=RUN ...]
"""
import csv
import gzip
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import balanced_accuracy_score, confusion_matrix, f1_score
from sklearn.preprocessing import StandardScaler

C_GRID = (0.01, 0.1, 1.0, 10.0)
MIN_TRAIN = 10
DRAWS, SEED = 1000, 20261026
USAGES = ('development_train', 'development_val', 'evaluation_only')
LABEL = os.environ.get('PROBE_LABEL', 'celltype_final')


def digest(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1 << 20), b''):
            h.update(block)
    return h.hexdigest()


def metrics(y, p, labels):
    return dict(accuracy=float(np.mean(y == p)), balanced_accuracy=float(balanced_accuracy_score(y, p)),
                macro_f1=float(f1_score(y, p, labels=labels, average='macro', zero_division=0)))


def fit_probe(x_train, y_train, x_val, y_val):
    """(selected C, fitted model, val balanced accuracy per C)."""
    scores, models = {}, {}
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=3000)
        model.fit(x_train, y_train)
        scores[c] = float(balanced_accuracy_score(y_val, model.predict(x_val)))
        models[c] = model
    best = max(scores.values())
    chosen = min(c for c in C_GRID if scores[c] == best)
    return chosen, models[chosen], scores


def main(prep, labels_path, out, runs):
    prep, out = Path(prep), Path(out)
    out.mkdir(parents=True, exist_ok=False)
    record = json.loads((prep / 'PREPARE.json').read_text())
    if digest(prep / 'public' / 'images.json') != record['images_sha256'] or \
            digest(prep / 'private' / 'cells.json') != record['private_cells_sha256']:
        raise ValueError('preparation changed')
    usage = {r['image_id']: r['usage'] for r in json.loads((prep / 'public' / 'images.json').read_text())}
    tile = {k: v['tile'] for k, v in json.loads((prep / 'private' / 'cells.json').read_text()).items()}
    with gzip.open(labels_path, 'rt', encoding='utf-8') as f:
        label = {r['cell_id']: r[LABEL] for r in csv.DictReader(f)}

    order, embeddings = None, {}
    for name, run in runs:
        run = Path(run)
        ids = [c['cell_id'] for c in json.loads((run / 'cells.json').read_text())]
        x = np.load(run / 'embedding.npy')
        if x.shape != (len(ids), 768) or not np.isfinite(x).all():
            raise ValueError(f'{name}: embedding {x.shape} does not match {len(ids)} cells or is not finite')
        if order is None:
            order = ids
        elif ids != order:
            raise ValueError(f'{name}: cell order differs from {runs[0][0]}')
        embeddings[name] = x / np.linalg.norm(x, axis=1, keepdims=True)
    missing = [c for c in order if c not in label]
    if missing or set(order) != set(usage):
        raise ValueError(f'{len(missing)} cells without a label, or the runs do not cover the preparation')

    y_all = np.array([label[c] for c in order])
    u_all = np.array([usage[c] for c in order])
    t_all = np.array([tile[c] for c in order])
    train_counts = {k: int(v) for k, v in zip(*np.unique(y_all[u_all == 'development_train'], return_counts=True))}
    classes = sorted(k for k, v in train_counts.items() if v >= MIN_TRAIN)
    dropped = {k: {u: int(np.sum((y_all == k) & (u_all == u))) for u in USAGES} for k in sorted(set(y_all) - set(classes))}
    keep = np.isin(y_all, classes)
    split = {u: np.flatnonzero(keep & (u_all == u)) for u in USAGES}
    counts = {u: {k: int(np.sum(y_all[idx] == k)) for k in classes} for u, idx in split.items()}
    tr, va, te = split['development_train'], split['development_val'], split['evaluation_only']
    y_tr, y_va, y_te = y_all[tr], y_all[va], y_all[te]

    rng = np.random.default_rng(SEED)
    test_tiles = np.unique(t_all[te])
    members = {t: np.flatnonzero(t_all[te] == t) for t in test_tiles}
    draws = [np.concatenate([members[t] for t in rng.choice(test_tiles, size=len(test_tiles), replace=True)])
             for _ in range(DRAWS)]

    def boot(pred):
        values = {k: [] for k in ('accuracy', 'balanced_accuracy', 'macro_f1')}
        for idx in draws:
            for k, v in metrics(y_te[idx], pred[idx], classes).items():
                values[k].append(v)
        return {k: np.array(v) for k, v in values.items()}

    results, preds, samples = {}, {}, {}

    def record(name, pred, **extra):
        preds[name] = pred
        samples[name] = boot(pred)
        f1 = f1_score(y_te, pred, labels=classes, average=None, zero_division=0)
        results[name] = dict(**extra, test=metrics(y_te, pred, classes),
                             test_ci95={k: [float(np.percentile(v, 2.5)), float(np.percentile(v, 97.5))]
                                        for k, v in samples[name].items()},
                             per_class_f1={k: float(v) for k, v in zip(classes, f1)})
        np.savetxt(out / f'confusion_{name}.csv', confusion_matrix(y_te, pred, labels=classes), fmt='%d',
                   delimiter=',', header=','.join(classes), comments='')
        print(name, json.dumps(extra.get('C')), json.dumps(results[name]['test']), flush=True)

    for name, x in embeddings.items():
        scaler = StandardScaler().fit(x[tr])
        c, model, val_scores = fit_probe(scaler.transform(x[tr]), y_tr, scaler.transform(x[va]), y_va)
        record(name, model.predict(scaler.transform(x[te])), C=c, val_balanced_accuracy=val_scores)
    # The chance group: each held-out cell gets a class drawn from the training-class frequencies (own seed), scored
    # on the same bootstrap draws as the arms; expected balanced accuracy 1 / K, expected accuracy sum_k p_train q_test.
    prior = np.array([np.sum(y_tr == k) for k in classes], dtype=float) / len(y_tr)
    record('chance', np.random.default_rng(SEED + 1).choice(np.array(classes), size=len(te), p=prior),
           rule='random class from the training-class frequencies',
           expected=dict(balanced_accuracy=1 / len(classes),
                         accuracy=float(sum(p * np.mean(y_te == k) for k, p in zip(classes, prior)))))

    majority = max(train_counts, key=lambda k: (train_counts[k], k))
    sanity = dict(majority_class=majority, majority=metrics(y_te, np.full(len(y_te), majority), classes))
    first = runs[0][0]
    shuffled = rng.permutation(y_tr)
    scaler = StandardScaler().fit(embeddings[first][tr])
    model = LogisticRegression(C=results[first]["C"], max_iter=3000).fit(
        scaler.transform(embeddings[first][tr]), shuffled)
    sanity['shuffled_labels_' + first] = metrics(y_te, model.predict(scaler.transform(embeddings[first][te])), classes)
    names = ['chance'] + list(embeddings)          # pairs read arm - chance and later arm - earlier arm
    paired = {}
    for i, a in enumerate(names):
        for b in names[i + 1:]:
            paired[f'{b}-{a}'] = {k: dict(delta=float(results[b]['test'][k] - results[a]['test'][k]),
                                          ci95=[float(np.percentile(samples[b][k] - samples[a][k], 2.5)),
                                                float(np.percentile(samples[b][k] - samples[a][k], 97.5))])
                                  for k in samples[a]}
    summary = dict(protocol=__doc__.strip().splitlines()[0], c_grid=C_GRID, min_train=MIN_TRAIN, draws=DRAWS, seed=SEED,
                   labels=str(labels_path), label_column=LABEL, labels_sha256=digest(labels_path), prep=str(prep),
                   prepare_sha256=digest(prep / 'PREPARE.json'), classes=classes, dropped_classes=dropped,
                   counts=counts, test_tiles=int(len(test_tiles)),
                   runs={n: dict(path=str(r), embedding_sha256=digest(Path(r) / 'embedding.npy')) for n, r in runs},
                   arms=results, paired_differences=paired, sanity=sanity)
    (out / 'metrics.json').write_text(json.dumps(summary, indent=2) + '\n')
    with open(out / 'predictions.csv', 'w', newline='') as f:
        w = csv.writer(f)
        w.writerow(['cell_id', 'tile', LABEL, *[f'pred_{n}' for n in names]])
        for j, i in enumerate(te):
            w.writerow([order[i], t_all[i], y_all[i], *[preds[n][j] for n in names]])
    print('sanity', json.dumps(sanity))
    print('paired', json.dumps({k: {m: round(v['delta'], 4) for m, v in d.items()} for k, d in paired.items()}))


if __name__ == '__main__':
    main(sys.argv[1], sys.argv[2], sys.argv[3], [tuple(a.split('=', 1)) for a in sys.argv[4:]])

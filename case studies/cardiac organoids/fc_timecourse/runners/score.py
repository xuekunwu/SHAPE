"""Trusted read-out and scoring: supervised probes per arm on its own frozen embeddings (development wells fit
and select; held-out wells are scored once), independent GFP read-outs inside the arm's own masks, segmentation
descriptives and stage costs. Primary read-out: the culture time point (eight-class day classifier; a ridge
regression of the day alongside). Secondary: the condition (Control vs FC). Everything is descriptive; intervals
are bootstrap over held-out wells.

With the cross-validation protocol (`data.CV_PROTOCOL`) there are no held-out wells: the read-outs are scored by
leave-one-plate-column-out cross-validation (for each outer fold the next fold validates, the other two fit, the
outer fold is scored once, so every image is scored out of fold) and the bootstrap is over the wells, out of fold."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
from pathlib import Path

import numpy as np
import pandas as pd
import tifffile

from . import data

C_GRID = (0.01, 0.1, 1.0, 10.0)
ALPHA_GRID = (0.1, 1.0, 10.0, 100.0)
THRESHOLD = 0.5
BOOTSTRAP_DRAWS = 1000
POSITIVE = 'FC'
DAYS = sorted(set(data.DAYS.values()))          # the eight time points, the classes of the primary read-out


def _split_indices(cells):
    usage = np.array([c['usage'] for c in cells])
    train, val = np.flatnonzero(usage == 'development_train'), np.flatnonzero(usage == 'development_val')
    if not len(train) or not len(val):
        raise ValueError('the probes need development_train and development_val crops')
    if {cells[i]['well'] for i in train} & {cells[i]['well'] for i in val}:
        raise ValueError('well leakage between train and val')
    if any(c['usage'] == 'evaluation_only' for i in np.concatenate([train, val]) for c in [cells[i]]):
        raise ValueError('held-out crops in a fitting split')
    return train, val


def _well_weights(cells, train):
    counts = Counter(cells[i]['well'] for i in train)
    weights = np.array([1 / counts[cells[i]['well']] for i in train])
    return weights * len(weights) / weights.sum(), sorted(counts)


def fit_day_classifier(x, cells, labels, seed):
    """The primary probe: multinomial logistic regression over the eight time points, scaler and model fitted on
    development_train crops (equal total weight per training well), C chosen by development_val balanced accuracy
    (ties: the smaller C), no refit on validation. Returns the class-probability matrix (crops x DAYS)."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import balanced_accuracy_score
    from sklearn.preprocessing import StandardScaler
    train, val = _split_indices(cells)
    y = np.array([DAYS.index(int(labels[c['image_id']]['day'])) for c in cells])
    if len(set(y[train].tolist())) < 2:
        raise ValueError('training crops cover fewer than two time points')
    weights, wells = _well_weights(cells, train)
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    trials, best = [], None
    for c in C_GRID:
        model = LogisticRegression(C=c, max_iter=3000, random_state=seed).fit(xt, y[train], sample_weight=weights)
        pred = model.predict(xv)
        quality = dict(accuracy=float(np.mean(pred == y[val])), balanced_accuracy=float(balanced_accuracy_score(y[val], pred)),
                       n=int(len(val)))
        trials.append(dict(C=c, validation=quality))
        if best is None or quality['balanced_accuracy'] > best[0]:
            best = (quality['balanced_accuracy'], c, model)
    model = best[2]
    full = np.zeros((len(cells), len(DAYS)))
    full[:, [int(k) for k in model.classes_]] = model.predict_proba(scaler.transform(x))
    record = dict(target='time point (day), eight classes', classes=DAYS, C=best[1], trials=trials,
                  selection='development_val balanced accuracy', fit_split='development_train',
                  selection_split='development_val', refit_on_validation=False, train_crops=int(len(train)),
                  validation_crops=int(len(val)), training_wells=wells,
                  training_weighting='equal total weight per training well',
                  features='row L2 norm, train-fitted standard scaling', classes_seen_in_training=[DAYS[int(k)] for k in model.classes_])
    return full, record


def _features(x: np.ndarray) -> np.ndarray:
    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or not np.isfinite(x).all():
        raise ValueError('invalid embedding matrix')
    norm = np.linalg.norm(x, axis=1, keepdims=True)
    return x / np.where(norm > 0, norm, 1.0)


def _binary(labels, probabilities, threshold=THRESHOLD):
    from sklearn.metrics import roc_auc_score
    y = np.asarray(labels, dtype=int)
    p = np.asarray(probabilities, dtype=float)
    if not len(y):
        return dict(n=0)
    pred = p >= threshold
    tp = int(np.sum(pred & (y == 1)))
    tn = int(np.sum(~pred & (y == 0)))
    fp = int(np.sum(pred & (y == 0)))
    fn = int(np.sum(~pred & (y == 1)))
    sens = tp / (tp + fn) if tp + fn else None
    spec = tn / (tn + fp) if tn + fp else None
    both = len(np.unique(y)) == 2
    return dict(n=int(len(y)), positives=int(y.sum()), accuracy=(tp + tn) / len(y),
                balanced_accuracy=(sens + spec) / 2 if both else None,
                auroc=float(roc_auc_score(y, p)) if both else None, sensitivity=sens, specificity=spec,
                confusion=dict(tp=tp, tn=tn, fp=fp, fn=fn))


def fit_condition_probe(x, cells, labels, seed):
    """Logistic regression on development crops: scaler and model fitted on development_train, C chosen by
    development_val balanced accuracy (ties: the smaller C), threshold 0.5, no refit on validation."""
    from sklearn.linear_model import LogisticRegression
    from sklearn.preprocessing import StandardScaler
    usage = np.array([c['usage'] for c in cells])
    y = np.array([int(labels[c['image_id']]['condition'] == POSITIVE) for c in cells])
    train, val = np.flatnonzero(usage == 'development_train'), np.flatnonzero(usage == 'development_val')
    if not len(train) or not len(val):
        raise ValueError('the probe needs development_train and development_val crops')
    if {cells[i]['well'] for i in train} & {cells[i]['well'] for i in val}:
        raise ValueError('well leakage between train and val')
    if set(y[train].tolist()) != {0, 1}:
        raise ValueError('training crops lack one condition')
    counts = Counter(cells[i]['well'] for i in train)
    weights = np.array([1 / counts[cells[i]['well']] for i in train])
    weights *= len(weights) / weights.sum()
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    trials, best = [], None
    grid = C_GRID if set(y[val].tolist()) == {0, 1} else (1.0,)
    for c in grid:
        model = LogisticRegression(C=c, max_iter=2000, random_state=seed).fit(xt, y[train], sample_weight=weights)
        quality = _binary(y[val], model.predict_proba(xv)[:, 1])
        trials.append(dict(C=c, validation=quality))
        value = quality['balanced_accuracy'] if quality['balanced_accuracy'] is not None else quality['accuracy']
        if best is None or value > best[0]:
            best = (value, c, model)
    probability = best[2].predict_proba(scaler.transform(x))[:, 1]
    params = dict(mean=scaler.mean_, scale=scaler.scale_, coef=best[2].coef_, intercept=best[2].intercept_)
    record = dict(target=f'condition == {POSITIVE}', C=best[1], trials=trials, classifier_probability_threshold=THRESHOLD,
                  selection='development_val balanced accuracy' if len(grid) > 1 else 'fixed C=1; single-class validation',
                  fit_split='development_train', selection_split='development_val', refit_on_validation=False,
                  train_crops=int(len(train)), validation_crops=int(len(val)), training_wells=sorted(counts),
                  training_weighting='equal total weight per training well', features='row L2 norm, train-fitted standard scaling')
    return probability, params, record


def fit_day_probe(x, cells, labels, seed):
    """Ridge regression for the culture day on the same features; alpha by development_val MAE."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    usage = np.array([c['usage'] for c in cells])
    day = np.array([float(labels[c['image_id']]['day']) for c in cells])
    train, val = np.flatnonzero(usage == 'development_train'), np.flatnonzero(usage == 'development_val')
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    trials, best = [], None
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha, random_state=seed).fit(xt, day[train])
        mae = float(np.mean(np.abs(model.predict(xv) - day[val])))
        trials.append(dict(alpha=alpha, validation_mae_days=mae))
        if best is None or mae < best[0]:
            best = (mae, alpha, model)
    prediction = best[2].predict(scaler.transform(x))
    return prediction, dict(target='day', alpha=best[1], trials=trials, selection='development_val MAE',
                            fit_split='development_train', refit_on_validation=False)


GFP_TARGET = 'gfp_integrated_bgsub'      # sum of GFP inside the object minus its area times the mean GFP outside every object


def object_gfp(prep: Path, run: Path, cells: list[dict], labels: dict) -> dict:
    """Private GFP read-outs of every delivered object inside the arm's own mask, keyed by cell id: integrated
    background-subtracted GFP (the target; user definition 2026-10-25), mean GFP, the image's background (mean and
    median of the pixels outside every object) and the GFP-positive pixel fraction at the frozen threshold."""
    threshold = data.gfp_threshold(prep)
    masks = {m['image_id']: m for m in data.read_json(run / 'segmentation.json')}
    by_image = defaultdict(list)
    for c in cells:
        by_image[c['image_id']].append(c)
    out = {}
    for ident, own in by_image.items():
        if ident not in masks:
            raise ValueError(f'no segmentation record for {ident}')
        if data.digest(Path(masks[ident]['mask'])) != masks[ident]['sha256']:
            raise ValueError(f'mask changed before GFP read-out: {ident}')
        mask = tifffile.imread(masks[ident]['mask'])
        gfp = tifffile.imread(data.within(prep / 'private', labels[ident]['gfp_file'])).astype(float)
        if mask.shape != gfp.shape:
            raise ValueError(f'mask and GFP shapes differ: {ident}')
        outside = gfp[mask == 0]
        if not outside.size:
            raise ValueError(f'no background pixels outside the objects: {ident}')
        bg_mean, bg_median = float(outside.mean()), float(np.median(outside))
        for c in own:
            if c.get('mask_label') is None:
                raise ValueError('crops without mask labels cannot be read out in GFP')
            inside = mask == c['mask_label']
            if not inside.any():
                raise ValueError(f'mask label {c["mask_label"]} absent from the mask: {ident}')
            values = gfp[inside]
            area = int(inside.sum())
            out[c['cell_id']] = dict(area=area, gfp_sum=float(values.sum()), gfp_background_mean=bg_mean, gfp_background_median=bg_median,
                                     gfp_integrated_bgsub=float(values.sum() - area * bg_mean), gfp_mean=float(values.mean()),
                                     gfp_positive_fraction=float(np.mean(values >= threshold)))
    return out


def fit_gfp_probe(x, cells, target, seed):
    """Ridge regression of the object's background-subtracted integrated GFP on the same features; alpha by development_val
    MAE; the target is standardised on development_train (the prediction is returned in GFP units)."""
    from sklearn.linear_model import Ridge
    from sklearn.preprocessing import StandardScaler
    train, val = _split_indices(cells)
    y = np.asarray(target, dtype=float)
    scaler = StandardScaler().fit(x[train])
    xt, xv = scaler.transform(x[train]), scaler.transform(x[val])
    mu, sd = float(y[train].mean()), float(y[train].std()) or 1.0
    trials, best = [], None
    for alpha in ALPHA_GRID:
        model = Ridge(alpha=alpha, random_state=seed).fit(xt, (y[train] - mu) / sd)
        mae = float(np.mean(np.abs(model.predict(xv) * sd + mu - y[val])))
        trials.append(dict(alpha=alpha, validation_mae=mae))
        if best is None or mae < best[0]:
            best = (mae, alpha, model)
    prediction = best[2].predict(scaler.transform(x)) * sd + mu
    return prediction, dict(target=GFP_TARGET, alpha=best[1], trials=trials, selection='development_val MAE',
                            fit_split='development_train', refit_on_validation=False, target_standardisation=dict(mean=mu, sd=sd))


def fit_gfp_label_baseline(cells, labels, target):
    """Reference without the image: least squares of the target on one-hot day plus condition, fitted on development_train
    objects. It bounds how much of the GFP read-out the withheld labels alone explain."""
    train, _ = _split_indices(cells)
    y = np.asarray(target, dtype=float)
    design = np.array([[1.0] + [float(int(labels[c['image_id']]['day']) == d) for d in DAYS[1:]] + [float(labels[c['image_id']]['condition'] == POSITIVE)]
                       for c in cells])
    coef, *_ = np.linalg.lstsq(design[train], y[train], rcond=None)
    return design @ coef, dict(target=GFP_TARGET, design='intercept + one-hot day (reference day 0) + condition', fit_split='development_train',
                               n_parameters=int(len(coef)))


def _fold_cells(cells: list[dict], labels: dict, test_fold: int) -> list[dict]:
    """The cells with `usage` relabelled for one outer fold: the fold's wells are evaluation_only, the next fold
    (cyclic) validates, the remaining folds fit. The probes above are unchanged and read only this usage."""
    folds = sorted({int(labels[c['image_id']]['cv_fold']) for c in cells})
    if len(folds) < 3 or test_fold not in folds:
        raise ValueError(f'cross-validation needs at least three folds and the test fold among them: {folds}, {test_fold}')
    validation = folds[(folds.index(test_fold) + 1) % len(folds)]
    out = []
    for c in cells:
        fold = int(labels[c['image_id']]['cv_fold'])
        usage = data.EVALUATION if fold == test_fold else 'development_val' if fold == validation else 'development_train'
        out.append(dict(c, usage=usage))
    return out


def _predict_cv(prep: Path, run: Path, cells: list[dict], config: dict, labels: dict, x: np.ndarray) -> dict:
    """The cross-validation protocol's predict stage: every outer fold fits the probes on its own training wells and
    predicts its own cells, so every cell is predicted once, out of fold."""
    gfp = object_gfp(prep, run, cells, labels)
    target = np.array([gfp[c['cell_id']][GFP_TARGET] for c in cells])
    folds = sorted({int(labels[c['image_id']]['cv_fold']) for c in cells})
    day_probabilities = np.full((len(cells), len(DAYS)), np.nan)
    day_prediction, probability = np.full(len(cells), np.nan), np.full(len(cells), np.nan)
    gfp_prediction, gfp_baseline = np.full(len(cells), np.nan), np.full(len(cells), np.nan)
    fold_records, params_all = [], {}
    for fold in folds:
        fold_cells = _fold_cells(cells, labels, fold)
        test = np.flatnonzero([c['usage'] == data.EVALUATION for c in fold_cells])
        q, day_class_record = fit_day_classifier(x, fold_cells, labels, config['seed'])
        d, day_record = fit_day_probe(x, fold_cells, labels, config['seed'])
        p, params, condition_record = fit_condition_probe(x, fold_cells, labels, config['seed'])
        g, gfp_record = fit_gfp_probe(x, fold_cells, target, config['seed'])
        b, baseline_record = fit_gfp_label_baseline(fold_cells, labels, target)
        day_probabilities[test], day_prediction[test], probability[test] = q[test], d[test], p[test]
        gfp_prediction[test], gfp_baseline[test] = g[test], b[test]
        params_all.update({f'fold{fold}_{k}': v for k, v in params.items()})
        fold_records.append(dict(
            fold=fold, test_wells=sorted({c['well'] for c in fold_cells if c['usage'] == data.EVALUATION}),
            validation_wells=sorted({c['well'] for c in fold_cells if c['usage'] == 'development_val'}),
            training_wells=sorted({c['well'] for c in fold_cells if c['usage'] == 'development_train'}),
            day_classifier=day_class_record, day_regression=day_record, condition_probe=condition_record,
            gfp_probe=gfp_record, gfp_label_baseline=baseline_record))
    if np.isnan(day_probabilities).any() or np.isnan(probability).any() or np.isnan(gfp_prediction).any():
        raise ValueError('a cell was not scored out of fold')
    np.savez(run / 'classifier.npz', **params_all)
    record = dict(primary='day_classifier',
                  evaluation='leave-one-plate-column-out cross-validation; every image is scored once, out of fold',
                  folds=fold_records, classifier_weights_sha256=data.digest(run / 'classifier.npz'),
                  input_embedding_sha256=data.digest(run / 'embedding.npy'),
                  prediction_channels=data.SPEC['prediction_channels'])
    data.write_json(run / 'CLASSIFIER.json', record)
    fold_of = {c['cell_id']: int(labels[c['image_id']]['cv_fold']) for c in cells}
    predictions = [dict(cell_id=c['cell_id'], image_id=c['image_id'], well=c['well'], usage=data.EVALUATION,
                        cv_fold=fold_of[c['cell_id']], mask_label=c.get('mask_label'), area=c['area'],
                        day_class_probabilities=[float(v) for v in q], day_class_prediction=int(DAYS[int(np.argmax(q))]),
                        day_regression=float(d), fc_probability=float(p), fc_prediction=int(p >= THRESHOLD),
                        gfp_prediction=float(g), gfp_label_baseline=float(b))
                   for c, q, d, p, g, b in zip(cells, day_probabilities, day_prediction, probability, gfp_prediction, gfp_baseline)]
    data.write_json(run / 'predictions.json', predictions)
    data.write_json(run / 'PREDICTIONS_COMPLETE.json', dict(
        status='frozen', predictions_sha256=data.digest(run / 'predictions.json'),
        classifier_sha256=data.digest(run / 'CLASSIFIER.json'), embedding_sha256=data.digest(run / 'embedding.npy'),
        cells_sha256=data.digest(run / 'cells.json'), segmentation_sha256=data.digest(run / 'segmentation.json'),
        scored_out_of_fold=True, labels_of_a_fold_used_to_score_it=False))
    return record


def predict(prep: Path, run: Path) -> dict:
    prep, run = Path(prep), Path(run)
    cells = data.read_json(run / 'cells.json')
    config = data.read_json(run / 'CONFIG.json')
    labels = data.private_images(prep)
    if any(c['usage'] not in data.USAGES for c in cells):
        raise ValueError('unknown usage in cells')
    x = _features(np.load(run / 'embedding.npy'))
    if len(x) != len(cells):
        raise ValueError('embedding rows do not match cells')
    if data.protocol_of(prep) == data.CV_PROTOCOL:
        return _predict_cv(prep, run, cells, config, labels, x)
    day_probabilities, day_class_record = fit_day_classifier(x, cells, labels, config['seed'])
    day_prediction, day_record = fit_day_probe(x, cells, labels, config['seed'])
    probability, params, condition_record = fit_condition_probe(x, cells, labels, config['seed'])
    gfp = object_gfp(prep, run, cells, labels)
    target = np.array([gfp[c['cell_id']][GFP_TARGET] for c in cells])
    gfp_prediction, gfp_record = fit_gfp_probe(x, cells, target, config['seed'])
    gfp_baseline, baseline_record = fit_gfp_label_baseline(cells, labels, target)
    np.savez(run / 'classifier.npz', **params)
    record = dict(primary='day_classifier', day_classifier=day_class_record, day_regression=day_record,
                  condition_probe=dict(condition_record, weights_sha256=data.digest(run / 'classifier.npz')),
                  gfp_probe=gfp_record, gfp_label_baseline=baseline_record,
                  input_embedding_sha256=data.digest(run / 'embedding.npy'),
                  prediction_channels=data.SPEC['prediction_channels'])
    data.write_json(run / 'CLASSIFIER.json', record)
    predictions = [dict(cell_id=c['cell_id'], image_id=c['image_id'], well=c['well'], usage=c['usage'],
                        mask_label=c.get('mask_label'), area=c['area'],
                        day_class_probabilities=[float(v) for v in q], day_class_prediction=int(DAYS[int(np.argmax(q))]),
                        day_regression=float(d), fc_probability=float(p), fc_prediction=int(p >= THRESHOLD),
                        gfp_prediction=float(g), gfp_label_baseline=float(b))
                   for c, q, d, p, g, b in zip(cells, day_probabilities, day_prediction, probability, gfp_prediction, gfp_baseline)]
    data.write_json(run / 'predictions.json', predictions)
    data.write_json(run / 'PREDICTIONS_COMPLETE.json', dict(
        status='frozen', predictions_sha256=data.digest(run / 'predictions.json'),
        classifier_sha256=data.digest(run / 'CLASSIFIER.json'), embedding_sha256=data.digest(run / 'embedding.npy'),
        cells_sha256=data.digest(run / 'cells.json'), segmentation_sha256=data.digest(run / 'segmentation.json'),
        heldout_labels_used=False))
    return record


def _image_table(predictions: list[dict], labels: dict, rows: list[dict], gfp: dict | None = None) -> pd.DataFrame:
    """One row per selected image: labels, the arm's object count, the mean day-class probabilities of its objects
    (the image's day prediction is their argmax), the mean day regression, the mean FC probability and, when the
    GFP table is given, the object-sum of the integrated GFP target with the summed predictions."""
    by_image = defaultdict(list)
    for p in predictions:
        by_image[p['image_id']].append(p)
    table = []
    for r in rows:
        own = by_image.get(r['image_id'], [])
        lab = labels[r['image_id']]
        entry = dict(image_id=r['image_id'], well=r['well'], usage=r['usage'], condition=lab['condition'],
                     day=int(lab['day']), fc=int(lab['condition'] == POSITIVE), n_objects=len(own),
                     day_class_prediction=np.nan, day_regression=np.nan, fc_probability=np.nan, largest_object_area=np.nan,
                     gfp_integrated_bgsub=np.nan, gfp_prediction=np.nan, gfp_label_baseline=np.nan)
        if own:
            mean_q = np.mean([p['day_class_probabilities'] for p in own], axis=0)
            entry.update(day_class_prediction=int(DAYS[int(np.argmax(mean_q))]),
                         day_regression=float(np.mean([p['day_regression'] for p in own])),
                         fc_probability=float(np.mean([p['fc_probability'] for p in own])),
                         largest_object_area=float(max(p['area'] for p in own)))
            if gfp is not None:
                entry.update(gfp_integrated_bgsub=float(sum(gfp[p['cell_id']][GFP_TARGET] for p in own)),
                             gfp_prediction=float(sum(p['gfp_prediction'] for p in own)),
                             gfp_label_baseline=float(sum(p['gfp_label_baseline'] for p in own)))
            for d, v in zip(DAYS, mean_q):
                entry[f'p_day_{d}'] = float(v)
        table.append(entry)
    return pd.DataFrame(table)


def _day_metrics(truth, predicted) -> dict:
    """Exact and adjacent-time-point accuracy, balanced accuracy and the confusion matrix over DAYS."""
    from sklearn.metrics import balanced_accuracy_score
    t = np.asarray(truth, dtype=int)
    p = np.asarray(predicted, dtype=int)
    if not len(t):
        return dict(n=0)
    rank = {d: i for i, d in enumerate(DAYS)}
    tr, pr = np.array([rank[v] for v in t]), np.array([rank[v] for v in p])
    confusion = pd.crosstab(pd.Categorical(t, categories=DAYS), pd.Categorical(p, categories=DAYS), dropna=False)
    return dict(n=int(len(t)), accuracy=float(np.mean(tr == pr)), within_one_timepoint=float(np.mean(np.abs(tr - pr) <= 1)),
                balanced_accuracy=float(balanced_accuracy_score(tr, pr)), mean_abs_rank_error=float(np.mean(np.abs(tr - pr))),
                chance_accuracy=1 / len(DAYS), confusion={'rows_true_cols_predicted': DAYS, 'matrix': confusion.values.tolist()})


def _heldout_metrics(images: pd.DataFrame, crops: pd.DataFrame, seed: int, draws: int, cv: bool = False) -> dict:
    from scipy.stats import spearmanr
    held = images[images.usage == 'evaluation_only']
    scored = held.dropna(subset=['day_class_prediction'])
    out = dict(heldout_images=int(len(held)), heldout_images_with_objects=int(len(scored)),
               heldout_wells=int(held.well.nunique()), classes=DAYS)
    if cv:
        out['evaluation'] = 'out-of-fold cross-validation: the heldout_* keys count the cells scored out of fold (all images)'
    hc = crops[crops.usage == 'evaluation_only']
    # primary: the time point
    out['day_image_level'] = _day_metrics(scored.day, scored.day_class_prediction) if len(scored) else dict(n=0)
    out['day_crop_level'] = _day_metrics(hc.day, hc.day_class_prediction) if len(hc) else dict(n=0)
    out['day_image_level_by_condition'] = {c: _day_metrics(g.day, g.day_class_prediction) for c, g in scored.groupby('condition')} if len(scored) else {}
    out['day_image_accuracy_by_day'] = {int(d): float(np.mean(g.day_class_prediction == g.day)) for d, g in scored.groupby('day')} if len(scored) else {}
    if len(scored) >= 3 and scored.day.nunique() > 1:
        rho = spearmanr(scored.day, scored.day_regression)
        out['day_regression'] = dict(mae_days=float(np.mean(np.abs(scored.day_regression - scored.day))),
                                     spearman_rho=float(rho.statistic), spearman_p=float(rho.pvalue), n_images=int(len(scored)))
    else:
        out['day_regression'] = dict(mae_days=None, spearman_rho=None, n_images=int(len(scored)))
    # secondary: the condition
    out['condition_image_level'] = _binary(scored.fc, scored.fc_probability) if len(scored) else dict(n=0)
    out['condition_crop_level'] = _binary(hc.fc, hc.fc_probability) if len(hc) else dict(n=0)
    out['condition_accuracy_by_day'] = {int(d): float(np.mean((g.fc_probability >= THRESHOLD) == (g.fc == 1)))
                                        for d, g in scored.groupby('day')} if len(scored) else {}
    wells = scored.groupby('well').agg(fc=('fc', 'first'), probability=('fc_probability', 'mean'))
    out['condition_well_level'] = _binary(wells.fc, wells.probability) if len(wells) else dict(n=0)
    # GFP: how well the embedding predicts the integrated background-subtracted GFP of the held-out objects, against the
    # label-only reference (one-hot day + condition)
    out['gfp_image_level'] = _gfp_metrics(scored) if len(scored) >= 3 else dict(n=int(len(scored)))
    out['gfp_crop_level'] = _gfp_metrics(hc) if len(hc) >= 3 else dict(n=int(len(hc)))
    out['bootstrap'] = None
    if len(wells) >= 2 and draws > 0:
        rng = np.random.default_rng(seed)
        names = list(wells.index)
        stats = defaultdict(list)
        for _ in range(draws):
            drawn = rng.choice(names, size=len(names), replace=True)
            sample = pd.concat([scored[scored.well == w] for w in drawn])
            dm = _day_metrics(sample.day, sample.day_class_prediction)
            for key in ('accuracy', 'within_one_timepoint', 'balanced_accuracy'):
                stats['day_' + key].append(dm[key])
            stats['day_regression_mae'].append(float(np.mean(np.abs(sample.day_regression - sample.day))))
            cm = _binary(sample.fc, sample.fc_probability)
            for key in ('accuracy', 'auroc'):
                if cm.get(key) is not None:
                    stats['condition_' + key].append(cm[key])
            gm = _gfp_metrics(sample)
            for key in ('spearman_rho', 'mae', 'baseline_spearman_rho', 'baseline_mae'):
                if gm.get(key) is not None:
                    stats['gfp_' + key].append(gm[key])
        out['bootstrap'] = dict(unit='well (out of fold)' if cv else 'held-out well', draws=draws, seed=seed,
                                image_level={k: dict(low=float(np.percentile(v, 2.5)), high=float(np.percentile(v, 97.5)), n_valid_draws=len(v))
                                             for k, v in stats.items()})
    return out


def _gfp_metrics(frame: pd.DataFrame) -> dict:
    """Spearman and MAE of the GFP probe and of the label-only reference against the integrated GFP target."""
    from scipy.stats import spearmanr
    t = frame[GFP_TARGET].to_numpy(dtype=float)
    out = dict(n=int(len(t)), target=GFP_TARGET)
    if len(t) < 3 or np.ptp(t) == 0:
        return out
    for prefix, column in (('', 'gfp_prediction'), ('baseline_', 'gfp_label_baseline')):
        p = frame[column].to_numpy(dtype=float)
        rho = spearmanr(t, p)
        out[prefix + 'spearman_rho'] = float(rho.statistic)
        out[prefix + 'spearman_p'] = float(rho.pvalue)
        out[prefix + 'mae'] = float(np.mean(np.abs(p - t)))
    return out


def _gfp_readouts(prep: Path, run: Path, predictions: list[dict], labels: dict, rows: list[dict]) -> tuple[pd.DataFrame, dict, dict]:
    """The per-object GFP table (private channel, the arm's own masks) with the probe outputs, its per-condition-and-day
    summary, and the object read-outs keyed by cell id."""
    threshold = data.gfp_threshold(prep)
    gfp = object_gfp(prep, run, predictions, labels)
    meta = {r['image_id']: r for r in rows}
    records = []
    for p in predictions:
        r, g, ident = meta[p['image_id']], gfp[p['cell_id']], p['image_id']
        records.append(dict(cell_id=p['cell_id'], image_id=ident, well=r['well'], usage=r['usage'],
                            condition=labels[ident]['condition'], day=int(labels[ident]['day']), **g,
                            gfp_prediction=p['gfp_prediction'], gfp_label_baseline=p['gfp_label_baseline'], fc_probability=p['fc_probability']))
    table = pd.DataFrame(records)
    summary = dict(gfp_positive_threshold=threshold, target=GFP_TARGET,
                   target_definition='sum of GFP inside the object minus area x mean GFP of the pixels outside every object of the image',
                   by_condition_day={}, heldout_spearman=None)
    if len(table):
        largest = table.sort_values('area', ascending=False).drop_duplicates('image_id')
        for (condition, day), g in largest.groupby(['condition', 'day']):
            summary['by_condition_day'][f'{condition}|{int(day)}'] = dict(
                n_images=int(len(g)), gfp_integrated_bgsub=float(g[GFP_TARGET].mean()),
                gfp_integrated_bgsub_sem=float(g[GFP_TARGET].sem()) if len(g) > 1 else None,
                gfp_mean=float(g.gfp_mean.mean()), gfp_mean_sem=float(g.gfp_mean.sem()) if len(g) > 1 else None,
                gfp_background_mean=float(g.gfp_background_mean.mean()),
                gfp_positive_fraction=float(g.gfp_positive_fraction.mean()), largest_object_area=float(g.area.mean()))
        held = table[table.usage == 'evaluation_only']
        if len(held) >= 3:
            from scipy.stats import spearmanr
            rho = spearmanr(held.fc_probability, held[GFP_TARGET])
            rho_mean = spearmanr(held.fc_probability, held.gfp_mean)
            summary['heldout_spearman'] = dict(fc_probability_vs_gfp_integrated_bgsub=float(rho.statistic), p=float(rho.pvalue),
                                               fc_probability_vs_gfp_mean=float(rho_mean.statistic), p_gfp_mean=float(rho_mean.pvalue),
                                               n_objects=int(len(held)))
    return table, summary, gfp


def _costs(run: Path, arm: str) -> dict:
    out = {}
    for stage in ('segment', 'crop', 'embed'):
        entry = {}
        complete = run / f'{stage}_COMPLETE.json'
        if complete.is_file():
            entry['seconds'] = data.read_json(complete).get('seconds')
        usage = run / f'{stage}_USAGE.json'
        if usage.is_file():
            entry['total_tokens'] = data.read_json(usage).get('total_tokens')
        if arm == 'codex':
            selected = run / 'agent' / stage / 'SELECTED.json'
            if selected.is_file():
                s = data.read_json(selected)
                entry.update(n_calls=s.get('n_calls'), n_rounds=s.get('n_rounds'))
        elif arm == 'shape' and stage in ('segment', 'crop'):
            searches = list((run / stage).glob('*/SEARCH.json'))
            entry['n_searched_images'] = len(searches)
            entry['n_tool_calls'] = int(sum(data.read_json(s).get('n_calls', 0) for s in searches))
            cohort = list((run / stage).glob('*/COHORT_SEARCH.json')) + list((run / stage).glob('COHORT_SEARCH.json'))
            if cohort:        # the cohort regimes: cohort-mode passes (segment: one per image; crop: one for all images)
                entry['n_cohort_searches'] = len(cohort)
                entry['n_cohort_tool_calls'] = int(sum(data.read_json(s).get('n_calls', 0) for s in cohort))
        out[stage] = entry
    return out


def score(prep: Path, run: Path) -> dict:
    prep, run = Path(prep), Path(run)
    config = data.read_json(run / 'CONFIG.json')
    frozen = data.read_json(run / 'PREDICTIONS_COMPLETE.json')
    if data.digest(run / 'predictions.json') != frozen['predictions_sha256']:
        raise ValueError('predictions changed after freezing')
    predictions = data.read_json(run / 'predictions.json')
    labels = data.private_images(prep)
    selected = set(config['images'])
    cv = data.protocol_of(prep) == data.CV_PROTOCOL
    rows = [dict(r, usage=data.EVALUATION) if cv else r for r in data.load_prepared(prep) if r['image_id'] in selected]   # cross-validation: every image is scored out of fold
    gfp_table, gfp_summary, gfp = _gfp_readouts(prep, run, predictions, labels, rows)
    images = _image_table(predictions, labels, rows, gfp)
    crops = pd.DataFrame(predictions)
    if len(crops):
        crops['fc'] = [int(labels[i]['condition'] == POSITIVE) for i in crops.image_id]
        crops['day'] = [int(labels[i]['day']) for i in crops.image_id]
        crops[GFP_TARGET] = [gfp[c][GFP_TARGET] for c in crops.cell_id]
    draws = 200 if config.get('smoke') else BOOTSTRAP_DRAWS
    metrics = dict(arm=config['arm'], model=config['model'], images=len(rows), crops=int(len(crops)),
                   prediction=_heldout_metrics(images, crops, config['seed'], draws, cv))
    metrics['gfp'] = gfp_summary
    seg = images.copy()
    metrics['segmentation'] = dict(
        images_with_objects=int((seg.n_objects > 0).sum()), images=int(len(seg)),
        objects_per_image=dict(mean=float(seg.n_objects.mean()), median=float(seg.n_objects.median()), max=int(seg.n_objects.max())),
        largest_object_area_by_condition_day={f'{c}|{int(d)}': dict(n=int(len(g)), mean=float(g.largest_object_area.mean()) if g.largest_object_area.notna().any() else None)
                                              for (c, d), g in seg.groupby(['condition', 'day'])})
    metrics['costs'] = _costs(run, config['arm'])
    metrics['caveats'] = ['descriptive comparison: 23 wells, 8 held-out wells; no test between arms',
                          'primary read-out: the time point (eight classes, chance 12.5 percent); the condition probe is secondary',
                          'each arm scores its own objects; object counts differ between arms',
                          'condition and plate row are confounded by the plate layout (Control rows B to D, FC rows E to G)',
                          'GFP read-outs depend on the arm\'s own masks and are not an independent accuracy test',
                          'GFP target = integrated GFP inside the object minus area x mean background outside every object; '
                          'the GFP probe is compared with a label-only reference (one-hot day + condition)']
    if cv:
        metrics['caveats'][0:1] = ['descriptive comparison: 23 wells scored by 4-fold leave-one-plate-column-out cross-validation; '
                                   'no test between arms; the four folds use four different classifiers',
                                   'the SSL stage (label-free) trained on 19 wells and validated on 4, and the arms embed every well, '
                                   'so the pixels of every scored well were available to the SSL stage; no label was']
    if config.get('variant') == 'ssl_two_channel':
        metrics['caveats'].append('ssl_two_channel variant: the GFP pixels were an input of the SSL crops, so the GFP '
                                  'probe and the condition read-out are not independent of GFP; the day read-out keeps '
                                  'its labels withheld')
    images.to_csv(run / 'per_image.csv', index=False)
    crops.to_csv(run / 'per_crop.csv', index=False)
    gfp_table.to_csv(run / 'per_object_gfp.csv', index=False)
    data.write_json(run / 'metrics.json', metrics)
    return metrics

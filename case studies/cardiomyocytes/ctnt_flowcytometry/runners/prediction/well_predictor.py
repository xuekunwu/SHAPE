"""Train-only well regression, validation selection and frozen percentage predictions."""
from pathlib import Path
import csv
import numpy as np
from ..dataset.preparation import TARGET
from ..dataset.preparation import digest
from ..dataset.preparation import read_json
from ..dataset.preparation import load_prepared
from ..dataset.preparation import write_json
from ..evaluation.metrics import metrics

ALPHAS = (0.01, 0.1, 1.0, 10.0, 100.0, 1000.0)


def fit_predict(x, table: list[dict], development_labels: dict, *, alphas=ALPHAS):
    """No evaluation targets are accepted by this fitting function."""
    from sklearn.linear_model import Ridge
    from sklearn.pipeline import make_pipeline
    from sklearn.preprocessing import StandardScaler

    x = np.asarray(x, dtype=float)
    if x.ndim != 2 or len(x) != len(table) or not np.isfinite(x).all():
        raise ValueError("invalid well profile matrix")
    wells = [r["well"] for r in table]
    if len(set(wells)) != len(wells):
        raise ValueError("duplicate well profiles")
    usage = np.array([r["usage"] for r in table])
    train, val = usage == "development_train", usage == "development_val"
    if train.sum()<2 or val.sum()<1 or not np.any(usage == "evaluation_only"):
        raise ValueError("need training, validation and held-out wells")
    wanted = {r["well"] for r in table if r["usage"] != "evaluation_only"}
    if set(development_labels) != wanted:
        raise ValueError("fitting must receive development targets only")
    y_train = np.array([development_labels[w] for w, keep in zip(wells, train) if keep])
    y_val = np.array([development_labels[w] for w, keep in zip(wells, val) if keep])
    if not np.isfinite(np.r_[y_train, y_val]).all() or np.any((np.r_[y_train,y_val]<0)|(np.r_[y_train,y_val]>100)):
        raise ValueError("invalid development percent")
    trials, best = [], None
    for alpha in alphas:
        model = make_pipeline(StandardScaler(), Ridge(alpha=alpha, solver="svd"))
        model.fit(x[train], y_train)
        value = metrics(y_val, np.clip(model.predict(x[val]), 0, 100))["mae_percentage_points"]
        trials.append({"alpha": alpha, "validation_mae_percentage_points": value})
        if best is None or value < best[0]:
            best = value, alpha, model
    model = best[2]
    raw = model.predict(x)
    predictions = np.clip(raw, 0, 100)
    scaler, ridge = model.steps[0][1], model.steps[1][1]
    parameters = {"mean": scaler.mean_, "scale": scaler.scale_, "coef": ridge.coef_,
                  "intercept": np.asarray(ridge.intercept_)}
    return predictions, raw, float(y_train.mean()), parameters, {"alpha": best[1], "trials": trials,
        "fit_unit": "well", "fit_split": "development_train", "selection_split": "development_val",
        "refit_on_validation": False, "target": TARGET, "clip_percent": [0, 100]}


def predict(prep: Path, run: Path):
    rows = load_prepared(prep)
    development_path = prep / "private" / "development_labels.json"
    if digest(development_path) != read_json(prep / "PREPARE.json")["development_labels_sha256"]:
        raise ValueError("development labels changed")
    labels = read_json(development_path)
    config = read_json(run / "CONFIG.json")
    selected = [r for r in rows if r["image_id"] in set(config["images"])]
    with (run / "profiles.csv").open(newline="", encoding="utf-8") as f:
        table = list(csv.DictReader(f))
    if {r["well"] for r in table} != {r["well"] for r in selected}:
        raise ValueError("profiles do not cover every scheduled well")
    for r in table:
        expected_usage = next(i["usage"] for i in selected if i["well"] == r["well"])
        if r["usage"] != expected_usage:
            raise ValueError("profile split changed")
    development = {r["well"]: labels[r["well"]][TARGET]
                   for r in table if r["usage"] != "evaluation_only"}
    predictions, raw, mean, params, selection = fit_predict(np.load(run / "corrected.npy"), table, development)
    np.savez(run / "regressor.npz", **params)
    cells = read_json(run / "cells.json")
    records = []
    for r, p, raw_p in zip(table, predictions, raw):
        images = [i["image_id"] for i in selected if i["well"] == r["well"]]
        image = next(i for i in selected if i["well"] == r["well"])
        records.append({"well": r["well"], "usage": r["usage"], "partition": image["partition"],
                        "experiment_date": image["experiment_date"], "plate_id": image["plate_id"],
                        "predicted_ctnt_positive_percent": float(p),
                        "raw_predicted_ctnt_positive_percent": float(raw_p), "baseline_ctnt_positive_percent": mean,
                        "image_ids": images, "retained_cells": sum(c["well"]==r["well"] for c in cells)})
    write_json(run / "predictions.json", records)
    write_json(run / "REGRESSOR.json", {**selection, "weights_sha256": digest(run / "regressor.npz"),
                "profile_sha256": digest(run / "corrected.npy"), "baseline_training_mean_percent": mean,
                "aggregation": "median of cells per view; equal-weight mean of all views per well"})

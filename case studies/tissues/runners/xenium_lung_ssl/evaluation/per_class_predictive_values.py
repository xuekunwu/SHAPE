"""Per-class PPV, NPV, sensitivity and specificity of each group on the held-out cells (one-vs-rest from the probe
predictions), with 95 percent intervals over 1000 bootstrap draws of held-out tiles (seed 20261026, as the probe)."""
import json
import numpy as np
import pandas as pd

m = json.load(open("metrics.json"))
classes, groups = m["classes"], ["control", "codex", "shape", "chance"]
p = pd.read_csv("predictions.csv")
y = p["celltype_final"].map({c: i for i, c in enumerate(classes)}).to_numpy()
pred = {g: p["pred_" + g].map({c: i for i, c in enumerate(classes)}).to_numpy() for g in groups}
K = len(classes)


def values(cm):
    tp = np.diag(cm).astype(float)
    fn, fp = cm.sum(1) - tp, cm.sum(0) - tp
    tn = cm.sum() - tp - fn - fp
    with np.errstate(invalid="ignore", divide="ignore"):
        return dict(ppv=tp / (tp + fp), npv=tn / (tn + fn), sensitivity=tp / (tp + fn), specificity=tn / (tn + fp))


def confusion(idx, g):
    return np.bincount(y[idx] * K + pred[g][idx], minlength=K * K).reshape(K, K)


rng = np.random.default_rng(20261026)
tiles = p["tile"].to_numpy()
uniq = np.unique(tiles)
members = {t: np.flatnonzero(tiles == t) for t in uniq}
draws = [np.concatenate([members[t] for t in rng.choice(uniq, size=len(uniq), replace=True)]) for _ in range(1000)]
rows = []
everyone = np.arange(len(y))
for g in groups:
    point = values(confusion(everyone, g))
    boot = {k: np.array([values(confusion(d, g))[k] for d in draws]) for k in point}
    for i, c in enumerate(classes):
        row = dict(group=g, celltype=c, n_true=int(np.sum(y == i)), n_pred=int(np.sum(pred[g] == i)))
        for k in point:
            row[k] = point[k][i]
            row[k + "_lo"], row[k + "_hi"] = np.nanpercentile(boot[k][:, i], [2.5, 97.5])
        rows.append(row)
pd.DataFrame(rows).to_csv("per_class_predictive_values.csv", index=False)
print(pd.DataFrame(rows).round(3).to_string(index=False))

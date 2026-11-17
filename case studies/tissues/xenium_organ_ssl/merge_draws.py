"""Merge the cell-type probes of an organ's three draws: per group and metric the three held-out values, their mean and
a 95 percent t interval over the draws (df = n_draws - 1); the same for every paired difference. The draws differ in
cells, split and training seed, so the interval includes training variation, unlike the per-draw tile bootstrap.
Also the pooled held-out cells of all draws (each draw's own probe predictions), with 95 percent intervals over 1000
bootstrap draws of (draw, tile) units (seed 20261026), the same draws for every group. Writes MERGED.md, merged.csv
and pooled.csv in eval/ORGAN/ (other probes than probe_v1: MERGED_<probe>.md, merged_<probe>.csv, pooled_<probe>.csv).
usage: merge_draws.py ORGAN [PROBE_NAME]"""
import csv
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd
from scipy import stats

R = Path.home() / 'shape' / 'runs' / 'xenium_ssl'
organ = sys.argv[1]
probe_name = sys.argv[2] if len(sys.argv) > 2 else 'probe_v1'
DRAWS = (1, 2, 3)
METRICS = ('macro_f1', 'balanced_accuracy', 'accuracy')
out = R / 'eval' / organ
suffix = '' if probe_name == 'probe_v1' else f'_{probe_name}'   # probe_v1 keeps its original file names
m ={d: json.loads((out / f'draw{d}' / probe_name / 'metrics.json').read_text()) for d in DRAWS}
groups, classes, pairs = list(m[1]['arms']), m[1]['classes'], list(m[1]['paired_differences'])
for d in DRAWS:
    if list(m[d]['arms']) != groups or m[d]['classes'] != classes or list(m[d]['paired_differences']) != pairs:
        raise ValueError(f'draw {d}: groups, classes or comparisons differ from draw 1')
groups = ['chance'] + [g for g in groups if g != 'chance']
if len({m[d]['prep'] for d in DRAWS}) != len(DRAWS):
    raise ValueError('the draws must be different preparations')


def summary(values):
    v = np.array(values, dtype=float)
    half = stats.t.ppf(0.975, len(v) - 1) * v.std(ddof=1) / np.sqrt(len(v))
    return float(v.mean()), float(v.std(ddof=1)), float(v.mean() - half), float(v.mean() + half)


def cost(d):
    path = out / f'draw{d}' / f'arm_summary{suffix}.csv'
    return {r['arm']: r for r in csv.DictReader(open(path))} if path.exists() else {}


costs = {d: cost(d) for d in DRAWS}
rows = []
for kind, names in (('group', groups), ('difference', pairs)):
    for name in names:
        for k in METRICS:
            per = [m[d]['arms'][name]['test'][k] if kind == 'group' else m[d]['paired_differences'][name][k]['delta']
                   for d in DRAWS]
            boot = [m[d]['arms'][name]['test_ci95'][k] if kind == 'group' else m[d]['paired_differences'][name][k]['ci95']
                    for d in DRAWS]
            mean, sd, lo, hi = summary(per)
            rows.append(dict(kind=kind, name=name, metric=k, **{f'draw{d}': v for d, v in zip(DRAWS, per)},
                             mean=mean, sd=sd, ci95_lo=lo, ci95_hi=hi,
                             draws_tile_ci_excluding_0=sum(1 for b in boot if b[0] > 0 or b[1] < 0)
                             if kind == 'difference' else ''))
with open(out / f'merged{suffix}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=list(rows[0]))
    w.writeheader()
    w.writerows(rows)


def row(kind, name, k):
    return next(r for r in rows if r['kind'] == kind and r['name'] == name and r['metric'] == k)


def fmt(r, sign=''):
    return f"{r['mean']:{sign}.3f} [{r['ci95_lo']:{sign}.3f}, {r['ci95_hi']:{sign}.3f}]"


def per_draw(r, sign=''):
    return ' / '.join(format(r['draw%d' % d], sign + '.3f') for d in DRAWS)


lines = [f'# {organ}: cell-type linear probe, {len(DRAWS)} draws merged', '',
         f"Probe `{probe_name}`, {len(classes)} classes. Mean over the draws with a 95 percent t interval "
         f"(df = {len(DRAWS) - 1}); the draws differ in cells, split and training seed. Per-draw values in "
         f"`merged.csv`; per-draw tile-bootstrap intervals in `draw*/SUMMARY.md`.", '',
         '| Group | Held-out accuracy | Held-out macro-F1 | Balanced accuracy | Macro-F1 per draw |',
         '|---|---|---|---|---|']
for g in groups:
    r = row('group', g, 'macro_f1')
    lines.append(f"| {g} | {fmt(row('group', g, 'accuracy'))} | {fmt(r)} | {fmt(row('group', g, 'balanced_accuracy'))} | "
                 f"{per_draw(r)} |")


def spent(g, key, scale=1.0):
    values = [float(costs[d][g][key] or 0) * scale for d in DRAWS]
    return values, f'{np.mean(values):.2f} ± {np.std(values, ddof=1):.2f}'


lines += ['', 'Cost and run errors per arm (mean ± SD over the draws; errors summed over the draws):', '',
          '| Arm | Tool calls | Tokens (millions) | Run time (min) | Run errors | Error detail per draw |', '|---|---|---|---|---|---|']
for g in groups[1:]:
    if not all(g in costs[d] for d in DRAWS):
        lines.append(f'| {g} | arm_summary missing | | | | |')
        continue
    errors = sum(int(costs[d][g]['run_errors'] or 0) for d in DRAWS)
    lines.append(f"| {g} | {spent(g, 'training_calls')[1]} | {spent(g, 'tokens', 1e-6)[1]} | {spent(g, 'stage_minutes')[1]} | "
                 f"{errors} | {' / '.join(costs[d][g]['error_detail'] for d in DRAWS)} |")

# Pooled held-out cells of the draws: per (draw, tile) unit confusion matrices, then 1000 bootstrap draws of units.
index = {c: i for i, c in enumerate(classes)}
K = len(classes)
frames = []
for d in DRAWS:
    p = pd.read_csv(out / f'draw{d}' / probe_name / 'predictions.csv', dtype=str)
    p['unit'] = f'{d}:' + p['tile']
    frames.append(p.rename(columns={p.columns[2]: 'label'}))
pool = pd.concat(frames, ignore_index=True)
units = pool['unit'].unique()
unit_of = pool['unit'].map({u: i for i, u in enumerate(units)}).to_numpy()
y = pool['label'].map(index).to_numpy()
cm = {g: np.zeros((len(units), K, K)) for g in groups}
for g in groups:
    np.add.at(cm[g], (unit_of, y, pool['pred_' + g].map(index).to_numpy()), 1)


def scores(c):
    tp, rows_, cols = np.diagonal(c, axis1=-2, axis2=-1), c.sum(-1), c.sum(-2)
    with np.errstate(invalid='ignore', divide='ignore'):
        f1 = np.where(rows_ + cols > 0, 2 * tp / (rows_ + cols), 0.0).mean(-1)
        recall = np.where(rows_ > 0, tp / rows_, np.nan)
    return dict(accuracy=tp.sum(-1) / c.sum((-2, -1)), macro_f1=f1, balanced_accuracy=np.nanmean(recall, -1))


boot = np.random.default_rng(20261026).integers(0, len(units), size=(1000, len(units)))
point = {g: scores(cm[g].sum(0)) for g in groups}
samples = {g: scores(np.stack([cm[g][b].sum(0) for b in boot])) for g in groups}
pooled = []
for g in groups:
    for k in METRICS:
        pooled.append(dict(kind='group', name=g, metric=k, value=float(point[g][k]),
                           ci95_lo=float(np.percentile(samples[g][k], 2.5)), ci95_hi=float(np.percentile(samples[g][k], 97.5))))
for pair in pairs:
    b, a = pair.split('-')
    for k in METRICS:
        delta = samples[b][k] - samples[a][k]
        pooled.append(dict(kind='difference', name=pair, metric=k, value=float(point[b][k] - point[a][k]),
                           ci95_lo=float(np.percentile(delta, 2.5)), ci95_hi=float(np.percentile(delta, 97.5))))
pd.DataFrame(pooled).to_csv(out / f'pooled{suffix}.csv', index=False)


def pfmt(name, k, sign=''):
    r = next(x for x in pooled if x['name'] == name and x['metric'] == k)
    return f"{r['value']:{sign}.3f} [{r['ci95_lo']:{sign}.3f}, {r['ci95_hi']:{sign}.3f}]"


lines += ['', f'Pooled held-out cells of the {len(DRAWS)} draws ({len(pool)} cells, {len(units)} tiles; each draw scored by '
              'its own probe; 95 percent bootstrap over (draw, tile) units):', '',
          '| Group | Held-out accuracy | Held-out macro-F1 | Balanced accuracy |', '|---|---|---|---|']
lines += [f"| {g} | {pfmt(g, 'accuracy')} | {pfmt(g, 'macro_f1')} | {pfmt(g, 'balanced_accuracy')} |" for g in groups]
lines += ['', '| Comparison (pooled) | Accuracy | Macro-F1 | Balanced accuracy |', '|---|---|---|---|']
lines += [f"| {p} | {pfmt(p, 'accuracy', '+')} | {pfmt(p, 'macro_f1', '+')} | {pfmt(p, 'balanced_accuracy', '+')} |"
          for p in pairs if not p.endswith('-chance')]
lines += ['', 'Paired differences (per draw on the same tile-bootstrap draws, then merged over draws):', '',
          '| Comparison | Macro-F1 | Balanced accuracy | Accuracy | Macro-F1 per draw | Draws whose tile CI excludes 0 (macro-F1) |',
          '|---|---|---|---|---|---|']
for p in pairs:
    if p.endswith('-chance'):
        continue
    r = row('difference', p, 'macro_f1')
    lines.append(f"| {p} | {fmt(r, '+')} | {fmt(row('difference', p, 'balanced_accuracy'), '+')} | "
                 f"{fmt(row('difference', p, 'accuracy'), '+')} | {per_draw(r, '+')} | "
                 f"{r['draws_tile_ci_excluding_0']} of {len(DRAWS)} |")
lines += ['', '| Cell type | Held-out cells (mean) | ' + ' | '.join(groups) + ' (F1, mean of draws) |',
          '|---|---|' + '---|' * len(groups)]
held = {c: np.mean([m[d]['counts']['evaluation_only'][c] for d in DRAWS]) for c in classes}
for c in sorted(classes, key=lambda c: -held[c]):
    lines.append(f'| {c} | {held[c]:.0f} | ' +
                 ' | '.join(f"{np.mean([m[d]['arms'][g]['per_class_f1'][c] for d in DRAWS]):.3f}" for g in groups) + ' |')
(out / f'MERGED{suffix}.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

"""Summary of one complete draw: each arm's delivered representation, cost and the cell-type probe (metrics.json of
linear_probe.py). Writes arm_summary.csv and SUMMARY.md next to the probe folder (other probes than probe_v1:
arm_summary_<probe>.csv and SUMMARY_<probe>.md).
usage: summarize_probe.py ORGAN DRAW [PROBE_NAME]"""
import csv
import json
from pathlib import Path
import sys

R = Path.home() / 'shape' / 'runs' / 'xenium_ssl'
organ, draw = sys.argv[1], int(sys.argv[2])
probe_name = sys.argv[3] if len(sys.argv) > 3 else 'probe_v1'
runs = R / organ / f'draw{draw}' / 'runs'
out = R / 'eval' / organ / f'draw{draw}'
m = json.loads((out / probe_name / 'metrics.json').read_text())
METRICS = ('accuracy', 'balanced_accuracy', 'macro_f1')


def load(path):
    return json.loads(path.read_text()) if path.exists() else {}


def cost(arm):
    """Delivered representation, training calls, tokens, stage minutes and run errors (tracebacks in the stage log;
    shape: execution errors, failed trials, reasoner failures, fallback rounds; codex: rejected rounds, failed calls)."""
    run = runs / arm
    comp, sel = load(run / 'embed_COMPLETE.json'), load(run / 'SSL_SELECTION.json')
    log = run / 'logs' / 'embed.log'
    tracebacks = log.read_text(errors='replace').count('Traceback') if log.exists() else 0
    row = dict(arm=arm, selection=sel.get('mode', ''), tokens=comp.get('usage', {}).get('total_tokens', 0),
               stage_minutes=round(comp.get('seconds', 0) / 60, 1), delivered='', training_calls=0, epochs='',
               run_errors=tracebacks, error_detail=f'tracebacks {tracebacks}')
    if arm == 'shape':
        row['training_calls'] = len(list((run / 'ssl').glob('call_*.result.json'))) - 1   # call_000 = the input
        row['delivered'] = (sel.get('checkpoint') or '').split('/runs/shape/')[-1]
        d = load(run / 'representation_SEARCH.json').get('details', {})
        parts = dict(execution_errors=len(d.get('execution_errors') or []), failed_trials=len(d.get('failed') or []),
                     reasoner_failures=int(d.get('reasoner_failures') or 0), fallback_rounds=int(d.get('fallback_rounds') or 0))
        row['delivered'] = (d.get('delivered') or {}).get('trial', row['delivered'])
    elif arm == 'codex':
        row['training_calls'] = len([p for p in (run / 'agent' / 'embed' / 'calls').glob('c*') if p.is_dir()])
        row['delivered'] = comp.get('usage', {}).get('deliveries', {}).get('dataset', {}).get('call', '')
        rounds = load(run / 'agent' / 'embed' / 'ROUNDS.json') or []
        calls = load(run / 'agent' / 'embed' / 'CALLS.json').get('dataset', {})
        parts = dict(rejected_rounds=sum(1 for r in rounds if r.get('outcome') == 'rejected'),
                     failed_calls=sum(1 for c in calls.values() if c.get('status') != 'complete'))
    else:
        parts = {}
    row['run_errors'] += sum(parts.values())
    row['error_detail'] = '; '.join([row['error_detail']] + [f'{k} {v}' for k, v in parts.items()])
    if arm in ('scportrait', 'dino4cells') and (run / 'TRAIN.json').exists():
        t = load(run / 'TRAIN.json')
        row.update(training_calls=1, delivered={'scportrait': 'MAE fine-tune', 'dino4cells': 'DINO continued'}[arm],
                   epochs=f"{t.get('epochs_trained')}/{t.get('max_epochs')} {t.get('stop_reason')}")
    elif arm in ('subcell', 'dino4cells'):                 # zero-shot (DINO4Cells since 2026-10-28)
        row.update(training_calls=0, delivered='frozen encoder')
    return row


arms = [n for n in m['arms'] if n != 'chance']
rows = []
for name in ['chance'] + arms:
    a = m['arms'][name]
    row = cost(name) if name != 'chance' else dict(arm='chance', selection='random class from training frequencies')
    row['probe_C'] = a.get('C', '')
    for k in METRICS:
        row[k], (row[k + '_lo'], row[k + '_hi']) = round(a['test'][k], 4), [round(v, 4) for v in a['test_ci95'][k]]
    rows.append(row)
fields = ['arm', 'selection', 'delivered', 'training_calls', 'epochs', 'tokens', 'stage_minutes', 'run_errors',
          'error_detail', 'probe_C'] + [f'{k}{s}' for k in METRICS for s in ('', '_lo', '_hi')]
suffix = '' if probe_name == 'probe_v1' else f'_{probe_name}'   # probe_v1 keeps its original file names
with open(out / f'arm_summary{suffix}.csv', 'w', newline='') as f:
    w = csv.DictWriter(f, fieldnames=fields, extrasaction='ignore')
    w.writeheader()
    w.writerows(rows)


def ci(a, k):
    return f"{a['test'][k]:.3f} [{a['test_ci95'][k][0]:.3f}, {a['test_ci95'][k][1]:.3f}]"


n_test = sum(m['counts']['evaluation_only'].values())
lines = [f'# {organ} draw {draw}: cell-type linear probe (chance / {" / ".join(arms)})', '',
         f"Probe `{probe_name}` ({m['protocol']}): {len(m['classes'])} classes, {n_test} held-out cells in "
         f"{m['test_tiles']} tiles; dropped classes (fewer than {m['min_train']} training cells): "
         f"{', '.join(c for c in m['dropped_classes'] if c not in m.get('excluded_classes', [])) or 'none'}"
         f"{'; excluded by design: ' + ', '.join(m['excluded_classes']) if m.get('excluded_classes') else ''}. "
         f"Brackets: 95 percent bootstrap over held-out tiles "
         f"({m['draws']} draws, seed {m['seed']}, the same draws for every group).", '',
         '| Group | Delivered | Training calls | Tokens | Stage min | Run errors | C | Accuracy | Balanced accuracy | Macro-F1 |',
         '|---|---|---|---|---|---|---|---|---|---|']
for row, name in zip(rows, ['chance'] + arms):
    a = m['arms'][name]
    delivered = f"{row.get('delivered', '')} {row.get('epochs', '')}".strip() or row['selection']
    lines.append(f"| {name} | {delivered} | {row.get('training_calls', '')} | {row.get('tokens', '')} | "
                 f"{row.get('stage_minutes', '')} | {row.get('run_errors', '')} | {row['probe_C']} | {ci(a, 'accuracy')} | "
                 f"{ci(a, 'balanced_accuracy')} | {ci(a, 'macro_f1')} |")
lines += ['', 'Paired differences (same bootstrap draws):', '', '| Comparison | Accuracy | Balanced accuracy | Macro-F1 |',
          '|---|---|---|---|']
for key, d in m['paired_differences'].items():
    cells = [f"{d[k]['delta']:+.3f} [{d[k]['ci95'][0]:+.3f}, {d[k]['ci95'][1]:+.3f}]" for k in METRICS]
    lines.append(f'| {key} | ' + ' | '.join(cells) + ' |')
lines += ['', '| Cell type | Held-out cells | ' + ' | '.join(['chance'] + arms) + ' (F1) |',
          '|---|---|' + '---|' * (len(arms) + 1)]
for c in sorted(m['classes'], key=lambda c: -m['counts']['evaluation_only'][c]):
    lines.append(f"| {c} | {m['counts']['evaluation_only'][c]} | " +
                 ' | '.join(f"{m['arms'][n]['per_class_f1'][c]:.3f}" for n in ['chance'] + arms) + ' |')
s = m['sanity']
lines += ['', f"Sanity: training-majority class ({s['majority_class']}) balanced accuracy "
              f"{s['majority']['balanced_accuracy']:.3f}; " +
          '; '.join(f"{k} balanced accuracy {v['balanced_accuracy']:.3f}" for k, v in s.items() if k.startswith('shuffled'))]
lines += ['', 'Run errors: ' + '; '.join(f"{r['arm']} ({r['error_detail']})" for r in rows if r['arm'] != 'chance')]
(out / f'SUMMARY{suffix}.md').write_text('\n'.join(lines) + '\n')
print('\n'.join(lines))

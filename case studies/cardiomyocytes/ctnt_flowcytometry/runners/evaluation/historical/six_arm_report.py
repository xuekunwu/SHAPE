"""Rebuild the six-arm report from a hash-verified, portable export (standard library only)."""
import csv
import hashlib
import json
from pathlib import Path

STAGES = ('segment', 'crop', 'embed', 'batch', 'predict')


def read(path):
    return json.loads(path.read_text(encoding='utf-8'))


def digest(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_csv(path, rows):
    with path.open('w', newline='', encoding='utf-8') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def tokens(usage):
    if not isinstance(usage, dict):
        return None
    if isinstance(usage.get('total_tokens'), (int, float)):
        return usage['total_tokens']
    return None


def reproduce(root):
    manifest = read(root/'EXPORT.json')
    for name, expected in manifest['files'].items():
        path = (root/name).resolve()
        if not path.is_relative_to(root.resolve()) or digest(path) != expected:
            raise ValueError(f'export evidence changed: {name}')
    arms = manifest['arms']
    data = {a: {n: read(root/a/(n+'.json')) for n in ('CONFIG','RESULT','STAGE_READOUTS')}
            for a in arms}
    report = {'scope': 'six_arm_pilot_descriptive_comparison', 'input_sha256': manifest['files'],
              'arms': {}, 'limitations': [
        'Same frozen 114 images / 18 wells; 7 train, 2 validation, 9 held-out wells.',
        'Updated SHAPE changes both code/policy and image-stage effort; this is not an isolated ablation.',
        'Claude Code uses Sonnet 5; the other agent arms use GPT-5.6-luna. Agent and model effects are confounded.',
        'Historical arms have different runtime/recovery policies; report source and elapsed time descriptively.',
        'Historical SHAPE crop usage is a lower bound after timeout/recovery; no gateway-billing reconciliation.',
        'No GT instance masks or cell-level cTnT labels: segmentation/crop readouts are QC proxies.',
        'Donors and batches can overlap between splits; no population confidence intervals or causal ranking.',
        'Biomni A1 and Claude Code select native-tool actions via the shared Codex protocol; Biomni built-in tools are disabled.',
        'Reported stage tokens include reasoning where the backend reports it; do not add reasoning tokens again.']}
    summary, process, paired, image_rows = [], [], [], []
    base = data['control']['RESULT']['primary']
    well_sets, truths = {}, {}
    for arm, record in data.items():
        cfg, result, stages = record['CONFIG'], record['RESULT'], record['STAGE_READOUTS']
        metrics = result['primary']
        row = {'arm':arm, 'model':cfg['model'], 'source_sha256':cfg['source_sha256'],
               'reasoning_profile':cfg.get('reasoning_profile','all-high'), **metrics}
        for key in ('mae_percentage_points','rmse_percentage_points','bias_percentage_points','r2'):
            row['delta_vs_control_'+key] = metrics[key]-base[key] if metrics[key] is not None and base[key] is not None else None
        summary.append(row)
        stage_rows = []
        for stage in STAGES:
            info = stages['process'][stage]
            entry = {'arm':arm, 'stage':stage, 'seconds':info.get('seconds'),
                     'reported_tokens':tokens(info.get('usage')),
                     'reasoning':json.dumps(info.get('reasoning'), sort_keys=True),
                     'usage_scope':'lower_bound_after_recovery' if arm=='shape' and stage=='crop' else 'recorded_stage_usage'}
            process.append(entry)
            stage_rows.append(entry)
        with (root/arm/'heldout_predictions.csv').open(newline='', encoding='utf-8') as f:
            predictions = list(csv.DictReader(f))
        well_sets[arm] = {r['well'] for r in predictions}
        if len(well_sets[arm]) != len(predictions):
            raise ValueError('duplicate held-out wells')
        for p in predictions:
            actual = float(p['ctnt_positive_percent'])
            if p['well'] in truths and truths[p['well']] != actual:
                raise ValueError('held-out target differs between arms')
            truths[p['well']] = actual
            predicted = float(p['predicted_ctnt_positive_percent'])
            paired.append({'arm':arm, **p, 'error_pp':predicted-actual, 'absolute_error_pp':abs(predicted-actual)})
        for stage, field in [('segment','per_image'),('crop','per_image')]:
            for item in stages[stage][field]:
                image_rows.append({'arm':arm, 'stage':stage, 'image_id':item['image_id'],
                                   'well':item['well'], 'readouts_json':json.dumps(item,sort_keys=True)})
        report['arms'][arm] = {'endpoint':metrics, 'batch_metrics':result['batch_metrics'],
            'segment':stages['segment']['well_macro'], 'crop':stages['crop']['well_macro_quality'],
            'embed':stages['embed'], 'batch':stages['batch'], 'process':stage_rows,
            'unavailable':stages['unavailable'], 'config':cfg}
    if any(w != well_sets['control'] for w in well_sets.values()):
        raise ValueError('held-out well sets differ')
    out = root/'compare'
    out.mkdir(exist_ok=True)
    write_csv(out/'summary.csv', summary)
    write_csv(out/'stage_process.csv', process)
    write_csv(out/'paired_per_well.csv', paired)
    write_csv(out/'paired_per_image.csv', image_rows)
    (out/'comparison.json').write_text(json.dumps(report,indent=2,allow_nan=False)+'\n',encoding='utf-8')
    def number(v):
        return 'NA' if v is None else f'{v:.4f}'
    lines = ['# cTnT six-arm pilot comparison', '',
        'All six arms completed the same frozen pilot. Lower held-out MAE/RMSE is better; differences are descriptive.', '',
        '| Arm | MAE (pp) | RMSE (pp) | Bias (pp) | R2 | MAE delta vs control |',
        '|---|---:|---:|---:|---:|---:|']
    for row in summary:
        lines.append('| '+row['arm']+' | '+' | '.join(number(row[k]) for k in (
            'mae_percentage_points','rmse_percentage_points','bias_percentage_points','r2',
            'delta_vs_control_mae_percentage_points'))+' |')
    best = min(summary,key=lambda r:r['mae_percentage_points'])
    lines += ['', f"The lowest observed held-out MAE is {best['arm']} ({best['mae_percentage_points']:.4f} pp). "
        'This does not establish superiority beyond these nine held-out wells.', '',
        '| Arm | Segment minutes / tokens | Crop minutes / tokens | Embed minutes / tokens | Batch minutes / tokens |',
        '|---|---:|---:|---:|---:|']
    for arm in arms:
        entries = [p for p in process if p['arm']==arm and p['stage']!='predict']
        lines.append('| '+arm+' | '+' | '.join(
            f"{p['seconds']/60:.1f} / {p['reported_tokens'] if p['reported_tokens'] is not None else 'NA'}"
            for p in entries)+' |')
    for title, stage in [('Segmentation QC (well macro)','segment'),('Crop QC (well macro)','crop')]:
        keys = sorted({key for a in arms for key,v in report['arms'][a][stage].items()
                       if isinstance(v,(int,float))})
        lines += ['', '## '+title, '', '| Metric | '+' | '.join(arms)+' |',
                  '|---|'+'---:|'*len(arms)]
        for key in keys:
            lines.append('| '+key+' | '+' | '.join(number(report['arms'][a][stage].get(key)) for a in arms)+' |')
    lines += ['', '## Stage readouts', '',
        'Segmentation and crop: well-weighted QC proxies and paired image records. '
        'Embedding: selection, split-wise representation health, image leakage and sibling-site retrieval. '
        'Batch: admission/abstention and date/plate structure. Prediction: well-level cTnT percentage regression. '
        'Full values and explicit unavailable metrics are in comparison.json.', '', '## Interpretation limits', '']
    lines.extend('- '+v for v in report['limitations'])
    lines += ['', '## Reproduction', '', 'Run `python compare/reproduce.py` from the export root. '
        'It verifies all evidence hashes and rebuilds the report and tables. '
        'EXPORT.json records source identities, native delivery validation and remote locations. '
        'Raw images, masks, crops, embeddings and checkpoints remain on the VM shared filesystem.', '']
    (out/'README.md').write_text('\n'.join(lines),encoding='utf-8')
    return report


if __name__ == '__main__':
    reproduce(Path(__file__).resolve().parent.parent)

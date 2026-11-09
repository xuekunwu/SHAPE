"""Export small deidentified smoke evidence and descriptive comparisons, without source images/weights."""
import argparse
from pathlib import Path
import csv
import json
import shutil
from . import data


def export(suite,out):
    if out.exists():raise FileExistsError(out)
    out.mkdir(parents=True)
    suite_record=data.read_json(suite/'SUITE_META.json')
    if suite_record['status'] not in ('complete','failed'):raise ValueError('suite is still active')
    data.write_json(out/'SUITE_META.json',suite_record)
    rows=[];threshold=None
    if (suite/'references/THRESHOLD.json').exists():
        threshold=data.read_json(suite/'references/THRESHOLD.json')
        data.write_json(out/'THRESHOLD.json',threshold)
        data.write_json(out/'REFERENCE_COMPLETE.json',data.read_json(suite/'references/REFERENCE_COMPLETE.json'))
    for arm in ['control','shape','codex']:
        target=out/arm;target.mkdir()
        for name in ['RESULT.json','STAGE_READOUTS.json','CLASSIFIER.json','PREFLIGHT.json','RUN_META.json',
            'DETECTION_BY_IMAGE.json','MATCHED_IMAGE_FRACTIONS.json']:
            src=suite/arm/name
            if src.exists():
                # Keep structured scientific evidence, never raw images, masks, model weights or fluorescence crops.
                data.write_json(target/name,data.read_json(src))
        if (suite/arm/'MATCHED_HELDOUT_CELLS.json').exists():
            matched=data.read_json(suite/arm/'MATCHED_HELDOUT_CELLS.json')
            if matched:
                with (target/'matched_heldout_cells.csv').open('w',newline='',encoding='utf-8') as f:
                    writer=csv.DictWriter(f,fieldnames=list(matched[0]));writer.writeheader();writer.writerows(matched)
        if (target/'RESULT.json').exists():
            r=data.read_json(target/'RESULT.json');primary=r['primary'];d=r['detection']
            rows.append(dict(arm=arm,auroc=primary['auroc'],auprc=primary['auprc'],accuracy=primary['accuracy'],
                balanced_accuracy=primary['balanced_accuracy'],f1=primary['f1'],brier=primary['brier'],
                reference_matching_coverage=d['reference_matching_coverage'],positive_end_to_end_recall=d['positive_end_to_end_recall'],
                label_coverage=r['label_coverage'],coverage_criterion_met=r['coverage_criterion']['met']))
        else:
            # A failed smoke is an exported failure, never presented as a completed scientific result.
            for src in [suite/(arm+'.log'),suite/arm/'logs/score.log',suite/arm/'logs/predict.log']:
                if src.exists():(target/(src.stem+'_tail.txt')).write_text(src.read_text(errors='replace')[-12000:],encoding='utf-8')
    if rows:
        with (out/'comparison.csv').open('w',newline='',encoding='utf-8') as f:
            writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    summary=dict(scope='exploratory_smoke_descriptive_comparison',status=suite_record['status'],arms=rows,
        prediction_channels=['Brightfield'],private_evaluation_channels=['Brightfield','DAPI','TroponinT'],
        interpretation=['No validated negative/positive fluorescence controls; training-only mixture proxy labels',
            'Fixed algorithmic whole-cell reference is a pseudo-reference, sharing its algorithm with CONTROL',
            'Composite channel alignment is unverified; no segmentation-ground-truth accuracy claim',
            'Inspect detection and label coverage alongside matched-cell classification',
            'Smoke search covers every selected image in both SHAPE and CODEX; all use the same complete-well subset'])
    data.write_json(out/'comparison.json',summary)
    lines=['# cTnT staining three-arm smoke','',f"Suite status: **{suite_record['status']}**.",'',
        'Brightfield-only prediction; identical crop boxes retain private DAPI/cTnT for evaluation.','',
        '| Arm | AUROC | AUPRC | Accuracy | Balanced accuracy | Reference coverage | Positive E2E recall |',
        '|---|---:|---:|---:|---:|---:|---:|']
    def fmt(x):return 'undefined' if x is None else f'{x:.4f}'
    for r in rows:lines.append('| '+' | '.join([r['arm'],*[fmt(r[k]) for k in ['auroc','auprc','accuracy','balanced_accuracy','reference_matching_coverage','positive_end_to_end_recall']]])+' |')
    lines+=['','## Interpretation','',*['- '+s for s in summary['interpretation']],'',
        'No source images, fluorescence crops, masks or checkpoints are included. Per-cell CSVs use opaque IDs.']
    if suite_record.get('error'):lines+=['','## Failure','',suite_record['error']]
    (out/'README.md').write_text('\n'.join(lines)+'\n',encoding='utf-8')
    files={p.relative_to(out).as_posix():data.digest(p) for p in out.rglob('*') if p.is_file()}
    data.write_json(out/'EXPORT.json',dict(status='complete',suite_status=suite_record['status'],files=files,
        source_sha256=suite_record['source_sha256'],boundary='small deidentified derived records; no image/weight transfer'))
    return out


def main():
    p=argparse.ArgumentParser();p.add_argument('--suite',type=Path,required=True);p.add_argument('--out',type=Path,required=True)
    a=p.parse_args();export(a.suite,a.out)


if __name__=='__main__':main()

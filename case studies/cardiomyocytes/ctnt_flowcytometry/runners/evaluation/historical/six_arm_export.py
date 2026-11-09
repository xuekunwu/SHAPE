"""Validate all completed arms on the VM, export compact evidence and reproduce the report."""

# Internal worker CLI; the supported public entrypoint is run.py.
if __name__ == "__main__" and not __package__:
    import sys as _sys
    from pathlib import Path as _Path
    _sys.path.insert(0, str(_Path(__file__).resolve().parents[6]))
    __package__ = 'case studies.cardiomyocytes.ctnt_flowcytometry.runners.evaluation.historical'

import argparse
import importlib
import json
from pathlib import Path
import shutil

from ...dataset import preparation as data
from ...dataset import sampling as pilot
from .six_arm_report import reproduce
from ..heldout_evaluation import score

ARMS = ('control','shape','codex','shape_new','biomni','claude')
MODELS = {a:'gpt-5.6-luna' for a in ARMS} | {'claude':'claude-sonnet-5'}
MATCH = ('images','pilot_sha256','seed','cores','train_cells_per_site','inference_chunk',
         'ssl_epoch_budget','protocol','target','batch_fields')


def export(prep, plan, out):
    if set(plan['runs']) != set(ARMS):
        raise ValueError('export requires all six arms')
    if out.exists():
        raise FileExistsError(out)
    verified = {}
    for arm in ARMS:
        entry = plan['runs'][arm]
        run = Path(entry['path'])
        config = data.read_json(run/'CONFIG.json')
        expected_arm = 'shape' if arm=='shape_new' else arm
        if (config['arm'] != expected_arm or config['model'] != MODELS[arm]
                or config['source_sha256'] != entry['source_sha256'] or config.get('smoke')):
            raise ValueError(f'{arm}: identity mismatch')
        if (run/'CTNT_FLOW_PILOT_COMPLETE').read_text().strip() != entry['source_sha256']:
            raise ValueError(f'{arm}: invalid completion marker')
        if data.read_json(run/'RUN_META.json')['status'] != 'complete':
            raise ValueError(f'{arm}: not complete')
        sample = pilot.verify(prep,run/'PILOT.json')
        if config['images'] != sample['images'] or data.digest(run/'PILOT.json') != config['pilot_sha256']:
            raise ValueError(f'{arm}: pilot mismatch')
        if verified and any(config.get(k)!=verified['control']['config'].get(k) for k in MATCH):
            raise ValueError(f'{arm}: frozen inputs or search budgets differ')
        saved = data.read_json(run/'RESULT.json')
        if saved['source_sha256'] != entry['source_sha256']:
            raise ValueError(f'{arm}: result source mismatch')
        if arm in ('biomni','claude') and any(
                config['reasoning']['stages'].get(s) != {'reasoning_effort':'high'}
                for s in ('segment','crop','embed','batch')):
            raise ValueError(f'{arm}: high reasoning not recorded')
        if arm == 'shape_new' and config.get('reasoning_profile') != 'image-medium':
            raise ValueError('updated SHAPE reasoning profile mismatch')
        if data.digest(run/'STAGE_READOUTS.json') != saved['stage_readouts']['sha256']:
            raise ValueError(f'{arm}: stage readouts changed')
        # Verifies every delivered mask/array/checkpoint hash before recomputing the frozen endpoint.
        fresh = score(prep,run,persist=False)
        if fresh != saved:
            raise ValueError(f'{arm}: scoring changed')
        verified[arm] = {'config':config, 'delivery_validated':True, **entry}
    out.mkdir(parents=True)
    for arm in ARMS:
        source, target = Path(verified[arm]['path']), out/arm
        target.mkdir()
        paths = [source/n for n in ('CONFIG.json','PREFLIGHT.json','PILOT.json','RUN_META.json','DELIVERY.json',
            'RESULT.json','STAGE_READOUTS.json','REGRESSOR.json','SSL_SELECTION.json','batch_SEARCH.json',
            'heldout_predictions.csv','predictions.json','CTNT_FLOW_PILOT_COMPLETE','RESUME.json') if (source/n).exists()]
        paths += list(source.glob('*_COMPLETE.json'))+list(source.glob('*_USAGE.json'))
        paths += list((source/'agent').glob('*/SELECTED.json'))+list((source/'agent').glob('*/ROUNDS.json'))
        for path in paths:
            destination = target/path.relative_to(source)
            destination.parent.mkdir(parents=True,exist_ok=True)
            shutil.copy2(path,destination)
        (target/'README.md').write_text(f'# {arm}\n\nModel: `{MODELS[arm]}`.\n\n'
            f'Source SHA256: `{verified[arm]["source_sha256"]}`.\n\n'
            f'Native run: `{source}`. Full delivery hashes verified before export.\n',encoding='utf-8')
    files = {p.relative_to(out).as_posix():data.digest(p) for p in sorted(out.rglob('*')) if p.is_file()}
    data.write_json(out/'EXPORT.json', {'arms':list(ARMS),'runs':verified,'files':files,
        'validation':'all native delivery hashes, frozen pilot, completed markers and recomputed endpoints'})
    (out/'compare').mkdir()
    shutil.copy2(Path(__file__).with_name('six_arm_report.py'),out/'compare'/'reproduce.py')
    reproduce(out)
    (out/'README.md').write_text('# cTnT six-arm pilot\n\n'
        'Completed runs, immutable evidence and descriptive analysis.\n\n'
        '[Comparison and limitations](compare/README.md) · [Evidence manifest](EXPORT.json)\n',encoding='utf-8')
    (out/'EXPORT_COMPLETE').write_text(data.digest(out/'EXPORT.json')+'\n',encoding='ascii')


if __name__ == '__main__':
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--prep',type=Path,required=True)
    p.add_argument('--plan',type=Path,required=True)
    p.add_argument('--out',type=Path,required=True)
    args=p.parse_args()
    export(args.prep,data.read_json(args.plan),args.out)

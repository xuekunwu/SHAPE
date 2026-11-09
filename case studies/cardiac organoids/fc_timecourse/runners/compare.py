"""The three-arm table from completed runs' metrics.json files; verifies that the arms ran the same images and source."""
from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd

if __package__ in (None, ''):  # executed as a script: import as the study package from the repository root
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[4]))
    __package__ = 'case studies.cardiac organoids.fc_timecourse.runners'

from . import data  # noqa: E402

ARMS = ('control', 'shape', 'codex')


def _get(d: dict, *keys):
    for k in keys:
        if not isinstance(d, dict) or k not in d or d[k] is None:
            return None
        d = d[k]
    return d


def compare(runs: dict, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    configs = {arm: data.read_json(path / 'CONFIG.json') for arm, path in runs.items()}
    metrics = {arm: data.read_json(path / 'metrics.json') for arm, path in runs.items()}
    images = {tuple(sorted(c['images'])) for c in configs.values()}
    if len(images) != 1:
        raise ValueError('the arms did not run the same images')
    sources = {c['source_sha256'] for c in configs.values()}
    rows = []
    for arm, m in metrics.items():
        p = m['prediction']
        boot = _get(p, 'bootstrap', 'image_level') or {}
        rows.append(dict(
            arm=arm, images=m['images'], crops=m['crops'], images_with_objects=_get(m, 'segmentation', 'images_with_objects'),
            day_image_accuracy=_get(p, 'day_image_level', 'accuracy'),
            day_image_accuracy_ci_low=_get(boot, 'day_accuracy', 'low'), day_image_accuracy_ci_high=_get(boot, 'day_accuracy', 'high'),
            day_within_one_timepoint=_get(p, 'day_image_level', 'within_one_timepoint'),
            day_balanced_accuracy=_get(p, 'day_image_level', 'balanced_accuracy'),
            day_crop_accuracy=_get(p, 'day_crop_level', 'accuracy'),
            day_regression_mae_days=_get(p, 'day_regression', 'mae_days'), day_regression_spearman_rho=_get(p, 'day_regression', 'spearman_rho'),
            condition_image_accuracy=_get(p, 'condition_image_level', 'accuracy'), condition_image_auroc=_get(p, 'condition_image_level', 'auroc'),
            condition_image_accuracy_ci_low=_get(boot, 'condition_accuracy', 'low'), condition_image_accuracy_ci_high=_get(boot, 'condition_accuracy', 'high'),
            gfp_probe_spearman_rho=_get(p, 'gfp_image_level', 'spearman_rho'), gfp_probe_mae=_get(p, 'gfp_image_level', 'mae'),
            gfp_probe_spearman_ci_low=_get(boot, 'gfp_spearman_rho', 'low'), gfp_probe_spearman_ci_high=_get(boot, 'gfp_spearman_rho', 'high'),
            gfp_label_baseline_spearman_rho=_get(p, 'gfp_image_level', 'baseline_spearman_rho'), gfp_label_baseline_mae=_get(p, 'gfp_image_level', 'baseline_mae'),
            fc_probability_vs_gfp_spearman=_get(m, 'gfp', 'heldout_spearman', 'fc_probability_vs_gfp_integrated_bgsub'),
            segment_seconds=_get(m, 'costs', 'segment', 'seconds'), crop_seconds=_get(m, 'costs', 'crop', 'seconds'),
            embed_seconds=_get(m, 'costs', 'embed', 'seconds'),
            total_tokens=sum((_get(m, 'costs', s, 'total_tokens') or 0) for s in ('segment', 'crop', 'embed')),
            cores=configs[arm]['cores']))
    table = pd.DataFrame(rows)
    table.to_csv(out / 'main_metrics.csv', index=False)
    trajectories = []
    for arm, m in metrics.items():
        for key, v in (m.get('gfp', {}).get('by_condition_day') or {}).items():
            condition, day = key.split('|')
            trajectories.append(dict(arm=arm, condition=condition, day=int(day), **v))
    pd.DataFrame(trajectories).to_csv(out / 'gfp_by_condition_day.csv', index=False)
    record = dict(arms=list(runs), same_images=True, same_source=len(sources) == 1, sources=sorted(sources),
                  main_metrics=rows, caveats=sorted({c for m in metrics.values() for c in m.get('caveats', [])}),
                  runs={arm: str(Path(p).resolve()) for arm, p in runs.items()})
    data.write_json(out / 'COMPARISON.json', record)
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        p.add_argument(f'--{arm}', type=Path, required=arm != 'codex')
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    runs = {arm: getattr(args, arm) for arm in ARMS if getattr(args, arm) is not None}
    print(compare(runs, args.out))


if __name__ == '__main__':
    main()

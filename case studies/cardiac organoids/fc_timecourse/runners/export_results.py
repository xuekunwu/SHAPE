"""Curate completed run directories into a results record: whitelisted files, cluster paths replaced, EXPORT.json with hashes."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
from pathlib import Path

ARMS = ('control', 'shape', 'codex')
# Per-arm files that are small, aggregate or decision-level. Masks, crops, embeddings, weights,
# per-call tool outputs (agent/*/CALLS.json) and per-image mask paths (segmentation.json) stay on the cluster.
ARM_FILES = (
    'metrics.json', 'per_image.csv', 'per_crop.csv', 'per_object_gfp.csv', 'CLASSIFIER.json', 'CONFIG.json',
    'PREFLIGHT.json', 'RESCORE.json', 'RUN_META.json', 'SSL_SELECTION.json', 'ENVIRONMENT.json',
    'segment_USAGE.json', 'crop_USAGE.json', 'embed_USAGE.json', 'segment_COMPLETE.json', 'crop_COMPLETE.json',
    'embed_COMPLETE.json', 'predict_COMPLETE.json', 'score_COMPLETE.json', 'representation_SEARCH.json',
    'SEARCH_WARNINGS.jsonl', 'agent/segment/ROUNDS.json', 'agent/segment/SELECTED.json', 'agent/crop/ROUNDS.json',
    'agent/crop/SELECTED.json', 'agent/embed/ROUNDS.json', 'agent/embed/SELECTED.json',
)
COMPARISON_FILES = ('main_metrics.csv', 'gfp_by_condition_day.csv', 'COMPARISON.json')
# Absolute cluster locations carry no information the record needs; the environment variables name the same roots.
PATH_SUBSTITUTIONS = ((re.compile(r'/scratch/users/[A-Za-z0-9_]+'), '$SCRATCH'),
                      (re.compile(r'/scratch/groups/[A-Za-z0-9_]+'), '$GROUP_SCRATCH'))


def _redacted(text: str) -> str:
    for pattern, replacement in PATH_SUBSTITUTIONS:
        text = pattern.sub(replacement, text)
    return text


def _copy_text(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_text(_redacted(src.read_text(encoding='utf-8')), encoding='utf-8', newline='\n')


def _export_arm(run: Path, out: Path) -> list[str]:
    copied = []
    for rel in ARM_FILES:
        src = run / rel
        if src.exists():
            _copy_text(src, out / rel)
            copied.append(rel)
    searches = sorted(run.glob('segment/*/SEARCH.json'))
    if searches:  # one record per image from the SHAPE per-image search, combined into a single file
        combined = {p.parent.name: json.loads(_redacted(p.read_text(encoding='utf-8'))) for p in searches}
        (out / 'segment_SEARCH.json').write_text(json.dumps(combined, indent=2) + '\n', encoding='utf-8', newline='\n')
        copied.append('segment_SEARCH.json')
    return copied


def export(runs: dict, comparison: Path, out: Path) -> dict:
    out = Path(out)
    out.mkdir(parents=True, exist_ok=True)
    for arm, run in runs.items():
        shutil.rmtree(out / arm, ignore_errors=True)
        _export_arm(Path(run), out / arm)
    shutil.rmtree(out / 'comparison', ignore_errors=True)
    for rel in COMPARISON_FILES:
        _copy_text(Path(comparison) / rel, out / 'comparison' / rel)
    identities = {}
    for arm, run in runs.items():
        config = json.loads((Path(run) / 'CONFIG.json').read_text(encoding='utf-8'))
        rescore_path = Path(run) / 'RESCORE.json'
        rescore = json.loads(rescore_path.read_text(encoding='utf-8')) if rescore_path.exists() else {}
        identities[arm] = dict(run_source_sha256=config['source_sha256'], prepare_sha256=config.get('prepare_sha256'),
                               rescore_source_sha256=rescore.get('rescore_source_sha256'), cores=config.get('cores'))
    files = {}
    for path in sorted(p for p in out.rglob('*') if p.is_file() and p.name != 'EXPORT.json'):
        files[path.relative_to(out).as_posix()] = dict(bytes=path.stat().st_size, sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    record = dict(schema=1, builder_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                  path_substitutions=[r for _, r in PATH_SUBSTITUTIONS], identities=identities, files=files)
    (out / 'EXPORT.json').write_text(json.dumps(record, indent=2) + '\n', encoding='utf-8', newline='\n')
    return record


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    for arm in ARMS:
        p.add_argument(f'--{arm}', type=Path, required=True)
    p.add_argument('--comparison', type=Path, required=True)
    p.add_argument('--out', type=Path, required=True)
    args = p.parse_args(argv)
    record = export({arm: getattr(args, arm) for arm in ARMS}, args.comparison, args.out)
    print(f"{len(record['files'])} files -> {args.out}")


if __name__ == '__main__':
    main()

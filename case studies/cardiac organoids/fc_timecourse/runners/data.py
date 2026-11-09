"""Prepare the FC time course: the phase-contrast channel is public, GFP and every label are private, one fixed
well split. Identifiers are pseudonymous; condition, day and well position never enter the public table.

Two split protocols. `co-fc-timecourse-well-stratified-11-4-8-v1` (the default): 11 training, 4 validation and 8
held-out wells, the held-out wells are never seen by the SSL stage or the read-outs. `co-fc-timecourse-ssl19-cvcol4-v1`:
no held-out wells in the SSL stage (19 wells train the SSL, four wells validate it: the public `usage`) and the
read-outs are scored by leave-one-plate-column-out cross-validation (the private `cv_fold`). The SSL stage is
label-free, so the pixels of every well may train it."""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import os
from pathlib import Path
import random
import re
import shutil

import numpy as np
import tifffile

from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json, within

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[3]
SPEC = json.loads((HERE.parent / 'harness' / 'dataset.json').read_text(encoding='utf-8'))
USAGES = ('development_train', 'development_val', 'evaluation_only')
EVALUATION = 'evaluation_only'          # the usage the scorer gives the cells of the fold being scored (cross-validation)
PROTOCOL = SPEC['split']['protocol']    # the default protocol: well-stratified held-out wells
CV_SPLIT = SPEC['split_cvcol4']
CV_PROTOCOL = CV_SPLIT['protocol']
PROTOCOL_USAGES = {PROTOCOL: USAGES, CV_PROTOCOL: USAGES[:2]}   # protocol -> the `usage` values of its image rows
DAYS = {k: int(v) for k, v in SPEC['timepoints'].items()}
CONDITION_OF_ROW = {row: condition for condition, rows in SPEC['conditions'].items() for row in rows}
PHASE_RE = re.compile(SPEC['phase_pattern'])
PLATE = 'p000'
ACQUISITION = 'a000'


def fresh_output(out: Path) -> None:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'output directory exists and is not empty: {out}')


def inventory(source: Path) -> list[dict]:
    """Every (well, time point) image of the source folder with its phase-contrast and GFP files, from the
    per-time-point folders only; a phase image without its GFP file raises."""
    source = Path(source)
    rows = []
    for folder, day in DAYS.items():
        directory = source / folder
        if not directory.is_dir():
            raise FileNotFoundError(directory)
        for name in sorted(os.listdir(directory)):
            match = PHASE_RE.match(name)
            if not match:
                continue
            well, seq = f'{match.group(1)}{match.group(2)}', match.group(3)
            gfp = directory / SPEC['gfp_pattern'].format(well=well, seq=seq)
            if not gfp.is_file():
                raise FileNotFoundError(f'GFP channel missing for {directory / name}')
            rows.append(dict(well=well, condition=CONDITION_OF_ROW[match.group(1)], day=day, timepoint=folder,
                             phase=directory / name, gfp=gfp))
    if len({(r['well'], r['day']) for r in rows}) != len(rows):
        raise ValueError('duplicate (well, day) image in the source folder')
    return rows


def split_wells(wells_by_condition: dict, seed: int) -> dict:
    """{well: usage}: a seeded shuffle within each condition, cut at the contract's exact counts."""
    assignments = {}
    for condition, wells in sorted(wells_by_condition.items()):
        counts = SPEC['split']['counts'][condition]
        if sum(counts.values()) != len(wells):
            raise ValueError(f'{condition}: {len(wells)} complete wells, the contract splits {sum(counts.values())}')
        order = sorted(wells)
        random.Random(f'{seed}|{condition}').shuffle(order)
        start = 0
        for usage in USAGES:
            for well in order[start:start + counts[usage]]:
                assignments[well] = usage
            start += counts[usage]
    return assignments


def split_wells_cv(wells: list[str]) -> dict:
    """{well: usage} of the cross-validation protocol: the contract's SSL validation wells validate, every other
    complete well trains."""
    validation = set(CV_SPLIT['ssl_validation_source_wells'])
    if not validation <= set(wells):
        raise ValueError(f'SSL validation wells {sorted(validation - set(wells))} are not complete wells')
    return {w: ('development_val' if w in validation else 'development_train') for w in wells}


def cv_fold(source_well: str) -> int:
    """The cross-validation fold of a well: its plate column."""
    return int(CV_SPLIT['cv']['fold_of_column'][source_well[1:]])


def pseudonyms(keys: list, prefix: str, width: int, seed: int, salt: str) -> dict:
    order = sorted(keys)
    random.Random(f'{seed}|{salt}').shuffle(order)
    return {key: f'{prefix}{index:0{width}d}' for index, key in enumerate(order)}


def _copy_verified(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    sha = digest(source)
    shutil.copyfile(source, target)
    if digest(target) != sha:
        raise ValueError(f'copy changed pixels: {source}')
    return sha


def _check_image(path: Path) -> np.ndarray:
    array = tifffile.imread(path)
    if array.ndim != 2 or list(array.shape) != SPEC['image_shape'] or str(array.dtype) != SPEC['dtype']:
        raise ValueError(f'{path}: shape {array.shape} dtype {array.dtype}, expected {SPEC["image_shape"]} {SPEC["dtype"]}')
    return array


PHASE_ONLY = ('PhaseContrast',)
TWO_CHANNEL = ('PhaseContrast', 'GFP')
VARIANTS = {PHASE_ONLY: 'phase_only', TWO_CHANNEL: 'ssl_two_channel'}


def prepare(source: Path, out: Path, seed: int | None = None, ssl_channels: tuple = PHASE_ONLY, protocol: str = PROTOCOL) -> dict:
    """Write the prepared study: public phase-contrast images, private GFP and labels, the fixed split.
    `ssl_channels` selects the variant: phase only (the original study), or `('PhaseContrast', 'GFP')`, where the
    public row also carries `raw_file`, a two-channel CYX stack that the crop stage cuts the SSL crops from
    (segmentation and crop decisions still read the phase-contrast image alone; user request 2026-10-25).
    `protocol` selects the split (see the module docstring); the cross-validation protocol was run with phase
    contrast only, so combining it with the two-channel variant is refused."""
    ssl_channels = tuple(ssl_channels)
    if ssl_channels not in VARIANTS:
        raise ValueError(f'unknown SSL channel set {ssl_channels}; one of {list(VARIANTS)}')
    if protocol not in PROTOCOL_USAGES:
        raise ValueError(f'unknown split protocol {protocol}; one of {list(PROTOCOL_USAGES)}')
    cv = protocol == CV_PROTOCOL
    if cv and ssl_channels != PHASE_ONLY:
        raise ValueError('the cross-validation protocol has no two-channel variant')
    variant, two_channel = VARIANTS[ssl_channels], ssl_channels == TWO_CHANNEL
    seed = SPEC['split']['seed'] if seed is None else int(seed)
    source, out = Path(source), Path(out)
    fresh_output(out)
    rows = inventory(source)
    by_well = defaultdict(list)
    for r in rows:
        by_well[r['well']].append(r)
    complete = {w: rs for w, rs in by_well.items() if {r['day'] for r in rs} == set(DAYS.values())}
    excluded = {w: dict(days=sorted(r['day'] for r in rs), reason='incomplete time course')
                for w, rs in by_well.items() if w not in complete}
    if len(complete) != SPEC['expected_wells'] or sum(len(v) for v in complete.values()) != SPEC['expected_images']:
        raise ValueError(f'{len(complete)} complete wells / {sum(len(v) for v in complete.values())} images; '
                         f'the contract expects {SPEC["expected_wells"]} / {SPEC["expected_images"]}')
    unexpected = set(excluded) - set(SPEC['known_incomplete_wells'])
    if unexpected:
        raise ValueError(f'unexpected incomplete wells: {sorted(unexpected)}')
    wells_by_condition = defaultdict(list)
    for w, rs in complete.items():
        wells_by_condition[rs[0]['condition']].append(w)
    if cv:
        assignments = split_wells_cv(list(complete))
        counts = {c: Counter(assignments[w] for w in ws) for c, ws in wells_by_condition.items()}
        if any(dict(counts[c]) != CV_SPLIT['counts'][c] for c in CV_SPLIT['counts']):
            raise ValueError(f'the split differs from the contract counts: {counts}')
    else:
        assignments = split_wells(wells_by_condition, seed)
    well_ids = pseudonyms(list(complete), 'w', 2, seed, 'wells')
    keys = [(r['well'], r['day']) for w in sorted(complete) for r in sorted(complete[w], key=lambda r: r['day'])]
    image_ids = pseudonyms(keys, 'i', 4, seed, 'images')
    out.mkdir(parents=True)
    public, private, baseline_gfp = [], {}, []
    for index, key in enumerate(sorted(keys, key=lambda k: image_ids[k])):
        r = next(x for x in complete[key[0]] if x['day'] == key[1])
        ident, well = image_ids[key], well_ids[key[0]]
        _check_image(r['phase'])
        gfp = _check_image(r['gfp'])
        if r['day'] == 0:
            baseline_gfp.append(gfp.ravel())
        phase_sha = _copy_verified(r['phase'], out / 'public' / 'images' / f'{ident}.tif')
        gfp_sha = _copy_verified(r['gfp'], out / 'private' / 'gfp' / f'{ident}.tif')
        usage = assignments[key[0]]
        row = dict(image_id=ident, file=f'images/{ident}.tif', segment_file=f'images/{ident}.tif',
                   sha256=phase_sha, segment_sha256=phase_sha, plate=PLATE, plate_id=PLATE, well=well,
                   site=ident, acquisition_id=ACQUISITION, experiment_date=ACQUISITION, usage=usage,
                   partition='held_out' if usage == USAGES[2] else 'seen', shape=[1, *SPEC['image_shape']],
                   dtype=SPEC['dtype'], calibration_group='', negative_control=False)
        if two_channel:     # the SSL stack: phase contrast and GFP, pixel-identical to the two source channels
            raw_path = out / 'public' / 'raw' / f'{ident}.tif'
            raw_path.parent.mkdir(parents=True, exist_ok=True)
            stack = np.stack([tifffile.imread(r['phase']), gfp]).astype(np.uint16)
            tifffile.imwrite(raw_path, stack, metadata={'axes': 'CYX', 'Channel': {'Name': list(TWO_CHANNEL)}})
            row.update(raw_file=f'raw/{ident}.tif', raw_sha256=digest(raw_path), shape=[2, *SPEC['image_shape']])
        public.append(row)
        private[ident] = dict(well=well, source_well=key[0], condition=r['condition'], day=r['day'],
                              timepoint=r['timepoint'], usage=usage, gfp_file=f'gfp/{ident}.tif', gfp_sha256=gfp_sha,
                              phase_source=str(r['phase']), gfp_source=str(r['gfp']), phase_sha256=phase_sha)
        if cv:
            private[ident]['cv_fold'] = cv_fold(key[0])
        if (index + 1) % 20 == 0:
            print(f'PREPARE {index + 1}/{len(keys)}', flush=True)
    threshold = float(np.percentile(np.concatenate(baseline_gfp), 99))
    write_json(out / 'public' / 'images.json', public)
    write_json(out / 'private' / 'images.json', private)
    if cv:
        splits = dict(
            protocol=protocol, seed=seed, unit='well', counts=dict(Counter(assignments.values())), cv=CV_SPLIT['cv'],
            cv_fold_sizes=dict(sorted(Counter(cv_fold(w) for w in complete).items())),
            wells={well_ids[w]: dict(usage=u, condition=complete[w][0]['condition'], cv_fold=cv_fold(w))
                   for w, u in assignments.items()})
    else:
        splits = dict(
            protocol=protocol, seed=seed, unit='well', counts=dict(Counter(assignments.values())),
            wells={well_ids[w]: dict(usage=u, condition=complete[w][0]['condition']) for w, u in assignments.items()},
            stratified_by='condition')
    write_json(out / 'private' / 'splits.json', splits)
    write_json(out / 'private' / 'gfp_threshold.json', dict(
        gfp_positive_threshold=threshold, rule=SPEC['gfp_positive_threshold'], n_baseline_images=len(baseline_gfp)))
    record = dict(status='complete', dataset=SPEC['dataset'], protocol=protocol, seed=seed, images=len(public),
                  wells=len(complete), wells_by_condition={c: len(w) for c, w in sorted(wells_by_condition.items())},
                  excluded_wells=excluded, images_sha256=digest(out / 'public' / 'images.json'),
                  private_images_sha256=digest(out / 'private' / 'images.json'),
                  splits_sha256=digest(out / 'private' / 'splits.json'),
                  gfp_threshold_sha256=digest(out / 'private' / 'gfp_threshold.json'),
                  variant=variant, ssl_channels=list(ssl_channels),
                  stage_channels=dict(segment=list(PHASE_ONLY), crop=list(PHASE_ONLY), embed=list(ssl_channels)),
                  prediction_channels=sorted(set(PHASE_ONLY) | set(ssl_channels), key=TWO_CHANNEL.index),
                  evaluation_channels=SPEC['evaluation_channels'],
                  source=str(source.resolve()), image_shape=SPEC['image_shape'], dtype=SPEC['dtype'],
                  identifier_policy='image, well and plate identifiers are seeded pseudonyms; the mapping is private')
    write_json(out / 'PREPARE.json', record)
    return record


def validate_rows(rows: list[dict], usages: tuple = USAGES) -> None:
    if not rows or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate image IDs')
    wells = defaultdict(list)
    two_channel = {('raw_file' in r) for r in rows}
    if len(two_channel) != 1:
        raise ValueError('rows mix the phase-only and the two-channel variant')
    for r in rows:
        if r['usage'] not in usages or r['file'] != r['segment_file'] or r['sha256'] != r['segment_sha256']:
            raise ValueError('invalid split or non-phase-contrast segmentation input')
        if 'raw_file' in r and (r['shape'][0] != 2 or not r.get('raw_sha256')):
            raise ValueError('two-channel row without a two-channel raw stack')
        wells[r['well']].append(r)
    for group in wells.values():
        if len({r['usage'] for r in group}) != 1:
            raise ValueError('well crosses splits')
    if {r['usage'] for r in rows} != set(usages):
        raise ValueError('missing split')


def load_prepared(prep: Path, *, verify_pixels: bool = False) -> list[dict]:
    prep = Path(prep)
    record = read_json(prep / 'PREPARE.json')
    path = prep / 'public' / 'images.json'
    if record.get('protocol') not in PROTOCOL_USAGES or record.get('status') != 'complete' or digest(path) != record['images_sha256']:
        raise ValueError('wrong or changed time-course preparation')
    rows = read_json(path)
    validate_rows(rows, PROTOCOL_USAGES[record['protocol']])
    if verify_pixels:
        for r in rows:
            if digest(within(prep / 'public', r['file'])) != r['sha256']:
                raise ValueError(f'phase-contrast pixels changed: {r["image_id"]}')
            if 'raw_file' in r and digest(within(prep / 'public', r['raw_file'])) != r['raw_sha256']:
                raise ValueError(f'two-channel stack changed: {r["image_id"]}')
    return rows


def protocol_of(prep: Path) -> str:
    """The split protocol of the prepared data."""
    return read_json(Path(prep) / 'PREPARE.json')['protocol']


def variant(prep: Path) -> dict:
    """The prepared data's variant record: `variant`, `ssl_channels`, `stage_channels` (phase-only runs prepared
    before the variants existed read as phase_only)."""
    record = read_json(Path(prep) / 'PREPARE.json')
    return dict(variant=record.get('variant', VARIANTS[PHASE_ONLY]), ssl_channels=record.get('ssl_channels', list(PHASE_ONLY)),
                stage_channels=record.get('stage_channels', dict(segment=list(PHASE_ONLY), crop=list(PHASE_ONLY), embed=list(PHASE_ONLY))))


def private_images(prep: Path) -> dict:
    prep = Path(prep)
    record = read_json(prep / 'PREPARE.json')
    path = prep / 'private' / 'images.json'
    if digest(path) != record['private_images_sha256']:
        raise ValueError('private label table changed')
    return read_json(path)


def gfp_threshold(prep: Path) -> float:
    prep = Path(prep)
    record = read_json(prep / 'PREPARE.json')
    path = prep / 'private' / 'gfp_threshold.json'
    if digest(path) != record['gfp_threshold_sha256']:
        raise ValueError('GFP threshold record changed')
    return float(read_json(path)['gfp_positive_threshold'])


def smoke_images(rows: list[dict], private: dict) -> list[dict]:
    """The technical smoke's images: `_smoke_images_cv` for the cross-validation protocol, else `_smoke_images_heldout`."""
    return (_smoke_images_cv if any('cv_fold' in v for v in private.values()) else _smoke_images_heldout)(rows, private)


def _smoke_images_heldout(rows: list[dict], private: dict) -> list[dict]:
    """Six images for a technical smoke: two training wells (one per condition) at day 0 and 16, one validation
    well and one held-out well at day 8. Deterministic (first pseudonymous well of each kind)."""
    chosen = set()
    train = defaultdict(list)
    for r in rows:
        if r['usage'] == 'development_train':
            train[private[r['image_id']]['condition']].append(r)
    for condition, group in sorted(train.items()):
        well = min(r['well'] for r in group)
        chosen.update(r['image_id'] for r in group if r['well'] == well and private[r['image_id']]['day'] in (0, 16))
    for usage in ('development_val', 'evaluation_only'):
        group = [r for r in rows if r['usage'] == usage]
        well = min(r['well'] for r in group)
        chosen.update(r['image_id'] for r in group if r['well'] == well and private[r['image_id']]['day'] == 8)
    selected = [r for r in rows if r['image_id'] in chosen]
    if len(selected) != 6:
        raise ValueError(f'smoke selection has {len(selected)} images, expected 6')
    return selected


def _smoke_images_cv(rows: list[dict], private: dict) -> list[dict]:
    """Thirty-two images for a technical smoke of the cross-validation protocol: one well per cross-validation fold
    (so every outer fold of the scorer has a test, a validation and training wells), two conditions among them, every
    time point (enough images for one segmentation cohort group, MIN_GROUP = 10). The SSL validation well is the one
    of fold 1. Deterministic (first pseudonymous well of each kind)."""
    by_well = defaultdict(list)
    for r in rows:
        by_well[r['well']].append(r)
    fold_of = {w: private[rs[0]['image_id']]['cv_fold'] for w, rs in by_well.items()}
    wanted = [(1, 'development_val', 'FC'), (2, 'development_train', 'Control'),
              (3, 'development_train', 'FC'), (4, 'development_train', 'Control')]
    chosen = set()
    for fold, usage, condition in wanted:
        wells = sorted(w for w, rs in by_well.items() if fold_of[w] == fold and rs[0]['usage'] == usage
                       and private[rs[0]['image_id']]['condition'] == condition)
        if not wells:
            raise ValueError(f'no {condition} {usage} well in fold {fold}')
        chosen.update(r['image_id'] for r in by_well[wells[0]])
    selected = [r for r in rows if r['image_id'] in chosen]
    if len(selected) != 32:
        raise ValueError(f'smoke selection has {len(selected)} images, expected 32')
    return selected

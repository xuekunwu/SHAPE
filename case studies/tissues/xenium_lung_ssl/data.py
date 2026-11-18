"""Prepare the Xenium lung cancer SSL study: a seeded 10 percent sample of the section's single-cell crops, split by
500 um spatial tiles into SSL training, SSL validation and held-out cells (4 : 1 : 5). Each crop is its own image and
site; a well is the crops of one 250 um neighborhood nested in a split tile, so SHAPE's replicate-tier proxy (sibling
sites of a well) is defined. The coordinates and the tile layout are private.

Protocol xenium-lung-ssl-sample10-tile500-well250-415-v2: held-out tiles never enter an SSL call; every sampled crop
is embedded after the fit."""
from __future__ import annotations

from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import random
import shutil

import tifffile

from benchmarks.full_CPJUMP1.harness.data import digest, read_json, write_json, within

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
SPEC = json.loads((HERE / 'dataset.json').read_text(encoding='utf-8'))
PROTOCOL = SPEC['split']['protocol']
USAGES = tuple(SPEC['split']['ratio'])          # development_train, development_val, evaluation_only
PLATE = 'p000'


def fresh_output(out: Path) -> None:
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        raise FileExistsError(f'output directory exists and is not empty: {out}')


def sample_cells(cell_ids: list[str], fraction: float, seed: int) -> list[str]:
    ids = sorted(cell_ids)
    return sorted(random.Random(seed).sample(ids, round(fraction * len(ids))))


def tile_of(x: float, y: float, tile_um: float) -> str:
    return f'{math.floor(x / tile_um)}_{math.floor(y / tile_um)}'


def split_tiles(tile_of_cell: dict, ratio: dict, seed: int) -> dict:
    """{tile: usage}: the tiles in a seeded order; a tile joins the first usage whose cumulative cell target the cells
    of the tiles before it have not reached, so whole tiles fill train, then val, then held-out."""
    sizes = Counter(tile_of_cell.values())
    order = sorted(sizes)
    random.Random(f'{seed}|tiles').shuffle(order)
    total, weight = sum(sizes.values()), sum(ratio.values())
    bounds, running = [], 0
    for usage, share in ratio.items():
        running += share
        bounds.append((usage, total * running / weight))
    assignment, before = {}, 0
    for tile in order:
        assignment[tile] = next(u for u, bound in bounds if before < bound)
        before += sizes[tile]
    return assignment


def _copy_verified(source: Path, target: Path) -> str:
    target.parent.mkdir(parents=True, exist_ok=True)
    sha = digest(source)
    shutil.copyfile(source, target)
    if digest(target) != sha:
        raise ValueError(f'copy changed pixels: {source}')
    return sha


def _check_crop(path: Path) -> tuple:
    array = tifffile.imread(path)
    side = SPEC['min_crop_side']
    if (array.ndim != 3 or array.shape[0] != len(SPEC['channels']) or str(array.dtype) != SPEC['dtype']
            or min(array.shape[1:]) < side):
        raise ValueError(f'{path}: shape {array.shape} dtype {array.dtype}, expected {len(SPEC["channels"])} x H x W '
                         f'{SPEC["dtype"]} with H, W >= {side}')
    return tuple(int(v) for v in array.shape)


def prepare(crops: Path, cells_parquet: Path, out: Path, seed: int | None = None) -> dict:
    """Write the prepared study: the sampled crops (copied byte for byte), the public table and the private
    coordinates and tile split."""
    import pandas as pd
    seed = SPEC['sample']['seed'] if seed is None else int(seed)
    crops, cells_parquet, out = Path(crops), Path(cells_parquet), Path(out)
    fresh_output(out)
    table = pd.read_parquet(cells_parquet).set_index('cell_id')
    if len(table) != SPEC['expected_cells'] or not table.index.is_unique:
        raise ValueError(f'{len(table)} cells, expected {SPEC["expected_cells"]} unique cell IDs')
    files = {p.name for p in crops.iterdir()}
    expected = {SPEC['crop_pattern'].format(cell_id=c) for c in table.index}
    if files != expected:
        raise ValueError(f'crop folder and cells.parquet differ: {len(files - expected)} extra crops, '
                         f'{len(expected - files)} missing crops')
    sampled = sample_cells(list(table.index), SPEC['sample']['fraction'], seed)
    tile_um, well_um = SPEC['split']['tile_um'], SPEC['split']['well_um']
    if tile_um % well_um:
        raise ValueError('a well must nest in one split tile')
    tiles = {c: tile_of(table.at[c, 'x_centroid'], table.at[c, 'y_centroid'], tile_um) for c in sampled}
    wells = {c: 'w' + tile_of(table.at[c, 'x_centroid'], table.at[c, 'y_centroid'], well_um) for c in sampled}
    usage_of_tile = split_tiles(tiles, SPEC['split']['ratio'], SPEC['split']['seed'])
    out.mkdir(parents=True)
    public, private = [], {}
    for index, cell in enumerate(sampled):
        name = SPEC['crop_pattern'].format(cell_id=cell)
        source = crops / name
        shape = _check_crop(source)
        sha = _copy_verified(source, out / 'public' / 'crops' / name)
        usage = usage_of_tile[tiles[cell]]
        row = table.loc[cell]
        public.append(dict(image_id=cell, file=f'crops/{name}', sha256=sha, plate=PLATE, plate_id=PLATE, well=wells[cell],
                           site=cell, usage=usage, shape=list(shape), dtype=SPEC['dtype'],
                           area=float(row['cell_area']), calibration_group='', negative_control=False))
        private[cell] = dict(x_centroid=float(row['x_centroid']), y_centroid=float(row['y_centroid']),
                             tile=tiles[cell], well=wells[cell], usage=usage, cell_area=float(row['cell_area']),
                             nucleus_area=float(row['nucleus_area']), transcript_counts=int(row['transcript_counts']),
                             source=str(source), sha256=sha)
        if (index + 1) % 2000 == 0:
            print(f'PREPARE {index + 1}/{len(sampled)}', flush=True)
    counts = Counter(r['usage'] for r in public)
    tile_counts = Counter(usage_of_tile.values())
    well_counts = Counter(r['usage'] for r in {r['well']: r for r in public}.values())
    write_json(out / 'public' / 'images.json', public)
    write_json(out / 'private' / 'cells.json', private)
    write_json(out / 'private' / 'splits.json', dict(
        protocol=PROTOCOL, seed=SPEC['split']['seed'], unit='spatial tile', tile_um=tile_um, well_um=well_um,
        ratio=SPEC['split']['ratio'], cells=dict(counts), tiles=dict(tile_counts), usage_of_tile=usage_of_tile))
    record = dict(status='complete', dataset=SPEC['dataset'], protocol=PROTOCOL, sample_seed=seed,
                  split_seed=SPEC['split']['seed'], cells_in_section=len(table), images=len(public),
                  cells_by_usage=dict(counts), tiles_by_usage=dict(tile_counts), wells_by_usage=dict(well_counts),
                  images_sha256=digest(out / 'public' / 'images.json'),
                  private_cells_sha256=digest(out / 'private' / 'cells.json'),
                  splits_sha256=digest(out / 'private' / 'splits.json'),
                  cells_parquet=str(cells_parquet.resolve()), cells_parquet_sha256=digest(cells_parquet),
                  crops=str(crops.resolve()), channels=SPEC['channels'], dtype=SPEC['dtype'])
    write_json(out / 'PREPARE.json', record)
    return record


def validate_rows(rows: list[dict]) -> None:
    if not rows or len({r['image_id'] for r in rows}) != len(rows):
        raise ValueError('empty or duplicate image IDs')
    if any(r['usage'] not in USAGES or r['site'] != r['image_id'] for r in rows):
        raise ValueError('invalid usage or a site that is not its crop')
    usages_of = {}
    for r in rows:
        usages_of.setdefault(r['well'], set()).add(r['usage'])
    if any(len(u) > 1 for u in usages_of.values()):
        raise ValueError('well crosses splits')
    if not {'development_train', 'development_val'} <= {r['usage'] for r in rows}:
        raise ValueError('missing SSL split')


def load_prepared(prep: Path, *, verify_pixels: bool = False) -> list[dict]:
    prep = Path(prep)
    record = read_json(prep / 'PREPARE.json')
    path = prep / 'public' / 'images.json'
    if record.get('protocol') != PROTOCOL or record.get('status') != 'complete' or digest(path) != record['images_sha256']:
        raise ValueError('wrong or changed Xenium preparation')
    rows = read_json(path)
    validate_rows(rows)
    if verify_pixels:
        for r in rows:
            if digest(within(prep / 'public', r['file'])) != r['sha256']:
                raise ValueError(f'crop pixels changed: {r["image_id"]}')
    return rows


def cells(prep: Path, rows: list[dict]) -> list[dict]:
    """The embed stage's cells.json: one record per crop, the prepared copy as its input."""
    return [dict(cell_id=r['image_id'], crop_path=str(within(Path(prep) / 'public', r['file']).resolve()),
                 area=r['area'], sha256=r['sha256'], image_id=r['image_id'], plate=r['plate'], well=r['well'],
                 usage=r['usage']) for r in rows]


def smoke_images(rows: list[dict]) -> list[dict]:
    """The technical smoke's crops: per usage, whole wells of at least two crops in sha256 order of the well ID until
    the SPEC smoke count is reached (at least two wells per usage, so the val split has MIN_VAL_WELLS wells)."""
    members = {}
    for r in rows:
        members.setdefault((r['usage'], r['well']), []).append(r['image_id'])
    chosen = set()
    for usage, n in SPEC['smoke'].items():
        if usage == 'rule':
            continue
        wells = sorted((w for (u, w), ids in members.items() if u == usage and len(ids) >= 2),
                       key=lambda w: hashlib.sha256(w.encode()).hexdigest())
        taken, count = 0, 0
        for well in wells:
            if count >= n and taken >= 2:
                break
            chosen.update(members[(usage, well)])
            taken, count = taken + 1, count + len(members[(usage, well)])
        if count < n or taken < 2:
            raise ValueError(f'{count} {usage} crops in {taken} wells, the smoke needs {n} in two or more wells')
    return [r for r in rows if r['image_id'] in chosen]

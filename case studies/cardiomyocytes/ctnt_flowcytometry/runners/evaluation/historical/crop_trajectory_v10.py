"""Re-select measured v10 crops post hoc; no LLM, new search, or TIFF delivery."""
import argparse
from collections import Counter
from copy import deepcopy
import gzip
import hashlib
import json
import math
import os
from pathlib import Path
import socket
import shutil
import sys
import time

from shape.agent.verification.crop_trajectory import CropTrajectory, POLICIES, EPS, observation
from shape.agent.verification.verifiers.crop import _edge_limited


def read(path):return json.loads(Path(path).read_text())
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def write(path, value):Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')


def decode_rows(rows):
    result = {(i, c): row for i, c, row in rows}
    if len(result) != len(rows):
        raise ValueError('Duplicate checkpoint crop identity')
    return result


def payload_sha(payload):
    data = json.dumps(payload, sort_keys=True, allow_nan=False, separators=(',', ':')).encode()
    return hashlib.sha256(data).hexdigest()


RUNNER = 'case studies/cardiomyocytes/ctnt_flowcytometry/runners/evaluation/historical/crop_trajectory_v10.py'
OVERLAY = (RUNNER, 'shape/agent/verification/crop_trajectory.py',
           'shape/agent/verification/test/test_crop_trajectory.py')


def prepare_output(run, out):
    """Overlay analysis only on the completed run's exact scientific source."""
    run, out = Path(run).resolve(), Path(out).resolve()
    if out.exists():
        if (out / 'SUMMARY.json').exists() or not (out / 'SOURCE_MANIFEST.json').is_file():
            raise ValueError('Use a fresh output directory or a prepared sealed analysis directory')
        return
    previous = read(run / 'SOURCE_MANIFEST.json')
    digest = sha(run / 'SOURCE_MANIFEST.json')
    assert digest == read(run / 'FINAL_SUMMARY.json')['source_manifest_sha256']
    assert digest == read(run / 'LAUNCH_PROVENANCE.json')['source_manifest_sha256']
    assert read(run / 'RUN_META.json')['status'] == 'COMPLETED'
    for name, expected in previous['files'].items():
        assert sha(run / 'source_v10' / name) == expected, name
    repo = Path(__file__).resolve().parents[6]
    out.mkdir(parents=True, exist_ok=False)
    shutil.copytree(run / 'source_v10', out / 'source')
    overlay = {}
    for name in OVERLAY:
        target = out / 'source' / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copyfile(repo / name, target)
        overlay[name] = sha(target)
    write(out / 'SOURCE_MANIFEST.json', dict(
        protocol='authorized-uncommitted-source-snapshot',
        branch=previous['branch'], base_commit=previous['base_commit'],
        revision='v10_best_observed_posthoc_v1', previous_manifest_sha256=digest,
        files={**previous['files'], **overlay}))


def main(run, out):
    import tifffile
    started = time.time();run = Path(run);out = Path(out)
    assert not (out / 'SUMMARY.json').exists()
    source = out / 'source';manifest = read(out / 'SOURCE_MANIFEST.json')
    for name, digest in manifest['files'].items():assert sha(source / name) == digest, name
    final = read(run / 'FINAL_SUMMARY.json');launch = read(run / 'LAUNCH_PROVENANCE.json')
    assert read(run / 'RUN_META.json')['status'] == 'COMPLETED'
    assert final['policy'] == 'native_guarded_frozen_conditional_global8_vrr_v10'
    assert final['source_manifest_sha256'] == launch['source_manifest_sha256'] == sha(run / 'SOURCE_MANIFEST.json')
    assert manifest['previous_manifest_sha256'] == final['source_manifest_sha256']
    assert final['objects'] == final['delivered'] == 28958
    envelope = read(run / 'NATIVE_SEARCH_CHECKPOINT.json');p = envelope['payload']
    assert envelope['sha256'] == payload_sha(p) and p['version'] == 3
    assert p['policy'] == final['policy'] and p['history'] == final['rounds']
    chunks = run / 'NATIVE_SEARCH_CHECKPOINT.chunks';verified = {}
    def chunk(digest):
        path = chunks / (digest + '.json');assert sha(path) == digest
        verified[digest] = path.stat().st_size
        return read(path)
    decode = decode_rows
    initial = decode(chunk(p['fixed']['initial']));initial_pixels = decode(chunk(p['fixed']['initial_pixels']))
    initial_overrides = chunk(p['fixed']['initial_overrides'])
    current_state = chunk(p['current']);current = decode(current_state['rows'])
    assert initial.keys() == current.keys() == initial_pixels.keys() and len(initial) == 28958
    # Actual effective global parameters, including basename-dependent defaults.
    delivered = read(run / 'SELECTED_GEOMETRY.json')['images']
    base = {i: {k: v for k, v in value['parameters'].items() if k != 'cell_overrides'}
            for i, value in delivered.items()}
    del delivered
    parameters = {k: {**base[k[0]], **initial_overrides[k[0]].get(k[1], {})} for k in initial}
    shapes = {}
    for i in base:
        with tifffile.TiffFile(run / 'inputs/images' / (i + '.tif')) as t:
            shapes[i] = tuple(n for n in t.series[0].shape if n != 1)
        assert len(shapes[i]) == 2
    cache = {}
    for digest in p['pixel_chunks']:
        for i, c, box, pixels in chunk(digest):
            key = i, c, tuple(box);assert key not in cache;cache[key] = pixels
    def pixels(rows):return {k: cache[(*k, tuple(r['crop_bbox']))] for k, r in rows.items()}
    assert pixels(initial) == initial_pixels
    def validate_bounds(rows):
        for (i, c), row in rows.items():
            y, x, h, w = row['crop_bbox'];cy, cx = row['centroid'];height, width = shapes[i]
            assert 0 <= y < h <= height and 0 <= x < w <= width
            assert y <= cy <= h and x <= cx <= w
    validate_bounds(initial);validate_bounds(current)
    trace = CropTrajectory(initial, initial_pixels, parameters)
    actual = {k: observation('actual_v10', 8, r, pixels({k: r})[k],
              {**base[k[0]], **current_state['overrides'][k[0]].get(k[1], {})}) for k, r in current.items()}
    def warning(k, option):
        r = option['row'];q = r['qc'];flags = []
        if q['target_retention'] < .95:flags.append('retention')
        if q['neighbor_contamination'] > .20:flags.append('neighbor')
        if q['centroid_offset'] > .20 and not _edge_limited(r['crop_bbox'], r['centroid'], shapes[k[0]]):flags.append('offset')
        return flags
    initial_good = {k for k in initial if not warning(k, trace.history[k][0])}
    def summary(selected):
        n = len(selected);assert n == 28958
        mean = lambda fn: math.fsum(fn(k, o) for k, o in selected.items()) / n
        quality = mean(lambda k, o: o['pixels']['crop_quality_score'])
        native = {metric: mean(lambda k, o: o['row']['qc'][metric]) for metric in
                  ('target_retention', 'neighbor_contamination', 'centroid_offset')}
        flags = {k: warning(k, o) for k, o in selected.items()}
        return dict(objects=n, pixel_quality=quality, **native,
            pixel_components={metric: mean(lambda k, o: o['pixels'][metric]) for metric in initial_pixels[next(iter(initial))]},
            changed=sum(o['row']['crop_bbox'] != initial[k]['crop_bbox'] for k, o in selected.items()),
            pixel_improved=sum(o['pixels']['crop_quality_score'] > initial_pixels[k]['crop_quality_score'] + EPS for k, o in selected.items()),
            pixel_worsened=sum(o['pixels']['crop_quality_score'] < initial_pixels[k]['crop_quality_score'] - EPS for k, o in selected.items()),
            retention_warning_objects=sum(o['row']['qc']['target_retention'] < .95 for o in selected.values()),
            min_retention=min(o['row']['qc']['target_retention'] for o in selected.values()),
            retention_loss_over_5pp=sum(initial[k]['qc']['target_retention'] - o['row']['qc']['target_retention'] > .05 + EPS for k, o in selected.items()),
            warning_objects=sum(bool(v) for v in flags.values()),
            strict_good=sum(not flags[k] and o['pixels']['crop_quality_score'] >= initial_pixels[k]['crop_quality_score'] - EPS for k, o in selected.items()),
            native_worsened={metric: sum((o['row']['qc'][metric] - initial[k]['qc'][metric]) * sign < -EPS
                for k, o in selected.items()) for metric, sign in [('target_retention', 1), ('neighbor_contamination', -1), ('centroid_offset', -1)]},
            chosen_from_round=dict(sorted(Counter(o['observed_round'] for o in selected.values()).items())),
            chosen_from_candidate=dict(sorted(Counter(o['candidate'] for o in selected.values()).items())))
    snapshots = [dict(round=0, candidates=[], policies={name: summary(values) for name, values in trace.best.items()})]
    candidate_hashes = dict(p['candidates']);origins = {};candidate_table = []
    for decision in p['history']:
        previous_best = {name: dict(values) for name, values in trace.best.items()}
        updated = {name: 0 for name in POLICIES};measured = []
        for proposal in decision.get('proposed', []):
            name = proposal['id']
            if name == 'current' or name in origins:continue
            record = chunk(candidate_hashes[name]);assert record['id'] == name
            rows = decode(record['rows']);overrides = decode(record['overrides'])
            assert not initial_good & rows.keys();validate_bounds(rows)
            params = {k: {**base[k[0]], **overrides[k]} for k in rows}
            changes = trace.add_candidate(name, decision['round'], rows, pixels(rows), params)
            for policy in POLICIES:updated[policy] += changes[policy]
            origins[name] = dict(round=decision['round'], chunk_sha256=candidate_hashes[name],
                                 whole_policy_allowed=record['loss_envelope']['allowed'])
            measured.append(name)
            candidate_table.append(dict(candidate=name, measured_round=decision['round'],
                whole_policy_allowed=record['loss_envelope']['allowed'],
                targeted_objects=len(rows), native_valid_objects=len(rows),
                score_improved_vs_initial=sum(cache[(*k, tuple(r['crop_bbox']))]['crop_quality_score'] > initial_pixels[k]['crop_quality_score'] + EPS for k, r in rows.items()),
                history_best_updates=changes))
        snapshots.append(dict(round=decision['round'], candidates=measured,
            history_best_update_events=updated,
            history_best_updates_unique={name: sum(o is not previous_best[name][k] for k, o in values.items())
                                         for name, values in trace.best.items()},
            actual_selected=decision.get('selected', []),
            policies={name: summary(values) for name, values in trace.best.items()}))
        write(out / 'PROGRESS.json', dict(stage='replay', round=decision['round'], total_rounds=8,
              candidates=len(origins), objects=len(initial), api_calls=0, updated_at=time.time()))
    assert set(origins) == set(candidate_hashes) and len(origins) == 11
    assert all(all(trace.best[name][k]['row'] == initial[k] for k in initial_good) for name in POLICIES)
    reconstructed = {k: values[0] for k, values in trace.history.items()}
    for decision in p['history']:
        for name in decision.get('selected', []):
            for k, values in trace.history.items():
                for o in values:
                    if o['candidate'] == name:reconstructed[k] = o
    assert all(reconstructed[k]['row'] == current[k] and
               reconstructed[k]['parameters'] == actual[k]['parameters'] for k in initial)
    actual = reconstructed
    baseline = summary({k: values[0] for k, values in trace.history.items()});actual_summary = summary(actual)
    assert math.isclose(actual_summary['pixel_quality'], final['initial_vs_final']['pixel_mean_after']['crop_quality_score'], abs_tol=1e-10)
    # Export one complete trajectory per crop, not one file per cell.
    with gzip.open(out / 'PER_CROP_TRAJECTORIES.jsonl.gz', 'wt', encoding='utf-8', compresslevel=3) as f:
        for k, values in sorted(trace.history.items()):
            positions = {id(o): j for j, o in enumerate(values)}
            f.write(json.dumps(dict(image_id=k[0], cell_id=k[1], initially_good=k in initial_good,
                observations=[dict(candidate=o['candidate'], measured_round=o['observed_round'],
                    crop_bbox=o['row']['crop_bbox'], native={m: o['row']['qc'][m] for m in
                    ('target_retention', 'neighbor_contamination', 'centroid_offset')},
                    pixels=o['pixels'], parameters=o['parameters'], routing_warnings=warning(k, o)) for o in values],
                actual_final_box=current[k]['crop_bbox'],
                chosen={name: positions[id(selected[k])] for name, selected in trace.best.items()}),
                separators=(',', ':')) + '\n')
    for name, selected in trace.best.items():
        with gzip.open(out / ('SELECTED_' + name + '.jsonl.gz'), 'wt', encoding='utf-8', compresslevel=3) as f:
            for k, o in sorted(selected.items()):
                f.write(json.dumps(dict(image_id=k[0], cell_id=k[1], **o), separators=(',', ':')) + '\n')
    # Independent source-pixel/native measurement check on fixed density-spaced
    # fields plus largest score gains and target losses. No model or TIFF writes.
    import numpy as np
    from skimage.measure import regionprops
    from shape.tools.single_cell_cropper.quant import crop_quality_score
    from shape.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
    inputs = read(run / 'inputs/MANIFEST.json');input_index = {e['relative']: e for e in inputs['entries']}
    counts = Counter(k[0] for k in initial);images = sorted(counts, key=lambda i: (counts[i], i))
    sample_keys = set()
    for i in (images[0], images[len(images) // 2], images[-1]):
        keys = sorted((k for k in initial if k[0] == i), key=lambda k: (initial[k]['area'], k))
        sample_keys.update((keys[0], keys[len(keys) // 2], keys[-1]))
    for selected in trace.best.values():
        sample_keys.update(sorted(selected, key=lambda k: (-(selected[k]['pixels']['crop_quality_score'] - initial_pixels[k]['crop_quality_score']), k))[:3])
        sample_keys.update(sorted(selected, key=lambda k: (selected[k]['row']['qc']['target_retention'], k))[:3])
    checked = [];tool = Single_Cell_Cropper_Tool()
    for i in sorted({k[0] for k in sample_keys}):
        for kind in ('images', 'masks'):
            relative = kind + '/' + i + '.tif';assert sha(run / 'inputs' / relative) == input_index[relative]['sha256']
        raw = np.squeeze(tifffile.imread(run / 'inputs/images' / (i + '.tif')))
        labels = np.squeeze(tifffile.imread(run / 'inputs/masks' / (i + '.tif')))
        regions = regionprops(labels);areas = np.bincount(labels.ravel())
        for k in sorted(key for key in sample_keys if key[0] == i):
            index = int(k[1].rsplit('_cell_', 1)[1]);region = regions[index]
            assert list(region.bbox) == initial[k]['original_bbox'] and float(region.area) == initial[k]['area']
            target = (labels == region.label).astype('uint8') * 255
            for name, selected in trace.best.items():
                o = selected[k];r = o['row'];y, x, h, w = r['crop_bbox']
                computed = crop_quality_score(raw[y:h, x:w], labels[y:h, x:w] == region.label)
                assert all(math.isclose(computed[m], o['pixels'][m], rel_tol=0, abs_tol=1e-12) for m in computed)
                args = {m: o['parameters'][m] for m in ('min_area', 'margin', 'crop_size', 'margin_frac', 'margin_mode', 'tight_buffer_px', 'recenter')}
                plan = tool.cell_crop_box(region, labels, target, labels.shape, o['parameters'],
                    idx=index, full_field_areas=areas, **args)
                assert list(plan['box']) == r['crop_bbox'], (k, name, plan['box'], r['crop_bbox'])
                target_count = int(np.count_nonzero(labels[y:h, x:w] == region.label));fg = int(np.count_nonzero(labels[y:h, x:w]))
                cy, cx = region.centroid
                q = dict(target_retention=target_count / float(region.area),
                    neighbor_contamination=(fg - target_count) / (fg + 1e-9),
                    centroid_offset=float(np.hypot(cy - (y + h) / 2., cx - (x + w) / 2.)) / (.5 * (h - y) + 1e-9))
                assert all(math.isclose(q[m], r['qc'][m], rel_tol=0, abs_tol=1e-9) for m in q)
                checked.append(dict(image_id=i, cell_id=k[1], policy=name, candidate=o['candidate']))
    summaries = {name: summary(values) for name, values in trace.best.items()}
    actual_adopted = {name for decision in p['history'] for name in decision.get('selected', [])}
    for name, values in trace.best.items():
        assert all(o['pixels']['crop_quality_score'] >= initial_pixels[k]['crop_quality_score'] - EPS for k, o in values.items())
        assert all(o['pixels']['crop_quality_score'] >= actual[k]['pixels']['crop_quality_score'] - EPS for k, o in values.items())
        summaries[name].update(
            pixel_improved_vs_actual=sum(o['pixels']['crop_quality_score'] > actual[k]['pixels']['crop_quality_score'] + EPS for k, o in values.items()),
            selected_from_whole_policy_veto=sum(o['candidate'] in origins and not origins[o['candidate']]['whole_policy_allowed'] for o in values.values()),
            selected_from_never_adopted_candidate=sum(o['candidate'] != 'initial' and o['candidate'] not in actual_adopted for o in values.values()))
    result = dict(status='COMPLETED_AND_VERIFIED',analysis='v10_best_observed_posthoc_v1',
        source_manifest_sha256=sha(out / 'SOURCE_MANIFEST.json'),parent_source_sha256=final['source_manifest_sha256'],
        parent_checkpoint_sha256=sha(run / 'NATIVE_SEARCH_CHECKPOINT.json'),input_bundle_sha256=final['input_bundle_sha256'],
        scope=dict(images=114, objects=28958, candidates=11, observed_model_replies=8),
        baseline=baseline, actual_v10=actual_summary, selected=summaries, rounds=snapshots,
        candidate_table=candidate_table, source_pixel_and_native_checks=len(checked),
        verified_chunks=verified, additional_api_calls=0, additional_model_tokens=0,
        scientific_search_rerun=False, raw_tiffs_written=0, seconds=time.time()-started,
        policies=POLICIES, selection='Highest strictly improved existing pixel proxy; exact ties retain earliest; native feasibility retained; warning thresholds do not filter candidates.',
        interpretation='Post-hoc recombination of a fixed adaptive trajectory, not a new online VRR result or independent biological/prediction validation. Quality-only selection has construction-driven nonregression of its own selected proxy. No optimality claim.',
        job=dict(job_id=os.environ.get('SLURM_JOB_ID'),step_id=os.environ.get('SLURM_STEP_ID'),host=socket.gethostname()))
    write(out / 'SOURCE_CHECKS.json', checked);write(out / 'SUMMARY.json', result)
    artifacts = {p.name: sha(p) for p in out.iterdir() if p.is_file() and p.name != 'ARTIFACTS.json'}
    write(out / 'ARTIFACTS.json', artifacts)
    print('VERIFIED_BEST_OBSERVED', json.dumps({k: result[k] for k in
        ('status', 'scope', 'baseline', 'actual_v10', 'selected', 'source_pixel_and_native_checks', 'seconds', 'job')}), flush=True)


if __name__ == '__main__':
    parser = argparse.ArgumentParser();parser.add_argument('--run-root', required=True);parser.add_argument('--output', required=True)
    args = parser.parse_args()
    prepare_output(args.run_root, args.output)
    sealed = Path(args.output).resolve() / 'source' / RUNNER
    if Path(__file__).resolve() != sealed:
        env = {**os.environ, 'PYTHONPATH': str(Path(args.output).resolve() / 'source')}
        os.execve(sys.executable, [sys.executable, '-u', str(sealed), '--run-root',
                  str(Path(args.run_root).resolve()), '--output', str(Path(args.output).resolve())], env)
    main(args.run_root, args.output)

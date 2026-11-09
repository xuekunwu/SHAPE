"""Seal and reuse an already selected native crop result without another tool call.

This is immediate delivery reuse, not a cross-run reasoning cache. Callers must
still apply the selected VRR exclusions and downstream scientific-pixel checks.
All crop bytes and metadata, input identities, effective levers and native code
are rechecked; damaged evidence raises instead of silently recomputing it.
"""
from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import time


POLICY = 'verified-selected-crop-delivery-v1'


def _digest(path):
    hasher = hashlib.sha256()
    with Path(path).open('rb') as handle:
        for block in iter(lambda: handle.read(1024*1024), b''):
            hasher.update(block)
    return hasher.hexdigest()


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _code():
    folder = Path(__file__).resolve().parent
    verifier = folder.parents[1]/'agent/verification/verifiers/crop.py'
    return {str(path): _digest(path) for path in (Path(__file__).resolve(), folder/'tool.py', verifier)}


def _proof(path):
    path = Path(path).resolve(strict=True)
    return dict(path=str(path), sha256=_digest(path))


def seal(result, *, image, mask, source_image_id, min_area, group, parameters,
         rejected_cells, receipt):
    """Persist a selected result's evidence closure. Never overwrite a receipt."""
    if result.get('error') or not result.get('cell_count'):
        raise ValueError('only successful nonempty selected crop results can be sealed')
    metadata = Path(result['cell_crops_metadata_path']).resolve(strict=True)
    rows = json.loads(metadata.read_text())['cell_metadata']
    ids = [row['cell_id'] for row in rows]
    if not rows or len(ids) != len(set(ids)) or len(rows) != result['cell_count']:
        raise ValueError('selected crop identities/count do not match result')
    rejected_cells = sorted(rejected_cells)
    if len(rejected_cells) != len(set(rejected_cells)) or set(rejected_cells)-set(ids):
        raise ValueError('selected rejection identities do not match crops')
    # Explicit paths (not a directory walk) keep cost proportional to delivered data.
    files = {str(metadata): _digest(metadata)}
    for prefix in ('cell_crops_kept_manifest_', 'cell_crops_triage_trace_'):
        companion = metadata.with_name(metadata.name.replace('cell_crops_metadata_', prefix))
        if companion.is_file():
            files[str(companion)] = _digest(companion)
    for row in rows:
        proof = _proof(row['crop_path'])
        files[proof['path']] = proof['sha256']
    record = dict(policy=POLICY, image=_proof(image), mask=_proof(mask),
                  source_image_id=source_image_id, min_area=min_area, group=group,
                  parameters=parameters, rejected_cells=rejected_cells,
                  code=_code(), files=files,
                  result={k: result[k] for k in ('cell_count', 'total_detected_objects',
                          'processing_statistics', 'seconds') if k in result})
    record['result']['cell_crops_metadata_path'] = str(metadata)
    # Fail before writing if arguments cannot be represented faithfully.
    payload = _canonical(record)+'\n'
    receipt = Path(receipt).resolve()
    receipt.parent.mkdir(parents=True, exist_ok=True)
    # A consumer only receives the pointer after the exclusive write and fsync.
    with receipt.open('x', encoding='utf-8') as handle:
        handle.write(payload)
        handle.flush()
        os.fsync(handle.fileno())
    return dict(path=str(receipt), sha256=_digest(receipt))


def reuse(pointer, *, image, mask, source_image_id, min_area, group, parameters,
          rejected_cells):
    """Return only attested native output, keeping validation time separate."""
    started = time.perf_counter()
    receipt = Path(pointer['path'])
    if _digest(receipt) != pointer['sha256']:
        raise ValueError('selected crop delivery receipt changed')
    record = json.loads(receipt.read_text())
    if record.get('policy') != POLICY or record.get('code') != _code():
        raise ValueError('selected crop delivery policy/code changed')
    expected = dict(source_image_id=source_image_id, min_area=min_area, group=group,
                    parameters=parameters, rejected_cells=sorted(rejected_cells))
    if any(_canonical(record.get(key)) != _canonical(value) for key, value in expected.items()):
        raise ValueError('selected crop delivery request differs from selected trial')
    for key, path in [('image', image), ('mask', mask)]:
        if _digest(path) != record[key]['sha256']:
            raise ValueError(f'selected crop delivery {key} changed')
    result = dict(record['result'])
    metadata = result['cell_crops_metadata_path']
    if metadata not in record['files']:
        raise ValueError('selected crop metadata missing from attestation')
    for path, sha256 in record['files'].items():
        if _digest(path) != sha256:
            raise ValueError('selected crop output changed: '+path)
    # Recheck the closure rather than trusting a well-hashed but incomplete receipt.
    rows = json.loads(Path(metadata).read_text())['cell_metadata']
    ids = [row['cell_id'] for row in rows]
    if (len(ids) != len(set(ids)) or len(ids) != result['cell_count']
            or set(rejected_cells)-set(ids)):
        raise ValueError('selected crop result identities changed')
    for row in rows:
        if str(Path(row['crop_path']).resolve()) not in record['files']:
            raise ValueError('selected crop file missing from attestation')
    result['delivery_reuse'] = dict(policy=POLICY, receipt=str(receipt),
        receipt_sha256=pointer['sha256'], native_call_executed=False,
        original_native_seconds=result.get('seconds'),
        verification_seconds=time.perf_counter()-started)
    return result

"""Batched visual Verify for statistical size alerts, with durable evidence.

This module has no dataset schema or prediction targets. Callers supply the
segmentation task and delivered native masks. An uncertain visual read retains
individual VRR. Receipts preserve model work; reasoning decisions are not cached.
"""
from __future__ import annotations

import hashlib
import io
import json
from pathlib import Path
import time
import uuid

import numpy as np
from PIL import Image, ImageDraw

from .segmentation_adaptive import REQUIRED_VIEWS, TriageReply, native_guard_ok, triage_batch_decisions
from shape.tools.cell_segmenter.trial_cache import atomic_json, canonical, digest


RENDER_VERSION = 'segmentation-triage-evidence-v1'


def _rgb(raw):
    raw = np.asarray(raw)
    if raw.ndim == 2:
        raw = np.repeat(raw[..., None], 3, axis=2)
    if raw.ndim != 3 or raw.shape[2] not in (1, 3, 4):
        raise ValueError('triage requires an explicit YX or YXC image')
    raw = raw[..., :3]
    if raw.shape[2] == 1:
        raw = np.repeat(raw, 3, axis=2)
    if not np.isfinite(raw).all():
        raise ValueError('nonfinite raw image')
    lo, hi = np.percentile(raw, [1, 99])
    if hi <= lo:
        raise ValueError('degenerate raw image')
    return np.clip((raw.astype(float) - lo) * 255 / (hi - lo), 0, 255).astype(np.uint8)


def evidence_card(raw, mask, image_id: str):
    """Raw/contour overviews and paired raw/contour size-extreme zooms.

    Label IDs are remapped before bincount, so sparse labels cannot cause an
    allocation proportional to the largest ID. Display scaling is shared across
    the raw and all zooms; no learned signal or endpoint labels enter rendering.
    """
    rgb, mask = _rgb(raw), np.asarray(mask)
    if (mask.ndim != 2 or mask.shape != rgb.shape[:2]
            or not np.issubdtype(mask.dtype, np.integer) or np.any(mask < 0)):
        raise ValueError('mask must be nonnegative integer YX aligned to the input')
    labels, inverse, counts = np.unique(mask, return_inverse=True, return_counts=True)
    objects = np.flatnonzero(labels > 0)
    if len(objects) == 0:
        raise ValueError('empty native segmentation')
    inverse = inverse.reshape(mask.shape)
    boundary = np.zeros(mask.shape, bool)
    boundary[1:] |= (mask[1:] != mask[:-1]) & (mask[1:] > 0)
    boundary[:-1] |= (mask[1:] != mask[:-1]) & (mask[:-1] > 0)
    boundary[:, 1:] |= (mask[:, 1:] != mask[:, :-1]) & (mask[:, 1:] > 0)
    boundary[:, :-1] |= (mask[:, 1:] != mask[:, :-1]) & (mask[:, :-1] > 0)
    overlay = rgb.copy()
    overlay[boundary] = (255, 40, 180)
    canvas = Image.new('RGB', (768, 900), 'white')
    draw = ImageDraw.Draw(canvas)
    draw.text((8, 8), image_id + ' | RAW overview / MASK overlay', fill='black')

    def paste(array, x, y, width, height):
        picture = Image.fromarray(array)
        scale = min(width/picture.width, height/picture.height)
        picture = picture.resize((max(1, round(picture.width*scale)), max(1, round(picture.height*scale))),
                                 Image.Resampling.LANCZOS)
        canvas.paste(picture, (x + (width-picture.width)//2, y + (height-picture.height)//2))

    paste(rgb, 0, 28, 384, 384)
    paste(overlay, 384, 28, 384, 384)
    ordered = sorted(objects, key=lambda i: (int(counts[i]), int(labels[i])))
    selected = list(dict.fromkeys(ordered[:2] + ordered[-2:]))
    zooms = []
    for row, index in enumerate(selected):
        yy, xx = np.nonzero(inverse == index)
        padding = max(8, int(np.sqrt(counts[index]) * .4))
        y0, y1 = max(0, int(yy.min())-padding), min(mask.shape[0], int(yy.max())+padding+1)
        x0, x1 = max(0, int(xx.min())-padding), min(mask.shape[1], int(xx.max())+padding+1)
        x, y = (row % 2)*384, 430 + (row//2)*230
        label = int(labels[index])
        draw.text((x+4, y), f'id={label} area={int(counts[index])} RAW / MASK', fill='black')
        paste(rgb[y0:y1, x0:x1], x, y+20, 192, 200)
        paste(overlay[y0:y1, x0:x1], x+192, y+20, 192, 200)
        zooms.append(dict(label=label, area=int(counts[index]), bbox=[y0, y1, x0, x1]))
    stream = io.BytesIO()
    canvas.save(stream, format='PNG')
    return stream.getvalue(), dict(views=sorted(REQUIRED_VIEWS), zooms=zooms,
                                  display_percentiles=[float(lo) for lo in np.percentile(np.asarray(raw), [1, 99])])


class VisualTriage:
    """A cohort callback: each batch gets one VLM question and an immutable receipt.

    Model identity is recorded for audit, not used to claim provider attestation.
    An exact input fingerprint binds source bytes, mask bytes, rendered cards,
    prompt, schema and requested model settings. Every invocation is charged and
    saved separately, including malformed responses and errors.
    """
    def __init__(self, engine, root, *, task: str, model_identity: dict, batch_size=4):
        if not task.strip() or not model_identity or not 1 <= batch_size <= 8:
            raise ValueError('explicit task, model identity and bounded batch size required')
        self.engine, self.root = engine, Path(root)
        self.task, self.identity, self.batch_size = task, json.loads(canonical(model_identity)), batch_size
        self.root.mkdir(parents=True, exist_ok=True)

    def _prepare(self, request, folder):
        from .segmentation_cohort import _segmentation
        resolution = _segmentation(request['resolutions'])
        ident = request['spec']['image_id']
        item = dict(image_id=ident, problems=request['problems'], guard_ok=native_guard_ok(resolution), evidence_views=[])
        if not item['guard_ok']:
            raise ValueError('native delivery guard failed')
        source = Path(request['spec']['path'])
        mask_path = Path(resolution.result['mask_path'])
        import tifffile
        before = dict(image_sha256=digest(source), mask_sha256=digest(mask_path))
        raw = tifffile.imread(source) if source.suffix.lower() in ('.tif', '.tiff') else np.asarray(Image.open(source))
        mask = np.load(mask_path, allow_pickle=False) if mask_path.suffix == '.npy' else tifffile.imread(mask_path)
        card, metadata = evidence_card(raw, mask, ident)
        if before != dict(image_sha256=digest(source), mask_sha256=digest(mask_path)):
            raise ValueError('native evidence changed during rendering')
        # Filenames do not contain caller-provided IDs or paths.
        file = folder / (uuid.uuid4().hex + '.png')
        file.write_bytes(card)
        item['evidence_views'] = metadata['views']
        evidence = dict(image_id=ident, **before, card=str(file), card_sha256=digest(file),
                        native_problems=request['problems'], **metadata)
        return item, card, evidence

    def __call__(self, requests):
        from shape.engine.errors import is_fatal_llm_error
        decisions = []
        for offset in range(0, len(requests), self.batch_size):
            folder = self.root / ('batch_' + uuid.uuid4().hex)
            folder.mkdir()
            items, cards, evidence, render_errors = [], [], [], []
            for request in requests[offset:offset+self.batch_size]:
                try:
                    item, card, record = self._prepare(request, folder)
                    items.append(item); cards.append(card); evidence.append(record)
                except Exception as exc:
                    item = dict(image_id=request['spec']['image_id'], guard_ok=False,
                                problems=request['problems'], evidence_views=[])
                    decisions.extend(triage_batch_decisions([item], None))
                    render_errors.append(dict(image_id=item['image_id'], error=type(exc).__name__, message=str(exc)))
            prompt = (
                'Verify instance segmentation against the task below. Each attached card has its image ID, '
                'a raw overview, mask-contour overview, and paired raw/contour zooms of small and large objects. '
                'Assess each image independently. Size-fraction alerts alone are not proof of errors. '
                'Accept with flags ONLY when all three views are sufficient and provide high-confidence '
                'evidence of normal heterogeneity with no actionable missed, merged, fragmented or spurious '
                'whole-cell instances. Otherwise request full_vrr or abstain. Cite the viewed evidence types; '
                'do not infer accuracy from count alone. Statistical alerts remain recorded on acceptance.\n'
                'Task:\n' + self.task + '\nImages and recorded alerts:\n' + canonical([
                    dict(image_id=i['image_id'], problems=i['problems']) for i in items]))
            request_record = dict(format=RENDER_VERSION, evidence=evidence, render_errors=render_errors,
                                  prompt=prompt, schema=TriageReply.model_json_schema(), model=self.identity,
                                  renderer_sha256=digest(Path(__file__)))
            fingerprint_record = {**request_record, 'evidence': [
                {k: v for k, v in e.items() if k != 'card'} for e in evidence]}
            request_record['fingerprint'] = hashlib.sha256(canonical(fingerprint_record).encode()).hexdigest()
            atomic_json(folder/'REQUEST.json', request_record)
            response, failure, reply, raw_answer = None, None, None, None
            started = time.perf_counter()
            try:
                if items:
                    response = self.engine.generate([prompt, *cards], response_format=TriageReply)
                    content = response.get('content') if isinstance(response, dict) else response
                    raw_answer = content.model_dump() if hasattr(content, 'model_dump') else content
                    if isinstance(content, str):
                        content = json.loads(content)
                    reply = TriageReply.model_validate(content).model_dump()
                    answer_ids = [v['image_id'] for v in reply['verdicts']]
                    if len(answer_ids) != len(set(answer_ids)) or set(answer_ids) != {i['image_id'] for i in items}:
                        raise ValueError('visual reply must cover each requested image ID exactly once')
            except Exception as exc:
                reply = None
                failure = dict(type=type(exc).__name__, message=str(exc))
                if is_fatal_llm_error(exc):
                    raise
            finally:
                receipt = dict(seconds=time.perf_counter()-started, reply=reply, raw_answer=raw_answer, error=failure,
                    usage=response.get('usage') if isinstance(response, dict) else None,
                    requested_model=self.identity, decision_reused=False)
                atomic_json(folder/'RECEIPT.json', receipt)
            checked = triage_batch_decisions(items, reply)
            for decision in checked:
                decision['receipt'] = str(folder/'RECEIPT.json')
            atomic_json(folder/'DECISIONS.json', checked)
            decisions.extend(checked)
        return decisions

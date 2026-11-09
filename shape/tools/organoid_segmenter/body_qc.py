"""GT-free dark-body coverage cues for organoid triage, never prediction generation."""
from __future__ import annotations
import numpy as np
from PIL import Image
from scipy import ndimage as ndi
from skimage.filters import threshold_otsu

POLICY = "dark_body_v2"
MIN_COVERAGE = 0.55
MAX_FRAGMENTS_PER_BODY = 3


def body_qc(image, masks, max_px=384):
    """Compare native labels to compact dark image components at thumbnail resolution.

    An unsupported/ambiguous image returns confident=0, not evidence of correctness.
    The image-only body estimate is discarded after QC and never becomes a mask.
    """
    out = dict(organoid_body_confident=0.0)
    raw, labels = np.asarray(image), np.asarray(masks)
    if raw.ndim == 3: raw = raw[..., 0]
    if raw.ndim != 2 or labels.shape != raw.shape or not np.all(np.isfinite(raw)):
        return out
    if not np.any(labels > 0): return out
    h, w = raw.shape
    scale = min(1.0, max_px / max(h, w))
    size = (max(1, round(w * scale)), max(1, round(h * scale)))
    gray = np.asarray(Image.fromarray(raw.astype(np.float32)).resize(size, Image.Resampling.BILINEAR))
    pred = np.asarray(Image.fromarray(labels.astype(np.int32)).resize(size, Image.Resampling.NEAREST))
    fg = pred > 0
    if not fg.any(): return out
    lo, hi = np.percentile(gray, [1, 99])
    if hi <= lo: return out
    gray = np.clip((gray - lo) / (hi - lo), 0, 1)
    smooth = ndi.gaussian_filter(gray, 1)
    dark = ndi.binary_fill_holes(smooth < threshold_otsu(smooth))
    components, n = ndi.label(dark)
    areas = np.bincount(components.ravel(), minlength=n + 1)
    overlap = np.bincount(components[fg], minlength=n + 1)
    selected = [i for i in range(1, n + 1)
                if .001 <= areas[i] / gray.size <= .45 and overlap[i] >= max(2, .02 * fg.sum())]
    if not selected: return out
    body = np.isin(components, selected)
    agreement = float(np.count_nonzero(body & fg) / fg.sum())
    if agreement < .70 or body.all(): return out
    contrast = float(np.median(gray[~body]) - np.median(gray[body]))
    if contrast < .15: return out
    covered = float(np.count_nonzero(body & fg) / body.sum())
    count = len(np.unique(pred[fg]))
    out.update(organoid_body_confident=1.0, organoid_body_coverage=covered,
               organoid_body_prediction_overlap=agreement,
               organoid_body_area_fraction=float(body.mean()),
               organoid_body_component_count=float(len(selected)),
               organoid_fragments_per_body=float(count / len(selected)))
    return out


def problems(metrics):
    if metrics.get("organoid_body_confident") != 1:
        return []
    issues = []
    coverage = metrics["organoid_body_coverage"]
    if coverage < MIN_COVERAGE:
        issues.append(f"whole-organoid body coverage {coverage:.3f} below {MIN_COVERAGE:g}; inspect raw image and overlay")
    fragments = metrics["organoid_fragments_per_body"]
    if fragments > MAX_FRAGMENTS_PER_BODY:
        issues.append(f"organoid fragments per dark body {fragments:.1f} above {MAX_FRAGMENTS_PER_BODY}; inspect instance splitting")
    return issues

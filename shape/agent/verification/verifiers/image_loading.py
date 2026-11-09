"""Verifier 1: Image loading (pure Python, NO tunable params).

This step cannot rerun acquisition, so its only legitimate actions are
exclude_samples / conceded (fault="upstream", IO config) / conceded (fault="ceiling").
There is no `retune_here` here -- that asymmetry is exactly why the loop must
distinguish action types rather than always "recommend params".
"""

from __future__ import annotations

import numpy as np

from shape.utils.schemas import VLRRResult

from ..base import BaseVerifier, normalize, weighted_score


def _laplacian_var(img: np.ndarray) -> float:
    # 3x3 Laplacian via finite differences; variance = sharpness proxy
    lap = (-4 * img
           + np.roll(img, 1, 0) + np.roll(img, -1, 0)
           + np.roll(img, 1, 1) + np.roll(img, -1, 1))
    return float(lap.var())


def per_image_qc(img: np.ndarray, bit_range: float) -> dict[str, float]:
    flat = img.ravel()
    p1, p99 = np.percentile(flat, [1, 99])
    h, w = img.shape[:2]
    cy, cx = h // 2, w // 2
    center = img[cy - h // 6: cy + h // 6, cx - w // 6: cx + w // 6]
    edge = np.concatenate([img[:h // 8].ravel(), img[-h // 8:].ravel()])
    bg = flat[flat < np.percentile(flat, 25)]
    bg_cv = float(bg.std() / (bg.mean() + 1e-9)) if bg.size else 0.0
    return {
        "saturation_fraction": float((flat > 0.98 * bit_range).mean()),
        "zero_fraction": float((flat < 0.02 * bit_range).mean()),
        "dynamic_range": float((p99 - p1) / bit_range),
        "sharpness": _laplacian_var(img.astype(np.float64)),
        "background_CV": bg_cv,
        "center_edge_ratio": float(center.mean() / (edge.mean() + 1e-9)),
    }


class ImageLoadingVerifier(BaseVerifier):
    name = "image_loading"

    # Per-image pixel-quality cuts used by quantify()'s per-image loop below. These are
    # absolute quality standards (a blank/blown-out/flat image is bad regardless of the
    # batch), so they are NOT data-derived. Config-declared gates (soft priors), not inline
    # literals -- mirrors CropVerifier._TRUNCATION_MAX -- and the deterministic FALLBACK
    # PRIOR for an agent-reasoned per-input override (no tool CARD.md documents this
    # verifier yet -- see coding log).
    _SATURATION_MAX = 0.05      # per-image saturation_fraction cut
    _ZERO_MAX = 0.10            # per-image zero_fraction cut
    _DYNAMIC_RANGE_MIN = 0.05   # per-image dynamic_range cut

    def __init__(self, bit_range: float, expected_channels: int,
                 exclude_cap: float = 0.05):
        self.bit_range = bit_range
        self.expected_channels = expected_channels
        self.exclude_cap = exclude_cap

    def quantify(self, tool_output, context) -> dict[str, float]:
        # tool_output: list[np.ndarray]; context carries declared load metadata
        imgs = tool_output
        per = [per_image_qc(im, self.bit_range) for im in imgs]
        sharp_p10 = float(np.percentile([m["sharpness"] for m in per], 10))

        n_fail = 0
        for m in per:
            if (m["saturation_fraction"] > self._SATURATION_MAX
                    or m["zero_fraction"] > self._ZERO_MAX
                    or m["dynamic_range"] < self._DYNAMIC_RANGE_MIN
                    or m["sharpness"] < sharp_p10):
                n_fail += 1

        loaded_channels = context.get("loaded_channels", self.expected_channels)
        integrity_ok = (loaded_channels == self.expected_channels
                        and context.get("bit_depth_ok", True))

        agg = {k: float(np.mean([m[k] for m in per])) for k in per[0]}
        agg.update({
            "n_images": float(len(imgs)),
            "n_failed_images": float(n_fail),
            "failed_fraction": n_fail / len(imgs),
            "load_integrity": 1.0 if integrity_ok else 0.0,
        })
        # group bias (multi_group): mean shift + KS distance
        if context.get("groups"):
            agg["group_effect_size"] = _group_effect(imgs, context["groups"])
        return agg

    def localize(self, metrics, context):
        triggered, affected = [], []
        if metrics["load_integrity"] < 1.0:
            triggered.append("load_integrity")
            return "tool_mismatch", triggered, affected   # IO config, not quality
        if metrics["failed_fraction"] > 0:
            triggered.append("failed_images")
            return "data_intrinsic", triggered, affected   # bad images, unfixable here
        if metrics.get("group_effect_size", 0.0) > 1.0:
            triggered.append("group_effect_size")
            return "upstream", triggered, affected         # input-loading context issue
        return "this_step", triggered, affected

    def adjudicate(self, metrics, locus, triggered, context) -> VLRRResult:
        parts = {
            "saturation_fraction": normalize(metrics["saturation_fraction"], 0.0, 0.05),
            "dynamic_range": normalize(metrics["dynamic_range"], 0.5, 0.05),
            "sharpness": 1.0 if metrics["failed_fraction"] == 0 else 0.0,
        }
        score = weighted_score(parts, {k: 1.0 for k in parts})

        if locus == "tool_mismatch":
            return VLRRResult("FAIL", score, {}, action="conceded", fault="upstream",
                              upstream_target="io_config",
                              notes="Channel/bit-depth mismatch: re-read with "
                                    "corrected IO config before judging quality.")
        if locus == "data_intrinsic":
            if metrics["failed_fraction"] <= self.exclude_cap:
                return VLRRResult("WARNING", score, {}, action="exclude_samples",
                                  notes="Drop the few unrecoverable images, proceed.")
            return VLRRResult("FAIL", score, {}, action="conceded", fault="ceiling",
                              notes="Too many bad images; data quality in doubt.")
        if locus == "upstream":
            return VLRRResult("WARNING", score, {}, action="conceded", fault="upstream",
                              upstream_target="image_loading",
                              expected_effect={"group_effect_size": "down"},
                              notes="Inter-group technical bias -> route to "
                                    "image-loading review before continuing.")
        return VLRRResult("PASS", score, {}, action="accept")


def _group_effect(imgs, groups) -> float:
    means = {}
    for im, g in zip(imgs, groups):
        means.setdefault(g, []).append(float(np.asarray(im).mean()))
    keys = list(means)
    if len(keys) < 2:
        return 0.0
    a, b = np.array(means[keys[0]]), np.array(means[keys[1]])
    pooled = np.sqrt((a.var() + b.var()) / 2) + 1e-9
    return float(abs(a.mean() - b.mean()) / pooled)

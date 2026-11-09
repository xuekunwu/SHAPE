"""The metadata profile of one solve() run: what the image specs declare about batches, groups and controls.

Built once, right after the image specs are normalised, by a deterministic rule (no LLM; the request's free text
is never a tier source), and handed to every managed stage along the request's path
(`Verifier.resolve_managed_stage(metadata_profile=)`). The tier says what the metadata supports:

- none: no batch labels (a single image, an image without a batch value, a single batch value, or a batch with
  fewer than `MIN_IMAGES_PER_BATCH` images);
- full: batch labels, at least `MIN_REPLICATED_GROUPS` non-control groups replicated in two or more batches (with
  fewer, the calibration readouts have no biology metric, so biology preservation cannot be verified), and the
  batch-metrics extra usable;
- partial: everything else (batch labels only; an unusable batch-metrics extra downgrades full to partial).

The batch-metrics extra (scib-metrics and its jax stack) is checked lazily, when the batch step first runs
(`resolve_batch_metrics`): a run without a batch step never imports jax. Until then a full tier is the design's,
and `batch_metrics` is None; the check sets it to "usable", or downgrades the tier to partial with the reason.

The batch value of an image is the first non-empty of `BATCH_KEYS`, the order `cell_state_analyzer.tool.
resolve_technical_metadata` reads (a test locks the equality). A group is the spec's `group` unless "default" or
empty; a control is a spec with `control: true`. The per-stage use of each tier is declared in each managed
tool's CARD front matter (`metadata_consumes`, `metadata_none`, `metadata_partial`, `metadata_full`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass, field, replace
from functools import lru_cache
import importlib
import os
import re
from typing import Iterable, Optional

PROFILE_VERSION = 1
# the analyzer's order (`resolve_technical_metadata`: plate, experiment, batch; each canonical, then `*_id`)
BATCH_KEYS = ("plate", "plate_id", "experiment", "experiment_id", "batch", "batch_id")
MIN_IMAGES_PER_BATCH = 2
MIN_REPLICATED_GROUPS = 2       # the full tier: biology preservation needs at least two labels to be scored
# what `quant.composite_all` imports for the calibration readouts (scib-metrics, its jax and pynndescent stack)
BATCH_METRICS_IMPORTS = ("jax", "pynndescent", "scib_metrics", "scib_metrics.nearest_neighbors")
_import = importlib.import_module
TIERS = ("none", "partial", "full")
# the request's "## Target" section (the agentic segmentation verifier's `TARGET_HEADING`)
_TARGET_HEADING = re.compile(r"^##[ \t]+target[ \t]*$", re.IGNORECASE | re.MULTILINE)


@dataclass(frozen=True)
class MetadataProfile:
    version: int
    tier: str                      # "none" | "partial" | "full"
    n_images: int
    batch: dict                    # available, key, labels (image_id -> label), n_batches, images_per_batch
    groups: dict                   # labels (image_id -> group), replicated, in_all_batches
    controls: dict                 # images, groups, per_batch, in_every_batch
    technical: dict                # well, site: whether every image declares one
    request: dict                  # target_section
    reasons: tuple = field(default_factory=tuple)
    batch_metrics: Optional[str] = None     # the extra's check: None (not run yet), "usable", or why not

    def to_dict(self) -> dict:
        out = asdict(self)
        out["reasons"] = list(self.reasons)
        return out


def _present(value) -> bool:
    return value is not None and bool(str(value).strip())


def batch_value(spec: dict) -> tuple[Optional[str], Optional[str]]:
    """(the image's batch label, the spec key it came from): the first non-empty of `BATCH_KEYS`."""
    for key in BATCH_KEYS:
        value = spec.get(key)
        if _present(value):
            return str(value), key
    return None, None


def declared_group(spec: dict) -> Optional[str]:
    group = spec.get("group")
    if not _present(group) or str(group) == "default":
        return None
    return str(group)


def is_control(spec: dict) -> bool:
    value = spec.get("control")
    return value is True or (isinstance(value, str) and value.strip().lower() == "true")


@lru_cache(maxsize=1)
def batch_metrics_problem() -> Optional[str]:
    """None when the batch-metrics extra the calibration readouts need imports (`BATCH_METRICS_IMPORTS`, the
    functions `quant.composite_all` calls), else why not. Imported once per process. JAX's GPU memory
    preallocation is turned off first (unless the caller set it), so that jax never takes the GPU memory the
    DINOv3 training of the same process needs."""
    os.environ.setdefault("XLA_PYTHON_CLIENT_PREALLOCATE", "false")
    try:
        for name in BATCH_METRICS_IMPORTS:
            _import(name)
        scib = _import("scib_metrics")
        for function in ("silhouette_batch", "silhouette_label", "ilisi_knn", "clisi_knn", "graph_connectivity",
                         "nmi_ari_cluster_labels_leiden"):
            getattr(scib, function)
        getattr(_import("scib_metrics.nearest_neighbors"), "pynndescent")
    except Exception as error:      # noqa: BLE001 -- any failure means the readouts cannot be computed
        return f"{type(error).__name__}: {error}"
    return None


def _image_id(spec: dict, index: int) -> str:
    return str(spec.get("image_id") or spec.get("path") or f"image_{index}")


def build_metadata_profile(question: Optional[str], specs: Iterable[dict]) -> MetadataProfile:
    """The run's `MetadataProfile` from its normalised image specs and the request (see the module docstring)."""
    specs = [s if isinstance(s, dict) else {"path": s} for s in (specs or [])]
    ids = [_image_id(s, i) for i, s in enumerate(specs)]
    reasons: list[str] = []

    labels, keys = {}, set()
    for image_id, spec in zip(ids, specs):
        value, key = batch_value(spec)
        labels[image_id] = value
        if key:
            keys.add(key[:-3] if key.endswith("_id") else key)
    present = [v for v in labels.values() if v is not None]
    per_batch: dict[str, int] = {}
    for value in present:
        per_batch[value] = per_batch.get(value, 0) + 1
    missing = [image_id for image_id, value in labels.items() if value is None]
    if len(specs) < 2:
        reasons.append(f"{len(specs)} image(s): batch labels need at least 2")
    elif missing and present:
        reasons.append(f"{len(missing)} of {len(specs)} images have no batch value (partial coverage)")
    elif missing:
        reasons.append("no image declares a batch value (plate / experiment / batch)")
    elif len(per_batch) < 2:
        reasons.append(f"one batch value ({present[0]}) for every image")
    small = sorted(b for b, n in per_batch.items() if n < MIN_IMAGES_PER_BATCH)
    if not reasons and small:
        reasons.append(f"batch(es) with fewer than {MIN_IMAGES_PER_BATCH} images: {', '.join(small)}")
    available = not reasons
    batch = {"available": available, "key": (keys.pop() if len(keys) == 1 else "batch") if present else None,
             "labels": labels, "n_batches": len(per_batch), "images_per_batch": dict(sorted(per_batch.items()))}

    groups_by_image = {image_id: declared_group(spec) for image_id, spec in zip(ids, specs)}
    control_ids = [image_id for image_id, spec in zip(ids, specs) if is_control(spec)]
    batches_of: dict[str, set] = {}
    for image_id, group in groups_by_image.items():
        if group is not None and image_id not in control_ids and labels[image_id] is not None:
            batches_of.setdefault(group, set()).add(labels[image_id])
    replicated = sorted(g for g, b in batches_of.items() if len(b) >= 2)
    in_all = sorted(g for g, b in batches_of.items() if per_batch and b >= set(per_batch))
    groups = {"labels": groups_by_image, "replicated": replicated if available else [],
              "in_all_batches": in_all if available else []}

    control_batches: dict[str, int] = {}
    for image_id in control_ids:
        if labels[image_id] is not None:
            control_batches[labels[image_id]] = control_batches.get(labels[image_id], 0) + 1
    controls = {"images": control_ids,
                "groups": sorted({g for i, g in groups_by_image.items() if i in control_ids and g is not None}),
                "per_batch": dict(sorted(control_batches.items())),
                "in_every_batch": bool(per_batch) and set(control_batches) >= set(per_batch)}

    technical = {"well": bool(specs) and all(_present(s.get("well")) or _present(s.get("well_id")) for s in specs),
                 "site": bool(specs) and all(_present(s.get("site")) or _present(s.get("site_id")) for s in specs)}
    request = {"target_section": bool(_TARGET_HEADING.search(
        (question or "").replace("\r\n", "\n").replace("\r", "\n")))}

    if not available:
        tier = "none"
    elif len(replicated) < MIN_REPLICATED_GROUPS:
        tier = "partial"
        reasons.append(f"fewer than {MIN_REPLICATED_GROUPS} non-control groups occur in 2 or more batches: "
                       "biology preservation cannot be verified")
    else:
        tier = "full"           # the design's; the batch-metrics extra is checked by `resolve_batch_metrics`
    return MetadataProfile(version=PROFILE_VERSION, tier=tier, n_images=len(specs), batch=batch, groups=groups,
                           controls=controls, technical=technical, request=request, reasons=tuple(reasons))


def resolve_batch_metrics(profile: MetadataProfile) -> MetadataProfile:
    """The profile once the batch-metrics extra is checked (`batch_metrics_problem`), for the full tier: the same
    tier with `batch_metrics` "usable", or the partial tier with the reason. Other tiers, and a profile already
    checked, are returned unchanged."""
    if profile.tier != "full" or profile.batch_metrics is not None:
        return profile
    problem = batch_metrics_problem()
    if problem is None:
        return replace(profile, batch_metrics="usable")
    return replace(profile, tier="partial", batch_metrics=problem, reasons=profile.reasons + (
        f"the batch-metrics extra (scib-metrics) is not usable ({problem}): no calibration readouts",))


def batch_routed(profile: Optional[MetadataProfile]) -> bool:
    """Whether the system routes the batch-correction step: batch labels exist (tier partial or full)."""
    return profile is not None and profile.tier != "none"

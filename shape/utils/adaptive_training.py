"""Data-dependent training budgets, convergence-based early stopping, and clustering-
representation dimensionality selection (task 075 M1)."""

from __future__ import annotations

import math
import os
from collections import deque
from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _available_memory_bytes(device) -> tuple[int, str]:
    """Return currently available CUDA or system memory without a hard dependency."""
    if getattr(device, "type", str(device)) == "cuda":
        import torch

        device_index = getattr(device, "index", None)
        if device_index is None:
            device_index = torch.cuda.current_device()
        try:
            free_bytes, _ = torch.cuda.mem_get_info(device_index)
            return int(free_bytes), f"cuda_mem_get_info:{device_index}"
        except (RuntimeError, ValueError):
            properties = torch.cuda.get_device_properties(device_index)
            reserved = torch.cuda.memory_reserved(device_index)
            free_estimate = max(0, int(properties.total_memory) - int(reserved))
            return free_estimate, f"cuda_total_minus_reserved:{device_index}"
    try:
        import psutil

        return int(psutil.virtual_memory().available), "psutil"
    except ImportError:
        if hasattr(os, "sysconf"):
            try:
                pages = os.sysconf("SC_AVPHYS_PAGES")
                page_size = os.sysconf("SC_PAGE_SIZE")
                return int(pages * page_size), "sysconf"
            except (OSError, ValueError):
                pass
    return 2 * 1024**3, "conservative_fallback"


def resolve_batch_size(
    num_samples: int,
    sample_bytes: int,
    model_parameter_bytes: int,
    device,
    requested_batch_size: Optional[int] = None,
) -> dict:
    """Choose a power-of-two batch from live memory and transformed crop size.

    The estimate reserves memory for optimizer state and framework overhead, then
    accounts for two augmented views plus transformer activations/backpropagation.
    A caller-provided batch size is treated as a ceiling, not an unsafe override.
    """
    samples = max(1, int(num_samples))
    available_bytes, memory_source = _available_memory_bytes(device)
    is_cuda = getattr(device, "type", str(device)) == "cuda"
    memory_fraction = 0.84 if is_cuda else 0.50
    optimizer_reserve = int(model_parameter_bytes * (3.0 if is_cuda else 2.0))
    usable_bytes = max(
        0,
        int(available_bytes * memory_fraction) - optimizer_reserve,
    )

    # ViT training retains substantially more than the input tensor itself.
    activation_multiplier = 96 if is_cuda else 96
    estimated_bytes_per_sample = max(
        24 * 1024**2,
        int(sample_bytes) * activation_multiplier,
    )
    memory_limited = max(1, usable_bytes // estimated_bytes_per_sample)
    automatic_ceiling = 128 if is_cuda else 32
    ceiling = min(samples, automatic_ceiling)
    if requested_batch_size is not None:
        ceiling = min(ceiling, max(1, int(requested_batch_size)))
    candidate = max(1, min(ceiling, memory_limited))
    resolved = 1 << (candidate.bit_length() - 1)

    return {
        "batch_size": resolved,
        "requested_batch_size": requested_batch_size,
        "available_memory_bytes": available_bytes,
        "usable_memory_bytes": usable_bytes,
        "sample_bytes": int(sample_bytes),
        "estimated_bytes_per_sample": estimated_bytes_per_sample,
        "model_parameter_bytes": int(model_parameter_bytes),
        "memory_source": memory_source,
        "device_type": getattr(device, "type", str(device)),
        "automatic_batch_ceiling": automatic_ceiling,
    }


def early_stopping_patience(budget: int) -> int:
    """The agentic VRR loops' early-stopping patience, their backstop stop, for a search of at most `budget`
    rounds: max(3, min(5, ceil(0.25 * budget))) tool calls in a row without an improvement beyond the
    noise end it (the agent may stop earlier). The proportion of `resolve_training_config`'s epoch
    patience; its floor is 2, this one's 3."""
    return max(3, min(5, math.ceil(0.25 * max(1, int(budget)))))


def resolve_training_config(
    num_samples: int,
    batch_size: int,
    max_epochs: Optional[int] = None,
    patience: Optional[int] = None,
    min_epochs: Optional[int] = None,
) -> dict:
    """Resolve an epoch budget from sample coverage, independent of batch size.

    min_epochs: the early stopper's floor; None = max(4, ceil(0.35 x max_epochs)), an explicit value is
    honored (at least 1, at most max_epochs); nothing else changes."""
    samples = max(1, int(num_samples))
    batch = max(1, min(int(batch_size), samples))
    steps_per_epoch = max(1, math.ceil(samples / batch))

    if max_epochs is None:
        # Auto path: small datasets need more augmented re-exposures; large datasets already
        # provide diversity within each pass. Batch size must not inflate epochs. Clamp to
        # [8, 20] and allow the budget-extension loop to grow toward a higher ceiling.
        resolved_max_epochs = round(8 + 6 * math.sqrt(128 / samples))
        resolved_max_epochs = max(8, min(20, resolved_max_epochs))
        max_epochs_hard_cap = min(60, 4 * resolved_max_epochs)
    else:
        # An EXPLICIT cap is authoritative: honor it exactly (no [8, 20] clamp, no
        # budget-extension growth), so callers can run cheap short trials for fast screening.
        resolved_max_epochs = max(1, int(max_epochs))
        max_epochs_hard_cap = resolved_max_epochs

    min_epochs = min(
        resolved_max_epochs,
        max(4, math.ceil(0.35 * resolved_max_epochs)) if min_epochs is None else max(1, int(min_epochs)),
    )
    resolved_patience = (
        max(2, min(5, math.ceil(0.25 * resolved_max_epochs)))
        if patience is None
        else max(1, int(patience))
    )
    relative_min_delta = max(0.002, min(0.01, 0.1 / math.sqrt(samples)))

    return {
        "max_epochs": resolved_max_epochs,
        "max_epochs_hard_cap": max_epochs_hard_cap,
        "min_epochs": min_epochs,
        "patience": resolved_patience,
        "relative_min_delta": relative_min_delta,
        "steps_per_epoch": steps_per_epoch,
        "planned_optimizer_steps": steps_per_epoch * resolved_max_epochs,
        "target_sample_exposures": samples * resolved_max_epochs,
    }


def next_epoch_budget(max_epochs: int, hard_cap: int, growth: float = 1.2) -> int:
    """Grow the epoch ceiling when training is still improving at the current one.

    Returns an increased ceiling (toward ``hard_cap``) so the loop keeps training
    instead of stopping with a still-falling loss; returns ``max_epochs`` unchanged
    once the hard cap is reached.
    """
    if max_epochs >= hard_cap:
        return int(max_epochs)
    return min(int(hard_cap), int(math.ceil(max_epochs * growth)))


def is_cuda_out_of_memory(error: BaseException) -> bool:
    message = str(error).lower()
    return "out of memory" in message and ("cuda" in message or "cublas" in message)


def resolve_clustering_representation(
    embedding,
    requested_n_pcs: Optional[int] = None,
) -> dict:
    """Choose the clustering representation (raw vs PCA, and how many PCs) from the
    embedding's OWN eigenvalue spectrum -- never from `scalar_score` or any verifier
    quality metric (task 075 M1 safety boundary: that score is degenerate, see
    tasks 069/072).

    Rule: the broken-stick null model (MacArthur 1957; applied to PCA dimensionality
    selection by Jackson 1993, Ecology 74(8):2204-2214; see also Legendre & Legendre,
    Numerical Ecology, 3rd ed., section 9.1). A unit-length stick broken at random into
    `p` pieces has an expected length for its k-th (1-indexed, longest-to-shortest) piece
    of ``(1/p) * sum_{i=k}^{p} (1/i)``; a PC is retained while its OBSERVED variance
    share still exceeds this chance expectation. This is threshold-free in the sense that
    matters here: no externally hard-coded variance-fraction cutoff (e.g. "keep 90% of
    variance") is imposed -- the cutoff falls out of comparing the data's own spectrum to
    its own null.

    Fail-safe (never crashes, never fabricates a choice from bad input) -- falls back to
    the raw (unreduced) representation, with the reason recorded, when:
      - any non-finite value is present in `embedding` (a collapse signal upstream of
        this function; the safe response is to NOT silently zero-fill and derive a
        plausible-looking cutoff from partially-fabricated data -- task 072's QC caught
        exactly this failure mode, a NaN path that silently scored maximal);
      - fewer than 2 usable components exist (``min(n_obs - 1, n_features) < 2`` -- too
        few cells, or fewer cells than dimensions collapsing the usable rank); or
      - the embedding has zero (or non-finite) total variance (e.g. all-identical
        points).

    `requested_n_pcs`: an explicit caller override (0 = raw). When given, the spectrum is
    NOT consulted -- the value is honored exactly, clamped only to the data's own usable
    rank (never silently ignored, never allowed to request an infeasible dimension).
    """
    x = np.asarray(embedding, dtype=np.float64)
    if x.ndim != 2:
        raise ValueError(f"embedding must be 2-D (n_obs x n_features); got shape {x.shape}")
    n_obs, n_features = x.shape
    max_components = max(0, min(n_obs - 1, n_features))
    base = {
        "n_obs": int(n_obs),
        "n_features": int(n_features),
        "max_components": int(max_components),
        "requested_n_pcs": requested_n_pcs,
        "method": "explicit_override" if requested_n_pcs is not None else "broken_stick",
    }

    if requested_n_pcs is not None:
        n_pcs = max(0, min(int(requested_n_pcs), max_components))
        base.update({"n_pcs": n_pcs, "representation": "raw" if n_pcs == 0 else "pca"})
        return base

    n_nonfinite = int((~np.isfinite(x)).sum())
    if n_nonfinite:
        base.update({
            "n_pcs": 0, "representation": "raw",
            "reason": "non_finite_values_present", "n_nonfinite_values": n_nonfinite,
        })
        return base
    if max_components < 2:
        base.update({
            "n_pcs": 0, "representation": "raw",
            "reason": "insufficient_rank_for_spectrum",
        })
        return base

    centered = x - x.mean(axis=0)
    zero_variance_features = int((centered.var(axis=0) <= 0).sum())
    singular_values = np.linalg.svd(centered, full_matrices=False, compute_uv=False)
    singular_values = singular_values[:max_components]
    eigenvalues = singular_values ** 2
    total_variance = float(eigenvalues.sum())
    base["zero_variance_features"] = zero_variance_features

    if not math.isfinite(total_variance) or total_variance <= 0:
        base.update({
            "n_pcs": 0, "representation": "raw",
            "reason": "zero_or_nonfinite_variance",
        })
        return base

    p = len(eigenvalues)
    explained = eigenvalues / total_variance
    # broken-stick null: expected share of the k-th (1-indexed) longest piece of a unit
    # stick randomly broken into p pieces = (1/p) * sum_{i=k}^{p} (1/i)
    tail_harmonic = np.cumsum(1.0 / np.arange(p, 0, -1))[::-1]
    broken_stick = tail_harmonic / p
    beats_chance = explained > broken_stick

    if not beats_chance[0]:
        n_pcs, reason = 0, "no_component_beats_broken_stick_null"
    elif beats_chance.all():
        n_pcs, reason = p, "all_components_beat_broken_stick_null"
    else:
        # first index where the leading run of "beats chance" components ends
        n_pcs, reason = int(np.argmax(~beats_chance)), "broken_stick_elbow"

    base.update({
        "n_pcs": n_pcs,
        "representation": "raw" if n_pcs == 0 else "pca",
        "reason": reason,
        "cumulative_explained_variance": float(explained[:n_pcs].sum()) if n_pcs > 0 else 0.0,
    })
    return base


@dataclass
class AdaptiveEarlyStopper:
    min_epochs: int
    patience: int
    relative_min_delta: float
    absolute_loss_target: Optional[float] = None
    best_loss: float = math.inf
    # windowed-slope plateau state (not set by callers)
    _window: deque = field(default_factory=deque)
    _loss_scale: float = 0.0

    def observe(self, loss: float, epoch_number: int) -> tuple[bool, bool, Optional[str]]:
        """Return (is_best, should_stop, reason) for a completed epoch.

        Stopping judges the SLOPE of the loss curve, not whether a new numeric
        minimum was hit: once the average per-epoch decline across the recent
        window -- normalized by the overall loss scale -- falls below
        ``relative_min_delta``, the curve has flattened and training stops. A loss
        that keeps ticking down by a hair (a slowly-decaying tail) therefore no
        longer counts as "still improving" and no longer extends the budget.
        """
        if not math.isfinite(loss):
            return False, True, "non_finite_loss"

        is_best = loss < self.best_loss
        if is_best:
            self.best_loss = loss
        # Stable reference for the curve's height: the running max is ~the initial
        # loss for a decreasing curve and is robust to a warmup spike. Normalizing
        # the slope by this (not by the shrinking current value) makes "flat" mean
        # flat on the loss plot, not merely a small relative-to-latest change.
        self._loss_scale = max(self._loss_scale, abs(loss))

        window = self.patience + 1            # the window spans `patience` steps
        self._window.append(loss)
        while len(self._window) > window:
            self._window.popleft()

        if epoch_number < self.min_epochs:
            return is_best, False, None
        if self.absolute_loss_target is not None and loss <= self.absolute_loss_target:
            return is_best, True, "absolute_loss_target"
        if len(self._window) >= window:
            span = len(self._window) - 1
            avg_decline_per_epoch = (self._window[0] - self._window[-1]) / span
            rate = avg_decline_per_epoch / max(self._loss_scale, 1e-8)
            if rate < self.relative_min_delta:
                # slope (incl. a flat or rising curve) below the rate threshold
                return is_best, True, "convergence_plateau"
        return is_best, False, None

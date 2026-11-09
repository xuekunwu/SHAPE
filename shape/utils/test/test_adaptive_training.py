"""Task 075 M1: `resolve_clustering_representation` -- the clustering representation
(raw vs PCA, and how many PCs) chosen from the embedding's OWN eigenvalue spectrum
(broken-stick null model), never from `scalar_score` or any verifier quality metric.

Covers the degenerate-input class task 072's QC flagged (a NaN path that silently
scored maximal): non-finite values, fewer cells than dimensions, all-identical points,
zero-variance columns, and the explicit-override / clamp path.
"""

import numpy as np
import pytest

from shape.utils.adaptive_training import resolve_clustering_representation


def _low_rank_embedding(n_obs=300, n_features=100, true_rank=8, noise=0.05, seed=0):
    rng = np.random.RandomState(seed)
    latent = rng.randn(n_obs, true_rank)
    basis = rng.randn(true_rank, n_features)
    return latent @ basis + noise * rng.randn(n_obs, n_features)


def test_low_rank_structure_recovers_a_reduced_pca_representation():
    x = _low_rank_embedding()
    info = resolve_clustering_representation(x)
    assert info["representation"] == "pca"
    assert info["method"] == "broken_stick"
    # far fewer PCs than the ambient 100 features, and comfortably above the true rank
    # floor (broken-stick is conservative, so it need not exactly equal true_rank)
    assert 0 < info["n_pcs"] < x.shape[1]


def test_pure_noise_does_not_fabricate_structure():
    """On iid noise with no real low-dimensional structure, broken-stick must not
    invent a PCA cutoff -- the honest, data-derived answer is 'no reduction'."""
    rng = np.random.RandomState(1)
    x = rng.randn(200, 50)
    info = resolve_clustering_representation(x)
    assert info["representation"] == "raw"
    assert info["reason"] == "no_component_beats_broken_stick_null"


def test_real_cached_embedding_reduces_far_below_768():
    """The measured effective dimensionality of the real embedding (069/072) is far
    below its raw 768-d width; regression-guard the qualitative direction (a real,
    substantial reduction, not an arbitrary specific number) without depending on the
    external cached file (kept in the tool-level integration test instead)."""
    x = _low_rank_embedding(n_obs=2421, n_features=768, true_rank=20, noise=0.3, seed=2)
    info = resolve_clustering_representation(x)
    assert info["representation"] == "pca"
    assert info["n_pcs"] < 200


@pytest.mark.parametrize("n_obs", [1, 2])
def test_too_few_cells_falls_back_to_raw_without_crashing(n_obs):
    x = np.random.RandomState(0).randn(n_obs, 50)
    info = resolve_clustering_representation(x)
    assert info["representation"] == "raw"
    assert info["n_pcs"] == 0
    assert info["reason"] == "insufficient_rank_for_spectrum"


def test_fewer_cells_than_dimensions_does_not_crash():
    x = np.random.RandomState(0).randn(3, 768)
    info = resolve_clustering_representation(x)
    assert info["max_components"] == 2
    assert info["n_pcs"] <= 2
    assert info["representation"] in ("raw", "pca")


def test_all_identical_points_falls_back_to_raw():
    x = np.ones((100, 50))
    info = resolve_clustering_representation(x)
    assert info["representation"] == "raw"
    assert info["n_pcs"] == 0
    assert info["reason"] == "zero_or_nonfinite_variance"


def test_zero_variance_columns_do_not_crash_or_get_selected():
    x = np.random.RandomState(0).randn(200, 50)
    x[:, :10] = 5.0   # 10 constant columns mixed with real-variance ones
    info = resolve_clustering_representation(x)
    assert info["zero_variance_features"] == 10
    assert np.isfinite(info["n_pcs"])


def test_non_finite_values_fall_back_to_raw_never_silently_zero_filled_and_scored():
    """Task 072's QC caught a NaN path that silently scored maximal; the fail-safe here
    is to refuse to derive a cutoff from partially-fabricated data, not to zero-fill and
    proceed as if nothing happened."""
    x = np.random.RandomState(0).randn(100, 50)
    x[0, 0] = np.nan
    info = resolve_clustering_representation(x)
    assert info["representation"] == "raw"
    assert info["reason"] == "non_finite_values_present"
    assert info["n_nonfinite_values"] == 1

    x2 = np.random.RandomState(0).randn(100, 50)
    x2[3, 3] = np.inf
    info2 = resolve_clustering_representation(x2)
    assert info2["representation"] == "raw"
    assert info2["reason"] == "non_finite_values_present"


def test_explicit_override_is_honored_exactly():
    x = _low_rank_embedding()
    info = resolve_clustering_representation(x, requested_n_pcs=5)
    assert info["method"] == "explicit_override"
    assert info["n_pcs"] == 5
    assert info["representation"] == "pca"


def test_explicit_zero_is_the_raw_escape_hatch():
    x = _low_rank_embedding()
    info = resolve_clustering_representation(x, requested_n_pcs=0)
    assert info["n_pcs"] == 0
    assert info["representation"] == "raw"


def test_explicit_override_beyond_feasible_rank_is_clamped_not_silently_ignored():
    x = _low_rank_embedding(n_obs=300, n_features=100)
    info = resolve_clustering_representation(x, requested_n_pcs=9999)
    assert info["requested_n_pcs"] == 9999          # the request is still recorded
    assert info["n_pcs"] == info["max_components"]  # but clamped to the usable rank
    assert info["n_pcs"] < 9999


def test_rejects_non_2d_input():
    with pytest.raises(ValueError):
        resolve_clustering_representation(np.zeros(10))

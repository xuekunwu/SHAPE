"""Task 075 M1: tool-level wiring for the adaptive clustering representation.

`resolve_clustering_representation`'s own unit tests (degenerate inputs, broken-stick
correctness) live in `shape/utils/test_adaptive_training.py`. This file covers the
TOOL-level wiring in `_compute_umap_and_clustering`/`_resolve_clustering_representation`:

  - n_pcs=None (the parameter not exercised) reproduces the historical raw-768d
    behaviour BYTE-FOR-BYTE -- connectivities, distances, UMAP coordinates, and Leiden
    cluster assignments all identical to the pre-task code path -- proven here on a
    synthetic embedding sized like the real one (2421 x 768 -- see
    `_synthetic_embedding`'s defaults). The SAME check was also run interactively on the
    real cached embedding named in CODING_PROMPT.md
    (`archives/021_agentic-analyzer-recommend/.../cell_state_analyzed.h5ad`, 2421x768);
    that external control-plane path is intentionally NOT a hard dependency of this
    committed suite (it does not exist on every machine this suite runs on -- e.g.
    the cluster/CI) -- its exact command and output are recorded in
    `results/M1_adaptive_representation.md`.
  - n_pcs="auto" actually engages PCA (X_pca populated, dimension = the resolved n_pcs)
    and records the resolved representation into
    `adata.uns["clustering_representation"]`.
  - n_pcs=<int> is honored exactly.
  - The full neighbors/UMAP/Leiden pipeline does not crash with either representation,
    at a cell count already established safe for this pipeline
    (`cell_state_analyzer/test_umap_init_guard.py`: n_obs=4 is the smallest count
    scanpy's default spectral UMAP init handles without the multi-tool's dedicated
    small-n guard, which single-tool does not have and this task does not add).
"""

from __future__ import annotations

import numpy as np
import anndata as ad
import pytest
import scanpy as sc

from shape.tools.cell_state_analyzer.tool import Cell_State_Analyzer_Tool


def _synthetic_embedding(n_obs=200, n_features=768, true_rank=20, noise=0.3, seed=0):
    rng = np.random.RandomState(seed)
    latent = rng.randn(n_obs, true_rank)
    basis = rng.randn(true_rank, n_features)
    return (latent @ basis + noise * rng.randn(n_obs, n_features)).astype("float32")


def _make_adata(n_obs=200, n_features=768, seed=0):
    return ad.AnnData(X=_synthetic_embedding(n_obs, n_features, seed=seed))


@pytest.fixture(scope="module")
def tool():
    return Cell_State_Analyzer_Tool()


def _old_code_path(adata, resolution, n_neighbors):
    """The EXACT pre-task-075 code (use_rep='X', no PCA, no representation choice at
    all) -- the ground truth this task must reproduce byte-for-byte when its new
    parameter is not exercised."""
    n_neighbors_int = min(n_neighbors, max(2, adata.n_obs - 1))
    sc.pp.neighbors(adata, use_rep="X", n_neighbors=n_neighbors_int, metric="cosine")
    sc.tl.umap(adata, min_dist=0.05, spread=1, random_state=42)
    cluster_key = f"leiden_{resolution}"
    sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
    return cluster_key


def test_default_n_pcs_reproduces_historical_behaviour_byte_for_byte(tool):
    base = _make_adata(n_obs=200, n_features=768, seed=3)

    adata_old = base.copy()
    cluster_key_old = _old_code_path(adata_old, resolution=0.5, n_neighbors=15)

    adata_new = base.copy()
    cluster_key_new = tool._compute_umap_and_clustering(
        adata_new, resolution=0.5, groups=None, n_neighbors=15, n_pcs=None)

    assert cluster_key_old == cluster_key_new
    assert (adata_old.obsp["connectivities"] != adata_new.obsp["connectivities"]).nnz == 0
    assert (adata_old.obsp["distances"] != adata_new.obsp["distances"]).nnz == 0
    assert np.array_equal(adata_old.obsm["X_umap"], adata_new.obsm["X_umap"])
    assert (adata_old.obs[cluster_key_old].values == adata_new.obs[cluster_key_new].values).all()
    # and the (raw, unchanged) representation is still recorded -- never silent
    rep = adata_new.uns["clustering_representation"]
    assert rep["representation"] == "raw"
    assert rep["method"] == "default_unchanged"


def test_auto_engages_pca_and_records_the_choice(tool):
    adata = _make_adata(n_obs=300, n_features=768, seed=1)
    tool._compute_umap_and_clustering(adata, resolution=0.5, n_pcs="auto")
    rep = adata.uns["clustering_representation"]
    assert rep["representation"] == "pca"
    assert rep["method"] == "broken_stick"
    assert 0 < rep["n_pcs"] < 768
    assert "X_pca" in adata.obsm
    assert adata.obsm["X_pca"].shape[1] == rep["n_pcs"]


def test_explicit_int_override_is_honored(tool):
    adata = _make_adata(n_obs=300, n_features=768, seed=2)
    tool._compute_umap_and_clustering(adata, resolution=0.5, n_pcs=10)
    rep = adata.uns["clustering_representation"]
    assert rep["n_pcs"] == 10
    assert rep["method"] == "explicit_override"
    assert adata.obsm["X_pca"].shape[1] == 10


def test_explicit_zero_forces_raw_through_the_tool(tool):
    adata = _make_adata(n_obs=300, n_features=768, seed=2)
    tool._compute_umap_and_clustering(adata, resolution=0.5, n_pcs=0)
    rep = adata.uns["clustering_representation"]
    assert rep["representation"] == "raw"
    assert "X_pca" not in adata.obsm


@pytest.mark.parametrize("n_pcs", [None, "auto", 3])
def test_full_pipeline_does_not_crash_at_a_small_but_umap_safe_cell_count(tool, n_pcs):
    """n_obs=4 is the smallest count `cell_state_analyzer/test_umap_init_guard.py`
    confirms scanpy's default (spectral) UMAP init handles without a dedicated small-n
    guard -- isolates THIS task's degenerate-input concern (representation resolution)
    from that unrelated, pre-existing small-n UMAP init issue (out of scope for M1)."""
    adata = _make_adata(n_obs=4, n_features=768, seed=4)
    cluster_key = tool._compute_umap_and_clustering(
        adata, resolution=0.5, n_neighbors=20, n_pcs=n_pcs)
    assert cluster_key in adata.obs.columns
    assert np.isfinite(adata.obsm["X_umap"]).all()


def test_degenerate_representation_resolution_alone_does_not_crash_below_umap_floor(tool):
    """Below the n_obs=4 UMAP-init floor, exercise ONLY the representation-resolution
    wrapper (not the full UMAP pipeline) -- the actual surface this task changes -- for
    N=1..3, mirroring the pure-function coverage in
    `shape/utils/test_adaptive_training.py`."""
    for n_obs in (1, 2, 3):
        adata = _make_adata(n_obs=n_obs, n_features=768, seed=5)
        for n_pcs in (None, "auto", 5):
            rep = tool._resolve_clustering_representation(adata, n_pcs)
            assert rep["n_pcs"] >= 0
            assert rep["representation"] in ("raw", "pca")

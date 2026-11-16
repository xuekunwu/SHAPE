"""Exact k-nearest-neighbours on the GPU (torch), for the readouts that otherwise run pynndescent /
sklearn / a numpy argsort on the CPU: scanpy's neighbour graph, the scib-metrics graph metrics and the
batch-corrector diagnostics.

Off unless SHAPE_GPU_KNN=1 (`enabled()`): the CPU paths are approximate (pynndescent) or differ in
tie order, so switching changes the neighbour sets slightly and a run must not change mid-experiment.
Without CUDA the same code runs on the CPU with torch (exact, multi-threaded)."""
from __future__ import annotations

import os

import numpy as np
import torch

ENV = "SHAPE_GPU_KNN"
CHUNK = 4096                    # query rows per block: CHUNK x n x 8 bytes of distances


def enabled() -> bool:
    return os.environ.get(ENV, "") == "1" and torch.cuda.is_available()


def knn(x, k: int, metric: str = "euclidean", include_self: bool = True, chunk: int = CHUNK):
    """The k nearest rows of every row of `x` (n, d): (indices int64 (n, k), distances float64 (n, k)),
    nearest first. include_self: column 0 is the row itself (distance 0) and k - 1 others follow, the
    layout scanpy and scib-metrics expect; otherwise the k nearest other rows. metric: "euclidean" or
    "cosine" (1 - cosine similarity). Exact, float64."""
    if metric not in ("euclidean", "cosine"):
        raise ValueError(f"metric must be euclidean or cosine, got {metric!r}")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = torch.as_tensor(np.ascontiguousarray(x), dtype=torch.float64, device=device)
    n = data.shape[0]
    k = int(min(k, n if include_self else n - 1))
    if metric == "cosine":
        data = data / data.norm(dim=1, keepdim=True).clamp_min(1e-12)
    sq = (data * data).sum(1)
    idx = torch.empty((n, k), dtype=torch.int64, device=device)
    dist = torch.empty((n, k), dtype=torch.float64, device=device)
    for start in range(0, n, chunk):
        stop = min(start + chunk, n)
        block = data[start:stop] @ data.T
        d = (1.0 - block) if metric == "cosine" else (sq[start:stop, None] + sq[None, :] - 2.0 * block)
        rows = torch.arange(stop - start, device=device)
        # the row itself first (include_self) or never
        d[rows, start + rows] = -1.0 if include_self else float("inf")
        values, order = torch.topk(d, k, dim=1, largest=False, sorted=True)
        if include_self:
            values[:, 0] = 0.0
        values = values.clamp_min(0.0)
        idx[start:stop], dist[start:stop] = order, (values if metric == "cosine" else values.sqrt())
    return idx.cpu().numpy(), dist.cpu().numpy()


class ScanpyKnnTransformer:
    """`transformer=` of `scanpy.pp.neighbors`: the exact GPU kNN as the sparse distance graph scanpy
    reads (n_neighbors entries per row, the cell itself first at distance 0)."""

    def __init__(self, n_neighbors: int = 15, metric: str = "euclidean"):
        self.n_neighbors, self.metric = int(n_neighbors), metric

    def get_params(self, deep: bool = True):
        return {"n_neighbors": self.n_neighbors, "metric": self.metric}

    def set_params(self, **params):
        for key, value in params.items():
            setattr(self, key, value)
        return self

    def fit(self, x, y=None):
        return self

    def fit_transform(self, x, y=None):
        from scipy.sparse import csr_matrix

        idx, dist = knn(x, self.n_neighbors, self.metric, include_self=True)
        n, k = idx.shape
        return csr_matrix((dist.ravel(), idx.ravel(), np.arange(0, n * k + 1, k)), shape=(n, n))

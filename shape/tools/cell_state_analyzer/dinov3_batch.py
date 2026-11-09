"""Client for the DINOv3 embedding service (mirrors sam_batch's client helpers).

  from shape.tools.cell_state_analyzer.dinov3_batch import dino_health, dino_embed, dino_embed_paths
  URL = "http://127.0.0.1:8237"
  emb = dino_embed_paths(URL, ["a_crop.tiff", "b_crop.tiff"])   # -> [N, 768] float32
  emb = dino_embed(URL, [rgb_uint8_a, rgb_uint8_b])             # arrays instead of paths

`url` may be a comma-separated list of server URLs; requests round-robin across them (run
several server processes for throughput, exactly like the SAM service). Requests are
chunked client-side so no single pickle payload gets huge and chunks spread over servers.
"""
from __future__ import annotations

import pickle
import urllib.request

import numpy as np


def _post(url: str, path: str, payload: dict, timeout: float = 600.0) -> dict:
    body = pickle.dumps(payload)
    req = urllib.request.Request(url.rstrip("/") + path, data=body,
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        return pickle.loads(resp.read())


def _get(url: str, path: str, timeout: float = 30.0) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + path, timeout=timeout) as resp:
        return pickle.loads(resp.read())


def dino_health(url: str) -> dict:
    """Health of the first server in `url` (comma-separated allowed)."""
    return _get(url.split(",")[0].strip(), "/health")


def _urls(url: str) -> list[str]:
    return [u.strip() for u in url.split(",") if u.strip()]


def _embed_chunked(url: str, items, key: str, batch_size: int, chunk: int,
                   concurrency: int | None = None) -> np.ndarray:
    """Split into chunks and embed them CONCURRENTLY, round-robining chunk i onto server
    i % n_servers. Concurrency defaults to 2x the number of server processes so every
    server replica stays busy (a sequential client would leave all-but-one replica idle)."""
    from concurrent.futures import ThreadPoolExecutor

    servers = _urls(url)
    path = "/embed_path" if key == "paths" else "/embed"
    chunks = [items[s:s + chunk] for s in range(0, len(items), chunk)]
    if not chunks:
        return np.zeros((0, dino_health(url)["dim"]), np.float32)
    concurrency = concurrency or max(2, 2 * len(servers))

    def _one(i_ch):
        i, ch = i_ch
        srv = servers[i % len(servers)]
        resp = _post(srv, path, {key: ch, "batch_size": batch_size})
        if "error" in resp:
            raise RuntimeError(f"dino server {srv}: {resp['error']}")
        return i, np.asarray(resp["emb"], dtype=np.float32)

    out = [None] * len(chunks)
    with ThreadPoolExecutor(max_workers=concurrency) as ex:
        for i, emb in ex.map(_one, enumerate(chunks)):
            out[i] = emb
    return np.concatenate(out, axis=0)


def dino_embed_paths(url: str, paths, batch_size: int = 256, chunk: int = 1024) -> np.ndarray:
    """Embed crops the server reads from disk (same-host / shared FS). -> [N, dim] float32."""
    return _embed_chunked(url, list(paths), "paths", batch_size, chunk)


def dino_embed(url: str, crops, batch_size: int = 256, chunk: int = 512) -> np.ndarray:
    """Embed in-memory HxWx(1|3|4) crops (bytes transport). -> [N, dim] float32."""
    return _embed_chunked(url, list(crops), "crops", batch_size, chunk)


if __name__ == "__main__":
    import argparse, glob
    ap = argparse.ArgumentParser(description="Smoke-test the DINOv3 service on a few crops.")
    ap.add_argument("--url", default="http://127.0.0.1:8237")
    ap.add_argument("--glob", default=None, help="glob of crop files to embed")
    ap.add_argument("--limit", type=int, default=8)
    a = ap.parse_args()
    print("health:", dino_health(a.url))
    if a.glob:
        paths = sorted(glob.glob(a.glob))[:a.limit]
        emb = dino_embed_paths(a.url, paths)
        print(f"embedded {len(paths)} crops -> {emb.shape} dtype={emb.dtype} "
              f"norm(mean)={np.linalg.norm(emb, axis=1).mean():.3f}")

"""Decoupled high-throughput Cellpose-SAM batch runner.

Segmentation (GPU service) and orchestration (this client) are DISAGGREGATED across processes:

    ┌───────────────────────────────┐        ┌────────────────────────────────────────┐
    │  P INDEPENDENT server procs    │  HTTP  │  sam_batch.py (this client)              │
    │  sam_server.py, R models each  │◀──────▶│  THREAD pool round-robins requests       │
    │  on ONE shared GPU             │  path  │  across the P process URLs               │
    └───────────────────────────────┘        └────────────────────────────────────────┘

Why P PROCESSES, not one process with more models (the key finding): cellpose eval's CPU-side
flow->mask reconstruction holds the GIL, so a SINGLE server process plateaus at ~2x serial no
matter how many model replicas it holds -- measured 0.59 img/s serial, 1.2 img/s at BOTH 4 and 8
in-process replicas (the GPU forward overlaps one CPU post-proc, then the GIL serializes the rest).
Throughput therefore scales with the number of independent OS processes (separate GILs), not with
replicas-per-process. One H200 has ample compute headroom (a single eval is far from saturating it)
and ~140GB VRAM at ~4GB/model, so P x R models fit comfortably and their post-processing runs truly
in parallel on the 96 cores. The client stays a cheap THREAD pool: a SAM call blocks on the socket
(GIL released) while a server process does the work, so many threads keep all P processes fed.

Transport is PATH-based: the request carries the image path + a mask output path; the server reads
the image, segments, writes the mask, and returns only small metadata -- no ~30MB array is pickled
through the server's single GIL-bound process per request (which capped the old /eval transport).

Usage:
    # Let this client boot P server processes (throughput lever is --server-procs, NOT --replicas):
    python -m shape.tools.cell_segmenter.sam_batch \
        --images-dir data/NeurIPS22-CellSeg/images --out-dir /tmp/masks \
        --autostart-sam --server-procs 8 --replicas 2
    # Or point at pre-started servers (comma-separated URLs), one per process:
    #   python -m shape.tools.cell_segmenter.sam_server --port 8137 --replicas 2 &  (x8, ports 8137..)
    #   python -m shape.tools.cell_segmenter.sam_batch ... --sam-url http://127.0.0.1:8137,...:8138,...
"""
from __future__ import annotations

import argparse
import glob
import os
import pickle
import subprocess
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path


# --------------------------------------------------------------------------- #
# SAM service client (pickle over HTTP, matches sam_server.py's /eval)
# --------------------------------------------------------------------------- #
def sam_eval(url: str, img, eval_kwargs: dict, timeout: float = 600.0):
    """One segmentation via the shared SAM server. Returns masks[0] (a 2-D label array).
    Thread-safe: no shared state, the socket call releases the GIL while the server works."""
    body = pickle.dumps({"img": img, "kwargs": eval_kwargs})
    req = urllib.request.Request(url.rstrip("/") + "/eval", data=body,
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = pickle.loads(resp.read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out["masks"][0]


def sam_eval_path(url: str, img_path: str, eval_kwargs: dict, mask_out: str,
                  timeout: float = 600.0) -> dict:
    """PATH transport: the server reads the image + writes the mask on disk (shared FS) and
    returns only small metadata {mask_path, n_cells, shape}. Avoids pickling ~30 MB of arrays
    per request through the server's single GIL-bound process -- the /eval transport's cap."""
    body = pickle.dumps({"img_path": img_path, "kwargs": eval_kwargs, "mask_out": mask_out})
    req = urllib.request.Request(url.rstrip("/") + "/eval_path", data=body,
                                 headers={"Content-Type": "application/octet-stream"})
    with urllib.request.urlopen(req, timeout=timeout) as resp:
        out = pickle.loads(resp.read())
    if "error" in out:
        raise RuntimeError(out["error"])
    return out


def sam_health(url: str, timeout: float = 10.0) -> dict:
    with urllib.request.urlopen(url.rstrip("/") + "/health", timeout=timeout) as resp:
        return pickle.loads(resp.read())


# --------------------------------------------------------------------------- #
# Worker (thread): one image via PATH transport. The GPU service reads the image +
# writes the mask + counts; this thread just carries small metadata. Threads are
# I/O-bound here (they block on the socket), so many keep the GPU pool saturated.
# --------------------------------------------------------------------------- #
def _segment_path(image_path: str, url: str, eval_kwargs: dict, out_dir: str,
                  save_masks: bool) -> dict:
    stem = Path(image_path).stem
    mask_out = str(Path(out_dir) / f"{stem}_pred.tiff")
    r = sam_eval_path(url, image_path, eval_kwargs, mask_out)
    if not save_masks:
        try:
            os.remove(mask_out)
        except OSError:
            pass
    return {"image": stem, "n_cells": r["n_cells"], "shape": r["shape"]}


# PICKLE-transport worker (fallback): the server returns the mask ARRAY; this thread carries
# it and a CPU-pool process does the post-proc. Kept for callers that need the array client-side.
def _segment_pickle(image_path: str, url: str, eval_kwargs: dict):
    from skimage.io import imread
    img = imread(image_path)
    mask = sam_eval(url, img, eval_kwargs)
    return image_path, mask


def _postprocess(image_path: str, mask, out_dir: str, save_masks: bool) -> dict:
    import numpy as np
    stem = Path(image_path).stem
    m = np.asarray(mask)
    n_cells = int(len(np.unique(m)) - 1)
    if save_masks:
        import tifffile
        tifffile.imwrite(str(Path(out_dir) / f"{stem}_pred.tiff"), m.astype(np.uint16))
    return {"image": stem, "n_cells": n_cells, "shape": list(m.shape)}


def _wait_healthy(url: str, tries: int = 300) -> dict:
    for _ in range(tries):
        try:
            h = sam_health(url, timeout=3)
            if h.get("status") == "ok":
                return h
        except Exception:
            time.sleep(1)
    raise SystemExit(f"[sam-batch] SAM server {url} did not come up in time.")


def _autostart_sam(port0: int, procs: int, replicas: int):
    """Launch `procs` INDEPENDENT server processes on consecutive ports, `replicas` models each.

    Independent processes (not one process with more threads) is the whole point: cellpose eval's
    CPU-side flow->mask reconstruction holds the GIL, so a single process plateaus at ~2x serial
    no matter how many model replicas it holds (measured: 4 and 8 replicas both ~1.2 img/s). Each
    OS process has its own GIL, so P processes give ~P x that -- until the GPU's compute saturates
    (one eval is far from full utilization, and VRAM is ~4GB/model out of ~140GB, so there is lots
    of room). Total concurrent evals = procs x replicas; the client round-robins requests across
    the process URLs so their GIL-bound post-processing overlaps on the many CPU cores."""
    urls, handles = [], []
    for i in range(procs):
        port = port0 + i
        handles.append(subprocess.Popen(
            [sys.executable, "-m", "shape.tools.cell_segmenter.sam_server",
             "--port", str(port), "--replicas", str(replicas)]))
        urls.append(f"http://127.0.0.1:{port}")
    try:
        for url in urls:
            h = _wait_healthy(url)
            print(f"[sam-batch] SAM server up @ {url}: {h['replicas']}x{h['model_type']} on "
                  f"{h['device']}", flush=True)
    except BaseException:  # a server failed to come up -> don't leak the ones that did
        for proc in handles:
            proc.terminate()
        raise
    return handles, urls


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images-dir", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--sam-url", default="http://127.0.0.1:8137",
                    help="Existing server URL(s), comma-separated, when NOT autostarting.")
    ap.add_argument("--autostart-sam", action="store_true",
                    help="Boot the SAM server(s) here (uses --server-procs x --replicas) and stop "
                         "them afterwards.")
    ap.add_argument("--sam-port", type=int, default=8137, help="First port; procs use consecutive.")
    ap.add_argument("--server-procs", type=int, default=8,
                    help="INDEPENDENT server processes to launch (--autostart-sam). Throughput "
                         "scales with THIS, not --replicas: each process is GIL-capped at ~2x "
                         "serial, so parallelism comes from separate processes. VRAM caps total "
                         "models (procs x replicas x ~4GB).")
    ap.add_argument("--replicas", type=int, default=2,
                    help="Model instances PER server process (2 saturates one process's ~2x GIL "
                         "ceiling; more per process does not help -- raise --server-procs instead).")
    ap.add_argument("--concurrency", type=int, default=0,
                    help="In-flight SAM requests (I/O threads). 0 = auto (2 x total models). Keep "
                         ">= total models so no server process starves; excess queues server-side.")
    ap.add_argument("--limit", type=int)
    ap.add_argument("--no-save-masks", action="store_true", help="Count only; do not write _pred.tiff.")
    ap.add_argument("--diameter", type=float, default=30.0)
    ap.add_argument("--flow-threshold", type=float, default=0.4)
    ap.add_argument("--cellprob-threshold", type=float, default=0.0)
    ap.add_argument("--min-size", type=int, default=15)
    args = ap.parse_args()

    exts = (".png", ".bmp", ".tif", ".tiff", ".jpg", ".jpeg")
    images = sorted(p for p in glob.glob(os.path.join(args.images_dir, "*"))
                    if os.path.splitext(p)[1].lower() in exts and not p.endswith("_seg"))
    if args.limit:
        images = images[:args.limit]
    if not images:
        raise SystemExit(f"[sam-batch] no images under {args.images_dir}")
    os.makedirs(args.out_dir, exist_ok=True)

    sam_procs, urls = [], [u.strip() for u in args.sam_url.split(",") if u.strip()]
    total_models = None
    if args.autostart_sam:
        sam_procs, urls = _autostart_sam(args.sam_port, args.server_procs, args.replicas)
        total_models = args.server_procs * args.replicas
    else:
        total_models = 0
        for u in urls:
            h = sam_health(u)
            total_models += int(h.get("replicas", 1))
            print(f"[sam-batch] using SAM server {u}: {h['replicas']}x{h['model_type']} on "
                  f"{h['device']} (idle {h['idle']})", flush=True)
    # Auto concurrency: 2x total models keeps every server process fed without oversubscribing.
    concurrency = args.concurrency if args.concurrency > 0 else max(4, 2 * total_models)

    eval_kwargs = dict(diameter=args.diameter, channels=[0, 0],
                       flow_threshold=args.flow_threshold,
                       cellprob_threshold=args.cellprob_threshold, min_size=args.min_size)

    print(f"[sam-batch] {len(images)} images | {len(urls)} server proc(s) x {args.replicas} "
          f"models | {concurrency} SAM threads (path transport, round-robin) -> {args.out_dir}",
          flush=True)
    t0 = time.perf_counter()
    results, errors = [], 0
    save_masks = not args.no_save_masks
    try:
        # PATH transport + ROUND-ROBIN across independent server processes: each process does seg +
        # mask write + count on its side and returns only small metadata, so a plain thread pool
        # keeps all processes busy with no large-array pickle. Requests are spread across the P
        # process URLs so their GIL-bound post-processing runs in parallel (separate GILs) -- the
        # actual throughput lever (a single process plateaus at ~2x serial; see _autostart_sam).
        with ThreadPoolExecutor(max_workers=concurrency) as io_pool:
            futs = {io_pool.submit(_segment_path, p, urls[k % len(urls)], eval_kwargs,
                                   args.out_dir, save_masks): p
                    for k, p in enumerate(images)}
            for i, sf in enumerate(as_completed(futs), 1):
                try:
                    results.append(sf.result())
                except Exception as exc:
                    errors += 1
                    print(f"  [seg-err] {Path(futs[sf]).stem}: {exc}", flush=True)
                if i % 100 == 0:
                    el = time.perf_counter() - t0
                    print(f"  segmented {i}/{len(images)} | {i/el:.1f} img/s "
                          f"(eta {(len(images)-i)/(i/el):.0f}s)", flush=True)
    finally:
        for proc in sam_procs:
            proc.terminate()
        for proc in sam_procs:
            try:
                proc.wait(timeout=10)
            except Exception:
                proc.kill()
        if sam_procs:
            print(f"[sam-batch] stopped {len(sam_procs)} SAM server(s)", flush=True)

    wall = time.perf_counter() - t0
    import json
    (Path(args.out_dir) / "sam_batch_summary.json").write_text(json.dumps({
        "n_images": len(images), "n_ok": len(results), "n_errors": errors,
        "wall_seconds": round(wall, 2), "throughput_img_per_s": round(len(results) / wall, 2),
        "server_procs": len(urls), "replicas_per_proc": args.replicas,
        "total_models": total_models, "concurrency": concurrency,
        "eval_kwargs": {k: v for k, v in eval_kwargs.items() if k != "channels"},
        "total_cells": sum(r["n_cells"] for r in results),
    }, indent=2), encoding="utf-8")
    print(f"[sam-batch] DONE {len(results)}/{len(images)} in {wall:.0f}s "
          f"= {len(results)/wall:.1f} img/s ({errors} errors) -> {args.out_dir}", flush=True)


if __name__ == "__main__":
    main()

"""Cellpose-SAM inference server: a POOL of K GPU-resident models, served over localhost HTTP.

Motivation
----------
Running the SHAPE agent on many images concurrently otherwise makes every worker load its own
Cellpose-SAM (N workers x one model = VRAM blow-up). This server loads a small POOL of K model
instances ONCE and answers segmentation requests; workers set ``$SHAPE_SAM_SERVER_URL`` and
``Cell_Segmenter_Tool`` delegates its one ``model.eval`` call here instead of loading anything.

Why a pool (not a single model + lock)
--------------------------------------
Measured on one H200: a single model saturates at ~24-28 concurrent workers (T~2s/eval, ~26
eval/min) -- beyond that the lock is the bottleneck and the queue grows unbounded. A single
eval does NOT saturate the GPU (the ~2s is transformer forward + CPU-side mask reconstruction),
and VRAM is huge (143GB / ~4GB per model), so hosting K models lets up to K evals run
concurrently -- one model's CPU post-processing overlaps another's GPU forward. Measured
aggregate: K=4 -> ~1.7x throughput (pushes saturation to ~42 workers). Requests borrow an idle
model from a thread-safe pool (blocking if all K are busy) and return it when done; different
requests use DISTINCT model objects, so concurrent eval is safe (no shared module state).

Heterogeneous hyperparameters are fine: diameter/flow_threshold/cellprob_threshold/... are
per-call ``eval`` kwargs (not model state), so every request carries its own set -- the pool
never merges requests, so there is no cross-request parameter contamination.

Contract (unchanged; workers/client need no changes)
----------------------------------------------------
  POST /eval   body = pickle({"img": np.ndarray, "kwargs": {...eval params...}})
               resp = pickle({"masks": [np.ndarray], "flows": [...]})  or {"error": str}
  GET  /health resp = pickle({"status": "ok", "model_type": str, "device": str,
                              "replicas": int, "idle": int})

Transport is pickle over HTTP bound to 127.0.0.1 ONLY (localhost trust boundary).

Usage
-----
    python -m shape.tools.cell_segmenter.sam_server --port 8137 --replicas 4 &
    export SHAPE_SAM_SERVER_URL=http://127.0.0.1:8137
"""

from __future__ import annotations

import argparse
import pickle
import queue
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

# Module-level state: a pool of loaded models + metadata.
_POOL: "queue.Queue" = queue.Queue()
_REPLICAS = 0
_MODEL_TYPE = "cpsam"
_DEVICE = "cpu"


def _auto_replicas(per_model_gb: float = 6.0, cap: int = 4) -> int:
    """Pick K from free VRAM (~per_model_gb each), capped (threaded scaling plateaus ~4)."""
    try:
        import torch
        if torch.cuda.is_available():
            free, _total = torch.cuda.mem_get_info()
            return max(1, min(cap, int((free / 1e9) // per_model_gb)))
    except Exception:
        pass
    return 1


def _load_models(replicas: int, model_path: str | None, model_name: str) -> None:
    """Load `replicas` Cellpose-SAM instances (GPU if available, CPU fallback) into the pool."""
    global _REPLICAS, _MODEL_TYPE, _DEVICE
    from cellpose import models
    from shape.utils.device import cellpose_precision_kwargs, resolve_torch_device

    device, warn = resolve_torch_device()
    if warn:
        print(f"[sam-server] {warn}")
    pretrained = model_path or "cpsam"
    for i in range(replicas):
        try:
            m = models.CellposeModel(device=device, pretrained_model=pretrained,
                                     **cellpose_precision_kwargs(device))
        except Exception as exc:  # GPU init failed -> CPU (only worth trying once)
            import torch
            print(f"[sam-server] GPU init failed ({exc}); retrying replica {i} on CPU")
            device = torch.device("cpu")
            m = models.CellposeModel(device=device, pretrained_model=pretrained,
                                     **cellpose_precision_kwargs(device))
        _POOL.put(m)
        print(f"[sam-server] loaded replica {i + 1}/{replicas}")
    _REPLICAS = replicas
    _MODEL_TYPE = "custom" if model_path else model_name
    _DEVICE = str(device)
    print(f"[sam-server] pool ready: {replicas} x {_MODEL_TYPE} on {_DEVICE}")


class _Handler(BaseHTTPRequestHandler):
    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802 (http.server API)
        if self.path == "/health":
            self._send(200, pickle.dumps(
                {"status": "ok", "model_type": _MODEL_TYPE, "device": _DEVICE,
                 "replicas": _REPLICAS, "idle": _POOL.qsize()}))
        else:
            self._send(404, b"")

    def do_POST(self):  # noqa: N802
        if self.path not in ("/eval", "/eval_path"):
            self._send(404, b"")
            return
        model = None
        try:
            n = int(self.headers.get("Content-Length", "0"))
            req = pickle.loads(self.rfile.read(n))
            kwargs = dict(req.get("kwargs") or {})
            if self.path == "/eval_path":
                # PATH transport (same-host, shared FS): read the image + write the mask on the
                # server, return only small metadata. Avoids pickling ~30 MB of arrays per request
                # in this single GIL-bound process -- the throughput cap of the /eval transport.
                import numpy as np
                from skimage.io import imread
                img = imread(req["img_path"])
                model = _POOL.get()
                masks = model.eval([img], **kwargs)[0]
                mask = np.asarray(masks[0]).astype(np.uint16)
                out_path = req["mask_out"]
                if str(out_path).endswith(".npy"):
                    np.save(out_path, mask)
                else:
                    import tifffile
                    tifffile.imwrite(out_path, mask)
                self._send(200, pickle.dumps({"mask_path": out_path,
                                              "n_cells": int(len(np.unique(mask)) - 1),
                                              "shape": list(mask.shape)}))
            else:
                img = req["img"]
                model = _POOL.get()                # borrow an idle model (blocks if all busy)
                out = model.eval([img], **kwargs)  # distinct model per thread -> safe concurrency
                masks, flows = out[0], out[1]      # drop styles (the tool discards it)
                # Compute the flow-error QC HERE, on the model's GPU: it re-runs cellpose
                # dynamics (_extend_centers_gpu), ~0.5s on GPU but ~34s on the client's CPU --
                # the real cost that made a remote-SAM client slow. The client reads this value
                # instead of recomputing it (req asks via "with_flow_error"; older clients that
                # omit the key still get masks/flows and recompute locally, unchanged).
                flow_error = None
                if req.get("with_flow_error"):
                    try:
                        import numpy as np
                        from shape.utils.segmentation_qc import compute_flow_error
                        flow_error = compute_flow_error(np.asarray(masks[0]), flows,
                                                        device=getattr(model, "device", None))
                    except Exception:
                        flow_error = None          # QC is best-effort; never fail an eval on it
                self._send(200, pickle.dumps({"masks": masks, "flows": flows,
                                              "flow_error": flow_error}))
        except Exception as exc:
            import traceback
            traceback.print_exc()
            self._send(200, pickle.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            if model is not None:
                _POOL.put(model)                   # return to the pool even on error

    def log_message(self, *_args):
        pass  # quiet -- one line per request would drown the benchmark log


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind host. Non-localhost requires --allow-remote (pickle transport).")
    ap.add_argument("--allow-remote", action="store_true",
                    help="Permit a non-localhost bind. SECURITY: the transport is pickle, so any "
                         "host that can reach the port can run arbitrary code here -- only on a "
                         "firewalled, trusted private network.")
    ap.add_argument("--port", type=int, default=8137)
    ap.add_argument("--replicas", type=int, default=0,
                    help="Number of model instances in the pool (max concurrent evals). "
                         "0 = auto-size from free VRAM (~6GB each, capped at 4).")
    ap.add_argument("--model-path", default=None,
                    help="Custom pretrained model path; default loads official 'cpsam'.")
    ap.add_argument("--model-name", default="cpsam")
    args = ap.parse_args()

    remote = args.host not in ("127.0.0.1", "localhost", "::1")
    if remote and not args.allow_remote:
        raise SystemExit("[sam-server] refusing non-localhost bind (pickle transport = remote code "
                         "execution if reachable). Pass --allow-remote to bind on a TRUSTED, "
                         "firewalled private network.")
    if remote:
        print("=" * 78)
        print(f"[sam-server] !!! SECURITY: binding {args.host}:{args.port} with a PICKLE transport.")
        print("[sam-server] !!! Anyone who can reach this port can execute ARBITRARY CODE on this")
        print("[sam-server] !!! host. Keep the port firewalled to a trusted private network only.")
        print("=" * 78)
    replicas = args.replicas if args.replicas > 0 else _auto_replicas()
    print(f"[sam-server] loading {replicas} replica(s) ...")
    _load_models(replicas, args.model_path, args.model_name)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[sam-server] serving on http://{args.host}:{args.port} "
          f"(POST /eval, GET /health) with {replicas} replica(s) -- Ctrl-C to stop")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("[sam-server] shutting down")
        server.shutdown()


if __name__ == "__main__":
    main()

"""Shared DINOv3 GPU embedding service (mirrors the Cellpose-SAM service).

A long-lived HTTP service that holds a pool of DINOv3 backbones on the GPU and answers
zero-shot embedding requests, so many crop-producing workers share ONE loaded backbone
instead of each re-loading the 768-d ViT. Same design + transport as
`shape/tools/cell_segmenter/sam_server.py`:

  GET  /health      -> pickle({"status","backbone","device","dim","replicas","idle"})
  POST /embed       body = pickle({"crops":[HxWx3 uint8 ndarray], "batch_size"})
                    resp = pickle({"emb": float32 ndarray [N, dim]})  or {"error"}
  POST /embed_path  body = pickle({"paths":[str], "batch_size"})   (server reads the crops)
                    resp = pickle({"emb": float32 ndarray [N, dim]})  or {"error"}

The embedding is the pretrained DINOv3 backbone CLS token (`use_projector=False`), read
through the SAME `DinoV3Projector` the Cell_State_Analyzer uses, so it is byte-identical to
the tool's zero-shot path: Resize((224,224)) -> ToTensor[0,1] -> identity RGB adapter ->
imagenet-normalize -> backbone CLS. No training, no projector head.

Transport is pickle over HTTP bound to 127.0.0.1 by default (localhost trust boundary);
a non-localhost bind needs --allow-remote (see the SAM service security note).

  # Must be started from a REAL (non-sandboxed) shell -- inside the Claude Code tool sandbox
  # background CUDA processes are SIGKILLed. Launch via the `!` prefix or your own terminal:
  setsid nohup python -u -m shape.tools.cell_state_analyzer.dinov3_server \
      --port 8237 --replicas 2 > dinov3_service_8237.log 2>&1 &
"""
from __future__ import annotations

import argparse
import pickle
import queue
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer

import numpy as np

_POOL: "queue.Queue" = queue.Queue()
_REPLICAS = 0
_DEVICE = "cpu"
_DIM = 768
_BACKBONE = "dinov3_vitb16"
_LOCK = threading.Lock()


def _load_models(replicas: int, backbone_name: str) -> None:
    """Load `replicas` DINOv3 backbones (GPU if available) into the pool, eval + frozen."""
    global _REPLICAS, _DEVICE, _DIM, _BACKBONE
    import torch
    from torchvision import transforms
    from shape.tools.cell_state_analyzer.tool import DinoV3Projector

    _DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
    _BACKBONE = backbone_name
    eval_tf = transforms.Compose([transforms.Resize((224, 224)), transforms.ToTensor()])
    for i in range(replicas):
        # input_channels=3 -> the channel adapter initialises to a DETERMINISTIC identity
        # (not random), so the frozen backbone CLS is a faithful zero-shot embedding.
        model = DinoV3Projector(backbone_name=backbone_name, input_channels=3,
                                freeze_backbone=True)
        model.eval().to(_DEVICE)
        _DIM = int(model.projector[0].in_features)   # backbone feature dim (768 for vitb16)
        _POOL.put((model, eval_tf, torch))
        print(f"[dino-server] loaded replica {i + 1}/{replicas}", flush=True)
    _REPLICAS = replicas
    print(f"[dino-server] pool ready: {replicas} x {backbone_name} (dim={_DIM}) on {_DEVICE}",
          flush=True)


def _to_rgb_uint8(arr: np.ndarray) -> np.ndarray:
    """Coerce any crop (gray / RGBA / uint16 / float) to HxWx3 uint8 for PIL."""
    a = np.asarray(arr)
    if a.dtype != np.uint8:                       # normalize dynamic range per-crop
        a = a.astype(np.float32)
        lo, hi = float(a.min()), float(a.max())
        a = np.zeros_like(a) if hi <= lo else (a - lo) / (hi - lo) * 255.0
        a = a.astype(np.uint8)
    if a.ndim == 2:
        a = np.stack([a] * 3, axis=-1)
    elif a.ndim == 3 and a.shape[2] == 1:
        a = np.repeat(a, 3, axis=2)
    elif a.ndim == 3 and a.shape[2] >= 3:
        a = a[:, :, :3]
    return np.ascontiguousarray(a)


def _embed(model_bundle, crops, batch_size: int) -> np.ndarray:
    from PIL import Image
    model, eval_tf, torch = model_bundle
    out = []
    with torch.no_grad():
        for s in range(0, len(crops), batch_size):
            chunk = crops[s:s + batch_size]
            batch = torch.stack([eval_tf(Image.fromarray(_to_rgb_uint8(c))) for c in chunk])
            batch = batch.to(_DEVICE)
            cls = model.encode_backbone(batch)     # backbone CLS token [B, dim]
            out.append(cls.float().cpu().numpy())
    return np.concatenate(out, axis=0).astype(np.float32) if out else np.zeros((0, _DIM), np.float32)


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *a):  # silence per-request logging
        pass

    def _send(self, code: int, body: bytes) -> None:
        self.send_response(code)
        self.send_header("Content-Type", "application/octet-stream")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def do_GET(self):  # noqa: N802
        if self.path == "/health":
            self._send(200, pickle.dumps(
                {"status": "ok", "backbone": _BACKBONE, "device": _DEVICE,
                 "dim": _DIM, "replicas": _REPLICAS, "idle": _POOL.qsize()}))
        else:
            self._send(404, pickle.dumps({"error": "not found"}))

    def do_POST(self):  # noqa: N802
        if self.path not in ("/embed", "/embed_path"):
            self._send(404, pickle.dumps({"error": "not found"}))
            return
        n = int(self.headers.get("Content-Length", 0))
        model_bundle = None
        try:
            req = pickle.loads(self.rfile.read(n))
            batch_size = int(req.get("batch_size", 256))
            if self.path == "/embed_path":
                from PIL import Image
                crops = [np.asarray(Image.open(p)) for p in req["paths"]]
            else:
                crops = req["crops"]
            model_bundle = _POOL.get()             # block until a replica is free
            emb = _embed(model_bundle, crops, batch_size)
            self._send(200, pickle.dumps({"emb": emb}))
        except Exception as exc:
            self._send(200, pickle.dumps({"error": f"{type(exc).__name__}: {exc}"}))
        finally:
            if model_bundle is not None:
                _POOL.put(model_bundle)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__,
                                 formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--host", default="127.0.0.1",
                    help="Bind host. Non-localhost requires --allow-remote (pickle transport).")
    ap.add_argument("--allow-remote", action="store_true",
                    help="Permit a non-localhost bind. SECURITY: pickle transport = RCE surface.")
    ap.add_argument("--port", type=int, default=8237)
    ap.add_argument("--replicas", type=int, default=2,
                    help="DINOv3 backbones in the pool (max concurrent /embed). vitb16 ~0.5GB each.")
    ap.add_argument("--backbone", default="dinov3_vitb16")
    args = ap.parse_args()

    remote = args.host not in ("127.0.0.1", "localhost", "::1")
    if remote and not args.allow_remote:
        raise SystemExit("[dino-server] refusing non-localhost bind (pickle transport = remote "
                         "code execution). Pass --allow-remote only on a trusted private network.")
    if remote:
        print(f"[dino-server] !!! SECURITY: binding {args.host}:{args.port} with a PICKLE transport.")

    print(f"[dino-server] loading {args.replicas} replica(s) ...", flush=True)
    _load_models(args.replicas, args.backbone)
    server = ThreadingHTTPServer((args.host, args.port), _Handler)
    print(f"[dino-server] serving http://{args.host}:{args.port} "
          f"(POST /embed, /embed_path, GET /health) -- Ctrl-C to stop", flush=True)
    server.serve_forever()


if __name__ == "__main__":
    main()

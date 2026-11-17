# Shared Cellpose-SAM GPU service

A long-lived HTTP service that holds a **pool of Cellpose-SAM models on the GPU** and answers
segmentation requests, so many workers (or other machines) share one GPU instead of each loading
its own model. Two files:

| File | Role |
|---|---|
| `sam_server.py` | The service: `K` GPU-resident models behind `ThreadingHTTPServer`. |
| `sam_batch.py`  | A decoupled client: fan out many images across one or more server processes. |

Transport is **pickle over HTTP**. `sam_server` also exposes the client helpers `sam_health`,
`sam_eval`, `sam_eval_path` from `sam_batch`.

---

## 0. TL;DR

```bash
cd /mnt/filesystem-r8/SHAPE

# start (localhost only -- safe default)
setsid nohup python -u -m shape.tools.cell_segmenter.sam_server \
  --port 8137 --replicas 4 > sam_service_8137.log 2>&1 &

# use it from the SHAPE agent
export SHAPE_SAM_SERVER_URL=http://127.0.0.1:8137
```

To reach it **from other machines on the internal network**, see §3 (needs `--allow-remote` and a
firewall). To go **fast**, see §5 (run more server *processes*, not more replicas).

---

## 1. Start the service

### Localhost (default, safe)
```bash
cd /mnt/filesystem-r8/SHAPE
setsid nohup python -u -m shape.tools.cell_segmenter.sam_server \
  --port 8137 --replicas 4 > sam_service_8137.log 2>&1 &
echo "PID $!"
```
- `setsid nohup … &` fully detaches it — survives the launching shell closing.
- `--replicas 4` = 4 models in the pool = up to 4 concurrent `eval`s in this one process.
- Model load takes ~30–60 s. The log prints `pool ready: 4 x cpsam on cuda` when up.

> Must be started from a **real (non-sandboxed) shell**. Inside the Claude Code tool sandbox,
> background CUDA processes are SIGKILLed and do not persist across calls — use the `!` prefix so
> the command runs in your own session, or run it in your own terminal.

### Verify it is up
```bash
python -c "from shape.tools.cell_segmenter.sam_batch import sam_health; print(sam_health('http://127.0.0.1:8137'))"
# -> {'status': 'ok', 'model_type': 'cpsam', 'device': 'cuda', 'replicas': 4, 'idle': 4}
```

---

## 2. Call it

### From the SHAPE agent (the tool auto-delegates)
```bash
export SHAPE_SAM_SERVER_URL=http://127.0.0.1:8137   # or the internal IP, see §3
# Cell_Segmenter_Tool now sends its model.eval to the service instead of loading a model.
```

### From Python (any client)
```python
from shape.tools.cell_segmenter.sam_batch import sam_health, sam_eval, sam_eval_path
from skimage.io import imread

URL = "http://127.0.0.1:8137"
kw = dict(diameter=30.0, channels=[0, 0], flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)

# (a) bytes transport: returns the mask ARRAY to the caller
mask = sam_eval(URL, imread("cell_00001.bmp"), kw)          # -> 2-D label array

# (b) path transport (same-host / shared FS): server reads image + writes mask, returns metadata
meta = sam_eval_path(URL, "cell_00001.bmp", kw, "/tmp/out/cell_00001_pred.tiff")
# -> {'mask_path': '/tmp/out/cell_00001_pred.tiff', 'n_cells': 143, 'shape': [512, 512]}
```

### Batch a whole folder
```bash
python -m shape.tools.cell_segmenter.sam_batch \
  --images-dir data/NeurIPS22-CellSeg/images --out-dir /tmp/masks \
  --sam-url http://127.0.0.1:8137
```

### HTTP endpoints
| Method | Path | Request (pickle) | Response (pickle) |
|---|---|---|---|
| GET  | `/health`    | — | `{status, model_type, device, replicas, idle}` |
| POST | `/eval`      | `{img: ndarray, kwargs}` | `{masks: [ndarray], flows}` or `{error}` |
| POST | `/eval_path` | `{img_path, kwargs, mask_out}` | `{mask_path, n_cells, shape}` or `{error}` |

Use `/eval_path` when the client shares a filesystem with the server (same host / shared mount):
it avoids pickling the ~30 MB image+mask arrays through the server per request.

---

## 3. Reach it from other machines (internal network)

The host's internal IP is on `eth0` (here: **`gpu-node.example.net`** — check yours with `hostname -I`).
A non-localhost bind is **refused by default** and requires an explicit opt-in:

```bash
cd /mnt/filesystem-r8/SHAPE
setsid nohup python -u -m shape.tools.cell_segmenter.sam_server \
  --host 0.0.0.0 --allow-remote --port 8137 --replicas 4 > sam_service_8137.log 2>&1 &
```

From another machine in the same VPC:
```bash
python -c "from shape.tools.cell_segmenter.sam_batch import sam_health; print(sam_health('http://gpu-node.example.net:8137'))"
export SHAPE_SAM_SERVER_URL=http://gpu-node.example.net:8137
# or: sam_batch ... --sam-url http://gpu-node.example.net:8137
```

> ### ⚠ SECURITY — this is a remote-code-execution surface
> The transport is **pickle**, and the server does `pickle.loads` on every request body. Anyone who
> can reach the port can execute **arbitrary code on the host**. `--allow-remote` is only acceptable
> on a **trusted, firewalled private network**:
> - Restrict the port in your **cloud security group / VPC firewall** to trusted internal CIDRs.
> - **Never** expose it to the public internet.
> - The server prints a loud SECURITY banner on a non-localhost bind so an exposed port is never silent.
>
> Safer alternative if the peers can SSH to this host: keep the default localhost bind and tunnel —
> `ssh -L 8137:127.0.0.1:8137 user@gpu-node.example.net`, then call `http://127.0.0.1:8137` on the peer.

---

## 4. Self-test (run on the host, real shell)

```bash
cd /mnt/filesystem-r8/SHAPE
# after the server is up, prove an end-to-end call over the internal IP:
python - <<'PY'
from shape.tools.cell_segmenter.sam_batch import sam_health, sam_eval_path
import glob
URL = "http://gpu-node.example.net:8137"
print("health:", sam_health(URL))
img = sorted(glob.glob("data/NeurIPS22-CellSeg/images/*.bmp"))[0]
kw = dict(diameter=30.0, channels=[0,0], flow_threshold=0.4, cellprob_threshold=0.0, min_size=15)
print("eval:", sam_eval_path(URL, img, kw, "/tmp/selftest_pred.tiff"))
PY
```
Expect `health: {...'status': 'ok'...}` and `eval: {'mask_path': ..., 'n_cells': <N>, 'shape': [...]}`.

---

## 5. Throughput: scale PROCESSES, not replicas

**Measured (one H200):**

| Config | img/s | vs serial |
|---|---|---|
| 1 model, serial | 0.59 | 1.0× |
| 1 process, 4 replicas (threads) | 1.17 | 2.0× |
| 1 process, 8 replicas (threads) | 1.20 | 2.0× |

A **single server process plateaus at ~2× serial** no matter how many replicas it holds: cellpose
`eval`'s CPU-side flow→mask reconstruction holds the Python GIL, so only the GPU forward overlaps.
Throughput scales with **independent server processes** (separate GILs), not replicas-per-process.

The GPU (single eval is far from saturating it) and VRAM (~4 GB/model of ~140 GB) have lots of room,
so run several processes on consecutive ports. `sam_batch` can launch and round-robin them for you:

```bash
python -m shape.tools.cell_segmenter.sam_batch \
  --images-dir data/NeurIPS22-CellSeg/images --out-dir /tmp/masks \
  --autostart-sam --server-procs 8 --replicas 2      # 8 processes x 2 models = 16 evals in flight
```
`--server-procs P` is the throughput lever (expected ≈ P × 1.2 img/s until the GPU saturates); it
launches P servers on ports `8137, 8138, …`, round-robins requests, and stops them at the end.
The run writes `sam_batch_summary.json` (`throughput_img_per_s`, `total_models`, …) to `--out-dir`.

For a **persistent** multi-process service, start the servers yourself and point the client at all of
them (comma-separated), so the client round-robins:
```bash
for p in 8137 8138 8139 8140 8141 8142 8143 8144; do
  setsid nohup python -u -m shape.tools.cell_segmenter.sam_server \
    --host 0.0.0.0 --allow-remote --port $p --replicas 2 > sam_service_$p.log 2>&1 &
done
# client:
sam_batch ... --sam-url http://gpu-node.example.net:8137,http://gpu-node.example.net:8138,...:8144
```

Where the ceiling actually sits: on a single GPU it is most likely the **GPU** (tens of img/s — sweep
`--server-procs` and watch `throughput_img_per_s` stop rising). With more GPUs, the **96 CPU cores**
become the wall (the post-processing is CPU-bound at ~0.8 s/eval → ~1 core each), around ~100 img/s.

---

## 6. Manage

```bash
grep -iE "security|pool ready|serving|error" sam_service_8137.log   # inspect
pkill -f "sam_server --port 8137"                                   # stop one
pkill -f "sam_server"                                               # stop all
```

## 7. Key flags

`sam_server`: `--port` · `--replicas N` (0 = auto-size from free VRAM) · `--host` · `--allow-remote`
· `--model-path` (custom weights; default official `cpsam`).

`sam_batch`: `--images-dir` · `--out-dir` · `--sam-url a,b,c` (existing servers) **or**
`--autostart-sam --server-procs P --replicas R` · `--concurrency` (0 = auto = 2 × total models) ·
`--limit` · `--no-save-masks` · `--diameter/--flow-threshold/--cellprob-threshold/--min-size`.

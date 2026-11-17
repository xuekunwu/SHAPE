#!/bin/bash
# One-shot GPU model services for the SHAPE benchmarks: start / stop / status the shared
# Cellpose-SAM and DINOv3 HTTP services on a GPU node, bound for remote access so CPU
# orchestration nodes on the same network can delegate to them (the
# full_CPJUMP1 segment pattern). Run this ON THE GPU NODE, from a real shell (background CUDA
# is SIGKILLed inside a sandbox).
#
#   bash benchmarks/gpu_services.sh start  [SAM_PROCS] [SAM_REPLICAS] [DINO_PROCS]
#   bash benchmarks/gpu_services.sh status
#   bash benchmarks/gpu_services.sh stop
#   bash benchmarks/gpu_services.sh urls           # print the SAM_URLS / DINO_URL to export
#
# Defaults: 4 SAM procs x 2 replicas (ports 8137-8140), 1 DINO proc x 2 replicas (port 8237).
# After `start`, it prints the exact SAM_URLS / DINO_URL (with this node's IP) to export on
# the CPU node. SECURITY: the transport is pickle (a remote-code-execution surface); only use
# --allow-remote on a trusted, firewalled private network.
set -eo pipefail
cd "$(cd "$(dirname "$0")/.." && pwd)"           # repo root
export PYTHONPATH="$PWD:${PYTHONPATH:-}"
set -a; [ -f .env ] && . ./.env; set +a
export HUGGINGFACE_TOKEN="${HUGGINGFACE_TOKEN:-$HF_TOKEN}"   # DINOv3 weights load needs it
PY="${PYTHON:-python}"
LOGDIR="$PWD/benchmarks/.gpu_service_logs"; mkdir -p "$LOGDIR"

CMD="${1:-status}"
SAM_PROCS="${2:-4}"; SAM_REPLICAS="${3:-2}"; DINO_PROCS="${4:-1}"
SAM_PORT0=8137; DINO_PORT0=8237

node_ip() { hostname -I 2>/dev/null | awk '{print $1}'; }

health() {  # health PORT -> prints status or DOWN
  "$PY" - "$1" <<'PY' 2>/dev/null || echo DOWN
import sys,urllib.request,pickle
try:
    r=pickle.loads(urllib.request.urlopen(f"http://127.0.0.1:{sys.argv[1]}/health",timeout=4).read())
    print(f"{r.get('status')} {r.get('device')} replicas={r.get('replicas')} idle={r.get('idle')}")
except Exception as e: print("DOWN")
PY
}

sam_ports()  { seq "$SAM_PORT0" $((SAM_PORT0 + SAM_PROCS - 1)); }
dino_ports() { seq "$DINO_PORT0" $((DINO_PORT0 + DINO_PROCS - 1)); }

do_start() {
  echo "[gpu-services] starting SAM x$SAM_PROCS (replicas $SAM_REPLICAS) + DINO x$DINO_PROCS on $(node_ip)"
  for p in $(sam_ports); do
    setsid nohup "$PY" -u -m shape.tools.cell_segmenter.sam_server \
      --host 0.0.0.0 --allow-remote --port "$p" --replicas "$SAM_REPLICAS" \
      > "$LOGDIR/sam_$p.log" 2>&1 &
  done
  for p in $(dino_ports); do
    setsid nohup "$PY" -u -m shape.tools.cell_state_analyzer.dinov3_server \
      --host 0.0.0.0 --allow-remote --port "$p" --replicas 2 \
      > "$LOGDIR/dinov3_$p.log" 2>&1 &
  done
  echo "[gpu-services] launched; models load in ~30-60s. Poll with: bash $0 status"
  # wait until all are healthy (or 90s)
  for _ in $(seq 1 30); do
    all_up=1
    for p in $(sam_ports) $(dino_ports); do [ "$(health "$p")" = DOWN ] && all_up=0; done
    [ "$all_up" = 1 ] && break || sleep 3
  done
  do_urls
}

do_status() {
  for p in $(sam_ports);  do echo "SAM  $p: $(health "$p")"; done
  for p in $(dino_ports); do echo "DINO $p: $(health "$p")"; done
}

do_urls() {
  ip="$(node_ip)"
  sam=""; for p in $(sam_ports);  do sam="${sam:+$sam,}http://$ip:$p"; done
  dino=""; for p in $(dino_ports); do dino="${dino:+$dino,}http://$ip:$p"; done
  echo "----- export these on the CPU orchestration node -----"
  echo "export SHAPE_SAM_SERVER_URL=\"$sam\""
  echo "export SAM_URLS=\"$sam\""
  echo "export DINO_URL=\"$dino\""
}

do_stop() {
  pkill -f "shape.tools.cell_segmenter.sam_server" 2>/dev/null && echo "[gpu-services] SAM stopped" || echo "[gpu-services] no SAM running"
  pkill -f "shape.tools.cell_state_analyzer.dinov3_server" 2>/dev/null && echo "[gpu-services] DINO stopped" || echo "[gpu-services] no DINO running"
}

case "$CMD" in
  start)  do_start ;;
  status) do_status ;;
  urls)   do_urls ;;
  stop)   do_stop ;;
  *) echo "usage: $0 {start [SAM_PROCS] [SAM_REPLICAS] [DINO_PROCS] | status | urls | stop}"; exit 2 ;;
esac

#!/usr/bin/env bash
# The run dir's non-image files as <run>.noimg.tgz (+ <run>.noimg.list) beside it, for a one-step download.
#
# usage: bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/pack_noimg.sh <run dir> [slurm job id ...]
#   Each job's scheduler .out / .err is copied into <run>/logs/ first. Kept (a whitelist): json, jsonl, csv, md,
#   log, txt, ok, out, err -- never the masks, overlays or figures (png, tif, tiff, npy, jpg, bmp).
#   The last line is "NOIMG <archive> (<n> files, <size>)".
set -eo pipefail

RUN=${1%/}
shift || true
[[ -n "$RUN" && -d "$RUN" ]] || { echo "pack_noimg: no run dir '$RUN'"; exit 2; }
LOGDIR=${SLURM_LOG_DIR:-/scratch/groups/your-group/your-user/log}
mkdir -p "$RUN/logs"
for job in "$@"; do
    out=$(scontrol show job "$job" 2>/dev/null | sed -n 's/.*StdOut=\([^ ]*\).*/\1/p' | head -1) || true
    [[ -f "$out" ]] || out=$(ls "$LOGDIR"/*-"$job".out 2>/dev/null | head -1) || true
    for f in "$out" "${out%.out}.err"; do
        if [[ -n "$f" && -f "$f" ]]; then cp "$f" "$RUN/logs/" && echo "pack_noimg: + logs/$(basename "$f")"; fi
    done
done
( cd "$RUN" && find . -type f \( -name '*.json' -o -name '*.jsonl' -o -name '*.csv' -o -name '*.md' -o -name '*.log' \
      -o -name '*.txt' -o -name '*.ok' -o -name '*.out' -o -name '*.err' \) | sort > "$RUN.noimg.list" )
tar czf "$RUN.noimg.tgz" -C "$RUN" -T "$RUN.noimg.list"
echo "NOIMG $RUN.noimg.tgz ($(wc -l < "$RUN.noimg.list") files, $(du -h "$RUN.noimg.tgz" | cut -f1))"

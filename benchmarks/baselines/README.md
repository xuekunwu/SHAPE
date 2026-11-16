# baselines

General coding-agent (Claude Code / Codex) baselines for the NeurIPS22-CellSeg segmentation and
cropper benchmarks, run as a SHAPE peer: the agent is sandboxed in an isolated workdir outside the
repo (inputs + a task prompt only, never SHAPE's code), runs headless, and is scored with SHAPE's
own metrics (`metrics/instance_ap.py`, `metrics/crop_qc.py`) so the comparison is apples-to-apples.
An outer feedback loop re-prompts the same agent session with its score + critique for up to
`--iters` attempts, keeping the best.

## Dataset

Both benchmarks read the same `NeurIPS22-CellSeg` data as their SHAPE counterparts
(`benchmarks/segmentation_NeurIPS22-CellSeg/`, `benchmarks/cropper_NeurIPS22-CellSeg/`): 1000
heterogeneous microscopy images (brightfield, phase contrast, DIC, fluorescence) with one
ground-truth instance mask each, laid out as

```
<data-root>/images/<name>.{png,bmp,tif,tiff,jpg,jpeg}
<data-root>/labels/<name>_label.tiff
```

`--data-root` defaults to the repo's own `data/NeurIPS22-CellSeg`; validate it first with
`benchmarks/segmentation_NeurIPS22-CellSeg/runners/prepare_dataset.py` (no download step -- the
data is already laid out locally). The segmentation adapter gives the agent raw images and asks it
to discover/install its own segmenter (Cellpose-SAM via the shared SAM service, or anything else);
the cropper adapter gives the agent images + their ground-truth instance masks and asks it to crop
each labeled cell cleanly.

## GPU services (segmentation baseline only)

The segmentation adapter's agent calls the shared Cellpose-SAM service over HTTP (`--sam-url`); the
cropper adapter needs no GPU (it crops from ground-truth masks directly). Start the shared SAM/DINO
services on a GPU node first:

```
bash benchmarks/gpu_services.sh start
bash benchmarks/gpu_services.sh urls      # prints this node's SAM_URLS / DINO_URL to export
```

## Running the baselines

One agent x one benchmark, with the feedback loop:

```
python -m benchmarks.baselines.run_baseline \
    --agent claude-code --benchmark segmentation \
    --data-root data/NeurIPS22-CellSeg --limit 5 --iters 3 --sam-url http://gpu-node.example.net:8137

python -m benchmarks.baselines.run_baseline \
    --agent codex --benchmark cropper \
    --data-root data/NeurIPS22-CellSeg --limit 5 --iters 3
```

Both agents (`claude-code`, `codex`) across both benchmarks, staged sequentially (so they never
oversubscribe the box or the APIs) -- run in a durable shell, it can run for hours:

```
python -m benchmarks.baselines.run_after_shape \
    --data-root data/NeurIPS22-CellSeg --limit 1000 --iters 10 \
    --sam-url http://gpu-node.example.net:8137
```

`--only claude-code,codex` or `--only seg,cropper` restricts which stages run; `--wait-dir` makes a
stage wait for an in-flight SHAPE run to finish first (so the baselines are compared against the
exact same drawn image set).

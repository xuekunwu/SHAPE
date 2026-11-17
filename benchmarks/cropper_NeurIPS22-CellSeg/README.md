# cropper_NeurIPS22-CellSeg

Single-cell cropper benchmark on NeurIPS22-CellSeg. Given the same ground-truth instance mask and
targets, five arms (maskcrop, cellprofiler, gpt, react, shape) each frame every target cell in one
square crop; scored by neighbor contamination, target retention, runtime and cost.

## Dataset

Shares the same `NeurIPS22-CellSeg` data as `segmentation_NeurIPS22-CellSeg/` (images + their
ground-truth instance masks, used directly -- no segmenter in the loop). From the repository root:

```
hf download SHAPE-agent/benchmark-data --repo-type dataset --include "NeurIPS22-CellSeg/*" --local-dir data
```

```
data/NeurIPS22-CellSeg/images/<name>.{png,bmp,tif,tiff,jpg,jpeg}
data/NeurIPS22-CellSeg/labels/<name>_label.tiff
```

Pass `--data-dir data/NeurIPS22-CellSeg` to `prepare`.

Validate it with `benchmarks/segmentation_NeurIPS22-CellSeg/runners/prepare_dataset.py`.

## Running the arms

```
python benchmarks/cropper_NeurIPS22-CellSeg/run_benchmark.py prepare --run-dir R --data-dir DATA --n-images 30
python benchmarks/cropper_NeurIPS22-CellSeg/run_benchmark.py run --run-dir R --arm maskcrop|cellprofiler|gpt|react|shape
python benchmarks/cropper_NeurIPS22-CellSeg/run_benchmark.py compare --run-dir R
```

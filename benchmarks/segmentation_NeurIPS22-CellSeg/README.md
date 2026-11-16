# segmentation_NeurIPS22-CellSeg

Instance cell segmentation on the NeurIPS 2022 Cell Segmentation Challenge test set (1000
heterogeneous images: brightfield, phase contrast, DIC, fluorescence). Eight arms segment the same
images (cellpose_sam, cellpose3, microsam, stardist, gpt, react, shape, shape_cohort); each
delivers one integer instance mask per image, scored against the ground-truth mask.

## Dataset

From the repository root, download it from Hugging Face into `data/` (git-ignored):

```
hf download SHAPE-agent/benchmark-data --repo-type dataset --include "NeurIPS22-CellSeg/*" --local-dir data
```

```
data/NeurIPS22-CellSeg/images/<name>.{png,bmp,tif,tiff,jpg,jpeg}
data/NeurIPS22-CellSeg/labels/<name>_label.tiff
data/NeurIPS22-CellSeg/NeurIPS22-CellSeg-cell_image_metrics.csv   # difficulty-band reference
```

`--data-root` defaults to the repo's own `data/NeurIPS22-CellSeg`. Validate it and the
difficulty-band reference CSV with:

```
python benchmarks/segmentation_NeurIPS22-CellSeg/runners/prepare_dataset.py
```

## GPU services

`cellpose_sam`, `react`, `shape` and `shape_cohort` call the shared Cellpose-SAM service; `cellpose3`,
`stardist` and `microsam` need their own one-time envs.

```
bash benchmarks/gpu_services.sh start
bash benchmarks/gpu_services.sh urls                                               # SAM_SERVER_URL to export
bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/setup_seg_base_envs.sh build # once, on the GPU node
```

## Running the arms

```
export SHAPE_SAM_SERVER_URL=http://...
bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh lists
bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh cellpose_sam|cellpose3|microsam|stardist|gpt|react|shape|shape_cohort
bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh compare
# or everything in sequence:
bash benchmarks/segmentation_NeurIPS22-CellSeg/runners/run_all_arms.sh all
```

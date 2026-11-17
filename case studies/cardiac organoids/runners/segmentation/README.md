# cardiac organoids/runners/segmentation

Whole-organoid instance segmentation on one frozen image / instance-mask manifest. Seven arms segment the same
images and are scored against the instance ground truth, which no arm sees:

| Arm | Inference / agent |
|---|---|
| `cellpose_sam` | generic Cellpose-SAM through `Cell_Segmenter_Tool`, once at its defaults |
| `shape_seg_co` | the trained organoid checkpoint through `Organoid_Segmenter_Tool`, once at its defaults |
| `microsam` | microSAM automatic instance segmentation (`vit_b_lm`) |
| `stardist` | StarDist2D (`2D_versatile_he` / `2D_versatile_fluo`) |
| `llm` | `gpt-5.6-luna` returns organoid polygons directly, no segmentation tool |
| `react` | the shared ReAct runner with `Organoid_Segmenter_Tool` |
| `shape` | full `solve()` with `Organoid_Segmenter_Tool`, per-image verification and VRR |

## Dataset

The in-house organoid images and their instance masks are private. `prepare` pairs each image with the mask at
the same relative path (same stem or `<stem>_seg`; flat Cellpose `<stem>_seg.npy` dictionaries work too) and
freezes the manifest with every file's sha256; `--sample-size` and `--seed` draw a subset once.

```
python "case studies/cardiac organoids/runners/segmentation/run_organoid_segmentation.py" prepare \
    --manifest MANIFEST --image-root IMAGES --label-root MASKS
```

## Running the arms

```
python "case studies/cardiac organoids/runners/segmentation/run_organoid_segmentation.py" run \
    --manifest MANIFEST --run-dir RUN --arm cellpose_sam|shape_seg_co|microsam|stardist|llm|react|shape \
    --model-path CO_CKPT --require-gpu
python "case studies/cardiac organoids/runners/segmentation/run_organoid_segmentation.py" compare --manifest MANIFEST --run-dir RUN
```

`--model-path` is the organoid checkpoint file (`shape_seg_co`); `microsam` and `stardist` need micro-sam and
StarDist (with TensorFlow) installed. `--resume` continues an interrupted arm on the identical manifest and
settings. `run_organoid_cohort.py` runs the
SHAPE arm as a cohort (calibration sample, fast pass for confident groups, per-image VRR for the rest), and
`run_organoid_bounded.py` the same cohort in fixed 64-image shards with per-pass checkpoints and `--resume`:

```
python "case studies/cardiac organoids/runners/segmentation/run_organoid_cohort.py" \
    --manifest MANIFEST --model-path CO_CKPT --expected-commit FULL_SHA --run-dir RUN
python "case studies/cardiac organoids/runners/segmentation/run_organoid_bounded.py" \
    --manifest MANIFEST --model-path CO_CKPT --expected-commit FULL_SHA --run-dir RUN
```

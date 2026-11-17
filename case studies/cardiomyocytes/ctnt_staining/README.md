# cardiomyocytes/ctnt_staining

Per-cell cTnT positivity of fixed iPSC-derived cardiomyocytes, predicted from brightfield alone and scored
against labels derived from the cTnT fluorescence of the same cells. Three arms (`control`, `shape`, `codex`)
segment, crop and embed the brightfield images (self-supervised DINOv3 fine-tune) and fit one per-cell
classifier on the same prepared images and the same frozen 40/10/50 well split (seed 20261017). DAPI and cTnT
enter only the reference labels and the scorer, never an arm's decisions.

## Dataset

[SHAPE-agent/iPSC-CM-cTnT_staining](https://huggingface.co/datasets/SHAPE-agent/iPSC-CM-cTnT_staining):
2,294 images of 574 wells (brightfield, DAPI, cTnT) in one archive. `dataset.json` pins its revision,
size and metadata hash; the metadata commands check them without downloading the images:

```
python "case studies/cardiomyocytes/ctnt_staining/run.py" inspect
python "case studies/cardiomyocytes/ctnt_staining/run.py" fetch-metadata --out OUT
```

`prepare-full` prepares every well, `prepare-pilot` the fixed 20% pilot (same image, well and split IDs), from
the extracted `cTnT_composite/` folder (`RAW`) and the frozen well split (`SPLIT`); `freeze-references`
fixes the matched reference cells. Both prepare commands pin the metadata of the original local backup
(sha256 `c1b2a9ea...`), not the public archive's donor-free CSV (`9514170e...`).

```
python "case studies/cardiomyocytes/ctnt_staining/run.py" prepare-full --raw RAW --frozen-split SPLIT --out PREP
python "case studies/cardiomyocytes/ctnt_staining/run.py" freeze-references --help
```

## Running the arms

```
python "case studies/cardiomyocytes/ctnt_staining/run.py" pack --out SNAPSHOT.tar.gz   # --expected-source is the source_sha256 it prints, not archive_sha256
python "case studies/cardiomyocytes/ctnt_staining/run.py" run --prep PREP --references REFS \
    --assets-root ASSETS --expected-source SHA256 --arm control|shape|codex --out RUN --allow-exploratory-full
```

Use a fresh `--out` per arm; `--smoke` runs a small technical check first. `--allow-exploratory-full` is needed to
score the full data against the fixed pseudo-reference; on the pilot pass `--pilot-manifest PREP/PILOT.json`
instead.

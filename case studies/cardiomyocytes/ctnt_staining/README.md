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

From the repository root, download and extract the archive (`RAW` is the extracted `cTnT_composite/` folder):

```
hf download SHAPE-agent/iPSC-CM-cTnT_staining --repo-type dataset     --revision 16fb6c44c1f9f90da6ab712966856e5973581891 --local-dir data/iPSC-CM-cTnT_staining
tar -xzf data/iPSC-CM-cTnT_staining/iPSC-CM-cTnT_staining.tar.gz -C data/iPSC-CM-cTnT_staining
```

`prepare-full` prepares every well, `prepare-pilot` the fixed 20% pilot (456 images of 114 wells; the same image,
well and split IDs). Both check the metadata against `dataset.json` and derive the 40/10/50 well split from it
(`--frozen-split` optionally checks it against an earlier preparation's `private/splits.json`):

```
RUN_PY="case studies/cardiomyocytes/ctnt_staining/run.py"
python "$RUN_PY" prepare-pilot --raw data/iPSC-CM-cTnT_staining/cTnT_composite --out PREP   # or prepare-full
```

## Running the arms

The reference cells are fixed once, before the arms: `reference-segment` segments the brightfield images with the
frozen Cellpose-SAM defaults (the fixed algorithmic pseudo-reference) and `freeze-references` matches them to the
fluorescence labels. `--fraction` (default 0.05) and `--seed` select the images; `reference-segment` and `run` must
use the same ones, so that every image an arm runs on has a reference.

```
python "$RUN_PY" pack --out SNAPSHOT.tar.gz   # --expected-source is the source_sha256 it prints, not archive_sha256
python "$RUN_PY" reference-segment --prep PREP --out REFSEG --assets-root ASSETS --expected-source SHA256
python "$RUN_PY" freeze-references --prep PREP --masks REFSEG --out REFS     --kind fixed_algorithmic_pseudo_reference --allow-exploratory
python "$RUN_PY" run --prep PREP --references REFS --assets-root ASSETS --expected-source SHA256     --arm control|shape|codex --out RUN --smoke
```

`--masks` is the `reference-segment` output folder. Use a fresh `--out` per arm; `--smoke` limits each arm's search
(two rounds) on the selected images. To score the whole prepared set, pass `--fraction 1` to `reference-segment`
and `run`, and to `run` either `--allow-exploratory-full` (prepare-full) or `--pilot-manifest PREP/PILOT.json`
(prepare-pilot).

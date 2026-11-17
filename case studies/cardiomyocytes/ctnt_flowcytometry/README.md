# cardiomyocytes/ctnt_flowcytometry

The cTnT-positive percentage of each culture well, predicted from that well's live-cell brightfield views and
scored against flow cytometry (one measurement per well). Each arm (`control`, `shape`, or an agent arm:
`codex`, `claude`, `biomni`) segments, crops and embeds the images (self-supervised DINOv3 fine-tune); a
well-level ridge head is fitted on the training wells, selected on the validation wells and scored once on the
held-out wells (MAE in percentage points). The split is fixed: 600 images, 87 wells, 35 / 9 / 43
train / validation / held-out wells, seed 20261015.

## Dataset

[SHAPE-agent/iPSC-CM-cTnT_flowcytometry](https://huggingface.co/datasets/SHAPE-agent/iPSC-CM-cTnT_flowcytometry):
an image archive and the standalone metadata CSV that `harness/dataset.json` pins. From the repository root:

```
hf download SHAPE-agent/iPSC-CM-cTnT_flowcytometry --repo-type dataset \
    --revision b4717abe28ecf9e2b82b02028bd9f2ed369b3243 --local-dir data/iPSC-CM-cTnT_flowcytometry
python "case studies/cardiomyocytes/ctnt_flowcytometry/runners/run.py" materialize \
    data/iPSC-CM-cTnT_flowcytometry data/ctnt_flowcytometry_raw          # the 600 pinned TIFFs, size and hash checked
python "case studies/cardiomyocytes/ctnt_flowcytometry/runners/run.py" prepare \
    --data-root data/ctnt_flowcytometry_raw --out PREP                    # pixel check and the fixed well split
```

`run.py inspect`, `fetch-metadata` and `audit-metadata` check the pinned contract without the images.

## Running the arms

```
python "case studies/cardiomyocytes/ctnt_flowcytometry/runners/run.py" pack --out SNAPSHOT.tar.gz   # --expected-source is the source_sha256 it prints, not archive_sha256
python "case studies/cardiomyocytes/ctnt_flowcytometry/runners/run.py" run --prep PREP \
    --arm control|shape|codex|claude|biomni --assets-root ASSETS --expected-source SHA256 --out RUN
```

`preflight` checks the model route and the runtime without running a stage; `--smoke` runs about 37 images. Two variants
share the dispatcher: `hybrid` (eligible-plate SSL with separate classifier experts and validation-only routing)
and `crop-from-masks` (re-crops a fixed mask set with the rule-first crop policy); see their `--help`.

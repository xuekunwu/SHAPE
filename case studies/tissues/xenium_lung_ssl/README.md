# tissues/xenium_lung_ssl

Self-supervised DINOv3 embeddings of four-channel single-cell crops of a Xenium Prime human lung cancer section, from
three arms (`control`, `shape`, `codex`), and the cell-type probe that scores them on held-out spatial tiles. The
inputs are crops, so the only tool stage is embed (the representation stage of the shared harness,
`benchmarks/full_CPJUMP1/harness`).

Protocol `xenium-lung-ssl-sample10-tile500-well250-415-v2` (`dataset.json`): a seeded 10% sample of the section's
278,328 cells (seed 20261025), split by 500 um spatial tiles into SSL training, SSL validation and held-out cells
(4 : 1 : 5); a well is a 250 um neighbourhood nested in one split tile, each crop one site. Held-out crops never enter
an SSL call; every sampled crop is embedded after the fit.

## Dataset

The 10x Genomics [Xenium Prime FFPE human lung cancer](https://www.10xgenomics.com/datasets) output bundle; of it,
`cells.parquet`, `cells.zarr.zip` and `morphology_focus/` are used. From the repository root:

```
mkdir -p data/Xenium && cd data/Xenium
curl -L -O https://cf.10xgenomics.com/samples/xenium/3.0.0/Xenium_Prime_Human_Lung_Cancer_FFPE/Xenium_Prime_Human_Lung_Cancer_FFPE_outs.zip
unzip Xenium_Prime_Human_Lung_Cancer_FFPE_outs.zip cells.parquet cells.zarr.zip 'morphology_focus/*' \
    -d Xenium_Prime_Human_Lung_Cancer_FFPE_outs && cd ../..
```

`make_crops.py` cuts one four-channel uint16 TIFF per cell (DAPI, ATP1A1/CD45/E-Cadherin, 18S, alphaSMA/Vimentin):
the square around the cell's pixels in the Xenium cell label mask, zero outside the cell. `prepare` samples, splits
and copies the crops:

```
OUTS=data/Xenium/Xenium_Prime_Human_Lung_Cancer_FFPE_outs
python "case studies/tissues/xenium_lung_ssl/make_crops.py" $OUTS data/Xenium/crops_lung --workers 32
python -m "case studies.tissues.xenium_lung_ssl.e2e" prepare --crops data/Xenium/crops_lung --cells $OUTS/cells.parquet --out PREP
```

The crops and the preparation reproduce the published ones: `PREP/PREPARE.json` records `images_sha256`
`fed0eba3...` and `splits_sha256` `408ae0d3...` (27,833 crops).

## Running the arms

`ASSETS` is the pinned asset root of `benchmarks/full_CPJUMP1` (`cli assets`); the `shape` and `codex` arms need
`LLM_API_KEY` and an https `LLM_BASE_URL`, the `codex` arm also the Codex CLI.

```
E2E="case studies.tissues.xenium_lung_ssl.e2e"
python -m "$E2E" pack --out SNAPSHOT.tar.gz          # --expected-source is the source_sha256 it prints, not archive_sha256
python -m "$E2E" run --prep PREP --out RUNS/control --arm control --assets-root ASSETS --expected-source SHA256 --cores 8
# the same for --arm shape and --arm codex; --smoke runs about 600 crops (whole wells per split) first
```

## Probe

The cell-type labels are CellTypist's, exported as `cell_id,celltype_final` to `celltype_final.csv.gz`.

```
python "case studies/tissues/xenium_lung_ssl/linear_probe.py" PREP celltype_final.csv.gz OUT \
    control=RUNS/control shape=RUNS/shape codex=RUNS/codex
cd OUT && python "$REPO/case studies/tissues/xenium_lung_ssl/per_class_predictive_values.py"
```

`per_class_predictive_values.py` reads the probe's `metrics.json` and `predictions.csv` from the current directory
(`REPO` is the repository root). The probe: L2-normalised embeddings, StandardScaler and multinomial logistic
regression fitted on the training cells, C chosen on the validation cells, scored once on the held-out tiles with
1,000 tile bootstrap draws (seed 20261026) shared by every arm, plus a chance group; `per_class_predictive_values.py`
adds PPV, NPV, sensitivity and specificity per class.

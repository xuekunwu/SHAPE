# cardiac organoids/timecourse

One 96-well plate of cardiac organoids, 12 Control and 11 FC wells imaged in phase contrast and GFP at eight
time points (baseline and days 2 to 16). From the phase-contrast image alone, each arm segments the organoid,
crops it, learns a representation and reads out the culture day and the condition on wells it never saw:

| Arm | Who sets the tool parameters |
|---|---|
| `control` | nobody: every tool at its CARD defaults, the frozen pretrained backbone (zero-shot) |
| `shape` | SHAPE's verify-reason-refine loops (organoid segmentation, crop, representation), `gpt-5.6-luna` at high effort |
| `codex` | the Codex CLI (`gpt-5.6-luna`, high) through the agent-arm stage protocol of `benchmarks/full_CPJUMP1` |

All arms get the same prepared inputs, the same well split (seed 20261024: 11 train, 4 validation, 8 held-out
wells, every time point of a well in one split) and the same tool-round and SSL-epoch budgets.

## Dataset

[SHAPE-agent/CO_timecourse](https://huggingface.co/datasets/SHAPE-agent/CO_timecourse): per time point a folder
(`bs`, `d2` ... `d16`) with `<well>_<seq>_1_1_Phase Contrast_001.tif` and its GFP image. `prepare` copies the
phase-contrast channel, keeps GFP, conditions and days private, pseudonymises the identifiers and freezes the
split (184 images; well F6, imaged on day 4 only, is excluded):

```
hf download SHAPE-agent/CO_timecourse --repo-type dataset --local-dir data/CO_timecourse
python -m "case studies.cardiac organoids.timecourse.e2e" prepare --source data/CO_timecourse --out PREP
```

## Running the arms

```
E2E="case studies.cardiac organoids.timecourse.e2e"
python -m "$E2E" pack --out SNAPSHOT.tar.gz            # --expected-source is the source_sha256 it prints, not archive_sha256
python -m "$E2E" run --prep PREP --out RUNS/control --arm control --assets-root ASSETS \
    --organoid-model CO_CKPT --expected-source SHA256 --cores 8
# the same for --arm shape and --arm codex; --smoke runs a 6-image technical check first
python -m "case studies.cardiac organoids.timecourse.compare" --control RUNS/control \
    --shape RUNS/shape --codex RUNS/codex --out RUNS/comparison
```

`--organoid-model` is the trained organoid segmenter checkpoint (`shape_seg_co`).
`e2e prepare --protocol co-fc-timecourse-ssl19-cvcol4-v1` selects the cross-validation protocol instead of the
held-out split.

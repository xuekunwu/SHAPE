# batch_correction_JUMP

Batch correction of JUMP Cell Painting well profiles on the scenarios of Arevalo et al. (2024,
*Nature Communications* 15:6516). Six arms -- the uncorrected `raw` reference plus `harmony`,
`scanorama`, `gpt_luna`, `react_luna_harmony` and `shape_luna_harmony` -- are scored on held-out
compounds with the authors' own metrics.

## Dataset

JUMP Cell Painting well profiles in the authors' `inputs/` layout (sources 2, 3, 6, 8 and 10, which
cover scenarios 1-5, plus the compound, plate and well metadata) and the pinned microscope table.
From the repository root:

```
hf download SHAPE-agent/benchmark-data --repo-type dataset --include "JUMP/*" --local-dir data
```

`prepare` builds a scenario's prep from it:

```
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py prepare \
    --base data/JUMP --output PREP --scenario scenario_N \
    --microscope-config data/JUMP/inputs/microscope_config.csv --arevalo-cache DIR
```

## Running the arms

```
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py control --prep PREP
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py suite --prep PREP --output RUN
python benchmarks/batch_correction_JUMP/run_batch_benchmark.py report --prep PREP --output RUN
```

`shape-only`, `react` and `effort-sweep` run one arm (or a reasoning-effort sweep of the SHAPE arm)
in isolation -- see the module docstring of `run_batch_benchmark.py` for their flags.

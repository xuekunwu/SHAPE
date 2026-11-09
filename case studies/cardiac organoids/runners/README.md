# Cardiac organoid runners

Study records on this branch:

| Study | Content |
|---|---|
| [define_candidates_96well](define_candidates_96well/README.md) | Run scripts for the 1141-image 96-well CO DEFINE candidates (segment, crop, two-channel SSL embed; train/validation 8:2 by plate-well). |

Use `run.py` as the public entrypoint, following the cardiomyocyte studies' separation of executable code in `runners/`.

| File | Responsibility |
| --- | --- |
| `run.py` | Compare, verify, or launch pinned native inference |
| `evaluation.py` | Authenticate/download the fixed HF export, validate inputs, match cohorts, aggregate and export |
| `native.py` | Require a clean checkout at the recorded inference commit and delegate to its original entrypoint |
| `stardist_isolation.py` | Portable spawn-worker isolation around that checkout's unchanged StarDist adapter; retain native failures and continue |
| `test/` | Denominator, pairing, input identity, native checkout and real spawned-worker regression checks |

## Reproduce the published comparison

Python 3.10+; offline replay and tests use only the standard library. The source dataset is private: use an account with access and the official HF login. Authentication never belongs in a command argument or Git.

```bash
python -m pip install -r "case studies/cardiac organoids/runners/requirements.txt"
hf auth login
python "case studies/cardiac organoids/runners/run.py" compare --download --output-dir /path/outside/repo/cardiac-comparison
python "case studies/cardiac organoids/runners/run.py" verify --download
```

Download is pinned to HF commit `cd0ab63f74fc304d481dad55912cc6afca00e780`, folder `cardiac_organoids_segmentation`. The runner checks the pinned manifest SHA256, all eight CSV byte hashes/sizes, schema, unique image IDs, source identities, status counts, input/GT hashes and dimensions, and shared CO checkpoint. It never silently averages different image cohorts or substitutes zero for unrecorded predictions. Failed recorded predictions retain the original scores.

For existing local exports, replace `--download` with `--input-dir /path/to/eight-csvs-and-manifest`. `verify` recomputes and compares all eight report files byte for byte, including the README and export manifest.

```bash
python -m unittest discover -s "case studies/cardiac organoids/runners/test" -v
CARDIAC_RESULTS_INPUT=/path/to/eight-csvs-and-manifest python -m unittest discover -s "case studies/cardiac organoids/runners/test" -v
```

On PowerShell, set `$env:CARDIAC_RESULTS_INPUT='C:\path\to\data'` before the second command. Without this variable, the three real-export tests skip; synthetic denominator/pairing tests and actual spawn-worker crash/continuation tests still run. These tests make no model or paid LLM calls. They do not establish biological accuracy or complete an unfinished SHAPE/LLM/ReAct evaluation.

## Native inference and checkpoint resume

Native inference code already exists in this repository at the exact recorded commits. It depends on changes throughout shared SHAPE, so use the **complete historical checkout**, rather than mixing its study runner with current `cs-cm` algorithms. This publication provides the launcher and portable isolation code here without duplicating shared segmentation implementations.

| Arms | Recorded native source |
| --- | --- |
| Cellpose-SAM, direct CO, microSAM, StarDist, OrganoID | `00fad58934b50008bfe4186a2e5e4cfd616662cd` |
| SHAPE, direct LLM, ReAct | `8f59e6b53b84229236bd091b9bb252ba83b266d9` |

Create separate clean worktrees from the repository root. Paths below are examples; keep them outside the current checkout and outside protected input directories.

```bash
git fetch origin 00fad58934b50008bfe4186a2e5e4cfd616662cd 8f59e6b53b84229236bd091b9bb252ba83b266d9
git worktree add --detach /path/native-baselines 00fad58934b50008bfe4186a2e5e4cfd616662cd
git worktree add --detach /path/native-agents 8f59e6b53b84229236bd091b9bb252ba83b266d9
python "case studies/cardiac organoids/runners/run.py" native --source-root /path/native-baselines --source-commit 00fad58934b50008bfe4186a2e5e4cfd616662cd -- --help
```

The native CLI supports `prepare`, `run`, and `compare`; the baseline README of that recorded checkout documents pairing, per-image audits, failure scoring and strict resume identity. Read its subcommand help for actual options. Original `compare` requires completed arms with matching source/settings; the new CSV replay explicitly labels its cross-commit partial comparisons and is not a replacement for that gate.

Use the same frozen manifest for every arm in a new comparison. This example assumes the protected input files, frozen manifest and backend environment are already available:

```bash
python "case studies/cardiac organoids/runners/run.py" native --source-root /path/native-baselines --source-commit 00fad58934b50008bfe4186a2e5e4cfd616662cd --python /path/backend/bin/python -- run --manifest /path/results/pairs.json --run-dir /path/results --arm shape_seg_co --model-path /path/exact/shape_seg_co --require-gpu
python "case studies/cardiac organoids/runners/run.py" native --source-root /path/native-baselines --source-commit 00fad58934b50008bfe4186a2e5e4cfd616662cd --python /path/stardist/bin/python --isolate-stardist -- run --manifest /path/results/pairs.json --run-dir /path/results --arm stardist --resume
```

Omit `--resume` for a fresh arm. Resume verifies the original manifest, source hashes, interpreter, package versions, settings and delivered mask hashes, then skips already recorded images **including failures**. It never retries a recorded failure; use a fresh run directory for retries. Do not regenerate a manifest to bypass an identity mismatch. The new isolation launcher validates the historical source checkout first and leaves its files unchanged. Isolation preserves the original model, normalization, tiling and scorer. A native abort records a failure and starts a new worker for the next image; Python errors also stay visible. Session provenance and worker logs go under the external run directory. Exit code 2 denotes completed predictions with failures.

This portable isolation adapter is newly published and tested with synthetic subprocess failures; it is not the exact external Slurm wrapper of the original run. The original recovery's input hash/mask validation and supplemental attestation remain separate historical evidence. Do not claim identical recovery bytes or full inference validation from the synthetic tests.

Native environments need the relevant packages and model assets: Cellpose 4 for CO/CPSAM; micro-sam and its predictor/segmenter dependencies for microSAM; StarDist, csbdeep and TensorFlow for StarDist. OrganoID needs the [upstream code](https://github.com/jono-m/OrganoID/tree/deb7aeaa) at the recorded revision, `ORGANOID_ID_REPO`, and the pinned OptimizedModel checked by the original adapter. Partial observed package versions are in the published comparison JSON; they are not complete environment locks. The private CO checkpoint and original images/GT are required and are not downloaded by this replay tool. Match the checkpoint hash recorded in the report.

For agent arms, use the agent checkout, `LLM_API_KEY` and `LLM_BASE_URL=https://llm-gateway.example.edu/v1` via the existing secret configuration. The source pins gpt-5.6-luna and defaults to high reasoning; no provider/model substitution or invented usage is permitted. The complete historical run settings are not recoverable from the CSVs alone. Run GPU inference only in an allocated compute job on the cluster; the launcher does not submit jobs, obtain resources or set hardware policy.

## Optimized CO cohort

[The managed cohort runner](segmentation/README.md#co-cohort-evaluation-and-gpu-smoke) provides calibration, selective per-image VRR and hash-verified native trial reuse. The GPU evaluation entrypoint is run_organoid_cohort.py under runners/segmentation.

### Export a completed cohort run

`export_cohort.py` uses only the Python standard library. It exports `shape_cohort.csv` and an attestation manifest from a completed `run_organoid_cohort.py` directory. It checks source and manifest identities, complete ordered image coverage, recorded failures, CUDA/checkpoint consistency, model configuration, audited call counts, reported usage, cache hits and aggregate scores. Unfinished or missing predictions are rejected. Failed recorded predictions stay in the denominator.

```bash
python "case studies/cardiac organoids/runners/export_cohort.py" --run-dir /path/to/completed/full150 --expected-commit a879dca9619645fdd670eb7818c4efc3bbf19203 --output-dir /path/outside/repo/cohort-csv --verify-mask-files
```

Run large mask rehashes inside Slurm on the cluster. Without `--verify-mask-files`, the manifest explicitly records that mask files were not rehashed; their original recorded hashes remain in the CSV. The exporter does not rescore images, perform inference or call an LLM.

`managed_pass_seconds` sums the managed passes for one image. Cohort and preflight LLM calls are shared, so LLM usage is retained globally in the export manifest rather than assigned to individual images. Concurrent image seconds cannot be summed into wall time. The runner's `stage_wall_seconds` includes model setup, gateway preflight, managed cohort/VRR and scoring, and excludes outer solve planning and scheduler waiting. It is not a matched runtime comparison with the historical full SHAPE pipeline. Errored calls without reported usage are counted explicitly as unmeasured.

Keep per-image exports in the private results dataset. Export regression checks are included in `test/test_cohort_export.py`.

### Compare the fixed 150-image cohort

`compare_cohort.py` compares the optimized cohort export with the first 150 **observed** records in the frozen historical SHAPE CSV, including its recorded failure. It verifies original historical file hashes, new CSV hashes, ordered cohort membership, image/GT identity, geometry, CO checkpoint and inference-source attestations. Per-image failures remain in the denominator. The report includes the five complete non-LLM baselines and historical SHAPE; the disjoint LLM/ReAct canaries are excluded.

Download the four new files from [the immutable private cohort export](https://huggingface.co/datasets/SHAPE-Morph/result/tree/55745dc96bc3b3b77a2524018813d4e6a111844f/cardiac_organoids_segmentation/cohort_20261020) into `cohort-exports/smoke5/` and `cohort-exports/full150/`. Keep the historical eight CSVs plus their manifest together in `historical-exports/` at revision `cd0ab63f74fc304d481dad55912cc6afca00e780`. Obtain these through your existing authorized HF client login.

```bash
python "case studies/cardiac organoids/runners/compare_cohort.py" --historical-dir historical-exports --cohort-dir cohort-exports --hf-revision 55745dc96bc3b3b77a2524018813d4e6a111844f --output-dir comparison-output
```

Add `--verify` and point `--output-dir` at an earlier comparison output to verify byte-identical replay. The stdlib comparison tests are `test/test_cohort_comparison.py`.

### Bounded VRR safety protocol

For recoverable larger runs, use [the schema-2 bounded cohort runner](segmentation/README.md#bounded-organoid-cohort-and-resume-schema-2), or add `--bounded` to the original cohort entrypoint with `--image-workers 1`. This adds independent raw-image review of empty predictions, cumulative managed VRR limits, a rescaled-pixel guard, one isolated GPU model worker, fixed shards and attested resume. Grouping preflights may overlap on up to four allocated CPU workers. Delivery and quality/review states are reported separately. A run under the original pinned protocol does not validate the new safety protocol. Its real GPU challenge/pilot gate remains pending.

## Bounded schema-2 validation

compare_bounded_validation.py compares Challenge24/Pilot64 against pinned eight-arm exports, checks input/GT identity and CO weights, retains failures, and preserves missing values. segmentation/prepare_bounded_review.py creates the post-inference review pack inside Slurm.

segmentation/validate_bounded_stage.py runs one clean pinned GPU cohort and its completed replay, requiring complete delivery, measured audits and byte-identical CSV. Optional --min-map is an explicit development regression floor. It never sets the biological quality gate or launches the full dataset.

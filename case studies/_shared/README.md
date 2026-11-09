# Shared Data Admission

`metadata.py` supplies authenticated Hugging Face inventory and metadata-only
fetch/audit operations for the two cardiomyocyte studies. Study-specific pinned
contracts live in each `harness/dataset.json`. No GPU, SHAPE model or LLM is loaded.

Authentication uses the normal HF token cache or `HF_TOKEN`; no credentials are
stored in the contracts. The staining CSV is read from at most 2 MiB of the
compressed archive prefix, with at most 16 MiB decompressed. Archive members are
read in memory, never extracted to paths. If metadata moves beyond this bound,
the operation fails and the data contract must be reviewed.

`fetch-metadata` saves private metadata outside the code checkout in a fresh
directory. `inspect` and `audit-metadata` can save aggregate reports without
donor identifiers, raw rows or labels. An audit checks bytes, CSV identities,
safe relative paths, well/assay consistency and label range, but does not verify
image existence, pixel layout, segmentation, annotation quality or model performance.
`metadata_verified` is not an end-to-end completion marker.

From the repository root:

```bash
python -m pytest "case studies/_shared/test" -q
```

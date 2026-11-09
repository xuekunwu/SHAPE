# Staining Study Harness

`dataset.json` pins the dataset revision and embedded metadata hash. `cli.py`
delegates inventory, bounded metadata fetch and audit to the shared admission code.
The CSV at `cTnT_composite/cTnT_composite_metadata.csv` lists 2294 image records
and 574 plate-qualified wells. It contains image/experimental metadata, not
per-cell cTnT intensity or binary labels.

All metadata status categories are retained in the audit. `layout_excluded`,
`ambiguous_source_metadata` and incomplete covariates require explicit cohort and
split decisions, not silent row deletion. Image channels/axes have not yet been
verified from pixels, and archive-prefix inspection is not a full image inventory.

Future modules here will prepare brightfield inputs and separate fluorescence
references, derive labels under a frozen protocol, and implement stage adapters,
the classifier and scorer. Follow the [study protocol](../README.md).

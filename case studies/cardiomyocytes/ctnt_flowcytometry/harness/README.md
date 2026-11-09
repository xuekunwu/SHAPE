# Flow Study Harness

`dataset.json` pins the dataset revision and **standalone** CSV hash. `cli.py`
delegates inventory, bounded metadata fetch and audit to the shared admission code.
The image manifest has 600 views, grouped by `(plate_id, well_id)` into 87 wells.
`flow_sample_id` is the assay identity; `ctnt_positive_percent` is in percent.

The older embedded CSV has 620 rows / 92 wells and 160 missing label entries.
It must not silently replace the pinned standalone table. Archive images not in
the pinned manifest are outside this study selection until explicitly reviewed.

Future modules here will implement preparation, split manifests, SHAPE stage
adapters, the well-level prediction head and scoring. They must follow the
[study protocol](../README.md); the current CLI only verifies metadata.

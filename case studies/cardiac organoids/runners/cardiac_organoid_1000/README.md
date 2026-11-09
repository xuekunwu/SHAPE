# Cardiac organoid 1000-image study

The worker and scorers of the SHAPE 1000-image comparison. Each run records their sha256 values in its own
`provenance.json`.

| File | Responsibility |
|---|---|
| `run_frozen_organoid_shard.py` | Run one fixed shard of 64 images for one arm with a clean checkout at the recorded commit; resumes with `--resume`; applies the cost guard (no unknown usage, no incomplete audit or tool records). |
| `full_thresholds/instance_ap.py` | AP(t) scorer: verbatim port of the harness matching (pairwise IoU, maximum-cardinality then maximum-IoU one-to-one matching, AP = TP / (TP + FP + FN)); the threshold grid is `AP_STEP` (0.05 default, 0.01 used) |
| `full_thresholds/ap_full_thresholds.py`, `patch_react_reused.py` | Recompute AP at 0.01 ... 1.00 for every arm from the delivered masks, verifying mask sha256 and the exported AP at 0.50 ... 0.95 per image; the patch script scores the reused ReAct rows located by mask sha256 |
| `cellpose3/run_cellpose3_arm.py`, `instance_ap.py` | The Cellpose3 (cyto3) baseline arm on the 1000 images (cellpose 3.1.1.3 env, size-model diameter, grayscale, tifffile with PIL fallback, `--resume`), scored with the same scorer at 0.01 ... 1.00 |

Authentication comes from the environment (`LLM_API_KEY`, `LLM_BASE_URL`, `LLM_MODEL`); the runner pins
`gpt-5.6-luna` at high reasoning effort regardless of `LLM_MODEL`. The scripts do not submit jobs themselves.

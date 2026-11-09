# Scientific workflows

All ordinary, HYBRID, fixed-mask crop, development QA, recovery and smoke runs live here. Workflows compose existing dataset/prediction/evaluation helpers and native tools through the shared runtime launcher. `stage_sequence.py` owns scientific order/environment/deadline; `runtime/stage_runner.py` only invokes one worker. Public commands remain in `run.py`.

`cohort_cropping.py` adapts the core rule-first crop policy to scientific raw
crops; it does not implement VRR. `crop-from-masks` uses the same policy by
default (`--crop-policy rule-first-balanced`); `--crop-policy legacy` retains
the previous search objective. Both verify the fixed input bundle. Each mask's
original `min_area` is preserved. `CROP_VRR_PROGRESS.json` reports round-level
problem, improvement, passing and rollback counts.

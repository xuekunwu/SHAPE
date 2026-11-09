# Cardiac Organoids

The segmentation evaluation compares whole-organoid instance masks on the Inhouse_Organoid_Seg images and paired instance annotations, over non-LLM baselines, SHAPE, and direct LLM/ReAct arms on matched images.

- [Runners](runners/): executable comparison replay, verification, tests and pinned native inference/resume launchers.
- [Harness](harness/): reserved for future study-specific data and biological endpoint adapters.
- [FC time course](fc_timecourse/README.md): end-to-end CONTROL / SHAPE / CODEX comparison on one Control-vs-FC 96-well plate imaged at eight time points (184 phase-contrast images, 23 wells); primary read-out is the culture day.

The direct CO, SHAPE and ReAct arms use the trained shape_seg_co checkpoint. Training/evaluation non-overlap has not been verified. Biological prediction, reference assays and the remaining end-to-end protocol are pending; follow the [shared case-study design](../README.md) when extending this application.

The current [CO cohort runner](runners/segmentation/README.md#co-cohort-evaluation-and-gpu-smoke) adds verified native-trial reuse and selective VRR fallback.

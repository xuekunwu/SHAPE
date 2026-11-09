---
tool: Analysis_Visualizer_Tool
category: bioimage_core
stage: none
depends_on: []
tunable: false
---
# Tool card: Analysis_Visualizer_Tool

**When to use:** to VISUALIZE already-computed results -- statistical bar/pie charts with
automatic significance testing (t-test for 2 groups, ANOVA for 3+), multi-group comparisons,
and summaries of segmentation or Cell_State_Analyzer outputs. **When NOT:** it does NOT
compute features or analysis -- run a Cell_State_Analyzer / segmenter first; this only plots
their outputs. Input: precomputed analysis results / per-group values. Output: figures.

Not tunable: the VLRR loop does not retune this tool; it has no correction parameters.

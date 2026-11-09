---
tool: Cell_Cluster_Functional_Hypothesis_Tool
category: bioimage_core
stage: none
depends_on: [Cell_State_Analyzer_Tool]
tunable: false
---
# Tool card: Cell_Cluster_Functional_Hypothesis_Tool

**When to use:** the DOWNSTREAM analysis step that takes morphological CLUSTERS (from a
Cell_State_Analyzer) and proposes functional hypotheses for them, grounded in literature via
PubMed. **When NOT:** it needs clusters first -- run a Cell_State_Analyzer before it; it does
not segment or cluster. Input: cluster descriptors / marker features. Output: per-cluster
functional hypotheses with literature evidence.

Depends on `Pubmed_Searcher_Tool` (knowledge) for literature grounding. Not tunable.

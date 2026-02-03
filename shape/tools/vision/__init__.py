"""
Vision tools for morphological analysis.

Tools in this module handle:
- Image preprocessing and enhancement
- Cell/nuclei/organoid segmentation
- Single-cell cropping
- Feature extraction and embedding
- Clustering and state analysis
- Visualization
"""

# Tool organization:
# Vision tools are currently implemented in octotools/tools/
# They will be migrated to shape/tools/vision/ in future refactoring
#
# Current vision tools:
# - image_preprocessor: Image preprocessing and enhancement
# - cell_segmenter: Cell segmentation (phase contrast)
# - nuclei_segmenter: Nuclei segmentation
# - organoid_segmenter: Organoid segmentation
# - single_cell_cropper: Extract individual cell crops
# - cell_state_analyzer_single: Self-supervised learning (single-channel)
# - cell_state_analyzer_multi: Self-supervised learning (multi-channel)
# - analysis_visualizer: Result visualization
# - image_captioner: Image description generation

__all__ = []


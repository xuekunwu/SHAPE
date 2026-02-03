"""
Schemas for morphological observations.

Morphological observations are first-class objects in SHAPE's reasoning process.
These schemas define the structure of observations produced by tools.
"""

from typing import List, Dict, Any, Optional
from pydantic import BaseModel


class SegmentationObservation(BaseModel):
    """Observation from a segmentation tool."""
    mask_path: str
    cell_count: int
    image_path: str
    metadata: Dict[str, Any] = {}


class CellCropObservation(BaseModel):
    """Observation from single-cell cropping."""
    crop_paths: List[str]
    cell_metadata: List[Dict[str, Any]]
    source_image: str
    total_cells: int


class EmbeddingObservation(BaseModel):
    """Observation from embedding extraction."""
    embedding_path: str
    cell_ids: List[str]
    dimensionality: int
    method: str


class ClusterObservation(BaseModel):
    """Observation from clustering analysis."""
    cluster_labels: List[int]
    cluster_key: str
    resolution: float
    n_clusters: int
    adata_path: Optional[str] = None


class SpatialObservation(BaseModel):
    """Observation from spatial analysis."""
    graph_path: Optional[str] = None
    neighbor_matrix_path: Optional[str] = None
    niche_labels: Optional[List[str]] = None
    metadata: Dict[str, Any] = {}


class MorphologicalObservation(BaseModel):
    """Composite morphological observation combining multiple analysis results."""
    segmentation: Optional[SegmentationObservation] = None
    crops: Optional[CellCropObservation] = None
    embeddings: Optional[EmbeddingObservation] = None
    clusters: Optional[ClusterObservation] = None
    spatial: Optional[SpatialObservation] = None
    metadata: Dict[str, Any] = {}


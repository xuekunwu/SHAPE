from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Literal, Any, Union
import uuid


class TaskType(str, Enum):
    CODING = "coding"
    ANALYSIS = "analysis"
    PIPELINE = "pipeline"
    DEBUG = "debug"


class ToolStage(str, Enum):
    """
    Tool stage classification for two-stage granularity separation.
    
    STAGE_1_IMAGE_LEVEL: Tools that operate on images as the unit.
    Includes preprocessing, segmentation, detection - no single-cell reasoning.
    
    STAGE_2_CELL_LEVEL: Tools that operate on cells as the unit.
    Includes single-cell cropping, feature extraction, classification, comparison.
    Must only operate after segmentation is complete.
    """
    STAGE_1_IMAGE_LEVEL = "stage_1_image_level"
    STAGE_2_CELL_LEVEL = "stage_2_cell_level"


@dataclass
class PlanStep:
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


@dataclass
class ActiveTask:
    task_id: str
    goal: str
    task_type: TaskType
    plan_steps: List[PlanStep] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, goal: str, task_type: TaskType = TaskType.ANALYSIS) -> "ActiveTask":
        return cls(task_id=str(uuid.uuid4()), goal=goal, task_type=task_type)


@dataclass
class PlanDelta:
    intent: Literal["NEW_TASK", "CONTINUE_TASK", "MODIFY_TASK"]
    added_steps: List[PlanStep] = field(default_factory=list)
    completed_step_ids: List[str] = field(default_factory=list)
    updated_goal: Optional[str] = None
    updated_task_type: Optional[TaskType] = None


@dataclass
class AnalysisInput:
    name: str
    path: str
    input_type: str = "image"  # image, folder, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisSession:
    inputs: Dict[str, AnalysisInput] = field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_input: Optional[str] = None
    compare_requested: bool = False


@dataclass
class InputDelta:
    new_inputs: Dict[str, AnalysisInput] = field(default_factory=dict)
    set_active: Optional[str] = None
    compare_requested: bool = False


@dataclass
class ConversationState:
    conversation: List[Any] = field(default_factory=list)
    active_task: Optional[ActiveTask] = None
    analysis_session: Optional[AnalysisSession] = None


class CellCollection:
    """
    Manager for collections of cells across multiple images and groups.
    Enforces that cells are the atomic unit for Stage 2 analysis.
    """
    
    def __init__(self):
        self.cells: List[CellCrop] = []
        self._index_by_image_id: Dict[str, List[CellCrop]] = {}
        self._index_by_group: Dict[str, List[CellCrop]] = {}
        self._index_by_cell_id: Dict[str, CellCrop] = {}
    
    def add_cell(self, cell: CellCrop) -> None:
        """Add a cell to the collection and update indices."""
        if cell.cell_id in self._index_by_cell_id:
            # Update existing cell
            old_cell = self._index_by_cell_id[cell.cell_id]
            self.cells.remove(old_cell)
            # Remove from old indices
            if old_cell.source_image_id in self._index_by_image_id:
                self._index_by_image_id[old_cell.source_image_id].remove(old_cell)
            if old_cell.group in self._index_by_group:
                self._index_by_group[old_cell.group].remove(old_cell)
        
        self.cells.append(cell)
        self._index_by_cell_id[cell.cell_id] = cell
        
        # Update image index
        if cell.source_image_id not in self._index_by_image_id:
            self._index_by_image_id[cell.source_image_id] = []
        self._index_by_image_id[cell.source_image_id].append(cell)
        
        # Update group index
        if cell.group not in self._index_by_group:
            self._index_by_group[cell.group] = []
        self._index_by_group[cell.group].append(cell)
    
    def add_cells(self, cells: List[CellCrop]) -> None:
        """Add multiple cells to the collection."""
        for cell in cells:
            self.add_cell(cell)
    
    def get_cells_by_image_id(self, image_id: str) -> List[CellCrop]:
        """Get all cells from a specific source image."""
        return self._index_by_image_id.get(image_id, []).copy()
    
    def get_cells_by_group(self, group: str) -> List[CellCrop]:
        """Get all cells from a specific group."""
        return self._index_by_group.get(group, []).copy()
    
    def get_cell_by_id(self, cell_id: str) -> Optional[CellCrop]:
        """Get a specific cell by its ID."""
        return self._index_by_cell_id.get(cell_id)
    
    def get_all_cells(self) -> List[CellCrop]:
        """Get all cells in the collection."""
        return self.cells.copy()
    
    def clear(self) -> None:
        """Clear all cells from the collection."""
        self.cells.clear()
        self._index_by_image_id.clear()
        self._index_by_group.clear()
        self._index_by_cell_id.clear()
    
    def get_groups(self) -> List[str]:
        """Get all unique groups in the collection."""
        return list(self._index_by_group.keys())
    
    def get_image_ids(self) -> List[str]:
        """Get all unique source image IDs in the collection."""
        return list(self._index_by_image_id.keys())
    
    def get_statistics(self) -> Dict[str, Any]:
        """Get statistics about the cell collection."""
        return {
            "total_cells": len(self.cells),
            "num_groups": len(self._index_by_group),
            "num_images": len(self._index_by_image_id),
            "cells_per_group": {group: len(cells) for group, cells in self._index_by_group.items()},
            "cells_per_image": {img_id: len(cells) for img_id, cells in self._index_by_image_id.items()}
        }


# Tool stage registry - maps tool names to their stages
TOOL_STAGE_REGISTRY: Dict[str, ToolStage] = {
    # Stage 1: Image-level tools
    "Image_Preprocessor_Tool": ToolStage.STAGE_1_IMAGE_LEVEL,
    "Nuclei_Segmenter_Tool": ToolStage.STAGE_1_IMAGE_LEVEL,
    "Cell_Segmenter_Tool": ToolStage.STAGE_1_IMAGE_LEVEL,  # For phase-contrast cell images
    "Organoid_Segmenter_Tool": ToolStage.STAGE_1_IMAGE_LEVEL,  # For organoid segmentation
    "Object_Detector_Tool": ToolStage.STAGE_1_IMAGE_LEVEL,
    "Advanced_Object_Detector_Tool": ToolStage.STAGE_1_IMAGE_LEVEL,
    "Relevant_Patch_Zoomer_Tool": ToolStage.STAGE_1_IMAGE_LEVEL,
    "Text_Detector_Tool": ToolStage.STAGE_1_IMAGE_LEVEL,
    
    # Stage 2: Cell-level tools
    "Single_Cell_Cropper_Tool": ToolStage.STAGE_2_CELL_LEVEL,
    "Cell_State_Analyzer_Tool": ToolStage.STAGE_2_CELL_LEVEL,  # Self-supervised learning for cell states
    "Fibroblast_Activation_Scorer_Tool": ToolStage.STAGE_2_CELL_LEVEL,
    
    # Non-image tools (not classified by stage)
    "Generalist_Solution_Generator_Tool": None,  # No stage classification
    "Python_Code_Generator_Tool": None,
    "ArXiv_Paper_Searcher_Tool": None,
    "Google_Search_Tool": None,
    "Wikipedia_Knowledge_Searcher_Tool": None,
    "URL_Text_Extractor_Tool": None,
    "Pubmed_Search_Tool": None,
    "Nature_News_Fetcher_Tool": None,
}


def get_tool_stage(tool_name: str) -> Optional[ToolStage]:
    """Get the stage classification for a tool."""
    return TOOL_STAGE_REGISTRY.get(tool_name)


def is_stage_1_tool(tool_name: str) -> bool:
    """Check if a tool is a Stage 1 (image-level) tool."""
    return TOOL_STAGE_REGISTRY.get(tool_name) == ToolStage.STAGE_1_IMAGE_LEVEL


def is_stage_2_tool(tool_name: str) -> bool:
    """Check if a tool is a Stage 2 (cell-level) tool."""
    return TOOL_STAGE_REGISTRY.get(tool_name) == ToolStage.STAGE_2_CELL_LEVEL


@dataclass
class BatchImage:
    group: str
    image_id: str
    image_path: str
    image_name: str


@dataclass
class CellCrop:
    """
    Atomic data object representing a single cell.
    All downstream analysis operates on lists of CellCrop objects.
    
    This is the fundamental unit of analysis after image segmentation.
    """
    # Core identifiers
    cell_id: str  # Unique identifier for this cell (e.g., "cell_0001")
    crop_id: str  # Unique identifier for this crop instance (can be same as cell_id)
    
    # Source tracking
    source_image_id: str  # ID of the source image this cell came from
    source_image_path: str  # Path to the source image
    group: str  # Group/condition label (e.g., "control", "treatment")
    
    # Cell data
    crop_path: Optional[str] = None  # Path to cropped cell image
    mask_path: Optional[str] = None  # Path to cell mask
    crop_image: Any = None  # Optional: in-memory image array (for efficiency)
    mask_image: Any = None  # Optional: in-memory mask array
    
    # Spatial metadata
    bbox: Optional[List[int]] = None  # [min_row, min_col, max_row, max_col] in source image
    centroid: Optional[List[float]] = None  # [row, col] in source image
    area: Optional[int] = None  # Pixel area of the cell
    
    # Extracted features (from feature extraction tools)
    features: Optional[Dict[str, Any]] = None  # Feature vectors, embeddings, etc.
    feature_vector: Optional[List[float]] = None  # Flattened feature vector for ML
    
    # Model predictions
    predictions: Optional[Dict[str, Any]] = None  # Model predictions, classifications, scores
    predicted_class: Optional[str] = None  # Predicted cell type/state
    confidence: Optional[float] = None  # Prediction confidence
    
    # Metadata
    metadata: Dict[str, Any] = field(default_factory=dict)  # Additional metadata
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert CellCrop to dictionary for serialization."""
        return {
            "cell_id": self.cell_id,
            "crop_id": self.crop_id,
            "source_image_id": self.source_image_id,
            "source_image_path": self.source_image_path,
            "group": self.group,
            "crop_path": self.crop_path,
            "mask_path": self.mask_path,
            "bbox": self.bbox,
            "centroid": self.centroid,
            "area": self.area,
            "features": self.features,
            "feature_vector": self.feature_vector,
            "predictions": self.predictions,
            "predicted_class": self.predicted_class,
            "confidence": self.confidence,
            "metadata": self.metadata
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> "CellCrop":
        """Create CellCrop from dictionary."""
        # Handle legacy format compatibility
        if "image_id" in data and "source_image_id" not in data:
            data["source_image_id"] = data.pop("image_id")
        if "path" in data and "crop_path" not in data:
            data["crop_path"] = data.pop("path")
        
        return cls(**data)



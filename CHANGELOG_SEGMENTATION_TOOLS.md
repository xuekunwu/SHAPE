# Changelog: Cell and Organoid Segmentation Tools

This document summarizes the changes made to extend the system from fibroblast-specific to general cell types and organoids.

## Overview

The system has been extended to support:
1. **Phase-contrast cell images** - Using Cell_Segmenter_Tool with CPSAM models
2. **Organoid segmentation** - Using Organoid_Segmenter_Tool with pretrained Cellpose models

## New Tools

### 1. Cell_Segmenter_Tool

**Location**: `octotools/tools/cell_segmenter/`

**Purpose**: Segments whole cells in phase-contrast microscopy images using Cellpose CPSAM model.

**Features**:
- Supports CPSAM model for phase-contrast cell images
- Falls back to Cellpose 'cyto' model if CPSAM not available
- Compatible with existing workflow (works with Image_Preprocessor_Tool, Single_Cell_Cropper_Tool)
- Generates cell masks (not just nuclei)

**Usage**:
```python
from octotools.tools.cell_segmenter.tool import Cell_Segmenter_Tool

tool = Cell_Segmenter_Tool()
result = tool.execute(
    image="path/to/phase_contrast_cells.png",
    diameter=30,
    flow_threshold=0.4,
    model_type="cpsam"
)
```

### 2. Organoid_Segmenter_Tool

**Location**: `octotools/tools/organoid_segmenter/`

**Purpose**: Segments organoids in microscopy images using pretrained Cellpose models.

**Features**:
- **REQUIRES** a specialized organoid segmentation model (mandatory)
- **Does NOT support** standard Cellpose models (cyto/cyto2) - these will not work for organoids
- Downloads organoid model from Hugging Face or accepts model_path parameter
- Optimized parameters for organoid segmentation (larger diameter defaults)
- Compatible with existing analysis pipeline

**Usage**:
```python
from octotools.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool

tool = Organoid_Segmenter_Tool()
result = tool.execute(
    image="path/to/organoids.png",
    diameter=100,  # Larger default for organoids
    flow_threshold=0.4,
    model_type="cyto2"
)
```

## Updated Components

### 1. Tool Priority System (`octotools/models/tool_priority.py`)

**Changes**:
- Added `Cell_Segmenter_Tool` and `Organoid_Segmenter_Tool` to HIGH priority
- Updated tool dependencies to include new segmentation tools
- Added keywords: 'organoid', 'organoids', 'tissue', 'spheroid', 'spheroids'

### 2. Task State Registry (`octotools/models/task_state.py`)

**Changes**:
- Registered both new tools as STAGE_1_IMAGE_LEVEL tools

### 3. Executor (`octotools/models/executor.py`)

**Changes**:
- Extended special handling for segmentation tools to support:
  - Processed image chaining from Image_Preprocessor_Tool
  - Mask chaining to Single_Cell_Cropper_Tool (supports nuclei_mask, cell_mask, organoid_mask)
- Updated tool command generation prompts

### 4. Planner (`octotools/models/planner.py`)

**Changes**:
- Added new tools to bioimage tools list
- Updated tool dependency descriptions
- Included new tools in example tool lists

### 5. Application Interface (`app.py`)

**Changes**:
- Added tool-specific descriptions for new segmentation tools
- Updated Single_Cell_Cropper_Tool description to reflect multi-source support

## Model Upload

See `MODEL_UPLOAD_GUIDE.md` for detailed instructions on:
- Uploading CPSAM models for Cell_Segmenter_Tool
- Uploading custom organoid models for Organoid_Segmenter_Tool
- Updating tool code to use custom models

## Workflow Integration

### Cell Segmentation Workflow

```
Image → Image_Preprocessor_Tool → Cell_Segmenter_Tool → Single_Cell_Cropper_Tool → Analysis
```

### Organoid Segmentation Workflow

```
Image → Image_Preprocessor_Tool → Organoid_Segmenter_Tool → Single_Cell_Cropper_Tool → Analysis
```

### Backward Compatibility

The existing workflow using `Nuclei_Segmenter_Tool` remains fully functional:
```
Image → Image_Preprocessor_Tool → Nuclei_Segmenter_Tool → Single_Cell_Cropper_Tool → Analysis
```

## Configuration

### Environment Variables

Ensure `HUGGINGFACE_TOKEN` is set for model downloads:
```bash
export HUGGINGFACE_TOKEN="your_token_here"
```

### Model Repository IDs

After uploading models, update the following in tool files:

**Cell_Segmenter_Tool** (`octotools/tools/cell_segmenter/tool.py`):
```python
repo_id="your-username/cell-segmenter-cpsam-model"  # Line ~51
filename="cpsam_model"  # Line ~52
```

**Organoid_Segmenter_Tool** (`octotools/tools/organoid_segmenter/tool.py`):
```python
repo_id="your-username/organoid-segmenter-model"  # Line ~47
filename="organoid_model"  # Line ~48
```

## Testing

Test the new tools with phase-contrast cell images and organoid images:

```python
# Test Cell_Segmenter_Tool
from octotools.tools.cell_segmenter.tool import Cell_Segmenter_Tool
tool = Cell_Segmenter_Tool()
result = tool.execute(image="examples/fibroblast.png")
print(f"Cells detected: {result['cell_count']}")

# Test Organoid_Segmenter_Tool
from octotools.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
tool = Organoid_Segmenter_Tool()
result = tool.execute(image="examples/organoid_image.png")
print(f"Organoids detected: {result['organoid_count']}")
```

## Future Enhancements

1. **Model Training Pipeline**: Scripts for training custom CPSAM models
2. **Multi-model Support**: Automatic model selection based on image characteristics
3. **Organoid Analysis Tools**: Specialized analysis tools for organoid morphology
4. **Batch Processing**: Enhanced batch processing for large organoid datasets

## Notes

- Both tools maintain backward compatibility with existing workflows
- The system automatically detects appropriate tools based on user queries
- All tools generate compatible mask formats for downstream analysis
- Visualization and output formats remain consistent across tools

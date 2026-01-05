# Project File Structure and Naming Convention Documentation

## Overview
This document describes the file naming and storage structure used throughout the project to ensure consistent file organization and prevent path-related errors.

## Directory Structure

### Base Structure
```
{solver_cache}/
└── {query_id}/                    # Query-specific cache directory
    └── tool_cache/                # All tool outputs go here
        ├── {group}/               # Group subdirectories (created by some tools)
        │   └── {files}            # Group-specific files
        └── {files}                # Direct tool outputs (metadata, etc.)
```

### Query Cache Directory
- **Location**: `{solver_cache}/{query_id}/`
- **Purpose**: Root directory for all files related to a specific query
- **Example**: `/tmp/solver_cache/2d1d6a184495417fa2294fb620d903cb/`

### Tool Cache Directory
- **Location**: `{query_cache_dir}/tool_cache/`
- **Purpose**: All tool outputs are stored here
- **Example**: `/tmp/solver_cache/2d1d6a184495417fa2294fb620d903cb/tool_cache/`

## Tool-Specific File Storage

### 1. Image_Preprocessor_Tool
- **Output Location**: `{tool_cache_dir}/{group}/`
- **File Naming**: `{image_identifier}_{group}_processed.png`
- **Example**: `tool_cache/default/A1_02_1_1_Phase Contrast_001_default_processed.png`
- **Note**: Creates group subdirectories

### 2. Nuclei_Segmenter_Tool / Cell_Segmenter_Tool
- **Output Location**: `{tool_cache_dir}/`
- **File Naming**: 
  - `nuclei_mask_{image_identifier}.tif` (for Nuclei_Segmenter_Tool)
  - `cell_mask_{image_identifier}.tif` (for Cell_Segmenter_Tool)
- **Example**: `tool_cache/nuclei_mask_c3e4dac8a1a740baa9cf843ce5873570.tif`

### 3. Single_Cell_Cropper_Tool
- **Output Location**: `{tool_cache_dir}/`
- **File Naming**:
  - Cell crops: `cell_{idx:04d}_crop.{format}`
  - Cell masks: `cell_{idx:04d}_mask.{format}`
  - **Metadata**: `cell_crops_metadata_{uuid8}.json` ⚠️ **CRITICAL**
- **Metadata Location**: `{tool_cache_dir}/cell_crops_metadata_*.json`
- **Example**: `tool_cache/cell_crops_metadata_a1b2c3d4.json`
- **Note**: 
  - Metadata is saved directly in `tool_cache_dir`, NOT in group subdirectories
  - Cell crops are also saved directly in `tool_cache_dir`
  - The `group` parameter is stored in metadata, not used for directory structure

### 4. Cell_State_Analyzer_Tool
- **Input**: Reads metadata from `{tool_cache_dir}/cell_crops_metadata_*.json`
- **Search Strategy**:
  1. First checks `{tool_cache_dir}/` directly
  2. Then checks all subdirectories in `{tool_cache_dir}/` (including group subdirectories)
  3. Merges all found metadata files for multi-image processing
- **Output Location**: `{tool_cache_dir}/`
- **File Naming**: `adata_{timestamp}.h5ad`, `cell_metadata_{timestamp}.csv`

## Critical Path Handling Rules

### Executor Path Management
1. **query_cache_dir**: Always the parent directory (without 'tool_cache')
   - Example: `/tmp/solver_cache/2d1d6a184495417fa2294fb620d903cb`
   - Normalized in `set_query_cache_dir()` to remove 'tool_cache' suffix if present

2. **tool_cache_dir**: Always `{query_cache_dir}/tool_cache`
   - Example: `/tmp/solver_cache/2d1d6a184495417fa2294fb620d903cb/tool_cache`
   - Set via `tool.set_custom_output_dir(self.tool_cache_dir)` in executor

### Single_Cell_Cropper_Tool Path Handling
1. Receives `query_cache_dir` parameter (parent directory)
2. Creates `tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")`
3. Saves metadata to: `{tool_cache_dir}/cell_crops_metadata_{uuid}.json`
4. **Does NOT** create group subdirectories for metadata

### Cell_State_Analyzer_Tool Path Handling
1. Receives `query_cache_dir` parameter (parent directory)
2. Searches in:
   - `{query_cache_dir}/tool_cache/` (primary location)
   - All subdirectories of `{query_cache_dir}/tool_cache/` (including group subdirectories)
3. Merges all found metadata files

## Common Issues and Solutions

### Issue 1: Metadata Not Found
**Symptom**: `No metadata files found. Searched in: {path}`

**Possible Causes**:
1. Single_Cell_Cropper_Tool not executed before Cell_State_Analyzer_Tool
2. Metadata saved in group subdirectory but Cell_State_Analyzer_Tool only checking root
3. Path mismatch between save and load locations

**Solution**: 
- Updated `_load_cell_data_from_metadata()` to search in all subdirectories
- Added comprehensive logging to track file locations

### Issue 2: Path Normalization
**Symptom**: Path contains 'tool_cache' when it shouldn't

**Solution**: 
- `set_query_cache_dir()` normalizes paths to ensure `query_cache_dir` is always parent directory
- Command generation uses normalized paths

## Best Practices

1. **Always use `query_cache_dir` parameter** (parent directory) when calling tools
2. **Never hardcode paths** - use `os.path.join()` for cross-platform compatibility
3. **Normalize paths** before comparison or storage
4. **Log paths** during save and load operations for debugging
5. **Check subdirectories** when searching for files that might be in group subdirectories

## File Naming Conventions

### Metadata Files
- Pattern: `{tool_name}_metadata_{uuid8}.json`
- Example: `cell_crops_metadata_a1b2c3d4.json`
- UUID: 8-character hexadecimal string

### Image Files
- Pattern: `{prefix}_{identifier}_{suffix}.{ext}`
- Examples:
  - `nuclei_mask_c3e4dac8a1a740baa9cf843ce5873570.tif`
  - `A1_02_1_1_Phase Contrast_001_default_processed.png`
  - `cell_0001_crop.png`

### Data Files
- Pattern: `{type}_{timestamp}.{ext}`
- Examples:
  - `adata_20240104_123456.h5ad`
  - `cell_metadata_20240104_123456.csv`

## Summary

**Key Points**:
1. All tool outputs go to `{query_cache_dir}/tool_cache/`
2. Some tools (Image_Preprocessor_Tool) create group subdirectories
3. Single_Cell_Cropper_Tool saves metadata directly in `tool_cache_dir`, NOT in subdirectories
4. Cell_State_Analyzer_Tool searches both root and subdirectories to find metadata
5. Path normalization ensures consistent directory structure
6. Comprehensive logging helps debug path-related issues


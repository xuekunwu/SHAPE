# Image ID Flow Documentation

## Overview
This document describes the complete flow of `image_id` through the entire pipeline, from image upload to final processing. The goal is to ensure consistent tracking and matching across all tools.

## Key Principle
**Always use `image_id` (UUID) for tracking, never use `image_name` for tool execution.**

## Complete Flow

### 1. Image Upload (`app.py: add_image_to_group`)
- **Input**: User uploads image(s)
- **Process**:
  - Generate unique `image_id = uuid.uuid4().hex` (e.g., `"7dd4a7833d27422fb4dca0cd49298867"`)
  - Save image as `{image_id}.jpg` in `images/{group}/` directory
  - Extract `image_name` from original filename (e.g., `"A2_02_1_1_Phase Contrast_001"`)
- **Storage**:
  ```python
  entry = {
      "image_id": image_id,        # UUID: "7dd4a783..."
      "image_name": image_name,    # Original filename: "A2_02_1_1_Phase Contrast_001"
      "image_path": str(image_path), # Full path: "images/Control/7dd4a783....jpg"
      "fingerprint": fingerprint,
      "features_path": feature_path
  }
  ```

### 2. Tool Execution (`app.py: stream_solve_user_problem`)
- **Input**: `img_item` from `image_groups[group]["images"]`
- **Process**:
  - Extract `image_id = img_item.get("image_id")` (UUID)
  - Extract `image_name = img_item.get("image_name")` (for reference only)
  - **Pass `image_id` to executor**: `executor.generate_tool_command(..., image_id=image_id)`
  - **DO NOT use `image_name` for tool execution** - it's only for display/reference

### 3. Image_Preprocessor_Tool
- **Input**: `image_id` parameter (UUID from step 2)
- **Process**:
  - Use `image_id` as `image_identifier` if provided
  - Otherwise, extract from filename (fallback)
  - Process image and save with `{image_identifier}_{group}_processed.png`
- **Output**:
  ```python
  {
      "image_id": image_identifier,  # Same as input image_id
      "image_identifier": image_identifier,  # Alias
      "processed_image_path": "...",
      "visual_outputs": [...],
      ...
  }
  ```
  - If multiple images: `{"per_image": [result1, result2, ...]}`

### 4. Cell_Segmenter_Tool / Nuclei_Segmenter_Tool
- **Input**: `image_id` parameter (UUID from step 2)
- **Process**:
  - Use `image_id` as `image_identifier` if provided
  - Otherwise, extract from filename (fallback)
  - Generate mask: `cell_mask_{image_identifier}.tif` or `nuclei_mask_{image_identifier}.tif`
- **Output**:
  ```python
  {
      "image_id": image_identifier,  # Same as input image_id
      "visual_outputs": [overlay_path, mask_path],
      "cell_count": n_cells,
      ...
  }
  ```
  - If multiple images: `{"per_image": [result1, result2, ...]}`

### 5. Single_Cell_Cropper_Tool (Executor Matching)
- **Input**: 
  - `image_id` parameter (UUID from step 2)
  - `previous_outputs` from segmentation tool (contains `per_image` structure)
- **Process**:
  - Match by `image_id`: `img_result.get('image_id') == image_id`
  - Extract `visual_outputs` from matched result
  - Find mask file in `visual_outputs`
  - Execute cropping with mask
- **Output**:
  ```python
  {
      "cell_crops_metadata_path": "...",
      "cell_count": n_cells,
      ...
  }
  ```

### 6. Cell_State_Analyzer_Tool
- **Input**: `query_cache_dir` (parent directory)
- **Process**:
  - Search for metadata files in `{query_cache_dir}/tool_cache/`
  - Load all metadata files (from all images)
  - Merge cell crops from all images
  - Perform unified analysis
- **Output**: Analysis results with merged data

## Critical Rules

1. **Always use `image_id` (UUID) for tool execution**
   - ✅ `executor.generate_tool_command(..., image_id=image_id)`
   - ❌ `executor.generate_tool_command(..., image_id=image_name)`

2. **Tools must return the same `image_id` they received**
   - If tool receives `image_id="7dd4a783..."`, it must return `image_id="7dd4a783..."`
   - This ensures matching works correctly

3. **Matching logic is simple**
   - `img_result.get('image_id') == image_id`
   - No path matching, no string parsing, just direct comparison

4. **`image_name` is for display only**
   - Used in UI, logs, file naming (readable)
   - Never used for tool execution or matching

## File Naming Convention

- **Stored images**: `{image_id}.jpg` (e.g., `7dd4a7833d27422fb4dca0cd49298867.jpg`)
- **Processed images**: `{image_id}_{group}_processed.png` (e.g., `7dd4a783..._Control_processed.png`)
- **Masks**: `cell_mask_{image_id}.tif` or `nuclei_mask_{image_id}.tif`
- **Metadata**: `cell_crops_metadata_{uuid}.json` (contains `source_image_id` field)

## Troubleshooting

If matching fails:
1. Check that `image_id` is passed correctly to `executor.generate_tool_command()`
2. Verify that tools return `image_id` in their output
3. Check that `per_image` structure contains `image_id` in each result
4. Ensure no string manipulation (e.g., `image_name` vs `image_id`) is used for matching


# Multi-Image Processing Workflow

## Current Implementation

### Image Processing Flow

1. **Image Collection**: Images are collected from `group_images` parameter
   - `image_items = group_images or []`
   - Each image item contains: `image_id`, `image_path`, `group`, `fingerprint`, `image_name`

2. **Tool Execution Strategy**:
   - **Per-Image Tools**: Most tools process each image separately
     - `Image_Preprocessor_Tool`
     - `Nuclei_Segmenter_Tool`
     - `Cell_Segmenter_Tool`
     - `Organoid_Segmenter_Tool`
     - `Single_Cell_Cropper_Tool`
   - **Merge-All Tools**: These tools analyze all images together (only execute once)
     - `Cell_State_Analyzer_Tool` - Merges all cell crops from all images
     - `Analysis_Visualizer_Tool` - Compares groups across all images

3. **Processing Loop** (lines 1190-1295 in app.py):
   ```python
   for img_idx, img_item in enumerate(image_items):
       # For merge-all tools: skip after first image
       if should_execute_once and img_idx > 0:
           # Reuse merged result
           continue
       
       # Check cache
       if cached_artifact:
           # Use cached result
       else:
           # Execute tool
           result = execute_tool_command(...)
       
       results_per_image.append(result)
   ```

4. **Result Aggregation**:
   - **Merge-all tools**: Return single merged result
   - **Per-image tools**: Return `{"per_image": [result1, result2, ...]}`

## Known Issues

### Issue 1: Only First Image Processed
**Symptom**: System shows "2 images from groups: A, B" but only processes first image.

**Possible Causes**:
1. `image_items` may only contain one image (check line 926)
2. Loop may exit early due to error (check error handling)
3. Tool execution failure on first image may prevent second image processing

**Debug Steps**:
- Check console logs for "Processing X image(s)" message
- Verify `len(image_items)` is correct
- Check if tool execution fails and prevents loop continuation

### Issue 2: Metadata File Not Found
**Symptom**: `Cell_State_Analyzer_Tool` fails with "No metadata files found"

**Root Cause**: 
- Each `Single_Cell_Cropper_Tool` execution creates a separate metadata file
- `Cell_State_Analyzer_Tool` needs to merge all metadata files
- Path handling issue: `query_cache_dir` may already contain `tool_cache`

**Fix Applied**:
- Modified `_load_cell_data_from_metadata` to merge all metadata files
- Fixed path handling in executor to correctly detect parent directory

## Debugging Checklist

1. **Verify Image Items**:
   ```python
   print(f"image_items count: {len(image_items)}")
   for idx, item in enumerate(image_items):
       print(f"  Image {idx}: {item.get('image_id')} (group: {item.get('group')})")
   ```

2. **Check Tool Type**:
   ```python
   print(f"Tool: {tool_name}, should_execute_once: {should_execute_once}")
   ```

3. **Monitor Loop Progress**:
   ```python
   print(f"Processing image {img_idx + 1}/{len(image_items)}")
   ```

4. **Check Results**:
   ```python
   print(f"results_per_image count: {len(results_per_image)}")
   ```

## Expected Behavior

### For Image_Preprocessor_Tool (Per-Image Tool):
- Should process **each image separately**
- Each image gets its own processed output
- Results stored as `{"per_image": [result1, result2]}`

### For Cell_State_Analyzer_Tool (Merge-All Tool):
- Should process **all images together** (only execute once)
- Merges all cell crops from all images
- Returns single merged result

## Next Steps

1. Add comprehensive logging to track image processing
2. Verify `image_items` contains all images
3. Check for early loop exits
4. Ensure error handling doesn't prevent subsequent image processing


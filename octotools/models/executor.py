import os
import json
# import sys
import importlib
import re
from typing import Dict, Any, List
from datetime import datetime

from octotools.engine.openai import ChatOpenAI 
from octotools.models.formatters import ToolCommand
from octotools.models.utils import get_llm_safe_result
from octotools.utils import logger, ResponseParser

import signal
from typing import Dict, Any, List, Optional
import uuid
from contextlib import redirect_stdout, redirect_stderr

class TimeoutError(Exception):
    pass

def timeout_handler(signum, frame):
    raise TimeoutError("Function execution timed out")

class Executor:
    def __init__(self, llm_engine_name: str, query_cache_dir: str = "solver_cache",  num_threads: int = 1, max_time: int = 120, max_output_length: int = 100000, enable_signal: bool = True, api_key: str = None, initializer=None):
        self.llm_engine_name = llm_engine_name
        self.query_cache_dir = query_cache_dir
        self.tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
        self.num_threads = num_threads
        self.max_time = max_time
        self.max_output_length = max_output_length
        self.enable_signal = enable_signal
        self.api_key = api_key
        self.initializer = initializer

    def set_query_cache_dir(self, query_cache_dir):
        if query_cache_dir:
            # Normalize query_cache_dir: if it ends with 'tool_cache', remove it to get parent directory
            # query_cache_dir should always be the parent directory (without 'tool_cache')
            normalized_query_cache_dir = query_cache_dir.rstrip(os.sep)
            if normalized_query_cache_dir.endswith('tool_cache'):
                # Remove 'tool_cache' suffix to get parent directory
                normalized_query_cache_dir = os.path.dirname(normalized_query_cache_dir)
            elif normalized_query_cache_dir.endswith(os.path.join('', 'tool_cache')):
                normalized_query_cache_dir = os.path.dirname(normalized_query_cache_dir)
            
            self.query_cache_dir = normalized_query_cache_dir
            # tool_cache_dir should always be query_cache_dir + 'tool_cache'
            self.tool_cache_dir = os.path.join(self.query_cache_dir, "tool_cache")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.query_cache_dir, timestamp)
            self.tool_cache_dir = os.path.join(self.query_cache_dir, "tool_cache")
        # Ensure both directories exist
        os.makedirs(self.query_cache_dir, exist_ok=True)
        os.makedirs(self.tool_cache_dir, exist_ok=True)
        logger.debug(f"Executor: Updated query_cache_dir to {self.query_cache_dir}")
        logger.debug(f"Executor: Updated tool_cache_dir to {self.tool_cache_dir}")
    
    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], memory=None, bytes_mode:bool = False, conversation_context: str = "", image_id: str = None, current_image_path: str = None, **kwargs) -> ToolCommand:
        """
        Generate a tool command based on the given information.
        
        Args:
            question: The user's question
            image: The image path
            context: The context information
            sub_goal: The sub-goal for this step
            tool_name: The name of the tool to execute
            tool_metadata: The metadata for the tool
            memory: The memory object containing previous actions
            bytes_mode: Whether the image is in bytes mode
            
        Returns:
            A ToolCommand object
        """
        actual_image_path = image if not bytes_mode else 'image.jpg'
        safe_path = actual_image_path.replace("\\", "\\\\") if actual_image_path else ""
        
        # Extract previous outputs from memory for tool chaining
        # Use full results (not LLM-safe) to get file paths for tool chaining
        previous_outputs = {}
        previous_outputs_for_llm = {}  # LLM-safe version (no file paths)
        if memory and hasattr(memory, 'get_actions'):
            actions = memory.get_actions(llm_safe=False)  # Get full results for executor use
            if actions:
                # Get the most recent action's result
                last_action = actions[-1]
                if 'result' in last_action:
                    raw_result = last_action['result']
                    # Handle per_image structure: keep the full structure for matching
                    # We need to match the correct image from per_image based on current_image_path or image_id
                    # Don't extract a single result here - let the tool-specific handlers do the matching
                    if isinstance(raw_result, dict) and 'per_image' in raw_result:
                        # Keep the full per_image structure for matching
                        previous_outputs = raw_result
                        logger.debug(f"Keeping per_image structure for matching (contains {len(raw_result['per_image'])} images)")
                    else:
                        previous_outputs = raw_result
                    # Create LLM-safe version (summary only, no file paths)
                    previous_outputs_for_llm = get_llm_safe_result(previous_outputs)
                    logger.debug(f"Extracted previous outputs: {list(previous_outputs.keys()) if isinstance(previous_outputs, dict) else 'Not a dict'}")
        
        # Special handling for Cell_State_Analyzer_Tool to use dynamic metadata file discovery
        state_analyzer_tools = ["Cell_State_Analyzer_Tool"]
        if tool_name in state_analyzer_tools:
            tool_label = "Cell_State_Analyzer_Tool"
            # Use dynamic paths from executor's cache directories
            # Escape backslashes for Windows paths in Python string
            metadata_dir_str = self.tool_cache_dir.replace("\\", "\\\\")
            query_cache_dir_str = self.query_cache_dir.replace("\\", "\\\\")
            return ToolCommand(
                analysis=f"Using dynamic metadata file discovery for {tool_label}",
                explanation=f"Automatically finding the most recent metadata file and loading cell data with improved format handling for {tool_label}",
                command=f"""import json
import os
import glob

# Dynamically find all metadata files (for multi-image processing)
# _load_cell_data_from_metadata expects query_cache_dir (parent directory) and constructs tool_cache path internally
# query_cache_dir should already be the parent directory (without 'tool_cache')
tool_cache_dir_str = r'{self.tool_cache_dir}'
query_cache_dir_str = r'{self.query_cache_dir}'

# query_cache_dir should already be the parent directory, use it directly
# If for some reason it contains 'tool_cache', extract parent directory
if 'tool_cache' in query_cache_dir_str:
    # Extract parent directory if query_cache_dir contains 'tool_cache'
    if query_cache_dir_str.endswith(os.path.sep + 'tool_cache') or query_cache_dir_str.endswith('/tool_cache'):
        query_cache_dir_parent = os.path.dirname(query_cache_dir_str)
    elif query_cache_dir_str.endswith('tool_cache'):
        query_cache_dir_parent = os.path.dirname(query_cache_dir_str)
    else:
        # Find the parent of the 'tool_cache' directory
        parts = query_cache_dir_str.split('tool_cache')
        query_cache_dir_parent = parts[0].rstrip(os.sep) if parts[0] else query_cache_dir_str
else:
    # query_cache_dir is already the parent directory
    query_cache_dir_parent = query_cache_dir_str

logger.info(f"Looking for metadata files in: {{tool_cache_dir_str}}")
logger.info(f"Using query_cache_dir_parent: {{query_cache_dir_parent}}")

# First, check if metadata files exist in tool_cache_dir directly
direct_metadata_files = glob.glob(os.path.join(tool_cache_dir_str, 'cell_crops_metadata_*.json'))
logger.info(f"Found {{len(direct_metadata_files)}} metadata file(s) directly in tool_cache_dir")

# Also check the path that _load_cell_data_from_metadata will use
expected_metadata_dir = os.path.join(query_cache_dir_parent, 'tool_cache')
expected_metadata_files = glob.glob(os.path.join(expected_metadata_dir, 'cell_crops_metadata_*.json'))
logger.info(f"Found {{len(expected_metadata_files)}} metadata file(s) in expected path: {{expected_metadata_dir}}")

try:
    # Use the tool's improved metadata loading method (merges all metadata files)
    # Pass query_cache_dir (parent directory), the method will construct tool_cache path internally
    cell_crops, cell_metadata = tool._load_cell_data_from_metadata(query_cache_dir_parent)
    
    if cell_crops and len(cell_crops) > 0:
        # Execute the tool with loaded data (merged from all metadata files)
        execution = tool.execute(
            cell_crops=cell_crops, 
            cell_metadata=cell_metadata, 
            max_epochs=25,
            early_stop_loss=0.5,
            batch_size=16,
            learning_rate=3e-5,
            cluster_resolution=0.5,
            query_cache_dir=query_cache_dir_parent
        )
    else:
        execution = {{"error": "No valid cell crops found in metadata", "status": "failed"}}
    
except Exception as e:
    logger.error("Error loading metadata: " + str(e))
    import traceback
    logger.error("Traceback: " + traceback.format_exc())
    execution = {{"error": "Failed to load metadata: " + str(e), "status": "failed"}}"""
            )
        
        # Special handling for segmentation tools (Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, Organoid_Segmenter_Tool) 
        # to use processed image from Image_Preprocessor_Tool
        segmentation_tools = ["Nuclei_Segmenter_Tool", "Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"]
        if tool_name in segmentation_tools and previous_outputs:
            # Handle per_image structure: find the processed_image_path for the current image
            processed_image_path = None
            if 'per_image' in previous_outputs:
                # If previous_outputs has per_image structure, find the matching image
                per_image_list = previous_outputs['per_image']
                # Try to match by image_id or image path
                for img_result in per_image_list:
                    if isinstance(img_result, dict):
                        # Check if this result matches the current image
                        # Match by checking if the processed_image_path contains the image_id or image name
                        if 'processed_image_path' in img_result:
                            candidate_path = img_result['processed_image_path']
                            # Try to match by image_id, image_path, or image name
                            matched = False
                            
                            # Method 1: Match by image_id if provided
                            if image_id and not matched:
                                # Extract image identifier from processed_image_path (filename without extension)
                                path_basename = os.path.basename(candidate_path)
                                path_identifier = os.path.splitext(path_basename)[0]
                                # Check if image_id matches (could be full name or part of it)
                                # Handle cases like "A2_02_1_1_Phase Contrast_001" matching "A2_02_1_1_Phase Contrast_001_default_processed"
                                if image_id in path_identifier or path_identifier.startswith(image_id) or image_id.split('_')[0] in path_identifier.split('_')[0]:
                                    processed_image_path = candidate_path
                                    matched = True
                                    logger.info(f"Matched processed_image_path for image_id '{image_id}': {processed_image_path}")
                            
                            # Method 2: Match by current_image_path if provided
                            if current_image_path and not matched:
                                # Extract original image name from current_image_path
                                current_image_name = os.path.splitext(os.path.basename(current_image_path))[0]
                                # Check if original_image_path in result matches
                                if 'original_image_path' in img_result:
                                    original_path = img_result['original_image_path']
                                    original_name = os.path.splitext(os.path.basename(original_path))[0]
                                    if current_image_name == original_name or current_image_path == original_path:
                                        processed_image_path = candidate_path
                                        matched = True
                                        logger.info(f"Matched processed_image_path for current_image_path '{current_image_path}': {processed_image_path}")
                            
                            # Method 3: Match by checking if image name appears in processed_image_path
                            if current_image_path and not matched:
                                current_image_name = os.path.splitext(os.path.basename(current_image_path))[0]
                                path_basename = os.path.basename(candidate_path)
                                if current_image_name in path_basename:
                                    processed_image_path = candidate_path
                                    matched = True
                                    logger.info(f"Matched processed_image_path by image name '{current_image_name}': {processed_image_path}")
                            
                            if matched:
                                break
            elif 'processed_image_path' in previous_outputs:
                # Direct processed_image_path (single image case)
                processed_image_path = previous_outputs['processed_image_path']
            
            if processed_image_path:
                # Normalize path for cross-platform compatibility
                # os is already imported at the top of the file
                processed_image_path = os.path.normpath(processed_image_path) if isinstance(processed_image_path, str) else str(processed_image_path)
                # Verify file exists before using it
                if not os.path.exists(processed_image_path):
                    logger.warning(f"processed_image_path does not exist: {processed_image_path}")
                    # Fall through to standard command generation
                else:
                    # Escape backslashes for Python string in command
                    safe_processed_path = processed_image_path.replace("\\", "\\\\")
                    # Include image_id and query_cache_dir if available for consistent naming
                    query_cache_dir_str = self.query_cache_dir.replace("\\", "\\\\")
                    image_id_param = f', image_id="{image_id}"' if image_id else ''
                    query_cache_dir_param = f', query_cache_dir=r"{query_cache_dir_str}"' if query_cache_dir_str else ''
                    tool_label = "nuclei segmentation" if tool_name == "Nuclei_Segmenter_Tool" else \
                                 "cell segmentation" if tool_name == "Cell_Segmenter_Tool" else \
                                 "organoid segmentation"
                    return ToolCommand(
                        analysis=f"Using the processed image from Image_Preprocessor_Tool for {tool_label}",
                        explanation=f"Using the processed image path '{processed_image_path}' from the previous Image_Preprocessor_Tool step (matched for image_id: {image_id})",
                        command=f"""execution = tool.execute(image="{safe_processed_path}"{image_id_param}{query_cache_dir_param})"""
                    )
        
        # Special handling for Analysis_Visualizer_Tool to use Cell_State_Analyzer_Tool results
        if tool_name == "Analysis_Visualizer_Tool" and previous_outputs:
            # Robust check: verify we have cell state analysis output
            has_adata_path = isinstance(previous_outputs, dict) and 'adata_path' in previous_outputs and previous_outputs.get('adata_path')
            has_analysis_type = isinstance(previous_outputs, dict) and previous_outputs.get('analysis_type') == 'cell_state_analysis'
            has_cluster_key = isinstance(previous_outputs, dict) and 'cluster_key' in previous_outputs
            
            logger.debug(f"Analysis_Visualizer_Tool special handling check: has_adata_path={has_adata_path}, "
                        f"has_analysis_type={has_analysis_type}, has_cluster_key={has_cluster_key}, "
                        f"previous_outputs_keys={list(previous_outputs.keys()) if isinstance(previous_outputs, dict) else 'not a dict'}")
            
            # Only proceed if we have the required fields
            if has_adata_path or (has_analysis_type and has_cluster_key):
                logger.info(f"✅ Using special handling for Analysis_Visualizer_Tool with Cell_State_Analyzer_Tool results")
                # Use adata_path directly from Cell_State_Analyzer_Tool output (should be well-defined)
                adata_path = previous_outputs.get('adata_path', '')
                cluster_key = previous_outputs.get('cluster_key', 'leiden_0.5')
                cluster_resolution = previous_outputs.get('cluster_resolution', 0.5)
                
                if not adata_path:
                    logger.error(f"❌ adata_path is empty in Cell_State_Analyzer_Tool output. Cannot proceed with visualization.")
                    # Fall through to standard command generation
                else:
                    logger.info(f"Using adata_path from Cell_State_Analyzer_Tool: {adata_path}")
                
                # Construct analysis_data dict for Analysis_Visualizer_Tool
                # Pass adata_path as-is: Cell_State_Analyzer_Tool should provide a well-defined path
                analysis_data_dict = {
                    'adata_path': adata_path,
                    'cluster_key': cluster_key,
                    'cluster_resolution': cluster_resolution,
                    'analysis_type': 'cell_state_analysis'
                }
                
                # Add metadata path if available
                if 'cell_metadata_path' in previous_outputs:
                    analysis_data_dict['cell_metadata_path'] = previous_outputs['cell_metadata_path']
                elif 'metadata_path' in previous_outputs:
                    analysis_data_dict['cell_metadata_path'] = previous_outputs['metadata_path']
                
                # Add cell count if available
                if 'cell_count' in previous_outputs:
                    analysis_data_dict['cell_count'] = previous_outputs['cell_count']
                
                # Use proper group_column - use 'group' if available in metadata, otherwise use cluster_key for cluster-based grouping
                group_col = 'group'  # Default to 'group' for multi-group comparison
                
                return ToolCommand(
                    analysis=f"Using Cell_State_Analyzer_Tool results for visualization",
                    explanation=f"Loading analysis data from Cell_State_Analyzer_Tool output (adata_path={adata_path}, cluster_key={cluster_key})",
                    command=f"""import json
analysis_data = {json.dumps(analysis_data_dict, indent=2).replace(chr(10), chr(10)+'    ')}
execution = tool.execute(
    analysis_data=analysis_data,
    chart_type='auto',
    comparison_metric='cell_count',
    group_column='{group_col}',
    output_dir='output_visualizations/cell_state_analysis',
    figure_size=(10, 6),
                    dpi=300
)"""
                )
            else:
                # Fallback: Let LLM generate command if fields are missing
                logger.warning(f"⚠️ Analysis_Visualizer_Tool special handling skipped: missing required fields. "
                             f"Has adata_path: {has_adata_path}, has analysis_type: {has_analysis_type}, "
                             f"has cluster_key: {has_cluster_key}. "
                             f"Previous outputs type: {type(previous_outputs)}, "
                             f"Keys: {list(previous_outputs.keys()) if isinstance(previous_outputs, dict) else 'N/A'}. "
                             f"Falling back to standard command generation.")
                # Continue to standard command generation below
        
        # Special handling for Single_Cell_Cropper_Tool to use mask from segmentation tools
        # Supports masks from Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, and Organoid_Segmenter_Tool
        if tool_name == "Single_Cell_Cropper_Tool" and previous_outputs:
            # Handle per_image structure: find the mask for the current image by image_id
            mask_path = None
            visual_outputs_list = []
            
            # Extract visual_outputs from previous_outputs (handle per_image structure)
            if 'per_image' in previous_outputs:
                # If previous_outputs has per_image structure, find the matching image by image_id
                per_image_list = previous_outputs['per_image']
                for img_result in per_image_list:
                    if isinstance(img_result, dict) and 'visual_outputs' in img_result:
                        # Direct match by image_id if provided
                        if image_id and img_result.get('image_id') == image_id:
                            visual_outputs_list = img_result['visual_outputs']
                            break
                # If no match found and image_id not provided, use first result as fallback
                if not visual_outputs_list and per_image_list:
                    visual_outputs_list = per_image_list[0].get('visual_outputs', [])
            elif 'visual_outputs' in previous_outputs:
                # Direct visual_outputs (single image case)
                visual_outputs_list = previous_outputs['visual_outputs']
            
            # Find mask file from visual_outputs
            logger.debug(f"Looking for mask in visual_outputs: {visual_outputs_list}")
            for output_path in visual_outputs_list:
                logger.debug(f"Checking output path: {output_path}")
                # Check for various mask types (support .png, .tif, .tiff formats)
                # Exclude visualization files (those with 'viz' in the name)
                output_path_lower = output_path.lower()
                is_mask_file = (
                    (output_path_lower.endswith('.png') or 
                     output_path_lower.endswith('.tif') or 
                     output_path_lower.endswith('.tiff')) and 
                    'viz' not in output_path_lower
                )
                
                if is_mask_file:
                    if 'nuclei_mask' in output_path_lower:
                        mask_path = output_path
                        logger.debug(f"Found nuclei mask path: {mask_path}")
                        break
                    elif 'cell_mask' in output_path_lower:
                        mask_path = output_path
                        logger.debug(f"Found cell mask path: {mask_path}")
                        break
                    elif 'organoid_mask' in output_path_lower:
                        mask_path = output_path
                        logger.debug(f"Found organoid mask path: {mask_path}")
                        break
            
            if mask_path:
                source_tool = "segmentation tool"
                if 'nuclei_mask' in mask_path.lower():
                    source_tool = "Nuclei_Segmenter_Tool"
                elif 'cell_mask' in mask_path.lower():
                    source_tool = "Cell_Segmenter_Tool"
                elif 'organoid_mask' in mask_path.lower():
                    source_tool = "Organoid_Segmenter_Tool"
                
                # Include query_cache_dir, source_image_id, and group to ensure metadata files are saved correctly
                # query_cache_dir should be the parent directory (without 'tool_cache')
                query_cache_dir_str = self.query_cache_dir.replace("\\", "\\\\")
                # Get group from kwargs first (passed from app.py), then fallback to extracting from image_id
                group = kwargs.get('group', "default")
                if group == "default" and image_id and '_' in image_id:
                    # Fallback: Extract group from image_id if available (e.g., "Control_7dd4a783..." -> "Control")
                    group = image_id.split('_')[0]
                
                # Use image_id as source_image_id for tracking
                source_image_id_param = f', source_image_id="{image_id}"' if image_id else ''
                group_param = f', group="{group}"'
                
                logger.info(f"Single_Cell_Cropper_Tool: Using query_cache_dir={self.query_cache_dir}, source_image_id={image_id}, group={group}")
                return ToolCommand(
                    analysis=f"Using the mask from {source_tool} for single cell cropping",
                    explanation=f"Using the mask path '{mask_path}' from the previous {source_tool} step (image: {image_id}, group: {group})",
                    command=f"""execution = tool.execute(original_image="{actual_image_path}", nuclei_mask="{mask_path}", min_area=50, margin=25, query_cache_dir=r'{query_cache_dir_str}'{source_image_id_param}{group_param})"""
                )
            else:
                logger.debug(f"No mask found in previous outputs: {previous_outputs['visual_outputs']}")
                # Fallback to standard command generation
                pass
        
        # Special handling for Image_Preprocessor_Tool to include group parameter
        if tool_name == "Image_Preprocessor_Tool" and 'group' in kwargs:
            group = kwargs.get('group', 'default')
            # Include group parameter in the prompt
            group_info = f"\nGroup: {group} (MUST include groups parameter in command: groups='{group}' or groups=['{group}'])"
        else:
            group_info = ""
        
        # For other tools, use the standard prompt
        # Include query_cache_dir in context for tools that need it
        query_cache_dir_str = self.query_cache_dir.replace("\\", "\\\\")
        prompt_generate_tool_command = f"""
Task: Generate a precise command to execute the selected tool based on the given information.

Conversation so far:
{conversation_context}

Query: {question}
Image Path: {safe_path}
Image ID: {image_id if image_id else "Not provided"}{group_info}
Context: {context}
Sub-Goal: {sub_goal}
Selected Tool: {tool_name}
Tool Metadata: {tool_metadata}
Previous Tool Outputs (summary only, file paths available for tool chaining): {previous_outputs_for_llm}
Query Cache Directory: {query_cache_dir_str} (use this for query_cache_dir parameter if the tool accepts it)

IMPORTANT: When the tool requires an image parameter, you MUST use the exact image path provided above: "{safe_path}"
{"IMPORTANT: Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, Organoid_Segmenter_Tool, and Image_Preprocessor_Tool accept the image_id parameter for consistent file naming and tracking. Include image_id parameter when available for these tools." if image_id else ""}
{f"CRITICAL: Image_Preprocessor_Tool requires the groups parameter when processing images. Use groups='{kwargs.get('group', 'default')}' or groups=['{kwargs.get('group', 'default')}'] in the command." if tool_name == "Image_Preprocessor_Tool" and 'group' in kwargs else ""}
{"CRITICAL: For Image_Preprocessor_Tool, if the query or context mentions 'organoid' or if Organoid_Segmenter_Tool was used previously, you MUST set skip_illumination_correction=True. Organoid images should NOT have illumination correction, only brightness adjustment." if tool_name == "Image_Preprocessor_Tool" and ('organoid' in question.lower() or 'organoid' in context.lower() or 'Organoid_Segmenter_Tool' in str(previous_outputs_for_llm)) else ""}

CRITICAL TOOL DEPENDENCY RULES:
- Fibroblast_Activation_Scorer_Tool MUST use the h5ad file output from Cell_State_Analyzer_Tool
- Fibroblast_Activation_Scorer_Tool CANNOT use cell_crops_metadata files directly
- If Cell_State_Analyzer_Tool has not been executed yet, Fibroblast_Activation_Scorer_Tool should not be executed
- The bioimage analysis tool chain must be: Image_Preprocessor_Tool -> (Cell_Segmenter_Tool/Nuclei_Segmenter_Tool/Organoid_Segmenter_Tool) -> Single_Cell_Cropper_Tool -> Cell_State_Analyzer_Tool -> Fibroblast_Activation_Scorer_Tool (optional)

Instructions:
1. Carefully review all provided information: the query, image path, context, sub-goal, selected tool, tool metadata, and previous tool outputs.
2. Analyze the tool's input_types from the metadata to understand required and optional parameters.
3. If previous tool outputs are available, use the appropriate file paths from those outputs (e.g., processed_image_path, nuclei_mask paths, h5ad files, etc.).
4. Construct a command or series of commands that aligns with the tool's usage pattern and addresses the sub-goal.
5. Ensure all required parameters are included and properly formatted.
6. Use appropriate values for parameters based on the given context, particularly the Context field which may contain relevant information from previous steps.
7. If multiple steps are needed to prepare data for the tool, include them in the command construction.
8. CRITICAL: If the tool requires an image parameter, use the exact image path "{safe_path}" provided above, unless a processed image path is available from previous outputs.
9. CRITICAL: For Fibroblast_Activation_Scorer_Tool, only use h5ad files from Cell_State_Analyzer_Tool, never use cell_crops_metadata files.

Output Format:
<analysis>: a step-by-step analysis of the context, sub-goal, and selected tool to guide the command construction.
<explanation>: a detailed explanation of the constructed command(s) and their parameters.
<command>: the Python code to execute the tool, which can be one of the following types:
    a. A single line command with execution = tool.execute().
    b. A multi-line command with complex data preparation, ending with execution = tool.execute().
    c. Multiple lines of execution = tool.execute() calls for processing multiple items.
```python
<your command here>
```

Rules:
1. The command MUST be valid Python code and include at least one call to tool.execute().
2. Each tool.execute() call MUST be assigned to the 'execution' variable in the format execution = tool.execute(...).
3. For multiple executions, use separate execution = tool.execute() calls for each execution.
4. The final output MUST be assigned to the 'execution' variable, either directly from tool.execute() or as a processed form of multiple executions.
5. Use the exact parameter names as specified in the tool's input_types.
6. Enclose string values in quotes, use appropriate data types for other values (e.g., lists, numbers).
7. Do not include any code or text that is not part of the actual command.
8. Ensure the command directly addresses the sub-goal and query.
9. Include ALL required parameters, data, and paths to execute the tool in the command itself.
10. If preparation steps are needed, include them as separate Python statements before the tool.execute() calls.
11. CRITICAL: If the tool requires an image parameter, use the exact image path "{safe_path}" provided above, unless a processed image path is available from previous outputs.
12. If previous tool outputs contain relevant file paths (e.g., processed_image_path, nuclei_mask paths, h5ad files), use those paths instead of the original image path when appropriate.
13. CRITICAL: Fibroblast_Activation_Scorer_Tool must use h5ad files from Cell_State_Analyzer_Tool, not cell_crops_metadata files.

Examples (Not to use directly unless relevant):

Example 1 (Single line command with actual image path):
<analysis>: The tool requires an image path and a list of labels for object detection.
<explanation>: We pass the actual image path and a list containing "baseball" as the label to detect.
<command>:
```python
execution = tool.execute(image="{safe_path}", labels=["baseball"])
```

Example 2 (Multi-line command with actual image path):
<analysis>: The tool requires an image path, multiple labels, and a threshold for object detection.
<explanation>: We prepare the data by defining variables for the image path, labels, and threshold, then pass these to the tool.execute() function.
<command>:
```python
image = "{safe_path}"
labels = ["baseball", "football", "basketball"]
threshold = 0.5
execution = tool.execute(image=image, labels=labels, threshold=threshold)
```

Example 3 (Image captioning with actual image path):
<analysis>: The tool requires an image path and an optional prompt for captioning.
<explanation>: We use the actual image path and provide a descriptive prompt.
<command>:
```python
execution = tool.execute(image="{safe_path}", prompt="Describe this image in detail.")
```

Some Wrong Examples:
<command>:
```python
execution1 = tool.execute(query="...")
execution2 = tool.execute(query="...")
```
Reason: only execution = tool.execute is allowed, not execution1 or execution2.

<command>:
```python
execution = tool.execute(image="path/to/image", labels=["baseball"])
```
Reason: Do not use placeholder paths like "path/to/image". Use the actual image path provided.

<command>:
```python
execution = tool.execute(cell_data="cell_crops_metadata.json")
```
Reason: Fibroblast_Activation_Scorer_Tool must use h5ad files from Cell_State_Analyzer_Tool, not cell_crops_metadata files.

Remember: Your <command> field MUST be valid Python code including any necessary data preparation steps and one or more execution = tool.execute( calls, without any additional explanatory text. The format execution = tool.execute must be strictly followed, and the last line must begin with execution = tool.execute to capture the final output. ALWAYS use the actual image path "{safe_path}" when the tool requires an image parameter, unless a processed image path is available from previous outputs. CRITICAL: Fibroblast_Activation_Scorer_Tool must use h5ad files from Cell_State_Analyzer_Tool.
"""

        try:
            llm_generate_tool_command = ChatOpenAI(model_string=self.llm_engine_name, is_multimodal=False, api_key=self.api_key)
            llm_response = llm_generate_tool_command.generate(prompt_generate_tool_command, response_format=ToolCommand)
            
            # Extract content and usage from response
            if isinstance(llm_response, dict) and 'content' in llm_response:
                tool_command = llm_response['content']
                usage_info = llm_response.get('usage', {})
                logger.debug(f"Tool command generation usage: {usage_info}")
            else:
                tool_command = llm_response
                usage_info = {}
            
            # Check if we got a string response (non-structured model like gpt-4-turbo) instead of ToolCommand object
            if isinstance(tool_command, str):
                logger.warning("Received string response instead of ToolCommand object")
                # Use unified parser
                try:
                    analysis, explanation, command = ResponseParser.parse_tool_command(tool_command)
                    tool_command = ToolCommand(
                        analysis=analysis,
                        explanation=explanation,
                        command=command
                    )
                    logger.debug(f"Created ToolCommand object from string: analysis length={len(analysis)}, explanation length={len(explanation)}, command length={len(command)}")
                except Exception as parse_error:
                    logger.error(f"Error parsing string response: {parse_error}")
                    # Create a default ToolCommand object
                    # Return error directly instead of trying to call tool.execute with error parameter
                    tool_command = ToolCommand(
                        analysis="Error parsing analysis",
                        explanation="Error parsing explanation",
                        command="execution = {'error': 'Error parsing command', 'status': 'failed'}"
                    )
            
            # Add usage information to the tool command for tracking
            if hasattr(tool_command, 'metadata'):
                tool_command.metadata = getattr(tool_command, 'metadata', {})
                tool_command.metadata['usage'] = usage_info
            else:
                # If ToolCommand doesn't have metadata, we'll handle this in the calling code
                pass
                
            return tool_command
        except Exception as e:
            logger.error(f"Error in tool command generation: {e}")
            # Fallback: create a basic ToolCommand with error information
            error_msg = str(e).replace("'", "\\'")  # Escape single quotes
            # Return error directly instead of trying to call tool.execute with error parameter
            return ToolCommand(
                analysis=f"Error generating tool command: {str(e)}",
                explanation="Failed to generate proper command due to response format parsing error",
                command=f"execution = {{'error': 'Command generation failed: {error_msg}', 'status': 'failed'}}"
            )

    def extract_explanation_and_command(self, response) -> tuple:
        """Extract analysis, explanation, and command from ToolCommand response."""
        try:
            # Use unified parser
            analysis, explanation, command = ResponseParser.parse_tool_command(response)
            logger.debug(f"Extracted: analysis length={len(analysis)}, explanation length={len(explanation)}, command length={len(command)}")
            return analysis, explanation, command
        except Exception as e:
            logger.error(f"Error extracting explanation and command: {str(e)}")
            return "Error extracting analysis", "Error extracting explanation", "execution = {'error': 'Error extracting command', 'status': 'failed'}"

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        # Check if tool_name contains error prefix (indicates normalization failed)
        if "No matched tool given: " in tool_name:
            # Extract the actual tool name from error message
            clean_tool_name = tool_name.split("No matched tool given: ")[-1].strip()
            error_msg = (
                f"Tool name normalization failed: '{clean_tool_name}' could not be matched to any available tool. "
                f"This may indicate a mismatch between the tool name returned by the planner and the available tools. "
                f"Please check that the tool is properly registered and available."
            )
            logger.error(error_msg)
            return {
                "error": error_msg,
                "summary": f"Failed to execute tool: {clean_tool_name} not found",
                "tool_name": clean_tool_name
            }
        
        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            # Use absolute path and include thread ID for thread safety in parallel execution
            import threading
            thread_id = threading.get_ident()
            output_file = os.path.join(self.tool_cache_dir, f"temp_output_{thread_id}_{uuid.uuid4().hex}.txt")
            try:
                # Ensure directory exists
                os.makedirs(os.path.dirname(output_file), exist_ok=True)
                
                # Write and read in separate with blocks to ensure file is properly closed
                with open(output_file, "w", encoding="utf-8") as f:
                    with redirect_stdout(f), redirect_stderr(f):
                        try:
                            # Use globals() like octotools_original to allow access to global namespace
                            exec(block, globals(), local_context)
                        except Exception as e:
                            # Store error in local_context so it can be retrieved
                            local_context["_execution_error"] = str(e)
                            raise e
                
                # Read output after file is closed from write
                output = ""
                if os.path.exists(output_file):
                    with open(output_file, "r", encoding="utf-8") as f:
                        output = f.read()
                
                # Check if execution variable exists, otherwise check for error
                if "execution" in local_context:
                    execution_result = local_context["execution"]
                    # Handle cached artifact case - return the cached result directly
                    if execution_result == 'cached_artifact':
                        # This should not happen here - cached results should be handled in app.py
                        # But if it does, return None to indicate it was cached
                        return None
                    return execution_result
                elif "_execution_error" in local_context:
                    return f"Error executing tool command: {local_context['_execution_error']}"
                else:
                    # If no execution variable and no error, return output or None
                    return output if output.strip() else None
            finally:
                # Clean up temp file - ensure it's closed before deletion
                if os.path.exists(output_file):
                    try:
                        # Small delay to ensure file handles are released
                        import time
                        time.sleep(0.01)
                        os.remove(output_file)
                    except (OSError, PermissionError) as e:
                        # File might be in use or already deleted, ignore
                        pass

        # Import the tool module and instantiate it (lazy loading)
        # Try to get tool class from initializer cache first if available
        tool_class = None
        if self.initializer and hasattr(self.initializer, '_tool_classes_cache'):
            tool_class = self.initializer._tool_classes_cache.get(tool_name)
        
        if tool_class is None:
            # Fallback: load tool class directly using Initializer's helper method
            if self.initializer:
                try:
                    tool_class = self.initializer._load_tool_class_only(tool_name)
                except Exception:
                    pass
            
            if tool_class is None:
                # Final fallback: manual loading
                tool_dir_parts = tool_name.replace('_Tool', '').split('_')
                tool_dir = '_'.join([p.lower() for p in tool_dir_parts])
                module_name = f"octotools.tools.{tool_dir}.tool"
                try:
                    module = importlib.import_module(module_name)
                    tool_class = getattr(module, tool_name)
                except (ImportError, AttributeError):
                    # Alternative path: try direct import
                    module_name = f"tools.{tool_dir}.tool"
                    module = importlib.import_module(module_name)
                    tool_class = getattr(module, tool_name)
        
        try:
            # Check if the tool requires an LLM engine or API key
            inputs = {}
            if getattr(tool_class, 'require_llm_engine', False):
                inputs['model_string'] = self.llm_engine_name
            if getattr(tool_class, 'require_api_key', False):
                inputs['api_key'] = self.api_key
            
            # Instantiate tool (lazy loading - only when actually called)
            tool = tool_class(**inputs)
            
            # Set the custom output directory
            if hasattr(tool, 'set_custom_output_dir'):
                tool.set_custom_output_dir(self.tool_cache_dir)
            
            # Create local context with tool object
            # Using globals() in exec() matches the approach in octotools_original
            local_context = {"tool": tool}
            
            # Preprocess command to fix common parameter mismatches
            # Some tools accept query_cache_dir but LLM may generate output_dir
            # Fix for Organoid_Segmenter_Tool, Cell_Segmenter_Tool, Nuclei_Segmenter_Tool
            if tool_name in ["Organoid_Segmenter_Tool", "Cell_Segmenter_Tool", "Nuclei_Segmenter_Tool"]:
                # Replace output_dir with query_cache_dir if present (handle both parameter and variable cases)
                if "output_dir" in command:
                    import re
                    # Replace output_dir= parameter with query_cache_dir=
                    command = re.sub(r'\boutput_dir\s*=', 'query_cache_dir=', command)
                    # Remove standalone output_dir variable references (if used as variable without definition)
                    # Pattern: output_dir) or output_dir, (but not output_dir=)
                    command = re.sub(r'\boutput_dir\s*([,)\)])', r'query_cache_dir\1', command)
                    logger.debug(f"Fixed parameter/variable name: replaced output_dir with query_cache_dir for {tool_name}")
            
            # Execute the entire command as a single block to preserve variable definitions
            result = execute_with_timeout(command, local_context)
            
            # Special handling for Cell_State_Analyzer_Tool to save h5ad file
            if tool_name == "Cell_State_Analyzer_Tool" and isinstance(result, dict):
                # Cell_State_Analyzer_Tool saves adata_path directly in result
                if 'adata_path' in result:
                    logger.debug(f"AnnData already saved by Cell_State_Analyzer_Tool: {result['adata_path']}")
            
            return result

        except TimeoutError:
            return f"Error: Tool execution timed out after {self.max_time} seconds."
        except Exception as e:
            return f"Error executing tool command: {e}"

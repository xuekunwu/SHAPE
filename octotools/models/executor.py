import os
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
            self.query_cache_dir = query_cache_dir
            # Also update tool_cache_dir to use the new query_cache_dir
            # If query_cache_dir already contains "tool_cache", use it directly
            # Otherwise, append "tool_cache" to it
            if "tool_cache" in query_cache_dir:
                self.tool_cache_dir = query_cache_dir
            else:
                self.tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.query_cache_dir, timestamp)
            self.tool_cache_dir = os.path.join(self.query_cache_dir, "tool_cache")
        # Ensure both directories exist
        os.makedirs(self.query_cache_dir, exist_ok=True)
        os.makedirs(self.tool_cache_dir, exist_ok=True)
        logger.debug(f"Executor: Updated query_cache_dir to {self.query_cache_dir}")
        logger.debug(f"Executor: Updated tool_cache_dir to {self.tool_cache_dir}")
    
    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], memory=None, bytes_mode:bool = False, conversation_context: str = "", image_id: str = None, **kwargs) -> ToolCommand:
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
                    previous_outputs = last_action['result']
                    # Create LLM-safe version (summary only, no file paths)
                    previous_outputs_for_llm = get_llm_safe_result(previous_outputs)
                    logger.debug(f"Extracted previous outputs: {list(previous_outputs.keys()) if isinstance(previous_outputs, dict) else 'Not a dict'}")
        
        # Special handling for Cell_State_Analyzer_Tool to use dynamic metadata file discovery
        state_analyzer_tools = ["Cell_State_Analyzer_Tool"]
        if tool_name in state_analyzer_tools:
            tool_label = "Cell_State_Analyzer_Tool"
            return ToolCommand(
                analysis=f"Using dynamic metadata file discovery for {tool_label}",
                explanation=f"Automatically finding the most recent metadata file and loading cell data with improved format handling for {tool_label}",
                command="""import json
import os
import glob

# Dynamically find the most recent metadata file
metadata_dir = 'solver_cache/temp/tool_cache'
metadata_files = glob.glob(os.path.join(metadata_dir, 'cell_crops_metadata_*.json'))
if not metadata_files:
    execution = {"error": "No metadata files found", "status": "failed"}
else:
    # Use the most recent metadata file
    latest_metadata_file = max(metadata_files, key=os.path.getctime)
    logger.info("Using metadata file: " + latest_metadata_file)
    
    try:
        # Use the tool's improved metadata loading method
        cell_crops, cell_metadata = tool._load_cell_data_from_metadata('solver_cache/temp/tool_cache')
        
        if cell_crops and len(cell_crops) > 0:
            # Execute the tool with loaded data
            execution = tool.execute(
                cell_crops=cell_crops, 
                cell_metadata=cell_metadata, 
                max_epochs=100,
                early_stop_loss=0.5,
                batch_size=16,
                learning_rate=3e-5,
                cluster_resolution=0.5,
                query_cache_dir='solver_cache/temp'
            )
        else:
            execution = {"error": "No valid cell crops found in metadata", "status": "failed"}
        
    except Exception as e:
        logger.error("Error loading metadata: " + str(e))
        execution = {"error": "Failed to load metadata: " + str(e), "status": "failed"}"""
            )
        
        # Special handling for segmentation tools (Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, Organoid_Segmenter_Tool) 
        # to use processed image from Image_Preprocessor_Tool
        segmentation_tools = ["Nuclei_Segmenter_Tool", "Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"]
        if tool_name in segmentation_tools and previous_outputs and 'processed_image_path' in previous_outputs:
            processed_image_path = previous_outputs['processed_image_path']
            # Normalize path for cross-platform compatibility
            import os
            processed_image_path = os.path.normpath(processed_image_path) if isinstance(processed_image_path, str) else str(processed_image_path)
            # Verify file exists before using it
            if not os.path.exists(processed_image_path):
                logger.warning(f"processed_image_path does not exist: {processed_image_path}")
                # Fall through to standard command generation
            else:
                # Escape backslashes for Python string in command
                safe_processed_path = processed_image_path.replace("\\", "\\\\")
                # Include image_id if available for consistent naming
                image_id_param = f', image_id="{image_id}"' if image_id else ''
                tool_label = "nuclei segmentation" if tool_name == "Nuclei_Segmenter_Tool" else \
                             "cell segmentation" if tool_name == "Cell_Segmenter_Tool" else \
                             "organoid segmentation"
                return ToolCommand(
                    analysis=f"Using the processed image from Image_Preprocessor_Tool for {tool_label}",
                    explanation=f"Using the processed image path '{processed_image_path}' from the previous Image_Preprocessor_Tool step",
                    command=f"""execution = tool.execute(image="{safe_processed_path}"{image_id_param})"""
                )
        
        # Special handling for Single_Cell_Cropper_Tool to use mask from segmentation tools
        # Supports masks from Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, and Organoid_Segmenter_Tool
        if tool_name == "Single_Cell_Cropper_Tool" and previous_outputs and 'visual_outputs' in previous_outputs:
            # Find mask file from previous outputs (support nuclei_mask, cell_mask, organoid_mask)
            mask_path = None
            mask_type = None
            logger.debug(f"Looking for mask in previous outputs: {previous_outputs['visual_outputs']}")
            for output_path in previous_outputs['visual_outputs']:
                logger.debug(f"Checking output path: {output_path}")
                # Check for various mask types
                if output_path.endswith('.png') and 'viz' not in output_path:
                    if 'nuclei_mask' in output_path:
                        mask_path = output_path
                        mask_type = "nuclei_mask"
                        logger.debug(f"Found nuclei mask path: {mask_path}")
                        break
                    elif 'cell_mask' in output_path:
                        mask_path = output_path
                        mask_type = "nuclei_mask"  # Use same parameter name for compatibility
                        logger.debug(f"Found cell mask path: {mask_path}")
                        break
                    elif 'organoid_mask' in output_path:
                        mask_path = output_path
                        mask_type = "nuclei_mask"  # Use same parameter name for compatibility
                        logger.debug(f"Found organoid mask path: {mask_path}")
                        break
            
            if mask_path:
                source_tool = "segmentation tool"
                if 'nuclei_mask' in mask_path:
                    source_tool = "Nuclei_Segmenter_Tool"
                elif 'cell_mask' in mask_path:
                    source_tool = "Cell_Segmenter_Tool"
                elif 'organoid_mask' in mask_path:
                    source_tool = "Organoid_Segmenter_Tool"
                return ToolCommand(
                    analysis=f"Using the mask from {source_tool} for single cell cropping",
                    explanation=f"Using the mask path '{mask_path}' from the previous {source_tool} step",
                    command=f"""execution = tool.execute(original_image="{actual_image_path}", nuclei_mask="{mask_path}", min_area=50, margin=25)"""
                )
            else:
                logger.debug(f"No mask found in previous outputs: {previous_outputs['visual_outputs']}")
                # Fallback to standard command generation
                pass
        
        # For other tools, use the standard prompt
        prompt_generate_tool_command = f"""
Task: Generate a precise command to execute the selected tool based on the given information.

Conversation so far:
{conversation_context}

Query: {question}
Image Path: {safe_path}
Image ID: {image_id if image_id else "Not provided"}
Context: {context}
Sub-Goal: {sub_goal}
Selected Tool: {tool_name}
Tool Metadata: {tool_metadata}
Previous Tool Outputs (summary only, file paths available for tool chaining): {previous_outputs_for_llm}

IMPORTANT: When the tool requires an image parameter, you MUST use the exact image path provided above: "{safe_path}"
{"IMPORTANT: Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, Organoid_Segmenter_Tool, and Image_Preprocessor_Tool accept the image_id parameter for consistent file naming and tracking. Include image_id parameter when available for these tools." if image_id else ""}

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
                    tool_command = ToolCommand(
                        analysis="Error parsing analysis",
                        explanation="Error parsing explanation",
                        command="execution = tool.execute(error='Error parsing command')"
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
            return ToolCommand(
                analysis=f"Error generating tool command: {str(e)}",
                explanation="Failed to generate proper command due to response format parsing error",
                command=f"execution = tool.execute(error='Command generation failed: {error_msg}')"
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
            return "Error extracting analysis", "Error extracting explanation", "execution = tool.execute(error='Error extracting command')"

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
            output_file = f"temp_output_{uuid.uuid4()}.txt"
            try:
                with open(output_file, "w", encoding="utf-8") as f, redirect_stdout(f), redirect_stderr(f):
                    try:
                        # Use globals() like octotools_original to allow access to global namespace
                        exec(block, globals(), local_context)
                    except Exception as e:
                        # Store error in local_context so it can be retrieved
                        local_context["_execution_error"] = str(e)
                        raise e
                with open(output_file, "r", encoding="utf-8") as f:
                    output = f.read()
                # Check if execution variable exists, otherwise check for error
                if "execution" in local_context:
                    return local_context["execution"]
                elif "_execution_error" in local_context:
                    return f"Error executing tool command: {local_context['_execution_error']}"
                else:
                    # If no execution variable and no error, return output or None
                    return output if output.strip() else None
            finally:
                # Clean up temp file
                if os.path.exists(output_file):
                    try:
                        os.remove(output_file)
                    except Exception:
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

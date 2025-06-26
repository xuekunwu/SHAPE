import os
# import sys
import importlib
import re
from typing import Dict, Any, List
from datetime import datetime

from octotools.engine.openai import ChatOpenAI 
from octotools.models.formatters import ToolCommand

import signal
from typing import Dict, Any, List, Optional
import uuid
from contextlib import redirect_stdout, redirect_stderr
import traceback

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
        else:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.query_cache_dir = os.path.join(self.query_cache_dir, timestamp)
        os.makedirs(self.query_cache_dir, exist_ok=True)
    
    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], memory=None, bytes_mode:bool = False) -> ToolCommand:
        actual_image_path = image if not bytes_mode else 'image.jpg'
        safe_path = actual_image_path.replace("\\", "\\\\") if actual_image_path else ""
        
        # Get previous tool outputs from memory to pass file paths correctly
        previous_outputs = {}
        if memory:
            actions = memory.get_actions()
            if actions:
                # Get the most recent action's result
                latest_action = actions[-1]
                if 'result' in latest_action:
                    previous_outputs = latest_action['result']
        
        # Special handling for Fibroblast_State_Analyzer_Tool to use dynamic metadata file discovery
        if tool_name == "Fibroblast_State_Analyzer_Tool":
            return ToolCommand(
                analysis="Using dynamic metadata file discovery for Fibroblast_State_Analyzer_Tool",
                explanation="Automatically finding the most recent metadata file and loading cell data with improved format handling",
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
    print("Using metadata file: " + latest_metadata_file)
    
    try:
        # Use the tool's improved metadata loading method
        cell_crops, cell_metadata = tool._load_cell_data_from_metadata('solver_cache/temp/tool_cache')
        
        if cell_crops and len(cell_crops) > 0:
            # Execute the tool with loaded data
            execution = tool.execute(
                cell_crops=cell_crops, 
                cell_metadata=cell_metadata, 
                confidence_threshold=0.5, 
                batch_size=16, 
                query_cache_dir='solver_cache/temp/tool_cache',
                visualization_type='all'
            )
            # 保存AnnData为h5ad文件，供下游激活评分工具使用
            # Note: This logic is now handled in execute_tool_command method
            # if hasattr(execution, 'adata'):
            #     adata_path = os.path.join('solver_cache/temp/tool_cache', 'fibroblast_state_analyzed.h5ad')
            #     execution['adata'].write_h5ad(adata_path)
            #     execution['analyzed_h5ad_path'] = adata_path
        else:
            execution = {"error": "No valid cell crops found in metadata", "status": "failed"}
        
    except Exception as e:
        print("Error loading metadata: " + str(e))
        execution = {"error": "Failed to load metadata: " + str(e), "status": "failed"}"""
            )
        
        # Special handling for Fibroblast_Activation_Scorer_Tool to use h5ad from state analyzer
        if tool_name == "Fibroblast_Activation_Scorer_Tool":
            return ToolCommand(
                analysis="Using AnnData h5ad from Fibroblast_State_Analyzer_Tool for activation scoring",
                explanation="Loading AnnData h5ad file from state analyzer output for activation scoring with reference data",
                command="""import os
# Find the analyzed h5ad file from state analyzer output
h5ad_path = os.path.join('solver_cache/temp/tool_cache', 'fibroblast_state_analyzed.h5ad')
if not os.path.exists(h5ad_path):
    execution = {"error": "No analyzed h5ad file found from state analyzer", "status": "failed"}
else:
    print(f'Using AnnData h5ad for activation scoring: {h5ad_path}')
    execution = tool.execute(
        cell_data=h5ad_path,
        reference_source='huggingface',
        reference_repo_id='5xuekun/adata_reference',
        reference_filename='adata_reference.h5ad',
        output_dir='output_visualizations',
        visualization_type='all',
        confidence_threshold=0.5,
        batch_size=100
    )"""
            )
        
        # Special handling for Nuclei_Segmenter_Tool to use processed image from Image_Preprocessor_Tool
        if tool_name == "Nuclei_Segmenter_Tool" and previous_outputs and 'processed_image_path' in previous_outputs:
            processed_image_path = previous_outputs['processed_image_path']
            return ToolCommand(
                analysis="Using the processed image from Image_Preprocessor_Tool for nuclei segmentation",
                explanation=f"Using the processed image path '{processed_image_path}' from the previous Image_Preprocessor_Tool step",
                command=f"""execution = tool.execute(image="{processed_image_path}")"""
            )
        
        # Special handling for Single_Cell_Cropper_Tool to use nuclei mask from Nuclei_Segmenter_Tool
        if tool_name == "Single_Cell_Cropper_Tool" and previous_outputs and 'visual_outputs' in previous_outputs:
            # Find nuclei mask file from previous outputs
            nuclei_mask_path = None
            for output_path in previous_outputs['visual_outputs']:
                if 'nuclei_mask' in output_path and output_path.endswith('.png'):
                    nuclei_mask_path = output_path
                    break
            
            if nuclei_mask_path:
                return ToolCommand(
                    analysis="Using the nuclei mask from Nuclei_Segmenter_Tool for single cell cropping",
                    explanation=f"Using the nuclei mask path '{nuclei_mask_path}' from the previous Nuclei_Segmenter_Tool step",
                    command=f"""execution = tool.execute(original_image="{actual_image_path}", nuclei_mask="{nuclei_mask_path}", min_area=50, margin=25)"""
                )
        
        # For other tools, use the standard prompt
        prompt_generate_tool_command = f"""
Task: Generate a precise command to execute the selected tool based on the given information.

Query: {question}
Image Path: {safe_path}
Context: {context}
Sub-Goal: {sub_goal}
Selected Tool: {tool_name}
Tool Metadata: {tool_metadata}
Previous Tool Outputs: {previous_outputs}

IMPORTANT: When the tool requires an image parameter, you MUST use the exact image path provided above: "{safe_path}"

Instructions:
1. Carefully review all provided information: the query, image path, context, sub-goal, selected tool, tool metadata, and previous tool outputs.
2. Analyze the tool's input_types from the metadata to understand required and optional parameters.
3. If previous tool outputs are available, use the appropriate file paths from those outputs (e.g., processed_image_path, nuclei_mask paths, etc.).
4. Construct a command or series of commands that aligns with the tool's usage pattern and addresses the sub-goal.
5. Ensure all required parameters are included and properly formatted.
6. Use appropriate values for parameters based on the given context, particularly the Context field which may contain relevant information from previous steps.
7. If multiple steps are needed to prepare data for the tool, include them in the command construction.
8. CRITICAL: If the tool requires an image parameter, use the exact image path "{safe_path}" provided above, unless a processed image path is available from previous outputs.

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
12. If previous tool outputs contain relevant file paths (e.g., processed_image_path, nuclei_mask paths), use those paths instead of the original image path when appropriate.

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

Remember: Your <command> field MUST be valid Python code including any necessary data preparation steps and one or more execution = tool.execute( calls, without any additional explanatory text. The format execution = tool.execute must be strictly followed, and the last line must begin with execution = tool.execute to capture the final output. ALWAYS use the actual image path "{safe_path}" when the tool requires an image parameter, unless a processed image path is available from previous outputs.
"""

        try:
            llm_generate_tool_command = ChatOpenAI(model_string=self.llm_engine_name, is_multimodal=False, api_key=self.api_key)
            llm_response = llm_generate_tool_command.generate(prompt_generate_tool_command, response_format=ToolCommand)
            
            # Extract content and usage from response
            if isinstance(llm_response, dict) and 'content' in llm_response:
                tool_command = llm_response['content']
                usage_info = llm_response.get('usage', {})
                print(f"Tool command generation usage: {usage_info}")
            else:
                tool_command = llm_response
                usage_info = {}
            
            # Check if we got a string response (non-structured model like gpt-4-turbo) instead of ToolCommand object
            if isinstance(tool_command, str):
                print("WARNING: Received string response instead of ToolCommand object")
                # Try to parse the string response to extract analysis, explanation, and command
                try:
                    lines = tool_command.split('\n')
                    analysis = ""
                    explanation = ""
                    command = ""
                    
                    for line in lines:
                        line = line.strip()
                        if line.lower().startswith('<analysis>') and not line.lower().startswith('<analysis>:'):
                            analysis = line.split('<analysis>')[1].split('</analysis>')[0].strip()
                        elif line.lower().startswith('analysis:'):
                            parts = line.split('analysis:', 1)
                            if len(parts) > 1:
                                analysis = parts[1].lstrip(' :')
                            else:
                                analysis = ""
                        elif line.lower().startswith('<explanation>') and not line.lower().startswith('<explanation>:'):
                            explanation = line.split('<explanation>')[1].split('</explanation>')[0].strip()
                        elif line.lower().startswith('explanation:'):
                            parts = line.split('explanation:', 1)
                            if len(parts) > 1:
                                explanation = parts[1].lstrip(' :')
                            else:
                                explanation = ""
                        elif line.lower().startswith('<command>') and not line.lower().startswith('<command>:'):
                            command = line.split('<command>')[1].split('</command>')[0].strip()
                        elif line.lower().startswith('command:'):
                            parts = line.split('command:', 1)
                            if len(parts) > 1:
                                command = parts[1].lstrip(' :')
                            else:
                                command = ""
                    
                    # If we couldn't parse properly, try alternative patterns
                    if not analysis or not explanation or not command:
                        for line in lines:
                            line = line.strip()
                            if line.lower().startswith('analysis:') and not analysis:
                                parts = line.split('analysis:', 1)
                                if len(parts) > 1:
                                    analysis = parts[1].lstrip(' :')
                            elif line.lower().startswith('explanation:') and not explanation:
                                parts = line.split('explanation:', 1)
                                if len(parts) > 1:
                                    explanation = parts[1].lstrip(' :')
                            elif line.lower().startswith('command:') and not command:
                                parts = line.split('command:', 1)
                                if len(parts) > 1:
                                    command = parts[1].lstrip(' :')
                    
                    # If still missing, use defaults
                    if not analysis:
                        analysis = "No analysis provided"
                    if not explanation:
                        explanation = "No explanation provided"
                    if not command:
                        command = "execution = tool.execute(error='No command provided')"
                    
                    # Create ToolCommand object manually
                    tool_command = ToolCommand(
                        analysis=analysis,
                        explanation=explanation,
                        command=command
                    )
                    print(f"Created ToolCommand object from string: analysis='{analysis[:50]}...', explanation='{explanation[:50]}...', command='{command[:50]}...'")
                    
                except Exception as parse_error:
                    print(f"Error parsing string response: {parse_error}")
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
            print(f"Error in tool command generation: {e}")
            # Fallback: create a basic ToolCommand with error information
            error_msg = str(e).replace("'", "\\'")  # Escape single quotes
            return ToolCommand(
                analysis=f"Error generating tool command: {str(e)}",
                explanation="Failed to generate proper command due to response format parsing error",
                command=f"execution = tool.execute(error='Command generation failed: {error_msg}')"
            )

    def extract_explanation_and_command(self, response) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove ```python at the beginning
            code = re.sub(r'^```python\s*', '', code)
            # Remove ``` at the end (handle both with and without newlines)
            code = re.sub(r'\s*```\s*$', '', code)
            return code.strip()
        
        try:
            # Check if response is a ToolCommand object
            if hasattr(response, 'analysis') and hasattr(response, 'explanation') and hasattr(response, 'command'):
                analysis = response.analysis.strip()
                explanation = response.explanation.strip()
                command = normalize_code(response.command.strip())
                return analysis, explanation, command
            # Check if response is a string (fallback for non-structured models like gpt-4-turbo)
            elif isinstance(response, str):
                print("WARNING: Received string response instead of ToolCommand object")
                # Try to parse the string response to extract analysis, explanation, and command
                try:
                    lines = response.split('\n')
                    analysis = ""
                    explanation = ""
                    command = ""
                    
                    for line in lines:
                        line = line.strip()
                        if line.lower().startswith('<analysis>') and not line.lower().startswith('<analysis>:'):
                            analysis = line.split('<analysis>')[1].split('</analysis>')[0].strip()
                        elif line.lower().startswith('analysis:'):
                            parts = line.split('analysis:', 1)
                            if len(parts) > 1:
                                analysis = parts[1].lstrip(' :')
                            else:
                                analysis = ""
                        elif line.lower().startswith('<explanation>') and not line.lower().startswith('<explanation>:'):
                            explanation = line.split('<explanation>')[1].split('</explanation>')[0].strip()
                        elif line.lower().startswith('explanation:'):
                            parts = line.split('explanation:', 1)
                            if len(parts) > 1:
                                explanation = parts[1].lstrip(' :')
                            else:
                                explanation = ""
                        elif line.lower().startswith('<command>') and not line.lower().startswith('<command>:'):
                            command = line.split('<command>')[1].split('</command>')[0].strip()
                        elif line.lower().startswith('command:'):
                            parts = line.split('command:', 1)
                            if len(parts) > 1:
                                command = parts[1].lstrip(' :')
                            else:
                                command = ""
                    
                    # If we couldn't parse properly, try alternative patterns
                    if not analysis or not explanation or not command:
                        for line in lines:
                            line = line.strip()
                            if line.lower().startswith('analysis:') and not analysis:
                                parts = line.split('analysis:', 1)
                                if len(parts) > 1:
                                    analysis = parts[1].lstrip(' :')
                            elif line.lower().startswith('explanation:') and not explanation:
                                parts = line.split('explanation:', 1)
                                if len(parts) > 1:
                                    explanation = parts[1].lstrip(' :')
                            elif line.lower().startswith('command:') and not command:
                                parts = line.split('command:', 1)
                                if len(parts) > 1:
                                    command = parts[1].lstrip(' :')
                    
                    # If still missing, use defaults
                    if not analysis:
                        analysis = "No analysis provided"
                    if not explanation:
                        explanation = "No explanation provided"
                    if not command:
                        command = "execution = tool.execute(error='No command provided')"
                    
                    # Normalize the command
                    command = normalize_code(command)
                    
                    print(f"Parsed from string: analysis='{analysis[:50]}...', explanation='{explanation[:50]}...', command='{command[:50]}...'")
                    return analysis, explanation, command
                    
                except Exception as parse_error:
                    print(f"Error parsing string response: {parse_error}")
                    return "Error parsing analysis", "Error parsing explanation", "execution = tool.execute(error='Error parsing command')"
            else:
                print(f"Unexpected response type: {type(response)}")
                return "Unknown analysis", "Unknown explanation", "execution = tool.execute(error='Unknown response type')"
                
        except Exception as e:
            print(f"Error extracting explanation and command: {str(e)}")
            return "Error extracting analysis", "Error extracting explanation", "execution = tool.execute(error='Error extracting command')"

    def execute_tool_command(self, tool_name: str, command: str) -> Any:
        def execute_with_timeout(block: str, local_context: dict) -> Optional[str]:
            output_file = f"temp_output_{uuid.uuid4()}.txt"
            with open(output_file, "w") as f, redirect_stdout(f), redirect_stderr(f):
                try:
                    exec(block, local_context)
                except Exception as e:
                    print(traceback.format_exc())
                    raise e
            with open(output_file, "r") as f:
                output = f.read()
            os.remove(output_file)
            return local_context.get("execution", output)

        # Import the tool module and instantiate it
        module_name = f"tools.{tool_name.lower().replace('_tool', '')}.tool"
        
        try:
            # Dynamically import the module
            module = importlib.import_module(module_name)

            # Get the tool class
            tool_class = getattr(module, tool_name)

            # Check if the tool requires an LLM engine or API key
            inputs = {}
            if getattr(tool_class, 'require_llm_engine', False):
                inputs['model_string'] = self.llm_engine_name
            if getattr(tool_class, 'require_api_key', False):
                inputs['api_key'] = self.api_key
            
            tool = tool_class(**inputs)
            
            # Set the custom output directory
            if hasattr(tool, 'set_custom_output_dir'):
                tool.set_custom_output_dir(self.tool_cache_dir)
            
            local_context = {"tool": tool}
            
            # Execute the entire command as a single block to preserve variable definitions
            result = execute_with_timeout(command, local_context)
            
            # Special handling for Fibroblast_State_Analyzer_Tool to save h5ad file
            if tool_name == "Fibroblast_State_Analyzer_Tool" and isinstance(result, dict) and 'adata' in result:
                try:
                    import anndata
                    adata_path = os.path.join(self.tool_cache_dir, 'fibroblast_state_analyzed.h5ad')
                    print(f"Saving AnnData to h5ad file: {adata_path}")
                    
                    # Ensure the directory exists
                    os.makedirs(os.path.dirname(adata_path), exist_ok=True)
                    
                    # Save the AnnData object
                    result['adata'].write_h5ad(adata_path)
                    result['analyzed_h5ad_path'] = adata_path
                    print(f"Successfully saved h5ad file: {adata_path}")
                    
                    # Verify the file was created
                    if os.path.exists(adata_path):
                        file_size = os.path.getsize(adata_path)
                        print(f"Verified h5ad file exists with size: {file_size} bytes")
                    else:
                        print(f"Warning: h5ad file was not created at {adata_path}")
                        
                except Exception as e:
                    print(f"Error saving h5ad file: {e}")
                    import traceback
                    traceback.print_exc()
                    # Don't fail the entire execution, just log the error
                    if 'analyzed_h5ad_path' not in result:
                        result['analyzed_h5ad_path'] = None
            
            return result

        except TimeoutError:
            return f"Error: Tool execution timed out after {self.max_time} seconds."
        except Exception as e:
            return f"Error executing tool command: {e}"
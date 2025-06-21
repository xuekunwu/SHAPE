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
import json

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
    
    def generate_tool_command(self, question: str, image: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], bytes_mode:bool = False) -> ToolCommand:
        actual_image_path = image if not bytes_mode else 'image.jpg'
        safe_path = actual_image_path.replace("\\", "\\\\") if actual_image_path else ""
        
        # Escape curly braces in tool_metadata to prevent f-string formatting errors
        safe_tool_metadata = str(tool_metadata).replace("{", "{{").replace("}", "}}")
        
        prompt_generate_tool_command = f"""
Task: Generate a precise command to execute the selected tool based on the given information.

Query: {question}
Image Path: {safe_path}
Context: {context}
Sub-Goal: {sub_goal}
Selected Tool: {tool_name}
Tool Metadata: {safe_tool_metadata}

IMPORTANT: When the tool requires an image parameter, you MUST use the exact image path provided above: "{safe_path}"

Instructions:
1. Carefully review all provided information: the query, image path, context, sub-goal, selected tool, and tool metadata.
2. Analyze the tool's input_types from the metadata to understand required and optional parameters.
3. Construct a command or series of commands that aligns with the tool's usage pattern and addresses the sub-goal.
4. Ensure all required parameters are included and properly formatted.
5. Use appropriate values for parameters based on the given context, particularly the `Context` field which may contain relevant information from previous steps.
6. If multiple steps are needed to prepare data for the tool, include them in the command construction.
7. CRITICAL: If the tool requires an image parameter, use the exact image path "{safe_path}" provided above.
8. CRITICAL: Include ALL necessary import statements (e.g., import json, import os, etc.) if the command requires them.

Output Format:
<analysis>: a step-by-step analysis of the context, sub-goal, and selected tool to guide the command construction.
<explanation>: a detailed explanation of the constructed command(s) and their parameters.
<command>: the Python code to execute the tool, which can be one of the following types:
    a. A single line command with `execution = tool.execute()`.
    b. A multi-line command with complex data preparation, ending with `execution = tool.execute()`.
    c. Multiple lines of `execution = tool.execute()` calls for processing multiple items.
```python
<your command here>
```

Rules:
1. The command MUST be valid Python code and include at least one call to `tool.execute()`.
2. Each `tool.execute()` call MUST be assigned to the 'execution' variable in the format `execution = tool.execute(...)`.
3. For multiple executions, use separate `execution = tool.execute()` calls for each execution.
4. The final output MUST be assigned to the 'execution' variable, either directly from `tool.execute()` or as a processed form of multiple executions.
5. Use the exact parameter names as specified in the tool's input_types.
6. Enclose string values in quotes, use appropriate data types for other values (e.g., lists, numbers).
7. Do not include any code or text that is not part of the actual command.
8. Ensure the command directly addresses the sub-goal and query.
9. Include ALL required parameters, data, and paths to execute the tool in the command itself.
10. If preparation steps are needed, include them as separate Python statements before the `tool.execute()` calls.
11. CRITICAL: If the tool requires an image parameter, use the exact image path "{safe_path}" provided above.
12. CRITICAL: Include ALL necessary import statements at the beginning of the command if needed (e.g., import json, import os, etc.).

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

Example 3 (Data loading with imports):
<analysis>: The tool requires data from a JSON file, so we need to import json and load the data.
<explanation>: We import json, load the metadata file, extract the required data, and pass it to the tool.
<command>:
```python
import json

# Load metadata from JSON file
with open('solver_cache/temp/tool_cache/cell_crops_metadata.json', 'r') as f:
    metadata = json.load(f)

# IMPORTANT: metadata is a list of dictionaries, not a dict with 'crops' key
# Each item in the list has: cell_id, crop_path, original_bbox, crop_bbox, area, centroid, crop_size
# Extract required data directly from the list
cell_crops = [item['crop_path'] for item in metadata]
cell_metadata = [{{'cell_id': item['cell_id']}} for item in metadata]

# Execute tool
execution = tool.execute(cell_crops=cell_crops, cell_metadata=cell_metadata)
```

Example 4 (Data loading with error handling):
<analysis>: The tool requires data from a JSON file with proper error handling.
<explanation>: We import json, load the metadata file with error handling, verify the structure, and extract data safely.
<command>:
```python
import json

# Load metadata from JSON file with error handling
try:
    with open('solver_cache/temp/tool_cache/cell_crops_metadata.json', 'r') as f:
        metadata = json.load(f)
    
    # Verify metadata is a list
    if not isinstance(metadata, list):
        raise ValueError("Metadata should be a list of cell data")
    
    # Extract required data safely
    cell_crops = []
    cell_metadata = []
    
    for item in metadata:
        if isinstance(item, dict) and 'crop_path' in item and 'cell_id' in item:
            cell_crops.append(item['crop_path'])
            cell_metadata.append({{'cell_id': item['cell_id']}})
    
    # Execute tool
    execution = tool.execute(cell_crops=cell_crops, cell_metadata=cell_metadata)
    
except Exception as e:
    print(f"Error loading metadata: {{e}}")
    execution = {{"error": f"Failed to load metadata: {{e}}", "status": "failed"}}
```

Example 5 (Image captioning with actual image path):
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
Reason: only `execution = tool.execute` is allowed, not `execution1` or `execution2`.

<command>:
```python
execution = tool.execute(image="path/to/image", labels=["baseball"])
```
Reason: Do not use placeholder paths like "path/to/image". Use the actual image path provided.

<command>:
```python
with open('data.json', 'r') as f:
    data = json.load(f)
execution = tool.execute(data=data)
```
Reason: Missing import json statement, which will cause "name 'json' is not defined" error.

<command>:
```python
import json
with open('metadata.json', 'r') as f:
    metadata = json.load(f)
cell_crops = [item['image_path'] for item in metadata['crops']]
```
Reason: Wrong structure assumption. metadata is a list, not a dict with 'crops' key. Use 'crop_path' not 'image_path'.

<command>:
```python
import json
with open('metadata.json', 'r') as f:
    metadata = json.load(f)
cell_crops = [item['crop_path'] for item in metadata['crops']]
```
Reason: Wrong structure assumption. metadata is a list, not a dict with 'crops' key. Access items directly: `[item['crop_path'] for item in metadata]`.

Remember: Your <command> field MUST be valid Python code including any necessary import statements, data preparation steps, and one or more `execution = tool.execute(` calls, without any additional explanatory text. The format `execution = tool.execute` must be strictly followed, and the last line must begin with `execution = tool.execute` to capture the final output. ALWAYS use the actual image path "{safe_path}" when the tool requires an image parameter. ALWAYS include necessary imports (e.g., import json) if your command requires them.
"""

        try:
            llm_generate_tool_command = ChatOpenAI(model_string=self.llm_engine_name, is_multimodal=False, api_key=self.api_key)
            tool_command = llm_generate_tool_command(prompt_generate_tool_command, response_format=ToolCommand)
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

    def extract_explanation_and_command(self, response: ToolCommand) -> tuple:
        def normalize_code(code: str) -> str:
            # Remove ```python at the beginning
            code = re.sub(r'^```python\s*', '', code)
            # Remove ``` at the end (handle both with and without newlines)
            code = re.sub(r'\s*```\s*$', '', code)
            return code.strip()
        
        analysis = response.analysis.strip()
        explanation = response.explanation.strip()
        command = normalize_code(response.command.strip())
        return analysis, explanation, command

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
            
            return result

        except TimeoutError:
            return f"Error: Tool execution timed out after {self.max_time} seconds."
        except Exception as e:
            return f"Error executing tool command: {e}"
import os
import importlib
import re
from typing import Dict, Any, List
from datetime import datetime

from octotools.engine.openai import ChatOpenAI 
from octotools.models.formatters import ToolCommand
from octotools.registry import REGISTRY
from octotools.models.tool_adapter import load_adapter, ToolAdapter

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
    
    def generate_tool_command(self, *args, **kwargs):  # Legacy signature retained to avoid breakage
        raise NotImplementedError("Deprecated: exec-based tool command generation removed.")

    def generate_tool_arguments(self, question: str, context: str, sub_goal: str, tool_name: str, tool_metadata: Dict[str, Any], adapter: ToolAdapter, memory=None, **kwargs) -> dict:
        """LLM produces JSON arguments only; validated against adapter input model."""
        schema_fields = getattr(adapter.input_model, "__fields__", {})
        prompt = f"""
Return ONLY JSON with arguments for tool {tool_name}.
Fields: {list(schema_fields.keys())}
Question: {question}
Context: {context}
Sub-goal: {sub_goal}
Tool metadata: {tool_metadata}
Previous steps: {memory.get_actions() if memory else []}
"""
        llm = ChatOpenAI(model_string=self.llm_engine_name, is_multimodal=False, api_key=self.api_key)
        response = llm.generate(prompt)
        if isinstance(response, dict) and 'content' in response:
            return response['content'] if isinstance(response['content'], dict) else {}
        if isinstance(response, dict):
            return response
        try:
            import json
            return json.loads(response) if isinstance(response, str) else {}
        except Exception:
            return {}

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

    def execute_tool(self, tool_name: str, args: dict) -> Any:
        """Execute tool via its adapter with validated args."""
        spec = REGISTRY.get(tool_name)
        if not spec:
            return {"error": f"Tool {tool_name} not registered"}
        try:
            adapter = load_adapter(spec.adapter_path)
        except Exception as e:
            return {"error": f"Failed to load adapter for {tool_name}: {e}"}

        # Validate args using adapter input model
        try:
            validated = adapter.input_model(**args) if hasattr(adapter, 'input_model') else args
        except Exception as e:
            return {"error": f"Invalid arguments for {tool_name}: {e}", "validation_errors": str(e)}

        try:
            result = adapter.run(validated, None)
        except TimeoutError:
            return {"error": f"Tool execution timed out after {self.max_time} seconds."}
        except Exception as e:
            return {"error": f"Error executing tool: {e}"}
        return result

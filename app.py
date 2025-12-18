import os
import sys
import json
import argparse
import time
import io
import uuid
import torch
import shutil
import logging
import tempfile
from PIL import Image
import numpy as np
from tifffile import imwrite as tiff_write
from typing import List, Dict, Any, Iterator, Optional
import matplotlib.pyplot as plt
import gradio as gr
from gradio import ChatMessage
from huggingface_hub import CommitScheduler
from pathlib import Path
import random
import traceback
import psutil  # For memory usage
from llm_evaluation_scripts.hf_model_configs import HF_MODEL_CONFIGS
from datetime import datetime
from octotools.models.utils import normalize_tool_name
from octotools.models.task_state import ConversationState, ActiveTask, TaskType, AnalysisSession, AnalysisInput, BatchImage, CellCrop
from dataclasses import dataclass, field
import importlib
from octotools.engine.openai import ChatOpenAI

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from octotools.models.initializer import Initializer

# Custom JSON encoder to handle ToolCommand objects
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        return super().default(obj)

def make_json_serializable(obj):
    """
    Recursively convert an object to be JSON serializable.
    Removes or converts non-serializable objects like AnnData.
    """
    if isinstance(obj, dict):
        return {key: make_json_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [make_json_serializable(item) for item in obj]
    elif hasattr(obj, '__class__') and 'AnnData' in obj.__class__.__name__:
        # Convert AnnData objects to a serializable representation
        return {
            "type": "AnnData",
            "shape": f"{obj.n_obs}x{obj.n_vars}",
            "obs_keys": list(obj.obs.keys()) if hasattr(obj, 'obs') else [],
            "var_keys": list(obj.var.keys()) if hasattr(obj, 'var') else [],
            "message": "AnnData object (removed for JSON serialization)"
        }
    elif hasattr(obj, '__dict__'):
        # For other objects, try to convert to dict
        try:
            return make_json_serializable(obj.__dict__)
        except:
            return str(obj)
    else:
        return obj

# Filter model configs to only include OpenAI models
def get_openai_model_configs():
    from llm_evaluation_scripts.hf_model_configs import HF_MODEL_CONFIGS
    return {k: v for k, v in HF_MODEL_CONFIGS.items() if v.get('model_type') == 'openai'}

OPENAI_MODEL_CONFIGS = get_openai_model_configs()

# In Gradio UI and all inference logic, only use OPENAI_MODEL_CONFIGS
# Example: model selection dropdown
# model_choices = list(OPENAI_MODEL_CONFIGS.keys())

# In the main inference logic, always use OpenAI API
# Remove any local/huggingface model inference branches

# Get Huggingface token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
IS_SPACES = os.getenv('SPACE_ID') is not None
DATASET_DIR = Path("solver_cache")  # the directory to save the dataset
DATASET_DIR.mkdir(parents=True, exist_ok=True) 
global QUERY_ID
QUERY_ID = None

# Comment out problematic CommitScheduler to avoid permission issues
# scheduler = CommitScheduler(
#     repo_id="lupantech/OctoTools-Gradio-Demo-User-Data",
#     repo_type="dataset",
#     folder_path=DATASET_DIR,
#     path_in_repo="solver_cache",  # Update path in repo
#     token=HF_TOKEN
# )

def save_query_data(query_id: str, query: str, image_path: str) -> None:
    """Save query data to Huggingface dataset"""
    # Save query metadata
    query_cache_dir = DATASET_DIR / query_id
    query_cache_dir.mkdir(parents=True, exist_ok=True)
    query_file = query_cache_dir / "query_metadata.json"

    query_metadata = {
        "query_id": query_id,
        "query_text": query,
        "datetime": time.strftime("%Y%m%d_%H%M%S"),
        "image_path": image_path if image_path else None
    }
    
    print(f"Saving query metadata to {query_file}")
    with query_file.open("w") as f:
        json.dump(query_metadata, f, indent=4)
    
    # # NOTE: As we are using the same name for the query cache directory as the dataset directory,
    # # NOTE: we don't need to copy the content from the query cache directory to the query directory.
    # # Copy all content from root_cache_dir to query_dir
    # import shutil
    # shutil.copytree(args.root_cache_dir, query_data_dir, dirs_exist_ok=True)


def save_feedback(query_id: str, feedback_type: str, feedback_text: str = None) -> None:
    """
    Save user feedback to the query directory.
    
    Args:
        query_id: Unique identifier for the query
        feedback_type: Type of feedback ('upvote', 'downvote', or 'comment')
        feedback_text: Optional text feedback from user
    """

    feedback_data_dir = DATASET_DIR / query_id
    feedback_data_dir.mkdir(parents=True, exist_ok=True)
    
    feedback_data = {
        "query_id": query_id,
        "feedback_type": feedback_type,
        "feedback_text": feedback_text,
        "datetime": time.strftime("%Y%m%d_%H%M%S")
    }
    
    # Save feedback in the query directory
    feedback_file = feedback_data_dir / "feedback.json"
    print(f"Saving feedback to {feedback_file}")
    
    # If feedback file exists, update it
    if feedback_file.exists():
        with feedback_file.open("r") as f:
            existing_feedback = json.load(f)
            # Convert to list if it's a single feedback entry
            if not isinstance(existing_feedback, list):
                existing_feedback = [existing_feedback]
            existing_feedback.append(feedback_data)
            feedback_data = existing_feedback
    
    # Write feedback data
    with feedback_file.open("w") as f:
        json.dump(feedback_data, f, indent=4)


def save_steps_data(query_id: str, memory) -> None:
    """Save steps data to Huggingface dataset"""
    steps_file = DATASET_DIR / query_id / "all_steps.json"

    memory_actions = memory.get_actions()
    memory_actions = make_json_serializable(memory_actions) # NOTE: make the memory actions serializable
    print("Memory actions: ", memory_actions)

    with steps_file.open("w") as f:
        json.dump(memory_actions, f, indent=4, cls=CustomEncoder)

    
def save_module_data(query_id: str, key: str, value: Any) -> None:
    """Save module data to Huggingface dataset"""
    try:
        key = key.replace(" ", "_").lower()
        module_file = DATASET_DIR / query_id / f"{key}.json"
        value = make_json_serializable(value)  # NOTE: make the value serializable
        with module_file.open("a") as f:
            json.dump(value, f, indent=4, cls=CustomEncoder)
    except Exception as e:
        print(f"Warning: Failed to save as JSON: {e}")
        # Fallback to saving as text file
        text_file = DATASET_DIR / query_id / f"{key}.txt"
        try:
            with text_file.open("a") as f:
                f.write(str(value) + "\n")
            print(f"Successfully saved as text file: {text_file}")
        except Exception as e:
            print(f"Error: Failed to save as text file: {e}")

########### End of Test Huggingface Dataset ###########


@dataclass
class AgentState(ConversationState):
    """Persistent session state to survive across turns."""
    last_context: str = ""
    last_sub_goal: str = ""
    last_visual_description: str = "*Ready to display analysis results and processed images.*"

def normalize_tool_name(tool_name: str, available_tools=None) -> str:
    """Normalize the tool name to match the available tools."""
    if available_tools is None:
        return tool_name
    for tool in available_tools:
        if tool.lower() in tool_name.lower():
            return tool
    return "No matched tool given: " + tool_name

class Solver:
    def __init__(
        self,
        planner,
        memory,
        executor,
        task: str,
        task_description: str,
        output_types: str = "base,final,direct",
        index: int = 0,
        verbose: bool = True,
        max_steps: int = 10,
        max_time: int = 60,
        query_cache_dir: str = "solver_cache",
        agent_state: AgentState = None,
        analysis_session: AnalysisSession = None
    ):
        self.planner = planner
        self.memory = memory
        self.executor = executor
        self.task = task
        self.task_description = task_description
        self.output_types = output_types
        self.index = index
        self.verbose = verbose
        self.max_steps = max_steps
        self.max_time = max_time
        self.query_cache_dir = query_cache_dir
        self.agent_state = agent_state or AgentState()
        self.analysis_session = analysis_session
        self.start_time = time.time()
        self.step_tokens = []
        # Initialize visual_outputs_for_gradio as instance variable to accumulate all visual outputs
        self.visual_outputs_for_gradio = []

        self.output_types = output_types.lower().split(',')
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."

        # Add statistics for evaluation
        self.step_times = []
        self.step_memory = []
        self.max_memory = 0
        self.step_costs = []
        self.total_cost = 0.0
        self.end_time = None
        
        # Add step information tracking
        self.step_info = []  # Store detailed information for each step
        self.model_config = self._get_model_config(planner.llm_engine_name)
        self.default_cost_per_token = self._get_default_cost_per_token()

    def _format_conversation_history(self) -> str:
        """Render conversation history into a plain-text transcript for prompts."""
        history = self.agent_state.conversation or []
        lines = []
        for msg in history:
            # ChatMessage stores role/content; be defensive about attributes
            role = getattr(msg, "role", "assistant")
            content = getattr(msg, "content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)

    def _get_model_config(self, model_id: str) -> dict:
        """Return the pricing config for the current model (cached on init)."""
        for config in OPENAI_MODEL_CONFIGS.values():
            if config.get("model_id") == model_id:
                return config
        return None

    def _get_default_cost_per_token(self) -> float:
        """Fallback pricing when model-specific costs are unavailable."""
        if self.model_config and 'expected_cost_per_1k_tokens' in self.model_config:
            return self.model_config['expected_cost_per_1k_tokens'] / 1000
        return 0.00001

    def _calculate_cost(self, input_tokens: int, output_tokens: int) -> float:
        """Calculate token cost using input/output pricing when available."""
        if self.model_config and 'input_cost_per_1k_tokens' in self.model_config and 'output_cost_per_1k_tokens' in self.model_config:
            input_cost = (input_tokens / 1000) * self.model_config['input_cost_per_1k_tokens']
            output_cost = (output_tokens / 1000) * self.model_config['output_cost_per_1k_tokens']
            return input_cost + output_cost
        total_tokens = input_tokens + output_tokens
        return total_tokens * self.default_cost_per_token

    def _collect_usage_and_cost(self, planner_usage=None, result=None):
        """
        Normalize usage stats from planner and executor outputs into input/output tokens
        plus a unified cost calculation.
        """
        planner_usage = planner_usage or {}
        input_tokens = planner_usage.get('prompt_tokens', 0)
        output_tokens = planner_usage.get('completion_tokens', 0)
        total_tokens = planner_usage.get('total_tokens', 0)

        if total_tokens and not (input_tokens or output_tokens):
            input_tokens = int(total_tokens * 0.7)
            output_tokens = total_tokens - input_tokens

        result_input = result_output = result_total = 0
        if isinstance(result, dict):
            if 'usage' in result:
                result_input = result['usage'].get('prompt_tokens', 0)
                result_output = result['usage'].get('completion_tokens', 0)
                result_total = result['usage'].get('total_tokens', result_input + result_output)
            elif 'token_usage' in result:
                result_total = result['token_usage']
                result_input = int(result_total * 0.7)
                result_output = result_total - result_input

        input_tokens += result_input
        output_tokens += result_output
        if input_tokens or output_tokens:
            total_tokens = input_tokens + output_tokens
        else:
            total_tokens += result_total

        cost = self._calculate_cost(input_tokens, output_tokens)
        return input_tokens, output_tokens, total_tokens, cost

    def stream_solve_user_problem(self, user_query: str, user_inputs: Optional[List[Dict[str, str]]], api_key: str, messages: List[ChatMessage]) -> Iterator:
        import time
        import os
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        visual_description = "*Ready to display analysis results and processed images.*"
        
        # Handle image inputs (multi-input aware)
        print(f"=== DEBUG: Image processing started ===")
        if user_inputs:
            print(f"DEBUG: user_inputs provided: {user_inputs}")
        else:
            print(f"DEBUG: user_inputs is None")

        if self.analysis_session is None:
            self.analysis_session = AnalysisSession()

        if user_inputs:
            for item in user_inputs:
                if not item or not item.get("path"):
                    continue
                name = (item.get("name") or "").strip()
                if not name:
                    messages.append(ChatMessage(role="assistant", content="Each uploaded image must have a non-empty name."))
                    yield messages, "", [], visual_description, "**Progress**: Error"
                    return
                self.analysis_session.inputs[name] = AnalysisInput(
                    name=name,
                    path=item["path"],
                    input_type=item.get("type", "image")
                )
            if not self.analysis_session.active_input and self.analysis_session.inputs:
                self.analysis_session.active_input = list(self.analysis_session.inputs.keys())[0]

        if not self.analysis_session.inputs:
            error_msg = "No images provided. Upload and name at least one image."
            messages.append(ChatMessage(role="assistant", content=error_msg))
            yield messages, "", [], visual_description, "**Progress**: Error"
            return

        active_input_name = self.analysis_session.active_input or list(self.analysis_session.inputs.keys())[0]
        img_path = self.analysis_session.inputs[active_input_name].path
        llm_img_path = normalize_image_for_llm(img_path, self.query_cache_dir)

        print(f"DEBUG: active_input={active_input_name}, img_path={img_path}")
        print(f"=== DEBUG: Image processing completed ===")

        # Set tool cache directory
        _tool_cache_dir = os.path.join(self.query_cache_dir, "tool_cache") # NOTE: This is the directory for tool cache
        self.executor.set_query_cache_dir(_tool_cache_dir) # NOTE: set query cache directory
        
        # Step 1: Display the received inputs
        messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}\n### 🖼️ Active Input: {active_input_name}"))
        yield messages, "", [], visual_description, "**Progress**: Input received"

        # [Step 3] Initialize problem-solving state
        step_count = 0
        json_data = {"query": user_query, "image": "Image received as bytes"}

        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### 🐙 Deep Thinking:"))
        yield messages, "", [], visual_description, "**Progress**: Starting analysis"

        # [Step 4] Query Analysis - This is the key step that should happen first
        print(f"Debug - Starting query analysis for: {user_query}")
        print(f"Debug - img_path for query analysis: {img_path} (LLM path: {llm_img_path})")
        query_analysis_start = time.time()
        try:
            conversation_text = self._format_conversation_history()
            query_analysis = self.planner.analyze_query(user_query, llm_img_path, conversation_text)
            query_analysis_end = time.time()
            query_analysis_time = query_analysis_end - query_analysis_start
            print(f"Debug - Query analysis completed: {len(query_analysis)} characters")
            
            # Track tokens for query analysis step
            planner_usage = self.planner.last_usage if hasattr(self.planner, 'last_usage') else None
            qa_input_tokens, qa_output_tokens, query_analysis_tokens, query_analysis_cost = self._collect_usage_and_cost(planner_usage)
            self.step_tokens.append(query_analysis_tokens)
            self.step_costs.append(query_analysis_cost)
            self.total_cost += query_analysis_cost

            if query_analysis_tokens:
                print(f"Query analysis - Input tokens: {qa_input_tokens}, Output tokens: {qa_output_tokens}")
                print(f"Query analysis cost: ${query_analysis_cost:.6f}")
            
            # Track time for query analysis step
            self.step_times.append(query_analysis_time)
            print(f"Query analysis time: {query_analysis_time:.2f}s")
            
            # Track memory for query analysis step
            mem_after_query_analysis = process.memory_info().rss / 1024 / 1024  # MB
            self.step_memory.append(mem_after_query_analysis)
            if mem_after_query_analysis > self.max_memory:
                self.max_memory = mem_after_query_analysis
            print(f"Query analysis memory usage: {mem_after_query_analysis:.2f} MB")
            
            # Record step information for query analysis
            step_info = {
                "step_number": 0,
                "step_type": "Query Analysis",
                "tool_name": "Query Analyzer",
                "description": "Analyze user query and determine required skills and tools",
                "time": query_analysis_time,
                "tokens": query_analysis_tokens,
                "cost": query_analysis_cost,
                "memory": mem_after_query_analysis,
                "input_tokens": qa_input_tokens,
                "output_tokens": qa_output_tokens
            }
            self.step_info.append(step_info)
            
            json_data["query_analysis"] = query_analysis
            query_analysis = query_analysis.replace("Concise Summary:", "**Concise Summary:**\n")
            query_analysis = query_analysis.replace("Required Skills:", "**Required Skills:**")
            query_analysis = query_analysis.replace("Relevant Tools:", "**Relevant Tools:**")
            query_analysis = query_analysis.replace("Additional Considerations:", "**Additional Considerations:**")
            messages.append(ChatMessage(role="assistant", 
                                        content=f"{query_analysis}",
                                        metadata={"title": "### 🔍 Step 0: Query Analysis"}))
            yield messages, query_analysis, [], visual_description, "**Progress**: Query analysis completed"

            # Save the query analysis data
            query_analysis_data = {"query_analysis": query_analysis, "time": round(time.time() - self.start_time, 5)}
            save_module_data(QUERY_ID, "step_0_query_analysis", query_analysis_data)
        except Exception as e:
            print(f"Error in query analysis: {e}")
            error_msg = f"⚠️ Error during query analysis: {str(e)}"
            messages.append(ChatMessage(role="assistant", 
                                        content=error_msg,
                                        metadata={"title": "### 🔍 Step 0: Query Analysis (Error)"}))
            yield messages, error_msg, [], visual_description, "**Progress**: Error in query analysis"
            return

        # Execution loop (similar to your step-by-step solver)
        while step_count < self.max_steps and (time.time() - self.start_time) < self.max_time:
            step_count += 1
            step_start = time.time()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            messages.append(ChatMessage(role="OctoTools", 
                                        content=f"Generating the {step_count}-th step...",
                                        metadata={"title": f"🔄 Step {step_count}"}))
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count}"

            # [Step 5] Generate the next step
            conversation_text = self._format_conversation_history()
            next_step = self.planner.generate_next_step(user_query, llm_img_path, query_analysis, self.memory, step_count, self.max_steps, conversation_context=conversation_text)
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
            context = context or self.agent_state.last_context or ""
            sub_goal = sub_goal or self.agent_state.last_sub_goal or ""
            step_data = {"step_count": step_count, "context": context, "sub_goal": sub_goal, "tool_name": tool_name, "time": round(time.time() - self.start_time, 5)}
            save_module_data(QUERY_ID, f"step_{step_count}_action_prediction", step_data)

            # Always normalize tool_name before use
            if hasattr(self.planner, 'available_tools'):
                tool_name = normalize_tool_name(tool_name, self.planner.available_tools)

            # Display the step information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Context:** {context}\n\n**Sub-goal:** {sub_goal}\n\n**Tool:** `{tool_name}`",
                metadata={"title": f"### 🎯 Step {step_count}: Action Prediction ({tool_name})"}))
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Action predicted"

            # Handle tool execution or errors
            if tool_name not in self.planner.available_tools:
                messages.append(ChatMessage(
                    role="assistant", 
                    content=f"⚠️ Error: Tool '{tool_name}' is not available."))
                yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Tool not available"
                continue

            # [Step 6-7] Generate and execute the tool command
            safe_path = img_path.replace("\\", "\\\\") if img_path else None
            conversation_text = self._format_conversation_history()
            tool_command = self.executor.generate_tool_command(user_query, safe_path, context, sub_goal, tool_name, self.planner.toolbox_metadata[tool_name], self.memory, conversation_context=conversation_text)
            analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
            result = self.executor.execute_tool_command(tool_name, command)
            result = make_json_serializable(result)
            print(f"Tool '{tool_name}' result:", result)

            # Persist result per input for later comparison
            if self.analysis_session and self.analysis_session.active_input:
                self.executor.record_result(self.analysis_session, self.analysis_session.active_input, f"step_{step_count}", result)
            
            # Generate dynamic visual description based on tool and results
            visual_description = self.generate_visual_description(tool_name, result, self.visual_outputs_for_gradio)
            
            if isinstance(result, dict):
                if "visual_outputs" in result:
                    visual_output_files = result["visual_outputs"]
                    # Append new visual outputs instead of reinitializing
                    for file_path in visual_output_files:
                        try:
                            # Skip comparison plots and non-image files
                            if "comparison" in os.path.basename(file_path).lower():
                                continue
                            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                                continue
                                
                            # Check if file exists and is readable
                            if not os.path.exists(file_path):
                                print(f"Warning: Image file not found: {file_path}")
                                continue
                            
                            # Check file size
                            if os.path.getsize(file_path) == 0:
                                print(f"Warning: Image file is empty: {file_path}")
                                continue
                                
                            # Use (image, label) tuple format to preserve filename for download
                            image = Image.open(file_path)
                            
                            # Validate image data
                            if image.size[0] == 0 or image.size[1] == 0:
                                print(f"Warning: Invalid image size: {file_path}")
                                continue
                            
                            # Convert to RGB if necessary for Gradio compatibility
                            if image.mode not in ['RGB', 'L', 'RGBA']:
                                try:
                                    image = image.convert('RGB')
                                except Exception as e:
                                    print(f"Warning: Failed to convert image {file_path} to RGB: {e}")
                                    continue
                            
                            # Additional validation for image data
                            try:
                                # Test if image can be converted to array
                                img_array = np.array(image)
                                if img_array.size == 0 or np.isnan(img_array).any():
                                    print(f"Warning: Invalid image data in {file_path}")
                                    continue
                            except Exception as e:
                                print(f"Warning: Failed to validate image data for {file_path}: {e}")
                                continue
                            
                            filename = os.path.basename(file_path)
                            
                            # Create descriptive label based on filename
                            if "processed" in filename.lower():
                                label = f"Processed Image: {filename}"
                            elif "corrected" in filename.lower():
                                label = f"Illumination Corrected: {filename}"
                            elif "segmented" in filename.lower():
                                label = f"Segmented Result: {filename}"
                            elif "detected" in filename.lower():
                                label = f"Detection Result: {filename}"
                            elif "zoomed" in filename.lower():
                                label = f"Zoomed Region: {filename}"
                            elif "crop" in filename.lower():
                                label = f"Single Cell Crop: {filename}"
                            else:
                                label = f"Analysis Result: {filename}"
                            
                            self.visual_outputs_for_gradio.append((image, label))
                            print(f"Successfully loaded image for Gradio: {filename}")
                            
                        except Exception as e:
                            print(f"Warning: Failed to load image {file_path} for Gradio. Error: {e}")
                            import traceback
                            print(f"Full traceback: {traceback.format_exc()}")
                            continue

            # Display the command generation information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Analysis:** {analysis}\n\n**Explanation:** {explanation}\n\n**Command:**\n```python\n{command}\n```",
                metadata={"title": f"### 📝 Step {step_count}: Command Generation ({tool_name})"}))
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Command generated"

            # Save the command generation data
            command_generation_data = {
                "analysis": analysis,
                "explanation": explanation,
                "command": command,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_generation", command_generation_data)
            
            # Display the command execution result
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Result:**\n```json\n{json.dumps(make_json_serializable(result), indent=4)}\n```",
                metadata={"title": f"### 🛠️ Step {step_count}: Command Execution ({tool_name})"}))
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Command executed"

            # If comparisons are requested and multiple inputs exist, generate a lightweight comparison summary
            if self.analysis_session and self.analysis_session.compare_requested and len(self.analysis_session.inputs) > 1:
                comparison_summary = self.executor.compare_results(self.analysis_session)
                if comparison_summary:
                    messages.append(ChatMessage(
                        role="assistant",
                        content=f"**Comparison across inputs:** {comparison_summary}",
                        metadata={"title": f"### 📊 Step {step_count}: Comparative Analysis"}))
                    yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Comparison updated"

            # Save the command execution data
            command_execution_data = {
                "result": result,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_execution", command_execution_data)

            # [Step 8] Memory update and stopping condition
            self.memory.add_action(step_count, tool_name, sub_goal, tool_command, result)
            conversation_text = self._format_conversation_history()
            stop_verification = self.planner.verificate_memory(user_query, llm_img_path, query_analysis, self.memory, conversation_context=conversation_text)
            context_verification, conclusion = self.planner.extract_conclusion(stop_verification)

            # Save the context verification data
            context_verification_data = {
                "stop_verification": context_verification,
                "conclusion": conclusion,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_context_verification", context_verification_data)    

            # Display the context verification result
            conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
            messages.append(ChatMessage(
                role="assistant", 
                content=f"**Analysis:**\n{context_verification}\n\n**Conclusion:** `{conclusion}` {conclusion_emoji}",
                metadata={"title": f"### 🤖 Step {step_count}: Context Verification"}))
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Context verified"

            # After tool execution, estimate tokens and cost
            planner_usage = self.planner.last_usage if hasattr(self.planner, 'last_usage') else None
            input_tokens, output_tokens, tokens_used, cost = self._collect_usage_and_cost(planner_usage, result)

            print(f"Step {step_count} - Input tokens: {input_tokens}, Output tokens: {output_tokens}")
            print(f"Step {step_count} - Total tokens: {tokens_used}, Cost: ${cost:.6f}")
            
            self.step_tokens.append(tokens_used)
            self.step_costs.append(cost)
            self.total_cost += cost
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            self.step_memory.append(mem_after)
            if mem_after > self.max_memory:
                self.max_memory = mem_after
            step_end = time.time()
            self.step_times.append(step_end - step_start)

            # Record step information for tool execution
            context_text = context or self.agent_state.last_context or ""
            sub_goal_text = sub_goal or self.agent_state.last_sub_goal or ""
            self.agent_state.last_context = context_text
            self.agent_state.last_sub_goal = sub_goal_text
            step_info = {
                "step_number": step_count,
                "step_type": "Tool Execution",
                "tool_name": tool_name,
                "description": f"Execute {tool_name} with sub-goal: {sub_goal_text[:100]}...",
                "time": step_end - step_start,
                "tokens": tokens_used,
                "cost": cost,
                "memory": mem_after,
                "input_tokens": input_tokens,
                "output_tokens": output_tokens,
                "context": context_text[:200] + "..." if len(context_text) > 200 else context_text,
                "sub_goal": sub_goal_text[:200] + "..." if len(sub_goal_text) > 200 else sub_goal_text
            }
            self.step_info.append(step_info)

            if conclusion == 'STOP':
                break

        self.end_time = time.time()

        # Step 7: Generate Final Output (if needed)
        final_answer = ""
        if 'direct' in self.output_types:
            messages.append(ChatMessage(role="assistant", content="<br>"))
            final_output_start = time.time()
            conversation_text = self._format_conversation_history()
            direct_output = self.planner.generate_direct_output(user_query, llm_img_path, self.memory, conversation_context=conversation_text)
            final_output_end = time.time()
            final_output_time = final_output_end - final_output_start
            
            # Track tokens for final output generation
            planner_usage = self.planner.last_usage if hasattr(self.planner, 'last_usage') else None
            direct_input_tokens, direct_output_tokens, final_output_tokens, final_output_cost = self._collect_usage_and_cost(planner_usage)
            self.step_tokens.append(final_output_tokens)
            self.step_costs.append(final_output_cost)
            self.total_cost += final_output_cost

            if final_output_tokens:
                print(f"Final output - Input tokens: {direct_input_tokens}, Output tokens: {direct_output_tokens}")
                print(f"Final output cost: ${final_output_cost:.6f}")
            
            # Track time for final output generation
            self.step_times.append(final_output_time)
            print(f"Final output time: {final_output_time:.2f}s")
            
            # Track memory for final output generation
            mem_after_final_output = process.memory_info().rss / 1024 / 1024  # MB
            self.step_memory.append(mem_after_final_output)
            if mem_after_final_output > self.max_memory:
                self.max_memory = mem_after_final_output
            print(f"Final output memory usage: {mem_after_final_output:.2f} MB")
            
            # Record step information for final output generation
            final_step_info = {
                "step_number": len(self.step_info),
                "step_type": "Final Output Generation",
                "tool_name": "Direct Output Generator",
                "description": "Generate final comprehensive answer based on all previous steps",
                "time": final_output_time,
                "tokens": final_output_tokens,
                "cost": final_output_cost,
                "memory": mem_after_final_output,
                "input_tokens": direct_input_tokens,
                "output_tokens": direct_output_tokens
            }
            self.step_info.append(final_step_info)
            
            # Extract conclusion from the final answer
            if isinstance(direct_output, str):
                conclusion_text = direct_output.strip()
            elif isinstance(direct_output, dict):
                # 你可以自定义错误信息或提取dict中的message
                conclusion_text = str(direct_output)
            else:
                conclusion_text = str(direct_output)
            
            # Step-by-step breakdown
            conclusion = f"🐙 **Conclusion:**\n{conclusion_text}\n\n---\n"
            conclusion += f"**📊 Detailed Performance Statistics**\n\n"
            
            # Step-by-step breakdown
            conclusion += f"**Step-by-Step Analysis:**\n"
            
            # Display detailed step information
            for i, step in enumerate(self.step_info):
                conclusion += f"**Step {step['step_number']}: {step['step_type']}**\n"
                conclusion += f"  • Tool: {step['tool_name']}\n"
                conclusion += f"  • Description: {step['description']}\n"
                conclusion += f"  • Time: {step['time']:.2f}s\n"
                conclusion += f"  • Tokens: {step['tokens']} (Input: {step['input_tokens']}, Output: {step['output_tokens']})\n"
                conclusion += f"  • Cost: ${step['cost']:.6f}\n"
                conclusion += f"  • Memory: {step['memory']:.2f} MB\n"
                
                # Add context and sub-goal for tool execution steps
                if step['step_type'] == "Tool Execution" and 'context' in step:
                    conclusion += f"  • Context: {step['context']}\n"
                    if 'sub_goal' in step and step['sub_goal'] != "":
                        conclusion += f"  • Sub-goal: {step['sub_goal']}\n"
                
                conclusion += "\n"
            
            # Summary statistics
            total_tokens_used = sum(self.step_tokens)
            conclusion += f"**📈 Summary Statistics:**\n"
            conclusion += f"  • Total Steps: {len(self.step_times)}\n"
            conclusion += f"  • Total Time: {self.end_time - self.start_time:.2f}s\n"
            conclusion += f"  • Total Tokens: {total_tokens_used}\n"
            conclusion += f"  • Total Cost: ${self.total_cost:.6f}\n"
            conclusion += f"  • Peak Memory: {self.max_memory:.2f} MB\n"
            avg_time_per_step = (self.end_time - self.start_time) / len(self.step_times) if self.step_times else 0
            conclusion += f"  • Average Time per Step: {avg_time_per_step:.2f}s\n"
            if total_tokens_used > 0:
                conclusion += f"  • Average Tokens per Step: {total_tokens_used / len(self.step_tokens):.1f}\n"
                conclusion += f"  • Cost per Token: ${self.total_cost / total_tokens_used:.8f}\n"
            
            # Raw data for reference
            conclusion += f"\n**📋 Raw Data (for reference):**\n"
            conclusion += f"  • Step Times: {[f'{t:.2f}s' for t in self.step_times]}\n"
            conclusion += f"  • Step Tokens: {self.step_tokens}\n"
            conclusion += f"  • Step Costs: {[f'${c:.6f}' for c in self.step_costs]}\n"
            conclusion += f"  • Step Memory: {[f'{m:.2f}MB' for m in self.step_memory]}\n"
            
            # Add model-specific pricing information
            model_config = None
            for config in OPENAI_MODEL_CONFIGS.values():
                if config.get("model_id") == self.planner.llm_engine_name:
                    model_config = config
                    break
            
            if model_config and 'input_cost_per_1k_tokens' in model_config:
                conclusion += f"\n**🔧 Model Configuration:**\n"
                conclusion += f"  • Model: {self.planner.llm_engine_name}\n"
                conclusion += f"  • Input Cost: ${model_config['input_cost_per_1k_tokens']:.6f} per 1K tokens\n"
                conclusion += f"  • Output Cost: ${model_config['output_cost_per_1k_tokens']:.6f} per 1K tokens\n"
                conclusion += f"  • Pricing Source: OpenAI Official Pricing (2024)\n"
            
            final_answer = f"{conclusion}"
            yield messages, final_answer, self.visual_outputs_for_gradio, visual_description, "**Progress**: Completed!"

            # Save the direct output data
            direct_output_data = {
                "direct_output": direct_output,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, "direct_output", direct_output_data)

        if 'final' in self.output_types:
            final_output_start = time.time()
            conversation_text = self._format_conversation_history()
            final_output = self.planner.generate_final_output(user_query, img_path, self.memory, conversation_context=conversation_text) # Disabled visibility for now
            final_output_end = time.time()
            final_output_time = final_output_end - final_output_start
            # messages.append(ChatMessage(role="assistant", content=f"🎯 Final Output:\n{final_output}"))
            # yield messages

            planner_usage = self.planner.last_usage if hasattr(self.planner, 'last_usage') else None
            final_input_tokens, final_output_tokens, final_total_tokens, final_output_cost = self._collect_usage_and_cost(planner_usage)
            self.step_tokens.append(final_total_tokens)
            self.step_costs.append(final_output_cost)
            self.total_cost += final_output_cost

            if final_total_tokens:
                print(f"Final output - Input tokens: {final_input_tokens}, Output tokens: {final_output_tokens}")
                print(f"Final output cost: ${final_output_cost:.6f}")

            # Track time for final output generation
            self.step_times.append(final_output_time)
            print(f"Final output time: {final_output_time:.2f}s")
            mem_after_final_generation = process.memory_info().rss / 1024 / 1024  # MB
            self.step_memory.append(mem_after_final_generation)
            if mem_after_final_generation > self.max_memory:
                self.max_memory = mem_after_final_generation

            final_step_info = {
                "step_number": len(self.step_info),
                "step_type": "Final Output Generation",
                "tool_name": "Final Output Generator",
                "description": "Generate final follow-up answer",
                "time": final_output_time,
                "tokens": final_total_tokens,
                "cost": final_output_cost,
                "memory": mem_after_final_generation,
                "input_tokens": final_input_tokens,
                "output_tokens": final_output_tokens
            }
            self.step_info.append(final_step_info)

            # Save the final output data
            final_output_data = {
                "final_output": final_output,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, "final_output", final_output_data)

        # Step 8: Completion Message
        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### ✅ Query Solved!"))
        # Use the final answer if available, otherwise use a default message
        completion_text = final_answer if final_answer else "Analysis completed successfully"
        yield messages, completion_text, self.visual_outputs_for_gradio, visual_description, "**Progress**: Analysis completed!"

    def generate_visual_description(self, tool_name: str, result: dict, visual_outputs: list) -> str:
        """
        Generate dynamic visual description based on tool type and results.
        """
        if not visual_outputs:
            return "*Ready to display analysis results and processed images.*"
        
        # Count different types of images with a single pass
        counts = {
            "processed": 0,
            "corrected": 0,
            "segmented": 0,
            "detected": 0,
            "zoomed": 0,
            "cropped": 0,
            "analyzed": 0
        }
        for _, label in visual_outputs:
            lower_label = str(label).lower()
            if "processed" in lower_label:
                counts["processed"] += 1
            if "corrected" in lower_label:
                counts["corrected"] += 1
            if "segmented" in lower_label:
                counts["segmented"] += 1
            if "detected" in lower_label:
                counts["detected"] += 1
            if "zoomed" in lower_label:
                counts["zoomed"] += 1
            if "crop" in lower_label:
                counts["cropped"] += 1
            if "analysis" in lower_label or "distribution" in lower_label:
                counts["analyzed"] += 1
        
        # Generate tool-specific descriptions
        tool_descriptions = {
            "Image_Preprocessor_Tool": f"*Displaying {counts['processed']} processed image(s) from illumination correction and brightness adjustment.*",
            "Object_Detector_Tool": f"*Showing {counts['detected']} detection result(s) with identified objects and regions of interest.*",
            "Image_Captioner_Tool": "*Displaying image analysis results with detailed morphological descriptions.*",
            "Relevant_Patch_Zoomer_Tool": f"*Showing {counts['zoomed']} zoomed region(s) highlighting key areas of interest.*",
            "Advanced_Object_Detector_Tool": f"*Displaying {counts['detected']} advanced detection result(s) with enhanced object identification.*",
            "Nuclei_Segmenter_Tool": f"*Showing {counts['segmented']} segmentation result(s) with identified nuclei regions.*",
            "Single_Cell_Cropper_Tool": f"*Displaying {counts['cropped']} single-cell crop(s) generated from nuclei segmentation results.*",
            "Cell_Morphology_Analyzer_Tool": "*Displaying cell morphology analysis results with detailed structural insights.*",
            "Fibroblast_Activation_Detector_Tool": "*Showing fibroblast activation state analysis with morphological indicators.*",
            "Fibroblast_State_Analyzer_Tool": f"*Displaying {counts['analyzed']} fibroblast state analysis result(s) with cell state distributions and statistics.*"
        }
        
        # Return tool-specific description or generic one
        if tool_name in tool_descriptions:
            return tool_descriptions[tool_name]
        else:
            total_images = len(visual_outputs)
            return f"*Displaying {total_images} analysis result(s) from {tool_name}.*"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the OctoTools demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--task", default="minitoolbench", help="Task to run.")
    parser.add_argument("--task_description", default="", help="Task description.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Generalist_Solution_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--query_id", default=None, help="Query ID.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")

    # NOTE: Add new arguments
    parser.add_argument("--run_baseline_only", type=bool, default=False, help="Run only the baseline (no toolbox).")
    parser.add_argument("--openai_api_source", default="we_provided", choices=["we_provided", "user_provided"], help="Source of OpenAI API key.")
    return parser.parse_args()


def build_image_table(files):
    rows = []
    if not files:
        return rows
    for f in files:
        path = getattr(f, "name", None) or getattr(f, "path", None) or str(f)
        rows.append([Path(path).stem])
    return rows


def normalize_image_for_llm(image_path: str, cache_dir: str) -> str:
    """
    Convert TIFF/TIF images to PNG for LLM compatibility while keeping
    the original path for downstream analysis.
    """
    if not image_path:
        return image_path
    ext = Path(image_path).suffix.lower()
    if ext not in [".tif", ".tiff"]:
        return image_path

    try:
        if Image is None:
            raise ImportError("Pillow (PIL) is required for TIFF normalization")
        os.makedirs(cache_dir, exist_ok=True)
        with Image.open(image_path) as img:
            # Always convert to RGB to avoid mode issues
            converted = img.convert("RGB")
            out_path = os.path.join(cache_dir, f"llm_normalized_{uuid.uuid4().hex}.png")
            converted.save(out_path, format="PNG")
            print(f"Normalized TIFF for LLM: {image_path} -> {out_path}")
            return out_path
    except Exception as e:
        print(f"Warning: failed to normalize TIFF for LLM ({image_path}): {e}")
        # Fall back to original; downstream may fail but avoid crash
        return image_path


def make_batch_image(image_path: str, group: str) -> BatchImage:
    """Create a BatchImage with provenance fields populated."""
    return BatchImage(
        group=group,
        image_id=str(uuid.uuid4()),
        image_path=image_path,
        image_name=os.path.basename(image_path)
    )


class BatchPipelineRunner:
    """Deterministic, group-aware batch pipeline (preprocess→segment→crop→feature→aggregate)."""

    def __init__(self, initializer: Initializer, api_key: str, cache_dir: str):
        self.initializer = initializer
        self.api_key = api_key
        self.cache_dir = cache_dir
        self.raw_dir = os.path.join(cache_dir, "raw_images")
        self.preprocess_dir = os.path.join(cache_dir, "preprocess")
        self.segment_dir = os.path.join(cache_dir, "segment")
        self.crops_dir = os.path.join(cache_dir, "crops")
        os.makedirs(self.raw_dir, exist_ok=True)
        os.makedirs(self.preprocess_dir, exist_ok=True)
        os.makedirs(self.segment_dir, exist_ok=True)
        os.makedirs(self.crops_dir, exist_ok=True)

    def _load_tool(self, tool_name: str):
        try:
            tool_dir = self.initializer.class_name_to_dir(tool_name)
            module = importlib.import_module(f"octotools.tools.{tool_dir}.tool")
            tool_class = getattr(module, tool_name)
            inputs = {}
            if getattr(tool_class, "require_llm_engine", False):
                inputs["model_string"] = self.initializer.model_string
            if getattr(tool_class, "require_api_key", False):
                inputs["api_key"] = self.api_key
            tool = tool_class(**inputs)
            if hasattr(tool, "set_custom_output_dir"):
                tool.set_custom_output_dir(os.path.join(self.cache_dir, "tool_cache"))
            return tool
        except Exception as e:
            print(f"Warning: failed to load tool {tool_name}: {e}")
            return None

    def preprocess_batch(self, batch_images: List[BatchImage]) -> Dict[str, str]:
        tool = self._load_tool("Image_Preprocessor_Tool")
        processed = {}
        for img in batch_images:
            out_path = img.image_path
            # Save original with provenance
            try:
                raw_target = os.path.join(self.raw_dir, f"{img.group}_{img.image_name}")
                shutil.copyfile(img.image_path, raw_target)
            except Exception as e:
                print(f"Raw copy failed for {img.image_path}: {e}")
            if tool:
                try:
                    res = tool.execute(image=img.image_path)
                    out_path = res.get("processed_image_path", out_path) if isinstance(res, dict) else out_path
                except Exception as e:
                    print(f"Preprocess failed for {img.image_path}: {e}")
            # Preserve provenance in filename
            try:
                ext = Path(out_path).suffix or ".png"
                target = os.path.join(self.preprocess_dir, f"{img.group}_{img.image_name}_processed{ext}")
                shutil.copyfile(out_path, target)
                out_path = target
            except Exception as e:
                print(f"Preprocess copy failed for {out_path}: {e}")
            processed[img.image_id] = out_path
        return processed

    def segment_batch(self, batch_images: List[BatchImage], preprocessed: Dict[str, str]) -> Dict[str, Dict[str, Any]]:
        tool = self._load_tool("Nuclei_Segmenter_Tool")
        results = {}
        for img in batch_images:
            img_path = preprocessed.get(img.image_id, img.image_path)
            seg_res = {}
            if tool:
                try:
                    seg_res = tool.execute(image=img_path)
                except Exception as e:
                    print(f"Segmentation failed for {img_path}: {e}")
            # Copy visual outputs with provenance
            vis = seg_res.get("visual_outputs") if isinstance(seg_res, dict) else None
            if isinstance(vis, list):
                new_vis = []
                for p in vis:
                    try:
                        target = os.path.join(self.segment_dir, f"{img.group}_{img.image_name}_{os.path.basename(p)}")
                        shutil.copyfile(p, target)
                        new_vis.append(target)
                    except Exception as e:
                        print(f"Failed to copy segmentation output {p}: {e}")
                seg_res["visual_outputs"] = new_vis
            results[img.image_id] = seg_res
        return results

    def crop_batch(self, batch_images: List[BatchImage], preprocessed: Dict[str, str], seg_results: Dict[str, Dict[str, Any]]) -> List[CellCrop]:
        tool = self._load_tool("Single_Cell_Cropper_Tool")
        crops: List[CellCrop] = []
        if not tool:
            return crops
        if Image is None or np is None:
            raise ImportError("Pillow and numpy are required for cropping and feature extraction")
        for img in batch_images:
            img_path = preprocessed.get(img.image_id, img.image_path)
            seg_res = seg_results.get(img.image_id, {})
            mask_path = None
            if isinstance(seg_res, dict):
                vis = seg_res.get("visual_outputs") or []
                for p in vis:
                    if p.lower().endswith(".png") and "mask" in os.path.basename(p).lower():
                        mask_path = p
                        break
            try:
                res = tool.execute(original_image=img_path, nuclei_mask=mask_path) if mask_path else tool.execute(original_image=img_path)
                crop_paths = res.get("cell_crops") or res.get("cropped_cells") or []
                for idx, cp in enumerate(crop_paths):
                    try:
                        with Image.open(cp) as cimg:
                            crop_arr = np.array(cimg)
                        cell_id = f"{img.image_id}_cell_{idx}"
                        crop_id = f"{img.image_id}_crop_{idx}"
                        target = os.path.join(self.crops_dir, f"{img.group}_{img.image_name}_cell_{idx}.png")
                        try:
                            shutil.copyfile(cp, target)
                        except Exception as e:
                            print(f"Failed to copy crop {cp} to {target}: {e}")
                            target = cp
                        crops.append(CellCrop(
                            crop_id=crop_id,
                            group=img.group,
                            image_id=img.image_id,
                            cell_id=cell_id,
                            image=crop_arr,
                            path=target
                        ))
                    except Exception as e:
                        print(f"Failed to load crop {cp}: {e}")
            except Exception as e:
                print(f"Cropping failed for {img_path}: {e}")
        return crops

    def feature_batch(self, crops: List[CellCrop]) -> List[Dict[str, Any]]:
        if np is None:
            raise ImportError("numpy is required for feature extraction")
        features = []
        for crop in crops:
            arr = crop.image
            area = int(arr.size) if hasattr(arr, "size") else 0
            features.append({
                "group": crop.group,
                "image_id": crop.image_id,
                "cell_id": crop.cell_id,
                "crop_id": crop.crop_id,
                "area": area
            })
        return features

    def aggregate(self, features: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        if np is None:
            raise ImportError("numpy is required for aggregation")
        grouped: Dict[str, List[Dict[str, Any]]] = {}
        for feat in features:
            grouped.setdefault(feat["group"], []).append(feat)
        summary = []
        for group, feats in grouped.items():
            areas = [f["area"] for f in feats] if feats else []
            cell_count = len(areas)
            mean_area = float(np.mean(areas)) if areas else 0.0
            std_area = float(np.std(areas)) if areas else 0.0
            summary.append({
                "group": group,
                "cell_count": int(cell_count),
                "mean_area": mean_area,
                "std_area": std_area
            })
        return summary

    def summarize_groups(self, aggregated: List[Dict[str, Any]]) -> str:
        llm_input = {
            "groups": [
                {
                    "group": row["group"],
                    "cell_count": row["cell_count"],
                    "mean_area": row["mean_area"],
                    "std_area": row["std_area"],
                }
                for row in aggregated
            ]
        }
        llm_input_text = json.dumps(llm_input, indent=2)
        print(f"[Batch] LLM summarization input length: {len(llm_input_text)} chars", flush=True)
        llm_summary = ""
        try:
            print("[Batch] LLM request started", flush=True)
            llm = ChatOpenAI(model_string=self.initializer.model_string, is_multimodal=False, api_key=self.api_key)
            prompt = f"Summarize key differences between groups based on cell counts and mean/std area.\nData:\n{llm_input_text}"
            llm_summary = llm.generate(prompt)
            print("[Batch] LLM request finished", flush=True)
        except Exception as e:
            llm_summary = f"(LLM summarization failed: {e})"
            print(f"[Batch] LLM request failed: {e}", flush=True)
        return llm_summary

    def run(self, grouped_inputs: Dict[str, List[str]]) -> Dict[str, Any]:
        if np is None:
            raise ImportError("numpy is required for batch pipeline execution")
        batch_images: List[BatchImage] = []
        for group, paths in grouped_inputs.items():
            for p in paths:
                batch_images.append(make_batch_image(p, group))

        print(f"[Batch] Starting preprocess for {len(batch_images)} images across {len(grouped_inputs)} groups", flush=True)
        preprocessed = self.preprocess_batch(batch_images)
        print(f"[Batch] Preprocess complete", flush=True)
        print(f"[Batch] Starting segmentation", flush=True)
        segmented = self.segment_batch(batch_images, preprocessed)
        print(f"[Batch] Segmentation complete", flush=True)
        print(f"[Batch] Starting cropping", flush=True)
        crops = self.crop_batch(batch_images, preprocessed, segmented)
        print(f"[Batch] Cropping complete; total crops: {len(crops)}", flush=True)
        print(f"[Batch] Starting feature extraction", flush=True)
        features = self.feature_batch(crops)
        print(f"[Batch] Feature extraction complete; total feature rows: {len(features)}", flush=True)
        print(f"[Batch] Starting aggregation", flush=True)
        aggregated = self.aggregate(features)
        print(f"[Batch] Aggregation complete; groups summarized: {len(aggregated)}", flush=True)

        summary_lines = ["### Group-level summary"]
        for row in aggregated:
            summary_lines.append(f"- {row['group']}: cells={row['cell_count']}, mean_area={row['mean_area']:.1f}, std_area={row['std_area']:.1f}")

        llm_input = {
            "groups": [
                {
                    "group": row["group"],
                    "cell_count": row["cell_count"],
                    "mean_area": row["mean_area"],
                    "std_area": row["std_area"],
                }
                for row in aggregated
            ]
        }
        llm_input_text = json.dumps(llm_input, indent=2)
        print(f"[Batch] LLM summarization input length: {len(llm_input_text)} chars", flush=True)
        llm_summary = ""
        try:
            print("[Batch] LLM request started", flush=True)
            llm = ChatOpenAI(model_string=self.initializer.model_string, is_multimodal=False, api_key=self.api_key)
            prompt = f"Summarize key differences between groups based on cell counts and mean/std area.\nData:\n{llm_input_text}"
            llm_summary = llm.generate(prompt)
            print("[Batch] LLM request finished", flush=True)
        except Exception as e:
            llm_summary = f"(LLM summarization failed: {e})"
            print(f"[Batch] LLM request failed: {e}", flush=True)

        final_md = "\n".join(summary_lines) + "\n\n### LLM Summary\n" + str(llm_summary)
        return {
            "crops": crops,
            "features": features,
            "aggregated": aggregated,
            "summary_md": final_md
        }


def solve_problem_gradio(user_query, user_images, image_table, max_steps=10, max_time=60, llm_model_engine=None, enabled_fibroblast_tools=None, enabled_general_tools=None, clear_previous_viz=False, conversation_history=None):
    """
    Solve a problem using the Gradio interface with optional visualization clearing.
    
    Args:
        user_query: The user's query
        user_images: List of uploaded images (Gradio Files)
        image_table: Dataframe of user-provided names and image paths
        max_steps: Maximum number of reasoning steps
        max_time: Maximum analysis time in seconds
        llm_model_engine: Language model engine (model_id from dropdown)
        enabled_fibroblast_tools: List of enabled fibroblast tools
        enabled_general_tools: List of enabled general tools
        clear_previous_viz: Whether to clear previous visualizations
        conversation_history: Persistent chat history to keep context across runs
    """
    # Pre-initialize all locals that are referenced later to avoid UnboundLocalError
    state: AgentState = conversation_history if isinstance(conversation_history, AgentState) else AgentState()
    state.conversation = list(state.conversation)
    state.analysis_session = state.analysis_session or AnalysisSession()
    messages: List[ChatMessage] = list(state.conversation)
    gallery_output: List[Any] = []
    grouped_preview: Dict[str, List[str]] = {}
    named_inputs: List[Dict[str, str]] = []

    # Normalize inputs into a single list of named inputs (works for single or multi-image)
    uploaded_files = user_images or []
    # Collect names from the table (single "name" column)
    table_names: List[str] = []
    if image_table is not None:
        # Gradio Dataframe returns a list-like or pandas.DataFrame; handle both
        try:
            iter_rows = image_table.values.tolist() if hasattr(image_table, "values") else image_table
        except Exception:
            iter_rows = image_table
        for row in iter_rows:
            if not row:
                continue
            name = str(row[0]).strip() if row[0] else ""
            if name:
                table_names.append(name)

    if uploaded_files:
        file_paths = []
        for f in uploaded_files:
            path = getattr(f, "name", None) or getattr(f, "path", None) or str(f)
            file_paths.append(path)

        # If no names provided in table, auto-fill from filenames; otherwise enforce 1:1
        if not table_names:
            table_names = [Path(p).stem for p in file_paths]
        if len(table_names) != len(file_paths):
            error_msg = f"Number of names ({len(table_names)}) does not match number of uploaded images ({len(file_paths)})."
            messages.append(ChatMessage(role="assistant", content=error_msg))
            state.conversation = messages
            return messages, "", [], "**Progress**: Error", state

        for name, path in zip(table_names, file_paths):
            if not name.strip():
                error_msg = "Each uploaded image must have a non-empty name."
                messages.append(ChatMessage(role="assistant", content=error_msg))
                state.conversation = messages
                return messages, "", [], "**Progress**: Error", state
            named_inputs.append({"name": name.strip(), "type": "image", "path": path})

    # Initialize or reuse persistent agent state
    state: AgentState = conversation_history if isinstance(conversation_history, AgentState) else AgentState()
    state.conversation = list(state.conversation)
    state.analysis_session = state.analysis_session or AnalysisSession()
    # Start with prior conversation so the session feels continuous
    messages: List[ChatMessage] = list(state.conversation)
    if user_query:
        messages.append(ChatMessage(role="user", content=str(user_query)))

    # Prepare grouped summary for interpretation
    grouped_preview: Dict[str, List[str]] = {}
    for item in named_inputs:
        grouped_preview.setdefault(item["name"], []).append(item["path"])

    # Query interpretation (lightweight, deterministic)
    interpretation_lines = [
        "### 🧭 Query Interpretation",
        f"- Task type: Batch image analysis",
        f"- Groups: {len(grouped_preview)}",
    ]
    for group, paths in grouped_preview.items():
        interpretation_lines.append(f"  - {group}: {len(paths)} image(s)")
    interpretation_lines.append("- Planned outputs: preprocessing visuals, segmentation overlays/masks")
    interpretation_lines.append("- Cell-level analysis: attempted if masks yield cells; skipped otherwise")
    interpretation_md = "\n".join(interpretation_lines)
    messages.append(ChatMessage(role="assistant", content=interpretation_md))
    yield messages, interpretation_md, [], "**Progress**: Interpretation ready", state
    
    # Find the model config by model_id
    selected_model_config = None
    for model_key, config in OPENAI_MODEL_CONFIGS.items():
        if config.get("model_id") == llm_model_engine:
            selected_model_config = config
            break
    
    # Use the model name for octotools
    model_name_for_octotools = llm_model_engine
    
    # Get API key from environment
    api_key = os.getenv("OPENAI_API_KEY")
    
    # Combine the tool lists
    enabled_tools = (enabled_fibroblast_tools or []) + (enabled_general_tools or [])

    # Generate a unique query ID
    query_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8] # e.g, 20250217_062225_612f2474
    print(f"Query ID: {query_id}")

    # NOTE: update the global variable to save the query ID
    global QUERY_ID
    QUERY_ID = query_id

    # Handle visualization clearing based on user preference
    if clear_previous_viz:
        print("🧹 Clearing output_visualizations directory as requested...")
        # Manually clear the directory
        output_viz_dir = os.path.join(os.getcwd(), 'output_visualizations')
        if os.path.exists(output_viz_dir):
            import shutil
            shutil.rmtree(output_viz_dir)
            print(f"✅ Cleared output directory: {output_viz_dir}")
        os.makedirs(output_viz_dir, exist_ok=True)
        print("✅ Output directory cleared successfully")
    else:
        print("📁 Preserving output_visualizations directory for continuity...")
        # Just ensure directory exists without clearing
        output_viz_dir = os.path.join(os.getcwd(), 'output_visualizations')
        os.makedirs(output_viz_dir, exist_ok=True)
        print("✅ Output directory preserved - all charts will be retained")

    # Create a directory for the query ID
    query_cache_dir = os.path.join(DATASET_DIR.name, query_id) # NOTE
    os.makedirs(query_cache_dir, exist_ok=True)

    if api_key is None or api_key.strip() == "":
        new_history = messages + [gr.ChatMessage(role="assistant", content="""⚠️ **API Key Configuration Required**

To use this application, you need to set up your OpenAI API key as an environment variable:

**Environment Variable Setup:**
Set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For Hugging Face Spaces, add this as a secret in your Space settings.

For more information about obtaining an OpenAI API key, visit: https://platform.openai.com/api-keys
""")]
        state.conversation = new_history
        return new_history, "", [], "**Progress**: Ready", state
    
    # Debug: Print enabled_tools
    print(f"Debug - enabled_tools: {enabled_tools}")
    print(f"Debug - type of enabled_tools: {type(enabled_tools)}")
    
    # Ensure enabled_tools is a list and not empty
    if not enabled_tools:
        print("⚠️ No tools selected in UI, defaulting to all available tools.")
        # Get all tools from the directory as a fallback
        tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'octotools', 'tools')
        enabled_tools = [
            d for d in os.listdir(tools_dir)
            if os.path.isdir(os.path.join(tools_dir, d)) and not d.startswith('__')
        ]
    elif isinstance(enabled_tools, str):
        enabled_tools = [enabled_tools]
    elif not isinstance(enabled_tools, list):
        enabled_tools = list(enabled_tools) if hasattr(enabled_tools, '__iter__') else []

    if not enabled_tools:
        print("❌ Critical Error: Could not determine a default tool list. Using Generalist_Solution_Generator_Tool as a last resort.")
        enabled_tools = ["Generalist_Solution_Generator_Tool"]

    print(f"Debug - final enabled_tools: {enabled_tools}")
    
    # Save the query data (use first image path if available)
    first_image_path = named_inputs[0]["path"] if named_inputs else None
    save_query_data(
        query_id=query_id,
        query=user_query,
        image_path=first_image_path
    )

    # Build named inputs from UI (files + editable table)
    if named_inputs:
        state.analysis_session = state.analysis_session or AnalysisSession()
        state.analysis_session.inputs = {
            item["name"]: AnalysisInput(name=item["name"], path=item["path"], input_type=item.get("type", "image"))
            for item in named_inputs
        }
        if not state.analysis_session.active_input:
            state.analysis_session.active_input = named_inputs[0]["name"]
    # Reject empty inputs early
    if not named_inputs:
        error_msg = "No images provided. Please upload and name at least one image."
        messages.append(ChatMessage(role="assistant", content=error_msg))
        state.conversation = messages
        return messages, "", [], "**Progress**: Error", state


    # Instantiate Initializer for deterministic batch pipeline
    try:
        initializer = Initializer(
            enabled_tools=enabled_tools,
            model_string=model_name_for_octotools,
            api_key=api_key
        )
        print(f"Debug - Initializer created successfully with {len(initializer.available_tools)} tools")
    except Exception as e:
        print(f"Error creating Initializer: {e}")
        new_history = messages + [gr.ChatMessage(role="assistant", content=f"⚠️ Error: Failed to initialize tools. {str(e)}")]
        state.conversation = new_history
        return new_history, "", [], "**Progress**: Error occurred", state

    # Deterministic batch pipeline execution
    runner = BatchPipelineRunner(initializer=initializer, api_key=api_key, cache_dir=query_cache_dir)
    grouped_inputs: Dict[str, List[str]] = {}
    for item in named_inputs:
        grouped_inputs.setdefault(item["name"], []).append(item["path"])

    try:
        messages.append(ChatMessage(role="assistant", content="### 📝 Received Query\nDeterministic batch pipeline starting..."))
        yield messages, "", [], "**Progress**: Starting batch analysis", state

        # Build batch images with provenance
        batch_images: List[BatchImage] = []
        for item in named_inputs:
            batch_images.append(make_batch_image(item["path"], item["name"]))

        gallery_output: List[Any] = []

        def add_to_gallery(img_path: str, label: str):
            try:
                if not os.path.exists(img_path):
                    return
                with Image.open(img_path) as im:
                    gallery_output.append((im.copy(), label))
            except Exception as e:
                print(f"Gallery load failed for {img_path}: {e}")

        # Stage: preprocess
        preprocessed = runner.preprocess_batch(batch_images)
        messages.append(ChatMessage(role="assistant", content="✅ Preprocessing complete"))
        for img in batch_images:
            processed_path = preprocessed.get(img.image_id, img.image_path)
            add_to_gallery(processed_path, f"{img.group}/{img.image_name} (preprocessed)")
            messages.append(ChatMessage(role="assistant", content=f"Processed {img.group}/{img.image_name}"))
            yield messages, "", gallery_output, f"**Progress**: Preprocessed {img.image_name}", state

        # Stage: segmentation
        segmented = runner.segment_batch(batch_images, preprocessed)
        messages.append(ChatMessage(role="assistant", content="✅ Segmentation complete (image-level outputs saved)"))
        for img in batch_images:
            seg_res = segmented.get(img.image_id, {})
            vis = seg_res.get("visual_outputs") if isinstance(seg_res, dict) else []
            for p in vis:
                add_to_gallery(p, f"{img.group}/{img.image_name} (seg)")
            messages.append(ChatMessage(role="assistant", content=f"Segmented {img.group}/{img.image_name}"))
            yield messages, "", gallery_output, f"**Progress**: Segmented {img.image_name}", state

        # Stage: cropping (optional)
        crops = runner.crop_batch(batch_images, preprocessed, segmented)
        if not crops:
            messages.append(ChatMessage(role="assistant", content="⚠️ No crops generated; skipping feature extraction and aggregation."))
            state.conversation = messages
            yield messages, "", gallery_output, "**Progress**: Cropping produced 0 crops", state
            return
        messages.append(ChatMessage(role="assistant", content=f"✅ Cropping complete ({len(crops)} crops)"))
        # Show up to a few crops per image to avoid overload
        crops_shown = 0
        for crop in crops:
            if crop.path and crops_shown < 12:
                add_to_gallery(crop.path, f"{crop.group}/{crop.image_id} {crop.cell_id}")
                crops_shown += 1
        yield messages, "", gallery_output, "**Progress**: Cropping complete", state

        # Stage: feature extraction
        features = runner.feature_batch(crops)
        if not features:
            messages.append(ChatMessage(role="assistant", content="⚠️ No features extracted; skipping aggregation and LLM summary."))
            state.conversation = messages
            yield messages, "", gallery_output, "**Progress**: No features extracted", state
            return
        messages.append(ChatMessage(role="assistant", content=f"✅ Feature extraction complete ({len(features)} feature rows)"))
        yield messages, "", gallery_output, "**Progress**: Feature extraction complete", state

        # Stage: aggregation
        aggregated = runner.aggregate(features)
        if not aggregated:
            messages.append(ChatMessage(role="assistant", content="⚠️ No aggregated data; skipping LLM summary."))
            state.conversation = messages
            yield messages, "", gallery_output, "**Progress**: No aggregated data", state
            return
        messages.append(ChatMessage(role="assistant", content=f"✅ Aggregation complete ({len(aggregated)} groups)"))
        yield messages, "", gallery_output, "**Progress**: Aggregation complete", state

        # Summary and optional LLM interpretation
        summary_lines = ["### Group-level summary"]
        for row in aggregated:
            summary_lines.append(f"- {row['group']}: cells={row['cell_count']}, mean_area={row['mean_area']:.1f}, std_area={row['std_area']:.1f}")
        summary_md = "\n".join(summary_lines)

        llm_summary = runner.summarize_groups(aggregated) if aggregated else ""
        final_md = summary_md + ("\n\n### LLM Summary\n" + str(llm_summary) if llm_summary else "")

        messages.append(ChatMessage(role="assistant", content=final_md))
        state.conversation = messages
        state.analysis_session = state.analysis_session or AnalysisSession()
        yield messages, final_md, gallery_output, "**Progress**: Completed", state
    except Exception as e:
        error_message = f"⚠️ Error during batch analysis: {e}"
        messages.append(ChatMessage(role="assistant", content=error_message))
        state.conversation = messages
        yield messages, "", [], "**Progress**: Error occurred", state


def main(args):
    #################### Gradio Interface ####################
    with gr.Blocks() as demo:
        # Theming https://www.gradio.app/guides/theming-guide
        
        gr.Markdown("# Chat with SHAPE: A self-supervised morphology agent for single-cell phenotype")  # Title
        gr.Markdown("""
        **SHPAE** is an open-source assistant for interpreting cell images, powered by large language models and tool-based reasoning.
        """)
        
        with gr.Row():
            # Left control panel
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### ⚙️ Model Configuration")
                
                # Model and limits
                multimodal_models = [m for m in OPENAI_MODEL_CONFIGS.values()]
                model_names = [m["model_id"] for m in multimodal_models]
                
                # Set default to first OpenAI model if available, otherwise first model
                default_model = None
                for model in multimodal_models:
                    if model.get("model_type") == "openai":
                        default_model = model["model_id"]
                        break
                if not default_model and model_names:
                    default_model = model_names[0]
                
                language_model = gr.Dropdown(
                    choices=model_names,
                    value=default_model,
                    label="Multimodal Large Language Model"
                )
                max_steps = gr.Slider(1, 15, value=10, label="Max Reasoning Steps")
                max_time = gr.Slider(60, 600, value=300, label="Max Analysis Time (seconds)")

                # Visualization options
                gr.Markdown("#### 📊 Visualization Options")
                clear_previous_viz = gr.Checkbox(
                    label="Clear previous visualizations", 
                    value=False,
                    info="Check this to clear all previous charts when starting new analysis"
                )

                # Tool selection
                gr.Markdown("#### 🛠️ Available Tools")
                
                # Fibroblast analysis tools
                fibroblast_tools = [
                    "Image_Preprocessor_Tool",
                    "Nuclei_Segmenter_Tool",
                    "Single_Cell_Cropper_Tool",
                    "Fibroblast_State_Analyzer_Tool",
                    "Fibroblast_Activation_Scorer_Tool"
                ]
                
                # General tools
                general_tools = [
                    "Generalist_Solution_Generator_Tool",
                    "Python_Code_Generator_Tool",
                    "ArXiv_Paper_Searcher_Tool",
                    "Pubmed_Search_Tool",
                    "Nature_News_Fetcher_Tool",
                    "Google_Search_Tool",
                    "Wikipedia_Knowledge_Searcher_Tool",
                    "URL_Text_Extractor_Tool",
                    "Object_Detector_Tool",
                    "Image_Captioner_Tool", 
                    "Relevant_Patch_Zoomer_Tool",
                    "Text_Detector_Tool",
                    "Advanced_Object_Detector_Tool"
                ]
                
                with gr.Accordion("🧬 Fibroblas Tools", open=True):
                    enabled_fibroblast_tools = gr.CheckboxGroup(
                        choices=fibroblast_tools, 
                        value=fibroblast_tools, 
                        label="Select Fibroblast Analysis Tools"
                    )

                with gr.Accordion("🧩 General Tools", open=False):
                    enabled_general_tools = gr.CheckboxGroup(
                        choices=general_tools, 
                        label="Select General Purpose Tools"
                    )

                with gr.Row():
                    gr.Button("Select Fibroblast Tools", size="sm").click(
                        lambda: fibroblast_tools, outputs=enabled_fibroblast_tools
                    )
                    gr.Button("Select All Tools", size="sm").click(
                        lambda: (fibroblast_tools, general_tools), 
                        outputs=[enabled_fibroblast_tools, enabled_general_tools]
                    )
                    gr.Button("Clear Selection", size="sm").click(
                        lambda: ([], []), 
                        outputs=[enabled_fibroblast_tools, enabled_general_tools]
                    )

            # Main interface
            with gr.Column(scale=5):
                # Input area
                gr.Markdown("### 📤 Data Input (multi-image, named conditions)")
                with gr.Row():
                    with gr.Column(scale=1):
                        user_images = gr.Files(
                            label="Upload Images (multiple)", 
                            file_types=["image"], 
                            file_count="multiple",
                            height=200
                        )
                        image_table = gr.Dataframe(
                            headers=["name"],
                            datatype=["str"],
                            row_count=(1, "dynamic"),
                            col_count=1,
                            label="Name your images (order matches uploads)",
                            interactive=True
                        )
                        user_images.change(
                            build_image_table,
                            inputs=user_images,
                            outputs=image_table
                        )
                    with gr.Column(scale=1):
                        user_query = gr.Textbox(
                            label="Analysis Question", 
                            placeholder="Describe the features or comparisons you want across these named images (e.g., compare control vs TGFB1)...", 
                            lines=15
                        )
                        
                # Submit button
                with gr.Row():
                    with gr.Column(scale=6):
                        run_button = gr.Button("🚀 Start Analysis", variant="primary", size="lg")
                        progress_md = gr.Markdown("**Progress**: Ready")
                        conversation_state = gr.State(AgentState())

                # Output area - two columns instead of three
                gr.Markdown("### 📊 Analysis Results")
                with gr.Row():
                    # Reasoning steps
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🔍 Reasoning Steps")
                        chatbot_output = gr.Chatbot(
                            type="messages", 
                            height=700,
                            show_label=False
                        )

                    # Combined analysis report and visual output
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 Analysis Report & Visual Output")
                        with gr.Group():
                            #gr.Markdown("*The final analysis conclusion and key findings will appear here.*")
                            text_output = gr.Markdown(
                                value="",
                                height=350
                            )
                            gallery_output = gr.Gallery(
                                label=None, 
                                show_label=False,
                                height=350,
                                columns=2,
                                rows=2
                            )

                # Bottom row for examples
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("## 💡 Try these examples with suggested tools.")
                        
                        # Define example lists
                        fibroblast_examples = [
                            ["Image Preprocessing", "examples/A5_01_1_1_Phase Contrast_001.png", "Normalize this phase contrast image.", 
                             "Image_Preprocessor_Tool", "Illumination-corrected and brightness-normalized phase contrast image."],
                            ["Cell Identification", "examples/A2_02_1_1_Phase Contrast_001.png", "How many cells are there in this image.", 
                             "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool", "258 cells are identified and their nuclei are labeled."],
                            ["Single-Cell Cropping", "examples/A3_02_1_1_Phase Contrast_001.png", "Crop single cells from the segmented nuclei in this image.", 
                             "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool", "Individual cell crops extracted from the image."],
                            ["Fibroblast State Analysis", "examples/A4_02_1_1_Phase Contrast_001.png", "Analyze the fibroblast cell states in this image.", 
                             "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool, Fibroblast_State_Analyzer_Tool", "540 cells identified and segmented successfully. Comprehensive analysis of fibroblast cell states have been performed with visualizations."],
                            ["Fibroblast Activation Scoring", "examples/A5_02_1_1_Phase Contrast_001.png", "Quantify the activation score of each fibroblast in this image.",
                             "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool, Fibroblast_State_Analyzer_Tool, Fibroblast_Activation_Scorer_Tool", "Activation scores for all fibroblasts have been computed and normalized based on the reference map."]
                        ]
                        
                        general_examples = [
                            ["Pathology Diagnosis", "examples/pathology.jpg", "What are the cell types in this image?", 
                             "Generalist_Solution_Generator_Tool, Image_Captioner_Tool, Relevant_Patch_Zoomer_Tool", "Need expert insights."],
                            ["Visual Reasoning", "examples/rotting_kiwi.png", "You are given a 3 x 3 grid in which each cell can contain either no kiwi, one fresh kiwi, or one rotten kiwi. Every minute, any fresh kiwi that is 4-directionally adjacent to a rotten kiwi also becomes rotten. What is the minimum number of minutes that must elapse until no cell has a fresh kiwi?", 
                             "Image_Captioner_Tool", "4 minutes"],
                            ["Scientific Research", None, "What are the research trends in tool agents with large language models for scientific discovery? Please consider the latest literature from ArXiv, PubMed, Nature, and news sources.", 
                             "ArXiv_Paper_Searcher_Tool, Pubmed_Search_Tool, Nature_News_Fetcher_Tool", "Open-ended question. No reference answer."]
                        ]

                        # Helper function to distribute tools
                        def distribute_tools(category, img, q, tools_str, ans):
                            selected_tools = [tool.strip() for tool in tools_str.split(',')]
                            selected_fibroblast = [tool for tool in selected_tools if tool in fibroblast_tools]
                            selected_general = [tool for tool in selected_tools if tool in general_tools]
                            return img, q, selected_fibroblast, selected_general

                        gr.Markdown("#### 🧬 Fibroblast Analysis Examples")
                        gr.Examples(
                            examples=fibroblast_examples,
                            inputs=[gr.Textbox(label="Category", visible=False), user_images, user_query, gr.Textbox(label="Select Tools", visible=False), gr.Textbox(label="Reference Answer", visible=False)],
                            outputs=[user_images, user_query, enabled_fibroblast_tools, enabled_general_tools],
                            fn=distribute_tools,
                            cache_examples=False
                        )
                        
                        gr.Markdown("#### 🧩 General Purpose Examples")
                        gr.Examples(
                            examples=general_examples,
                            inputs=[gr.Textbox(label="Category", visible=False), user_images, user_query, gr.Textbox(label="Select Tools", visible=False), gr.Textbox(label="Reference Answer", visible=False)],
                            outputs=[user_images, user_query, enabled_fibroblast_tools, enabled_general_tools],
                            fn=distribute_tools,
                            cache_examples=False
                        )

        # Button click event
        run_button.click(
            solve_problem_gradio,
            [user_query, user_images, image_table, max_steps, max_time, language_model, enabled_fibroblast_tools, enabled_general_tools, clear_previous_viz, conversation_state],
            [chatbot_output, text_output, gallery_output, progress_md, conversation_state]
        )

    #################### Gradio Interface ####################

    # Launch configuration
    if IS_SPACES:
        # HuggingFace Spaces config
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
    else:
        # Local development config
        demo.launch(
            server_name="0.0.0.0",
            server_port=1048,
            debug=True,
            share=False
        )

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set default API source to use environment variables
    if not hasattr(args, 'openai_api_source') or args.openai_api_source is None:
        args.openai_api_source = "we_provided"

    # All available tools
    all_tools = [
        # Cell analysis tools
        "Object_Detector_Tool",           # Cell detection and counting
        "Image_Captioner_Tool",           # Cell morphology description
        "Relevant_Patch_Zoomer_Tool",     # Cell region zoom analysis
        "Text_Detector_Tool",             # Text recognition in images
        "Advanced_Object_Detector_Tool",  # Advanced cell detection
        "Image_Preprocessor_Tool",        # Image preprocessing and enhancement
        "Nuclei_Segmenter_Tool",          # Nuclei segmentation
        "Single_Cell_Cropper_Tool",        # Single cell cropping
        "Fibroblast_State_Analyzer_Tool",  # Fibroblast state analysis
        
        # General analysis tools
        "Generalist_Solution_Generator_Tool",  # Comprehensive analysis generation
        "Python_Code_Generator_Tool",          # Code generation
        
        # Research literature tools
        "ArXiv_Paper_Searcher_Tool",      # arXiv paper search
        "Pubmed_Search_Tool",             # PubMed literature search
        "Nature_News_Fetcher_Tool",       # Nature news fetching
        "Google_Search_Tool",             # Google search
        "Wikipedia_Knowledge_Searcher_Tool",  # Wikipedia search
        "URL_Text_Extractor_Tool",        # URL text extraction
    ]
    args.enabled_tools = all_tools

    # NOTE: Use the same name for the query cache directory as the dataset directory
    args.root_cache_dir = DATASET_DIR.name
    
    # Print environment information
    print("\n=== Environment Information ===")
    print(f"Running in HuggingFace Spaces: {IS_SPACES}")
    if torch:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("CUDA Available: torch not installed")
    #print(f"API Key Source: {args.openai_api_source}")
    print("==============================\n")
    
    main(args)

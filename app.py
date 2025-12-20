import os
import sys
repo_root = os.path.dirname(os.path.abspath(__file__))
if repo_root not in sys.path:
    sys.path.insert(0, repo_root)
import json
import argparse
import time
import io
import uuid
try:
    import torch
    TORCH_AVAILABLE = True
except ImportError:
    torch = None
    TORCH_AVAILABLE = False
import shutil
import logging
import tempfile
from PIL import Image
import numpy as np
from tifffile import imwrite as tiff_write
from typing import List, Dict, Any, Iterator
import matplotlib.pyplot as plt
import gradio as gr
from gradio import ChatMessage
from pathlib import Path
import hashlib
from huggingface_hub import CommitScheduler
from octotools.models.formatters import ToolCommand
import random
import traceback
import psutil  # For memory usage
from llm_evaluation_scripts.hf_model_configs import HF_MODEL_CONFIGS
from datetime import datetime
from octotools.models.utils import make_json_serializable, VisualizationConfig, normalize_tool_name
from dataclasses import dataclass, field

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from octotools.models.initializer import Initializer
from octotools.models.planner import Planner
from octotools.models.memory import Memory
from octotools.models.executor import Executor

# Custom JSON encoder to handle ToolCommand objects
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ToolCommand):
            return str(obj)  # Convert ToolCommand to its string representation
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


def save_steps_data(query_id: str, memory: Memory) -> None:
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


def ensure_session_dirs(session_id: str):
    """Create and return session-scoped directories for caching."""
    session_dir = DATASET_DIR / session_id
    images_dir = session_dir / "images"
    features_dir = session_dir / "features"
    session_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, images_dir, features_dir


def compute_image_fingerprint(image) -> str:
    """Return a stable hash for the uploaded image to detect reuse."""
    hasher = hashlib.sha256()
    try:
        if isinstance(image, dict) and 'path' in image:
            with open(image['path'], "rb") as f:
                hasher.update(f.read())
        elif isinstance(image, str) and os.path.exists(image):
            with open(image, "rb") as f:
                hasher.update(f.read())
        elif hasattr(image, "save"):
            buf = io.BytesIO()
            image.save(buf, format="PNG")
            hasher.update(buf.getvalue())
        else:
            return ""
        return hasher.hexdigest()
    except Exception as e:
        print(f"Warning: failed to compute image fingerprint: {e}")
        return ""


def encode_image_features(image_path: str, features_dir: Path) -> str:
    """
    Compute a lightweight cached encoding for an image.
    This runs once per upload and is reused for all subsequent questions.
    """
    try:
        features_dir.mkdir(parents=True, exist_ok=True)
        feature_path = features_dir / (Path(image_path).stem + "_features.npy")
        if feature_path.exists():
            return str(feature_path)
        img = Image.open(image_path).convert("RGB").resize((64, 64))
        arr = np.asarray(img, dtype=np.float32) / 255.0
        pooled = np.concatenate([
            arr.mean(axis=(0, 1)),
            arr.std(axis=(0, 1)),
            arr.max(axis=(0, 1)),
            arr.min(axis=(0, 1))
        ])
        np.save(feature_path, pooled)
        return str(feature_path)
    except Exception as e:
        print(f"Warning: failed to encode image features for {image_path}: {e}")
        return ""


def make_artifact_key(tool_name: str, image_path: str, context: str = "", sub_goal: str = "") -> str:
    """Deterministic key for caching tool outputs tied to inputs."""
    hasher = hashlib.sha256()
    hasher.update(tool_name.encode())
    hasher.update(str(image_path or "").encode())
    hasher.update(str(context or "").encode())
    hasher.update(str(sub_goal or "").encode())
    return hasher.hexdigest()


def get_cached_artifact(state: AgentState, group_name: str, tool_name: str, key: str):
    group = state.image_groups.get(group_name, {})
    artifacts = group.get("artifacts", {}).get(tool_name, [])
    for art in artifacts:
        if art.get("key") == key:
            return art
    return None


def store_artifact(state: AgentState, group_name: str, tool_name: str, key: str, result: Any):
    state.image_groups.setdefault(group_name, {"images": [], "features": [], "artifacts": {}})
    state.image_groups[group_name].setdefault("artifacts", {}).setdefault(tool_name, [])
    entry = {
        "key": key,
        "result": result,
        "created_at": time.time()
    }
    state.image_groups[group_name]["artifacts"][tool_name].append(entry)


def add_image_to_group(group_name: str, user_image, state: "AgentState", images_dir: Path, features_dir: Path) -> str:
    """Store an uploaded image into a session-level group and cache its features."""
    if not user_image:
        return "⚠️ No image provided."
    group = group_name.strip() or "default"
    state.image_groups.setdefault(group, {"images": [], "features": [], "artifacts": {}})

    fingerprint = compute_image_fingerprint(user_image)
    for entry in state.image_groups[group]["images"]:
        if entry.get("fingerprint") == fingerprint:
            state.last_group_name = group
            state.image_context = ImageContext(
                image_id=entry["image_id"],
                image_path=entry["image_path"],
                features_path=entry.get("features_path", ""),
                fingerprint=fingerprint,
                source_type="group"
            )
            return f"✅ Image already cached in group '{group}'. Reusing existing features."

    image_id = uuid.uuid4().hex
    group_image_dir = images_dir / group
    group_image_dir.mkdir(parents=True, exist_ok=True)
    image_path = group_image_dir / f"{image_id}.jpg"
    try:
        if isinstance(user_image, dict) and 'path' in user_image:
            shutil.copy(user_image['path'], image_path)
        elif isinstance(user_image, str) and os.path.exists(user_image):
            shutil.copy(user_image, image_path)
        elif hasattr(user_image, "save"):
            user_image.save(image_path)
        else:
            raise ValueError(f"Unsupported image type: {type(user_image)}")
    except Exception as e:
        print(f"Error caching uploaded image: {e}")
        traceback.print_exc()
        return f"❌ Failed to save image to group '{group}': {e}"

    feature_path = encode_image_features(str(image_path), features_dir / group)
    entry = {
        "image_id": image_id,
        "image_path": str(image_path),
        "fingerprint": fingerprint,
        "features_path": feature_path
    }
    state.image_groups[group]["images"].append(entry)
    if feature_path:
        state.image_groups[group]["features"].append(feature_path)
    # Preserve existing artifacts; already initialized above
    state.last_group_name = group
    state.image_context = ImageContext(
        image_id=image_id,
        image_path=str(image_path),
        features_path=feature_path,
        fingerprint=fingerprint,
        source_type="group"
    )
    return f"✅ Added image to group '{group}' and cached features."

########### End of Test Huggingface Dataset ###########


@dataclass
class ImageContext:
    """Persistent image metadata for the session."""
    image_id: str
    image_path: str
    features_path: str = ""
    source_type: str = "uploaded"
    created_at: float = field(default_factory=time.time)
    fingerprint: str = ""


@dataclass
class AgentState:
    """Persistent session state to survive across turns."""
    conversation: List[ChatMessage] = field(default_factory=list)
    last_context: str = ""
    last_sub_goal: str = ""
    last_visual_description: str = "*Ready to display analysis results and processed images.*"
    image_context: ImageContext = None
    image_groups: Dict[str, Dict[str, Any]] = field(default_factory=dict)  # {group: {"images": [...], "features": [...], "artifacts": {tool_name: [..]}} }
    last_group_name: str = ""
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

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
        agent_state: AgentState = None
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

    def stream_solve_user_problem(self, user_query: str, image_context: ImageContext, api_key: str, messages: List[ChatMessage]) -> Iterator:
        import time
        import os
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        visual_description = "*Ready to display analysis results and processed images.*"

        # Pull image path from cached session context
        img_path_for_tools = image_context.image_path if image_context else None
        # NOTE: Features are for downstream reasoning/caching only; they must never replace images in vision prompts
        img_path_for_analysis = None
        img_id = image_context.image_id if image_context else None
        self.agent_state.image_context = image_context
        group_name = getattr(self.agent_state, "last_group_name", "")
        analysis_img_ref = img_path_for_tools
        print(f"=== DEBUG: Image context ===")
        print(f"DEBUG: img_path_for_tools: {img_path_for_tools}")
        print(f"DEBUG: img_path_for_analysis: {img_path_for_analysis}")
        print(f"DEBUG: image_id: {img_id}")

        # Set tool cache directory
        _tool_cache_dir = os.path.join(self.query_cache_dir, "tool_cache") # NOTE: This is the directory for tool cache
        self.executor.set_query_cache_dir(_tool_cache_dir) # NOTE: set query cache directory
        
        # Step 1: Display the received inputs
        if image_context and img_path_for_tools:
            group_label = f" in group `{group_name}`" if group_name else ""
            messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}\n### 🖼️ Using session image `{img_id}`{group_label}"))
        else:
            messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}"))
        yield messages, "", [], visual_description, "**Progress**: Input received"

        # [Step 3] Initialize problem-solving state
        step_count = 0
        json_data = {"query": user_query, "image_id": img_id}

        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### 🐙 Deep Thinking:"))
        yield messages, "", [], visual_description, "**Progress**: Starting analysis"

        # [Step 4] Query Analysis - This is the key step that should happen first
        print(f"Debug - Starting query analysis for: {user_query}")
        print(f"Debug - img_path for query analysis: {analysis_img_ref}")
        query_analysis_start = time.time()
        try:
            conversation_text = self._format_conversation_history()
            query_analysis = self.planner.analyze_query(user_query, analysis_img_ref, conversation_text)
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
            next_step = self.planner.generate_next_step(user_query, analysis_img_ref, query_analysis, self.memory, step_count, self.max_steps, conversation_context=conversation_text)
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

            # [Step 6-7] Generate and execute the tool command (with artifact reuse)
            safe_path = img_path_for_tools.replace("\\", "\\\\") if img_path_for_tools else None
            conversation_text = self._format_conversation_history()
            artifact_key = make_artifact_key(tool_name, safe_path, context, sub_goal)
            cached_artifact = get_cached_artifact(self.agent_state, group_name, tool_name, artifact_key)

            if cached_artifact:
                result = cached_artifact.get("result")
                analysis = "Cached result reused"
                explanation = "Found matching artifact in session; skipping execution."
                command = "execution = 'cached_artifact'"
                messages.append(ChatMessage(
                    role="assistant",
                    content=f"♻️ Reusing cached {tool_name} result from previous turn.",
                    metadata={"title": f"### 🛠️ Step {step_count}: Cached Execution ({tool_name})"}
                ))
                print(f"Reused cached artifact for {tool_name} (key={artifact_key})")
            else:
                tool_command = self.executor.generate_tool_command(user_query, safe_path, context, sub_goal, self.planner.toolbox_metadata[tool_name], self.memory, conversation_context=conversation_text)
                analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
                result = self.executor.execute_tool_command(tool_name, command)
                result = make_json_serializable(result)
                store_artifact(self.agent_state, group_name, tool_name, artifact_key, result)
                print(f"Tool '{tool_name}' result:", result)
            
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

            # Save the command execution data
            command_execution_data = {
                "result": result,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_execution", command_execution_data)

            # [Step 8] Memory update and stopping condition
            self.memory.add_action(step_count, tool_name, sub_goal, tool_command, result)
            conversation_text = self._format_conversation_history()
            stop_verification = self.planner.verificate_memory(user_query, analysis_img_ref, query_analysis, self.memory, conversation_context=conversation_text)
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
            direct_output = self.planner.generate_direct_output(user_query, analysis_img_ref, self.memory, conversation_context=conversation_text)
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
            final_output = self.planner.generate_final_output(user_query, analysis_img_ref, self.memory, conversation_context=conversation_text) # Disabled visibility for now
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


def upload_image_to_group(user_image, group_name, conversation_state):
    """Gradio handler: add image to a named group and cache features."""
    state: AgentState = conversation_state if isinstance(conversation_state, AgentState) else AgentState()
    _, images_dir, features_dir = ensure_session_dirs(state.session_id)
    status = add_image_to_group(group_name or "default", user_image, state, images_dir, features_dir)
    progress = f"**Progress**: {status}"
    return state, progress


def solve_problem_gradio(user_query, group_name, max_steps=10, max_time=60, llm_model_engine=None, enabled_fibroblast_tools=None, enabled_general_tools=None, clear_previous_viz=False, conversation_history=None):
    """
    Solve a problem using the Gradio interface with optional visualization clearing.
    
    Args:
        user_query: The user's query
        group_name: The target image group name to analyze
        max_steps: Maximum number of reasoning steps
        max_time: Maximum analysis time in seconds
        llm_model_engine: Language model engine (model_id from dropdown)
        enabled_fibroblast_tools: List of enabled fibroblast tools
        enabled_general_tools: List of enabled general tools
        clear_previous_viz: Whether to clear previous visualizations
        conversation_history: Persistent chat history to keep context across runs
    """
    # Initialize or reuse persistent agent state
    state: AgentState = conversation_history if isinstance(conversation_history, AgentState) else AgentState()
    state.conversation = list(state.conversation)
    session_dir, images_dir, features_dir = ensure_session_dirs(state.session_id)

    # Start with prior conversation so the session feels continuous
    messages: List[ChatMessage] = list(state.conversation)
    if user_query:
        messages.append(ChatMessage(role="user", content=str(user_query)))
    
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
    # Legacy logging directory (keeps previous dataset layout)
    os.makedirs(DATASET_DIR / query_id, exist_ok=True)

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
    query_cache_dir = os.path.join(str(session_dir), query_id) # scoped per session + query
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
    
    # Instantiate Initializer
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

    # Instantiate Planner
    try:
        planner = Planner(
            llm_engine_name=model_name_for_octotools,
            toolbox_metadata=initializer.toolbox_metadata,
            available_tools=initializer.available_tools,
            api_key=api_key
        )
        print(f"Debug - Planner created successfully")
    except Exception as e:
        print(f"Error creating Planner: {e}")
        new_history = messages + [gr.ChatMessage(role="assistant", content=f"⚠️ Error: Failed to initialize planner. {str(e)}")]
        state.conversation = new_history
        return new_history, "", [], "**Progress**: Error occurred", state

    # Resolve target image group and cached features
    group_name = (group_name or state.last_group_name or "").strip()
    if not group_name and len(state.image_groups) == 1:
        group_name = next(iter(state.image_groups.keys()))
    if not group_name or group_name not in state.image_groups or not state.image_groups[group_name]["images"]:
        prompt_msg = "⚠️ Please upload an image into a group before asking a question."
        new_history = messages + [gr.ChatMessage(role="assistant", content=prompt_msg)]
        state.conversation = new_history
        return new_history, "", [], "**Progress**: Waiting for image upload", state

    group_entry = state.image_groups[group_name]
    representative = group_entry["images"][0]
    state.image_context = ImageContext(
        image_id=representative["image_id"],
        image_path=representative["image_path"],
        features_path=representative.get("features_path", ""),
        fingerprint=representative.get("fingerprint", ""),
        source_type="group"
    )
    state.last_group_name = group_name

    # Save the query data with resolved group context
    save_query_data(
        query_id=query_id,
        query=user_query,
        image_path=state.image_context.image_path if state.image_context else None
    )

    # Instantiate Memory
    memory = Memory()

    # Instantiate Executor
    executor = Executor(
        llm_engine_name=model_name_for_octotools,
        query_cache_dir=query_cache_dir, # NOTE
        enable_signal=False,
        api_key=api_key,
        initializer=initializer
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        memory=memory,
        executor=executor,
        task="minitoolbench",  # Default task
        task_description="",   # Default empty description
        output_types="base,final,direct",  # Default output types
        verbose=True,          # Default verbose
        max_steps=max_steps,
        max_time=max_time,
        query_cache_dir=query_cache_dir, # NOTE
        agent_state=state
    )

    if solver is None:
        new_history = messages + [gr.ChatMessage(role="assistant", content="⚠️ Error: Failed to initialize solver.")]
        state.conversation = new_history
        return new_history, "", [], "**Progress**: Error occurred", state

    try:
        # Stream the solution
        for messages, text_output, gallery_output, visual_desc, progress_md in solver.stream_solve_user_problem(user_query, state.image_context, api_key, messages):
            # Save steps data
            save_steps_data(query_id, memory)
            
            # Return the current state
            state.conversation = messages
            state.last_visual_description = visual_desc
            yield messages, text_output, gallery_output, progress_md, state
            
    except Exception as e:
        print(f"Error in solve_problem_gradio: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Full traceback: {error_traceback}")
        
        # Create error message for UI
        error_message = f"⚠️ Error occurred during analysis:\n\n**Error Type:** {type(e).__name__}\n**Error Message:** {str(e)}\n\nPlease check your input and try again."
        
        # Return error message in the expected format
        error_messages = messages + [gr.ChatMessage(role="assistant", content=error_message)]
        state.conversation = error_messages
        yield error_messages, "", [], "**Progress**: Error occurred", state
    finally:
        print(f"Task completed for query_id: {query_id}. Preparing to clean up cache directory: {query_cache_dir}")
        try:
            # Add a check to prevent deleting the root solver_cache
            if query_cache_dir != DATASET_DIR.name and DATASET_DIR.name in query_cache_dir:
                # Preserve output_visualizations directory - DO NOT CLEAR IT
                # This allows users to keep all generated charts until they start a new analysis
                output_viz_dir = os.path.join(os.getcwd(), 'output_visualizations')
                if os.path.exists(output_viz_dir):
                    print(f"📁 Preserving output_visualizations directory: {output_viz_dir}")
                    print(f"💡 All generated charts are preserved for review")
                
                # Add a small delay to ensure files are written
                time.sleep(1)
                
                # Clean up the cache directory (but preserve visualizations)
                shutil.rmtree(query_cache_dir)
                print(f"✅ Successfully cleaned up cache directory: {query_cache_dir}")
                print(f"💡 Note: All visualization files are preserved in output_visualizations/ directory")
            else:
                print(f"⚠️ Skipping cleanup for safety. Path was: {query_cache_dir}")
        except Exception as e:
            print(f"❌ Error cleaning up cache directory {query_cache_dir}: {e}")


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
                # Image group manager (uploads only)
                gr.Markdown("### 📤 Image Groups")
                gr.Markdown("Upload images into named groups (e.g., control, drugA). Each upload appends to the group and caches features; questions always reuse cached features.")
                with gr.Row():
                    with gr.Column(scale=1):
                        user_image = gr.Image(
                            label="Upload an Image", 
                            type="pil", 
                            height=300
                        )
                        group_name_input = gr.Textbox(
                            label="Image Group Name",
                            placeholder="e.g., control, drugA, replicate1",
                            value="control"
                        )
                        upload_btn = gr.Button("Add Image to Group", variant="primary")
                        upload_status_md = gr.Markdown("**Upload Status**: No uploads yet")

                # Conversation (question-driven execution)
                gr.Markdown("### 🗣️ Conversation")
                chatbot_output = gr.Chatbot(
                    type="messages", 
                    height=550,
                    show_label=False
                )
                user_query = gr.Textbox(
                    label="Ask a question about your groups", 
                    placeholder="e.g., Compare cell counts between control and drugA", 
                    lines=4
                )
                run_button = gr.Button("🚀 Ask Question", variant="primary", size="lg")
                progress_md = gr.Markdown("**Progress**: Ready")
                conversation_state = gr.State(AgentState())

                # Visual outputs
                gr.Markdown("### 🖼️ Visual Outputs")
                gallery_output = gr.Gallery(
                    label=None, 
                    show_label=False,
                    height=350,
                    columns=2,
                    rows=2
                )
                text_output = gr.Markdown(value="", visible=False)  # compatibility placeholder

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
                            inputs=[gr.Textbox(label="Category", visible=False), user_image, user_query, gr.Textbox(label="Select Tools", visible=False), gr.Textbox(label="Reference Answer", visible=False)],
                            outputs=[user_image, user_query, enabled_fibroblast_tools, enabled_general_tools],
                            fn=distribute_tools,
                            cache_examples=False
                        )
                        
                        gr.Markdown("#### 🧩 General Purpose Examples")
                        gr.Examples(
                            examples=general_examples,
                            inputs=[gr.Textbox(label="Category", visible=False), user_image, user_query, gr.Textbox(label="Select Tools", visible=False), gr.Textbox(label="Reference Answer", visible=False)],
                            outputs=[user_image, user_query, enabled_fibroblast_tools, enabled_general_tools],
                            fn=distribute_tools,
                            cache_examples=False
                        )

        # Button click event
        upload_btn.click(
            upload_image_to_group,
            inputs=[user_image, group_name_input, conversation_state],
            outputs=[conversation_state, upload_status_md]
        )

        run_button.click(
            solve_problem_gradio,
            [user_query, group_name_input, max_steps, max_time, language_model, enabled_fibroblast_tools, enabled_general_tools, clear_previous_viz, conversation_state],
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
    if TORCH_AVAILABLE:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch not installed; running in CPU-only mode.")
    #print(f"API Key Source: {args.openai_api_source}")
    print("==============================\n")
    
    main(args)

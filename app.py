from __future__ import annotations
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
# Progress bar removed per user request
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
from typing import List, Dict, Any, Iterator
import matplotlib.pyplot as plt
import gradio as gr
from gradio import ChatMessage
from pathlib import Path
import hashlib
from octotools.models.formatters import ToolCommand
import traceback
import psutil
from model_configs import HF_MODEL_CONFIGS
from datetime import datetime
from octotools.models.utils import make_json_serializable, normalize_tool_name
from dataclasses import dataclass, field

_AVAILABLE_TOOLS_CACHE = None

def get_available_tools() -> List[str]:
    """Dynamically discover all available tools from octotools/tools directory."""
    global _AVAILABLE_TOOLS_CACHE
    if _AVAILABLE_TOOLS_CACHE is not None:
        return _AVAILABLE_TOOLS_CACHE
    
    tools = []
    current_dir = os.path.dirname(os.path.abspath(__file__))
    tools_dir = os.path.join(current_dir, 'octotools', 'tools')
    
    if not os.path.exists(tools_dir):
        print(f"Warning: Tools directory not found: {tools_dir}")
        return tools
    
    # Iterate through all items in tools directory
    for item in os.listdir(tools_dir):
        tool_dir = os.path.join(tools_dir, item)
        
        # Skip if not a directory or starts with underscore/dot
        if not os.path.isdir(tool_dir) or item.startswith('_') or item.startswith('.'):
            continue
        
        # Check if tool.py exists in this directory
        tool_file = os.path.join(tool_dir, 'tool.py')
        if os.path.isfile(tool_file):
            # Special cases: handle tools with non-standard capitalization
            special_cases = {
                'url_text_extractor': 'URL_Text_Extractor_Tool',  # URL is all uppercase
                'arxiv_paper_searcher': 'ArXiv_Paper_Searcher_Tool',  # ArXiv is mixed case
            }
            
            if item in special_cases:
                class_name = special_cases[item]
            else:
                # Standard conversion: snake_case -> Pascal_Case_Tool
                # Example: single_cell_cropper -> Single_Cell_Cropper_Tool
                parts = item.split('_')
                class_name = '_'.join([p.capitalize() for p in parts]) + '_Tool'
            
            tools.append(class_name)
            print(f"✓ Discovered tool: {item} -> {class_name}")
    
    tools.sort()
    _AVAILABLE_TOOLS_CACHE = tools
    print(f"Total tools discovered: {len(tools)}")
    return tools

current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from octotools.models.initializer import Initializer
from octotools.models.planner import Planner
from octotools.models.memory import Memory
from octotools.models.executor import Executor

class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ToolCommand):
            return str(obj)
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
        return {
            "type": "AnnData",
            "shape": f"{obj.n_obs}x{obj.n_vars}",
            "obs_keys": list(obj.obs.keys()) if hasattr(obj, 'obs') else [],
            "var_keys": list(obj.var.keys()) if hasattr(obj, 'var') else [],
            "message": "AnnData object (removed for JSON serialization)"
        }
    elif hasattr(obj, '__dict__'):
        try:
            return make_json_serializable(obj.__dict__)
        except:
            return str(obj)
    else:
        return obj

def get_openai_model_configs():
    from model_configs import HF_MODEL_CONFIGS
    return {k: v for k, v in HF_MODEL_CONFIGS.items() if v.get('model_type') == 'openai'}

OPENAI_MODEL_CONFIGS = get_openai_model_configs()

HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
IS_SPACES = os.getenv('SPACE_ID') is not None

if IS_SPACES:
    DATASET_DIR = Path("/tmp/solver_cache")
else:
    DATASET_DIR = Path("solver_cache")

DATASET_DIR.mkdir(parents=True, exist_ok=True) 
global QUERY_ID
QUERY_ID = None

def save_query_data(query_id: str, query: str, image_path: str) -> None:
    """Save query data to Huggingface dataset"""
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
    
    if feedback_file.exists():
        with feedback_file.open("r") as f:
            existing_feedback = json.load(f)
            if not isinstance(existing_feedback, list):
                existing_feedback = [existing_feedback]
            existing_feedback.append(feedback_data)
            feedback_data = existing_feedback
    
    with feedback_file.open("w") as f:
        json.dump(feedback_data, f, indent=4)


def save_steps_data(query_id: str, memory: Memory) -> None:
    """Save steps data to Huggingface dataset"""
    steps_file = DATASET_DIR / query_id / "all_steps.json"

    memory_actions = memory.get_actions()
    memory_actions = make_json_serializable(memory_actions)
    print("Memory actions: ", memory_actions)

    with steps_file.open("w") as f:
        json.dump(memory_actions, f, indent=4, cls=CustomEncoder)

    
def save_module_data(query_id: str, key: str, value: Any) -> None:
    """Save module data to Huggingface dataset"""
    try:
        key = key.replace(" ", "_").lower()
        module_file = DATASET_DIR / query_id / f"{key}.json"
        value = make_json_serializable(value)
        with module_file.open("a") as f:
            json.dump(value, f, indent=4, cls=CustomEncoder)
    except Exception as e:
        print(f"Warning: Failed to save as JSON: {e}")
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


# Global cross-session cache: {fingerprint: {tool_name: [artifacts]}}
# This allows reusing tool results across different conversation sessions for the same image
_global_artifact_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


def make_artifact_key(tool_name: str, image_path: str, context: str = "", sub_goal: str = "", image_id: str = "") -> str:
    """Deterministic key for caching tool outputs tied to inputs (image-aware)."""
    hasher = hashlib.sha256()
    hasher.update(tool_name.encode())
    hasher.update(str(image_path or "").encode())
    hasher.update(str(context or "").encode())
    hasher.update(str(sub_goal or "").encode())
    hasher.update(str(image_id or "").encode())
    return hasher.hexdigest()


def make_fingerprint_based_key(tool_name: str, image_fingerprint: str) -> str:
    """
    Create a cache key based on image fingerprint for cross-session caching.
    Only uses tool_name and image_fingerprint to maximize cache hit rate.
    For the same image and tool, the result should be the same regardless of context/sub_goal.
    """
    hasher = hashlib.sha256()
    hasher.update(tool_name.encode())
    hasher.update(str(image_fingerprint or "").encode())
    return hasher.hexdigest()


def get_cached_artifact(state: "AgentState", group_name: str, tool_name: str, key: str, image_fingerprint: str = None):
    """
    Get cached artifact from current session or global cross-session cache.
    
    Args:
        state: Current AgentState
        group_name: Image group name
        tool_name: Tool name
        key: Artifact key (session-specific)
        image_fingerprint: Image fingerprint for cross-session lookup
    """
    # First, check current session cache
    group = state.image_groups.get(group_name, {})
    artifacts = group.get("artifacts", {}).get(tool_name, [])
    for art in artifacts:
        if art.get("key") == key:
            return art
    
    # If not found in current session and fingerprint is provided, check global cache
    if image_fingerprint:
        fingerprint_key = make_fingerprint_based_key(tool_name, image_fingerprint)
        global_artifacts = _global_artifact_cache.get(image_fingerprint, {}).get(tool_name, [])
        for art in global_artifacts:
            if art.get("fingerprint_key") == fingerprint_key:
                print(f"Found cached artifact in global cache for fingerprint {image_fingerprint[:8]}... and tool {tool_name}")
                return art
    
    return None


def store_artifact(state: "AgentState", group_name: str, tool_name: str, key: str, result: Any, image_fingerprint: str = None):
    """
    Store artifact in both current session and global cross-session cache.
    
    Args:
        state: Current AgentState
        group_name: Image group name
        tool_name: Tool name
        key: Artifact key (session-specific)
        result: Tool execution result
        image_fingerprint: Image fingerprint for cross-session caching
    """
    # Store in current session
    state.image_groups.setdefault(group_name, {"images": [], "features": [], "artifacts": {}})
    state.image_groups[group_name].setdefault("artifacts", {}).setdefault(tool_name, [])
    entry = {
        "key": key,
        "result": result,
        "created_at": time.time()
    }
    state.image_groups[group_name]["artifacts"][tool_name].append(entry)
    
    # Also store in global cross-session cache if fingerprint is provided
    if image_fingerprint:
        fingerprint_key = make_fingerprint_based_key(tool_name, image_fingerprint)
        if image_fingerprint not in _global_artifact_cache:
            _global_artifact_cache[image_fingerprint] = {}
        if tool_name not in _global_artifact_cache[image_fingerprint]:
            _global_artifact_cache[image_fingerprint][tool_name] = []
        
        existing = False
        for art in _global_artifact_cache[image_fingerprint][tool_name]:
            if art.get("fingerprint_key") == fingerprint_key:
                art["result"] = result
                art["created_at"] = time.time()
                existing = True
                break
        
        if not existing:
            global_entry = {
                "fingerprint_key": fingerprint_key,
                "result": result,
                "created_at": time.time(),
                "tool_name": tool_name
            }
            _global_artifact_cache[image_fingerprint][tool_name].append(global_entry)
            print(f"Stored artifact in global cache for fingerprint {image_fingerprint[:8]}... and tool {tool_name}")


def get_question_result(state: "AgentState", question: str) -> QuestionResult | None:
    for qr in state.question_results:
        if qr.question.strip() == str(question).strip():
            return qr
    return None


def record_question_result(state: "AgentState", question: str, status: str, final_answer: str = "", error: str = ""):
    existing = get_question_result(state, question)
    if existing:
        existing.status = status
        existing.final_answer = final_answer
        existing.error = error
        existing.created_at = time.time()
    else:
        state.question_results.append(QuestionResult(question=question, status=status, final_answer=final_answer, error=error))


def add_image_to_group(group_name: str, user_image, state: "AgentState", images_dir: Path, features_dir: Path) -> str:
    """Store uploaded image(s) into a session-level group and cache their features."""
    if not user_image:
        return "⚠️ No image provided."
    images = user_image if isinstance(user_image, list) else [user_image]
    group = group_name.strip() or "default"
    state.image_groups.setdefault(group, {"images": [], "features": [], "artifacts": {}})

    added = 0
    reused = 0
    group_image_dir = images_dir / group
    group_image_dir.mkdir(parents=True, exist_ok=True)

    for img in images:
        fingerprint = compute_image_fingerprint(img)
        already = False
        for entry in state.image_groups[group]["images"]:
            if entry.get("fingerprint") == fingerprint and fingerprint:
                reused += 1
                already = True
                state.image_context = ImageContext(
                    image_id=entry["image_id"],
                    image_path=entry["image_path"],
                    features_path=entry.get("features_path", ""),
                    fingerprint=fingerprint,
                    source_type="group"
                )
                break
        if already:
            continue

        image_id = uuid.uuid4().hex
        image_path = group_image_dir / f"{image_id}.jpg"
        try:
            if isinstance(img, dict) and 'path' in img:
                shutil.copy(img['path'], image_path)
            elif isinstance(img, str) and os.path.exists(img):
                shutil.copy(img, image_path)
            elif hasattr(img, "save"):
                img.save(image_path)
            else:
                raise ValueError(f"Unsupported image type: {type(img)}")
        except Exception as e:
            print(f"Error caching uploaded image: {e}")
            traceback.print_exc()
            continue

        feature_path = encode_image_features(str(image_path), features_dir / group)
        # Extract original image name from upload_path_map or image path
        image_name = None
        # Try to find in upload_path_map by matching paths
        if state.upload_path_map:
            img_source_path = None
            if isinstance(img, dict) and 'path' in img:
                img_source_path = img['path']
            elif isinstance(img, str):
                img_source_path = img
            
            if img_source_path:
                # Normalize paths for comparison
                img_source_path_norm = os.path.normpath(str(img_source_path))
                for orig_name, orig_path in state.upload_path_map.items():
                    orig_path_norm = os.path.normpath(str(orig_path))
                    if orig_path_norm == img_source_path_norm:
                        image_name = os.path.splitext(orig_name)[0]  # Remove extension
                        break
        
        # If not found in upload_path_map, extract from source path
        if not image_name:
            if isinstance(img, dict) and 'path' in img:
                image_name = os.path.splitext(os.path.basename(img['path']))[0]
            elif isinstance(img, str):
                image_name = os.path.splitext(os.path.basename(img))[0]
        
        # Keep original image_name (including spaces and special characters) for consistent file naming
        # The tools will handle the filename appropriately when saving files
        
        entry = {
            "image_id": image_id,
            "image_name": image_name,  # Store original image name for consistent naming
            "image_path": str(image_path),
            "fingerprint": fingerprint,
            "features_path": feature_path
        }
        state.image_groups[group]["images"].append(entry)
        if feature_path:
            state.image_groups[group]["features"].append(feature_path)
        added += 1
        state.image_context = ImageContext(
            image_id=image_id,
            image_path=str(image_path),
            features_path=feature_path,
            fingerprint=fingerprint,
            source_type="group"
        )

    state.last_group_name = group
    if added and reused:
        return f"✅ Added {added} new image(s) to group '{group}', reused {reused} existing."
    elif added:
        return f"✅ Added {added} new image(s) to group '{group}'."
    elif reused:
        return f"♻️ Reused {reused} existing image(s) in group '{group}'."
    return f"⚠️ No images added."


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
class QuestionResult:
    question: str
    status: str  # "SUCCESS" | "FAILED"
    final_answer: str = ""
    error: str = ""
    created_at: float = field(default_factory=time.time)


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
    question_results: List[QuestionResult] = field(default_factory=list)
    upload_path_map: Dict[str, str] = field(default_factory=dict)  # image_name -> full path
    session_id: str = field(default_factory=lambda: uuid.uuid4().hex)

def normalize_tool_name(tool_name: str, available_tools=None) -> str:
    """Normalize the tool name to match the available tools."""
    if available_tools is None:
        return tool_name
    
    # Strip any "No matched tool given: " prefix if present (handle recursive calls)
    clean_name = tool_name
    if "No matched tool given: " in tool_name:
        # Handle multiple nested prefixes
        while "No matched tool given: " in clean_name:
            clean_name = clean_name.split("No matched tool given: ")[-1].strip()
    
    # First try exact match (case-insensitive)
    for tool in available_tools:
        if tool.lower() == clean_name.lower():
            print(f"app.normalize_tool_name: Exact match found: '{tool_name}' -> '{tool}'")
            return tool
    
    # Then try partial match (tool name contained in the given string)
    for tool in available_tools:
        if tool.lower() in clean_name.lower() or clean_name.lower() in tool.lower():
            print(f"app.normalize_tool_name: Partial match found: '{tool_name}' -> '{tool}'")
            return tool
    
    # If still no match, return error with cleaned name
    print(f"app.normalize_tool_name: No match found for '{tool_name}' (cleaned: '{clean_name}'). Available tools: {available_tools[:5] if available_tools else 'None'}...")
    return "No matched tool given: " + clean_name


def _read_group_table_rows(group_table):
    """Extract rows from group_table (DataFrame or list format)."""
    import pandas as pd
    if isinstance(group_table, pd.DataFrame) and not group_table.empty:
        rows = []
        for row_values in group_table.values:
            image_name = str(row_values[0]).strip() if len(row_values) > 0 and row_values[0] is not None else ""
            group = str(row_values[1]).strip() if len(row_values) > 1 and row_values[1] is not None and str(row_values[1]).strip() else ""
            rows.append([image_name, group])
        return rows
    elif isinstance(group_table, list):
        return group_table
    else:
        return []


def _check_tool_execution_error(result, tool_name: str) -> tuple[bool, str]:
    """Check if tool execution failed and return (is_error, error_message)."""
    if result is None:
        return True, f"Tool '{tool_name}' execution returned None (no output)."
    elif isinstance(result, str):
        if result.startswith("Error") or result.startswith("error"):
            return True, result
        elif len(result.strip()) == 0:
            return True, f"Tool '{tool_name}' returned empty response."
        return False, ""  # Valid string response
    elif isinstance(result, dict) and "error" in result:
        error_msg = result.get("error", "Unknown error")
        # Include error_details if available for better debugging
        if "error_details" in result:
            error_details = result.get("error_details", {})
            details_str = ", ".join([f"{k}={v}" for k, v in error_details.items()])
            error_msg = f"{error_msg} (Details: {details_str})"
        return True, error_msg
    return False, ""  # No error detected


def _collect_visual_outputs(result, visual_outputs_list):
    """
    Collect visual outputs from tool result and add to visual_outputs_list.
    Unified collection logic: handles both single image and per_image structures.
    """
    visual_output_files = []
    seen_paths = set()  # Track seen paths to avoid duplicates
    
    def add_path(path):
        """Helper to add path if valid and not duplicate."""
        if path and path not in seen_paths and isinstance(path, str):
            visual_output_files.append(path)
            seen_paths.add(path)
    
    if isinstance(result, dict):
        # Handle per_image structure (multi-image results)
        if "per_image" in result:
            for img_result in result["per_image"]:
                if isinstance(img_result, dict):
                    # Collect from visual_outputs list
                    if "visual_outputs" in img_result:
                        for path in img_result["visual_outputs"]:
                            add_path(path)
                    # Collect from individual keys
                    for key in ["overlay_path", "mask_path", "processed_image_path", "output_path"]:
                        if key in img_result:
                            add_path(img_result[key])
        else:
            # Single image result: collect from top level
            if "visual_outputs" in result:
                for path in result["visual_outputs"]:
                    add_path(path)
            # Also check individual keys for single results
            for key in ["overlay_path", "mask_path", "processed_image_path", "output_path"]:
                if key in result:
                    add_path(result[key])
    
    # Process collected files
    for file_path in visual_output_files:
        try:
            # Don't skip comparison charts - they are important visualizations
            # Only skip if it's a duplicate comparison plot (e.g., from image_preprocessor)
            filename = os.path.basename(file_path).lower()
            if "comparison" in filename and "preprocessed" in filename:
                # Skip preprocessing comparison plots (we already show processed images)
                continue
                
            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                continue
            if not os.path.exists(file_path) or os.path.getsize(file_path) == 0:
                continue
                
            image = Image.open(file_path)
            if image.size[0] == 0 or image.size[1] == 0:
                continue
            
            if image.mode not in ['RGB', 'L', 'RGBA']:
                image = image.convert('RGB')
            
            img_array = np.array(image)
            if img_array.size == 0 or np.isnan(img_array).any():
                continue
            
            filename = os.path.basename(file_path)
            filename_lower = filename.lower()
            
            # Extract image identifier from filename (consistent naming: tool_type_imageid.ext)
            # Patterns: nuclei_overlay_<image_id>.png, nuclei_mask_<image_id>.png, <image_id>_default_processed.png
            image_id = None
            if "_overlay_" in filename_lower or "_mask_" in filename_lower:
                # Pattern: nuclei_overlay_<image_id>.png
                parts = filename.rsplit(".", 1)[0].split("_")
                if len(parts) >= 3:
                    image_id = "_".join(parts[2:])  # Everything after tool_type
            elif "_processed" in filename_lower:
                # Pattern: <image_id>_default_processed.png
                parts = filename.split("_")
                if len(parts) >= 2:
                    image_id = parts[0]
            
            # Format label with image identifier
            image_id_display = f" ({image_id[:8]}...)" if image_id and len(image_id) > 8 else (f" ({image_id})" if image_id else "")
            
            # Generate descriptive label
            if "overlay" in filename_lower:
                label = f"Segmentation Overlay{image_id_display}"
            elif "mask" in filename_lower and "viz" not in filename_lower:
                label = f"Segmentation Mask{image_id_display}"
            elif "processed" in filename_lower:
                label = f"Processed Image{image_id_display}"
            elif "comparison" in filename_lower or "bar" in filename_lower:
                label = f"Comparison Chart: {filename}"
            else:
                label = f"Analysis Result{image_id_display}" if image_id_display else f"Analysis Result: {filename}"
            
            visual_outputs_list.append((image, label))
        except Exception as e:
            print(f"Warning: Failed to load image {file_path}: {e}")
            continue


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
        self.visual_outputs_for_gradio = []

        self.output_types = output_types.lower().split(',')
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."

        self.step_times = []
        self.step_memory = []
        self.max_memory = 0
        self.step_costs = []
        self.total_cost = 0.0
        self.end_time = None
        self.step_info = []  # Store detailed information for each step
        self.model_config = self._get_model_config(planner.llm_engine_name)
        self.default_cost_per_token = self._get_default_cost_per_token()

    def _format_conversation_history(self) -> str:
        """Render conversation history into a plain-text transcript for prompts."""
        history = self.agent_state.conversation or []
        lines = []
        for msg in history:
            role = getattr(msg, "role", "assistant")
            content = getattr(msg, "content", "")
            lines.append(f"{role}: {content}")
        return "\n".join(lines)
    
    def _extract_result_summary(self, result: Any) -> str:
        """Extract a brief summary from tool execution result."""
        if result is None:
            return "No result"
        elif isinstance(result, str):
            return result[:200] + "..." if len(result) > 200 else result
        elif isinstance(result, dict):
            summary_parts = []
            if "summary" in result:
                summary_parts.append(f"Summary: {result['summary']}")
            if "cell_count" in result:
                summary_parts.append(f"Cells: {result['cell_count']}")
            if "processing_statistics" in result:
                stats = result["processing_statistics"]
                if isinstance(stats, dict):
                    if "final_brightness" in stats:
                        summary_parts.append(f"Brightness: {stats['final_brightness']}")
            if "visual_outputs" in result:
                summary_parts.append(f"Generated {len(result['visual_outputs'])} visual output(s)")
            if "per_image" in result:
                per_image = result["per_image"]
                if isinstance(per_image, list) and len(per_image) > 0:
                    image_summaries = []
                    for i, img_result in enumerate(per_image):
                        if isinstance(img_result, dict):
                            if "summary" in img_result:
                                image_summaries.append(f"Image {i+1}: {img_result['summary']}")
                            elif "cell_count" in img_result:
                                cell_count = img_result['cell_count']
                                image_summaries.append(f"Image {i+1}: {cell_count} cells")
                            elif "error" in img_result:
                                image_summaries.append(f"Image {i+1}: Error")
                    if image_summaries:
                        summary_parts.append(f"Processed {len(per_image)} image(s)")
            return "; ".join(summary_parts) if summary_parts else "Tool executed successfully"
        else:
            return str(result)[:200] + "..." if len(str(result)) > 200 else str(result)

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

    def stream_solve_user_problem(self, user_query: str, image_context: ImageContext, group_name: str, group_images: List[Dict[str, Any]], api_key: str, messages: List[ChatMessage]) -> Iterator:
        import time
        import os
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        visual_description = "*Ready to display analysis results and processed images.*"

        image_items = group_images or []
        img_path_for_tools = image_items[0]["image_path"] if image_items else (image_context.image_path if image_context else None)
        img_id = image_items[0]["image_id"] if image_items else (image_context.image_id if image_context else None)
        img_path_for_analysis = img_path_for_tools
        self.agent_state.image_context = image_context
        group_name = group_name or getattr(self.agent_state, "last_group_name", "")
        analysis_img_ref = img_path_for_tools

        session_dir = os.path.join(str(DATASET_DIR), self.agent_state.session_id)
        _tool_cache_dir = os.path.join(session_dir, "tool_cache")
        os.makedirs(_tool_cache_dir, exist_ok=True)
        print(f"Using persistent tool cache directory: {_tool_cache_dir} (session-based, won't be deleted)")
        self.executor.set_query_cache_dir(_tool_cache_dir)
        
        if image_context and img_path_for_tools:
            if len(image_items) > 1:
                groups_used = set(img.get("group", "") for img in image_items if img.get("group"))
                group_label = f" ({len(image_items)} images from groups: {', '.join(sorted(groups_used))})" if groups_used else f" ({len(image_items)} images)"
                messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}\n### 🖼️ Using session images{group_label}"))
            else:
                group_label = f" in group `{group_name}`" if group_name else ""
                messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}\n### 🖼️ Using session image `{img_id}`{group_label}"))
        else:
            messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}"))
        yield messages, "", [], visual_description, "**Progress**: Input received"

        step_count = 0
        json_data = {"query": user_query, "image_id": img_id}

        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### 🐙 Deep Thinking:"))
        yield messages, "", [], visual_description, "**Progress**: Starting analysis"
        query_analysis_start = time.time()
        try:
            conversation_text = self._format_conversation_history()
            query_analysis = self.planner.analyze_query(user_query, analysis_img_ref, conversation_text)
            query_analysis_end = time.time()
            query_analysis_time = query_analysis_end - query_analysis_start
            
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
            # Extract only Concise Summary section for display
            concise_summary = ""
            if "Concise Summary:" in query_analysis:
                # Find the start of Concise Summary
                start_idx = query_analysis.find("Concise Summary:")
                # Find the end (next section or end of string)
                end_markers = ["Required Skills:", "Relevant Tools:", "Additional Considerations:"]
                end_idx = len(query_analysis)
                for marker in end_markers:
                    marker_idx = query_analysis.find(marker, start_idx)
                    if marker_idx != -1 and marker_idx < end_idx:
                        end_idx = marker_idx
                # Extract the summary text (skip "Concise Summary:" label)
                summary_text = query_analysis[start_idx + len("Concise Summary:"):end_idx].strip()
                concise_summary = summary_text
            else:
                # Fallback: use original if pattern not found
                concise_summary = query_analysis
            
            # Keep full query_analysis for internal use, but display only concise summary
            messages.append(ChatMessage(role="assistant", 
                                        content=f"{concise_summary}",
                                        metadata={"title": "### 🔍 Step 0: Query Analysis"}))
            yield messages, concise_summary, self.visual_outputs_for_gradio, visual_description, "**Progress**: Query analysis completed"

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

        if image_items and len(image_items) > 0:
            for img_item in image_items:
                original_img_path = img_item.get("image_path")
                if original_img_path and os.path.exists(original_img_path):
                    try:
                        original_image = Image.open(original_img_path)
                        
                        supported_formats = ['PNG', 'JPEG', 'JPG', 'GIF', 'WEBP']
                        file_ext = os.path.splitext(original_img_path)[1].upper().lstrip('.')
                        
                        if original_image.mode not in ['RGB', 'L', 'RGBA']:
                            try:
                                original_image = original_image.convert('RGB')
                            except Exception as e:
                                print(f"Warning: Failed to convert image {original_img_path} to RGB: {e}")
                                continue
                        
                        if file_ext not in supported_formats:
                            print(f"Converting unsupported format {file_ext} to PNG for display")
                            from io import BytesIO
                            png_buffer = BytesIO()
                            if original_image.mode in ('RGBA', 'LA', 'P'):
                                rgb_image = Image.new('RGB', original_image.size, (255, 255, 255))
                                if original_image.mode == 'P':
                                    original_image = original_image.convert('RGBA')
                                rgb_image.paste(original_image, mask=original_image.split()[-1] if original_image.mode in ('RGBA', 'LA') else None)
                                original_image = rgb_image
                            elif original_image.mode != 'RGB':
                                original_image = original_image.convert('RGB')
                            original_image.save(png_buffer, format='PNG')
                            png_buffer.seek(0)
                            original_image = Image.open(png_buffer)
                        
                        if original_image.size[0] == 0 or original_image.size[1] == 0:
                            print(f"Warning: Invalid image size: {original_img_path}")
                            continue
                        
                        try:
                            img_array = np.array(original_image)
                            if img_array.size == 0 or np.isnan(img_array).any():
                                print(f"Warning: Invalid image data in {original_img_path}")
                                continue
                        except Exception as e:
                            print(f"Warning: Failed to validate image data for {original_img_path}: {e}")
                            continue
                        
                        filename = os.path.basename(original_img_path)
                        label = f"Original Image: {filename}"
                        self.visual_outputs_for_gradio.insert(0, (original_image, label))
                        print(f"Successfully added original image to visual outputs: {filename}")
                        
                    except Exception as e:
                        print(f"Warning: Failed to load original image {original_img_path} for display. Error: {e}")
                        import traceback
                        print(f"Full traceback: {traceback.format_exc()}")
                        continue
        
        # Update visual description after adding original images
        if self.visual_outputs_for_gradio:
            visual_description = f"*Displaying {len(self.visual_outputs_for_gradio)} image(s), including original and processed results.*"
        
        # Execution loop (similar to your step-by-step solver)
        execution_successful = False
        tool_execution_failed = False
        failed_tool_names = []
        successful_steps = []  # Track successful steps: [{"step": N, "tool": "ToolName", "result": {...}}]
        successful_tools = set()  # Track successfully executed tools
        
        while step_count < self.max_steps:  # Removed time limit check
            step_count += 1
            step_start = time.time()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            
            # Simple progress message without progress bar
            progress_msg = f"**Progress**: Step {step_count}/{self.max_steps}"
            
            messages.append(ChatMessage(role="OctoTools", 
                                        content=f"Generating the {step_count}-th step...",
                                        metadata={"title": f"🔄 Step {step_count}"}))
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg

            conversation_text = self._format_conversation_history()
            next_step = self.planner.generate_next_step(user_query, analysis_img_ref, query_analysis, self.memory, step_count, self.max_steps, conversation_context=conversation_text)
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
            context = context or self.agent_state.last_context or ""
            sub_goal = sub_goal or self.agent_state.last_sub_goal or ""
            step_data = {"step_count": step_count, "context": context, "sub_goal": sub_goal, "tool_name": tool_name, "time": round(time.time() - self.start_time, 5)}
            save_module_data(QUERY_ID, f"step_{step_count}_action_prediction", step_data)

            # Tool name normalization is already done in Planner.extract_context_subgoal_and_tool()
            # which calls ResponseParser.parse_next_step() that normalizes the tool name
            # No need for additional normalization here

            # Display the step information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Context:** {context}\n\n**Sub-goal:** {sub_goal}\n\n**Tool:** `{tool_name}`",
                metadata={"title": f"### 🎯 Step {step_count}: Action Prediction ({tool_name})"}))
            
            # Simple progress message
            progress_msg_predicted = f"**Progress**: Step {step_count}/{self.max_steps}"
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg_predicted

            # Handle tool execution or errors
            # Check if tool_name contains error prefix (normalization failed)
            if "No matched tool given: " in tool_name:
                # Extract the actual tool name from error message
                actual_tool_name = tool_name.split("No matched tool given: ")[-1].strip()
                # Handle nested prefixes
                while "No matched tool given: " in actual_tool_name:
                    actual_tool_name = actual_tool_name.split("No matched tool given: ")[-1].strip()
                error_msg = f"Tool '{actual_tool_name}' could not be matched to any available tool. Available tools: {self.planner.available_tools[:10]}..."
                tool_execution_failed = True
                if actual_tool_name not in failed_tool_names:
                    failed_tool_names.append(actual_tool_name)
                messages.append(ChatMessage(
                    role="assistant", 
                    content=f"⚠️ **Tool Matching Error:** {error_msg}\n\nThis may indicate that the tool is not properly registered or loaded."))
                
                # Simple progress message for error
                progress_msg_error = f"**Progress**: Step {step_count}/{self.max_steps} ⚠️"
                yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg_error
                continue
            
            if tool_name not in self.planner.available_tools:
                tool_execution_failed = True
                if tool_name not in failed_tool_names:
                    failed_tool_names.append(tool_name)
                messages.append(ChatMessage(
                    role="assistant", 
                    content=f"⚠️ Error: Tool '{tool_name}' is not available."))
                
                # Simple progress message for tool not available
                progress_msg_unavailable = f"**Progress**: Step {step_count}/{self.max_steps} ⚠️"
                yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg_unavailable
                continue

            results_per_image = []
            tool_command = None  # Initialize tool_command outside loop
            analysis = ""  # Initialize analysis, explanation, command for display
            explanation = ""
            command = ""
            
            # Tools that merge all images should only execute once, not per-image
            # These tools analyze all images together, so result is always a single merged result
            tools_that_merge_all_images = ["Cell_State_Analyzer_Tool", "Analysis_Visualizer_Tool"]
            should_execute_once = tool_name in tools_that_merge_all_images and len(image_items) > 1
            
            merged_result = None  # Store merged result for tools that combine all images
            
            for img_idx, img_item in enumerate(image_items):
                safe_path = img_item["image_path"].replace("\\", "\\\\") if img_item.get("image_path") else None
                conversation_text = self._format_conversation_history()
                artifact_key = make_artifact_key(tool_name, safe_path, context, sub_goal, image_id=img_item.get("image_id"))
                
                # Get image fingerprint for cross-session caching
                image_fingerprint = img_item.get("fingerprint") or (image_context.fingerprint if image_context else None)
                
                # For tools that merge all images: only execute on first image, reuse result for others
                if should_execute_once and img_idx > 0:
                    # Reuse the merged result from first image
                    if merged_result is not None:
                        result = merged_result
                        messages.append(ChatMessage(
                            role="assistant",
                            content=f"♻️ Reusing merged analysis result for image `{img_item.get('image_id')}` (all images analyzed together).",
                            metadata={"title": f"### 🛠️ Step {step_count}: Merged Analysis ({tool_name})"}
                        ))
                        print(f"Reusing merged result for {tool_name} (image {img_item.get('image_id')}, index {img_idx})")
                        results_per_image.append(result)
                        continue  # Skip to next image
                    else:
                        # If no merged result yet, continue to execute (shouldn't happen, but safety check)
                        pass
                
                # Check cache (both session and global cross-session)
                cached_artifact = get_cached_artifact(self.agent_state, group_name, tool_name, artifact_key, image_fingerprint)

                if cached_artifact:
                    result = cached_artifact.get("result")
                    analysis = "Cached result reused"
                    explanation = "Found matching artifact in session or previous conversation; skipping execution."
                    command = "execution = 'cached_artifact'"
                    cache_source = "previous conversation" if image_fingerprint and cached_artifact.get("fingerprint_key") else "current session"
                    
                    # Create a ToolCommand object for memory consistency
                    from octotools.models.formatters import ToolCommand
                    if tool_command is None:  # Only create once if multiple images
                        tool_command = ToolCommand(
                            analysis=analysis,
                            explanation=explanation,
                            command=command
                        )
                    
                    messages.append(ChatMessage(
                        role="assistant",
                        content=f"♻️ Reusing cached {tool_name} result for image `{img_item.get('image_id')}` (from {cache_source}).",
                        metadata={"title": f"### 🛠️ Step {step_count}: Cached Execution ({tool_name})"}
                    ))
                    print(f"Reused cached artifact for {tool_name} (key={artifact_key}, source={cache_source})")
                    
                    # Store merged result for tools that combine all images
                    if should_execute_once and merged_result is None:
                        merged_result = result
                else:
                    # Pass image_id and image_name for consistent file naming
                    image_id = img_item.get("image_id")
                    image_name = img_item.get("image_name")  # Use original image name if available
                    # Prefer image_name over image_id for file naming (more readable)
                    identifier_for_naming = image_name if image_name else image_id
                    tool_command = self.executor.generate_tool_command(
                        user_query, safe_path, context, sub_goal, 
                        self.planner.toolbox_metadata[tool_name], self.memory, 
                        conversation_context=conversation_text, image_id=identifier_for_naming
                    )
                    analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
                    result = self.executor.execute_tool_command(tool_name, command)
                    result = make_json_serializable(result)
                    
                    # Check if tool execution failed
                    execution_failed, error_msg = _check_tool_execution_error(result, tool_name)
                    
                    if execution_failed:
                        tool_execution_failed = True
                        if tool_name not in failed_tool_names:
                            failed_tool_names.append(tool_name)
                        messages.append(ChatMessage(
                            role="assistant",
                            content=f"⚠️ **Tool Execution Failed:** {error_msg}\n\n**Tool:** `{tool_name}`\n**Command:**\n```python\n{command}\n```",
                            metadata={"title": f"### ❌ Step {step_count}: Tool Execution Failed ({tool_name})"}
                        ))
                        
                        # Simple progress message for execution failure
                        progress_msg_failed = f"**Progress**: Step {step_count}/{self.max_steps} ❌"
                        yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg_failed
                        # Store the error result
                        store_artifact(self.agent_state, group_name, tool_name, artifact_key, {"error": error_msg, "result": None}, image_fingerprint)
                        result = {"error": error_msg, "result": None}
                    else:
                        store_artifact(self.agent_state, group_name, tool_name, artifact_key, result, image_fingerprint)
                        print(f"Tool '{tool_name}' result for image {img_item.get('image_id')}: {result}")
                        # Track successful tool execution
                        if tool_name not in successful_tools:
                            successful_tools.add(tool_name)
                        
                        # Store merged result for tools that combine all images
                        if should_execute_once and merged_result is None:
                            merged_result = result
                    
                results_per_image.append(result)

            # For downstream steps, if only one image use its result, else aggregate
            # EXCEPTION: Tools that merge all images always return a single merged result (not per_image structure)
            if should_execute_once and len(results_per_image) > 0:
                # Use the merged result directly (from first execution, reused for all images)
                result = results_per_image[0]
            elif len(results_per_image) == 1:
                result = results_per_image[0]
            else:
                result = {"per_image": results_per_image}
            
            # Generate dynamic visual description based on tool and results
            visual_description = self.generate_visual_description(tool_name, result, self.visual_outputs_for_gradio)
            
            # Collect and add visual outputs
            _collect_visual_outputs(result, self.visual_outputs_for_gradio)
            
            # Track successful step (if not failed in this iteration)
            step_has_error = any(
                isinstance(r, dict) and ("error" in r or r.get("result") is None) 
                for r in results_per_image
            ) if results_per_image else False
            
            if not step_has_error:
                step_summary = {
                    "step": step_count,
                    "tool": tool_name,
                    "sub_goal": sub_goal,
                    "result_summary": self._extract_result_summary(result)
                }
                successful_steps.append(step_summary)

            # Display the command generation information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Analysis:** {analysis}\n\n**Explanation:** {explanation}\n\n**Command:**\n```python\n{command}\n```",
                metadata={"title": f"### 📝 Step {step_count}: Command Generation ({tool_name})"}))
            
            # Simple progress message for command generation
            progress_msg_command = f"**Progress**: Step {step_count}/{self.max_steps}"
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg_command

            # Save the command generation data
            command_generation_data = {
                "analysis": analysis,
                "explanation": explanation,
                "command": command,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_generation", command_generation_data)
            
            # Display the command execution result
            # Include training logs if available (for Cell_State_Analyzer_Tool)
            result_content = ""
            if isinstance(result, dict):
                if "training_logs" in result and result["training_logs"]:
                    # Show training logs prominently
                    result_content += f"**Training Progress:**\n```\n{result['training_logs']}\n```\n\n"
                if "summary" in result and result["summary"]:
                    result_content += f"**Summary:**\n{result['summary']}\n\n"
                # Show full result in JSON for other details
                result_content += f"**Full Result:**\n```json\n{json.dumps(make_json_serializable(result), indent=4)}\n```"
            else:
                result_content = f"**Result:**\n```json\n{json.dumps(make_json_serializable(result), indent=4)}\n```"
            
            messages.append(ChatMessage(
                role="assistant",
                content=result_content,
                metadata={"title": f"### 🛠️ Step {step_count}: Command Execution ({tool_name})"}))
            
            # Simple progress message for command execution
            progress_msg_executed = f"**Progress**: Step {step_count}/{self.max_steps}"
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg_executed

            # Save the command execution data
            command_execution_data = {
                "result": result,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_execution", command_execution_data)

            if tool_command is None:
                # Fallback: create a minimal ToolCommand if somehow not set
                from octotools.models.formatters import ToolCommand
                tool_command = ToolCommand(
                    analysis="Tool command not available",
                    explanation="Command was not generated (possibly all cached)",
                    command="execution = 'unknown'"
                )
            self.memory.add_action(step_count, tool_name, sub_goal, tool_command, result)
            
            # Check if tool execution failed - if same tool failed multiple times, stop to avoid infinite loop
            recent_actions = self.memory.get_actions()[-3:]  # Check last 3 actions
            same_tool_failures = 0
            for action in recent_actions:
                if action.get('tool_name') == tool_name:
                    action_result = action.get('result')
                    # Check various failure conditions
                    if action_result is None:
                        same_tool_failures += 1
                    elif isinstance(action_result, dict):
                        if action_result.get('error') or action_result.get('result') is None:
                            same_tool_failures += 1
                    elif isinstance(action_result, str) and action_result.startswith("Error"):
                        same_tool_failures += 1
            
            if same_tool_failures >= 2:
                messages.append(ChatMessage(
                    role="assistant",
                    content=f"⚠️ **Stopping execution:** Tool '{tool_name}' has failed {same_tool_failures} times consecutively. Stopping to avoid infinite loop.\n\n**Possible reasons:**\n- Tool parameters may be incorrect\n- Required dependencies may be missing\n- Image format may not be supported\n- Tool may have internal errors",
                    metadata={"title": f"### 🛑 Step {step_count}: Stopping due to repeated failures"}
                ))
                
                # Simple progress message for stopping
                progress_msg_stop = f"**Progress**: Step {step_count}/{self.max_steps} 🛑"
                yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg_stop
                execution_successful = False
                break
            
            conversation_text = self._format_conversation_history()
            stop_verification = self.planner.verificate_memory(user_query, analysis_img_ref, query_analysis, self.memory, conversation_context=conversation_text)
            context_verification, conclusion = self.planner.extract_conclusion(stop_verification)

            # Calculate step duration before using it in progress messages
            step_end = time.time()
            step_duration = step_end - step_start

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
            
            # Simple progress message after context verification
            progress_msg_verified = f"**Progress**: Step {step_count}/{self.max_steps} ✓"
            yield messages, query_analysis, self.visual_outputs_for_gradio, visual_description, progress_msg_verified

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
            
            # step_duration already calculated above, now update tracking
            self.step_times.append(step_duration)

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
                execution_successful = True
                break

        self.end_time = time.time()
        
        # Determine if execution was successful
        # Success means we stopped because conclusion == 'STOP', not because we ran out of steps or had tool errors
        failure_reason = ""
        if not execution_successful:
            # Check why we exited the loop
            if step_count >= self.max_steps:
                failure_reason = f"Reached maximum step limit ({self.max_steps} steps) without completing the query."
            else:
                failure_reason = "Execution stopped unexpectedly."
            
            if tool_execution_failed:
                failure_reason += f"\n\nAdditionally, {len(failed_tool_names)} tool(s) failed to execute: {', '.join(failed_tool_names)}"

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
            
            conclusion = f"🐙 **Conclusion:**\n{conclusion_text}\n\n---\n"
            conclusion += f"**📊 Detailed Performance Statistics**\n\n"
            conclusion += f"**Step-by-Step Analysis:**\n"
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
            # Record success for this question
            record_question_result(self.agent_state, user_query, status="SUCCESS", final_answer=final_answer)

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
            # Record success for this question if not already recorded
            if not get_question_result(self.agent_state, user_query):
                record_question_result(self.agent_state, user_query, status="SUCCESS", final_answer=str(final_output))

        messages.append(ChatMessage(role="assistant", content="<br>"))
        if execution_successful:
            messages.append(ChatMessage(role="assistant", content="### ✅ Query Solved!"))
            # Use the final answer if available, otherwise use a default message
            completion_text = final_answer if final_answer else "Analysis completed successfully"
            yield messages, completion_text, self.visual_outputs_for_gradio, visual_description, "**Progress**: Analysis completed!"
        else:
            # Execution failed - provide comprehensive summary including successful parts
            error_message = "### ⚠️ Partial Execution Summary\n\n"
            
            # Summary of successful parts
            if successful_steps:
                error_message += "## ✅ Successfully Completed Steps:\n\n"
                for step_info in successful_steps:
                    error_message += f"**Step {step_info['step']}: {step_info['tool']}**\n"
                    error_message += f"- Sub-goal: {step_info['sub_goal']}\n"
                    error_message += f"- Result: {step_info['result_summary']}\n\n"
            
            if successful_tools:
                error_message += f"## ✅ Successfully Executed Tools: {', '.join(sorted(successful_tools))}\n\n"
            
            if self.visual_outputs_for_gradio:
                error_message += f"## 🖼️ Generated Visual Outputs: {len(self.visual_outputs_for_gradio)} image(s)\n\n"
            
            # Failure information
            error_message += "## ❌ Execution Issues:\n\n"
            error_message += f"**Reason:** {failure_reason}\n\n"
            
            if tool_execution_failed:
                error_message += f"**Failed Tools:** {', '.join(failed_tool_names)}\n\n"
            
            # Partial results guidance
            if successful_steps:
                error_message += "## 💡 Partial Results Available:\n\n"
                error_message += "Although the query was not fully completed, the following results are available:\n"
                error_message += "- Review the visual outputs above for processed images\n"
                error_message += "- Check the successful steps for intermediate results\n"
                error_message += "- You can use these partial results or retry with adjusted parameters\n\n"
            
            # Solutions
            error_message += "## 🔧 Possible Solutions:\n"
            error_message += "1. Check if all required tools are available and properly configured\n"
            if tool_execution_failed:
                error_message += "2. Verify the tool names and ensure they match the available tools list\n"
            if step_count >= self.max_steps:
                error_message += f"3. Try increasing the maximum step limit (currently {self.max_steps} steps)\n"
                error_message += "4. Break down your query into smaller, more specific tasks\n"
            error_message += "5. Review the error messages above for specific tool execution issues\n"
            error_message += "6. Ensure all required input data (images, files, etc.) are properly provided\n"
            
            messages.append(ChatMessage(role="assistant", content=error_message))
            # Record failure for this question if not already recorded
            if not get_question_result(self.agent_state, user_query):
                record_question_result(self.agent_state, user_query, status="FAILED", final_answer=error_message)
            yield messages, error_message, self.visual_outputs_for_gradio, visual_description, f"**Progress**: Partial execution - {len(successful_steps)} step(s) completed"

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
            "Cell_Segmenter_Tool": f"*Showing {counts['segmented']} segmentation result(s) with identified cell regions in phase-contrast images.*",
            "Organoid_Segmenter_Tool": f"*Showing {counts['segmented']} segmentation result(s) with identified organoid regions.*",
            "Single_Cell_Cropper_Tool": f"*Displaying {counts['cropped']} single-cell crop(s) generated from segmentation results (nuclei, cells, or organoids).*",
            "Cell_State_Analyzer_Tool": f"*Displaying cell state analysis results from self-supervised learning with UMAP visualizations and clustering.*",
            "Cell_Morphology_Analyzer_Tool": "*Displaying cell morphology analysis results with detailed structural insights.*",
            "Fibroblast_Activation_Detector_Tool": "*Showing fibroblast activation state analysis with morphological indicators.*",
            "Cell_State_Analyzer_Tool": f"*Displaying {counts['analyzed']} cell state analysis result(s) with cell state distributions and statistics.*"
        }
        
        # Return tool-specific description or generic one
        if tool_name in tool_descriptions:
            return tool_descriptions[tool_name]
        else:
            total_images = len(visual_outputs)
            return f"*Displaying {total_images} analysis result(s) from {tool_name}.*"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the OctoTools demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-5-mini", help="LLM engine name.")
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

    parser.add_argument("--run_baseline_only", type=bool, default=False, help="Run only the baseline (no toolbox).")
    parser.add_argument("--openai_api_source", default="we_provided", choices=["we_provided", "user_provided"], help="Source of OpenAI API key.")
    return parser.parse_args()


def prepare_group_assignment(uploaded_files, conversation_state):
    """Prepare per-file group assignment table after upload and track path mapping."""
    state: AgentState = conversation_state if isinstance(conversation_state, AgentState) else AgentState()
    state.upload_path_map = {}
    if not uploaded_files:
        return gr.update(value=None, visible=True), "**Upload Status**: No files uploaded.", state
    files = uploaded_files if isinstance(uploaded_files, list) else [uploaded_files]
    if len(files) == 1:
        f = files[0]
        path = f if isinstance(f, str) else f.get("name", "")
        image_name = os.path.basename(path)
        state.upload_path_map[image_name] = path
        # Show table even for single image so user can edit group
        return gr.update(value=[[image_name, "default"]], visible=True), "**Upload Status**: Single image detected. You can edit the group if needed.", state
    rows = []
    for f in files:
        path = f if isinstance(f, str) else f.get("name", "")
        image_name = os.path.basename(path)
        state.upload_path_map[image_name] = path
        rows.append([image_name, ""])
    return gr.update(value=rows, visible=True), "**Upload Status**: Multiple images detected. Please assign a group per image.", state


def upload_image_to_group(user_image, group_table, conversation_state):
    """Gradio handler: add image(s) to group(s) and cache features."""
    state: AgentState = conversation_state if isinstance(conversation_state, AgentState) else AgentState()
    _, images_dir, features_dir = ensure_session_dirs(state.session_id)

    files = user_image if isinstance(user_image, list) else [user_image] if user_image else []
    if not files:
        return state, "**Progress**: ⚠️ No image provided.", gr.update()
    
    # Build upload_path_map from user_image directly (don't rely on state)
    # This ensures we always have the correct mapping even if state is out of sync
    upload_path_map = {}
    for f in files:
        path = f if isinstance(f, str) else f.get("name", "")
        if not path:
            continue  # Skip invalid paths
        image_name = os.path.basename(path)
        if image_name:  # Only add if image_name is not empty
            upload_path_map[image_name] = path
    
    # Update state's upload_path_map if it's empty or incomplete
    if not state.upload_path_map or len(state.upload_path_map) < len(upload_path_map):
        state.upload_path_map = upload_path_map

    # If single image, check if group_table has a group assignment
    if len(files) == 1:
        # Check if group_table exists and has a group value
        import pandas as pd
        if group_table is not None:
            if isinstance(group_table, pd.DataFrame) and not group_table.empty:
                # Extract group from the table
                group = str(group_table.iloc[0, 1]).strip() if len(group_table.columns) > 1 and len(group_table) > 0 else "default"
            elif isinstance(group_table, list) and len(group_table) > 0 and len(group_table[0]) > 1:
                # Extract group from list format
                group = str(group_table[0][1]).strip() if group_table[0][1] else "default"
            else:
                group = "default"
        else:
            group = "default"
        status = add_image_to_group(group or "default", files[0], state, images_dir, features_dir)
        if status.startswith("✅") or status.startswith("♻️"):
            import pandas as pd
            if group_table is not None and isinstance(group_table, pd.DataFrame) and not group_table.empty:
                return state, f"**Progress**: {status}", gr.update(value=group_table.values.tolist(), visible=True)
        return state, f"**Progress**: {status}", gr.update()

    # Multiple images: require group_table rows [filepath, group]
    import pandas as pd
    if group_table is None:
        return state, "**Progress**: ⚠️ Multiple images uploaded. Please assign a group per image before adding.", gr.update()
    
    # Convert group_table to proper format
    import pandas as pd
    if isinstance(group_table, pd.DataFrame) and group_table.empty:
        return state, "**Progress**: ⚠️ Multiple images uploaded. Please assign a group per image before adding.", gr.update()
    
    rows = _read_group_table_rows(group_table)
    
    added_msgs = []
    # Find first valid group as default (but don't use for last row)
    default_group = "default"
    for row in rows:
        if len(row) >= 2 and row[1]:
            default_group = str(row[1]).strip()
            break
    
    # Process twice to handle Gradio DataFrame sync issues with last cell
    # add_image_to_group has deduplication, so repeated calls are safe
    processed_images = set()  # Track which images have been successfully processed
    
    for attempt in range(2):
        if attempt > 0:
            # Second attempt: re-read group_table values in case they were updated
            rows = _read_group_table_rows(group_table)
        
        updated_rows = []
        
        for i, row in enumerate(rows):
            if len(row) < 2:
                updated_rows.append(row)
                continue
            image_name = str(row[0]).strip() if row[0] else ""
            original_group = str(row[1]).strip() if row[1] else ""
            is_last_row = (i == len(rows) - 1)
            
            # Special handling for last row: if empty, skip on first attempt, use second attempt value
            # For non-last rows, use default_group if empty
            if is_last_row and not original_group and attempt == 0:
                # Last row is empty on first attempt - skip it, will try again on second attempt
                updated_rows.append(row)
                continue
            
            # Use original_group if available, otherwise default_group (but not for last row on first attempt)
            group = original_group if original_group else (default_group if not is_last_row else "default")
            
            if not image_name:
                updated_rows.append(row)
                continue
            
            # Skip if already successfully processed in first attempt
            if image_name in processed_images:
                updated_rows.append([image_name, original_group if original_group else group])
                continue
            
            # Use the local upload_path_map (built from user_image) instead of state
            full_path = upload_path_map.get(image_name)
            if not full_path:
                # Try case-insensitive match
                for key, path in upload_path_map.items():
                    if key.lower() == image_name.lower():
                        full_path = path
                        break
            
            if not full_path:
                if attempt == 0:  # Only show error on first attempt
                    added_msgs.append(f"⚠️ Skipped file '{image_name}' because original path not found.")
                updated_rows.append(row)
                continue
            
            try:
                # For last row, skip if group is still "default" (meaning it was empty)
                if is_last_row and group == "default" and not original_group:
                    if attempt == 1:  # Second attempt and still empty
                        added_msgs.append(f"⚠️ Last image '{image_name}' has no group assigned, skipped.")
                    updated_rows.append(row)
                    continue
                
                status = add_image_to_group(group, full_path, state, images_dir, features_dir)
                if attempt == 0 or image_name not in processed_images:
                    added_msgs.append(status)
                
                # Check if addition was successful (status starts with ✅ or ♻️)
                if status.startswith("✅") or status.startswith("♻️"):
                    processed_images.add(image_name)
                    updated_rows.append([image_name, original_group if original_group else group])
                else:
                    updated_rows.append(row)
            except Exception as e:
                error_msg = f"⚠️ Error processing '{image_name}': {str(e)}"
                if attempt == 0:  # Only show error on first attempt
                    print(f"Error processing image '{image_name}': {e}")
                    traceback.print_exc()
                    added_msgs.append(error_msg)
                updated_rows.append(row)
    
    # Update group_table
    import pandas as pd
    updated_table = pd.DataFrame(updated_rows, columns=["image_name", "group"]) if updated_rows else None
    
    progress = "**Progress**:\n" + "\n".join(f"- {m}" for m in added_msgs) if added_msgs else "**Progress**: ⚠️ No images processed."
    
    if updated_table is not None and not updated_table.empty:
        return state, progress, gr.update(value=updated_table.values.tolist(), visible=True)
    elif rows:
        return state, progress, gr.update(value=rows, visible=True)
    else:
        return state, progress, gr.update()


def solve_problem_gradio(user_query, llm_model_engine=None, conversation_history=None):
    """
    Solve a problem using the Gradio interface with optional visualization clearing.
    
    Args:
        user_query: The user's query
        llm_model_engine: Language model engine (model_id from dropdown)
        conversation_history: Persistent chat history to keep context across runs
    """
    # Initialize or reuse persistent agent state
    state: AgentState = conversation_history if isinstance(conversation_history, AgentState) else AgentState()
    state.conversation = list(state.conversation)
    session_dir, images_dir, features_dir = ensure_session_dirs(state.session_id)

    # Start with prior conversation so the session feels continuous
    # Ensure messages is a list of ChatMessage objects
    messages: List[ChatMessage] = []
    if state.conversation:
        # Convert state.conversation to list of ChatMessage objects if needed
        for msg in state.conversation:
            if isinstance(msg, ChatMessage):
                messages.append(msg)
            elif isinstance(msg, dict):
                messages.append(ChatMessage(role=msg.get('role', 'assistant'), content=msg.get('content', '')))
            else:
                # Fallback: try to create ChatMessage from object
                messages.append(ChatMessage(role=getattr(msg, 'role', 'assistant'), content=str(getattr(msg, 'content', ''))))
    
    # Immediately add and display user query
    if user_query:
        user_msg = ChatMessage(role="user", content=str(user_query))
        messages.append(user_msg)
        # Immediately yield to show user message in conversation
        yield messages, "", [], "**Progress**: Processing your question...", state

    # Short-circuit if we already answered this question successfully
    cached_qr = get_question_result(state, user_query) if user_query else None
    if cached_qr and cached_qr.status == "SUCCESS":
        reuse_msg = f"♻️ Previously answered this question. Reusing stored result."
        answer_msg = cached_qr.final_answer or "Stored result (no answer text)."
        new_history = messages + [ChatMessage(role="assistant", content=reuse_msg), ChatMessage(role="assistant", content=answer_msg)]
        state.conversation = new_history
        yield new_history, cached_qr.final_answer, [], "**Progress**: Reused prior answer", state
        return
    
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
    
    # Check API key early and return error message if missing
    if api_key is None or api_key.strip() == "":
        print("⚠️ OPENAI_API_KEY not found in environment variables")
        api_key_error_msg = """⚠️ **API Key Configuration Required**

To use this application, you need to set up your OpenAI API key as an environment variable:

**Environment Variable Setup:**
Set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

For Windows PowerShell:
```powershell
$env:OPENAI_API_KEY="your-api-key-here"
```

For Hugging Face Spaces, add this as a secret in your Space settings.

For more information about obtaining an OpenAI API key, visit: https://platform.openai.com/api-keys
"""
        new_history = messages + [ChatMessage(role="assistant", content=api_key_error_msg)]
        state.conversation = new_history
        yield new_history, "", [], "**Progress**: API Key Required", state
        return
    
    # Get available tools dynamically
    enabled_tools = get_available_tools()

    # Generate a unique query ID
    query_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8] # e.g, 20250217_062225_612f2474
    print(f"Query ID: {query_id}")

    global QUERY_ID
    QUERY_ID = query_id
    os.makedirs(DATASET_DIR / query_id, exist_ok=True)
    if IS_SPACES:
        output_viz_dir = "/tmp/output_visualizations"
    else:
        output_viz_dir = os.path.join(os.getcwd(), 'output_visualizations')
    os.makedirs(output_viz_dir, exist_ok=True)
    print(f"✅ Output directory: {output_viz_dir}")

    query_cache_dir = os.path.join(str(session_dir), query_id)
    os.makedirs(query_cache_dir, exist_ok=True)
    
    # Debug: Print enabled_tools
    print(f"Debug - enabled_tools: {enabled_tools}")
    print(f"Debug - type of enabled_tools: {type(enabled_tools)}")
    
    # Ensure enabled_tools is a list and not empty
    if not enabled_tools:
        print("⚠️ No tools selected in UI, defaulting to all available tools.")
        enabled_tools = get_available_tools()
    elif isinstance(enabled_tools, str):
        enabled_tools = [enabled_tools]
    elif not isinstance(enabled_tools, list):
        enabled_tools = list(enabled_tools) if hasattr(enabled_tools, '__iter__') else []

    if not enabled_tools:
        print("❌ Critical Error: Could not determine a default tool list. Using Generalist_Solution_Generator_Tool as a last resort.")
        enabled_tools = ["Generalist_Solution_Generator_Tool"]

    # Create octotools components (Initializer → Planner, Memory, Executor)
    # Dependency chain: Initializer provides metadata → Planner uses it → Executor uses Initializer for tool loading
    try:
        initializer = Initializer(
            enabled_tools=enabled_tools,
            model_string=model_name_for_octotools,
            api_key=api_key
        )
        
        # Get toolbox metadata with lazy loading (only loads when first accessed)
        toolbox_metadata = initializer.get_toolbox_metadata()
        planner = Planner(
            llm_engine_name=model_name_for_octotools,
            toolbox_metadata=toolbox_metadata,
            available_tools=initializer.available_tools,
            api_key=api_key
        )
        
        memory = Memory()
        
        session_tool_cache_dir = os.path.join(str(session_dir), "tool_cache")
        os.makedirs(session_tool_cache_dir, exist_ok=True)
        executor = Executor(
            llm_engine_name=model_name_for_octotools,
            query_cache_dir=session_tool_cache_dir,
            enable_signal=False,
            api_key=api_key,
            initializer=initializer
        )
    except Exception as e:
        print(f"Error creating octotools components: {e}")
        import traceback
        traceback.print_exc()
        new_history = messages + [ChatMessage(role="assistant", content=f"⚠️ Error: Failed to initialize components. {str(e)}")]
        state.conversation = new_history
        yield new_history, "", [], "**Progress**: Error occurred", state
        return

    # Collect all images from all groups for analysis (let planner decide which to use)
    if len(state.image_groups) == 0:
        prompt_msg = "⚠️ Please upload an image into a group before asking a question."
        new_history = messages + [ChatMessage(role="assistant", content=prompt_msg)]
        state.conversation = new_history
        yield new_history, "", [], "**Progress**: Waiting for image upload", state
        return
    
    # Collect all images from all groups with group metadata
    all_group_images = []
    for gname, gentry in state.image_groups.items():
        for img in gentry.get("images", []):
            img_with_group = img.copy()
            img_with_group["group"] = gname
            all_group_images.append(img_with_group)
    
    if not all_group_images:
        prompt_msg = "⚠️ No images found in any group."
        new_history = messages + [ChatMessage(role="assistant", content=prompt_msg)]
        state.conversation = new_history
        yield new_history, "", [], "**Progress**: Waiting for image upload", state
        return
    
    # Use first image as representative for context
    representative = all_group_images[0]
    group_name = next(iter(state.image_groups.keys()))  # Use first group name
    group_images = all_group_images
    
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

    # Instantiate Solver (components already created above)
    solver = Solver(
        planner=planner,
        memory=memory,
        executor=executor,
        task="minitoolbench",  # Default task
        task_description="",   # Default empty description
        output_types="base,final,direct",  # Default output types
        verbose=True,          # Default verbose
        max_steps=10,
        max_time=999999,  # Effectively no time limit
        query_cache_dir=query_cache_dir,
        agent_state=state
    )

    if solver is None:
        new_history = messages + [ChatMessage(role="assistant", content="⚠️ Error: Failed to initialize solver.")]
        state.conversation = new_history
        yield new_history, "", [], "**Progress**: Error occurred", state
        return

    try:
        # Stream the solution - same as original version
        for messages, text_output, gallery_output, visual_desc, progress_md in solver.stream_solve_user_problem(user_query, state.image_context, group_name, group_images, api_key, messages):
            # Save steps data
            save_steps_data(query_id, memory)
            
            # Return the current state
            state.conversation = messages
            state.last_visual_description = visual_desc
            # Direct yield messages (already a list of ChatMessage objects)
            yield messages, text_output, gallery_output, progress_md, state
            
    except Exception as e:
        print(f"Error in solve_problem_gradio: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Full traceback: {error_traceback}")
        record_question_result(state, user_query, status="FAILED", final_answer="", error=str(e))
        
        # Create error message for UI - same as original version
        error_message = f"⚠️ Error occurred during analysis:\n\n**Error Type:** {type(e).__name__}\n**Error Message:** {str(e)}\n\nPlease check your input and try again."
        error_messages = messages + [ChatMessage(role="assistant", content=error_message)]
        state.conversation = error_messages
        yield error_messages, "", [], "**Progress**: Error occurred", state
    finally:
        print(f"Task completed for query_id: {query_id}. Preparing to clean up cache directory: {query_cache_dir}")
        try:
            # Add a check to prevent deleting the root solver_cache
            if query_cache_dir != DATASET_DIR.name and DATASET_DIR.name in query_cache_dir:
                # Preserve output_visualizations directory - DO NOT CLEAR IT
                # This allows users to keep all generated charts until they start a new analysis
                # For Spaces, use /tmp directory; for local, use current working directory
                if IS_SPACES:
                    output_viz_dir = "/tmp/output_visualizations"
                else:
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
        
        gr.Markdown("# Chat with SHAPE: A self-supervised morphology agent for cellular phenotype")  # Title
        gr.Markdown("""
        **SHPAE** is an open-source assistant for interpreting cell images, powered by large language models and tool-based reasoning.
        """)
        
        # Model / tool configuration (top row, equal height)
        with gr.Row(equal_height=True):
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### ⚙️ LLM Configuration")
                multimodal_models = [m for m in OPENAI_MODEL_CONFIGS.values()]
                model_names = [m["model_id"] for m in multimodal_models]
                # Prefer gpt-5-mini as default (latest cost-effective mini model)
                default_model = next((m["model_id"] for m in multimodal_models if m.get("model_id") == "gpt-5-mini"),
                                   next((m["model_id"] for m in multimodal_models if m.get("model_id") == "gpt-4o-mini"),
                                       next((m["model_id"] for m in multimodal_models if m.get("model_type") == "openai"), 
                                           model_names[0] if model_names else None)))
                language_model = gr.Dropdown(choices=model_names, value=default_model)
            with gr.Column(scale=1):
                gr.Markdown("### 🛠️ Available Tools")
                with gr.Accordion("🛠️ Available Tools", open=False):
                    gr.Markdown("\n".join([f"- {t}" for t in get_available_tools()]))

        # Main interaction row: left (uploads + question), right (conversation)
        with gr.Row():
            with gr.Column(scale=1):
                gr.Markdown("### 📤 Image Groups")
                user_image = gr.File(label="Upload Images", file_count="multiple", type="filepath")
                group_table = gr.Dataframe(headers=["image_name", "group"], row_count=0, wrap=True, interactive=True, visible=True)
                group_prompt = gr.Markdown("**Upload Status**: No uploads yet")
                upload_btn = gr.Button("Add Group(s) to Image(s)", variant="primary")
                gr.Markdown("### ❓ Ask Question")
                user_query = gr.Textbox(label="Ask about your groups", placeholder="e.g., Compare cell counts between control and drugA", lines=5)
                run_button = gr.Button("🚀 Ask Question", variant="primary", size="lg")
                progress_md = gr.Markdown("**Progress**: Ready")
                conversation_state = gr.State(AgentState())
            with gr.Column(scale=1):
                gr.Markdown("### 🗣️ Conversation")
                # Use type="messages" directly like original version
                chatbot_output = gr.Chatbot(type="messages", height=700, show_label=False)

        # Bottom full-width: summary and visual outputs
        gr.Markdown("### 🧾 Summary")
        text_output = gr.Markdown(value="")
        gr.Markdown("### 🖼️ Visual Outputs")
        gallery_output = gr.Gallery(label=None, show_label=False, height=350, columns=3, rows=2)

        # Examples row (optional, no nesting issues)
        with gr.Row():
            with gr.Column(scale=5):
                gr.Markdown("## 💡 Try these examples with suggested tools.")
                examples = [
                    ["Image Preprocessing", "examples/A5_01_1_1_Phase Contrast_001.png", "Normalize this phase contrast image.", "Image_Preprocessor_Tool", "Illumination-corrected and brightness-normalized phase contrast image."],
                    ["Cell Identification", "examples/A2_02_1_1_Phase Contrast_001.png", "How many cells are there in this image.", "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool", "258 cells are identified and their nuclei are labeled."],
                    ["Single-Cell Cropping", "examples/A3_02_1_1_Phase Contrast_001.png", "Crop single cells from the segmented nuclei in this image.", "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool", "Individual cell crops extracted from the image."],
                    ["Cell State Analysis", "examples/A4_02_1_1_Phase Contrast_001.png", "Analyze the cell states in this image.", "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool, Cell_State_Analyzer_Tool", "540 cells identified and segmented successfully. Comprehensive analysis of cell states have been performed with visualizations."],
                    ["Cell State Analysis", "examples/A5_02_1_1_Phase Contrast_001.png", "Analyze and visualize cell states in this image.", "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool, Cell_State_Analyzer_Tool", "Cell states have been analyzed and visualized with clustering results."],
                    ["Pathology Diagnosis", "examples/pathology.jpg", "What are the cell types in this image?", "Generalist_Solution_Generator_Tool, Image_Captioner_Tool, Relevant_Patch_Zoomer_Tool", "Need expert insights."],
                    ["Visual Reasoning", "examples/rotting_kiwi.png", "You are given a 3 x 3 grid in which each cell can contain either no kiwi, one fresh kiwi, or one rotten kiwi. Every minute, any fresh kiwi that is 4-directionally adjacent to a rotten kiwi also becomes rotten. What is the minimum number of minutes that must elapse until no cell has a fresh kiwi?", "Image_Captioner_Tool", "4 minutes"],
                    ["Scientific Research", None, "What are the research trends in tool agents with large language models for scientific discovery? Please consider the latest literature from ArXiv, PubMed, Nature, and news sources.", "ArXiv_Paper_Searcher_Tool, Pubmed_Search_Tool, Nature_News_Fetcher_Tool", "Open-ended question. No reference answer."]
                ]
                def distribute_tools(category, img, q, tools_str, ans):
                    selected_tools = [tool.strip() for tool in tools_str.split(',')]
                    selected = [tool for tool in selected_tools if tool in get_available_tools()]
                    return img, q
                gr.Markdown("#### 🧬 Analysis Examples")
                gr.Examples(
                    examples=examples,
                    inputs=[gr.Textbox(label="Category", visible=False), user_image, user_query, gr.Textbox(label="Select Tools", visible=False), gr.Textbox(label="Reference Answer", visible=False)],
                    outputs=[user_image, user_query],
                    fn=distribute_tools,
                    cache_examples=False
                )
        # Button click event
        user_image.change(
            prepare_group_assignment,
            inputs=[user_image, conversation_state],
            outputs=[group_table, group_prompt, conversation_state]
        )

        upload_btn.click(
            upload_image_to_group,
            inputs=[user_image, group_table, conversation_state],
            outputs=[conversation_state, group_prompt, group_table]
        )

        run_button.click(
            solve_problem_gradio,
            [user_query, language_model, conversation_state],
            [chatbot_output, text_output, gallery_output, progress_md, conversation_state]
        )

    #################### Gradio Interface ####################

    if IS_SPACES:
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False,
            show_error=True
        )
    else:
        import os
        server_port = os.getenv("GRADIO_SERVER_PORT")
        if server_port:
            try:
                server_port = int(server_port)
            except ValueError:
                server_port = None
        else:
            server_port = None
        
        demo.launch(
            server_name="127.0.0.1",
            server_port=server_port,
            debug=True,
            share=False
        )

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set default API source to use environment variables
    if not hasattr(args, 'openai_api_source') or args.openai_api_source is None:
        args.openai_api_source = "we_provided"

    args.enabled_tools = get_available_tools()
    args.root_cache_dir = DATASET_DIR.name
    
    print("\n=== Environment Information ===")
    print(f"Running in HuggingFace Spaces: {IS_SPACES}")
    print(f"Dataset/Cache Directory: {DATASET_DIR}")
    if IS_SPACES:
        print("⚠️ Note: Using /tmp directory - data will be cleared on restart")
    if TORCH_AVAILABLE:
        print(f"CUDA Available: {torch.cuda.is_available()}")
        if torch.cuda.is_available():
            print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    else:
        print("PyTorch not installed; running in CPU-only mode.")
    if not os.getenv("OPENAI_API_KEY"):
        print("⚠️ Warning: OPENAI_API_KEY not set - API calls will fail")
    else:
        print("✅ OPENAI_API_KEY is set")
    print("==============================\n")
    
    main(args)

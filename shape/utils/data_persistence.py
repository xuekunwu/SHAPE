"""
Data persistence utilities for saving query data, feedback, steps, and module data.

This module provides functions for persisting various types of data to the file system,
organized by query_id and session_id.
"""

from __future__ import annotations

import os
import json
import time
from pathlib import Path
from typing import TYPE_CHECKING, Any

from shape.schemas.formatters import ToolCommand

if TYPE_CHECKING:
    from shape.agent.memory import Memory


class CustomEncoder(json.JSONEncoder):
    """Custom JSON encoder for ToolCommand objects."""
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


def get_dataset_dir() -> Path:
    """
    Get the dataset directory path based on environment.
    
    Returns:
        Path to the dataset directory
    """
    IS_SPACES = os.getenv('SPACE_ID') is not None
    if IS_SPACES:
        dataset_dir = Path("/tmp/solver_cache")
    else:
        dataset_dir = Path("solver_cache")
    
    dataset_dir.mkdir(parents=True, exist_ok=True)
    return dataset_dir


def save_query_data(query_id: str, query: str, image_path: str, dataset_dir: Path = None) -> None:
    """
    Save query data to local cache.
    
    Args:
        query_id: Unique identifier for the query
        query: Query text
        image_path: Path to the image (if any)
        dataset_dir: Dataset directory (if None, uses get_dataset_dir())
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    
    query_cache_dir = dataset_dir / query_id
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


def save_feedback(query_id: str, feedback_type: str, feedback_text: str = None, dataset_dir: Path = None) -> None:
    """
    Save user feedback to the query directory.
    
    Args:
        query_id: Unique identifier for the query
        feedback_type: Type of feedback ('upvote', 'downvote', or 'comment')
        feedback_text: Optional text feedback from user
        dataset_dir: Dataset directory (if None, uses get_dataset_dir())
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    
    feedback_data_dir = dataset_dir / query_id
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


def save_steps_data(query_id: str, memory: Memory, dataset_dir: Path = None) -> None:
    """
    Save steps data to local cache.
    
    Args:
        query_id: Unique identifier for the query
        memory: Memory object containing all steps
        dataset_dir: Dataset directory (if None, uses get_dataset_dir())
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    
    steps_file = dataset_dir / query_id / "all_steps.json"

    memory_actions = memory.get_actions()
    memory_actions = make_json_serializable(memory_actions)
    print("Memory actions: ", memory_actions)

    with steps_file.open("w") as f:
        json.dump(memory_actions, f, indent=4, cls=CustomEncoder)

    
def save_module_data(query_id: str, key: str, value: Any, dataset_dir: Path = None) -> None:
    """
    Save module data to local cache.
    
    Args:
        query_id: Unique identifier for the query
        key: Module key (will be normalized)
        value: Value to save
        dataset_dir: Dataset directory (if None, uses get_dataset_dir())
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    
    try:
        key = key.replace(" ", "_").lower()
        module_file = dataset_dir / query_id / f"{key}.json"
        value = make_json_serializable(value)
        with module_file.open("a") as f:
            json.dump(value, f, indent=4, cls=CustomEncoder)
    except Exception as e:
        print(f"Warning: Failed to save as JSON: {e}")
        text_file = dataset_dir / query_id / f"{key}.txt"
        try:
            with text_file.open("a") as f:
                f.write(str(value) + "\n")
            print(f"Successfully saved as text file: {text_file}")
        except Exception as e:
            print(f"Error: Failed to save as text file: {e}")


def ensure_session_dirs(session_id: str, dataset_dir: Path = None):
    """
    Create and return session-scoped directories for caching.
    
    Args:
        session_id: Session identifier
        dataset_dir: Dataset directory (if None, uses get_dataset_dir())
    
    Returns:
        Tuple of (session_dir, images_dir, features_dir)
    """
    if dataset_dir is None:
        dataset_dir = get_dataset_dir()
    
    session_dir = dataset_dir / session_id
    images_dir = session_dir / "images"
    features_dir = session_dir / "features"
    session_dir.mkdir(parents=True, exist_ok=True)
    images_dir.mkdir(parents=True, exist_ok=True)
    features_dir.mkdir(parents=True, exist_ok=True)
    return session_dir, images_dir, features_dir


def save_run_trace(dataset_dir, stop_reason, verification_summary, steps=None) -> str:
    """Persist one VLRR run's trace + terminal `stop_reason` as `vlrr_trace.json`.

    `dataset_dir` is the run's cache dir (solve()'s `query_cache_dir`). Best-effort: writes small JSON
    only, and solve() calls it inside a try/except, so it never fails a run. Reuses
    `make_json_serializable` + `CustomEncoder` so AnnData/ToolCommand payloads don't break `json.dump`.
    Returns the path written.
    """
    d = Path(dataset_dir)
    d.mkdir(parents=True, exist_ok=True)
    trace_file = d / "vlrr_trace.json"
    payload = {
        "stop_reason": stop_reason,
        "datetime": time.strftime("%Y%m%d_%H%M%S"),
        "verification_summary": make_json_serializable(verification_summary),
    }
    if steps is not None:
        payload["steps"] = make_json_serializable(steps)
    try:
        from shape.engine import usage as _usage
        payload["usage"] = _usage.snapshot()   # token counts for the later cost benchmark (no cap)
    except Exception:
        pass
    with trace_file.open("w") as f:
        json.dump(payload, f, indent=2, cls=CustomEncoder)
    return str(trace_file)



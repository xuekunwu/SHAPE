"""
Artifact caching utilities for tool execution results.

This module provides functions for caching and retrieving tool execution artifacts,
supporting both session-specific and global cross-session caching.
"""

import hashlib
import time
from typing import Dict, List, Any, Optional
from pathlib import Path


# Global cross-session cache: {fingerprint: {tool_name: [artifacts]}}
# This allows reusing tool results across different conversation sessions for the same image
_global_artifact_cache: Dict[str, Dict[str, List[Dict[str, Any]]]] = {}


def make_artifact_key(tool_name: str, image_path: str, context: str = "", sub_goal: str = "", image_id: str = "") -> str:
    """
    Create a deterministic key for caching tool outputs tied to inputs (image-aware).
    
    Args:
        tool_name: Name of the tool
        image_path: Path to the image
        context: Context string
        sub_goal: Sub-goal string
        image_id: Image identifier
    
    Returns:
        SHA256 hash of the combined inputs
    """
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
    
    Args:
        tool_name: Name of the tool
        image_fingerprint: Image fingerprint (SHA256 hash)
    
    Returns:
        SHA256 hash of tool_name and image_fingerprint
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
    
    Returns:
        Cached artifact dict or None if not found
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
            _global_artifact_cache[image_fingerprint][tool_name].append({
                "fingerprint_key": fingerprint_key,
                "result": result,
                "created_at": time.time()
            })
        
        print(f"Stored artifact in global cache for fingerprint {image_fingerprint[:8]}... and tool {tool_name}")


def clear_global_cache():
    """Clear the global artifact cache. Useful for testing or memory management."""
    global _global_artifact_cache
    _global_artifact_cache = {}


def get_global_cache_stats() -> Dict[str, Any]:
    """
    Get statistics about the global artifact cache.
    
    Returns:
        Dictionary with cache statistics
    """
    total_fingerprints = len(_global_artifact_cache)
    total_artifacts = sum(
        len(tool_artifacts)
        for fingerprint_cache in _global_artifact_cache.values()
        for tool_artifacts in fingerprint_cache.values()
    )
    
    return {
        "total_fingerprints": total_fingerprints,
        "total_artifacts": total_artifacts,
        "cache_size_mb": 0  # Could be calculated if needed
    }


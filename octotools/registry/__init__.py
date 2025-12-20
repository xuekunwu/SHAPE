"""Centralized tool registry for capability-driven orchestration."""

# octotools/registry/__init__.py

from .tool_registry import (
    REGISTRY,
    ToolRegistry,
    ToolSpec,
    normalize_tool_name,
)

__all__ = [
    "REGISTRY",
    "ToolRegistry",
    "ToolSpec",
    "normalize_tool_name",
]


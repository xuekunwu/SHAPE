"""
Memory system for agent state tracking.

Maintains history of tool executions and observations, enabling
multi-step reasoning and context-aware planning.
"""

from shape.models.memory import Memory as _Memory

# Re-export with SHAPE namespace
Memory = _Memory

__all__ = ["Memory"]



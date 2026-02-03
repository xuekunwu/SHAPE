"""
Tool Executor for morphological analysis.

The executor takes planned steps and executes them using the available tools,
managing tool dependencies and result chaining.
"""

from shape.models.executor import Executor as _Executor

# Re-export with SHAPE namespace
Executor = _Executor

__all__ = ["Executor"]



"""
Intelligent Planner for morphological reasoning.

The planner analyzes queries and generates step-by-step plans using LLM-driven
decision making. It treats morphology as a first-class reasoning object.
"""

# Import from octotools for now (implementation detail)
# TODO: Refactor to move implementation into shape/ directly
from octotools.models.planner import Planner as _Planner

# Re-export with SHAPE namespace
Planner = _Planner

__all__ = ["Planner"]


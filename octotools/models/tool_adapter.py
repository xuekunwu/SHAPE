"""
Tool adapter interface to replace exec-based invocation.

Each concrete tool should provide an Adapter class in its package exposing:
- input_model: pydantic BaseModel describing inputs
- output_model: optional pydantic BaseModel for structured outputs
- run(args, session_state): executes underlying tool safely
"""

import importlib
from typing import Any, Optional, Tuple
from pydantic import BaseModel


class ToolAdapter:
    input_model: BaseModel
    output_model: Optional[BaseModel] = None

    def run(self, args: BaseModel, session_state: Any = None) -> Any:
        raise NotImplementedError("Adapter must implement run")


def load_adapter(adapter_path: str) -> ToolAdapter:
    """Load adapter from dotted path 'module:Class'."""
    if ":" not in adapter_path:
        raise ValueError(f"Invalid adapter path: {adapter_path}")
    module_path, class_name = adapter_path.split(":")
    module = importlib.import_module(module_path)
    adapter_cls = getattr(module, class_name)
    return adapter_cls()

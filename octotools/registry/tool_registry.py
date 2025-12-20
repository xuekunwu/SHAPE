"""
Capability-driven tool registry.

This provides a single source of truth for tool metadata and replaces
ad-hoc tool discovery/heuristics. Planner and executor should consult
this registry to resolve tools by capability and to load adapters safely.
"""

from dataclasses import dataclass, field
from typing import Dict, List, Optional, Callable


@dataclass
class ToolSpec:
    """Declarative specification for a tool."""

    name: str
    description: str
    capabilities: List[str]
    domain: str  # e.g., "general", "fibroblast"
    adapter_path: str  # python import path to adapter class
    input_schema: Dict[str, str] = field(default_factory=dict)
    output_schema: Dict[str, str] = field(default_factory=dict)


class ToolRegistry:
    """Central registry of tool specifications."""

    def __init__(self):
        self._tools: Dict[str, ToolSpec] = {}
        self._capability_index: Dict[str, List[str]] = {}

    def register(self, spec: ToolSpec) -> None:
        """Register or replace a tool spec."""
        self._tools[spec.name] = spec
        for cap in spec.capabilities:
            self._capability_index.setdefault(cap, [])
            if spec.name not in self._capability_index[cap]:
                self._capability_index[cap].append(spec.name)

    def get(self, name: str) -> Optional[ToolSpec]:
        return self._tools.get(name)

    def by_capability(self, capability: str) -> List[ToolSpec]:
        names = self._capability_index.get(capability, [])
        return [self._tools[n] for n in names if n in self._tools]

    def all(self) -> List[ToolSpec]:
        return list(self._tools.values())


# Global singleton used across planner/executor
REGISTRY = ToolRegistry()


def normalize_tool_name(tool_name: str) -> str:
    """
    Normalize a tool name using the registry entries instead of ad-hoc matching.
    """
    if tool_name in REGISTRY._tools:
        return tool_name
    # Try case-insensitive exact match
    for name in REGISTRY._tools:
        if name.lower() == tool_name.lower():
            return name
    return tool_name

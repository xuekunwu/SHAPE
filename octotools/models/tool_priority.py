"""
Tool Priority Configuration System for LLM-Orchestrated Single-Cell Bioimage Analysis.

This module defines priority levels and domain-specific tool filtering for ensuring
that the most relevant tools are selected for bioimage analysis tasks.
"""

from typing import Dict, List, Set, Optional, Tuple
from enum import IntEnum


class ToolPriority(IntEnum):
    """Priority levels for tool selection (lower number = higher priority)."""
    HIGH = 1          # High priority tools for bioimage analysis
    MEDIUM = 2        # Medium priority tools
    LOW = 3           # Low priority tools (use sparingly)
    EXCLUDED = 99     # Tools that should be excluded for bioimage tasks


# Tool priority mapping for bioimage analysis domain
BIOIMAGE_TOOL_PRIORITIES: Dict[str, ToolPriority] = {
    # HIGH: Core image processing, segmentation, and specialized analysis tools
    "Image_Preprocessor_Tool": ToolPriority.HIGH,
    "Nuclei_Segmenter_Tool": ToolPriority.HIGH,
    "Cell_Segmenter_Tool": ToolPriority.HIGH,  # For phase-contrast cell images
    "Organoid_Segmenter_Tool": ToolPriority.HIGH,  # For organoid segmentation
    "Single_Cell_Cropper_Tool": ToolPriority.HIGH,
    "Cell_State_Analyzer_Tool": ToolPriority.HIGH,  # Self-supervised learning for cell state analysis
    "Fibroblast_State_Analyzer_Tool": ToolPriority.HIGH,  # Keep for backward compatibility
    "Fibroblast_Activation_Scorer_Tool": ToolPriority.HIGH,
    "Analysis_Visualizer_Tool": ToolPriority.HIGH,
    
    # MEDIUM: General image analysis tools
    "Object_Detector_Tool": ToolPriority.MEDIUM,
    "Advanced_Object_Detector_Tool": ToolPriority.MEDIUM,
    "Image_Captioner_Tool": ToolPriority.MEDIUM,
    
    # LOW: Utility tools and code generation tools (use sparingly)
    "Text_Detector_Tool": ToolPriority.LOW,
    "Python_Code_Generator_Tool": ToolPriority.LOW,
    
    # EXCLUDED: Tools not relevant for bioimage analysis
    "Generalist_Solution_Generator_Tool": ToolPriority.EXCLUDED,  # Not useful, excluded
    "Google_Search_Tool": ToolPriority.EXCLUDED,
    "Pubmed_Search_Tool": ToolPriority.EXCLUDED,
    "Arxiv_Paper_Searcher_Tool": ToolPriority.EXCLUDED,
    "Nature_News_Fetcher_Tool": ToolPriority.EXCLUDED,
    "Wikipedia_Knowledge_Searcher_Tool": ToolPriority.EXCLUDED,
    "Url_Text_Extractor_Tool": ToolPriority.EXCLUDED,
    "Relevant_Patch_Zoomer_Tool": ToolPriority.EXCLUDED,  # Not suitable for bioimages
}

# Tool dependency chains for workflow optimization
TOOL_DEPENDENCIES: Dict[str, List[str]] = {
    "Single_Cell_Cropper_Tool": ["Nuclei_Segmenter_Tool", "Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"],  # Cropper needs segmentation
    "Cell_State_Analyzer_Tool": ["Single_Cell_Cropper_Tool"],  # Needs cell crops
    "Fibroblast_State_Analyzer_Tool": ["Single_Cell_Cropper_Tool", "Nuclei_Segmenter_Tool", "Cell_Segmenter_Tool"],  # Keep for backward compatibility
    "Fibroblast_Activation_Scorer_Tool": ["Fibroblast_State_Analyzer_Tool"],
    "Analysis_Visualizer_Tool": [],  # Can work with any analysis output
}

# Keywords for detecting task domains
BIOIMAGE_KEYWORDS: Set[str] = {
    'cell', 'nucleus', 'nuclei', 'fibroblast', 'segment', 'crop',
    'microscopy', 'microscope', 'phase contrast', 'fluorescence',
    'single cell', 'cell state', 'activation', 'morphology',
    'bioimage', 'biological image', 'cell analysis', 'cellular',
    'organoid', 'organoids', 'tissue', 'spheroid', 'spheroids'
}

SEARCH_KEYWORDS: Set[str] = {
    'search', 'find paper', 'lookup', 'information about', 'what is',
    'pubmed', 'arxiv', 'paper', 'article', 'literature'
}

TEXT_EXTRACTION_KEYWORDS: Set[str] = {
    'extract text', 'read url', 'get text from', 'parse webpage'
}


class ToolPriorityManager:
    """Manages tool priorities and filtering based on task domain."""
    
    def __init__(self, priorities: Optional[Dict[str, ToolPriority]] = None):
        """
        Initialize the priority manager.
        
        Args:
            priorities: Custom priority mapping. Defaults to BIOIMAGE_TOOL_PRIORITIES.
        """
        self.priorities = priorities or BIOIMAGE_TOOL_PRIORITIES.copy()
        self.default_priority = ToolPriority.MEDIUM
    
    def get_priority(self, tool_name: str) -> ToolPriority:
        """Get priority for a tool."""
        # Normalize tool name (handle variations)
        normalized = self._normalize_tool_name(tool_name)
        return self.priorities.get(normalized, self.default_priority)
    
    def _normalize_tool_name(self, tool_name: str) -> str:
        """Normalize tool name for lookup."""
        # Handle common variations
        if tool_name.endswith('_Tool'):
            return tool_name
        return f"{tool_name}_Tool" if not tool_name.endswith('_Tool') else tool_name
    
    def detect_task_domain(self, query: str, query_analysis: str = "") -> str:
        """
        Detect the task domain from query and analysis.
        
        Returns:
            'bioimage', 'search', 'text_extraction', or 'general'
        """
        query_lower = query.lower()
        analysis_lower = query_analysis.lower() if query_analysis else ""
        combined = f"{query_lower} {analysis_lower}"
        
        # Check for search tasks
        if any(keyword in combined for keyword in SEARCH_KEYWORDS):
            return 'search'
        
        # Check for text extraction tasks
        if any(keyword in combined for keyword in TEXT_EXTRACTION_KEYWORDS):
            return 'text_extraction'
        
        # Check for bioimage tasks (default for this system)
        if any(keyword in combined for keyword in BIOIMAGE_KEYWORDS):
            return 'bioimage'
        
        return 'general'
    
    def filter_tools_for_domain(
        self, 
        available_tools: List[str], 
        domain: str,
        exclude_excluded: bool = True
    ) -> Tuple[List[str], List[str]]:
        """
        Filter and prioritize tools based on domain.
        
        Args:
            available_tools: List of all available tool names
            domain: Task domain ('bioimage', 'search', 'text_extraction', 'general')
            exclude_excluded: Whether to exclude EXCLUDED priority tools
        
        Returns:
            Tuple of (filtered_tools, excluded_tools) where filtered_tools are sorted by priority
        """
        filtered = []
        excluded = []
        
        for tool in available_tools:
            priority = self.get_priority(tool)
            
            # Exclude EXCLUDED tools if requested
            if exclude_excluded and priority == ToolPriority.EXCLUDED:
                excluded.append(tool)
                continue
            
            # For bioimage domain, prioritize bioimage-specific tools
            if domain == 'bioimage':
                if priority == ToolPriority.EXCLUDED:
                    excluded.append(tool)
                    continue
                # Allow all non-excluded tools but prioritize them
                filtered.append(tool)
            elif domain == 'search':
                # For search tasks, allow search tools
                if 'Search' in tool or 'Searcher' in tool or 'Fetcher' in tool:
                    filtered.append(tool)
                elif priority == ToolPriority.EXCLUDED and 'Search' not in tool and 'Searcher' not in tool:
                    excluded.append(tool)
                else:
                    filtered.append(tool)
            else:
                # For other domains, use all tools except EXCLUDED
                if priority != ToolPriority.EXCLUDED:
                    filtered.append(tool)
                else:
                    excluded.append(tool)
        
        # Sort filtered tools by priority (lower number = higher priority)
        filtered.sort(key=lambda t: (self.get_priority(t).value, t))
        
        return filtered, excluded
    
    def get_tool_priority_groups(self, tools: List[str]) -> Dict[ToolPriority, List[str]]:
        """Group tools by priority level."""
        groups: Dict[ToolPriority, List[str]] = {}
        for tool in tools:
            priority = self.get_priority(tool)
            if priority not in groups:
                groups[priority] = []
            groups[priority].append(tool)
        return groups
    
    def get_recommended_next_tools(
        self,
        available_tools: List[str],
        used_tools: List[str],
        domain: str = 'bioimage'
    ) -> List[str]:
        """
        Get recommended next tools based on dependencies and priorities.
        
        Args:
            available_tools: All available tools
            used_tools: Tools already used in the workflow
            domain: Task domain
        
        Returns:
            List of recommended tools in priority order
        """
        filtered_tools, _ = self.filter_tools_for_domain(available_tools, domain)
        
        # Check dependencies
        recommended = []
        used_set = set(used_tools)
        
        for tool in filtered_tools:
            if tool in used_set:
                continue
            
            # Check if dependencies are satisfied
            deps = TOOL_DEPENDENCIES.get(tool, [])
            if all(dep in used_set for dep in deps):
                recommended.append(tool)
            elif not deps:  # No dependencies
                recommended.append(tool)
        
        # Sort by priority
        recommended.sort(key=lambda t: (self.get_priority(t).value, t))
        
        return recommended
    
    def format_tools_by_priority(self, tools: List[str]) -> str:
        """Format tools list with priority indicators for LLM prompts."""
        groups = self.get_tool_priority_groups(tools)
        
        lines = []
        priority_names = {
            ToolPriority.HIGH: "HIGH (High Priority - Use First)",
            ToolPriority.MEDIUM: "MEDIUM (Medium Priority - General Tools)",
            ToolPriority.LOW: "LOW (Low Priority - Use Sparingly)",
        }
        
        for priority in sorted(groups.keys()):
            if priority == ToolPriority.EXCLUDED:
                continue
            tool_list = groups[priority]
            if tool_list:
                priority_label = priority_names.get(priority, f"Priority {priority.value}")
                lines.append(f"{priority_label}:")
                for tool in tool_list:
                    lines.append(f"  - {tool}")
        
        return "\n".join(lines) if lines else "No tools available"
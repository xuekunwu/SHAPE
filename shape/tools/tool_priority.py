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


# The bioimage-domain priority TABLE is no longer hard-coded -- it is DERIVED from each
# tool's CARD.md `category` (the single source of truth). Adding or recategorizing a tool is
# a card edit, and the LLM planner reasons over the cards' when-to-use text
# (see `format_tools_by_priority`) rather than a fixed per-tool ranking.
_CATEGORY_TO_PRIORITY: Dict[str, "ToolPriority"] = {
    "bioimage_core": ToolPriority.HIGH,
    "utility": ToolPriority.LOW,
    "knowledge": ToolPriority.EXCLUDED,   # excluded for the BIOIMAGE domain; the
                                          # knowledge-domain filter still surfaces them
}


def _derive_priorities_from_cards() -> Dict[str, ToolPriority]:
    """Build {tool: priority} from the tools' CARD.md categories; empty on any failure."""
    try:
        from shape.tools.base import card_index
        return {name: _CATEGORY_TO_PRIORITY[hdr["category"]]
                for name, (_d, hdr, _b) in card_index().items()
                if hdr.get("category") in _CATEGORY_TO_PRIORITY}
    except Exception:
        return {}


BIOIMAGE_TOOL_PRIORITIES: Dict[str, ToolPriority] = _derive_priorities_from_cards()

# Tool dependency chains for workflow optimization
# Bioimage analysis chain: (Cell_Segmenter/Organoid_Segmenter) -> Single_Cell_Cropper -> Cell_State_Analyzer -> Analysis_Visualizer

# CRITICAL: Tool role definitions
# - Analysis Tools: Extract features, perform statistical analysis, generate embeddings/clusters
#   * Cell_State_Analyzer_Tool: Auto-channel feature extraction + clustering
# - Visualization Tools: Only visualize pre-computed analysis results
#   * Analysis_Visualizer_Tool: Visualizes results from Cell_State_Analyzer or segmentation outputs

TOOL_DEPENDENCIES: Dict[str, List[str]] = {
    "Single_Cell_Cropper_Tool": ["Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"],  # Cropper needs segmentation
    "Cell_State_Analyzer_Tool": ["Single_Cell_Cropper_Tool"],
    "Cell_Cluster_Functional_Hypothesis_Tool": ["Cell_State_Analyzer_Tool"],
    # Analysis_Visualizer_Tool has flexible dependencies:
    # - For morphology/comparison queries: REQUIRES Cell_State_Analyzer_Tool (feature extraction needed)
    # - For simple counting: Can work with segmentation results directly
    # - For cell state visualization: REQUIRES Cell_State_Analyzer_Tool
    "Analysis_Visualizer_Tool": [],  # Flexible: can work with segmentation or analyzer outputs
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

# Keywords for detecting knowledge-based analysis tasks (non-image)
KNOWLEDGE_KEYWORDS: Set[str] = {
    'literature', 'literature mining', 'gene annotation', 'gene function',
    'pathway enrichment', 'go enrichment', 'functional enrichment',
    'functional annotation', 'gene cluster', 'marker gene', 'cell type marker',
    'pubmed', 'database', 'functional analysis', 'biological interpretation',
    'transcriptomics', 'genomics', 'bioinformatics', 'gene expression',
    'pathway analysis', 'enrichment analysis', 'gene ontology', 'kegg',
    'reactome', 'david', 'g:profiler', 'enrichr', 'cellmarker', 'panglaodb',
    'cell type', 'cell identity', 'cluster identity', 'biological classification',
    'orthogonal validation', 'ihc marker', 'if marker', 'validation marker',
    'no image', 'external analysis', 'text-based', 'knowledge-based',
    'functional hypothesis', 'functional state', 'regulator gene', 'regulator genes',
    'cell cluster', 'cluster annotation', 'functional inference', 'cluster functional'
}

# Keywords for detecting cell cluster functional annotation tasks (use specialized tool)
CLUSTER_ANNOTATION_KEYWORDS: Set[str] = {
    'cluster annotation', 'cluster functional', 'functional hypothesis', 'functional state',
    'regulator gene', 'regulator genes', 'top regulator', 'top regulators',
    'cell cluster', 'gene cluster', 'cluster identity', 'cluster classification',
    'cluster marker', 'cluster markers', 'cluster function', 'cluster functions'
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
            'bioimage', 'search', 'text_extraction', 'knowledge', or 'general'
        """
        query_lower = query.lower()
        analysis_lower = query_analysis.lower() if query_analysis else ""
        combined = f"{query_lower} {analysis_lower}"
        
        has_bioimage_signal = any(keyword in combined for keyword in BIOIMAGE_KEYWORDS)
        explicitly_non_image = any(term in combined for term in (
            "no image", "without image", "text-based only", "literature only",
        ))

        # Mixed prompts often request both image analysis and downstream
        # biological interpretation. Keep those in the bioimage workflow.
        if has_bioimage_signal and not explicitly_non_image:
            return 'bioimage'

        if any(keyword in combined for keyword in KNOWLEDGE_KEYWORDS):
            return 'knowledge'
        
        # Check for search tasks
        if any(keyword in combined for keyword in SEARCH_KEYWORDS):
            return 'search'
        
        # Check for text extraction tasks
        if any(keyword in combined for keyword in TEXT_EXTRACTION_KEYWORDS):
            return 'text_extraction'
        
        return 'general'

    def is_cluster_annotation_task(self, query: str, query_analysis: str = "") -> bool:
        """Return whether the request is specifically about annotating clusters."""
        combined = f"{query} {query_analysis}".lower()
        return any(keyword in combined for keyword in CLUSTER_ANNOTATION_KEYWORDS)

    def filter_segmentation_tools(
        self,
        tools: List[str],
        query: str,
        query_analysis: str = "",
        image_info: Optional[Dict] = None,
    ) -> Tuple[List[str], List[str]]:
        """Rank the likely segmenter first without removing alternatives."""
        image_info = image_info or {}
        channel_names = " ".join(image_info.get("channel_names") or []).lower()
        combined = f"{query} {query_analysis} {channel_names}".lower()

        if any(term in combined for term in ("organoid", "organoids", "spheroid", "spheroids")):
            selected = "Organoid_Segmenter_Tool"
        else:
            # Cellpose-SAM (CPSAM) is the single adaptive whole-cell segmenter: it handles
            # phase-contrast, live-cell AND fluorescence/nuclei (DAPI) inputs, so there is no
            # separate nuclei segmenter to route to.
            selected = "Cell_Segmenter_Tool"

        segmenters = {
            "Cell_Segmenter_Tool",
            "Organoid_Segmenter_Tool",
        }
        ranked = sorted(
            tools,
            key=lambda tool: (
                0 if tool == selected else 1 if tool not in segmenters else 2,
                self.get_priority(tool).value,
                tool,
            ),
        )
        return ranked, []
    
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
            domain: Task domain ('bioimage', 'search', 'text_extraction', 'knowledge', 'general')
            exclude_excluded: Whether to exclude EXCLUDED priority tools
        
        Returns:
            Tuple of (filtered_tools, excluded_tools) where filtered_tools are sorted by priority
        """
        filtered = []
        excluded = []
        
        for tool in available_tools:
            priority = self.get_priority(tool)
            
            # For knowledge domain (non-image analysis tasks)
            if domain == 'knowledge':
                if tool == 'Cell_Cluster_Functional_Hypothesis_Tool':
                    # HIGHEST priority for cluster functional hypothesis tool in knowledge domain
                    # This tool internally handles search + synthesis, so it should be preferred
                    filtered.append(tool)
                elif 'Search' in tool or 'Searcher' in tool or 'Fetcher' in tool:
                    # Search tools for general knowledge tasks (even if EXCLUDED for bioimage)
                    filtered.append(tool)
                elif tool == 'Generalist_Solution_Generator_Tool':
                    # Synthesis tool (use after search, or as fallback)
                    filtered.append(tool)
                elif priority == ToolPriority.EXCLUDED:
                    # Exclude image processing tools for knowledge tasks
                    excluded.append(tool)
                elif priority != ToolPriority.EXCLUDED:
                    # Allow other non-excluded tools
                    filtered.append(tool)
                continue
            
            # Bioimage workflows may still need literature or hypothesis tools
            # after image-derived evidence exists. Keep them available at the
            # lowest priority instead of removing them outright.
            if exclude_excluded and priority == ToolPriority.EXCLUDED and domain != "bioimage":
                excluded.append(tool)
                continue
            
            # For bioimage domain, prioritize bioimage-specific tools
            if domain == 'bioimage':
                # Keep all tools available; priority controls recommendation.
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
        # Special handling: prioritize Cell_Cluster_Functional_Hypothesis_Tool for knowledge domain
        if domain == 'knowledge':
            # Custom sort: Cell_Cluster_Functional_Hypothesis_Tool first, then by priority
            def sort_key(t):
                if t == 'Cell_Cluster_Functional_Hypothesis_Tool':
                    return (0, t)  # Highest priority
                priority = self.get_priority(t)
                return (priority.value, t)
            filtered.sort(key=sort_key)
        else:
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
        domain: str = 'bioimage',
        query: str = "",
        query_analysis: str = ""
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
        if domain == "bioimage":
            filtered_tools, _ = self.filter_segmentation_tools(
                filtered_tools, query, query_analysis
            )
        
        # For knowledge domain with cluster annotation tasks, prioritize specialized tool
        if domain == 'knowledge' and self.is_cluster_annotation_task(query, query_analysis):
            if 'Cell_Cluster_Functional_Hypothesis_Tool' in filtered_tools and 'Cell_Cluster_Functional_Hypothesis_Tool' not in used_tools:
                return ['Cell_Cluster_Functional_Hypothesis_Tool'] + [t for t in filtered_tools if t != 'Cell_Cluster_Functional_Hypothesis_Tool']
        
        # Check dependencies
        recommended = []
        used_set = set(used_tools)
        
        # CRITICAL: Enforce tool dependency chain
        segmentation_tools = ["Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"]
        analyzer_tools = ["Cell_State_Analyzer_Tool"]
        last_tool = used_tools[-1] if used_tools else None
        
        # Rule 1: If segmentation tool was just used -> Single_Cell_Cropper_Tool must be next
        if last_tool in segmentation_tools:
            if "Single_Cell_Cropper_Tool" in filtered_tools and "Single_Cell_Cropper_Tool" not in used_set:
                recommended.append("Single_Cell_Cropper_Tool")
        
        # Rule 2: If Single_Cell_Cropper_Tool was just used -> Cell_State_Analyzer_Tool must be next
        if last_tool == "Single_Cell_Cropper_Tool":
            # Check image channels to recommend appropriate analyzer (single vs multi)
            # For now, recommend both - planner will choose based on image info
            for analyzer in analyzer_tools:
                if analyzer in filtered_tools and analyzer not in used_set:
                    recommended.append(analyzer)
        
        # Rule 3: If Cell_State_Analyzer_Tool was just used -> Analysis_Visualizer_Tool should be next
        if last_tool in analyzer_tools:
            if "Analysis_Visualizer_Tool" in filtered_tools and "Analysis_Visualizer_Tool" not in used_set:
                recommended.append("Analysis_Visualizer_Tool")
        
        for tool in filtered_tools:
            if tool in used_set:
                continue
            
            # Skip if already added by rules above
            if tool in recommended:
                continue
            
            # Check if dependencies are satisfied
            deps = TOOL_DEPENDENCIES.get(tool, [])
            # For Analysis_Visualizer_Tool, check if ANY analyzer was used
            if tool == "Analysis_Visualizer_Tool":
                analyzer_used = any(a in used_set for a in analyzer_tools)
                if analyzer_used:
                    recommended.append(tool)
            elif all(dep in used_set for dep in deps):
                recommended.append(tool)
            elif not deps:  # No dependencies
                recommended.append(tool)
        
        # Sort by priority (but keep Single_Cell_Cropper_Tool first if segmentation just completed)
        if last_tool in segmentation_tools and "Single_Cell_Cropper_Tool" in recommended:
            # Single_Cell_Cropper_Tool is already at the top, sort the rest
            rest = [t for t in recommended if t != "Single_Cell_Cropper_Tool"]
            rest.sort(key=lambda t: (self.get_priority(t).value, t))
            recommended = ["Single_Cell_Cropper_Tool"] + rest
        else:
            # Normal sorting
            recommended.sort(key=lambda t: (self.get_priority(t).value, t))
        
        return recommended
    
    def format_tools_by_priority(self, tools: List[str]) -> str:
        """Present the available tools grouped by CARD CATEGORY, each with its when-to-use
        line, so the LLM REASONS the selection from the cards rather than a fixed ranking."""
        from shape.tools.base import card_index, card_summary
        idx = card_index()
        order = [
            ("bioimage_core", "BIOIMAGE TOOLS (the analysis pipeline -- prefer these for image tasks):"),
            ("utility", "UTILITY TOOLS:"),
            ("knowledge", "KNOWLEDGE TOOLS (literature / web -- only when the task needs external knowledge):"),
        ]
        labels = dict(order)
        by_cat: Dict[str, List[str]] = {}
        for tool in tools:
            cat = idx.get(tool, (None, {}, None))[1].get("category", "utility")
            by_cat.setdefault(cat, []).append(tool)

        lines: List[str] = []
        for cat, label in order:
            group = by_cat.get(cat)
            if not group:
                continue
            lines.append(label)
            for tool in sorted(group):
                summary = card_summary(tool)
                lines.append(f"  - {tool}: {summary}" if summary else f"  - {tool}")
        # tools without a recognized card category -- list plainly so nothing disappears
        leftover = [t for t in sorted(tools)
                    if idx.get(t, (None, {}, None))[1].get("category") not in labels]
        if leftover:
            lines.append("OTHER TOOLS:")
            lines.extend(f"  - {t}" for t in leftover)

        return "\n".join(lines) if lines else "No tools available"

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
    "Cell_State_Analyzer_Single_Tool": ToolPriority.HIGH,  # Self-supervised learning for single-channel images
    "Cell_State_Analyzer_Multi_Tool": ToolPriority.HIGH,  # Self-supervised learning for multi-channel images (2+ channels)
    "Analysis_Visualizer_Tool": ToolPriority.HIGH,
        
    # LOW: Utility tools and code generation tools (use sparingly)
    "Object_Detector_Tool": ToolPriority.LOW,
    "Advanced_Object_Detector_Tool": ToolPriority.LOW,
    "Text_Detector_Tool": ToolPriority.LOW,
    "Python_Code_Generator_Tool": ToolPriority.LOW,
    
    # EXCLUDED: Tools not relevant for bioimage analysis
    "Generalist_Solution_Generator_Tool": ToolPriority.EXCLUDED,  # Not useful, excluded
    "Google_Search_Tool": ToolPriority.EXCLUDED,
    "Pubmed_Search_Tool": ToolPriority.EXCLUDED,
    "ArXiv_Paper_Searcher_Tool": ToolPriority.EXCLUDED,  # Note: ArXiv is mixed case (as defined in tool class)
    "Nature_News_Fetcher_Tool": ToolPriority.EXCLUDED,
    "Wikipedia_Knowledge_Searcher_Tool": ToolPriority.EXCLUDED,
    "URL_Text_Extractor_Tool": ToolPriority.EXCLUDED,  # Note: URL is all uppercase (as defined in tool class)
    "Relevant_Patch_Zoomer_Tool": ToolPriority.EXCLUDED,  # Not suitable for bioimages
    "Cell_Cluster_Functional_Hypothesis_Tool": ToolPriority.EXCLUDED,  # For knowledge domain only
}

# Tool dependency chains for workflow optimization
# Bioimage analysis chain: Image_Preprocessor 鈫?(Cell_Segmenter/Nuclei_Segmenter/Organoid_Segmenter) 鈫?Single_Cell_Cropper 鈫?Cell_State_Analyzer 鈫?Analysis_Visualizer

# CRITICAL: Tool role definitions
# - Analysis Tools: Extract features, perform statistical analysis, generate embeddings/clusters
#   * Cell_State_Analyzer_Single_Tool: Feature extraction + clustering for single-channel images
#   * Cell_State_Analyzer_Multi_Tool: Feature extraction + clustering for multi-channel images
# - Visualization Tools: Only visualize pre-computed analysis results
#   * Analysis_Visualizer_Tool: Visualizes results from Cell_State_Analyzer or segmentation outputs

TOOL_DEPENDENCIES: Dict[str, List[str]] = {
    "Single_Cell_Cropper_Tool": ["Nuclei_Segmenter_Tool", "Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"],  # Cropper needs segmentation
    "Cell_State_Analyzer_Single_Tool": ["Single_Cell_Cropper_Tool"],  # Needs cell crops (single-channel)
    "Cell_State_Analyzer_Multi_Tool": ["Single_Cell_Cropper_Tool"],  # Needs cell crops (multi-channel)
    # Analysis_Visualizer_Tool has flexible dependencies:
    # - For morphology/comparison queries: REQUIRES Cell_State_Analyzer_*_Tool (feature extraction needed)
    # - For simple counting: Can work with segmentation results directly
    # - For cell state visualization: REQUIRES Cell_State_Analyzer_*_Tool
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
        
        # Check for knowledge-based analysis tasks (non-image: literature, gene annotation, etc.)
        # This should be checked before bioimage to avoid false positives
        if any(keyword in combined for keyword in KNOWLEDGE_KEYWORDS):
            return 'knowledge'
        
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
                # Check if this is a cluster annotation task (needs specialized tool)
                is_cluster_task = self.is_cluster_annotation_task(query="", query_analysis="")
                # Note: We can't access query here, so we'll handle this in the sorting/priority logic
                
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
        
        # For knowledge domain with cluster annotation tasks, prioritize specialized tool
        if domain == 'knowledge' and self.is_cluster_annotation_task(query, query_analysis):
            if 'Cell_Cluster_Functional_Hypothesis_Tool' in filtered_tools and 'Cell_Cluster_Functional_Hypothesis_Tool' not in used_tools:
                return ['Cell_Cluster_Functional_Hypothesis_Tool'] + [t for t in filtered_tools if t != 'Cell_Cluster_Functional_Hypothesis_Tool']
        
        # Check dependencies
        recommended = []
        used_set = set(used_tools)
        
        # CRITICAL: Enforce tool dependency chain
        segmentation_tools = ["Cell_Segmenter_Tool", "Nuclei_Segmenter_Tool", "Organoid_Segmenter_Tool"]
        analyzer_tools = ["Cell_State_Analyzer_Single_Tool", "Cell_State_Analyzer_Multi_Tool"]
        last_tool = used_tools[-1] if used_tools else None
        
        # Rule 1: If segmentation tool was just used 鈫?Single_Cell_Cropper_Tool must be next
        if last_tool in segmentation_tools:
            if "Single_Cell_Cropper_Tool" in filtered_tools and "Single_Cell_Cropper_Tool" not in used_set:
                recommended.append("Single_Cell_Cropper_Tool")
        
        # Rule 2: If Single_Cell_Cropper_Tool was just used 鈫?Cell_State_Analyzer_*_Tool must be next
        if last_tool == "Single_Cell_Cropper_Tool":
            # Check image channels to recommend appropriate analyzer (single vs multi)
            # For now, recommend both - planner will choose based on image info
            for analyzer in analyzer_tools:
                if analyzer in filtered_tools and analyzer not in used_set:
                    recommended.append(analyzer)
        
        # Rule 3: If Cell_State_Analyzer_*_Tool was just used 鈫?Analysis_Visualizer_Tool should be next
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


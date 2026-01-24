"""
Intelligent Planner - Goal-Oriented Planning for Bioimage Analysis

Design Philosophy (inspired by Biomni):
- Goal-Oriented Action Planning (GOAP)
- LLM-driven intelligent decision making
- Minimal hardcoded rules, maximum flexibility
- Rich context provision for LLM reasoning
- Essential bioimage analysis chain awareness
- Multi-image and group comparison support
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from io import BytesIO

from octotools.engine.openai import ChatOpenAI
from octotools.models.memory import Memory
from octotools.models.formatters import QueryAnalysis, NextStep, MemoryVerification
from octotools.models.tool_priority import ToolPriorityManager
from octotools.utils import logger, ResponseParser
from octotools.utils.image_processor import ImageProcessor


class Planner:
    """
    Intelligent planner for bioimage analysis system.
    
    Core principles:
    - Trust LLM intelligence for tool selection
    - Provide rich context (query, memory, tools, metadata)
    - Guide through prompts, not hardcoded rules
    - Retain essential bioimage analysis chain awareness
    - Support multi-image and group comparison scenarios
    """
    
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None, 
                 available_tools: List = None, api_key: str = None):
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []
        self.api_key = api_key
        
        # Initialize LLM engines
        self.llm_engine = ChatOpenAI(
            model_string=llm_engine_name, 
            is_multimodal=False, 
            api_key=api_key
        )
        self.llm_engine_mm = ChatOpenAI(
            model_string=llm_engine_name, 
            is_multimodal=True, 
            api_key=api_key
        )
        
        # Tool priority manager
        self.priority_manager = ToolPriorityManager()
        
        # State tracking
        self.query_analysis = None
        self.last_usage = {}
        self.base_response = None
        self.detected_domain = 'general'
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Extract comprehensive image metadata for context."""
        if not image_path or not os.path.isfile(image_path):
            return {}
        
        try:
            img_data = ImageProcessor.load_image(image_path)
            return {
                "width": img_data.shape[1],
                "height": img_data.shape[0],
                "num_channels": img_data.num_channels,
                "is_multi_channel": img_data.is_multi_channel,
                "channel_names": getattr(img_data, 'channel_names', []),
                "dtype": str(img_data.dtype),
                "format": "multi-channel TIFF" if img_data.is_multi_channel else "single-channel"
            }
        except Exception as e:
            logger.debug(f"ImageProcessor failed, using PIL fallback: {e}")
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                return {
                    "width": width,
                    "height": height,
                    "num_channels": 1,
                    "is_multi_channel": False
                }
            except Exception:
                return {}
    
    def get_image_info_bytes(self, bytes: str) -> Dict[str, Any]:
        """Extract image info from bytes (for compatibility)."""
        try:
            with Image.open(BytesIO(bytes)) as img:
                width, height = img.size
            return {
                "width": width,
                "height": height,
                "num_channels": 1,
                "is_multi_channel": False
            }
        except Exception:
            return {}
    
    def _format_memory(self, memory: Memory, max_actions: int = 10) -> str:
        """Format memory for prompt - concise and relevant."""
        actions = memory.get_actions(llm_safe=True)
        if not actions:
            return "No previous steps."
        
        recent = actions[-max_actions:]
        formatted = []
        for i, action in enumerate(recent, 1):
            tool = action.get('tool_name', 'Unknown')
            goal = action.get('sub_goal', '')
            result = action.get('result', {})
            
            # Truncate long results
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "... (truncated)"
            
            formatted.append(
                f"Step {i}: {tool}\n"
                f"  Goal: {goal[:200] if len(goal) > 200 else goal}\n"
                f"  Result: {result_str}"
            )
        
        if len(actions) > max_actions:
            formatted.insert(0, f"[Showing last {max_actions} of {len(actions)} steps. Earlier steps omitted to save context.]")
        
        return "\n\n".join(formatted)
    
    def _format_memory_for_prompt(self, memory: Memory, max_recent_actions: int = 10) -> str:
        """Alias for _format_memory (for compatibility)."""
        return self._format_memory(memory, max_recent_actions)
    
    def analyze_query(self, question: str, image: str, bytes_mode: bool = False,
                     conversation_context: str = "", group_images: List[Dict[str, Any]] = None,
                     **kwargs) -> str:
        """
        Analyze query to understand intent and requirements.
        Returns structured query analysis for display and planning.
        """
        image_info = self.get_image_info(image) if not bytes_mode else {}
        
        # Detect domain and filter tools
        self.detected_domain = self.priority_manager.detect_task_domain(question, "")
        available_tools, excluded_tools = self.priority_manager.filter_tools_for_domain(
            self.available_tools, self.detected_domain, exclude_excluded=True
        )
        
        # Multi-image and group context
        num_images = len(group_images) if group_images else 1
        groups = set()
        if group_images:
            groups = {img.get("group", "default") for img in group_images if img.get("group")}
        is_group_comparison = len(groups) > 1 if groups else False
        
        # Filter metadata
        toolbox_metadata = {
            tool: self.toolbox_metadata[tool]
            for tool in available_tools
            if tool in self.toolbox_metadata
        }
        
        # Build multi-image context
        multi_image_context = ""
        if num_images > 1:
            groups_summary = ", ".join(sorted(groups)) if groups else "default"
            multi_image_context = f"""
MULTI-IMAGE CONTEXT:
- Number of images: {num_images}
- Groups detected: {len(groups)} ({groups_summary})
- Group comparison: {'YES' if is_group_comparison else 'NO (all images in same group)'}

MULTI-IMAGE PROCESSING:
- Per-image tools: Process each image independently (segmenters, croppers)
- Merge-all tools: Execute once, merging all images with group labels (analyzers, visualizers)
- For comparison: Analysis tool extracts features → Visualization tool displays results
"""
        
        # Enhanced prompt with structured output
        prompt = f"""Analyze this bioimage analysis query and determine the minimum necessary skills and tools.

Conversation so far:
{conversation_context}

Query: {question}
Image info: {image_info}
{multi_image_context}

Available tools: {available_tools}
Tool metadata: {toolbox_metadata}

CRITICAL INSTRUCTIONS:
1. Understand the query precisely - identify EXACTLY what is being asked
2. Determine the SINGLE main objective:
   - "How many cells/organoids" → Only counting needed (segmentation + counting, may visualize counts)
   - "What cell states" or "analyze cell states" → Full cell state analysis (preprocessing → segmentation → cropping → analysis → visualization)
   - "Compare" or "morphology" → Full pipeline (analysis tool extracts features, visualizer displays)
   - Analysis_Visualizer_Tool requires pre-computed results from Cell_State_Analyzer_*_Tool
3. Select minimum necessary tools based on query requirements
4. Do not skip analysis tool for morphology/comparison queries
5. Use intelligent planning: Consider whether additional context (e.g., image type analysis) would help, but prioritize efficiency - if the task can be accomplished directly with specialized tools, use them directly

Provide your analysis in structured format:
- Concise Summary: Brief summary of what the query asks
- Query Type: One of [Simple Counting / Basic Morphology / Cell State Analysis / Comparison]
- Required Analysis Pipeline: Describe steps needed (e.g., "Image preprocessing (optional) → Segmentation → Count extraction")
- Key Requirements: Specific requirements, constraints, considerations
- Expected Tools: Tools likely needed in order

Provide a clear, actionable analysis that helps plan the bioimage analysis workflow."""
        
        # Try structured format first, fallback to string
        try:
            response = self.llm_engine_mm.generate([prompt], response_format=QueryAnalysis)
            if isinstance(response, dict) and 'content' in response:
                self.query_analysis = response['content']
                self.last_usage = response.get('usage', {})
            else:
                self.query_analysis = str(response)
        except Exception as e:
            logger.debug(f"Structured format failed, using text: {e}")
            response = self.llm_engine.generate(prompt, max_tokens=500)
            if isinstance(response, dict) and 'content' in response:
                self.query_analysis = response['content']
            elif isinstance(response, str):
                self.query_analysis = response
            else:
                self.query_analysis = str(response)
        
        return str(self.query_analysis)
    
    def generate_next_step(self, question: str, image: str, query_analysis: str,
                           memory: Memory, step_count: int, max_step_count: int,
                           bytes_mode: bool = False, conversation_context: str = "",
                           group_images: List[Dict[str, Any]] = None, **kwargs) -> NextStep:
        """
        Generate next step using intelligent LLM-based planning.
        
        Core approach:
        - Provide rich context (query, memory, tools, metadata)
        - Guide through prompts with bioimage analysis chain awareness
        - Trust LLM's intelligence for tool selection
        - Support multi-image and group comparison scenarios
        """
        # Ensure query_analysis is a string
        if not isinstance(query_analysis, str):
            query_analysis = str(query_analysis) if query_analysis else ""
        
        # Get context
        image_info = self.get_image_info(image) if not bytes_mode else {}
        detected_domain = self.priority_manager.detect_task_domain(question, query_analysis)
        available_tools, excluded_tools = self.priority_manager.filter_tools_for_domain(
            self.available_tools, detected_domain, exclude_excluded=True
        )
        
        # Get used tools
        actions = memory.get_actions(llm_safe=True)
        used_tools = [a.get('tool_name', '') for a in actions if 'tool_name' in a]
        
        # Filter metadata
        toolbox_metadata = {
            tool: self.toolbox_metadata[tool]
            for tool in available_tools
            if tool in self.toolbox_metadata
        }
        
        # Multi-image context
        num_images = len(group_images) if group_images else 1
        groups = set()
        if group_images:
            groups = {img.get("group", "default") for img in group_images if img.get("group")}
        is_group_comparison = len(groups) > 1 if groups else False
        
        # Build multi-image prompt section
        multi_image_prompt = ""
        if num_images > 1:
            groups_summary = ", ".join(sorted(groups)) if groups else "default"
            multi_image_prompt = f"""
MULTI-IMAGE PROCESSING CONTEXT:
- Processing {num_images} image(s) across {len(groups)} group(s): {groups_summary}
- Group comparison enabled: {'YES - tools should compare groups statistically' if is_group_comparison else 'NO'}
- Tool execution modes:
  * Per-image tools (segmenters, croppers): Process each of {num_images} images independently
  * Merge-all tools (Cell_State_Analyzer_*_Tool, Analysis_Visualizer_Tool): Execute ONCE, merging all crops from all {num_images} images with group labels
- Cell_State_Analyzer_*_Tool automatically loads all crop images from Single_Cell_Cropper_Tool metadata (from all {num_images} images)
"""
        
        # Get tools by priority
        tools_by_priority = self.priority_manager.format_tools_by_priority(available_tools)
        
        # Get recommended next tools
        recommended_tools = self.priority_manager.get_recommended_next_tools(
            available_tools, used_tools, detected_domain
        )
        recommended_str = ", ".join(recommended_tools[:5]) if recommended_tools else "None"
        
        # Build intelligent prompt with bioimage analysis chain awareness
        prompt = f"""You are an intelligent planner for a bioimage analysis system specializing in SINGLE-CELL level analysis.
Your goal: Select the optimal next tool to progress toward answering the query.

QUERY: {question}
QUERY ANALYSIS: {query_analysis}
IMAGE INFO: {image_info}
STEP: {step_count}/{max_step_count}
{multi_image_prompt}

PREVIOUS STEPS:
{self._format_memory(memory)}

AVAILABLE TOOLS (by priority):
{tools_by_priority}

RECOMMENDED NEXT TOOLS: {recommended_str}

TOOL METADATA (includes dependencies and capabilities):
{self._format_tool_metadata(toolbox_metadata)}

CRITICAL: Bioimage Analysis Chain for Single-Cell Analysis

This system performs SINGLE-CELL level analysis. For queries involving cell states, cell counts, or cell-level comparisons:

**Tool Role & Pipeline:**
- Analysis Tools: Cell_State_Analyzer_*_Tool analyzes individual cell/organoid crops - REQUIRES Single_Cell_Cropper_Tool output
- Visualization Tool: Analysis_Visualizer_Tool ONLY visualizes pre-computed analysis results

**Pipeline Flow:**
1. Preprocessing (optional) → Image_Preprocessor_Tool
2. Segmentation → [Cell/Nuclei/Organoid_Segmenter_Tool] → produces masks
3. Cropping → Single_Cell_Cropper_Tool → REQUIRED: produces individual crop images (one per cell/organoid)
4. Analysis → Cell_State_Analyzer_Single_Tool (1 channel) | Cell_State_Analyzer_Multi_Tool (2+ channels)
   - Input: Individual crop images from step 3 (NOT original images or masks)
   - Processes each crop to extract features, perform clustering
5. Visualization → Analysis_Visualizer_Tool

**CRITICAL Input Requirements:**
- Cell_State_Analyzer_*_Tool REQUIRES cell_crops parameter: List of paths to individual crop images from Single_Cell_Cropper_Tool
- Input format: cell_crops=["path/to/crop1.tiff", "path/to/crop2.tiff", ...] (individual crop images)
- Single_Cell_Cropper_Tool MUST be executed first - it produces the crop images that Cell_State_Analyzer_*_Tool needs
- When Single_Cell_Cropper_Tool completes, it saves metadata in query_cache_dir/tool_cache/ - Cell_State_Analyzer_*_Tool automatically loads crops from this metadata

**Query → Pipeline Mapping:**
- Counting: Segmentation → Count (visualize if needed)
- Morphology/Comparison/States: Full pipeline (Segmentation → Cropping → Analysis → Visualization)
- Analysis tools analyze individual crops - cannot skip cropping step

**Intelligent Planning Guidelines:**
- For counting queries: Segmentation tools (Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, Organoid_Segmenter_Tool) are designed to handle counting directly. Consider whether additional image analysis would add value or if direct segmentation is more efficient.
- For ambiguous image types: Segmentation tools have built-in detection capabilities. Consider whether explicit image analysis would significantly improve results or if direct segmentation with auto-detection is sufficient.
- Tool selection should balance: (1) Task efficiency - can the task be accomplished directly? (2) Result quality - would additional context improve outcomes? (3) User experience - minimize unnecessary steps while ensuring accuracy.
- Image_Captioner_Tool can provide valuable context for complex or ambiguous cases, but for standard bioimage workflows, specialized tools often work more efficiently.

TASK DOMAIN AWARENESS:

**Knowledge Domain (Non-Image Analysis):**
The system has detected this as a knowledge-based task (literature mining, gene annotation, pathway enrichment, functional analysis, etc.).
- **CRITICAL: Use actual search tools FIRST to retrieve real knowledge from databases**
  - For literature mining: Use Pubmed_Search_Tool to search PubMed for relevant articles
  - For general knowledge: Use Google_Search_Tool to find current information
  - For academic papers: Use ArXiv_Paper_Searcher_Tool for preprints
  - These tools provide REAL, VERIFIABLE information with actual citations
- **Then use Generalist_Solution_Generator_Tool** to synthesize the search results into comprehensive answers
- These tasks require knowledge retrieval from external sources, NOT just language model knowledge
- Examples: "Perform literature mining for gene clusters" → First use Pubmed_Search_Tool to find papers, then use Generalist_Solution_Generator_Tool to synthesize
- Workflow: Search tools (Pubmed_Search_Tool, Google_Search_Tool) → Generalist_Solution_Generator_Tool (synthesis with references)

**Bioimage Domain (Image Analysis):**
For image analysis tasks, follow the bioimage analysis chain below.

INSTRUCTIONS:
1. Review the query analysis and detected domain to understand what type of analysis is needed
2. For knowledge domain: 
   - FIRST: Use search tools (Pubmed_Search_Tool, Google_Search_Tool, ArXiv_Paper_Searcher_Tool) to retrieve REAL knowledge from databases
   - THEN: Use Generalist_Solution_Generator_Tool to synthesize search results with proper citations
   - DO NOT use Generalist_Solution_Generator_Tool alone - it should synthesize actual search results
3. For bioimage domain: Check what has been done (PREVIOUS STEPS) and what is still needed
4. For bioimage domain: Follow the bioimage analysis chain appropriate for the query type
5. Use intelligent reasoning: Evaluate whether each step adds value or if a more direct approach would be better
6. Select ONE tool that is the logical next step for the identified domain
7. Ensure dependencies are satisfied (e.g., need segmentation before cropping, need cropping before analysis)
8. For Cell_State_Analyzer_*_Tool: Do NOT pass original images or masks - tool automatically loads crop images from Single_Cell_Cropper_Tool metadata (query_cache_dir parameter)
9. Formulate a clear sub-goal explaining what this tool will accomplish, referencing the correct input format

Output format:
<justification>: Why this tool is the best next step, referencing the analysis pipeline
<context>: All necessary information from previous steps (file paths, data, variables)
<sub_goal>: Specific, achievable objective for this tool
<tool_name>: Exact tool name from available tools list"""
        
        # Generate next step
        try:
            next_step = self.llm_engine.generate(prompt, response_format=NextStep)
        except Exception as e:
            logger.warning(f"Structured format failed, using text: {e}")
            # Fallback to text generation
            response = self.llm_engine.generate(prompt, max_tokens=500)
            # Parse manually if needed
            next_step = response
        
        # Extract and store usage
        if isinstance(next_step, dict) and 'content' in next_step:
            result = next_step['content']
            self.last_usage = next_step.get('usage', {})
        else:
            result = next_step
            self.last_usage = {}
        
        # Post-LLM enforcement: Check tool-declared dependencies
        # Intelligent approach: Tools declare their required_next_tools, Planner enforces them
        actions = memory.get_actions(llm_safe=True)
        used_tools = [a.get('tool_name', '') for a in actions if 'tool_name' in a]
        
        # Check the last tool's result for required_next_tools
        if actions:
            last_action = actions[-1]
            last_result = last_action.get('result', {})
            if isinstance(last_result, dict):
                # If last tool declares no required_next_tools and can_terminate_after_chain, prevent further tool selection
                required_next = last_result.get('required_next_tools', [])
                can_terminate = last_result.get('can_terminate_after_chain', False)
                
                if not required_next and can_terminate:
                    # Tool chain is complete, prevent further tool selection
                    # Return None to signal completion (this will be handled by the caller)
                    logger.info(f"Tool chain complete after {last_action.get('tool_name', 'Unknown')}, preventing further tool selection")
                    # Note: We can't return None here as it would break the flow
                    # Instead, we'll let verificate_memory handle termination
                
                if required_next:
                    # Find the first required tool that hasn't been used
                    for req_tool in required_next:
                        if req_tool not in used_tools:
                            # Force the required tool as next step
                            if hasattr(result, 'tool_name'):
                                if result.tool_name != req_tool:
                                    logger.info(f"Enforcing {req_tool} as required by {last_action.get('tool_name', 'Unknown')}")
                                    result.tool_name = req_tool
                                    result.sub_goal = f"Complete workflow by running {req_tool} as required by previous tool"
                                    result.justification = f"{req_tool} is required by {last_action.get('tool_name', 'Unknown')} to complete the analysis workflow"
                            elif isinstance(result, dict):
                                if result.get('tool_name') != req_tool:
                                    logger.info(f"Enforcing {req_tool} as required by {last_action.get('tool_name', 'Unknown')}")
                                    result['tool_name'] = req_tool
                                    result['sub_goal'] = f"Complete workflow by running {req_tool} as required by previous tool"
                                    result['justification'] = f"{req_tool} is required by {last_action.get('tool_name', 'Unknown')} to complete the analysis workflow"
                            break  # Only enforce the first uncompleted required tool
        
        return result
    
    def _format_tools_list(self, available_tools: List[str], used_tools: List[str]) -> str:
        """Format tools list with usage indicators."""
        used_set = set(used_tools)
        lines = []
        for tool in available_tools:
            status = "✓ Used" if tool in used_set else "Available"
            lines.append(f"  - {tool} ({status})")
        return "\n".join(lines) if lines else "No tools available"
    
    def _format_tool_metadata(self, toolbox_metadata: Dict[str, Any]) -> str:
        """Format tool metadata concisely."""
        if not toolbox_metadata:
            return "No tool metadata available"
        
        lines = []
        for tool_name, metadata in list(toolbox_metadata.items())[:10]:
            desc = metadata.get('description', 'No description')[:200]
            # Emphasize critical role distinctions and input requirements
            if 'Cell_State_Analyzer' in tool_name:
                lines.append(f"{tool_name}: {desc} [REQUIRES: cell crops from Single_Cell_Cropper_Tool]")
            elif tool_name == "Analysis_Visualizer_Tool":
                lines.append(f"{tool_name}: {desc} [VISUALIZATION ONLY: requires pre-computed analysis results]")
            else:
                lines.append(f"{tool_name}: {desc}")
        
        if len(toolbox_metadata) > 10:
            lines.append(f"... and {len(toolbox_metadata) - 10} more tools")
        
        return "\n".join(lines)
    
    def verificate_memory(self, question: str, image: str, query_analysis: str,
                         memory: Memory, bytes_mode: bool = False,
                         conversation_context: str = "", **kwargs) -> MemoryVerification:
        """Verify if memory contains sufficient information to answer query."""
        # Ensure query_analysis is a string
        if not isinstance(query_analysis, str):
            query_analysis = str(query_analysis) if query_analysis else ""
        
        # Check for termination signals from tools BEFORE calling LLM
        # Intelligent approach: Tools declare their dependencies via required_next_tools
        # Only allow termination after all required_next_tools are completed
        actions = memory.get_actions(llm_safe=True)
        used_tools = [a.get('tool_name', '') for a in actions if 'tool_name' in a]
        
        # Check if any tool has uncompleted required_next_tools
        has_uncompleted_requirements = False
        for action in actions:
            result = action.get('result', {})
            if isinstance(result, dict):
                required_next = result.get('required_next_tools', [])
                if required_next:
                    # Check if all required tools have been used
                    for req_tool in required_next:
                        if req_tool not in used_tools:
                            has_uncompleted_requirements = True
                            break
                    if has_uncompleted_requirements:
                        break
        
        # If there are uncompleted required_next_tools, don't check termination yet
        if not has_uncompleted_requirements:
            # Check for termination signals only after all required_next_tools are completed
            for action in actions:
                result = action.get('result', {})
                if isinstance(result, dict):
                    # Check for can_terminate_after_chain field (new name, more semantic)
                    # Also support legacy termination_recommended for backward compatibility
                    can_terminate = result.get('can_terminate_after_chain', False) or result.get('termination_recommended', False)
                    if can_terminate:
                        termination_reason = result.get('termination_reason', '')
                        summary = result.get('summary', '')
                        tool_name = action.get('tool_name', 'Unknown Tool')
                        
                        # Build analysis message explaining why termination is recommended
                        analysis = (
                            f"**Tool Termination Signal Detected:**\n\n"
                            f"Tool '{tool_name}' has indicated that after completing the tool chain, "
                            f"the analysis results cannot fully answer the query requirements.\n\n"
                            f"**Reason:**\n{termination_reason}\n\n"
                            f"**Tool Summary:**\n{summary}\n\n"
                            f"**Recommendation:** Terminate analysis and inform user about the limitations. "
                            f"This is not an error, but a signal that the current data/configuration cannot answer the query."
                        )
                        
                        # Return stop signal
                        return MemoryVerification(analysis=analysis, stop_signal=True)
        
        image_info = self.get_image_info(image) if not bytes_mode else {}
        
        # Check detected domain for knowledge tasks
        detected_domain = getattr(self, 'detected_domain', 'general')
        is_knowledge_domain = (detected_domain == 'knowledge')
        
        prompt = f"""Verify if the following steps contain sufficient information to answer the query.

Conversation so far:
{conversation_context}

Query: {question}
Query Analysis: {query_analysis}
Image: {image_info}
Detected Domain: {detected_domain}

Previous Steps:
{self._format_memory(memory)}

{"**KNOWLEDGE DOMAIN TASK:**" if is_knowledge_domain else ""}
{"For knowledge-based tasks (literature mining, gene annotation, pathway enrichment, functional analysis):" if is_knowledge_domain else ""}
{"- CRITICAL: Knowledge tasks require a TWO-STEP workflow:" if is_knowledge_domain else ""}
{"  1. Search tools (Pubmed_Search_Tool, Google_Search_Tool) retrieve REAL knowledge from databases" if is_knowledge_domain else ""}
{"  2. Generalist_Solution_Generator_Tool synthesizes search results into comprehensive answers with citations" if is_knowledge_domain else ""}
{"- DO NOT stop after only search tools - synthesis step is REQUIRED" if is_knowledge_domain else ""}
{"- For literature mining: Must have BOTH (1) search results from Pubmed_Search_Tool AND (2) synthesis from Generalist_Solution_Generator_Tool" if is_knowledge_domain else ""}
{"- For gene annotation: Must have BOTH (1) search results AND (2) synthesized annotations with references" if is_knowledge_domain else ""}
{"- STOP only when BOTH search AND synthesis steps are complete, with comprehensive answers including references" if is_knowledge_domain else ""}
{"- If only search tools have been used, CONTINUE to use Generalist_Solution_Generator_Tool for synthesis" if is_knowledge_domain else ""}

CRITICAL: Understand the query type and verify ONLY what is explicitly asked:
- If query asks "how many cells" → Verify that cell count is available (from segmentation result) → STOP if count exists
- If query asks "what cell states" → Verify that cell state analysis (clustering + visualization) is complete → STOP if complete
- If query asks "compare" → Verify that comparison analysis is complete → STOP if complete
- DO NOT require information beyond what is explicitly asked in the query

CRITICAL: Tool Chain Completion Rules:
- If Analysis_Visualizer_Tool has been executed → The analysis pipeline is COMPLETE → STOP
- Analysis_Visualizer_Tool is the final step in the cell state analysis pipeline (Segmentation → Cropping → Cell_State_Analyzer → Analysis_Visualizer)
- After Analysis_Visualizer_Tool completes, no further tools are needed for the analysis workflow
- DO NOT re-run Cell_State_Analyzer after Analysis_Visualizer_Tool has completed

CRITICAL CHECKLIST FOR STOPPING:
- Has the MAIN query been answered EXACTLY as asked? If yes, STOP immediately.
- For "how many cells" queries: If cell count is available (from segmentation), STOP.
- For "what cell states" queries: If cell state analysis (clustering + visualization) is complete, STOP.
- If Analysis_Visualizer_Tool has been executed → STOP (pipeline complete)
- Are there analysis results, visualizations, counts, or statistics that DIRECTLY answer the query? If yes, STOP.
- DO NOT continue just because there are unused tools available.
- DO NOT re-run tools that have already completed their workflow.

Determine if the query can be answered with the information available, or if more steps are needed."""
        
        # Try structured format first
        try:
            input_data = [prompt]
            if image_info and 'image_path' in image_info:
                try:
                    with open(image_info["image_path"], 'rb') as f:
                        input_data.append(f.read())
                except Exception:
                    pass
            
            response = self.llm_engine_mm.generate(input_data, response_format=MemoryVerification)
            if isinstance(response, dict) and 'content' in response:
                result = response['content']
                self.last_usage = response.get('usage', {})
            else:
                result = response
                self.last_usage = {}
            
            # Handle string response
            if isinstance(result, str):
                analysis, stop_signal = ResponseParser.parse_memory_verification(result)
                result = MemoryVerification(analysis=analysis, stop_signal=stop_signal)
            
            return result
        except Exception as e:
            logger.warning(f"Structured format failed, using text: {e}")
            # Fallback
            response = self.llm_engine.generate(prompt, max_tokens=500)
            if isinstance(response, dict) and 'content' in response:
                content = response['content']
            else:
                content = str(response)
            
            # Parse manually
            analysis, stop_signal = ResponseParser.parse_memory_verification(content)
            return MemoryVerification(analysis=analysis, stop_signal=stop_signal)
    
    def generate_final_output(self, question: str, image: str, memory: Memory,
                             bytes_mode: bool = False, conversation_context: str = "",
                             **kwargs) -> str:
        """Generate final answer from memory."""
        image_info = self.get_image_info(image) if not bytes_mode else {}
        
        prompt = f"""Based on the following analysis steps, provide a comprehensive answer to the query.

Conversation so far:
{conversation_context}

Query: {question}
Image: {image_info}

Analysis Steps:
{self._format_memory(memory, max_actions=20)}

Instructions:
1. Review the query and all actions taken
2. Consider results from each tool execution
3. Incorporate relevant information from memory
4. Generate a well-organized, step-by-step final output
5. CRITICAL: The Summary section always provide a summary even if results are incomplete

Output Structure:
1. Summary: Brief overview of query and main findings
   - If results are complete: summarize the key findings
   - If results are incomplete: summarize what was found and what is missing
   - Always include morphological features if available (area, perimeter, circularity, etc.)
2. Detailed Analysis: Step-by-step process breakdown
3. Key Findings: Most important discoveries
   - Include morphological measurements if available
   - Include statistical comparisons if performed
4. Answer to Query: Direct, clear answer to original question
5. Additional Insights: Relevant information beyond direct answer
6. Conclusion: Summary and potential next steps

CRITICAL: The Summary section MUST NOT be empty. Even if results are incomplete, provide a summary of:
- What analysis was performed
- What morphological features were extracted (if any)
- What statistical comparisons were made (if any)
- What limitations exist (if any)

Provide a clear answer to the query based on the analysis results."""
        
        input_data = [prompt]
        if image_info and 'image_path' in image_info:
            try:
                with open(image_info["image_path"], 'rb') as f:
                    input_data.append(f.read())
            except Exception:
                pass
        
        response = self.llm_engine_mm.generate(input_data)
        if isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def generate_base_response(self, question: str, image: str, max_tokens: str = "4000",
                              bytes_mode: bool = False) -> str:
        """Generate base response (for compatibility)."""
        prompt = f"Answer this question about the image: {question}"
        response = self.llm_engine.generate(prompt, max_tokens=int(max_tokens))
        if isinstance(response, dict) and 'content' in response:
            self.base_response = response['content']
        elif isinstance(response, str):
            self.base_response = response
        else:
            self.base_response = str(response)
        return self.base_response
    
    def extract_context_subgoal_and_tool(self, response) -> Tuple[str, str, str]:
        """Extract context, sub_goal, and tool_name from LLM response."""
        try:
            context, sub_goal, tool_name = ResponseParser.parse_next_step(response, self.available_tools)
            logger.debug(f"Extracted: context='{context[:50]}...', sub_goal='{sub_goal[:50]}...', tool_name='{tool_name}'")
            return context, sub_goal, tool_name
        except Exception as e:
            logger.error(f"Error in extract_context_subgoal_and_tool: {e}")
            return "", "", "Error extracting tool name"
    
    def extract_conclusion(self, response: MemoryVerification) -> Tuple[str, str]:
        """Extract context_verification and conclusion from verification response."""
        try:
            if hasattr(ResponseParser, 'parse_memory_verification'):
                analysis, stop_signal = ResponseParser.parse_memory_verification(response)
                conclusion = 'STOP' if stop_signal else 'CONTINUE'
                return analysis, conclusion
        except Exception:
            pass
        
        # Fallback: extract from response object
        if isinstance(response, dict):
            context_verification = response.get('context_verification', response.get('analysis', ''))
            conclusion = response.get('conclusion', response.get('stop_signal', 'CONTINUE'))
        elif hasattr(response, 'context_verification') and hasattr(response, 'conclusion'):
            context_verification = response.context_verification
            conclusion = response.conclusion
        elif hasattr(response, 'analysis') and hasattr(response, 'stop_signal'):
            context_verification = response.analysis
            conclusion = 'STOP' if response.stop_signal else 'CONTINUE'
        else:
            context_verification = str(response) if response else ''
            conclusion = 'CONTINUE'
        return context_verification, conclusion
    
    def generate_direct_output(self, question: str, image: str, memory: Memory,
                              bytes_mode: bool = False, conversation_context: str = "",
                              **kwargs) -> str:
        """Generate direct output (alias for generate_final_output)."""
        return self.generate_final_output(question, image, memory, bytes_mode, conversation_context, **kwargs)

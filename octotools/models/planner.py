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

IMPORTANT FOR MULTI-IMAGE PROCESSING:
- Per-image tools (segmenters, croppers): Process each of {num_images} images independently
- Merge-all tools (Cell_State_Analyzer_Tool, Analysis_Visualizer_Tool): Execute ONCE, merging all {num_images} images with group labels
- For group comparison queries: Ensure using merge-all tools after per-image processing for statistical comparison
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
   - "How many cells" → Only counting needed (segmentation + counting)
   - "What cell states" or "analyze cell states" → Full cell state analysis (preprocessing → segmentation → cropping → clustering → visualization)
   - "Compare images" or "compare groups" → Comparison analysis (requires cell state analysis for meaningful comparison)
3. Select MINIMUM necessary tools - only what is DIRECTLY required
4. DO NOT over-extend - if query asks for count, don't suggest cell state analysis

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
  * Merge-all tools (Cell_State_Analyzer_Tool, Analysis_Visualizer_Tool): Execute ONCE, merging all {num_images} images with group labels
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

**Standard Pipeline for Cell-Level Analysis:**
1. Image Preprocessing (optional) → Image_Preprocessor_Tool
2. Segmentation → [Cell_Segmenter_Tool | Nuclei_Segmenter_Tool | Organoid_Segmenter_Tool]
3. Single Cell Cropping → Single_Cell_Cropper_Tool (MANDATORY for cell-level analysis)
4. Cell State Analysis → [Cell_State_Analyzer_Single_Tool | Cell_State_Analyzer_Multi_Tool]
   - Use Cell_State_Analyzer_Single_Tool for single-channel images (1 channel)
   - Use Cell_State_Analyzer_Multi_Tool for multi-channel images (2+ channels)
5. Visualization → Analysis_Visualizer_Tool

**Query-Specific Guidelines:**
- "cell count" or "how many cells": Segmentation → Count (may skip cropping if only counting)
- "cell states", "cell types", "clustering", "UMAP": FULL pipeline (Segmentation → Cropping → Analysis → Visualization)
- "compare" queries at cell level: FULL pipeline with group comparison
- Basic morphology (area, size): Segmentation → (optional Cropping) → Visualization

**Tool Selection Rules:**
1. If query involves cell states/cell types → MUST use full pipeline (cannot skip cropping or analysis)
2. If Single_Cell_Cropper_Tool was used → MUST use Cell_State_Analyzer next (check channel count for Single vs Multi)
3. If Cell_State_Analyzer was used → MUST use Analysis_Visualizer_Tool next
4. Check image channel count: multi-channel (2+) → Cell_State_Analyzer_Multi_Tool, single-channel → Cell_State_Analyzer_Single_Tool

INSTRUCTIONS:
1. Review the query analysis to understand what type of analysis is needed
2. Check what has been done (PREVIOUS STEPS) and what is still needed
3. Follow the bioimage analysis chain appropriate for the query type
4. Select ONE tool that is the logical next step in the pipeline
5. Ensure dependencies are satisfied (e.g., need segmentation before cropping)
6. Formulate a clear sub-goal explaining what this tool will accomplish

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
        
        image_info = self.get_image_info(image) if not bytes_mode else {}
        
        prompt = f"""Verify if the following steps contain sufficient information to answer the query.

Conversation so far:
{conversation_context}

Query: {question}
Query Analysis: {query_analysis}
Image: {image_info}

Previous Steps:
{self._format_memory(memory)}

CRITICAL: Understand the query type and verify ONLY what is explicitly asked:
- If query asks "how many cells" → Verify that cell count is available (from segmentation result) → STOP if count exists
- If query asks "what cell states" → Verify that cell state analysis (clustering + visualization) is complete → STOP if complete
- If query asks "compare" → Verify that comparison analysis is complete → STOP if complete
- DO NOT require information beyond what is explicitly asked in the query

CRITICAL CHECKLIST FOR STOPPING:
- Has the MAIN query been answered EXACTLY as asked? If yes, STOP immediately.
- For "how many cells" queries: If cell count is available (from segmentation), STOP.
- For "what cell states" queries: If cell state analysis (clustering + visualization) is complete, STOP.
- Are there analysis results, visualizations, counts, or statistics that DIRECTLY answer the query? If yes, STOP.
- DO NOT continue just because there are unused tools available.

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

Output Structure:
1. Summary: Brief overview of query and main findings
2. Detailed Analysis: Step-by-step process breakdown
3. Key Findings: Most important discoveries
4. Answer to Query: Direct, clear answer to original question
5. Additional Insights: Relevant information beyond direct answer
6. Conclusion: Summary and potential next steps

Provide a clear, complete answer to the query based on the analysis results."""
        
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

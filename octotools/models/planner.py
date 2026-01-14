"""
Simplified Planner - Intelligent Goal-Oriented Planning

Inspired by Biomni's planning strategy:
- Goal-Oriented Action Planning (GOAP)
- Minimal hardcoded rules
- LLM-driven intelligent decision making
- Simplified, maintainable code
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
    Intelligent planner that uses LLM to make goal-oriented decisions.
    
    Core philosophy:
    - Trust LLM's intelligence for tool selection
    - Provide rich context, not rigid rules
    - Let tool metadata guide dependencies
    - Minimize hardcoded enforcement
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
        
        # Tool priority manager (for filtering, not enforcement)
        self.priority_manager = ToolPriorityManager()
        
        # State
        self.query_analysis = None
        self.last_usage = {}
        self.base_response = None  # For compatibility
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Extract image metadata for context."""
        if not image_path or not os.path.isfile(image_path):
            return {}
        
        try:
            img_data = ImageProcessor.load_image(image_path)
            return {
                "width": img_data.shape[1],
                "height": img_data.shape[0],
                "num_channels": img_data.num_channels,
                "is_multi_channel": img_data.is_multi_channel,
                "format": "multi-channel TIFF" if img_data.is_multi_channel else "single-channel"
            }
        except Exception as e:
            logger.debug(f"Image info extraction failed: {e}")
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                return {"width": width, "height": height, "num_channels": 1, "is_multi_channel": False}
            except Exception:
                return {}
    
    def get_image_info_bytes(self, bytes: str) -> Dict[str, Any]:
        """Extract image info from bytes (for compatibility)."""
        try:
            with Image.open(BytesIO(bytes)) as img:
                width, height = img.size
            return {"width": width, "height": height, "num_channels": 1, "is_multi_channel": False}
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
            if len(result_str) > 300:
                result_str = result_str[:300] + "..."
            
            formatted.append(
                f"Step {i}: {tool}\n"
                f"  Goal: {goal[:150] if len(goal) > 150 else goal}\n"
                f"  Result: {result_str}"
            )
        
        if len(actions) > max_actions:
            formatted.insert(0, f"[Showing last {max_actions} of {len(actions)} steps]")
        
        return "\n\n".join(formatted)
    
    def _format_memory_for_prompt(self, memory: Memory, max_recent_actions: int = 10) -> str:
        """Alias for _format_memory (for compatibility)."""
        return self._format_memory(memory, max_recent_actions)
    
    def analyze_query(self, question: str, image: str, bytes_mode: bool = False,
                     conversation_context: str = "", group_images: List[Dict[str, Any]] = None,
                     **kwargs) -> str:
        """
        Analyze query to understand intent and requirements.
        Returns query analysis string for use in planning.
        """
        image_info = self.get_image_info(image) if not bytes_mode else {}
        
        # Detect domain and filter tools
        detected_domain = self.priority_manager.detect_task_domain(question, "")
        available_tools, _ = self.priority_manager.filter_tools_for_domain(
            self.available_tools, detected_domain, exclude_excluded=True
        )
        
        # Multi-image context
        num_images = len(group_images) if group_images else 1
        groups = set()
        if group_images:
            groups = {img.get("group", "default") for img in group_images if img.get("group")}
        
        # Simple, focused prompt
        prompt = f"""Analyze this bioimage analysis query and determine:
1. Query type (counting, morphology, cell state analysis, comparison)
2. Required analysis depth (minimal, intermediate, full pipeline)
3. Key requirements and constraints

Query: {question}
Image info: {image_info}
Number of images: {num_images}
Groups: {', '.join(sorted(groups)) if groups else 'single group'}

Provide a concise analysis focusing on what needs to be done to answer the query."""
        
        response = self.llm_engine.generate(prompt, max_tokens=500)
        # Handle dict response (with 'content' key) or direct string
        if isinstance(response, dict) and 'content' in response:
            self.query_analysis = response['content']
        elif isinstance(response, str):
            self.query_analysis = response
        else:
            self.query_analysis = str(response)
        return self.query_analysis
    
    def generate_next_step(self, question: str, image: str, query_analysis: str,
                           memory: Memory, step_count: int, max_step_count: int,
                           bytes_mode: bool = False, conversation_context: str = "",
                           group_images: List[Dict[str, Any]] = None, **kwargs) -> NextStep:
        """
        Generate next step using intelligent LLM-based planning.
        
        Core approach:
        - Provide rich context (query, memory, tools, metadata)
        - Let LLM understand dependencies from tool metadata
        - Trust LLM's intelligence for tool selection
        - Minimal hardcoded rules
        """
        # Ensure query_analysis is a string
        if not isinstance(query_analysis, str):
            query_analysis = str(query_analysis) if query_analysis else ""
        
        # Get context
        image_info = self.get_image_info(image) if not bytes_mode else {}
        detected_domain = self.priority_manager.detect_task_domain(question, query_analysis)
        available_tools, _ = self.priority_manager.filter_tools_for_domain(
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
        
        # Build intelligent prompt - concise but informative
        prompt = f"""You are an intelligent planner for a bioimage analysis system. 
Your goal: Select the optimal next tool to progress toward answering the query.

QUERY: {question}
QUERY ANALYSIS: {query_analysis}
IMAGE INFO: {image_info}
STEP: {step_count}/{max_step_count}
{'MULTI-IMAGE: ' + str(num_images) + ' images, groups: ' + ', '.join(sorted(groups)) if num_images > 1 else ''}

PREVIOUS STEPS:
{self._format_memory(memory)}

AVAILABLE TOOLS:
{self._format_tools_list(available_tools, used_tools)}

TOOL METADATA (includes dependencies and capabilities):
{self._format_tool_metadata(toolbox_metadata)}

INSTRUCTIONS:
1. Understand the query goal and what has been done so far
2. Check tool metadata for dependencies (some tools require others to run first)
3. Select ONE tool that best progresses toward answering the query
4. Consider:
   - What information is still needed?
   - What tools can provide that information?
   - Are dependencies satisfied?
   - For multi-channel images (2+ channels), use Cell_State_Analyzer_Multi_Tool
   - For single-channel images, use Cell_State_Analyzer_Single_Tool
5. Formulate a clear sub-goal for the selected tool

Output format:
<justification>: Why this tool is the best next step
<context>: All necessary information from previous steps (file paths, data, variables)
<sub_goal>: Specific, achievable objective for this tool
<tool_name>: Exact tool name from available tools list"""
        
        # Generate next step
        next_step = self.llm_engine.generate(prompt, response_format=NextStep)
        
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
        for tool_name, metadata in list(toolbox_metadata.items())[:10]:  # Limit to avoid overflow
            desc = metadata.get('description', 'No description')[:200]
            lines.append(f"{tool_name}: {desc}")
        
        if len(toolbox_metadata) > 10:
            lines.append(f"... and {len(toolbox_metadata) - 10} more tools")
        
        return "\n".join(lines)
    
    def verificate_memory(self, question: str, image: str, query_analysis: str,
                         memory: Memory, bytes_mode: bool = False,
                         conversation_context: str = "", **kwargs) -> MemoryVerification:
        """Verify if memory contains sufficient information to answer query."""
        prompt = f"""Verify if the following steps contain sufficient information to answer the query.

Query: {question}
Query Analysis: {query_analysis}

Previous Steps:
{self._format_memory(memory)}

Determine if the query can be answered with the information available, or if more steps are needed."""
        
        return self.llm_engine.generate(prompt, response_format=MemoryVerification)
    
    def generate_final_output(self, question: str, image: str, memory: Memory,
                             bytes_mode: bool = False, conversation_context: str = "",
                             **kwargs) -> str:
        """Generate final answer from memory."""
        prompt = f"""Based on the following analysis steps, provide a comprehensive answer to the query.

Query: {question}

Analysis Steps:
{self._format_memory(memory, max_actions=20)}

Provide a clear, complete answer to the query based on the analysis results."""
        
        response = self.llm_engine.generate(prompt, max_tokens=2000)
        # Handle dict response (with 'content' key) or direct string
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
        # Handle dict response (with 'content' key) or direct string
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
            # Use unified parser if available
            if hasattr(ResponseParser, 'parse_memory_verification'):
                analysis, stop_signal = ResponseParser.parse_memory_verification(response)
                # Map to expected format: (context_verification, conclusion)
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

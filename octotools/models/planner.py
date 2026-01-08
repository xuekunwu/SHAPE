import os
import re
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List, Tuple, Optional

from octotools.engine.openai import ChatOpenAI
from octotools.models.memory import Memory
from octotools.models.formatters import QueryAnalysis, NextStep, MemoryVerification
from octotools.models.utils import normalize_tool_name
from octotools.models.tool_priority import ToolPriorityManager, ToolPriority, TOOL_DEPENDENCIES
from octotools.utils import logger, ResponseParser
from octotools.utils.image_processor import ImageProcessor
from octotools.models.image_data import ImageData

class Planner:
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None, available_tools: List = None, api_key: str = None):
        self.llm_engine_name = llm_engine_name
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []
        self.api_key = api_key
        
        # Initialize tool priority manager
        self.priority_manager = ToolPriorityManager()
        
        # Initialize LLM engines
        self.llm_engine = ChatOpenAI(model_string=llm_engine_name, is_multimodal=False, api_key=api_key)
        self.llm_engine_mm = ChatOpenAI(model_string=llm_engine_name, is_multimodal=True, api_key=api_key)
        
        # Initialize response storage
        self.base_response = None
        self.query_analysis = None
        self.detected_domain = 'general'  # Track detected task domain
        
        # Initialize token usage tracking
        self.last_usage = {}
    
    def _format_memory_for_prompt(self, memory: Memory, max_recent_actions: int = 10) -> str:
        """
        Format memory actions for prompt, using LLM-safe summaries and limiting to recent actions.
        This prevents context length overflow while preserving essential information.
        
        Args:
            memory: Memory object
            max_recent_actions: Maximum number of recent actions to include
            
        Returns:
            Formatted string of actions
        """
        actions = memory.get_actions(llm_safe=True)  # Use LLM-safe summaries
        if not actions:
            return "No previous steps"
        
        # Take only the most recent actions to prevent context overflow
        recent_actions = actions[-max_recent_actions:] if len(actions) > max_recent_actions else actions
        
        # Format actions compactly
        formatted = []
        for i, action in enumerate(recent_actions, 1):
            tool_name = action.get('tool_name', 'Unknown')
            sub_goal = action.get('sub_goal', '')
            result_summary = action.get('result', {})
            
            # Truncate long summaries
            if isinstance(result_summary, dict):
                result_str = str(result_summary)
            elif isinstance(result_summary, str):
                result_str = result_summary
            else:
                result_str = str(result_summary)
            
            # Limit result summary length
            if len(result_str) > 500:
                result_str = result_str[:500] + "... (truncated)"
            
            formatted.append(f"Step {i}: {tool_name}\n  Goal: {sub_goal[:200] if len(sub_goal) > 200 else sub_goal}\n  Result: {result_str}")
        
        if len(actions) > max_recent_actions:
            formatted.insert(0, f"[Showing last {max_recent_actions} of {len(actions)} steps. Earlier steps omitted to save context.]")
        
        return "\n\n".join(formatted)

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """
        Enhanced image info extraction using unified ImageProcessor.
        Extracts comprehensive metadata including channel information for better planning.
        """
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                # Use unified ImageProcessor for comprehensive metadata
                try:
                    img_data = ImageProcessor.load_image(image_path)
                    image_info.update({
                        "width": img_data.shape[1],
                        "height": img_data.shape[0],
                        "num_channels": img_data.num_channels,
                        "is_multi_channel": img_data.is_multi_channel,
                        "channel_names": img_data.channel_names,
                        "dtype": str(img_data.dtype),
                        "format": "multi-channel TIFF" if img_data.is_multi_channel else "single-channel"
                    })
                except Exception as img_proc_error:
                    # Fallback to basic PIL info if ImageProcessor fails
                    logger.debug(f"ImageProcessor failed, using PIL fallback: {img_proc_error}")
                    with Image.open(image_path) as img:
                        width, height = img.size
                    image_info.update({
                        "width": width,
                        "height": height,
                        "num_channels": 1,  # Default assumption
                        "is_multi_channel": False
                    })
            except Exception as e:
                logger.error(f"Error processing image file: {str(e)}")
        return image_info
    
    def get_image_info_bytes(self, bytes: str) -> Dict[str, Any]:
        image_info = {}
        if bytes:
            try:
                with Image.open(BytesIO(bytes)) as img:
                    width, height = img.size
                image_info.update({
                    "image_path": 'image.jpg', # generic image name
                    "width": width,
                    "height": height
                })
            except Exception as e:
                logger.error(f"Error processing image bytes: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: str = 4000, bytes_mode: bool = False) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]

        self.base_response = self.llm_engine_mm(input_data, max_tokens=max_tokens)

        return self.base_response


    def analyze_query(self, question: str, image: str, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> str:
        image_info = self.get_image_info(image)
        logger.debug(f"image_info: {image_info}")
        
        # Detect task domain using priority manager
        self.detected_domain = self.priority_manager.detect_task_domain(question, "")
        
        # Filter tools based on detected domain
        available_tools, excluded_tools = self.priority_manager.filter_tools_for_domain(
            self.available_tools, 
            self.detected_domain,
            exclude_excluded=True
        )
        
        # Filter metadata to match filtered tools
        toolbox_metadata = {
            tool: self.toolbox_metadata[tool] 
            for tool in available_tools 
            if tool in self.toolbox_metadata
        }
        
        if excluded_tools:
            logger.info(f"Task domain detected: {self.detected_domain}")
            logger.debug(f"Excluded tools for this domain: {excluded_tools}")
            logger.debug(f"Available tools after filtering: {available_tools}")

        query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the MINIMUM necessary skills and tools needed to address it directly and precisely.

Conversation so far:
{conversation_context}

Available tools: {available_tools}

Metadata for the tools: {toolbox_metadata}

Image: {image_info}

Query: {question}

CRITICAL INSTRUCTIONS - READ CAREFULLY:
1. **Understand the query precisely**: Read the query word-by-word and identify EXACTLY what is being asked. Do NOT add assumptions or extend the query beyond what is explicitly stated.

2. **Identify the core objective**: Determine the SINGLE main objective or task. If the query asks for:
   - "How many cells" → ONLY cell counting is needed (segmentation + counting)
   - "What cell states" or "analyze cell states" → Full cell state analysis is needed (preprocessing → segmentation → cropping → clustering → visualization)
   - "Compare images" → Comparison analysis is needed (may require cell state analysis if comparing at cell state level)
   - "Detect objects" → Object detection is needed (NOT cell state analysis)
   
3. **DO NOT over-extend**: 
   - If the query asks for cell count, DO NOT suggest cell state analysis tools
   - If the query asks for cell state analysis, DO NOT suggest additional validation or biological interpretation tools
   - Only include tools that are DIRECTLY necessary to answer the query
   - Do NOT add "nice-to-have" or "comprehensive" analysis steps that are not explicitly requested

4. **Select MINIMUM necessary tools**: 
   - For "how many cells": Image_Preprocessor_Tool (if needed) → Cell_Segmenter_Tool → STOP (count is in segmentation result)
   - For "what cell states": Full pipeline (preprocessing → segmentation → cropping → clustering → visualization)
   - For "compare cell states": Full pipeline + group comparison
   - Only add tools that are REQUIRED, not optional enhancements

5. **Match query type to tool chain**:
   - Simple counting/detection queries → Minimal tool chain (preprocessing → segmentation)
   - State analysis queries → Full analysis chain (preprocessing → segmentation → cropping → clustering → visualization)
   - Comparison queries → Full chain + comparison analysis

Your response should include:
1. A precise summary of what the query is asking for (EXACTLY, without additions).
2. A list of MINIMUM required skills (only what is necessary to answer the query).
3. A list of MINIMUM necessary tools (only tools directly required to answer the query, no optional enhancements).
4. A clear explanation of why each tool is necessary (must directly contribute to answering the query).

IMPORTANT: Do NOT suggest tools for analysis that is not explicitly requested in the query.
"""

        input_data = [query_prompt]

        llm_response = self.llm_engine_mm.generate(input_data, response_format=QueryAnalysis)
        
        # Extract content and usage from response
        if isinstance(llm_response, dict) and 'content' in llm_response:
            self.query_analysis = llm_response['content']
            # Store usage info for later access
            self.last_usage = llm_response.get('usage', {})
            logger.debug(f"Query analysis usage: {self.last_usage}")
        else:
            self.query_analysis = llm_response
            self.last_usage = {}

        # Check if we got a string response (non-structured model like gpt-4-turbo) instead of QueryAnalysis object
        if isinstance(self.query_analysis, str):
            logger.warning("Received string response instead of QueryAnalysis object")
            # Try to parse the string response to extract the analysis components
            try:
                # For string responses, we'll use the entire response as the analysis
                # This is simpler since QueryAnalysis is mainly used for display
                analysis_text = self.query_analysis.strip()
                logger.debug(f"Using string response as query analysis: {len(analysis_text)} characters")
            except Exception as parse_error:
                logger.error(f"Error parsing string response: {parse_error}")
                analysis_text = "Error parsing query analysis"
        else:
            analysis_text = str(self.query_analysis).strip()

        return analysis_text

    def extract_context_subgoal_and_tool(self, response) -> Tuple[str, str, str]:
        """Extract context, sub_goal, and tool_name from LLM response."""
        try:
            # Use unified response parser
            context, sub_goal, tool_name = ResponseParser.parse_next_step(response, self.available_tools)
            logger.debug(f"Extracted: context='{context[:50]}...', sub_goal='{sub_goal[:50]}...', tool_name='{tool_name}'")
            return context, sub_goal, tool_name
        except Exception as e:
            logger.error(f"Error in extract_context_subgoal_and_tool: {e}")
            return "", "", "Error extracting tool name"
    
    def _is_bioimage_task(self, question: str, query_analysis: str, memory: Memory) -> bool:
        """Detect if the task is related to bioimage/single-cell analysis."""
        # Use priority manager's domain detection
        domain = self.priority_manager.detect_task_domain(question, query_analysis)
        
        # Also check if bioimage tools have been used in memory
        if domain == 'bioimage':
            return True
        
        # Check if bioimage tools have been used
        bioimage_tools = [
            'Nuclei_Segmenter_Tool', 'Cell_Segmenter_Tool', 'Organoid_Segmenter_Tool',
            'Single_Cell_Cropper_Tool',
            'Cell_State_Analyzer_Tool',
            'Image_Preprocessor_Tool'
        ]
        actions = memory.get_actions(llm_safe=True)
        for action in actions:
            tool_name = action.get('tool_name', '')
            if any(bio_tool in tool_name for bio_tool in bioimage_tools):
                return True
        
        return False
    
    def _requires_full_cell_state_analysis(self, question: str) -> bool:
        """
        Determine if the query requires full cell state analysis pipeline.
        Returns True only if the query explicitly asks for cell state analysis, clustering, or comparison at cell state level.
        Returns False for simple counting queries.
        """
        question_lower = question.lower()
        
        # Keywords that indicate full cell state analysis is needed
        state_analysis_keywords = [
            'cell state', 'cell states', 'cell state analysis', 'analyze cell state',
            'cell clustering', 'cluster', 'clustering', 'cell type', 'cell types',
            'phenotype', 'phenotypic', 'morphological state', 'cell morphology analysis',
            'compare.*cell state', 'difference.*cell state', 'cell state.*difference',
            'umap', 'embedding', 'cell state level', 'at the cell state level'
        ]
        
        # Keywords that indicate simple counting (do NOT require full analysis)
        counting_keywords = [
            'how many', 'count', 'number of', 'quantity', 'total cells',
            'cell count', 'how many cells', 'number of cells'
        ]
        
        # Check for counting keywords first (higher priority)
        for keyword in counting_keywords:
            if keyword in question_lower:
                # If it's explicitly about cell states, still need full analysis
                if any(state_kw in question_lower for state_kw in ['cell state', 'cell states', 'state']):
                    return True
                # Otherwise, it's just counting
                return False
        
        # Check for state analysis keywords
        for keyword in state_analysis_keywords:
            if keyword in question_lower:
                return True
        
        # Default: if unclear, don't force full analysis (let LLM decide)
        return False
    
    def _try_rule_based_decision(self, question: str, image: str, memory: Memory, available_tools: List[str]) -> Optional[NextStep]:
        """
        Rule-based decision making for common scenarios to avoid LLM calls.
        Returns NextStep if a rule matches, None otherwise.
        
        This improves planning efficiency by handling simple, well-defined cases directly.
        """
        question_lower = question.lower()
        actions = memory.get_actions(llm_safe=True)
        used_tools = [action.get('tool_name', '') for action in actions if 'tool_name' in action]
        
        # Rule 1: Simple counting query with no steps taken
        # Pattern: "how many cells" + no tools used -> Image_Preprocessor_Tool or Segmenter
        if not used_tools:
            if any(kw in question_lower for kw in ['how many', 'count', 'number of']):
                # Check image type to select appropriate segmenter
                image_info = self.get_image_info(image) if image else {}
                is_multi_channel = image_info.get('is_multi_channel', False)
                
                # Select segmenter based on query keywords
                if 'organoid' in question_lower and 'Organoid_Segmenter_Tool' in available_tools:
                    return NextStep(
                        justification="Rule-based: Simple counting query for organoids. Starting with Organoid_Segmenter_Tool.",
                        context="No previous steps. Image ready for segmentation.",
                        sub_goal="Segment organoids in the image and count them.",
                        tool_name="Organoid_Segmenter_Tool"
                    )
                elif 'nuclei' in question_lower and 'Nuclei_Segmenter_Tool' in available_tools:
                    return NextStep(
                        justification="Rule-based: Simple counting query for nuclei. Starting with Nuclei_Segmenter_Tool.",
                        context="No previous steps. Image ready for segmentation.",
                        sub_goal="Segment nuclei in the image and count them.",
                        tool_name="Nuclei_Segmenter_Tool"
                    )
                elif 'Cell_Segmenter_Tool' in available_tools:
                    return NextStep(
                        justification="Rule-based: Simple counting query for cells. Starting with Cell_Segmenter_Tool.",
                        context="No previous steps. Image ready for segmentation.",
                        sub_goal="Segment cells in the image and count them.",
                        tool_name="Cell_Segmenter_Tool"
                    )
        
        # Rule 2: Counting query after segmentation -> STOP (count is in result)
        # Pattern: "how many" + segmentation tool used -> No next step needed
        if used_tools:
            last_tool = used_tools[-1]
            segmentation_tools = ["Cell_Segmenter_Tool", "Nuclei_Segmenter_Tool", "Organoid_Segmenter_Tool"]
            if last_tool in segmentation_tools:
                if any(kw in question_lower for kw in ['how many', 'count', 'number of']):
                    # Check if it's about cell states (needs full analysis)
                    if not any(state_kw in question_lower for state_kw in ['cell state', 'cell states', 'state', 'cluster', 'umap']):
                        logger.info("Rule-based: Counting query completed after segmentation. Count available in result.")
                        # Return None to signal completion (handled by caller)
                        return None
        
        # Rule 3: Multi-channel image detection -> Pre-select multi-channel aware tools
        # This is informational, not a direct decision, but helps in planning
        image_info = self.get_image_info(image) if image else {}
        if image_info.get('is_multi_channel', False):
            num_channels = image_info.get('num_channels', 1)
            logger.debug(f"Rule-based: Detected multi-channel image ({num_channels} channels). Planning will prioritize multi-channel aware tools.")
        
        # No rule matched - return None to proceed with LLM-based planning
        return None
    
    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> NextStep:
        image_info = self.get_image_info(image) if not bytes_mode else self.get_image_info_bytes(image)
        
        # Detect domain if not already detected (fallback to detection)
        if not hasattr(self, 'detected_domain') or not self.detected_domain:
            self.detected_domain = self.priority_manager.detect_task_domain(question, query_analysis)
        
        # Filter tools based on detected domain using priority manager
        available_tools, excluded_tools = self.priority_manager.filter_tools_for_domain(
            self.available_tools,
            self.detected_domain,
            exclude_excluded=True
        )
        
        # Try rule-based decision first (efficiency optimization)
        rule_decision = self._try_rule_based_decision(question, image, memory, available_tools)
        if rule_decision is not None:
            # Rule matched - return rule-based decision (saves LLM call)
            logger.info(f"✅ Rule-based decision applied: {rule_decision.tool_name}")
            return rule_decision
        # If rule_decision is None but we want to signal completion, handle it here
        # (For now, continue with LLM-based planning)
        
        # Filter metadata to match filtered tools
        toolbox_metadata = {
            tool: self.toolbox_metadata[tool]
            for tool in available_tools
            if tool in self.toolbox_metadata
        }
        
        if excluded_tools:
            logger.debug(f"Step {step_count}: Domain={self.detected_domain}, Excluded tools: {excluded_tools}")
        
        # Get tools grouped by priority for prompt
        tools_by_priority = self.priority_manager.format_tools_by_priority(available_tools)
        
        # Get used tools from memory for dependency checking (use llm_safe to avoid loading full results)
        actions_for_tools = memory.get_actions(llm_safe=True)
        used_tools = [action.get('tool_name', '') for action in actions_for_tools if 'tool_name' in action]
        
        # CRITICAL: Force Single_Cell_Cropper_Tool if last tool was a segmentation tool
        # BUT ONLY if the query requires full cell state analysis (not just counting)
        if used_tools:
            last_tool = used_tools[-1]
            segmentation_tools = ["Cell_Segmenter_Tool", "Nuclei_Segmenter_Tool", "Organoid_Segmenter_Tool"]
            if last_tool in segmentation_tools:
                # Check if query requires full cell state analysis
                requires_full_analysis = self._requires_full_cell_state_analysis(question)
                
                if requires_full_analysis:
                    # Override LLM selection - MUST use Single_Cell_Cropper_Tool next
                    logger.info(f"⚠️ FORCING Single_Cell_Cropper_Tool: Last tool '{last_tool}' was a segmentation tool, and query requires full cell state analysis")
                    if "Single_Cell_Cropper_Tool" not in available_tools:
                        logger.error(f"Single_Cell_Cropper_Tool not available! Available: {available_tools}")
                    else:
                        # Create a forced next step
                        forced_context = self._format_memory_for_prompt(memory)
                        return NextStep(
                            justification=f"MANDATORY: Previous tool '{last_tool}' was a segmentation tool, and the query requires full cell state analysis. Single_Cell_Cropper_Tool MUST be called next according to the bioimage analysis chain.",
                            context=forced_context,
                            sub_goal="Generate single-cell crops from the segmentation mask produced in the previous step. Extract individual cell regions with appropriate margins for downstream analysis.",
                            tool_name="Single_Cell_Cropper_Tool"
                        )
                else:
                    # Query is just counting - segmentation result contains the count, no need to force cropping
                    logger.info(f"ℹ️ Query appears to be counting-only. Segmentation tool '{last_tool}' completed. " +
                              f"Cell count should be available in segmentation result. Not forcing Single_Cell_Cropper_Tool.")
        recommended_tools = self.priority_manager.get_recommended_next_tools(
            available_tools, used_tools, self.detected_domain
        )
        
        # Format recommended tools for prompt with special emphasis
        if recommended_tools:
            first_tool = recommended_tools[0]
            # If first tool is Single_Cell_Cropper_Tool and last used was a segmentation tool, emphasize it
            last_tool = used_tools[-1] if used_tools else None
            segmentation_tools = ["Cell_Segmenter_Tool", "Nuclei_Segmenter_Tool", "Organoid_Segmenter_Tool"]
            if first_tool == "Single_Cell_Cropper_Tool" and last_tool in segmentation_tools:
                recommended_tools_str = f"⚡ {first_tool} (MANDATORY - must be next)" + \
                                      (f", {', '.join(recommended_tools[1:5])}" if len(recommended_tools) > 1 else "")
            else:
                recommended_tools_str = ", ".join(recommended_tools[:5])
            if len(recommended_tools) > 5:
                recommended_tools_str += f" (and {len(recommended_tools) - 5} more)"
        else:
            recommended_tools_str = "None"
        
        prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

CRITICAL: Understand the query type and select ONLY the minimum necessary tools:
- If query asks "how many cells" → Only need: Image_Preprocessor_Tool → Cell_Segmenter_Tool → STOP (count is in result)
- If query asks "what cell states" or "analyze cell states" → Need full pipeline: preprocessing → segmentation → cropping → clustering → visualization
- If query asks "compare" at cell state level → Need full pipeline + comparison
- DO NOT add tools beyond what is explicitly needed to answer the query

Context:
Query: {question}
Image: {image if not bytes_mode else 'image.jpg'}
Query Analysis: {query_analysis}
Detected Task Domain: {self.detected_domain}

Available Tools (organized by priority):
{tools_by_priority}

Recommended Next Tools (considering dependencies and priorities):
{recommended_tools_str}

⚠️ CRITICAL CHECK: Look at "Previous Steps and Their Results" above. 

   IMPORTANT: Only enforce tool chain if the query requires full cell state analysis:
   - If query asks "how many cells" → After segmentation, STOP (count is in result). Do NOT force Single_Cell_Cropper_Tool.
   - If query asks "what cell states" or "analyze cell states" → Follow full pipeline below.
   
   If the query requires full cell state analysis AND the LAST tool executed was:
   - Cell_Segmenter_Tool, OR
   - Nuclei_Segmenter_Tool, OR  
   - Organoid_Segmenter_Tool
   
   Then you MUST select Single_Cell_Cropper_Tool as the next tool. This is MANDATORY.
   Do NOT select any other tool. Do NOT skip this step.
   
   If the LAST tool executed was:
   - Cell_State_Analyzer_Tool
   
   Then you MUST select Analysis_Visualizer_Tool as the next tool. This is MANDATORY.
   Do NOT select any other tool. Do NOT skip this step.
   
   If the LAST tool executed was:
   - Analysis_Visualizer_Tool
   
   Then you SHOULD consider using Image_Captioner_Tool to generate a final summary and interpretation of the visualizations. 
   This is RECOMMENDED (not mandatory) to provide a comprehensive final answer to the user's query.

Tool Metadata:
{toolbox_metadata}

Previous Steps and Their Results:
{self._format_memory_for_prompt(memory)}

Tools Already Used: {used_tools if used_tools else "None"}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. **First, identify the query type precisely**:
   - Is this a simple counting query ("how many cells")? → Use minimal tools (preprocessing → segmentation → STOP)
   - Is this a cell state analysis query ("what cell states", "analyze cell states")? → Use full pipeline
   - Is this a comparison query? → Determine if comparison is at cell state level or just counts
   - DO NOT assume the query needs full analysis if it only asks for counts

2. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

3. Determine the most appropriate next step by considering:
   - **Query type and minimum requirements**: What is the MINIMUM needed to answer the query?
   - Key objectives from the query analysis (ONLY what is explicitly stated)
   - Capabilities of available tools (see priority grouping above)
   - Logical progression of problem-solving (but only for what is needed)
   - Outcomes from previous steps
   - Tool dependencies (some tools require other tools to run first)
   - Current step count and remaining steps
   
4. **CRITICAL: Do NOT over-extend**:
   - If the query asks for cell count and segmentation is done → STOP, do NOT proceed to cell state analysis
   - If the query asks for cell state analysis and visualization is done → STOP, do NOT add unnecessary validation steps
   - Only proceed to next tool if it is REQUIRED to answer the query

3. Tool Selection Priority (IMPORTANT - FOLLOW THIS ORDER):
   Tools are organized by priority level. You MUST follow this priority order:
   
   - HIGH Priority: Use these tools FIRST if they are relevant to the query
     * Core tools for bioimage analysis and specialized analysis tools
     * Examples: Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, Organoid_Segmenter_Tool,
                Single_Cell_Cropper_Tool, Cell_State_Analyzer_Tool, Analysis_Visualizer_Tool
   
   - MEDIUM Priority: General-purpose tools (rarely used for bioimage tasks)
   
   - LOW Priority: Use sparingly, only when necessary
     * Utility tools and code generation tools (use only when no other tool can solve the query)
     * Examples: Object_Detector_Tool, Advanced_Object_Detector_Tool, Image_Captioner_Tool, Text_Detector_Tool, Python_Code_Generator_Tool
   
   IMPORTANT: Always prefer tools from higher priority levels (HIGH > MEDIUM > LOW).
   Do NOT use LOW priority code generation tools if any higher-priority tool can address the query.

4. CRITICAL: Bioimage Analysis Chain Priority (MUST FOLLOW THIS ORDER ONLY IF QUERY REQUIRES IT):
   IMPORTANT: Only follow the full chain if the query explicitly asks for cell state analysis, clustering, or comparison at cell state level.
   For simple counting queries ("how many cells"), only use: Image_Preprocessor_Tool → Segmenter → STOP (count is in segmentation result).
   
   For cell state analysis queries, you MUST follow this specific tool chain order:
   
   Step 1: Image_Preprocessor_Tool (if image quality needs improvement)
   Step 2: Choose ONE segmentation tool based on image type:
           - Cell_Segmenter_Tool (for phase-contrast cell images)
           - Nuclei_Segmenter_Tool (for nuclei/fluorescence images)
           - Organoid_Segmenter_Tool (for organoid images)
   Step 3: Single_Cell_Cropper_Tool (REQUIRED - must be called IMMEDIATELY after Step 2, ONLY if query requires cell state analysis)
          ⚠️ MANDATORY: If a segmentation tool was just executed AND the query requires cell state analysis, 
          you MUST select Single_Cell_Cropper_Tool as the next tool. 
          Do NOT skip this step. Do NOT select any other tool.
          ⚠️ NOT REQUIRED: If the query is just counting cells, STOP after segmentation (count is in result).
   Step 4: Cell_State_Analyzer_Tool (requires single-cell crops from Step 3)
   
   This chain MUST be followed in order ONLY for cell state analysis queries: 
   Image_Preprocessor → Segmenter → Single_Cell_Cropper → Cell_State_Analyzer
   Do NOT skip steps or use tools out of order.
   
   CRITICAL RULE: When a segmentation tool completes:
   - If query requires cell state analysis → Next tool MUST be Single_Cell_Cropper_Tool
   - If query is just counting → STOP (count is available in segmentation result)

5. Check Tool Dependencies:
   Some tools require other tools to run first:
   - Single_Cell_Cropper_Tool requires Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, or Organoid_Segmenter_Tool
   - Cell_State_Analyzer_Tool requires Single_Cell_Cropper_Tool
   - Analysis_Visualizer_Tool requires Cell_State_Analyzer_Tool (for cell state visualization)
   
   Ensure all dependencies are satisfied before selecting a tool.

5. Select ONE tool best suited for the next step, keeping in mind:
   - The priority order above
   - Tool dependencies
   - The limited number of remaining steps
   - Recommended tools listed above

6. Formulate a specific, achievable sub-goal for the selected tool that maximizes progress towards answering the query.

Output Format:
<justification>: detailed explanation of why the selected tool is the best choice for the next step, considering the context and previous outcomes.
<context>: MUST include ALL necessary information for the tool to function, structured as follows:
    * Relevant data from previous steps
    * File names or paths created or used in previous steps (list EACH ONE individually)
    * Variable names and their values from previous steps' results
    * Any other context-specific information required by the tool
<sub_goal>: a specific, achievable objective for the tool, based on its metadata and previous outcomes. It MUST contain any involved data, file names, and variables from Previous Steps and Their Results that the tool can act upon.
<tool_name>: MUST be the exact name of a tool from the available tools list.

Rules:
- Select only ONE tool for this step.
- The sub-goal MUST directly address the query and be achievable by the selected tool.
- The Context section MUST include ALL necessary information for the tool to function, including ALL relevant file paths, data, and variables from previous steps.
- The tool name MUST exactly match one from the available tools list: {available_tools}.
- Avoid redundancy by considering previous steps and building on prior results.

Example (do not copy, use only as reference):
<justification>: [Your detailed explanation here]
<context>: Image path: "example/image.jpg", Previous detection results: [list of objects]
<sub_goal>: Detect and count the number of specific objects in the image "example/image.jpg"
<tool_name>: Object_Detector_Tool
        """
        llm_response = self.llm_engine.generate(prompt_generate_next_step, response_format=NextStep)
        
        # Extract content and usage from response
        if isinstance(llm_response, dict) and 'content' in llm_response:
            next_step = llm_response['content']
            # Store usage info for later access
            self.last_usage = llm_response.get('usage', {})
            logger.debug(f"Next step usage: {self.last_usage}")
        else:
            next_step = llm_response
            self.last_usage = {}
        
        # CRITICAL: Code-level enforcement - verify LLM selection matches forced rules
        # (This is a safety check in case the pre-LLM check was somehow bypassed)
        # BUT ONLY if the query requires full cell state analysis
        if used_tools:
            last_tool = used_tools[-1]
            segmentation_tools = ["Cell_Segmenter_Tool", "Nuclei_Segmenter_Tool", "Organoid_Segmenter_Tool"]
            if last_tool in segmentation_tools:
                # Check if query requires full cell state analysis
                requires_full_analysis = self._requires_full_cell_state_analysis(question)
                
                if requires_full_analysis:
                    # Extract tool name from next_step
                    selected_tool = getattr(next_step, 'tool_name', '') if hasattr(next_step, 'tool_name') else ''
                    # Parse from string if needed
                    if not selected_tool and isinstance(next_step, str):
                        from octotools.utils.response_parser import ResponseParser
                        _, _, selected_tool = ResponseParser.parse_next_step(next_step, available_tools)
                    
                    # Enforce: If LLM didn't select the correct tool, override it
                    if selected_tool != 'Single_Cell_Cropper_Tool':
                        logger.warning(f"⚠️ CODE ENFORCEMENT: LLM selected '{selected_tool}' after segmentation tool '{last_tool}', " 
                                     f"overriding to Single_Cell_Cropper_Tool (query requires full cell state analysis)")
                        # Override: Create forced next step
                        forced_context = self._format_memory_for_prompt(memory)
                        forced_next_step = NextStep(
                            justification=f"MANDATORY ENFORCEMENT: Previous tool '{last_tool}' was a segmentation tool, and the query requires full cell state analysis. "
                                        f"Single_Cell_Cropper_Tool MUST be called next according to the bioimage analysis chain. "
                                        f"LLM selected '{selected_tool}' which was overridden by code-level enforcement.",
                            context=forced_context,
                            sub_goal="Generate single-cell crops from the segmentation mask produced in the previous step. "
                                   "Extract individual cell regions with appropriate margins for downstream analysis.",
                            tool_name="Single_Cell_Cropper_Tool"
                        )
                        return forced_next_step
                # If query is just counting, don't enforce - let LLM decide or stop
            
            # CRITICAL: Code-level enforcement for Analysis_Visualizer_Tool after Cell_State_Analyzer_Tool
            if last_tool == "Cell_State_Analyzer_Tool":
                # Extract tool name from next_step
                selected_tool = getattr(next_step, 'tool_name', '') if hasattr(next_step, 'tool_name') else ''
                # Parse from string if needed
                if not selected_tool and isinstance(next_step, str):
                    from octotools.utils.response_parser import ResponseParser
                    _, _, selected_tool = ResponseParser.parse_next_step(next_step, available_tools)
                
                # Enforce: If LLM didn't select Analysis_Visualizer_Tool, override it
                if selected_tool != 'Analysis_Visualizer_Tool' and 'Analysis_Visualizer_Tool' in available_tools:
                    logger.warning(f"⚠️ CODE ENFORCEMENT: LLM selected '{selected_tool}' after Cell_State_Analyzer_Tool, "
                                 f"overriding to Analysis_Visualizer_Tool")
                    # Override: Create forced next step
                    forced_context = self._format_memory_for_prompt(memory)
                    forced_next_step = NextStep(
                        justification=f"MANDATORY ENFORCEMENT: Previous tool 'Cell_State_Analyzer_Tool' completed cell state analysis. "
                                    f"Analysis_Visualizer_Tool MUST be called next to visualize the analysis results (UMAP, clusters, exemplars). "
                                    f"LLM selected '{selected_tool}' which was overridden by code-level enforcement.",
                        context=forced_context,
                        sub_goal="Visualize cell state analysis results from Cell_State_Analyzer_Tool. "
                               "Generate publication-quality UMAP plots, cluster composition charts, and exemplar cell montages.",
                        tool_name="Analysis_Visualizer_Tool"
                    )
                    return forced_next_step
            
            # RECOMMENDATION: Suggest Image_Captioner_Tool after Analysis_Visualizer_Tool (if not already used)
            if last_tool == "Analysis_Visualizer_Tool":
                # Extract tool name from next_step
                selected_tool = getattr(next_step, 'tool_name', '') if hasattr(next_step, 'tool_name') else ''
                # Parse from string if needed
                if not selected_tool and isinstance(next_step, str):
                    from octotools.utils.response_parser import ResponseParser
                    _, _, selected_tool = ResponseParser.parse_next_step(next_step, available_tools)
                
                # Check if Image_Captioner_Tool has been used
                used_tools = [action.get('tool_name') for action in memory.get_actions(llm_safe=True) if 'tool_name' in action]
                
                # If Image_Captioner_Tool hasn't been used and LLM didn't select it, suggest it
                if 'Image_Captioner_Tool' not in used_tools and selected_tool != 'Image_Captioner_Tool' and 'Image_Captioner_Tool' in available_tools:
                    logger.info(f"💡 RECOMMENDATION: Analysis_Visualizer_Tool completed. Suggesting Image_Captioner_Tool for final summary. "
                              f"LLM selected '{selected_tool}', but Image_Captioner_Tool is recommended.")
                    # Override: Create recommended next step
                    forced_context = self._format_memory_for_prompt(memory)
                    forced_next_step = NextStep(
                        justification=f"RECOMMENDED: Previous tool 'Analysis_Visualizer_Tool' completed visualization generation. "
                                    f"Image_Captioner_Tool is recommended to generate a final summary and interpretation of the visualizations "
                                    f"to provide a comprehensive answer to the user's query. "
                                    f"LLM selected '{selected_tool}' which was overridden by recommendation.",
                        context=forced_context,
                        sub_goal="Generate a final summary and interpretation of the analysis visualizations using Image_Captioner_Tool. "
                               "Provide a comprehensive answer to the user's query based on the generated visualizations.",
                        tool_name="Image_Captioner_Tool"
                    )
                    return forced_next_step
        
        # Validate the selected tool
        if hasattr(next_step, 'tool_name') and next_step.tool_name:
            validation_result = self._validate_tool_selection(
                next_step.tool_name, available_tools, used_tools, self.detected_domain
            )
            if not validation_result['valid']:
                logger.warning(f"Tool selection validation failed: {validation_result['reason']}")
                # The tool is still returned, but warning is logged
                # In production, you might want to retry or use a fallback
        
        return next_step
    
    def _validate_tool_selection(
        self, 
        tool_name: str, 
        available_tools: List[str],
        used_tools: List[str],
        domain: str
    ) -> Dict[str, Any]:
        """
        Validate that the selected tool is appropriate for the task.
        
        Returns:
            dict with 'valid' (bool) and 'reason' (str) keys
        """
        # Normalize tool name
        normalized_tool = self.priority_manager._normalize_tool_name(tool_name)
        
        # Check if tool is in available tools
        if normalized_tool not in available_tools:
            # Try to find similar tool names
            similar_tools = [t for t in available_tools if normalized_tool.lower() in t.lower() or t.lower() in normalized_tool.lower()]
            if similar_tools:
                return {
                    'valid': False,
                    'reason': f"Tool '{tool_name}' not in available tools. Did you mean: {similar_tools[0]}?",
                    'suggestion': similar_tools[0]
                }
            return {
                'valid': False,
                'reason': f"Tool '{tool_name}' not in available tools list: {available_tools}",
                'suggestion': None
            }
        
        # Check tool priority for domain
        priority = self.priority_manager.get_priority(normalized_tool)
        
        if priority == ToolPriority.EXCLUDED:
            return {
                'valid': False,
                'reason': f"Tool '{tool_name}' is EXCLUDED for {domain} domain tasks",
                'suggestion': None
            }
        
        # Check dependencies
        dependencies = TOOL_DEPENDENCIES.get(normalized_tool, [])
        missing_deps = [dep for dep in dependencies if dep not in used_tools]
        
        if missing_deps:
            return {
                'valid': False,
                'reason': f"Tool '{tool_name}' requires dependencies that haven't been used: {missing_deps}",
                'suggestion': f"Use {missing_deps[0]} first" if missing_deps else None
            }
        
        # Warn if LOW priority code generation tool is selected and there are higher priority alternatives
        if priority == ToolPriority.LOW:
            # Check if it's a code generation tool
            code_gen_tools = ['Python_Code_Generator_Tool']  # Generalist_Solution_Generator_Tool is excluded
            if normalized_tool in code_gen_tools:
                higher_priority_tools = [
                    t for t in available_tools 
                    if t not in used_tools and self.priority_manager.get_priority(t) < ToolPriority.LOW
                ]
                if higher_priority_tools:
                    return {
                        'valid': True,  # Still valid, but warn
                        'reason': f"LOW priority code generation tool selected, but higher priority tools available: {higher_priority_tools[:3]}",
                        'suggestion': higher_priority_tools[0] if higher_priority_tools else None
                    }
        
        return {'valid': True, 'reason': 'Tool selection is valid', 'suggestion': None}

    def verificate_memory(self, question: str, image: str, query_analysis: str, memory: Memory, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> MemoryVerification:
        if bytes_mode:
            image_info = self.get_image_info_bytes(image)
        else:
            image_info = self.get_image_info(image)

        # Special handling for fibroblast analysis pipeline
        # Check if this is a fibroblast analysis that needs activation scoring
        fibroblast_keywords = ["fibroblast", "activation", "cell state", "cell analysis", "quantify", "score"]
        is_fibroblast_query = any(keyword.lower() in question.lower() for keyword in fibroblast_keywords)
        
        # Get finished tools from memory
        finished_tools = [action['tool_name'] for action in memory.get_actions(llm_safe=True) if 'tool_name' in action]
        
        # Special case: If this is a fibroblast analysis and cell state analyzer just finished,
        # but activation scorer hasn't run yet, we should continue

        prompt_memory_verification = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

CRITICAL: Understand the query type and verify ONLY what is explicitly asked:
- If query asks "how many cells" → Verify that cell count is available (from segmentation result) → STOP if count exists
- If query asks "what cell states" → Verify that cell state analysis (clustering + visualization) is complete → STOP if complete
- If query asks "compare" → Verify that comparison analysis is complete → STOP if complete
- DO NOT require information beyond what is explicitly asked in the query

Conversation so far:
{conversation_context}

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.available_tools}
Toolbox Metadata: {self.toolbox_metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {self._format_memory_for_prompt(memory)}

Detailed Instructions:
1. **First, identify the query type precisely**:
   - Simple counting query ("how many cells")? → Only need cell count from segmentation
   - Cell state analysis query ("what cell states", "analyze cell states")? → Need full analysis pipeline
   - Comparison query? → Need comparison results
   - DO NOT assume the query needs full analysis if it only asks for counts

2. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the EXACT objectives of the query (ONLY what is explicitly stated).
   - Note any specific requirements or constraints mentioned (do NOT add assumptions).
   - If an image is provided, consider its relevance and what information it contributes (only for what is asked).

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address the SPECIFIC aspects asked in the query?
      - Identify ONLY the parts of the query that remain unanswered (do NOT add requirements beyond the query).
      - For "how many cells" queries: If cell count is available (from segmentation), the query is COMPLETE. Do NOT require cell state analysis.
      - For "what cell states" queries: If cell state analysis (clustering + visualization) is complete, the query is COMPLETE. Do NOT require additional validation.
      - Consider if all relevant information has been extracted from the image (ONLY for what is explicitly asked).
      - IMPORTANT: Match the query type to required results:
        * Counting queries → Only need counts (from segmentation or detection)
        * State analysis queries → Need full analysis pipeline results
        * Comparison queries → Need comparison results
      - CRITICAL: If the query asks for analysis and you see analysis results with visualizations, distributions, and statistics, the task is COMPLETE.
      - For cell state analysis: If preprocessing → segmentation → cropping → clustering → visualization is complete, STOP.
      - Technical analysis results (clusters, UMAP, exemplars) are SUFFICIENT - do NOT require biological label mapping or validation.
      - DO NOT require information that is not explicitly asked in the query.

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.
      - Pay special attention to analysis tools that could provide insights from prepared data.
      - IMPORTANT: If the main analysis has been completed and only unused tools remain for minor enhancements, this does NOT justify continuing.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.
      - IMPORTANT: Do NOT require verification or QC steps if technical analysis is complete. Only flag critical errors or failures.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.
      - IMPORTANT: Do NOT flag technical analysis results as "ambiguous" just because they lack biological labels.
      - Do NOT require additional tools for "validation" if the main analysis pipeline is complete.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.
   
   CRITICAL CHECKLIST FOR STOPPING (PRIORITY: Stop early if query is satisfied):
   - Has the MAIN query been answered EXACTLY as asked? If yes, STOP immediately.
   - For "how many cells" queries: If cell count is available (from segmentation), STOP. Do NOT require cell state analysis.
   - For "what cell states" queries: If cell state analysis (clustering + visualization) is complete, STOP. Do NOT require additional steps.
   - Are there analysis results, visualizations, counts, or statistics that DIRECTLY answer the query? If yes, STOP.
   - If the query asked for "compare", "count", "analyze", "detect" and you have those results, STOP.
   - If the query asked for specific outputs (charts, counts, comparisons) and they exist, STOP.
   - DO NOT continue just because there are unused tools available.
   - DO NOT use Python_Code_Generator_Tool unless NO other tools can solve the query.
   - STOP if the query is satisfied, even if some tools haven't been used.
   - DO NOT require information beyond what is explicitly asked in the query.
   
   IMPORTANT: For cell state analysis queries ("what cell states", "analyze cell states", etc.):
   - If Analysis_Visualizer_Tool has been executed and produced visualizations (UMAP, clusters, exemplars), the task is COMPLETE. STOP.
   - If you see the full pipeline: preprocessing → segmentation → cropping → clustering → visualization, STOP immediately.
   - Technical analysis results (UMAP plots, cluster assignments, exemplar images, statistics) are SUFFICIENT.
   - DO NOT require biological label mapping, marker validation, manual QC, or additional verification steps.
   - DO NOT suggest "missing biological interpretation" - technical cluster analysis IS the answer to "what cell states" queries.
   - The presence of cluster analysis with visualizations means the task is DONE. Stop immediately.

   CRITICAL CHECKLIST FOR CONTINUING (Only continue if query is NOT satisfied):
   - Is the MAIN query still UNANSWERED?
   - Are there UNANSWERED parts that require specific tools (not generalist/code generators)?
   - Is the current state only data preparation without the requested analysis?
   - Only continue if the query specifically requires additional tools AND those tools are NOT Python_Code_Generator_Tool (use only as last resort).

Response Format:
You MUST respond with exactly two fields:
1. analysis: A detailed analysis of why the memory is sufficient or insufficient. Reference specific information from the memory and explain its relevance to each aspect of the task. Address how each main point of the query has been satisfied or what is still missing.
2. stop_signal: A boolean value (True or False) indicating whether to stop the problem solving process and proceed to generating the final output.
    * True: if the memory is sufficient for addressing the query to proceed and no additional available tools need to be used. If ONLY manual verification without tools is needed, choose True.
    * False: if the memory is insufficient and needs more information from additional tool usage.

IMPORTANT: The response must be structured exactly as specified above with both 'analysis' and 'stop_signal' fields present.

For text-based responses, format your answer as:
analysis: [Your detailed analysis here]
stop_signal: [True or False]
"""

        input_data = [prompt_memory_verification]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                logger.error(f"Error reading image file: {str(e)}")

        try:
            llm_response = self.llm_engine_mm.generate(input_data, response_format=MemoryVerification)
            
            # Extract content and usage from response
            if isinstance(llm_response, dict) and 'content' in llm_response:
                stop_verification = llm_response['content']
                # Store usage info for later access
                self.last_usage = llm_response.get('usage', {})
                logger.debug(f"Memory verification usage: {self.last_usage}")
            else:
                stop_verification = llm_response
                self.last_usage = {}
            
            # Check if we got a string response (non-structured model) instead of MemoryVerification object
            if isinstance(stop_verification, str):
                logger.warning("Received string response instead of MemoryVerification object")
                # Use unified parser
                analysis, stop_signal = ResponseParser.parse_memory_verification(stop_verification)
                stop_verification = MemoryVerification(
                    analysis=analysis,
                    stop_signal=stop_signal
                )
                logger.debug(f"Created MemoryVerification object: analysis length={len(analysis)}, stop_signal={stop_signal}")
                
        except Exception as e:
            logger.error(f"Error in response format parsing: {e}")
            # Fallback: try without response format
            try:
                raw_response = self.llm_engine_mm.generate(input_data)
                
                # Extract content and usage from fallback response
                if isinstance(raw_response, dict) and 'content' in raw_response:
                    raw_content = raw_response['content']
                    self.last_usage = raw_response.get('usage', {})
                else:
                    raw_content = raw_response
                    self.last_usage = {}
                    
                logger.debug(f"Raw response length: {len(str(raw_content))}")
                # Create a basic MemoryVerification object with default values
                stop_verification = MemoryVerification(
                    analysis=raw_content,
                    stop_signal=False  # Default to continue
                )
            except Exception as fallback_error:
                logger.error(f"Fallback error: {fallback_error}")
                # Create a minimal MemoryVerification object
                stop_verification = MemoryVerification(
                    analysis="Error in memory verification",
                    stop_signal=False
                )

        return stop_verification

    def extract_conclusion(self, response: MemoryVerification) -> str:
        try:
            # Use unified parser
            analysis, stop_signal = ResponseParser.parse_memory_verification(response)
            logger.debug(f"Extract conclusion - Analysis length: {len(analysis)}, Stop signal: {stop_signal}")
            
            if stop_signal:
                return analysis, 'STOP'
            else:
                return analysis, 'CONTINUE'
        except Exception as e:
            logger.error(f"Error accessing MemoryVerification attributes: {e}")
            # Fallback: try to extract from string representation or default to continue
            try:
                if hasattr(response, 'analysis'):
                    analysis = response.analysis
                else:
                    analysis = str(response)
                
                # Default to continue if we can't determine stop_signal
                return analysis, 'CONTINUE'
            except Exception as fallback_error:
                logger.error(f"Fallback error: {fallback_error}")
                return "Error processing verification response", 'CONTINUE'

    def generate_final_output(self, question: str, image: str, memory: Memory, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> str:
        if bytes_mode:
            image_info = self.get_image_info_bytes(image)
        else:
            image_info = self.get_image_info(image)

        prompt_generate_final_output = f"""
Task: Generate the final output based on the query, image, and tools used in the process.

Conversation so far:
{conversation_context}

Context:
Query: {question}
Image: {image_info}
Actions Taken:
{self._format_memory_for_prompt(memory)}

Instructions:
1. Review the query, image, and all actions taken during the process.
2. Consider the results obtained from each tool execution.
3. Incorporate the relevant information from the memory to generate the step-by-step final output.
4. The final output should be consistent and coherent using the results from the tools.

Output Structure:
Your response should be well-organized and include the following sections:

1. Summary:
   - Provide a brief overview of the query and the main findings.

2. Detailed Analysis:
   - Break down the process of answering the query step-by-step.
   - For each step, mention the tool used, its purpose, and the key results obtained.
   - Explain how each step contributed to addressing the query.

3. Key Findings:
   - List the most important discoveries or insights gained from the analysis.
   - Highlight any unexpected or particularly interesting results.

4. Answer to the Query:
   - Directly address the original question with a clear and concise answer.
   - If the query has multiple parts, ensure each part is answered separately.

5. Additional Insights (if applicable):
   - Provide any relevant information or insights that go beyond the direct answer to the query.
   - Discuss any limitations or areas of uncertainty in the analysis.

6. Conclusion:
   - Summarize the main points and reinforce the answer to the query.
   - If appropriate, suggest potential next steps or areas for further investigation.
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                logger.error(f"Error reading image file: {str(e)}")

        llm_response = self.llm_engine_mm.generate(input_data)
        
        # Extract content and usage from response
        if isinstance(llm_response, dict) and 'content' in llm_response:
            final_output = llm_response['content']
            # Store usage info for later access
            self.last_usage = llm_response.get('usage', {})
            logger.debug(f"Final output usage: {self.last_usage}")
        else:
            final_output = llm_response
            self.last_usage = {}

        return final_output


    def generate_direct_output(self, question: str, image: str, memory: Memory, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> str:
        if bytes_mode:
            image_info = self.get_image_info_bytes(image)
        else:
            image_info = self.get_image_info(image)

        prompt_generate_final_output = f"""
Conversation so far:
{conversation_context}

Context:
Query: {question}
Image: {image_info}
Initial Analysis:
{self.query_analysis}
Actions Taken:
{self._format_memory_for_prompt(memory)}

Please generate the concise output based on the query, image information, initial analysis, and actions taken. Break down the process into clear, logical, and conherent steps. Conclude with a precise and direct answer to the query.

Answer:
"""

        input_data = [prompt_generate_final_output]
        if image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                logger.error(f"Error reading image file: {str(e)}")

        llm_response = self.llm_engine_mm.generate(input_data)
        
        # Extract content and usage from response
        if isinstance(llm_response, dict) and 'content' in llm_response:
            final_output = llm_response['content']
            # Store usage info for later access
            self.last_usage = llm_response.get('usage', {})
            logger.debug(f"Direct output usage: {self.last_usage}")
        else:
            final_output = llm_response
            self.last_usage = {}

        return final_output
    
    def run_activation_scorer(self, input_adata_path, reference_path):
        scorer = ActivationScorerTool()
        output_adata_path = scorer.run(input_adata_path, reference_path)
        return output_adata_path
    

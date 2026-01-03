import os
import re
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List, Tuple

from octotools.engine.openai import ChatOpenAI
from octotools.models.memory import Memory
from octotools.models.formatters import QueryAnalysis, NextStep, MemoryVerification
from octotools.models.utils import normalize_tool_name
from octotools.models.tool_priority import ToolPriorityManager, ToolPriority, TOOL_DEPENDENCIES

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

    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        image_info = {}
        if image_path and os.path.isfile(image_path):
            image_info["image_path"] = image_path
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                image_info.update({
                    "width": width,
                    "height": height
                })
            except Exception as e:
                print(f"Error processing image file: {str(e)}")
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
                print(f"Error processing image bytes: {str(e)}")
        return image_info

    def generate_base_response(self, question: str, image: str, max_tokens: str = 4000, bytes_mode: bool = False) -> str:
        image_info = self.get_image_info(image)

        input_data = [question]

        self.base_response = self.llm_engine_mm(input_data, max_tokens=max_tokens)

        return self.base_response


    def analyze_query(self, question: str, image: str, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> str:
        image_info = self.get_image_info(image)
        print("image_info: ", image_info)
        
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
            print(f"Task domain detected: {self.detected_domain}")
            print(f"Excluded tools for this domain: {excluded_tools}")
            print(f"Available tools after filtering: {available_tools}")

        query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Conversation so far:
{conversation_context}

Available tools: {available_tools}

Metadata for the tools: {toolbox_metadata}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Examine the available tools in the toolbox and determine which ones are relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
"""

        input_data = [query_prompt]

        llm_response = self.llm_engine_mm.generate(input_data, response_format=QueryAnalysis)
        
        # Extract content and usage from response
        if isinstance(llm_response, dict) and 'content' in llm_response:
            self.query_analysis = llm_response['content']
            # Store usage info for later access
            self.last_usage = llm_response.get('usage', {})
            print(f"Query analysis usage: {self.last_usage}")
        else:
            self.query_analysis = llm_response
            self.last_usage = {}

        # Check if we got a string response (non-structured model like gpt-4-turbo) instead of QueryAnalysis object
        if isinstance(self.query_analysis, str):
            print("WARNING: Received string response instead of QueryAnalysis object")
            # Try to parse the string response to extract the analysis components
            try:
                # For string responses, we'll use the entire response as the analysis
                # This is simpler since QueryAnalysis is mainly used for display
                analysis_text = self.query_analysis.strip()
                print(f"Using string response as query analysis: {len(analysis_text)} characters")
            except Exception as parse_error:
                print(f"Error parsing string response: {parse_error}")
                analysis_text = "Error parsing query analysis"
        else:
            analysis_text = str(self.query_analysis).strip()

        return analysis_text

    def extract_context_subgoal_and_tool(self, response) -> Tuple[str, str, str]:
        def normalize_tool_name(tool_name: str) -> str:
            for tool in self.available_tools:
                if tool.lower() in tool_name.lower():
                    return tool
            return "No matched tool given: " + tool_name
        
        try:
            print(f"DEBUG: extract_context_subgoal_and_tool - response type: {type(response)}")
            print(f"DEBUG: extract_context_subgoal_and_tool - response: {response}")
            
            # Check if response is a NextStep object
            if hasattr(response, 'context') and hasattr(response, 'sub_goal') and hasattr(response, 'tool_name'):
                print(f"DEBUG: Processing NextStep object")
                print(f"DEBUG: response.context: {response.context}")
                print(f"DEBUG: response.sub_goal: {response.sub_goal}")
                print(f"DEBUG: response.tool_name: {response.tool_name}")
                print(f"DEBUG: available_tools: {self.available_tools}")
                
                context = response.context.strip()
                sub_goal = response.sub_goal.strip()
                tool_name = normalize_tool_name(response.tool_name.strip())
                
                print(f"DEBUG: Normalized tool_name: {tool_name}")
                
                return context, sub_goal, tool_name
            # Check if response is a string (fallback for non-structured models)
            elif isinstance(response, str):
                print("WARNING: Received string response instead of NextStep object")
                # Try to parse the string response to extract context, sub_goal, and tool_name
                try:
                    lines = response.split('\n')
                    context = ""
                    sub_goal = ""
                    tool_name = ""
                    
                    for line in lines:
                        line = line.strip()
                        if line.lower().startswith('<context>') and not line.lower().startswith('<context>:'):
                            # Only handle <context>...</context> tags, not <context>: format
                            context = line.split('<context>')[1].split('</context>')[0].strip()
                        elif line.lower().startswith('context:'):
                            # Handle both <context>: and context: formats
                            parts = line.split('context:', 1)
                            if len(parts) > 1:
                                context = parts[1].lstrip(' :')
                            else:
                                context = ""
                        elif line.lower().startswith('<sub_goal>') and not line.lower().startswith('<sub_goal>:'):
                            sub_goal = line.split('<sub_goal>')[1].split('</sub_goal>')[0].strip()
                        elif line.lower().startswith('sub_goal:'):
                            parts = line.split('sub_goal:', 1)
                            if len(parts) > 1:
                                sub_goal = parts[1].lstrip(' :')
                            else:
                                sub_goal = ""
                        elif line.lower().startswith('<tool_name>') and not line.lower().startswith('<tool_name>:'):
                            tool_name = line.split('<tool_name>')[1].split('</tool_name>')[0].strip()
                        elif line.lower().startswith('tool_name:'):
                            parts = line.split('tool_name:', 1)
                            if len(parts) > 1:
                                tool_name = parts[1].lstrip(' :')
                            else:
                                tool_name = ""
                    
                    # Normalize tool name
                    tool_name = normalize_tool_name(tool_name)
                    
                    return context, sub_goal, tool_name
                    
                except Exception as parse_error:
                    print(f"Error parsing string response: {parse_error}")
                    return "", "", "Error parsing tool name"
            else:
                return "", "", "Unknown response type"
        except Exception as e:
            print(f"Error in extract_context_subgoal_and_tool: {e}")
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
            'Cell_State_Analyzer_Tool', 'Fibroblast_Activation_Scorer_Tool',
            'Image_Preprocessor_Tool'
        ]
        actions = memory.get_actions()
        for action in actions:
            tool_name = action.get('tool_name', '')
            if any(bio_tool in tool_name for bio_tool in bioimage_tools):
                return True
        
        return False
    
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
        
        # Filter metadata to match filtered tools
        toolbox_metadata = {
            tool: self.toolbox_metadata[tool]
            for tool in available_tools
            if tool in self.toolbox_metadata
        }
        
        if excluded_tools:
            print(f"Step {step_count}: Domain={self.detected_domain}, Excluded tools: {excluded_tools}")
        
        # Get tools grouped by priority for prompt
        tools_by_priority = self.priority_manager.format_tools_by_priority(available_tools)
        
        # Get used tools from memory for dependency checking
        used_tools = [action.get('tool_name', '') for action in memory.get_actions() if 'tool_name' in action]
        recommended_tools = self.priority_manager.get_recommended_next_tools(
            available_tools, used_tools, self.detected_domain
        )
        
        # Format recommended tools for prompt (first 5 for brevity)
        recommended_tools_str = ", ".join(recommended_tools[:5]) if recommended_tools else "None"
        if len(recommended_tools) > 5:
            recommended_tools_str += f" (and {len(recommended_tools) - 5} more)"
        
        prompt_generate_next_step = f"""
Task: Determine the optimal next step to address the given query based on the provided analysis, available tools, and previous steps taken.

Context:
Query: {question}
Image: {image if not bytes_mode else 'image.jpg'}
Query Analysis: {query_analysis}
Detected Task Domain: {self.detected_domain}

Available Tools (organized by priority):
{tools_by_priority}

Recommended Next Tools (considering dependencies and priorities):
{recommended_tools_str}

Tool Metadata:
{toolbox_metadata}

Previous Steps and Their Results:
{memory.get_actions()}

Tools Already Used: {used_tools if used_tools else "None"}

Current Step: {step_count} in {max_step_count} steps
Remaining Steps: {max_step_count - step_count}

Instructions:
1. Analyze the context thoroughly, including the query, its analysis, any image, available tools and their metadata, and previous steps taken.

2. Determine the most appropriate next step by considering:
   - Key objectives from the query analysis
   - Capabilities of available tools (see priority grouping above)
   - Logical progression of problem-solving
   - Outcomes from previous steps
   - Tool dependencies (some tools require other tools to run first)
   - Current step count and remaining steps

3. Tool Selection Priority (IMPORTANT - FOLLOW THIS ORDER):
   Tools are organized by priority level. You MUST follow this priority order:
   
   - HIGH Priority: Use these tools FIRST if they are relevant to the query
     * Core tools for bioimage analysis and specialized analysis tools
     * Examples: Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, Organoid_Segmenter_Tool,
                Single_Cell_Cropper_Tool, Cell_State_Analyzer_Tool, Fibroblast_Activation_Scorer_Tool, Analysis_Visualizer_Tool
   
   - MEDIUM Priority: Use these general-purpose tools when needed
     * Examples: Object_Detector_Tool, Image_Captioner_Tool
   
   - LOW Priority: Use sparingly, only when necessary
     * Utility tools and code generation tools (use only when no other tool can solve the query)
     * Examples: Text_Detector_Tool, Python_Code_Generator_Tool
   
   IMPORTANT: Always prefer tools from higher priority levels (HIGH > MEDIUM > LOW).
   Do NOT use LOW priority code generation tools if any higher-priority tool can address the query.

4. CRITICAL: Bioimage Analysis Chain Priority (MUST FOLLOW THIS ORDER):
   For bioimage analysis queries, you MUST follow this specific tool chain order:
   
   Step 1: Image_Preprocessor_Tool (if image quality needs improvement)
   Step 2: Choose ONE segmentation tool based on image type:
           - Cell_Segmenter_Tool (for phase-contrast cell images)
           - Nuclei_Segmenter_Tool (for nuclei/fluorescence images)
           - Organoid_Segmenter_Tool (for organoid images)
   Step 3: Single_Cell_Cropper_Tool (requires segmentation output from Step 2)
   Step 4: Cell_State_Analyzer_Tool (requires single-cell crops from Step 3)
   
   This chain MUST be followed in order: Image_Preprocessor → Segmenter → Single_Cell_Cropper → Cell_State_Analyzer
   Do NOT skip steps or use tools out of order.

5. Check Tool Dependencies:
   Some tools require other tools to run first:
   - Single_Cell_Cropper_Tool requires Nuclei_Segmenter_Tool, Cell_Segmenter_Tool, or Organoid_Segmenter_Tool
   - Cell_State_Analyzer_Tool requires Single_Cell_Cropper_Tool
   - Fibroblast_Activation_Scorer_Tool requires Cell_State_Analyzer_Tool (uses h5ad output)
   
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
            print(f"Next step usage: {self.last_usage}")
        else:
            next_step = llm_response
            self.last_usage = {}
        
        # Validate the selected tool
        if hasattr(next_step, 'tool_name') and next_step.tool_name:
            validation_result = self._validate_tool_selection(
                next_step.tool_name, available_tools, used_tools, self.detected_domain
            )
            if not validation_result['valid']:
                print(f"WARNING: Tool selection validation failed: {validation_result['reason']}")
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
        finished_tools = [action['tool_name'] for action in memory.get_actions() if 'tool_name' in action]
        
        # Special case: If this is a fibroblast analysis and cell state analyzer just finished,
        # but activation scorer hasn't run yet, we should continue
        if (is_fibroblast_query and 
            "Cell_State_Analyzer_Tool" in finished_tools and 
            "Fibroblast_Activation_Scorer_Tool" not in finished_tools and
            "Fibroblast_Activation_Scorer_Tool" in self.available_tools):
            
            print(f"DEBUG: Fibroblast analysis detected - cell state analyzer finished but activation scorer not run yet")
            print(f"DEBUG: Continuing to activation scorer for complete fibroblast analysis")
            
            return MemoryVerification(
                analysis="Cell state analysis completed successfully. However, the complete fibroblast analysis pipeline requires activation scoring to provide quantitative activation scores. The Fibroblast_Activation_Scorer_Tool is available and should be run to complete the analysis.",
                stop_signal=False
            )

        prompt_memory_verification = f"""
Task: Thoroughly evaluate the completeness and accuracy of the memory for fulfilling the given query, considering the potential need for additional tool usage.

Conversation so far:
{conversation_context}

Context:
Query: {question}
Image: {image_info}
Available Tools: {self.available_tools}
Toolbox Metadata: {self.toolbox_metadata}
Initial Analysis: {query_analysis}
Memory (tools used and results): {memory.get_actions(llm_safe=True)}

Detailed Instructions:
1. Carefully analyze the query, initial analysis, and image (if provided):
   - Identify the main objectives of the query.
   - Note any specific requirements or constraints mentioned.
   - If an image is provided, consider its relevance and what information it contributes.

2. Review the available tools and their metadata:
   - Understand the capabilities and limitations and best practices of each tool.
   - Consider how each tool might be applicable to the query.

3. Examine the memory content in detail:
   - Review each tool used and its execution results.
   - Assess how well each tool's output contributes to answering the query.

4. Critical Evaluation (address each point explicitly):
   a) Completeness: Does the memory fully address all aspects of the query?
      - Identify any parts of the query that remain unanswered.
      - Consider if all relevant information has been extracted from the image (if applicable).
      - IMPORTANT: For analysis tasks, ensure that the actual analysis has been performed, not just data preparation.
      - For example: If the query asks to "analyze cell states", ensure that cell state analysis has been performed, not just cell cropping.
      - CRITICAL: If the query asks for analysis and you see analysis results with visualizations, distributions, and statistics, the task is COMPLETE.

   b) Unused Tools: Are there any unused tools that could provide additional relevant information?
      - Specify which unused tools might be helpful and why.
      - Pay special attention to analysis tools that could provide insights from prepared data.
      - IMPORTANT: If the main analysis has been completed and only unused tools remain for minor enhancements, this does NOT justify continuing.

   c) Inconsistencies: Are there any contradictions or conflicts in the information provided?
      - If yes, explain the inconsistencies and suggest how they might be resolved.

   d) Verification Needs: Is there any information that requires further verification due to tool limitations?
      - Identify specific pieces of information that need verification and explain why.

   e) Ambiguities: Are there any unclear or ambiguous results that could be clarified by using another tool?
      - Point out specific ambiguities and suggest which tools could help clarify them.

5. Final Determination:
   Based on your thorough analysis, decide if the memory is complete and accurate enough to generate the final output, or if additional tool usage is necessary.
   
   CRITICAL CHECKLIST FOR STOPPING (PRIORITY: Stop early if query is satisfied):
   - Has the MAIN query been answered? If yes, STOP immediately.
   - Are there analysis results, visualizations, counts, or statistics that answer the query? If yes, STOP.
   - If the query asked for "compare", "count", "analyze", "detect" and you have those results, STOP.
   - If the query asked for specific outputs (charts, counts, comparisons) and they exist, STOP.
   - DO NOT continue just because there are unused tools available.
   - DO NOT use Python_Code_Generator_Tool unless NO other tools can solve the query.
   - STOP if the query is satisfied, even if some tools haven't been used.

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
                print(f"Error reading image file: {str(e)}")

        try:
            llm_response = self.llm_engine_mm.generate(input_data, response_format=MemoryVerification)
            
            # Extract content and usage from response
            if isinstance(llm_response, dict) and 'content' in llm_response:
                stop_verification = llm_response['content']
                # Store usage info for later access
                self.last_usage = llm_response.get('usage', {})
                print(f"Memory verification usage: {self.last_usage}")
            else:
                stop_verification = llm_response
                self.last_usage = {}
            
            # Debug: Check if the response is properly formatted
            print(f"Stop verification response type: {type(stop_verification)}")
            print(f"Stop verification response: {stop_verification}")
            
            # Check if we got a string response (non-structured model) instead of MemoryVerification object
            if isinstance(stop_verification, str):
                print("WARNING: Received string response instead of MemoryVerification object")
                # Try to parse the string response to extract analysis and stop_signal
                try:
                    # Look for patterns in the response to extract information
                    lines = stop_verification.split('\n')
                    analysis = ""
                    stop_signal = False
                    
                    for line in lines:
                        line = line.strip()
                        if line.lower().startswith('analysis:'):
                            analysis = line[9:].strip()
                        elif line.lower().startswith('stop_signal:'):
                            signal_text = line[12:].strip().lower()
                            stop_signal = signal_text in ['true', 'yes', '1']
                    
                    # If we couldn't parse properly, use the whole response as analysis
                    if not analysis:
                        analysis = stop_verification
                    
                    # Create MemoryVerification object manually
                    stop_verification = MemoryVerification(
                        analysis=analysis,
                        stop_signal=stop_signal
                    )
                    print(f"Created MemoryVerification object: analysis='{analysis}', stop_signal={stop_signal}")
                    
                except Exception as parse_error:
                    print(f"Error parsing string response: {parse_error}")
                    # Create a default MemoryVerification object
                    stop_verification = MemoryVerification(
                        analysis=stop_verification,
                        stop_signal=False
                    )
            
            if hasattr(stop_verification, 'analysis'):
                print(f"Analysis attribute exists: {stop_verification.analysis}")
            if hasattr(stop_verification, 'stop_signal'):
                print(f"Stop signal attribute exists: {stop_verification.stop_signal}")
            else:
                print("WARNING: stop_signal attribute not found!")
                
        except Exception as e:
            print(f"Error in response format parsing: {e}")
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
                    
                print(f"Raw response: {raw_content}")
                # Create a basic MemoryVerification object with default values
                stop_verification = MemoryVerification(
                    analysis=raw_content,
                    stop_signal=False  # Default to continue
                )
            except Exception as fallback_error:
                print(f"Fallback error: {fallback_error}")
                # Create a minimal MemoryVerification object
                stop_verification = MemoryVerification(
                    analysis="Error in memory verification",
                    stop_signal=False
                )

        return stop_verification

    def extract_conclusion(self, response: MemoryVerification) -> str:
        try:
            print(f"Extract conclusion - Response type: {type(response)}")
            print(f"Extract conclusion - Response: {response}")
            
            analysis = response.analysis
            stop_signal = response.stop_signal
            print(f"Extract conclusion - Analysis: {analysis}")
            print(f"Extract conclusion - Stop signal: {stop_signal}")
            
            if stop_signal:
                return analysis, 'STOP'
            else:
                return analysis, 'CONTINUE'
        except AttributeError as e:
            print(f"Error accessing MemoryVerification attributes: {e}")
            print(f"Response object type: {type(response)}")
            print(f"Response object attributes: {dir(response)}")
            # Fallback: try to extract from string representation or default to continue
            try:
                if hasattr(response, 'analysis'):
                    analysis = response.analysis
                else:
                    analysis = str(response)
                
                # Default to continue if we can't determine stop_signal
                return analysis, 'CONTINUE'
            except Exception as fallback_error:
                print(f"Fallback error: {fallback_error}")
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
{memory.get_actions()}

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
                print(f"Error reading image file: {str(e)}")

        llm_response = self.llm_engine_mm.generate(input_data)
        
        # Extract content and usage from response
        if isinstance(llm_response, dict) and 'content' in llm_response:
            final_output = llm_response['content']
            # Store usage info for later access
            self.last_usage = llm_response.get('usage', {})
            print(f"Final output usage: {self.last_usage}")
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
{memory.get_actions()}

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
                print(f"Error reading image file: {str(e)}")

        llm_response = self.llm_engine_mm.generate(input_data)
        
        # Extract content and usage from response
        if isinstance(llm_response, dict) and 'content' in llm_response:
            final_output = llm_response['content']
            # Store usage info for later access
            self.last_usage = llm_response.get('usage', {})
            print(f"Direct output usage: {self.last_usage}")
        else:
            final_output = llm_response
            self.last_usage = {}

        return final_output
    
    def run_activation_scorer(self, input_adata_path, reference_path):
        scorer = ActivationScorerTool()
        output_adata_path = scorer.run(input_adata_path, reference_path)
        return output_adata_path
    

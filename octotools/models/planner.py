import os
import re
from PIL import Image
from io import BytesIO
from typing import Dict, Any, List, Tuple

from octotools.engine.openai import ChatOpenAI
from octotools.models.memory import Memory
from octotools.models.formatters import QueryAnalysis, NextStep, MemoryVerification, StepPlan
from octotools.registry import REGISTRY

class Planner:
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None, available_tools: List = None, api_key: str = None):
        self.llm_engine_name = llm_engine_name
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []
        self.api_key = api_key
        
        # Initialize LLM engines
        self.llm_engine = ChatOpenAI(model_string=llm_engine_name, is_multimodal=False, api_key=api_key)
        self.llm_engine_mm = ChatOpenAI(model_string=llm_engine_name, is_multimodal=True, api_key=api_key)
        
        # Initialize response storage
        self.base_response = None
        self.query_analysis = None
        
        # Initialize token usage tracking
        self.last_usage = {}

    def _available_capabilities(self) -> list:
        return list(REGISTRY._capability_index.keys())

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
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

        self.base_response = self.llm_engine_mm(input_data, max_tokens=max_tokens)

        return self.base_response


    def analyze_query(self, question: str, image: str, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> str:
        image_info = self.get_image_info(image)
        print("image_info: ", image_info)

        query_prompt = f"""
Task: Analyze the given query with accompanying inputs and determine the skills and tools needed to address it effectively.

Conversation so far:
{conversation_context}

Available tools: {self.available_tools}

Metadata for the tools: {self.toolbox_metadata}

Image: {image_info}

Query: {question}

Instructions:
1. Carefully read and understand the query and any accompanying inputs.
2. Identify the main objectives or tasks within the query.
3. List the specific skills that would be necessary to address the query comprehensively.
4. Examine the available tools in the toolbox and determine which ones are relevant and useful for addressing the query. Make sure to consider the user metadata for each tool, including limitations and potential applications (if available).
5. Provide a brief explanation for each skill and tool you've identified, describing how it would contribute to answering the query.

SPECIAL CONSIDERATIONS FOR FIBROBLAST ANALYSIS:
- If the query involves fibroblast analysis, cell state analysis, or activation scoring, you MUST include the complete analysis pipeline tools
- For fibroblast analysis tasks, the relevant tools should include: Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool, Fibroblast_State_Analyzer_Tool
- If the query specifically asks for activation scoring, quantification, or numerical analysis, also include Fibroblast_Activation_Scorer_Tool
- The tools should be listed in the logical order they would be used in the analysis pipeline

TASK RECOGNITION GUIDELINES:
- Cell analysis tasks: Include preprocessing, segmentation, and analysis tools
- Image processing tasks: Include preprocessing and relevant analysis tools
- Object detection tasks: Include detection and analysis tools
- Text extraction tasks: Include text detection and extraction tools
- Research tasks: Include search and knowledge tools
- General analysis tasks: Include appropriate preprocessing and analysis tools

Your response should include:
1. A concise summary of the query's main points and objectives, as well as content in any accompanying inputs.
2. A list of required skills, with a brief explanation for each.
3. A list of relevant tools from the toolbox, with a brief explanation of how each tool would be utilized and its potential limitations. List tools in the logical order they would be used.
4. Any additional considerations that might be important for addressing the query effectively.

Please present your analysis in a clear, structured format.
"""

        input_data = [query_prompt]
        if image_info and "image_path" in image_info:
            try:
                with open(image_info["image_path"], 'rb') as file:
                    image_bytes = file.read()
                input_data.append(image_bytes)
            except Exception as e:
                print(f"Error reading image file: {str(e)}")

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
        """Simplified: return tool name if present; context/sub_goal are unused for capability plan."""
        try:
            if isinstance(response, StepPlan):
                return "", "", response.tool_name or ""
            if hasattr(response, 'context') and hasattr(response, 'sub_goal') and hasattr(response, 'tool_name'):
                return response.context.strip(), response.sub_goal.strip(), response.tool_name.strip()
            if isinstance(response, str):
                return "", "", response
            return "", "", "Unknown response type"
        except Exception as e:
            print(f"Error in extract_context_subgoal_and_tool: {e}")
            return "", "", "Error extracting tool name"
    
    def generate_next_step(self, question: str, image: str, query_analysis: str, memory: Memory, step_count: int, max_step_count: int, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> StepPlan:
        """Capability-driven next step selection using the registry (no heuristics)."""
        available_specs = [REGISTRY.get(name) for name in self.available_tools if REGISTRY.get(name)]
        available_caps = sorted({cap for spec in available_specs for cap in spec.capabilities}) or self._available_capabilities()
        prompt_generate_next_step = f"""
Task: Determine the single best next action to advance the user's goal.

Conversation so far:
{conversation_context}

Query: {question}
Image: {image if not bytes_mode else 'image.jpg'}
Query Analysis: {query_analysis}

Available Capabilities: {available_caps}
Available Tools: {[spec.name for spec in available_specs]}
Tool Descriptions: {[{'name': spec.name, 'capabilities': spec.capabilities, 'description': spec.description} for spec in available_specs]}

Previous Steps: {memory.get_actions()}
Current Step: {step_count} of {max_step_count}

Return JSON with fields:
- capability: one of Available Capabilities
- tool_name: exact tool name from Available Tools or null
- required_inputs: list of concrete inputs needed from prior steps or the query
- rationale: brief reason this step advances the goal
"""
        next_step_response = self.llm_engine.generate(prompt_generate_next_step, response_format=StepPlan)
        
        if isinstance(next_step_response, dict) and 'content' in next_step_response:
            self.last_usage = next_step_response.get('usage', {})
            plan = next_step_response['content']
        else:
            plan = next_step_response
            self.last_usage = {}

        if isinstance(plan, StepPlan):
            return plan
        if isinstance(plan, dict):
            return StepPlan(**plan)
        fallback_cap = available_caps[0] if available_caps else "general"
        return StepPlan(capability=fallback_cap, tool_name=None, required_inputs=[], rationale="Fallback plan")

    def verificate_memory(self, question: str, image: str, query_analysis: str, memory: Memory, bytes_mode: bool = False, conversation_context: str = "", **kwargs) -> MemoryVerification:
        if bytes_mode:
            image_info = self.get_image_info_bytes(image)
        else:
            image_info = self.get_image_info(image)

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
Memory (tools used and results): {memory.get_actions()}

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
   
   CRITICAL CHECKLIST FOR STOPPING:
   - Has the query been fully answered with actual analysis results?
   - Are there analysis results with visualizations, distributions, or statistics?
   - Does the current state represent the final analysis, or just intermediate data preparation?
   - If the query asked for analysis and analysis results are present, STOP.
   - If the query asked for classification and classification results are present, STOP.
   - If the query asked for detection and detection results are present, STOP.
   - Only continue if there are UNANSWERED aspects of the query that require additional tools.

   CRITICAL CHECKLIST FOR CONTINUING:
   - Are there UNANSWERED parts of the query that require additional tools?
   - Is the current state only data preparation without actual analysis?
   - Are there analysis tools available that could provide insights from prepared data AND the query specifically asks for such analysis?

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
    

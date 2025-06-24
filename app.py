import os
import sys
import json
import argparse
import time
import io
import uuid
import torch
import shutil
from PIL import Image
import numpy as np
from tifffile import imwrite as tiff_write
from typing import List, Dict, Any, Iterator
import matplotlib.pyplot as plt
import gradio as gr
from gradio import ChatMessage
from pathlib import Path
from huggingface_hub import CommitScheduler
from octotools.models.formatters import ToolCommand
import random
import traceback
import psutil  # For memory usage
from llm_evaluation_scripts.hf_model_configs import HF_MODEL_CONFIGS

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from octotools.models.initializer import Initializer
from octotools.models.planner import Planner
from octotools.models.memory import Memory
from octotools.models.executor import Executor
from octotools.models.utils import make_json_serializable, VisualizationConfig

# Custom JSON encoder to handle ToolCommand objects
class CustomEncoder(json.JSONEncoder):
    def default(self, obj):
        if isinstance(obj, ToolCommand):
            return str(obj)  # Convert ToolCommand to its string representation
        return super().default(obj)

# Get Huggingface token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")
IS_SPACES = os.getenv('SPACE_ID') is not None
DATASET_DIR = Path("solver_cache")  # the directory to save the dataset
DATASET_DIR.mkdir(parents=True, exist_ok=True) 
global QUERY_ID
QUERY_ID = None

# Comment out problematic CommitScheduler to avoid permission issues
# scheduler = CommitScheduler(
#     repo_id="lupantech/OctoTools-Gradio-Demo-User-Data",
#     repo_type="dataset",
#     folder_path=DATASET_DIR,
#     path_in_repo="solver_cache",  # Update path in repo
#     token=HF_TOKEN
# )


def save_query_data(query_id: str, query: str, image_path: str) -> None:
    """Save query data to Huggingface dataset"""
    # Save query metadata
    query_cache_dir = DATASET_DIR / query_id
    query_cache_dir.mkdir(parents=True, exist_ok=True)
    query_file = query_cache_dir / "query_metadata.json"

    query_metadata = {
        "query_id": query_id,
        "query_text": query,
        "datetime": time.strftime("%Y%m%d_%H%M%S"),
        "image_path": image_path if image_path else None
    }
    
    print(f"Saving query metadata to {query_file}")
    with query_file.open("w") as f:
        json.dump(query_metadata, f, indent=4)
    
    # # NOTE: As we are using the same name for the query cache directory as the dataset directory,
    # # NOTE: we don't need to copy the content from the query cache directory to the query directory.
    # # Copy all content from root_cache_dir to query_dir
    # import shutil
    # shutil.copytree(args.root_cache_dir, query_data_dir, dirs_exist_ok=True)


def save_feedback(query_id: str, feedback_type: str, feedback_text: str = None) -> None:
    """
    Save user feedback to the query directory.
    
    Args:
        query_id: Unique identifier for the query
        feedback_type: Type of feedback ('upvote', 'downvote', or 'comment')
        feedback_text: Optional text feedback from user
    """

    feedback_data_dir = DATASET_DIR / query_id
    feedback_data_dir.mkdir(parents=True, exist_ok=True)
    
    feedback_data = {
        "query_id": query_id,
        "feedback_type": feedback_type,
        "feedback_text": feedback_text,
        "datetime": time.strftime("%Y%m%d_%H%M%S")
    }
    
    # Save feedback in the query directory
    feedback_file = feedback_data_dir / "feedback.json"
    print(f"Saving feedback to {feedback_file}")
    
    # If feedback file exists, update it
    if feedback_file.exists():
        with feedback_file.open("r") as f:
            existing_feedback = json.load(f)
            # Convert to list if it's a single feedback entry
            if not isinstance(existing_feedback, list):
                existing_feedback = [existing_feedback]
            existing_feedback.append(feedback_data)
            feedback_data = existing_feedback
    
    # Write feedback data
    with feedback_file.open("w") as f:
        json.dump(feedback_data, f, indent=4)


def save_steps_data(query_id: str, memory: Memory) -> None:
    """Save steps data to Huggingface dataset"""
    steps_file = DATASET_DIR / query_id / "all_steps.json"

    memory_actions = memory.get_actions()
    memory_actions = make_json_serializable(memory_actions) # NOTE: make the memory actions serializable
    print("Memory actions: ", memory_actions)

    with steps_file.open("w") as f:
        json.dump(memory_actions, f, indent=4, cls=CustomEncoder)

    
def save_module_data(query_id: str, key: str, value: Any) -> None:
    """Save module data to Huggingface dataset"""
    try:
        key = key.replace(" ", "_").lower()
        module_file = DATASET_DIR / query_id / f"{key}.json"
        value = make_json_serializable(value)  # NOTE: make the value serializable
        with module_file.open("a") as f:
            json.dump(value, f, indent=4, cls=CustomEncoder)
    except Exception as e:
        print(f"Warning: Failed to save as JSON: {e}")
        # Fallback to saving as text file
        text_file = DATASET_DIR / query_id / f"{key}.txt"
        try:
            with text_file.open("a") as f:
                f.write(str(value) + "\n")
            print(f"Successfully saved as text file: {text_file}")
        except Exception as e:
            print(f"Error: Failed to save as text file: {e}")

########### End of Test Huggingface Dataset ###########

class Solver:
    def __init__(
        self,
        planner,
        memory,
        executor,
        task: str,
        task_description: str,
        output_types: str = "base,final,direct",
        index: int = 0,
        verbose: bool = True,
        max_steps: int = 10,
        max_time: int = 60,
        query_cache_dir: str = "solver_cache"
    ):
        self.planner = planner
        self.memory = memory
        self.executor = executor
        self.task = task
        self.task_description = task_description
        self.index = index
        self.verbose = verbose
        self.max_steps = max_steps
        self.max_time = max_time
        self.query_cache_dir = query_cache_dir

        self.output_types = output_types.lower().split(',')
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."

        # Add statistics for evaluation
        self.step_times = []
        self.step_tokens = []
        self.total_tokens = 0
        self.step_memory = []
        self.max_memory = 0
        self.step_costs = []
        self.total_cost = 0.0
        self.start_time = None
        self.end_time = None

    def stream_solve_user_problem(self, user_query: str, user_image, api_key: str, messages: List[ChatMessage]) -> Iterator:
        import time
        import os
        self.start_time = time.time()
        process = psutil.Process(os.getpid())
        visual_outputs_for_gradio = []
        visual_description = "*Ready to display analysis results and processed images.*"
        
        # Handle image input - simplified logic based on original OctoTools
        print(f"=== DEBUG: Image processing started ===")
        print(f"DEBUG: user_image type: {type(user_image)}")
        print(f"DEBUG: user_image is None: {user_image is None}")
        
        if user_image:
            print(f"DEBUG: user_image exists, processing...")
            # Handle different image input formats from Gradio
            if isinstance(user_image, dict) and 'path' in user_image:
                img_path = user_image['path']
                print(f"DEBUG: extracted path from dict: {img_path}")
            elif isinstance(user_image, str) and os.path.exists(user_image):
                img_path = user_image
                print(f"DEBUG: user_image is valid string path: {img_path}")
            elif hasattr(user_image, 'save'):
                print(f"DEBUG: user_image is a PIL Image, saving...")
                # It's a PIL Image object - save it like in original version
                img_path = os.path.join(self.query_cache_dir, 'query_image.jpg')
                print(f"DEBUG: saving to path: {img_path}")
                print(f"DEBUG: query_cache_dir exists: {os.path.exists(self.query_cache_dir)}")
                try:
                    user_image.save(img_path)
                    print(f"DEBUG: Image saved successfully to: {img_path}")
                    print(f"DEBUG: file exists after save: {os.path.exists(img_path)}")
                except Exception as e:
                    print(f"DEBUG: Error saving image: {e}")
                    import traceback
                    print(f"DEBUG: Full traceback: {traceback.format_exc()}")
                    img_path = None
            else:
                print(f"DEBUG: user_image is not a recognized format: {type(user_image)}")
                if user_image:
                    print(f"DEBUG: user_image attributes: {dir(user_image)}")
                img_path = None
        else:
            print(f"DEBUG: no user_image provided")
            img_path = None

        print(f"DEBUG: final img_path: {img_path}")
        print(f"=== DEBUG: Image processing completed ===")

        # Set tool cache directory
        _tool_cache_dir = os.path.join(self.query_cache_dir, "tool_cache") # NOTE: This is the directory for tool cache
        self.executor.set_query_cache_dir(_tool_cache_dir) # NOTE: set query cache directory
        
        # Step 1: Display the received inputs
        if user_image:
            messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}\n### 🖼️ Image Uploaded"))
        else:
            messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}"))
        yield messages, "", [], visual_description, "**Progress**: Input received"

        # [Step 3] Initialize problem-solving state
        step_count = 0
        json_data = {"query": user_query, "image": "Image received as bytes"}

        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### 🐙 Deep Thinking:"))
        yield messages, "", [], visual_description, "**Progress**: Starting analysis"

        # [Step 4] Query Analysis - This is the key step that should happen first
        print(f"Debug - Starting query analysis for: {user_query}")
        print(f"Debug - img_path for query analysis: {img_path}")
        try:
            query_analysis = self.planner.analyze_query(user_query, img_path)
            print(f"Debug - Query analysis completed: {len(query_analysis)} characters")
            json_data["query_analysis"] = query_analysis
            query_analysis = query_analysis.replace("Concise Summary:", "**Concise Summary:**\n")
            query_analysis = query_analysis.replace("Required Skills:", "**Required Skills:**")
            query_analysis = query_analysis.replace("Relevant Tools:", "**Relevant Tools:**")
            query_analysis = query_analysis.replace("Additional Considerations:", "**Additional Considerations:**")
            messages.append(ChatMessage(role="assistant", 
                                        content=f"{query_analysis}",
                                        metadata={"title": "### 🔍 Step 0: Query Analysis"}))
            yield messages, query_analysis, [], visual_description, "**Progress**: Query analysis completed"

            # Save the query analysis data
            query_analysis_data = {"query_analysis": query_analysis, "time": round(time.time() - self.start_time, 5)}
            save_module_data(QUERY_ID, "step_0_query_analysis", query_analysis_data)
        except Exception as e:
            print(f"Error in query analysis: {e}")
            error_msg = f"⚠️ Error during query analysis: {str(e)}"
            messages.append(ChatMessage(role="assistant", 
                                        content=error_msg,
                                        metadata={"title": "### 🔍 Step 0: Query Analysis (Error)"}))
            yield messages, error_msg, [], visual_description, "**Progress**: Error in query analysis"
            return

        # Execution loop (similar to your step-by-step solver)
        while step_count < self.max_steps and (time.time() - self.start_time) < self.max_time:
            step_count += 1
            step_start = time.time()
            mem_before = process.memory_info().rss / 1024 / 1024  # MB
            messages.append(ChatMessage(role="OctoTools", 
                                        content=f"Generating the {step_count}-th step...",
                                        metadata={"title": f"🔄 Step {step_count}"}))
            yield messages, query_analysis, visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count}"

            # [Step 5] Generate the next step
            next_step = self.planner.generate_next_step(user_query, img_path, query_analysis, self.memory, step_count, self.max_steps)
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
            step_data = {"step_count": step_count, "context": context, "sub_goal": sub_goal, "tool_name": tool_name, "time": round(time.time() - self.start_time, 5)}
            save_module_data(QUERY_ID, f"step_{step_count}_action_prediction", step_data)

            # Display the step information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Context:** {context}\n\n**Sub-goal:** {sub_goal}\n\n**Tool:** `{tool_name}`",
                metadata={"title": f"### 🎯 Step {step_count}: Action Prediction ({tool_name})"}))
            yield messages, query_analysis, visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Action predicted"

            # Handle tool execution or errors
            if tool_name not in self.planner.available_tools:
                messages.append(ChatMessage(
                    role="assistant", 
                    content=f"⚠️ Error: Tool '{tool_name}' is not available."))
                yield messages, query_analysis, visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Tool not available"
                continue

            # [Step 6-7] Generate and execute the tool command
            safe_path = img_path.replace("\\", "\\\\") if img_path else None
            tool_command = self.executor.generate_tool_command(user_query, safe_path, context, sub_goal, tool_name, self.planner.toolbox_metadata[tool_name])
            analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
            result = self.executor.execute_tool_command(tool_name, command)
            result = make_json_serializable(result)
            print(f"Tool '{tool_name}' result:", result)
            
            # Generate dynamic visual description based on tool and results
            visual_description = self.generate_visual_description(tool_name, result, visual_outputs_for_gradio)
            
            if isinstance(result, dict):
                if "visual_outputs" in result:
                    visual_output_files = result["visual_outputs"]
                    visual_outputs_for_gradio = []
                    for file_path in visual_output_files:
                        try:
                            # Skip comparison plots and non-image files
                            if "comparison" in os.path.basename(file_path).lower():
                                continue
                            if not file_path.lower().endswith(('.png', '.jpg', '.jpeg', '.tif', '.tiff')):
                                continue
                                
                            # Check if file exists and is readable
                            if not os.path.exists(file_path):
                                print(f"Warning: Image file not found: {file_path}")
                                continue
                            
                            # Check file size
                            if os.path.getsize(file_path) == 0:
                                print(f"Warning: Image file is empty: {file_path}")
                                continue
                                
                            # Use (image, label) tuple format to preserve filename for download
                            image = Image.open(file_path)
                            
                            # Validate image data
                            if image.size[0] == 0 or image.size[1] == 0:
                                print(f"Warning: Invalid image size: {file_path}")
                                continue
                            
                            # Convert to RGB if necessary for Gradio compatibility
                            if image.mode not in ['RGB', 'L', 'RGBA']:
                                try:
                                    image = image.convert('RGB')
                                except Exception as e:
                                    print(f"Warning: Failed to convert image {file_path} to RGB: {e}")
                                    continue
                            
                            # Additional validation for image data
                            try:
                                # Test if image can be converted to array
                                img_array = np.array(image)
                                if img_array.size == 0 or np.isnan(img_array).any():
                                    print(f"Warning: Invalid image data in {file_path}")
                                    continue
                            except Exception as e:
                                print(f"Warning: Failed to validate image data for {file_path}: {e}")
                                continue
                            
                            filename = os.path.basename(file_path)
                            
                            # Create descriptive label based on filename
                            if "processed" in filename.lower():
                                label = f"Processed Image: {filename}"
                            elif "corrected" in filename.lower():
                                label = f"Illumination Corrected: {filename}"
                            elif "segmented" in filename.lower():
                                label = f"Segmented Result: {filename}"
                            elif "detected" in filename.lower():
                                label = f"Detection Result: {filename}"
                            elif "zoomed" in filename.lower():
                                label = f"Zoomed Region: {filename}"
                            elif "crop" in filename.lower():
                                label = f"Single Cell Crop: {filename}"
                            else:
                                label = f"Analysis Result: {filename}"
                            
                            visual_outputs_for_gradio.append((image, label))
                            print(f"Successfully loaded image for Gradio: {filename}")
                            
                        except Exception as e:
                            print(f"Warning: Failed to load image {file_path} for Gradio. Error: {e}")
                            import traceback
                            print(f"Full traceback: {traceback.format_exc()}")
                            continue

            # Display the command generation information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Analysis:** {analysis}\n\n**Explanation:** {explanation}\n\n**Command:**\n```python\n{command}\n```",
                metadata={"title": f"### 📝 Step {step_count}: Command Generation ({tool_name})"}))
            yield messages, query_analysis, visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Command generated"

            # Save the command generation data
            command_generation_data = {
                "analysis": analysis,
                "explanation": explanation,
                "command": command,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_generation", command_generation_data)
            
            # Display the command execution result
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Result:**\n```json\n{json.dumps(result, indent=4)}\n```",
                metadata={"title": f"### 🛠️ Step {step_count}: Command Execution ({tool_name})"}))
            yield messages, query_analysis, visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Command executed"

            # Save the command execution data
            command_execution_data = {
                "result": result,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_execution", command_execution_data)

            # [Step 8] Memory update and stopping condition
            self.memory.add_action(step_count, tool_name, sub_goal, tool_command, result)
            stop_verification = self.planner.verificate_memory(user_query, img_path, query_analysis, self.memory)
            context_verification, conclusion = self.planner.extract_conclusion(stop_verification)

            # Save the context verification data
            context_verification_data = {
                "stop_verification": context_verification,
                "conclusion": conclusion,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_context_verification", context_verification_data)    

            # Display the context verification result
            conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
            messages.append(ChatMessage(
                role="assistant", 
                content=f"**Analysis:**\n{context_verification}\n\n**Conclusion:** `{conclusion}` {conclusion_emoji}",
                metadata={"title": f"### 🤖 Step {step_count}: Context Verification"}))
            yield messages, query_analysis, visual_outputs_for_gradio, visual_description, f"**Progress**: Step {step_count} - Context verified"

            # After tool execution, estimate tokens and cost
            # Try to get token usage from result if available, else estimate
            tokens_used = 0
            cost = 0.0
            if isinstance(result, dict):
                if 'usage' in result and 'total_tokens' in result['usage']:
                    tokens_used = result['usage']['total_tokens']
                elif 'token_usage' in result:
                    tokens_used = result['token_usage']
                # Cost estimation (example: OpenAI $0.00001/token)
                if 'usage' in result and 'total_tokens' in result['usage']:
                    cost = result['usage']['total_tokens'] * 0.00001
                elif 'token_usage' in result:
                    cost = result['token_usage'] * 0.00001
            self.step_tokens.append(tokens_used)
            self.total_tokens += tokens_used
            self.step_costs.append(cost)
            self.total_cost += cost
            mem_after = process.memory_info().rss / 1024 / 1024  # MB
            self.step_memory.append(mem_after)
            if mem_after > self.max_memory:
                self.max_memory = mem_after
            step_end = time.time()
            self.step_times.append(step_end - step_start)

            if conclusion == 'STOP':
                break

        self.end_time = time.time()

        # Step 7: Generate Final Output (if needed)
        final_answer = ""
        if 'direct' in self.output_types:
            messages.append(ChatMessage(role="assistant", content="<br>"))
            direct_output = self.planner.generate_direct_output(user_query, img_path, self.memory)
            
            # Extract conclusion from the final answer
            conclusion = ""
            if "### Conclusion:" in direct_output:
                conclusion = direct_output.split("### Conclusion:")[1].strip()
            elif "Conclusion:" in direct_output:
                conclusion = direct_output.split("Conclusion:")[1].strip()
            elif "**Conclusion:**" in direct_output:
                conclusion = direct_output.split("**Conclusion:**")[1].strip()
            else:
                # If no clear conclusion section, use the entire output
                # This ensures we always have content to display
                conclusion = direct_output.strip()
            
            conclusion += f"\n\n---\n"
            conclusion += f"**Step-wise elapsed time (s):** {self.step_times}\n"
            conclusion += f"**Total elapsed time (s):** {self.end_time - self.start_time:.2f}\n"
            conclusion += f"**Step-wise tokens used:** {self.step_tokens}\n"
            conclusion += f"**Total tokens used:** {self.total_tokens}\n"
            conclusion += f"**Step-wise memory usage (MB):** {self.step_memory}\n"
            conclusion += f"**Max memory usage (MB):** {self.max_memory:.2f}\n"
            conclusion += f"**Step-wise estimated cost ($):** {[round(c,6) for c in self.step_costs]}\n"
            conclusion += f"**Total estimated cost ($):** {round(self.total_cost,6)}\n"
            
            final_answer = f"🐙 **Conclusion:**\n{conclusion}"
            # Remove the ChatMessage that displays final answer in reasoning steps
            # messages.append(ChatMessage(role="assistant", content=f"### 🐙 Final Answer:\n{direct_output}"))
            yield messages, final_answer, visual_outputs_for_gradio, visual_description, "**Progress**: Completed!"

            # Save the direct output data
            direct_output_data = {
                "direct_output": direct_output,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, "direct_output", direct_output_data)

        if 'final' in self.output_types:
            final_output = self.planner.generate_final_output(user_query, img_path, self.memory) # Disabled visibility for now
            # messages.append(ChatMessage(role="assistant", content=f"🎯 Final Output:\n{final_output}"))
            # yield messages

            # Save the final output data
            final_output_data = {
                "final_output": final_output,
                "time": round(time.time() - self.start_time, 5)
            }
            save_module_data(QUERY_ID, "final_output", final_output_data)

        # Step 8: Completion Message
        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### ✅ Query Solved!"))
        # Use the final answer if available, otherwise use a default message
        completion_text = final_answer if final_answer else "Analysis completed successfully"
        yield messages, completion_text, visual_outputs_for_gradio, visual_description, "**Progress**: Analysis completed!"

    def generate_visual_description(self, tool_name: str, result: dict, visual_outputs: list) -> str:
        """
        Generate dynamic visual description based on tool type and results.
        """
        if not visual_outputs:
            return "*Ready to display analysis results and processed images.*"
        
        # Count different types of images
        processed_count = sum(1 for _, label in visual_outputs if "processed" in label.lower())
        corrected_count = sum(1 for _, label in visual_outputs if "corrected" in label.lower())
        segmented_count = sum(1 for _, label in visual_outputs if "segmented" in label.lower())
        detected_count = sum(1 for _, label in visual_outputs if "detected" in label.lower())
        zoomed_count = sum(1 for _, label in visual_outputs if "zoomed" in label.lower())
        cropped_count = sum(1 for _, label in visual_outputs if "crop" in label.lower())
        analyzed_count = sum(1 for _, label in visual_outputs if "analysis" in label.lower() or "distribution" in label.lower())
        
        # Generate tool-specific descriptions
        tool_descriptions = {
            "Image_Preprocessor_Tool": f"*Displaying {processed_count} processed image(s) from illumination correction and brightness adjustment.*",
            "Object_Detector_Tool": f"*Showing {detected_count} detection result(s) with identified objects and regions of interest.*",
            "Image_Captioner_Tool": "*Displaying image analysis results with detailed morphological descriptions.*",
            "Relevant_Patch_Zoomer_Tool": f"*Showing {zoomed_count} zoomed region(s) highlighting key areas of interest.*",
            "Advanced_Object_Detector_Tool": f"*Displaying {detected_count} advanced detection result(s) with enhanced object identification.*",
            "Nuclei_Segmenter_Tool": f"*Showing {segmented_count} segmentation result(s) with identified nuclei regions.*",
            "Single_Cell_Cropper_Tool": f"*Displaying {cropped_count} single-cell crop(s) generated from nuclei segmentation results.*",
            "Cell_Morphology_Analyzer_Tool": "*Displaying cell morphology analysis results with detailed structural insights.*",
            "Fibroblast_Activation_Detector_Tool": "*Showing fibroblast activation state analysis with morphological indicators.*",
            "Fibroblast_State_Analyzer_Tool": f"*Displaying {analyzed_count} fibroblast state analysis result(s) with cell state distributions and statistics.*"
        }
        
        # Return tool-specific description or generic one
        if tool_name in tool_descriptions:
            return tool_descriptions[tool_name]
        else:
            total_images = len(visual_outputs)
            return f"*Displaying {total_images} analysis result(s) from {tool_name}.*"


def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the OctoTools demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--task", default="minitoolbench", help="Task to run.")
    parser.add_argument("--task_description", default="", help="Task description.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Generalist_Solution_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--query_id", default=None, help="Query ID.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")

    # NOTE: Add new arguments
    parser.add_argument("--run_baseline_only", type=bool, default=False, help="Run only the baseline (no toolbox).")
    parser.add_argument("--openai_api_source", default="we_provided", choices=["we_provided", "user_provided"], help="Source of OpenAI API key.")
    return parser.parse_args()


def solve_problem_gradio(user_query, user_image, max_steps=10, max_time=60, api_key=None, llm_model_engine=None, enabled_fibroblast_tools=None, enabled_general_tools=None, clear_previous_viz=False):
    """
    Solve a problem using the Gradio interface with optional visualization clearing.
    
    Args:
        user_query: The user's query
        user_image: The user's image
        max_steps: Maximum number of reasoning steps
        max_time: Maximum analysis time in seconds
        api_key: OpenAI API key
        llm_model_engine: Language model engine
        enabled_fibroblast_tools: List of enabled fibroblast tools
        enabled_general_tools: List of enabled general tools
        clear_previous_viz: Whether to clear previous visualizations
    """
    # Combine the tool lists
    enabled_tools = (enabled_fibroblast_tools or []) + (enabled_general_tools or [])
    
    # Generate a unique query ID
    query_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8] # e.g, 20250217_062225_612f2474
    print(f"Query ID: {query_id}")

    # NOTE: update the global variable to save the query ID
    global QUERY_ID
    QUERY_ID = query_id

    # Handle visualization clearing based on user preference
    if clear_previous_viz:
        print("🧹 Clearing output_visualizations directory as requested...")
        # Manually clear the directory
        output_viz_dir = os.path.join(os.getcwd(), 'output_visualizations')
        if os.path.exists(output_viz_dir):
            import shutil
            shutil.rmtree(output_viz_dir)
            print(f"✅ Cleared output directory: {output_viz_dir}")
        os.makedirs(output_viz_dir, exist_ok=True)
        print("✅ Output directory cleared successfully")
    else:
        print("📁 Preserving output_visualizations directory for continuity...")
        # Just ensure directory exists without clearing
        output_viz_dir = os.path.join(os.getcwd(), 'output_visualizations')
        os.makedirs(output_viz_dir, exist_ok=True)
        print("✅ Output directory preserved - all charts will be retained")

    # Create a directory for the query ID
    query_cache_dir = os.path.join(DATASET_DIR.name, query_id) # NOTE
    os.makedirs(query_cache_dir, exist_ok=True)

    if api_key is None or api_key.strip() == "":
        return [[gr.ChatMessage(role="assistant", content="""⚠️ **API Key Configuration Required**

To use this application, you need to set up your OpenAI API key. You can do this in one of two ways:

**Option 1: Environment Variable (Recommended)**
Set the `OPENAI_API_KEY` environment variable:
```bash
export OPENAI_API_KEY="your-api-key-here"
```

**Option 2: Manual Input**
If you prefer to enter the API key manually, please contact the administrator to enable manual input mode.

For more information about obtaining an OpenAI API key, visit: https://platform.openai.com/api-keys
""")]], "", [], "**Progress**: Ready"
    
    # Debug: Print enabled_tools
    print(f"Debug - enabled_tools: {enabled_tools}")
    print(f"Debug - type of enabled_tools: {type(enabled_tools)}")
    
    # Ensure enabled_tools is a list and not empty
    if not enabled_tools:
        print("⚠️ No tools selected in UI, defaulting to all available tools.")
        # Get all tools from the directory as a fallback
        tools_dir = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'octotools', 'tools')
        enabled_tools = [
            d for d in os.listdir(tools_dir)
            if os.path.isdir(os.path.join(tools_dir, d)) and not d.startswith('__')
        ]
    elif isinstance(enabled_tools, str):
        enabled_tools = [enabled_tools]
    elif not isinstance(enabled_tools, list):
        enabled_tools = list(enabled_tools) if hasattr(enabled_tools, '__iter__') else []

    if not enabled_tools:
        print("❌ Critical Error: Could not determine a default tool list. Using Generalist_Solution_Generator_Tool as a last resort.")
        enabled_tools = ["Generalist_Solution_Generator_Tool"]

    print(f"Debug - final enabled_tools: {enabled_tools}")
    
    # Save the query data
    save_query_data(
        query_id=query_id,
        query=user_query,
        image_path=os.path.join(query_cache_dir, 'query_image.jpg') if user_image else None
    )

    # Instantiate Initializer
    try:
        initializer = Initializer(
            enabled_tools=enabled_tools,
            model_string=llm_model_engine,
            api_key=api_key
        )
        print(f"Debug - Initializer created successfully with {len(initializer.available_tools)} tools")
    except Exception as e:
        print(f"Error creating Initializer: {e}")
        return [[gr.ChatMessage(role="assistant", content=f"⚠️ Error: Failed to initialize tools. {str(e)}")]], "", []

    # Instantiate Planner
    try:
        planner = Planner(
            llm_engine_name=llm_model_engine,
            toolbox_metadata=initializer.toolbox_metadata,
            available_tools=initializer.available_tools,
            api_key=api_key
        )
        print(f"Debug - Planner created successfully")
    except Exception as e:
        print(f"Error creating Planner: {e}")
        return [[gr.ChatMessage(role="assistant", content=f"⚠️ Error: Failed to initialize planner. {str(e)}")]], "", []

    # Instantiate Memory
    memory = Memory()

    # Instantiate Executor
    executor = Executor(
        llm_engine_name=llm_model_engine,
        query_cache_dir=query_cache_dir, # NOTE
        enable_signal=False,
        api_key=api_key,
        initializer=initializer
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        memory=memory,
        executor=executor,
        task="minitoolbench",  # Default task
        task_description="",   # Default empty description
        output_types="base,final,direct",  # Default output types
        verbose=True,          # Default verbose
        max_steps=max_steps,
        max_time=max_time,
        query_cache_dir=query_cache_dir # NOTE
    )

    if solver is None:
        return [[gr.ChatMessage(role="assistant", content="⚠️ Error: Failed to initialize solver.")]], "", []

    # Initialize messages list
    messages = []
    
    try:
        # Stream the solution
        for messages, text_output, gallery_output, visual_desc, progress_md in solver.stream_solve_user_problem(user_query, user_image, api_key, messages):
            # Save steps data
            save_steps_data(query_id, memory)
            
            # Return the current state
            yield messages, text_output, gallery_output, progress_md
            
    except Exception as e:
        print(f"Error in solve_problem_gradio: {e}")
        import traceback
        error_traceback = traceback.format_exc()
        print(f"Full traceback: {error_traceback}")
        
        # Create error message for UI
        error_message = f"⚠️ Error occurred during analysis:\n\n**Error Type:** {type(e).__name__}\n**Error Message:** {str(e)}\n\nPlease check your input and try again."
        
        # Return error message in the expected format
        error_messages = [gr.ChatMessage(role="assistant", content=error_message)]
        yield error_messages, "", [], "**Progress**: Error occurred"
    finally:
        print(f"Task completed for query_id: {query_id}. Preparing to clean up cache directory: {query_cache_dir}")
        try:
            # Add a check to prevent deleting the root solver_cache
            if query_cache_dir != DATASET_DIR.name and DATASET_DIR.name in query_cache_dir:
                # Preserve output_visualizations directory - DO NOT CLEAR IT
                # This allows users to keep all generated charts until they start a new analysis
                output_viz_dir = os.path.join(os.getcwd(), 'output_visualizations')
                if os.path.exists(output_viz_dir):
                    print(f"📁 Preserving output_visualizations directory: {output_viz_dir}")
                    print(f"💡 All generated charts are preserved for review")
                
                # Add a small delay to ensure files are written
                time.sleep(1)
                
                # Clean up the cache directory (but preserve visualizations)
                shutil.rmtree(query_cache_dir)
                print(f"✅ Successfully cleaned up cache directory: {query_cache_dir}")
                print(f"💡 Note: All visualization files are preserved in output_visualizations/ directory")
            else:
                print(f"⚠️ Skipping cleanup for safety. Path was: {query_cache_dir}")
        except Exception as e:
            print(f"❌ Error cleaning up cache directory {query_cache_dir}: {e}")


def main(args):
    #################### Gradio Interface ####################
    with gr.Blocks() as demo:
        # Theming https://www.gradio.app/guides/theming-guide
        
        gr.Markdown("# Chat with FBagent: An augmented agentic approach to resolve fibroblast states at single-cell multimodal resolution")  # Title
        gr.Markdown("""
        **FBagent** is an open-source assistant for interpreting cell images, powered by large language models and tool-based reasoning. It supports morphological reasoning, patch extraction, and multi-omic integration.
        """)
        
        with gr.Row():
            # Left control panel
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### ⚙️ Model Configuration")
                
                # API Key - Manual input option
                api_key = gr.Textbox(
                    placeholder="Enter your OpenAI API key",
                    type="password",
                    label="🔑 OpenAI API Key",
                    value=os.getenv("OPENAI_API_KEY", "")
                )

                # Model and limits
                model_choices = list(HF_MODEL_CONFIGS.keys())
                language_model = gr.Dropdown(
                    choices=model_choices,
                    value="gpt-4o",
                    label="Language Model"
                )
                max_steps = gr.Slider(1, 15, value=10, label="Max Reasoning Steps")
                max_time = gr.Slider(60, 600, value=300, label="Max Analysis Time (seconds)")

                # Visualization options
                gr.Markdown("#### 📊 Visualization Options")
                clear_previous_viz = gr.Checkbox(
                    label="Clear previous visualizations", 
                    value=False,
                    info="Check this to clear all previous charts when starting new analysis"
                )

                # Tool selection
                gr.Markdown("#### 🛠️ Available Tools")
                
                # Fibroblast analysis tools
                fibroblast_tools = [
                    "Image_Preprocessor_Tool",
                    "Nuclei_Segmenter_Tool",
                    "Single_Cell_Cropper_Tool",
                    "Fibroblast_State_Analyzer_Tool"
                ]
                
                # General tools
                general_tools = [
                    "Generalist_Solution_Generator_Tool",
                    "Python_Code_Generator_Tool",
                    "ArXiv_Paper_Searcher_Tool",
                    "Pubmed_Search_Tool",
                    "Nature_News_Fetcher_Tool",
                    "Google_Search_Tool",
                    "Wikipedia_Knowledge_Searcher_Tool",
                    "URL_Text_Extractor_Tool",
                    "Object_Detector_Tool",
                    "Image_Captioner_Tool", 
                    "Relevant_Patch_Zoomer_Tool",
                    "Text_Detector_Tool",
                    "Advanced_Object_Detector_Tool"
                ]
                
                with gr.Accordion("🧬 Fibroblas Tools", open=True):
                    enabled_fibroblast_tools = gr.CheckboxGroup(
                        choices=fibroblast_tools, 
                        value=fibroblast_tools, 
                        label="Select Fibroblast Analysis Tools"
                    )

                with gr.Accordion("🧩 General Tools", open=False):
                    enabled_general_tools = gr.CheckboxGroup(
                        choices=general_tools, 
                        label="Select General Purpose Tools"
                    )

                with gr.Row():
                    gr.Button("Select Fibroblast Tools", size="sm").click(
                        lambda: fibroblast_tools, outputs=enabled_fibroblast_tools
                    )
                    gr.Button("Select All Tools", size="sm").click(
                        lambda: (fibroblast_tools, general_tools), 
                        outputs=[enabled_fibroblast_tools, enabled_general_tools]
                    )
                    gr.Button("Clear Selection", size="sm").click(
                        lambda: ([], []), 
                        outputs=[enabled_fibroblast_tools, enabled_general_tools]
                    )

            # Main interface
            with gr.Column(scale=5):
                # Input area
                gr.Markdown("### 📤 Data Input")
                with gr.Row():
                    with gr.Column(scale=1):
                        user_image = gr.Image(
                            label="Upload an Image", 
                            type="pil", 
                            height=350
                        )
                    with gr.Column(scale=1):
                        user_query = gr.Textbox(
                            label="Analysis Question", 
                            placeholder="Describe the cell features or states you want to analyze...", 
                            lines=15
                        )
                        
                # Submit button
                with gr.Row():
                    with gr.Column(scale=6):
                        run_button = gr.Button("🚀 Start Analysis", variant="primary", size="lg")
                        progress_md = gr.Markdown("**Progress**: Ready")

                # Output area - two columns instead of three
                gr.Markdown("### 📊 Analysis Results")
                with gr.Row():
                    # Reasoning steps
                    with gr.Column(scale=1):
                        gr.Markdown("#### 🔍 Reasoning Steps")
                        chatbot_output = gr.Chatbot(
                            type="messages", 
                            height=700,
                            show_label=False
                        )

                    # Combined analysis report and visual output
                    with gr.Column(scale=1):
                        gr.Markdown("#### 📝 Analysis Report & Visual Output")
                        with gr.Group():
                            #gr.Markdown("*The final analysis conclusion and key findings will appear here.*")
                            text_output = gr.Markdown(
                                value="",
                                height=350
                            )
                            gallery_output = gr.Gallery(
                                label=None, 
                                show_label=False,
                                height=350,
                                columns=2,
                                rows=2
                            )

                # Bottom row for examples
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("## 💡 Try these examples with suggested tools.")
                        
                        # Define example lists
                        fibroblast_examples = [
                            ["Image Preprocessing", "examples/A5_01_1_1_Phase Contrast_001.png", "Normalize this phase contrast image.", 
                             "Image_Preprocessor_Tool", "Illumination-corrected and brightness-normalized phase contrast image."],
                            ["Cell Identification", "examples/A5_01_1_1_Phase Contrast_001.png", "How many cells are there in this image.", 
                             "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool", "258 cells are identified and their nuclei are labeled."],
                            ["Single-Cell Cropping", "examples/A5_01_1_1_Phase Contrast_001.png", "Crop single cells from the segmented nuclei in this image.", 
                             "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool", "Individual cell crops extracted from the image."],
                            ["Fibroblast State Analysis", "examples/fibroblast.png", "Analyze the fibroblast cell states in this image.", 
                             "Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool, Fibroblast_State_Analyzer_Tool", "540 cells identified and segmented successfully. Comprehensive analysis of fibroblast cell states have been performed with visualizations."]
                        ]
                        
                        general_examples = [
                            ["Pathology Diagnosis", "examples/pathology.jpg", "What are the cell types in this image?", 
                             "Generalist_Solution_Generator_Tool, Image_Captioner_Tool, Relevant_Patch_Zoomer_Tool", "Need expert insights."],
                            ["Visual Reasoning", "examples/rotting_kiwi.png", "You are given a 3 x 3 grid in which each cell can contain either no kiwi, one fresh kiwi, or one rotten kiwi. Every minute, any fresh kiwi that is 4-directionally adjacent to a rotten kiwi also becomes rotten. What is the minimum number of minutes that must elapse until no cell has a fresh kiwi?", 
                             "Image_Captioner_Tool", "4 minutes"],
                            ["Scientific Research", None, "What are the research trends in tool agents with large language models for scientific discovery? Please consider the latest literature from ArXiv, PubMed, Nature, and news sources.", 
                             "ArXiv_Paper_Searcher_Tool, Pubmed_Search_Tool, Nature_News_Fetcher_Tool", "Open-ended question. No reference answer."]
                        ]

                        # Helper function to distribute tools
                        def distribute_tools(category, img, q, tools_str, ans):
                            selected_tools = [tool.strip() for tool in tools_str.split(',')]
                            selected_fibroblast = [tool for tool in selected_tools if tool in fibroblast_tools]
                            selected_general = [tool for tool in selected_tools if tool in general_tools]
                            return img, q, selected_fibroblast, selected_general

                        gr.Markdown("#### 🧬 Fibroblast Analysis Examples")
                        gr.Examples(
                            examples=fibroblast_examples,
                            inputs=[gr.Textbox(label="Category", visible=False), user_image, user_query, gr.Textbox(label="Select Tools", visible=False), gr.Textbox(label="Reference Answer", visible=False)],
                            outputs=[user_image, user_query, enabled_fibroblast_tools, enabled_general_tools],
                            fn=distribute_tools,
                            cache_examples=False
                        )
                        
                        gr.Markdown("#### 🧩 General Purpose Examples")
                        gr.Examples(
                            examples=general_examples,
                            inputs=[gr.Textbox(label="Category", visible=False), user_image, user_query, gr.Textbox(label="Select Tools", visible=False), gr.Textbox(label="Reference Answer", visible=False)],
                            outputs=[user_image, user_query, enabled_fibroblast_tools, enabled_general_tools],
                            fn=distribute_tools,
                            cache_examples=False
                        )

        # Button click event
        run_button.click(
            solve_problem_gradio,
            [user_query, user_image, max_steps, max_time, api_key, language_model, enabled_fibroblast_tools, enabled_general_tools, clear_previous_viz],
            [chatbot_output, text_output, gallery_output, progress_md]
        )

    #################### Gradio Interface ####################

    # Launch configuration
    if IS_SPACES:
        # HuggingFace Spaces config
        demo.launch(
            server_name="0.0.0.0",
            server_port=7860,
            share=False,
            debug=False
        )
    else:
        # Local development config
        demo.launch(
            server_name="0.0.0.0",
            server_port=1048,
            debug=True,
            share=False
        )

if __name__ == "__main__":
    args = parse_arguments()
    
    # Set default API source to use environment variables
    if not hasattr(args, 'openai_api_source') or args.openai_api_source is None:
        args.openai_api_source = "we_provided"

    # All available tools
    all_tools = [
        # Cell analysis tools
        "Object_Detector_Tool",           # Cell detection and counting
        "Image_Captioner_Tool",           # Cell morphology description
        "Relevant_Patch_Zoomer_Tool",     # Cell region zoom analysis
        "Text_Detector_Tool",             # Text recognition in images
        "Advanced_Object_Detector_Tool",  # Advanced cell detection
        "Image_Preprocessor_Tool",        # Image preprocessing and enhancement
        "Nuclei_Segmenter_Tool",          # Nuclei segmentation
        "Single_Cell_Cropper_Tool",        # Single cell cropping
        "Fibroblast_State_Analyzer_Tool",  # Fibroblast state analysis
        
        # General analysis tools
        "Generalist_Solution_Generator_Tool",  # Comprehensive analysis generation
        "Python_Code_Generator_Tool",          # Code generation
        
        # Research literature tools
        "ArXiv_Paper_Searcher_Tool",      # arXiv paper search
        "Pubmed_Search_Tool",             # PubMed literature search
        "Nature_News_Fetcher_Tool",       # Nature news fetching
        "Google_Search_Tool",             # Google search
        "Wikipedia_Knowledge_Searcher_Tool",  # Wikipedia search
        "URL_Text_Extractor_Tool",        # URL text extraction
    ]
    args.enabled_tools = all_tools

    # NOTE: Use the same name for the query cache directory as the dataset directory
    args.root_cache_dir = DATASET_DIR.name
    
    # Print environment information
    print("\n=== Environment Information ===")
    print(f"Running in HuggingFace Spaces: {IS_SPACES}")
    print(f"CUDA Available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        print(f"CUDA Device: {torch.cuda.get_device_name(0)}")
    #print(f"API Key Source: {args.openai_api_source}")
    print("==============================\n")
    
    main(args)

    if __name__ == '__main__':
        from generate_llm_eval_visualization import (
            plot_time_distribution, plot_token_distribution, plot_success_rate
        )
        import pandas as pd
        # 假设你的example数据变量名为example_data
        df = pd.DataFrame(example_data)
        plot_time_distribution(df)
        plot_token_distribution(df)
        plot_success_rate(df)


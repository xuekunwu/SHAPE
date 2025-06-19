import os
import sys
import json
import argparse
import time
import io
import uuid
import torch
from PIL import Image
import numpy as np
from tifffile import imwrite as tiff_write
from typing import List, Dict, Any, Iterator
import matplotlib.pyplot as plt
import gradio as gr
from gradio import ChatMessage
from pathlib import Path
from huggingface_hub import CommitScheduler

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from octotools.models.initializer import Initializer
from octotools.models.planner import Planner
from octotools.models.memory import Memory
from octotools.models.executor import Executor
from octotools.models.utils import make_json_serializable

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
        json.dump(memory_actions, f, indent=4)

    
def save_module_data(query_id: str, key: str, value: Any) -> None:
    """Save module data to Huggingface dataset"""
    try:
        key = key.replace(" ", "_").lower()
        module_file = DATASET_DIR / query_id / f"{key}.json"
        value = make_json_serializable(value)  # NOTE: make the value serializable
        with module_file.open("a") as f:
            json.dump(value, f, indent=4)
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


    def stream_solve_user_problem(self, user_query: str, user_image, api_key: str, messages: List[ChatMessage]) -> Iterator:
        visual_output_files = []
        
        if user_image:
            # Handle different image input formats from Gradio
            if isinstance(user_image, dict):
                # Gradio Image component returns dict with 'image' key
                if 'image' in user_image:
                    user_image = user_image['image']
                else:
                    # Try to get the first value if it's a dict
                    user_image = list(user_image.values())[0] if user_image else None
            
            if user_image and hasattr(user_image, 'format'):
                # It's a PIL Image object
                try:
                    original_format = (user_image.format() if callable(user_image.format) else user_image.format or 'PNG').upper()
                except:
                    original_format = 'PNG'
                    
                if original_format in ['TIFF', 'TIF']:
                    img_ext = 'query_image.tif'
                    img_path = os.path.join(self.query_cache_dir, img_ext)
                    tiff_write(img_path, np.array(user_image))
                else:
                    img_ext = 'query_image.png'
                    img_path = os.path.join(self.query_cache_dir, img_ext)
                    user_image.save(img_path)
            elif user_image:
                # It's a numpy array or other format
                img_ext = 'query_image.png'
                img_path = os.path.join(self.query_cache_dir, img_ext)
                if isinstance(user_image, np.ndarray):
                    from PIL import Image
                    Image.fromarray(user_image).save(img_path)
                else:
                    # Try to save as is
                    try:
                        user_image.save(img_path)
                    except:
                        img_path = None
            else:
                img_path = None
        else:
            img_path = None

        # Set tool cache directory
        _tool_cache_dir = os.path.join(self.query_cache_dir, "tool_cache") # NOTE: This is the directory for tool cache
        self.executor.set_query_cache_dir(_tool_cache_dir) # NOTE: set query cache directory
        
        # Step 1: Display the received inputs
        if user_image:
            messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}\n### 🖼️ Image Uploaded"))
        else:
            messages.append(ChatMessage(role="assistant", content=f"### 📝 Received Query:\n{user_query}"))
        yield messages, [], None, None

        # [Step 3] Initialize problem-solving state
        start_time = time.time()
        step_count = 0
        json_data = {"query": user_query, "image": "Image received as bytes"}

        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### 🐙 Reasoning Steps"))
        yield messages, [], None, None

        # [Step 4] Query Analysis
        print(f"Debug - Starting query analysis for: {user_query}")
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
            yield messages, [], None, None

            # Save the query analysis data
            query_analysis_data = {"query_analysis": query_analysis, "time": round(time.time() - start_time, 5)}
            save_module_data(QUERY_ID, "step_0_query_analysis", query_analysis_data)
        except Exception as e:
            print(f"Error in query analysis: {e}")
            error_msg = f"⚠️ Error during query analysis: {str(e)}"
            messages.append(ChatMessage(role="assistant", 
                                        content=error_msg,
                                        metadata={"title": "### 🔍 Step 0: Query Analysis (Error)"}))
            yield messages, [], None, None
            return

        # Execution loop (similar to your step-by-step solver)
        while step_count < self.max_steps and (time.time() - start_time) < self.max_time:
            step_count += 1
            messages.append(ChatMessage(role="OctoTools", 
                                        content=f"Generating the {step_count}-th step...",
                                        metadata={"title": f"🔄 Step {step_count}"}))
            yield messages, [], None, None

            # [Step 5] Generate the next step
            next_step = self.planner.generate_next_step(user_query, img_path, query_analysis, self.memory, step_count, self.max_steps)
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
            step_data = {"step_count": step_count, "context": context, "sub_goal": sub_goal, "tool_name": tool_name, "time": round(time.time() - start_time, 5)}
            save_module_data(QUERY_ID, f"step_{step_count}_action_prediction", step_data)

            # Display the step information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Context:** {context}\n\n**Sub-goal:** {sub_goal}\n\n**Tool:** `{tool_name}`",
                metadata={"title": f"### 🎯 Step {step_count}: Action Prediction ({tool_name})"}))
            yield messages, [], None, None

            # Handle tool execution or errors
            if tool_name not in self.planner.available_tools:
                messages.append(ChatMessage(
                    role="assistant", 
                    content=f"⚠️ Error: Tool '{tool_name}' is not available."))
                yield messages, [], None, None
                continue

            # [Step 6-7] Generate and execute the tool command
            tool_command = self.executor.generate_tool_command(user_query, img_path, context, sub_goal, tool_name, self.planner.toolbox_metadata[tool_name])
            analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
            result = self.executor.execute_tool_command(tool_name, command)
            print(f"Tool '{tool_name}' result:", result)
            if isinstance(result, dict):
                if "visual_outputs" in result:
                    visual_output_files = result["visual_outputs"]  # Use the processed image paths directly

            # Display the command generation information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Analysis:** {analysis}\n\n**Explanation:** {explanation}\n\n**Command:**\n```python\n{command}\n```",
                metadata={"title": f"### 📝 Step {step_count}: Command Generation ({tool_name})"}))
            yield messages, [], None, None

            # Save the command generation data
            command_generation_data = {
                "analysis": analysis,
                "explanation": explanation,
                "command": command,
                "time": round(time.time() - start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_generation", command_generation_data)
            
            # Display the command execution result
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Result:**\n```json\n{json.dumps(result, indent=4)}\n```",
                metadata={"title": f"### 🛠️ Step {step_count}: Command Execution ({tool_name})"}))
            yield messages, [], None, None

            # Save the command execution data
            command_execution_data = {
                "result": result,
                "time": round(time.time() - start_time, 5)
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
                "time": round(time.time() - start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_context_verification", context_verification_data)    

            # Display the context verification result
            conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
            messages.append(ChatMessage(
                role="assistant", 
                content=f"**Analysis:**\n{context_verification}\n\n**Conclusion:** `{conclusion}` {conclusion_emoji}",
                metadata={"title": f"### 🤖 Step {step_count}: Context Verification"}))
            yield messages, [], None, None

            if conclusion == 'STOP':
                break

        # Step 7: Generate Final Output (if needed)
        if 'direct' in self.output_types:
            messages.append(ChatMessage(role="assistant", content="<br>"))
            direct_output = self.planner.generate_direct_output(user_query, img_path, self.memory)
            messages.append(ChatMessage(role="assistant", content=f"### 🐙 Final Answer:\n{direct_output}"))
            yield messages, [], None, None

            # Save the direct output data
            direct_output_data = {
                "direct_output": direct_output,
                "time": round(time.time() - start_time, 5)
            }
            save_module_data(QUERY_ID, "direct_output", direct_output_data)

        if 'final' in self.output_types:
            final_output = self.planner.generate_final_output(user_query, img_path, self.memory)
            # messages.append(ChatMessage(role="assistant", content=f"🎯 Final Output:\n{final_output}"))
            # yield messages

            # Save the final output data
            final_output_data = {
                "final_output": final_output,
                "time": round(time.time() - start_time, 5)
            }
            save_module_data(QUERY_ID, "final_output", final_output_data)

        # Step 8: Completion Message
        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### ✅ Query Solved!"))
        messages.append(ChatMessage(role="assistant", content="How do you like the output from OctoTools 🐙? Please give us your feedback below. \n\n👍 If the answer is correct or the reasoning steps are helpful, please upvote the output. \n👎 If it is incorrect or the reasoning steps are not helpful, please downvote the output. \n💬 If you have any suggestions or comments, please leave them below.\n\nThank you for using OctoTools! 🐙"))
        yield messages, [], None, None
        

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


def solve_problem_gradio(user_query, user_image, max_steps=10, max_time=60, api_key=None, llm_model_engine=None, enabled_tools=None):
    """
    Wrapper function to connect the solver to Gradio.
    Streams responses from `solver.stream_solve_user_problem` for real-time UI updates.
    """

    # Generate Unique Query ID (Date and first 8 characters of UUID)
    query_id = time.strftime("%Y%m%d_%H%M%S") + "_" + str(uuid.uuid4())[:8] # e.g, 20250217_062225_612f2474
    print(f"Query ID: {query_id}")

    # NOTE: update the global variable to save the query ID
    global QUERY_ID
    QUERY_ID = query_id

    # Create a directory for the query ID
    query_cache_dir = os.path.join(DATASET_DIR.name, query_id) # NOTE
    os.makedirs(query_cache_dir, exist_ok=True)

    if api_key is None:
        return [[gr.ChatMessage(role="assistant", content="⚠️ Error: OpenAI API Key is required.")]], "", []
    
    # Debug: Print enabled_tools
    print(f"Debug - enabled_tools: {enabled_tools}")
    print(f"Debug - type of enabled_tools: {type(enabled_tools)}")
    
    # Ensure enabled_tools is a list
    if enabled_tools is None:
        enabled_tools = ["Generalist_Solution_Generator_Tool"]
    elif isinstance(enabled_tools, str):
        enabled_tools = [enabled_tools]
    elif not isinstance(enabled_tools, list):
        enabled_tools = list(enabled_tools) if hasattr(enabled_tools, '__iter__') else ["Generalist_Solution_Generator_Tool"]
    
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
        api_key=api_key
    )

    # Instantiate Solver
    solver = Solver(
        planner=planner,
        memory=memory,
        executor=executor,
        task=args.task,
        task_description=args.task_description,
        output_types=args.output_types,  # Add new parameter
        verbose=args.verbose,
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
        for messages, text_output, gallery_output, progress_md in solver.stream_solve_user_problem(user_query, user_image, api_key, messages):
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


def main(args):
    #################### Gradio Interface ####################
    with gr.Blocks(theme=gr.themes.Soft()) as demo:
        
        gr.Markdown("# Chat with FBagent: An augmented agentic approach to resolve fibroblast states at single-cell multimodal resolution")  # Title
        gr.Markdown("""
        **FB Agent** is an open-source assistant for interpreting cell images, powered by large language models and tool-based reasoning. It supports morphological reasoning, patch extraction, and multi-omic integration.
        """)
        
        with gr.Row():
            # Left control panel
            with gr.Column(scale=1, min_width=250):
                gr.Markdown("### ⚙️ Analysis Settings")
                
                # API Key
                if args.openai_api_source == "user_provided":
                    api_key = gr.Textbox(
                        placeholder="Enter your OpenAI API key",
                        type="password",
                        label="🔑 API Key"
                    )
                else:
                    api_key = gr.Textbox(
                        value=os.getenv("OPENAI_API_KEY"),
                        visible=False,
                        interactive=False
                    )

                # Model and limits
                gr.Markdown("#### 🤖 AI Model Configuration")
                llm_model_engine = gr.Dropdown(
                    choices=["gpt-4o"], value="gpt-4o", label="Language Model"
                )
                max_steps = gr.Slider(1, 15, value=10, label="Max Reasoning Steps")
                max_time = gr.Slider(60, 600, value=300, label="Max Analysis Time (seconds)")

                # Tool selection
                gr.Markdown("#### 🛠️ Analysis Tools")
                
                # Cell analysis tools
                cell_analysis_tools = [
                    "Object_Detector_Tool",
                    "Image_Captioner_Tool", 
                    "Relevant_Patch_Zoomer_Tool",
                    "Text_Detector_Tool",
                    "Advanced_Object_Detector_Tool"
                ]
                
                # General tools
                general_tools = [
                    "Generalist_Solution_Generator_Tool",
                    "Python_Code_Generator_Tool",
                    "ArXiv_Paper_Searcher_Tool",
                    "Pubmed_Search_Tool"
                ]
                
                # Specialized cell analysis tools (reserved)
                specialized_tools = [
                    "Nuclei_Segmenter_Tool",
                    "Cell_Morphology_Analyzer_Tool", 
                    "Fibroblast_Activation_Detector_Tool"
                ]
                
                all_tools = cell_analysis_tools + general_tools + specialized_tools
                
                enabled_tools = gr.CheckboxGroup(
                    choices=all_tools, 
                    value=cell_analysis_tools, 
                    label="Select Analysis Tools"
                )
                
                with gr.Row():
                    gr.Button("Select Cell Analysis Tools", size="sm").click(
                        lambda: cell_analysis_tools, outputs=enabled_tools
                    )
                    gr.Button("Select All Tools", size="sm").click(
                        lambda: all_tools, outputs=enabled_tools
                    )
                    gr.Button("Clear Selection", size="sm").click(
                        lambda: [], outputs=enabled_tools
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

                # Output area - three columns
                gr.Markdown("### 📊 Analysis Results")
                with gr.Row():
                    # Reasoning steps
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🔍 Reasoning Steps")
                        chatbot_output = gr.Chatbot(
                            type="messages", 
                            height=450,
                            show_label=False
                        )

                    # Text report
                    with gr.Column(scale=2):
                        gr.Markdown("#### 📝 Analysis Report")
                        text_output = gr.Textbox(
                            interactive=False,
                            lines=20,
                            placeholder="The analysis report will appear here...",
                            show_label=False
                        )

                    # Visual output
                    with gr.Column(scale=2):
                        gr.Markdown("#### 🖼️ Visual Output")
                        gallery_output = gr.Gallery(
                            label=None, 
                            show_label=False,
                            height=450,
                            columns=2,
                            rows=3
                        )

                # Bottom row for examples
                with gr.Row():
                    with gr.Column(scale=5):
                        gr.Markdown("")
                        gr.Markdown("""
                                    ## 💡 Try these examples with suggested tools.
                                    """)
                        gr.Examples(
                            examples=[
                                # [ None, "Who is the president of the United States?", ["Google_Search_Tool"]],
                                [ "Logical Reasoning",
                                 None,                             
                                 "How many r letters are in the word strawberry?", 
                                 ["Generalist_Solution_Generator_Tool", "Python_Code_Generator_Tool"], 
                                 "3"],

                                [ "Web Search", 
                                 None, 
                                 "What's up with the upcoming Apple Launch? Any rumors?", 
                                 ["Generalist_Solution_Generator_Tool", "Google_Search_Tool", "Wikipedia_Knowledge_Searcher_Tool", "URL_Text_Extractor_Tool"], 
                                 "Apple's February 19, 2025, event may feature the iPhone SE 4, new iPads, accessories, and rumored iPhone 17 and Apple Watch Series 10."],

                                [ "Arithmetic Reasoning", 
                                 None, 
                                 "Which is bigger, 9.11 or 9.9?", 
                                 ["Generalist_Solution_Generator_Tool", "Python_Code_Generator_Tool"], 
                                 "9.9"],

                                [ "Multi-step Reasoning", 
                                 None, 
                                 "Using the numbers [1, 1, 6, 9], create an expression that equals 24. You must use basic arithmetic operations (+, -, ×, /) and parentheses. For example, one solution for [1, 2, 3, 4] is (1+2+3)×4.", ["Python_Code_Generator_Tool"], 
                                 "((1 + 1) * 9) + 6"],

                                [ "Scientific Research",
                                 None, 
                                 "What are the research trends in tool agents with large language models for scientific discovery? Please consider the latest literature from ArXiv, PubMed, Nature, and news sources.", ["ArXiv_Paper_Searcher_Tool", "Pubmed_Search_Tool", "Nature_News_Fetcher_Tool"],
                                 "Open-ended question. No reference answer."],

                                [ "Visual Perception", 
                                 "examples/baseball.png", 
                                 "How many baseballs are there?", 
                                 ["Object_Detector_Tool"], 
                                 "20"],

                                [ "Visual Reasoning",  
                                 "examples/rotting_kiwi.png", 
                                 "You are given a 3 x 3 grid in which each cell can contain either no kiwi, one fresh kiwi, or one rotten kiwi. Every minute, any fresh kiwi that is 4-directionally adjacent to a rotten kiwi also becomes rotten. What is the minimum number of minutes that must elapse until no cell has a fresh kiwi?", ["Image_Captioner_Tool"], 
                                 "4 minutes"],

                                [ "Medical Image Analysis",
                                 "examples/lung.jpg", 
                                 "What is the organ on the left side of this image?", 
                                 ["Image_Captioner_Tool", "Relevant_Patch_Zoomer_Tool"],
                                 "Lung"],

                                [ "Pathology Diagnosis",
                                 "examples/pathology.jpg", 
                                 "What are the cell types in this image?", 
                                 ["Generalist_Solution_Generator_Tool", "Image_Captioner_Tool", "Relevant_Patch_Zoomer_Tool"],
                                 "Need expert insights."],

                            ],
                            inputs=[gr.Textbox(label="Category", visible=False), user_image, user_query, enabled_tools, gr.Textbox(label="Reference Answer", visible=False)],
                            # label="Try these examples with suggested tools."
                        )

        # Button click event
        run_button.click(
            fn=solve_problem_gradio,
            inputs=[user_query, user_image, max_steps, max_time, api_key, llm_model_engine, enabled_tools],
            outputs=[chatbot_output, text_output, gallery_output, progress_md],
            preprocess=False,
            queue=True,
            show_progress=True
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
            server_port=1019,
            debug=True,
            share=False
        )

if __name__ == "__main__":
    args = parse_arguments()

    # All available tools
    all_tools = [
        # Cell analysis tools
        "Object_Detector_Tool",           # Cell detection and counting
        "Image_Captioner_Tool",           # Cell morphology description
        "Relevant_Patch_Zoomer_Tool",     # Cell region zoom analysis
        "Text_Detector_Tool",             # Text recognition in images
        "Advanced_Object_Detector_Tool",  # Advanced cell detection
        
        # General analysis tools
        "Generalist_Solution_Generator_Tool",  # Comprehensive analysis generation
        "Python_Code_Generator_Tool",          # Code generation
        "Image_Preprocessing_Tool",            # Image preprocessing
        
        # Research literature tools
        "ArXiv_Paper_Searcher_Tool",      # arXiv paper search
        "Pubmed_Search_Tool",             # PubMed literature search
        "Nature_News_Fetcher_Tool",       # Nature news fetching
        "Google_Search_Tool",             # Google search
        "Wikipedia_Knowledge_Searcher_Tool",  # Wikipedia search
        "URL_Text_Extractor_Tool",        # URL text extraction
        
        # Specialized cell analysis tools (reserved)
        "Nuclei_Segmenter_Tool",          # Nuclei segmentation
        "Cell_Morphology_Analyzer_Tool",  # Cell morphology analyzer
        "Fibroblast_Activation_Detector_Tool",  # Fibroblast activation detector
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
    print("==============================\n")
    
    main(args)


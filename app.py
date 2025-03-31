import os
import sys
import json
import argparse
import time
import io
import uuid
from PIL import Image
from typing import List, Dict, Any, Iterator

import gradio as gr
from gradio import ChatMessage

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(current_dir)))
sys.path.insert(0, project_root)

from octotools.models.initializer import Initializer
from octotools.models.planner import Planner
from octotools.models.memory import Memory
from octotools.models.executor import Executor
from octotools.models.utils import make_json_serializable


from pathlib import Path
from huggingface_hub import CommitScheduler

# Get Huggingface token from environment variable
HF_TOKEN = os.getenv("HUGGINGFACE_TOKEN")

########### Test Huggingface Dataset ###########
# Update the HuggingFace dataset constants
DATASET_DIR = Path("solver_cache")  # the directory to save the dataset
DATASET_DIR.mkdir(parents=True, exist_ok=True) 

global QUERY_ID
QUERY_ID = None

scheduler = CommitScheduler(
    repo_id="lupantech/OctoTools-Gradio-Demo-User-Data",
    repo_type="dataset",
    folder_path=DATASET_DIR,
    path_in_repo="solver_cache",  # Update path in repo
    token=HF_TOKEN
)


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


    def stream_solve_user_problem(self, user_query: str, user_image: Image.Image, api_key: str, messages: List[ChatMessage]) -> Iterator[List[ChatMessage]]:
        """
        Streams intermediate thoughts and final responses for the problem-solving process based on user input.
        
        Args:
            user_query (str): The text query input from the user.
            user_image (Image.Image): The uploaded image from the user (PIL Image object).
            messages (list): A list of ChatMessage objects to store the streamed responses.
        """

        if user_image:
            # # Convert PIL Image to bytes (for processing)
            # img_bytes_io = io.BytesIO()
            # user_image.save(img_bytes_io, format="PNG")  # Convert image to PNG bytes
            # img_bytes = img_bytes_io.getvalue()  # Get bytes
            
            # Use image paths instead of bytes,
            # os.makedirs(os.path.join(self.root_cache_dir, 'images'), exist_ok=True)
            # img_path = os.path.join(self.root_cache_dir, 'images', str(uuid.uuid4()) + '.jpg')

            img_path = os.path.join(self.query_cache_dir,  'query_image.jpg')
            user_image.save(img_path)
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
        yield messages

        # # Step 2: Add "thinking" status while processing
        # messages.append(ChatMessage(
        #     role="assistant",
        #     content="",
        #     metadata={"title": "⏳ Thinking: Processing input..."}
        # ))

        # [Step 3] Initialize problem-solving state
        start_time = time.time()
        step_count = 0
        json_data = {"query": user_query, "image": "Image received as bytes"}

        messages.append(ChatMessage(role="assistant", content="<br>"))
        messages.append(ChatMessage(role="assistant", content="### 🐙 Reasoning Steps from OctoTools (Deep Thinking...)"))
        yield messages

        # [Step 4] Query Analysis
        query_analysis = self.planner.analyze_query(user_query, img_path)
        json_data["query_analysis"] = query_analysis
        query_analysis = query_analysis.replace("Concise Summary:", "**Concise Summary:**\n")
        query_analysis = query_analysis.replace("Required Skills:", "**Required Skills:**")
        query_analysis = query_analysis.replace("Relevant Tools:", "**Relevant Tools:**")
        query_analysis = query_analysis.replace("Additional Considerations:", "**Additional Considerations:**")
        messages.append(ChatMessage(role="assistant", 
                                    content=f"{query_analysis}",
                                    metadata={"title": "### 🔍 Step 0: Query Analysis"}))
        yield messages

        # Save the query analysis data
        query_analysis_data = {
            "query_analysis": query_analysis,
            "time": round(time.time() - start_time, 5)
        }
        save_module_data(QUERY_ID, "step_0_query_analysis", query_analysis_data)



        # Execution loop (similar to your step-by-step solver)
        while step_count < self.max_steps and (time.time() - start_time) < self.max_time:
            step_count += 1
            messages.append(ChatMessage(role="OctoTools", 
                                        content=f"Generating the {step_count}-th step...",
                                        metadata={"title": f"🔄 Step {step_count}"}))
            yield messages

            # [Step 5] Generate the next step
            next_step = self.planner.generate_next_step(
                user_query, img_path, query_analysis, self.memory, step_count, self.max_steps
            )
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)
            step_data = {
                "step_count": step_count,
                "context": context,
                "sub_goal": sub_goal,
                "tool_name": tool_name,
                "time": round(time.time() - start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_action_prediction", step_data)

            # Display the step information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Context:** {context}\n\n**Sub-goal:** {sub_goal}\n\n**Tool:** `{tool_name}`",
                metadata={"title": f"### 🎯 Step {step_count}: Action Prediction ({tool_name})"}))
            yield messages

            # Handle tool execution or errors
            if tool_name not in self.planner.available_tools:
                messages.append(ChatMessage(
                    role="assistant", 
                    content=f"⚠️ Error: Tool '{tool_name}' is not available."))
                yield messages
                continue

            # [Step 6-7] Generate and execute the tool command
            tool_command = self.executor.generate_tool_command(
                user_query, img_path, context, sub_goal, tool_name, self.planner.toolbox_metadata[tool_name]
            )
            analysis, explanation, command = self.executor.extract_explanation_and_command(tool_command)
            result = self.executor.execute_tool_command(tool_name, command)
            result = make_json_serializable(result)

            # Display the ommand generation information
            messages.append(ChatMessage(
                role="assistant",
                content=f"**Analysis:** {analysis}\n\n**Explanation:** {explanation}\n\n**Command:**\n```python\n{command}\n```",
                metadata={"title": f"### 📝 Step {step_count}: Command Generation ({tool_name})"}))
            yield messages

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
                # content=f"**Result:**\n```json\n{result}\n```",
                metadata={"title": f"### 🛠️ Step {step_count}: Command Execution ({tool_name})"}))
            yield messages

            # Save the command execution data
            command_execution_data = {
                "result": result,
                "time": round(time.time() - start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_command_execution", command_execution_data)

            # [Step 8] Memory update and stopping condition
            self.memory.add_action(step_count, tool_name, sub_goal, tool_command, result)
            stop_verification = self.planner.verificate_memory(user_query, img_path, query_analysis, self.memory)
            conclusion = self.planner.extract_conclusion(stop_verification)

            # Save the context verification data
            context_verification_data = {
                "stop_verification": stop_verification,
                "conclusion": conclusion,
                "time": round(time.time() - start_time, 5)
            }
            save_module_data(QUERY_ID, f"step_{step_count}_context_verification", context_verification_data)    

            # Display the context verification result
            conclusion_emoji = "✅" if conclusion == 'STOP' else "🛑"
            messages.append(ChatMessage(
                role="assistant", 
                content=f"**Analysis:** {analysis}\n\n**Conclusion:** `{conclusion}` {conclusion_emoji}",
                metadata={"title": f"### 🤖 Step {step_count}: Context Verification"}))
            yield messages

            if conclusion == 'STOP':
                break

        # Step 7: Generate Final Output (if needed)
        if 'direct' in self.output_types:
            messages.append(ChatMessage(role="assistant", content="<br>"))
            direct_output = self.planner.generate_direct_output(user_query, img_path, self.memory)
            messages.append(ChatMessage(role="assistant", content=f"### 🐙 Final Answer:\n{direct_output}"))
            yield messages

            # Save the direct output data
            direct_output_data = {
                "direct_output": direct_output,
                "time": round(time.time() - start_time, 5)
            }
            save_module_data(QUERY_ID, "direct_output", direct_output_data)


        if 'final' in self.output_types:
            final_output = self.planner.generate_final_output(user_query, img_path, self.memory) # Disabled visibility for now
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
        yield messages
        

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
        return [["assistant", "⚠️ Error: OpenAI API Key is required."]]
    
    # Save the query data
    save_query_data(
        query_id=query_id,
        query=user_query,
        image_path=os.path.join(query_cache_dir, 'query_image.jpg') if user_image else None
    )

    # # Initialize Tools
    # enabled_tools = args.enabled_tools.split(",") if args.enabled_tools else []

    # # Hack enabled_tools
    # enabled_tools = ["Generalist_Solution_Generator_Tool"]

    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        model_string=llm_model_engine,
        api_key=api_key
    )

    # Instantiate Planner
    planner = Planner(
        llm_engine_name=llm_model_engine,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        api_key=api_key
    )

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
        return [["assistant", "⚠️ Error: Solver is not initialized. Please restart the application."]]


    messages = []  # Initialize message list
    for message_batch in solver.stream_solve_user_problem(user_query, user_image, api_key, messages):
        yield [msg for msg in message_batch]  # Ensure correct format for Gradio Chatbot

    # Save steps
    save_steps_data(
        query_id=query_id,
        memory=memory
    )


def main(args):
    #################### Gradio Interface ####################
    with gr.Blocks() as demo:
    # with gr.Blocks(theme=gr.themes.Soft()) as demo:
        # Theming https://www.gradio.app/guides/theming-guide

        gr.Markdown("# 🐙 Chat with OctoTools: An Agentic Framework with Extensive Tools for Complex Reasoning")  # Title
        # gr.Markdown("[![OctoTools](https://img.shields.io/badge/OctoTools-Agentic%20Framework%20for%20Complex%20Reasoning-blue)](https://octotools.github.io/)")  # Title
        gr.Markdown("""
        **OctoTools** is a training-free, user-friendly, and easily extensible open-source agentic framework designed to tackle complex reasoning across diverse domains. 
        It introduces standardized **tool cards** to encapsulate tool functionality, a **planner** for both high-level and low-level planning, and an **executor** to carry out tool usage. 
                    
        [Website](https://octotools.github.io/) | 
        [Github](https://github.com/octotools/octotools) | 
        [arXiv](https://arxiv.org/abs/2502.11271) | 
        [Paper](https://arxiv.org/pdf/2502.11271) | 
        [Daily Paper](https://huggingface.co/papers/2502.11271) | 
        [Tool Cards](https://octotools.github.io/#tool-cards) | 
        [Example Visualizations](https://octotools.github.io/#visualization) | 
        [YouTube](https://www.youtube.com/watch?v=4828sGfx7dk&t=1176s&ab_channel=DiscoverAI) | 
        [Coverage](https://x.com/lupantech/status/1892260474320015861) | 
        [Discord](https://discord.gg/F4x9m7Cf)
        """)

        with gr.Row():
            # Left column for settings
            with gr.Column(scale=1):
                with gr.Row():
                    if args.openai_api_source == "user_provided":
                        print("Using API key from user input.")
                        api_key = gr.Textbox(
                            show_label=True,
                            placeholder="Your API key will not be stored in any way.",
                            type="password", 
                            label="OpenAI API Key",
                            # container=False
                        )
                    else:
                        print(f"Using local API key from environment variable: ...{os.getenv('OPENAI_API_KEY')[-4:]}")
                        api_key = gr.Textbox(
                            value=os.getenv("OPENAI_API_KEY"),
                            visible=False,
                            interactive=False
                        )

                with gr.Row():
                    llm_model_engine = gr.Dropdown(
                        choices=["gpt-4o", "gpt-4o-2024-11-20", "gpt-4o-2024-08-06", "gpt-4o-2024-05-13",
                                "gpt-4o-mini", "gpt-4o-mini-2024-07-18"], 
                        value="gpt-4o", 
                        label="LLM Model"
                    )
                with gr.Row():
                    max_steps = gr.Slider(value=8, minimum=1, maximum=10, step=1, label="Max Steps")
                
                with gr.Row():
                    max_time = gr.Slider(value=240, minimum=60, maximum=300, step=30, label="Max Time (seconds)")

                with gr.Row():
                    # Container for tools section
                    with gr.Column():

                        # First row for checkbox group
                        enabled_tools = gr.CheckboxGroup(
                            choices=all_tools,
                            value=all_tools,
                            label="Selected Tools",
                        )

                        # Second row for buttons
                        with gr.Row():
                            enable_all_btn = gr.Button("Select All Tools")
                            disable_all_btn = gr.Button("Clear All Tools")
                        
                        # Add click handlers for the buttons
                        enable_all_btn.click(
                            lambda: all_tools,
                            outputs=enabled_tools
                        )
                        disable_all_btn.click(
                            lambda: [],
                            outputs=enabled_tools
                        )

            with gr.Column(scale=5):
                
                with gr.Row():
                    # Middle column for the query
                    with gr.Column(scale=2):
                        user_image = gr.Image(type="pil", label="Upload an Image (Optional)", height=500)  # Accepts multiple formats
                        
                        with gr.Row():
                            user_query = gr.Textbox( placeholder="Type your question here...", label="Question (Required)")

                        with gr.Row():
                            run_button = gr.Button("🐙 Submit and Run", variant="primary")  # Run button with blue color

                    # Right column for the output
                    with gr.Column(scale=3):
                        chatbot_output = gr.Chatbot(type="messages", label="Step-wise Problem-Solving Output", height=500)

                        # TODO: Add actions to the buttons
                        with gr.Row(elem_id="buttons") as button_row:
                            upvote_btn = gr.Button(value="👍  Upvote", interactive=True, variant="primary") # TODO
                            downvote_btn = gr.Button(value="👎  Downvote", interactive=True, variant="primary") # TODO
                            # stop_btn = gr.Button(value="⛔️  Stop", interactive=True) # TODO
                            # clear_btn = gr.Button(value="🗑️  Clear history", interactive=True) # TODO

                        # TODO: Add comment textbox
                        with gr.Row():
                            comment_textbox = gr.Textbox(value="", 
                                                        placeholder="Feel free to add any comments here. Thanks for using OctoTools!",
                                                        label="💬 Comment (Type and press Enter to submit.)", interactive=True) # TODO
                            
                        # Update the button click handlers
                        upvote_btn.click(
                            fn=lambda: save_feedback(QUERY_ID, "upvote"),
                            inputs=[],
                            outputs=[]
                        )
                        
                        downvote_btn.click(
                            fn=lambda: save_feedback(QUERY_ID, "downvote"),
                            inputs=[],
                            outputs=[]
                        )

                        # Add handler for comment submission
                        comment_textbox.submit(
                            fn=lambda comment: save_feedback(QUERY_ID, "comment", comment),
                            inputs=[comment_textbox],
                            outputs=[]
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

        # Link button click to function
        run_button.click(
            fn=solve_problem_gradio, 
            inputs=[user_query, user_image, max_steps, max_time, api_key, llm_model_engine, enabled_tools], 
            outputs=chatbot_output
        )
    #################### Gradio Interface ####################

    # Launch the Gradio app
    # demo.launch(ssr_mode=False)
    demo.launch(ssr_mode=False, share=True)  # Added share=True parameter

if __name__ == "__main__":
    args = parse_arguments()

    # All tools
    all_tools = [
        "Generalist_Solution_Generator_Tool",

        "Image_Captioner_Tool",
        "Object_Detector_Tool",
        "Relevant_Patch_Zoomer_Tool",
        "Text_Detector_Tool",

        "Python_Code_Generator_Tool",

        "ArXiv_Paper_Searcher_Tool",
        "Google_Search_Tool",
        "Nature_News_Fetcher_Tool",
        "Pubmed_Search_Tool",
        "URL_Text_Extractor_Tool",
        "Wikipedia_Knowledge_Searcher_Tool"
    ]
    args.enabled_tools = ",".join(all_tools)

    # NOTE: Use the same name for the query cache directory as the dataset directory
    args.root_cache_dir = DATASET_DIR.name
    main(args)


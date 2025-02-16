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

from opentools.models.initializer import Initializer
from opentools.models.planner import Planner
from opentools.models.memory import Memory
from opentools.models.executor import Executor
from opentools.models.utils import make_json_serializable

# class ChatMessage:
#     def __init__(self, role: str, content: str, metadata: dict = None):
#         self.role = role
#         self.content = content
#         self.metadata = metadata or {}

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
        output_json_dir: str = "results",
        root_cache_dir: str = "cache"
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
        self.output_json_dir = output_json_dir
        self.root_cache_dir = root_cache_dir

        self.output_types = output_types.lower().split(',')
        assert all(output_type in ["base", "final", "direct"] for output_type in self.output_types), "Invalid output type. Supported types are 'base', 'final', 'direct'."

        # self.benchmark_data = self.load_benchmark_data()



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
            os.makedirs(os.path.join(self.root_cache_dir, 'images'), exist_ok=True)
            img_path = os.path.join(self.root_cache_dir, 'images', str(uuid.uuid4()) + '.jpg')
            user_image.save(img_path)
        else:
            img_path = None

        # Set query cache
        _cache_dir = os.path.join(self.root_cache_dir)
        self.executor.set_query_cache_dir(_cache_dir)
        
        # Step 1: Display the received inputs
        if user_image:
            messages.append(ChatMessage(role="assistant", content=f"📝 Received Query: {user_query}\n🖼️ Image Uploaded"))
        else:
            messages.append(ChatMessage(role="assistant", content=f"📝 Received Query: {user_query}"))
        yield messages

        # # Step 2: Add "thinking" status while processing
        # messages.append(ChatMessage(
        #     role="assistant",
        #     content="",
        #     metadata={"title": "⏳ Thinking: Processing input..."}
        # ))

        # Step 3: Initialize problem-solving state
        start_time = time.time()
        step_count = 0
        json_data = {"query": user_query, "image": "Image received as bytes"}

        # Step 4: Query Analysis
        query_analysis = self.planner.analyze_query(user_query, img_path)
        json_data["query_analysis"] = query_analysis
        messages.append(ChatMessage(role="assistant", 
                                    content=f"{query_analysis}", 
                                    metadata={"title": "🔍 Query Analysis"}))
        yield messages

        # Step 5: Execution loop (similar to your step-by-step solver)
        while step_count < self.max_steps and (time.time() - start_time) < self.max_time:
            step_count += 1
            # messages.append(ChatMessage(role="assistant", 
            #                             content=f"Generating next step...",
            #                             metadata={"title": f"🔄 Step {step_count}"}))
            yield messages

            # Generate the next step
            next_step = self.planner.generate_next_step(
                user_query, img_path, query_analysis, self.memory, step_count, self.max_steps
            )
            context, sub_goal, tool_name = self.planner.extract_context_subgoal_and_tool(next_step)

            # Display the step information
            messages.append(ChatMessage(
                role="assistant",
                content=f"- Context: {context}\n- Sub-goal: {sub_goal}\n- Tool: {tool_name}",
                metadata={"title": f"📌 Step {step_count}: {tool_name}"}
            ))
            yield messages

            # Handle tool execution or errors
            if tool_name not in self.planner.available_tools:
                messages.append(ChatMessage(
                    role="assistant", 
                    content=f"⚠️ Error: Tool '{tool_name}' is not available."))
                yield messages
                continue

            # Execute the tool command
            tool_command = self.executor.generate_tool_command(
                user_query, img_path, context, sub_goal, tool_name, self.planner.toolbox_metadata[tool_name]
            )
            explanation, command = self.executor.extract_explanation_and_command(tool_command)
            result = self.executor.execute_tool_command(tool_name, command)
            result = make_json_serializable(result)

            messages.append(ChatMessage(
                role="assistant", 
                content=f"{json.dumps(result, indent=4)}",
                metadata={"title": f"✅ Step {step_count} Result: {tool_name}"}))
            yield messages

            # Step 6: Memory update and stopping condition
            self.memory.add_action(step_count, tool_name, sub_goal, tool_command, result)
            stop_verification = self.planner.verificate_memory(user_query, img_path, query_analysis, self.memory)
            conclusion = self.planner.extract_conclusion(stop_verification)

            messages.append(ChatMessage(
                role="assistant", 
                content=f"🛑 Step {step_count} Conclusion: {conclusion}"))
            yield messages

            if conclusion == 'STOP':
                break

        # Step 7: Generate Final Output (if needed)
        if 'final' in self.output_types:
            final_output = self.planner.generate_final_output(user_query, img_path, self.memory)
            messages.append(ChatMessage(role="assistant", content=f"🎯 Final Output:\n{final_output}"))
            yield messages

        if 'direct' in self.output_types:
            direct_output = self.planner.generate_direct_output(user_query, img_path, self.memory)
            messages.append(ChatMessage(role="assistant", content=f"🔹 Direct Output:\n{direct_output}"))
            yield messages

        # Step 8: Completion Message
        messages.append(ChatMessage(role="assistant", content="✅ Problem-solving process complete."))
        yield messages
            
def parse_arguments():
    parser = argparse.ArgumentParser(description="Run the OpenTools demo with specified parameters.")
    parser.add_argument("--llm_engine_name", default="gpt-4o", help="LLM engine name.")
    parser.add_argument("--max_tokens", type=int, default=2000, help="Maximum tokens for LLM generation.")
    parser.add_argument("--run_baseline_only", type=bool, default=False, help="Run only the baseline (no toolbox).")
    parser.add_argument("--task", default="minitoolbench", help="Task to run.")
    parser.add_argument("--task_description", default="", help="Task description.")
    parser.add_argument(
        "--output_types",
        default="base,final,direct",
        help="Comma-separated list of required outputs (base,final,direct)"
    )
    parser.add_argument("--enabled_tools", default="Generalist_Solution_Generator_Tool", help="List of enabled tools.")
    parser.add_argument("--root_cache_dir", default="demo_solver_cache", help="Path to solver cache directory.")
    parser.add_argument("--output_json_dir", default="demo_results", help="Path to output JSON directory.")
    parser.add_argument("--verbose", type=bool, default=True, help="Enable verbose output.")
    return parser.parse_args()


def solve_problem_gradio(user_query, user_image, max_steps=10, max_time=60, api_key=None):
    """
    Wrapper function to connect the solver to Gradio.
    Streams responses from `solver.stream_solve_user_problem` for real-time UI updates.
    """

    if api_key is None:
        return [["assistant", "⚠️ Error: API Key is required."]]
    
    # Initialize Tools
    enabled_tools = args.enabled_tools.split(",") if args.enabled_tools else []

    # Instantiate Initializer
    initializer = Initializer(
        enabled_tools=enabled_tools,
        model_string=args.llm_engine_name,
        api_key=api_key
    )

    # Instantiate Planner
    planner = Planner(
        llm_engine_name=args.llm_engine_name,
        toolbox_metadata=initializer.toolbox_metadata,
        available_tools=initializer.available_tools,
        api_key=api_key
    )

    # Instantiate Memory
    memory = Memory()

    # Instantiate Executor
    executor = Executor(
        llm_engine_name=args.llm_engine_name,
        root_cache_dir=args.root_cache_dir,
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
        output_json_dir=args.output_json_dir,
        root_cache_dir=args.root_cache_dir
    )

    if solver is None:
        return [["assistant", "⚠️ Error: Solver is not initialized. Please restart the application."]]

    messages = []  # Initialize message list
    for message_batch in solver.stream_solve_user_problem(user_query, user_image, api_key, messages):
        yield [msg for msg in message_batch]  # Ensure correct format for Gradio Chatbot



def main(args):
    # ========== Gradio Interface ==========
    with gr.Blocks() as demo:
        gr.Markdown("# 🧠 OctoTools AI Solver")  # Title

        with gr.Row():
            with gr.Column(scale=1):
                api_key = gr.Textbox(show_label=False, placeholder="Your API key will not be stored in any way.", type="password", container=False)
                user_image = gr.Image(type="pil", label="Upload an image")  # Accepts multiple formats
                max_steps = gr.Slider(value=5, minimum=1, maximum=10, step=1)
                max_time = gr.Slider(value=180, minimum=60, maximum=300, step=20)
            with gr.Column(scale=3):
                chatbot_output = gr.Chatbot(type="messages", label="Problem-Solving Output")
                # chatbot_output.like(lambda x: print(f"User liked: {x}"))
                with gr.Row():
                    with gr.Column(scale=8):
                        user_query = gr.Textbox(show_label=False, placeholder="Type your question here...", container=False)
                    with gr.Column(scale=1):
                        run_button = gr.Button("Run")  # Run button
                with gr.Row(elem_id="buttons") as button_row:
                    upvote_btn = gr.Button(value="👍  Upvote", interactive=False)
                    downvote_btn = gr.Button(value="👎  Downvote", interactive=False)
                    clear_btn = gr.Button(value="🗑️  Clear history", interactive=False)

        # Link button click to function
        run_button.click(fn=solve_problem_gradio, inputs=[user_query, user_image, max_steps, max_time, api_key], outputs=chatbot_output)

    # Launch the Gradio app
    demo.launch()



if __name__ == "__main__":
    args = parse_arguments()
    main(args)
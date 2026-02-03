"""
Basic usage example for SHAPE framework.

This example demonstrates how to use SHAPE programmatically
for morphological analysis of cell images.
"""

import os
from shape.agent.planner import Planner
from shape.agent.executor import Executor
from shape.agent.memory import Memory

# TODO: Import tool discovery function
# from shape.tools import get_available_tools


def analyze_cell_states(image_path: str, query: str, api_key: str):
    """
    Analyze cell states in an image using SHAPE.
    
    Args:
        image_path: Path to the image file
        query: Natural language query (e.g., "What cell states are present?")
        api_key: OpenAI API key
    
    Returns:
        Analysis results
    """
    # Initialize agent components
    planner = Planner(
        llm_engine_name="gpt-4o",
        available_tools=get_available_tools(),  # TODO: Implement
        api_key=api_key
    )
    
    executor = Executor(
        llm_engine_name="gpt-4o",
        query_cache_dir="output/cache",
        api_key=api_key
    )
    
    memory = Memory()
    memory.set_query(query)
    
    # Analyze query
    query_analysis = planner.analyze_query(
        question=query,
        image=image_path
    )
    
    # Execute analysis steps
    max_steps = 10
    for step in range(1, max_steps + 1):
        # Generate next step
        next_step = planner.generate_next_step(
            question=query,
            image=image_path,
            query_analysis=query_analysis,
            memory=memory,
            step_count=step,
            max_step_count=max_steps
        )
        
        # Verify if we can stop
        verification = planner.verificate_memory(
            question=query,
            image=image_path,
            query_analysis=query_analysis,
            memory=memory
        )
        
        if verification.stop_signal:
            break
        
        # Execute tool
        tool_command = executor.generate_tool_command(
            question=query,
            image=image_path,
            context=next_step.context,
            sub_goal=next_step.sub_goal,
            tool_name=next_step.tool_name,
            tool_metadata={},  # TODO: Get from registry
            memory=memory
        )
        
        result = executor.execute_tool_command(
            tool_name=next_step.tool_name,
            command=tool_command.command
        )
        
        # Update memory
        memory.add_action(
            step_count=step,
            tool_name=next_step.tool_name,
            sub_goal=next_step.sub_goal,
            command=tool_command.command,
            result=result
        )
    
    # Generate final answer
    final_answer = planner.generate_final_output(
        question=query,
        image=image_path,
        memory=memory
    )
    
    return final_answer


if __name__ == "__main__":
    # Example usage
    image_path = "examples/example_image.tif"
    query = "What cell states are present in this image?"
    api_key = os.getenv("OPENAI_API_KEY")
    
    if not api_key:
        raise ValueError("Please set OPENAI_API_KEY environment variable")
    
    result = analyze_cell_states(image_path, query, api_key)
    print(result)


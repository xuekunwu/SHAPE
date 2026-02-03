"""
Basic usage example for SHAPE framework.

This example demonstrates how to use SHAPE programmatically
for morphological analysis of cell images.
"""

import os
import sys

# Add parent directory to path to import shape
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from shape import solve, construct_solver


def analyze_cell_states(image_path: str, query: str, api_key: str):
    """
    Analyze cell states in an image using SHAPE.
    
    Args:
        image_path: Path to the image file
        query: Natural language query (e.g., "What cell states are present?")
        api_key: OpenAI API key
    
    Returns:
        Analysis results dictionary
    """
    # Use the high-level solve function
    result = solve(
        question=query,
        image_path=image_path,
        llm_engine_name="gpt-4o",
        api_key=api_key,
        max_steps=10
    )
    
    return result


def analyze_with_custom_setup(image_path: str, query: str, api_key: str):
    """
    Example using construct_solver for more control.
    
    Args:
        image_path: Path to the image file
        query: Natural language query
        api_key: OpenAI API key
    
    Returns:
        Analysis results
    """
    # Construct solver components
    solver = construct_solver(
        llm_engine_name="gpt-4o",
        api_key=api_key
    )
    
    planner = solver["planner"]
    executor = solver["executor"]
    memory = solver["memory"]
    
    # Set query
    memory.set_query(query)
    
    # Analyze query
    query_analysis = planner.analyze_query(
        question=query,
        image=image_path
    )
    
    # Execute steps (simplified - see solver.py for full implementation)
    max_steps = 10
    for step in range(1, max_steps + 1):
        next_step = planner.generate_next_step(
            question=query,
            image=image_path,
            query_analysis=query_analysis,
            memory=memory,
            step_count=step,
            max_step_count=max_steps
        )
        
        verification = planner.verificate_memory(
            question=query,
            image=image_path,
            query_analysis=query_analysis,
            memory=memory
        )
        
        if verification.stop_signal == "STOP":
            break
        
        # Get tool metadata
        tool_metadata = solver["initializer"].get_toolbox_metadata().get(
            next_step.tool_name, {}
        )
        
        # Generate and execute tool command
        tool_command = executor.generate_tool_command(
            question=query,
            image=image_path,
            context=next_step.context,
            sub_goal=next_step.sub_goal,
            tool_name=next_step.tool_name,
            tool_metadata=tool_metadata,
            memory=memory
        )
        
        result = executor.execute_tool_command(
            tool_name=next_step.tool_name,
            command=tool_command.command
        )
        
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
    
    return {
        "direct_output": final_answer.direct_output,
        "memory": memory
    }


if __name__ == "__main__":
    # Example usage
    import argparse
    
    parser = argparse.ArgumentParser(description="SHAPE basic usage example")
    parser.add_argument(
        "--image",
        type=str,
        default="examples/iPSC-cardiomyocyte.tif",
        help="Path to image file"
    )
    parser.add_argument(
        "--query",
        type=str,
        default="What cell states are present in this image?",
        help="Query to analyze"
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=None,
        help="OpenAI API key (or set OPENAI_API_KEY env var)"
    )
    
    args = parser.parse_args()
    
    # Get API key
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise ValueError(
            "Please provide API key via --api-key argument "
            "or set OPENAI_API_KEY environment variable"
        )
    
    # Check if image exists
    if not os.path.exists(args.image):
        print(f"Warning: Image file not found: {args.image}")
        print("Using query-only mode (no image)")
        args.image = None
    
    # Run analysis
    print(f"Query: {args.query}")
    if args.image:
        print(f"Image: {args.image}")
    
    result = analyze_cell_states(args.image, args.query, api_key)
    
    print("\n" + "="*50)
    print("Analysis Result:")
    print("="*50)
    print(result["direct_output"])


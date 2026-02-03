"""
SHAPE Solver - Main entry point for the framework.

This module provides a high-level interface for using SHAPE programmatically.
"""

import os
from typing import List, Optional
from shape.models.initializer import Initializer
from shape.models.planner import Planner
from shape.models.executor import Executor
from shape.models.memory import Memory


def get_available_tools(enabled_tools: Optional[List[str]] = None) -> List[str]:
    """
    Get list of available tools.
    
    Args:
        enabled_tools: Optional list of tool names to enable. If None, all tools are discovered.
    
    Returns:
        List of available tool names
    """
    initializer = Initializer(enabled_tools=enabled_tools or [])
    return initializer.available_tools


def construct_solver(
    llm_engine_name: str = "gpt-4o",
    enabled_tools: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    query_cache_dir: str = "solver_cache"
):
    """
    Construct a SHAPE solver with all necessary components.
    
    Args:
        llm_engine_name: Name of the LLM engine to use (e.g., "gpt-4o", "gpt-4")
        enabled_tools: Optional list of tool names to enable. If None, all tools are enabled.
        api_key: OpenAI API key. If None, will try to get from OPENAI_API_KEY environment variable.
        query_cache_dir: Directory for caching query results
    
    Returns:
        Dictionary containing solver components:
        - planner: Planner instance
        - executor: Executor instance
        - memory: Memory instance
        - initializer: Initializer instance
        - available_tools: List of available tool names
    """
    # Get API key from environment if not provided
    if api_key is None:
        api_key = os.getenv("OPENAI_API_KEY")
        if api_key is None:
            raise ValueError(
                "API key not provided. Set OPENAI_API_KEY environment variable "
                "or pass api_key parameter."
            )
    
    # Initialize tools
    initializer = Initializer(
        enabled_tools=enabled_tools or [],
        model_string=llm_engine_name,
        api_key=api_key
    )
    
    # Get toolbox metadata
    toolbox_metadata = initializer.get_toolbox_metadata()
    
    # Initialize planner
    planner = Planner(
        llm_engine_name=llm_engine_name,
        toolbox_metadata=toolbox_metadata,
        available_tools=initializer.available_tools,
        api_key=api_key
    )
    
    # Initialize executor
    executor = Executor(
        llm_engine_name=llm_engine_name,
        query_cache_dir=query_cache_dir,
        api_key=api_key,
        initializer=initializer
    )
    
    # Initialize memory
    memory = Memory()
    
    return {
        "planner": planner,
        "executor": executor,
        "memory": memory,
        "initializer": initializer,
        "available_tools": initializer.available_tools
    }


def solve(
    question: str,
    image_path: Optional[str] = None,
    llm_engine_name: str = "gpt-4o",
    enabled_tools: Optional[List[str]] = None,
    api_key: Optional[str] = None,
    max_steps: int = 10,
    query_cache_dir: str = "solver_cache"
):
    """
    Solve a problem using SHAPE framework.
    
    Args:
        question: Natural language question to answer
        image_path: Optional path to image file
        llm_engine_name: Name of the LLM engine to use
        enabled_tools: Optional list of tool names to enable
        api_key: OpenAI API key
        max_steps: Maximum number of steps to execute
        query_cache_dir: Directory for caching query results
    
    Returns:
        Dictionary containing:
        - direct_output: Final answer text
        - memory: Memory object with execution history
        - steps: List of executed steps
    """
    # Construct solver
    solver = construct_solver(
        llm_engine_name=llm_engine_name,
        enabled_tools=enabled_tools,
        api_key=api_key,
        query_cache_dir=query_cache_dir
    )
    
    planner = solver["planner"]
    executor = solver["executor"]
    memory = solver["memory"]
    
    # Set query in memory
    memory.set_query(question)
    
    # Analyze query
    query_analysis = planner.analyze_query(
        question=question,
        image=image_path
    )
    
    # Execute steps
    for step in range(1, max_steps + 1):
        # Generate next step
        next_step = planner.generate_next_step(
            question=question,
            image=image_path,
            query_analysis=query_analysis,
            memory=memory,
            step_count=step,
            max_step_count=max_steps
        )
        
        # Verify if we can stop
        verification = planner.verificate_memory(
            question=question,
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
        
        # Generate tool command
        tool_command = executor.generate_tool_command(
            question=question,
            image=image_path,
            context=next_step.context,
            sub_goal=next_step.sub_goal,
            tool_name=next_step.tool_name,
            tool_metadata=tool_metadata,
            memory=memory
        )
        
        # Execute tool
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
        question=question,
        image=image_path,
        memory=memory
    )
    
    return {
        "direct_output": final_answer.direct_output,
        "memory": memory,
        "steps": memory.get_actions()
    }



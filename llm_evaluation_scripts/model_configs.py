"""
Model Configuration Templates for LLM Evaluation
Defines configurations for different LLM models to be evaluated
"""

import os
from typing import Dict, Any

# Base system prompt for fibroblast analysis
FIBROBLAST_SYSTEM_PROMPT = """You are an expert AI assistant specialized in fibroblast biology and single-cell image analysis. Your expertise includes:

1. **Cell Biology Knowledge**: Deep understanding of fibroblast activation states, cell morphology, and biological processes
2. **Image Analysis**: Expertise in analyzing microscopy images, cell segmentation, and feature extraction
3. **Scientific Communication**: Ability to explain complex biological concepts clearly and accurately
4. **Tool Integration**: Skilled at using specialized tools for image preprocessing, cell detection, and state classification

Your primary role is to:
- Analyze fibroblast cell images at single-cell resolution
- Classify cells into appropriate activation states (Dead, np-MyoFb, p-MyoFb, proto-MyoFb, q-Fb)
- Provide detailed explanations of your analysis and reasoning
- Generate appropriate visualizations and statistical summaries
- Integrate findings with relevant scientific literature when requested

Always maintain scientific accuracy and provide clear, evidence-based interpretations."""

# Model configurations for evaluation
MODEL_CONFIGS = {
    "gpt-4o": {
        "model_string": "gpt-4o",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.005,
        "capabilities": ["multimodal", "structured_output", "tool_calling"],
        "api_key": os.getenv("OPENAI_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    },
    
    "gpt-4o-mini": {
        "model_string": "gpt-4o-mini",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.00015,
        "capabilities": ["multimodal", "structured_output", "tool_calling"],
        "api_key": os.getenv("OPENAI_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    },
    
    "gpt-4o-mini-2024-07-18": {
        "model_string": "gpt-4o-mini-2024-07-18",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.00015,
        "capabilities": ["multimodal", "structured_output", "tool_calling"],
        "api_key": os.getenv("OPENAI_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    },
    
    "gpt-4o-2024-08-06": {
        "model_string": "gpt-4o-2024-08-06",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.005,
        "capabilities": ["multimodal", "structured_output", "tool_calling"],
        "api_key": os.getenv("OPENAI_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    },
    
    "claude-3-5-sonnet": {
        "model_string": "claude-3-5-sonnet-20241022",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.003,
        "capabilities": ["multimodal", "tool_calling"],
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    },
    
    "claude-3-5-haiku": {
        "model_string": "claude-3-5-haiku-20241022",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.00025,
        "capabilities": ["multimodal", "tool_calling"],
        "api_key": os.getenv("ANTHROPIC_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    },
    
    "gemini-1.5-pro": {
        "model_string": "gemini-1.5-pro",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.0025,
        "capabilities": ["multimodal", "tool_calling"],
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    },
    
    "gemini-1.5-flash": {
        "model_string": "gemini-1.5-flash",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.000075,
        "capabilities": ["multimodal", "tool_calling"],
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    },
    
    "deepseek": {
        "model_string": "deepseek",
        "temperature": 0.3,
        "max_tokens": 4000,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "expected_cost_per_1k_tokens": 0.001,
        "capabilities": ["multimodal", "structured_output", "tool_calling"],
        "api_key": os.getenv("DEEPSEEK_API_KEY"),
        "is_multimodal": True,
        "enable_cache": False
    }
}

# Model categories for different evaluation scenarios
MODEL_CATEGORIES = {
    "premium": ["gpt-4o", "claude-3-5-sonnet", "gemini-1.5-pro"],
    "standard": ["gpt-4o-mini", "claude-3-5-haiku", "gemini-1.5-flash"],
    "budget": ["gpt-4o-mini-2024-07-18", "deepseek"],
    "multimodal": ["gpt-4o", "gpt-4o-mini", "claude-3-5-sonnet", "gemini-1.5-pro"],
    "fast": ["gpt-4o-mini", "claude-3-5-haiku", "gemini-1.5-flash"]
}

# Performance expectations for different model tiers
PERFORMANCE_EXPECTATIONS = {
    "premium": {
        "min_success_rate": 0.85,
        "min_output_quality": 8.0,
        "max_execution_time": 30.0,
        "max_cost_per_task": 0.10
    },
    "standard": {
        "min_success_rate": 0.75,
        "min_output_quality": 7.0,
        "max_execution_time": 45.0,
        "max_cost_per_task": 0.05
    },
    "budget": {
        "min_success_rate": 0.65,
        "min_output_quality": 6.0,
        "max_execution_time": 60.0,
        "max_cost_per_task": 0.02
    }
}

def get_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
        
    Raises:
        ValueError: If model is not found in configurations
    """
    if model_name not in MODEL_CONFIGS:
        available_models = list(MODEL_CONFIGS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return MODEL_CONFIGS[model_name].copy()

def get_models_by_category(category: str) -> list:
    """
    Get list of models in a specific category
    
    Args:
        category: Category name (premium, standard, budget, multimodal, fast)
        
    Returns:
        List of model names in the category
        
    Raises:
        ValueError: If category is not found
    """
    if category not in MODEL_CATEGORIES:
        available_categories = list(MODEL_CATEGORIES.keys())
        raise ValueError(f"Category '{category}' not found. Available categories: {available_categories}")
    
    return MODEL_CATEGORIES[category]

def validate_model_config(model_config: Dict[str, Any]) -> bool:
    """
    Validate model configuration
    
    Args:
        model_config: Model configuration dictionary
        
    Returns:
        True if configuration is valid
        
    Raises:
        ValueError: If configuration is invalid
    """
    required_fields = ["model_string", "temperature", "max_tokens", "system_prompt"]
    
    for field in required_fields:
        if field not in model_config:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate temperature range
    if not 0 <= model_config["temperature"] <= 2:
        raise ValueError("Temperature must be between 0 and 2")
    
    # Validate max_tokens
    if model_config["max_tokens"] <= 0:
        raise ValueError("max_tokens must be positive")
    
    # Validate API key presence
    if not model_config.get("api_key"):
        raise ValueError("API key is required")
    
    return True

def get_cost_estimate(model_name: str, estimated_tokens: int) -> float:
    """
    Get cost estimate for a model based on token usage
    
    Args:
        model_name: Name of the model
        estimated_tokens: Estimated number of tokens
        
    Returns:
        Estimated cost in USD
    """
    config = get_model_config(model_name)
    cost_per_1k = config["expected_cost_per_1k_tokens"]
    return (estimated_tokens / 1000) * cost_per_1k

def get_performance_targets(model_name: str) -> Dict[str, float]:
    """
    Get performance targets for a model based on its category
    
    Args:
        model_name: Name of the model
        
    Returns:
        Dictionary of performance targets
    """
    # Determine category
    category = None
    for cat, models in MODEL_CATEGORIES.items():
        if model_name in models:
            category = cat
            break
    
    if category and category in PERFORMANCE_EXPECTATIONS:
        return PERFORMANCE_EXPECTATIONS[category]
    else:
        # Default targets
        return {
            "min_success_rate": 0.70,
            "min_output_quality": 6.5,
            "max_execution_time": 60.0,
            "max_cost_per_task": 0.05
        } 
"""
Model configurations for LLM engines.
Simplified version for production use - only includes OpenAI models.
"""

# OpenAI model configurations
HF_MODEL_CONFIGS = {
    "gpt-4o": {
        "model_id": "gpt-4o",
        "model_type": "openai",
        "multimodal": True,
        "input_cost_per_1k_tokens": 0.0025,
        "output_cost_per_1k_tokens": 0.01,
    },
    "gpt-4o-mini": {
        "model_id": "gpt-4o-mini",
        "model_type": "openai",
        "multimodal": True,
        "input_cost_per_1k_tokens": 0.00015,
        "output_cost_per_1k_tokens": 0.0006,
    },
    "gpt-4-turbo": {
        "model_id": "gpt-4-turbo",
        "model_type": "openai",
        "multimodal": True,
        "input_cost_per_1k_tokens": 0.01,
        "output_cost_per_1k_tokens": 0.03,
    },
    "gpt-4": {
        "model_id": "gpt-4",
        "model_type": "openai",
        "multimodal": False,
        "input_cost_per_1k_tokens": 0.03,
        "output_cost_per_1k_tokens": 0.06,
    },
    "gpt-3.5-turbo": {
        "model_id": "gpt-3.5-turbo",
        "model_type": "openai",
        "multimodal": False,
        "input_cost_per_1k_tokens": 0.0005,
        "output_cost_per_1k_tokens": 0.0015,
    },
    # Placeholder for future models
    "gpt-5-mini": {
        "model_id": "gpt-5-mini",
        "model_type": "openai",
        "multimodal": True,
        "input_cost_per_1k_tokens": 0.0001,
        "output_cost_per_1k_tokens": 0.0004,
    },
}

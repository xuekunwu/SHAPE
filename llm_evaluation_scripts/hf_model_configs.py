"""
Hugging Face Model Configurations for LLM Evaluation
Supports various open-source LLMs available on Hugging Face
"""

import os
import base64
from typing import Dict, Any, List, Optional
import torch
from transformers import AutoProcessor, AutoModelForCausalLM
from PIL import Image
import openai
import logging

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

# Hugging Face Model Configurations
HF_MODEL_CONFIGS = {
    "gpt-4o": {
        "model_id": "gpt-4o",
        "model_type": "openai",
        "hardware_requirements": "OpenAI API",
        "is_multimodal": True
    },
    "gpt-4o-mini": {
        "model_id": "gpt-4o-mini",
        "model_type": "openai",
        "hardware_requirements": "OpenAI API",
        "is_multimodal": True
    },
    "gpt-4-turbo": {
        "model_id": "gpt-4-turbo",
        "model_type": "openai",
        "hardware_requirements": "OpenAI API",
        "is_multimodal": True
    },
    "gpt-3.5-turbo": {
        "model_id": "gpt-3.5-turbo",
        "model_type": "openai",
        "hardware_requirements": "OpenAI API",
        "is_multimodal": False
    },
    # 流行开源Hugging Face模型
    "phi-3-mini": {
        "model_id": "microsoft/phi-3-mini-4k-instruct",
        "model_type": "hf",
        "hardware_requirements": "8GB VRAM (4bit)",
        "is_multimodal": False
    },
    "llama-3-8b": {
        "model_id": "meta-llama/Meta-Llama-3-8B-Instruct",
        "model_type": "hf",
        "hardware_requirements": "12GB VRAM (4bit)",
        "is_multimodal": False
    },
    "gemma-2-9b": {
        "model_id": "google/gemma-2b-it",
        "model_type": "hf",
        "hardware_requirements": "8GB VRAM (4bit)",
        "is_multimodal": False
    },
    "mixtral-8x7b": {
        "model_id": "mistralai/Mixtral-8x7B-Instruct-v0.1",
        "model_type": "hf",
        "hardware_requirements": "24GB VRAM (4bit)",
        "is_multimodal": False
    },
    "qwen2-72b": {
        "model_id": "Qwen/Qwen2-72B-Instruct",
        "model_type": "hf",
        "hardware_requirements": "48GB VRAM (4bit)",
        "is_multimodal": False
    },
    "llava-1.5-7b": {
        "model_id": "llava-hf/llava-1.5-7b-hf",
        "model_type": "hf",
        "hardware_requirements": "24GB VRAM (4bit/8bit)",
        "is_multimodal": True
    },
    "llava-1.5-13b": {
        "model_id": "llava-hf/llava-1.5-13b-hf",
        "model_type": "hf",
        "hardware_requirements": "24GB VRAM (4bit, 单进程)",
        "is_multimodal": True
    },
    "qwen-vl": {
        "model_id": "Qwen/Qwen-VL",
        "model_type": "hf",
        "hardware_requirements": "24GB VRAM (4bit/8bit)",
        "is_multimodal": True
    },
    "qwen-vl-plus": {
        "model_id": "Qwen/Qwen-VL-Plus",
        "model_type": "hf",
        "hardware_requirements": "24GB VRAM (4bit, 单进程)",
        "is_multimodal": True
    },
    "phi-3-vision": {
        "model_id": "microsoft/phi-3-vision-128k-instruct",
        "model_type": "hf",
        "hardware_requirements": "16GB VRAM (4bit/8bit)",
        "is_multimodal": True
    },
    # Llama 3 Models
    "llama-3-70b": {
        "model_id": "meta-llama/Meta-Llama-3-70B-Instruct",
        "model_type": "llama",
        "max_length": 4096,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "instruction_following"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "40GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    
    # Mistral Models
    "mistral-7b-instruct": {
        "model_id": "mistralai/Mistral-7B-Instruct-v0.2",
        "model_type": "mistral",
        "max_length": 32768,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "instruction_following"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "16GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    
    # Qwen Models
    "qwen2-7b-instruct": {
        "model_id": "Qwen/Qwen2-7B-Instruct",
        "model_type": "qwen2",
        "max_length": 32768,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "instruction_following"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "16GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    
    # Phi Models
    "phi-3-medium": {
        "model_id": "microsoft/Phi-3-medium-4k-instruct",
        "model_type": "phi",
        "max_length": 4096,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "instruction_following"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "16GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    
    # Gemma Models
    "gemma-2-27b-it": {
        "model_id": "google/gemma-2-27b-it",
        "model_type": "gemma",
        "max_length": 8192,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "instruction_following"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "48GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    
    # Chinese Models
    "qwen1.5-7b-chat": {
        "model_id": "Qwen/Qwen1.5-7B-Chat",
        "model_type": "qwen",
        "max_length": 32768,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "instruction_following"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "16GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    
    "chatglm3-6b": {
        "model_id": "THUDM/chatglm3-6b",
        "model_type": "chatglm",
        "max_length": 8192,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "instruction_following"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "16GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    
    # Multimodal Models
    "llava-v1.6-7b": {
        "model_id": "llava-hf/llava-1.6-7b-hf",
        "model_type": "llava",
        "max_length": 4096,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "multimodal", "image_analysis"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "16GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": True
    },
    
    "llava-v1.6-13b": {
        "model_id": "llava-hf/llava-1.6-13b-hf",
        "model_type": "llava",
        "max_length": 4096,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "multimodal", "image_analysis"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "24GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": True
    },
    
    # Specialized Models
    "codellama-7b-instruct": {
        "model_id": "codellama/CodeLlama-7b-Instruct-hf",
        "model_type": "llama",
        "max_length": 4096,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "code_generation"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "16GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    
    "deepseek-coder-6.7b": {
        "model_id": "deepseek-ai/deepseek-coder-6.7b-instruct",
        "model_type": "deepseek",
        "max_length": 16384,
        "temperature": 0.3,
        "system_prompt": FIBROBLAST_SYSTEM_PROMPT,
        "capabilities": ["text_generation", "code_generation"],
        "expected_cost_per_1k_tokens": 0.0,
        "hardware_requirements": "16GB+ VRAM",
        "quantization": ["4bit", "8bit"],
        "is_multimodal": False
    },
    "qwen-vl-plus": {
        "model_id": "Qwen/Qwen-VL-Plus",
        "model_type": "hf",
        "hardware_requirements": "32GB+ VRAM (4bit)",
        "is_multimodal": True
    }
}

# Model categories for different evaluation scenarios
HF_MODEL_CATEGORIES = {
    "lightweight": ["phi-3-mini", "llama-3-8b", "mistral-7b-instruct", "qwen2-7b-instruct"],
    "medium": ["mixtral-8x7b-instruct", "qwen2-72b-instruct", "gemma-2-9b-it", "llava-v1.6-7b"],
    "heavyweight": ["llama-3-70b", "gemma-2-27b-it", "llava-v1.6-13b", "qwen-vl-plus"],
    "multimodal": ["llava-v1.6-7b", "llava-v1.6-13b", "qwen-vl-plus"],
    "code_specialized": ["codellama-7b-instruct", "deepseek-coder-6.7b"],
    "chinese": ["qwen1.5-7b-chat", "chatglm3-6b"]
}

# Performance expectations for different model tiers
HF_PERFORMANCE_EXPECTATIONS = {
    "lightweight": {
        "min_success_rate": 0.60,
        "min_output_quality": 5.5,
        "max_execution_time": 120.0,
        "max_memory_usage": "16GB"
    },
    "medium": {
        "min_success_rate": 0.70,
        "min_output_quality": 6.5,
        "max_execution_time": 90.0,
        "max_memory_usage": "24GB"
    },
    "heavyweight": {
        "min_success_rate": 0.80,
        "min_output_quality": 7.5,
        "max_execution_time": 60.0,
        "max_memory_usage": "48GB"
    }
}

def get_hf_model_config(model_name: str) -> Dict[str, Any]:
    """
    Get configuration for a specific Hugging Face model
    
    Args:
        model_name: Name of the model
        
    Returns:
        Model configuration dictionary
        
    Raises:
        ValueError: If model is not found in configurations
    """
    if model_name not in HF_MODEL_CONFIGS:
        available_models = list(HF_MODEL_CONFIGS.keys())
        raise ValueError(f"Model '{model_name}' not found. Available models: {available_models}")
    
    return HF_MODEL_CONFIGS[model_name].copy()

def get_hf_models_by_category(category: str) -> List[str]:
    """
    Get list of models in a specific category
    
    Args:
        category: Category name (lightweight, medium, heavyweight, multimodal, etc.)
        
    Returns:
        List of model names in the category
        
    Raises:
        ValueError: If category is not found
    """
    if category not in HF_MODEL_CATEGORIES:
        available_categories = list(HF_MODEL_CATEGORIES.keys())
        raise ValueError(f"Category '{category}' not found. Available categories: {available_categories}")
    
    return HF_MODEL_CATEGORIES[category]

def get_hf_models_by_hardware(vram_gb: int) -> List[str]:
    """
    Get models that can run on hardware with specified VRAM
    
    Args:
        vram_gb: Available VRAM in GB
        
    Returns:
        List of compatible model names
    """
    compatible_models = []
    
    for model_name, config in HF_MODEL_CONFIGS.items():
        # Extract VRAM requirement from hardware_requirements
        req_str = config["hardware_requirements"]
        try:
            req_vram = int(req_str.split("GB")[0])
            if vram_gb >= req_vram:
                compatible_models.append(model_name)
        except (ValueError, IndexError):
            # If parsing fails, assume it's compatible
            compatible_models.append(model_name)
    
    return compatible_models

def get_quantization_options(model_name: str) -> List[str]:
    """
    Get available quantization options for a model
    
    Args:
        model_name: Name of the model
        
    Returns:
        List of quantization options
    """
    config = get_hf_model_config(model_name)
    return config.get("quantization", ["4bit", "8bit"])

def estimate_memory_usage(model_name: str, quantization: str = "4bit") -> Dict[str, float]:
    """
    Estimate memory usage for a model with specific quantization
    
    Args:
        model_name: Name of the model
        quantization: Quantization level (4bit, 8bit, 16bit)
        
    Returns:
        Dictionary with memory estimates in GB
    """
    config = get_hf_model_config(model_name)
    
    # Base memory estimates (rough approximations)
    base_memory = {
        "llama-3-8b": 16,
        "llama-3-70b": 140,
        "mistral-7b-instruct": 14,
        "mixtral-8x7b-instruct": 24,
        "qwen2-7b-instruct": 14,
        "qwen2-72b-instruct": 144,
        "phi-3-mini": 8,
        "phi-3-medium": 16,
        "gemma-2-9b-it": 18,
        "gemma-2-27b-it": 54,
        "qwen1.5-7b-chat": 14,
        "chatglm3-6b": 12,
        "llava-v1.6-7b": 16,
        "llava-v1.6-13b": 26,
        "qwen-vl-plus": 24,
        "codellama-7b-instruct": 14,
        "deepseek-coder-6.7b": 14
    }
    
    base_gb = base_memory.get(model_name, 16)
    
    # Apply quantization reduction
    quantization_factor = {
        "4bit": 0.25,
        "8bit": 0.5,
        "16bit": 1.0
    }
    
    model_memory = base_gb * quantization_factor.get(quantization, 0.25)
    
    return {
        "model_memory_gb": model_memory,
        "total_memory_gb": model_memory + 2,  # Add buffer for activations
        "recommended_vram_gb": model_memory + 4  # Add safety margin
    }

def get_optimal_models_for_hardware(vram_gb: int, include_multimodal: bool = True) -> Dict[str, List[str]]:
    """
    Get optimal model recommendations for given hardware
    
    Args:
        vram_gb: Available VRAM in GB
        include_multimodal: Whether to include multimodal models
        
    Returns:
        Dictionary with model recommendations by category
    """
    recommendations = {
        "best_performance": [],
        "best_efficiency": [],
        "multimodal": [],
        "code_specialized": []
    }
    
    compatible_models = get_hf_models_by_hardware(vram_gb)
    
    # Best performance (largest models that fit)
    if vram_gb >= 48:
        recommendations["best_performance"] = ["llama-3-70b", "qwen2-72b-instruct", "gemma-2-27b-it"]
    elif vram_gb >= 24:
        recommendations["best_performance"] = ["mixtral-8x7b-instruct", "llava-v1.6-13b", "qwen-vl-plus"]
    elif vram_gb >= 16:
        recommendations["best_performance"] = ["llama-3-8b", "mistral-7b-instruct", "qwen2-7b-instruct"]
    else:
        recommendations["best_performance"] = ["phi-3-mini"]
    
    # Best efficiency (smaller, faster models)
    recommendations["best_efficiency"] = ["phi-3-mini", "llama-3-8b", "mistral-7b-instruct"]
    
    # Multimodal models
    if include_multimodal:
        multimodal_models = [m for m in compatible_models if m in ["llava-v1.6-7b", "llava-v1.6-13b", "qwen-vl-plus"]]
        recommendations["multimodal"] = multimodal_models
    
    # Code specialized models
    code_models = [m for m in compatible_models if m in ["codellama-7b-instruct", "deepseek-coder-6.7b"]]
    recommendations["code_specialized"] = code_models
    
    # Filter to only include compatible models
    for category in recommendations:
        recommendations[category] = [m for m in recommendations[category] if m in compatible_models]
    
    return recommendations

def run_llava_inference(image_path, question, model_id):
    """
    Run inference with LLaVA or similar multimodal models
    
    Args:
        image_path: Path to the input image
        question: Text question about the image
        model_id: Model identifier from HF_MODEL_CONFIGS
        
    Returns:
        Dict containing the response and metadata
    """
    try:
        model_config = HF_MODEL_CONFIGS[model_id]
        
        if model_config["model_type"] == "openai":
            # Use OpenAI API for inference
            return _run_openai_inference(image_path, question, model_config)
        elif model_config["model_type"] in ["hf", "llava", "qwen", "phi", "llama", "mistral", "gemma", "chatglm", "deepseek"]:
            # Use local transformers for inference
            return _run_local_inference(image_path, question, model_config)
        else:
            raise ValueError(f"Unknown model type: {model_config['model_type']}")
            
    except Exception as e:
        logging.error(f"Error in run_llava_inference: {str(e)}")
        return {
            "response": f"Error during inference: {str(e)}",
            "success": False,
            "error": str(e)
        }

def _run_openai_inference(image_path: str, question: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference using OpenAI API"""
    try:
        # Initialize OpenAI client
        client = openai.OpenAI(api_key=os.getenv("OPENAI_API_KEY"))
        
        # Read and encode image
        with open(image_path, "rb") as image_file:
            image_data = base64.b64encode(image_file.read()).decode('utf-8')
        
        # Prepare the message
        messages = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": question
                    },
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": f"data:image/jpeg;base64,{image_data}"
                        }
                    }
                ]
            }
        ]
        
        # Make API call
        response = client.chat.completions.create(
            model=model_config["model_id"],
            messages=messages,
            max_tokens=model_config.get("max_tokens", 1000),
            temperature=model_config.get("temperature", 0.7)
        )
        
        return {
            "response": response.choices[0].message.content,
            "success": True,
            "model": model_config["model_id"],
            "usage": {
                "prompt_tokens": response.usage.prompt_tokens,
                "completion_tokens": response.usage.completion_tokens,
                "total_tokens": response.usage.total_tokens
            }
        }
        
    except Exception as e:
        logging.error(f"OpenAI API error: {str(e)}")
        return {
            "response": f"OpenAI API error: {str(e)}",
            "success": False,
            "error": str(e)
        }

def _run_local_inference(image_path: str, question: str, model_config: Dict[str, Any]) -> Dict[str, Any]:
    """Run inference using local transformers"""
    try:
        # Load model and processor (with caching)
        model_name = model_config["model_id"]
        
        # Check if model is already loaded
        if not hasattr(_run_local_inference, '_loaded_models'):
            _run_local_inference._loaded_models = {}
        
        if model_name not in _run_local_inference._loaded_models:
            logging.info(f"Loading model: {model_name}")
            
            # Load processor and model
            processor = AutoProcessor.from_pretrained(model_name)
            model = AutoModelForCausalLM.from_pretrained(
                model_name,
                torch_dtype=torch.float16,
                device_map="auto",
                trust_remote_code=True
            )
            
            _run_local_inference._loaded_models[model_name] = {
                'processor': processor,
                'model': model
            }
        
        processor = _run_local_inference._loaded_models[model_name]['processor']
        model = _run_local_inference._loaded_models[model_name]['model']
        
        # Load and preprocess image
        image = Image.open(image_path).convert('RGB')
        
        # Prepare inputs
        inputs = processor(
            text=question,
            images=image,
            return_tensors="pt"
        )
        
        # Move inputs to device
        inputs = {k: v.to(model.device) for k, v in inputs.items()}
        
        # Generate response
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=model_config.get("max_tokens", 1000),
                temperature=model_config.get("temperature", 0.7),
                do_sample=True,
                pad_token_id=processor.tokenizer.eos_token_id
            )
        
        # Decode response
        response_text = processor.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract only the generated part (remove input)
        input_length = inputs['input_ids'].shape[1]
        generated_text = response_text[input_length:].strip()
        
        return {
            "response": generated_text,
            "success": True,
            "model": model_config["model_id"],
            "usage": {
                "input_tokens": input_length,
                "generated_tokens": len(outputs[0]) - input_length,
                "total_tokens": len(outputs[0])
            }
        }
        
    except Exception as e:
        logging.error(f"Local inference error: {str(e)}")
        return {
            "response": f"Local inference error: {str(e)}",
            "success": False,
            "error": str(e)
        }

def cleanup_loaded_models():
    """Clean up loaded models to free memory"""
    if hasattr(_run_local_inference, '_loaded_models'):
        for model_name, model_data in _run_local_inference._loaded_models.items():
            del model_data['model']
            del model_data['processor']
        _run_local_inference._loaded_models.clear()
        if torch.cuda.is_available():
            torch.cuda.empty_cache() 

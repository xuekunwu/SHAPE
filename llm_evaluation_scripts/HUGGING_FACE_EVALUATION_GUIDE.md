# 🤖 Hugging Face LLM Evaluation Guide

Complete guide for evaluating different LLMs on the FBagent platform using Hugging Face models.

## 🚀 Quick Start

### 1. Deploy on Hugging Face Spaces

**Easiest way to get started with free GPU access:**

1. **Fork this repository** to your Hugging Face account
2. **Create a new Space:**
   - Go to [huggingface.co/spaces](https://huggingface.co/spaces)
   - Click "Create new Space"
   - Choose "Gradio" as SDK
   - Select "GPU" as hardware (free tier available)
   - Connect your forked repository

3. **Set environment variables:**
   ```bash
   HUGGINGFACE_TOKEN=your_hf_token_here
   ```

4. **The Space will automatically build and deploy!**

### 2. Local Installation

```bash
# Install minimal dependencies (avoids auto-gptq issues)
pip install -r llm_evaluation_scripts/hf_requirements_minimal.txt

# Run quick evaluation
python llm_evaluation_scripts/hf_quick_start.py
```

## 📋 Supported Models

### Lightweight Models (4-8B parameters)
- **phi-3-mini**: Microsoft's efficient 3.8B model
- **llama-3-8b**: Meta's latest 8B model
- **gemma-2-9b**: Google's 9B model
- **qwen2-7b**: Alibaba's 7B model

### Medium Models (13-72B parameters)
- **mixtral-8x7b**: MoE model with 47B effective parameters
- **qwen2-72b**: Alibaba's 72B model
- **llama-3-70b**: Meta's 70B model

### Heavyweight Models (70B+ parameters)
- **llama-3-70b**: Meta's largest model
- **gemma-2-27b**: Google's 27B model

### Multimodal Models
- **llava-1.5-7b**: Vision-language model
- **llava-1.5-13b**: Larger vision-language model

## 🎯 Hardware Requirements

### GPU Recommendations

| VRAM | Recommended Models | Quantization |
|------|-------------------|--------------|
| 24GB+ | llama-3-70b, gemma-2-27b | 4bit/8bit |
| 16-24GB | mixtral-8x7b, qwen2-72b | 4bit |
| 8-16GB | llama-3-8b, phi-3-mini | 4bit |
| <8GB | phi-3-mini only | 4bit |

### CPU Mode
- **Recommended**: phi-3-mini, llama-3-8b (4bit)
- **Avoid**: Models > 7B parameters
- **Performance**: 10-50x slower than GPU

## 🔧 Configuration

### Model Selection

```python
# Example: Select models based on hardware
import torch

if torch.cuda.is_available():
    vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
    
    if vram_gb >= 24:
        models = ["llama-3-70b", "gemma-2-27b"]
    elif vram_gb >= 16:
        models = ["mixtral-8x7b", "qwen2-72b"]
    else:
        models = ["phi-3-mini", "llama-3-8b"]
else:
    models = ["phi-3-mini"]  # CPU mode
```

### Quantization Settings

```python
# 4bit quantization (recommended for most cases)
config = {
    "load_in_4bit": True,
    "bnb_4bit_compute_dtype": torch.float16,
    "bnb_4bit_use_double_quant": True,
    "bnb_4bit_quant_type": "nf4"
}

# 8bit quantization (better quality, more memory)
config = {
    "load_in_8bit": True,
    "device_map": "auto"
}
```

## 📊 Evaluation Metrics

### Performance Indicators

1. **Success Rate** (0-100%)
   - Percentage of tasks completed successfully
   - Accounts for crashes, timeouts, and errors

2. **Output Quality** (1-10 scale)
   - Accuracy of analysis results
   - Relevance of generated insights
   - Completeness of responses

3. **Response Time** (seconds)
   - Average processing time per task
   - Includes model loading and inference

4. **Memory Usage** (GB)
   - Peak GPU/CPU memory consumption
   - Important for resource planning

5. **Cost Efficiency**
   - Computational cost per task
   - API calls (if applicable)

### Test Categories

#### 1. Basic Functionality
- Image upload and preprocessing
- Cell segmentation accuracy
- Single-cell cropping quality
- State classification precision

#### 2. Batch Processing
- Multiple image handling
- Memory efficiency
- Processing speed scaling

#### 3. Error Recovery
- Invalid input handling
- Graceful failure modes
- Error message quality

#### 4. Edge Cases
- Extreme image conditions
- Very small/large images
- Unusual cell morphologies

#### 5. Performance Testing
- Large dataset processing
- Memory management
- Cache efficiency

#### 6. User Experience
- Interface responsiveness
- Progress feedback
- Result presentation

## 🚀 Running Evaluations

### Method 1: Gradio Interface (Recommended)

1. **Launch the app:**
   ```bash
   python app.py
   ```

2. **Go to "🤖 LLM Evaluation" tab**

3. **Select models and settings:**
   - Choose models from the list
   - Set quantization level
   - Click "Start Evaluation"

4. **Monitor progress:**
   - Real-time progress updates
   - Live results display
   - Automatic result saving

### Method 2: Command Line

```bash
# Quick evaluation with default settings
python llm_evaluation_scripts/hf_quick_start.py

# Full evaluation with custom parameters
python llm_evaluation_scripts/run_hf_evaluation.py \
    --models phi-3-mini llama-3-8b \
    --quantization 4bit \
    --test_suite llm_evaluation_scripts/test_suite.json \
    --output_dir hf_evaluation_results
```

### Method 3: Programmatic

```python
from llm_evaluation_scripts.hf_evaluation_framework import HuggingFaceEvaluator

# Initialize evaluator
evaluator = HuggingFaceEvaluator(
    test_suite_path="llm_evaluation_scripts/test_suite.json",
    output_dir="hf_evaluation_results"
)

# Evaluate single model
result = evaluator.evaluate_model(
    model_name="phi-3-mini",
    config=HF_MODEL_CONFIGS["phi-3-mini"],
    quantization="4bit"
)

# Compare multiple models
results = {}
for model in ["phi-3-mini", "llama-3-8b"]:
    results[model] = evaluator.evaluate_model(
        model_name=model,
        config=HF_MODEL_CONFIGS[model],
        quantization="4bit"
    )

comparison = evaluator.compare_models(results)
```

## 📈 Results Analysis

### Understanding Scores

| Score Range | Performance Level | Recommendation |
|-------------|-------------------|----------------|
| 90-100 | Excellent | Production-ready |
| 70-89 | Good | Minor optimizations needed |
| 50-69 | Acceptable | Consider model changes |
| <50 | Poor | Requires investigation |

### Cost Analysis

#### Hugging Face Models (Free)
- **Compute Time**: Only cost is GPU/CPU time
- **Model Download**: One-time bandwidth cost
- **Storage**: Minimal disk space for models

#### Commercial APIs (For Comparison)
- **OpenAI**: ~$0.01-0.10 per evaluation
- **Anthropic**: ~$0.02-0.15 per evaluation
- **Google**: ~$0.01-0.08 per evaluation

### Result Files

After evaluation, you'll find:

```
hf_evaluation_results/
├── evaluation_report.json          # Complete results
├── model_comparison.json           # Model comparison data
├── performance_charts/             # Visualization charts
│   ├── success_rate_comparison.png
│   ├── quality_scores.png
│   └── memory_usage.png
└── individual_results/
    ├── phi-3-mini_results.json
    ├── llama-3-8b_results.json
    └── ...
```

## 🔧 Troubleshooting

### Common Issues

#### 1. Installation Problems

**auto-gptq installation fails:**
```bash
# Use minimal requirements instead
pip install -r llm_evaluation_scripts/hf_requirements_minimal.txt
```

**CUDA version mismatch:**
```bash
# Install appropriate PyTorch version
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

#### 2. Memory Issues

**Out of memory errors:**
- Reduce model size
- Use 4bit quantization
- Clear GPU cache between evaluations
- Use CPU mode for very large models

**Memory leak:**
```python
import gc
import torch

# Clear cache after each evaluation
torch.cuda.empty_cache()
gc.collect()
```

#### 3. Model Loading Issues

**Model not found:**
- Check internet connection
- Verify model name spelling
- Ensure sufficient disk space

**Permission errors:**
- Set `HUGGINGFACE_TOKEN` environment variable
- Accept model terms on Hugging Face website

#### 4. Performance Issues

**Slow loading:**
- Use quantization
- Enable model caching
- Consider smaller models

**Inconsistent results:**
- Set random seeds
- Use deterministic settings
- Run multiple evaluations

### Performance Optimization

#### 1. Model Loading
```python
# Enable model caching
from transformers import AutoTokenizer, AutoModelForCausalLM

# Models will be cached locally
tokenizer = AutoTokenizer.from_pretrained("microsoft/phi-3-mini")
model = AutoModelForCausalLM.from_pretrained("microsoft/phi-3-mini")
```

#### 2. Batch Processing
```python
# Process multiple tasks together
tasks = [task1, task2, task3]
results = evaluator.evaluate_batch(tasks, model)
```

#### 3. Memory Management
```python
# Clear cache between evaluations
def clear_memory():
    torch.cuda.empty_cache()
    gc.collect()
    
# Use after each model evaluation
clear_memory()
```

## 📊 Example Results

### Sample Evaluation Output

```json
{
  "model": "phi-3-mini",
  "evaluation_date": "2024-01-15T10:30:00Z",
  "hardware": "NVIDIA RTX 4090 (24GB)",
  "quantization": "4bit",
  "results": {
    "success_rate": 92.5,
    "avg_output_quality": 8.2,
    "avg_response_time": 2.3,
    "peak_memory_usage_gb": 4.1,
    "total_tasks": 40,
    "successful_tasks": 37
  },
  "category_scores": {
    "basic_functionality": 95.0,
    "batch_processing": 88.0,
    "error_recovery": 90.0,
    "edge_cases": 85.0,
    "performance": 92.0,
    "user_experience": 89.0
  }
}
```

### Model Comparison Chart

```
Model Comparison Results:
┌─────────────────┬──────────────┬──────────────┬──────────────┐
│ Model           │ Success Rate │ Avg Quality  │ Memory (GB)  │
├─────────────────┼──────────────┼──────────────┼──────────────┤
│ phi-3-mini      │     92.5%    │     8.2      │     4.1      │
│ llama-3-8b      │     89.0%    │     8.5      │     6.8      │
│ mixtral-8x7b    │     94.0%    │     8.8      │    12.3      │
└─────────────────┴──────────────┴──────────────┴──────────────┘
```

## 🎯 Best Practices

### 1. Model Selection
- Start with lightweight models for testing
- Scale up based on hardware capabilities
- Consider use case requirements

### 2. Evaluation Strategy
- Run multiple evaluations for consistency
- Test edge cases and error conditions
- Monitor resource usage

### 3. Result Interpretation
- Focus on success rate and quality scores
- Consider memory usage for deployment
- Balance performance vs. resource cost

### 4. Continuous Monitoring
- Regular re-evaluation with new models
- Track performance over time
- Update model recommendations

## 🚀 Next Steps

1. **Deploy on Hugging Face Spaces** for free GPU access
2. **Run initial evaluation** with phi-3-mini
3. **Compare multiple models** based on your hardware
4. **Analyze results** and select optimal model
5. **Integrate into production** workflow

---

**Ready to evaluate LLMs? Start with Hugging Face Spaces for the easiest setup! 🚀** 
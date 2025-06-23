# LLM Performance Evaluation Framework for FBagent

This directory contains a comprehensive framework for evaluating different LLM models' performance in the FBagent system.

## Overview

The evaluation framework provides:
- **Standardized test cases** for fibroblast analysis tasks
- **Automated evaluation** across multiple LLM models
- **Comprehensive metrics** including success rate, quality, cost, and speed
- **Visualization tools** for result analysis and comparison
- **Configurable model settings** for different providers

## Directory Structure

```
llm_evaluation_scripts/
├── evaluation_framework.py    # Core evaluation engine
├── model_configs.py          # Model configurations and settings
├── test_suite.json           # Standardized test cases
├── run_evaluation.py         # Main evaluation runner
├── visualize_results.py      # Results visualization tools
└── README.md                 # This file
```

## Quick Start

### 1. Setup Environment

```bash
# Install required dependencies
pip install pandas matplotlib seaborn numpy

# Set up environment variables
export OPENAI_API_KEY="your-openai-api-key"
export ANTHROPIC_API_KEY="your-anthropic-api-key"  # Optional
export GOOGLE_API_KEY="your-google-api-key"        # Optional
```

### 2. Run Basic Evaluation

```bash
# Evaluate all available models
python run_evaluation.py \
    --test-suite test_suite.json \
    --output-dir results/

# Evaluate specific models
python run_evaluation.py \
    --test-suite test_suite.json \
    --output-dir results/ \
    --models gpt-4o gpt-4o-mini

# Evaluate models by category
python run_evaluation.py \
    --test-suite test_suite.json \
    --output-dir results/ \
    --category premium
```

### 3. Generate Visualizations

```bash
# Create all visualizations
python visualize_results.py \
    --results-file results/evaluation_report.json \
    --output-dir results/visualizations/

# Create only the main dashboard
python visualize_results.py \
    --results-file results/evaluation_report.json \
    --output-dir results/visualizations/ \
    --dashboard-only
```

## Test Suite

The test suite includes 15 standardized test cases covering:

### Basic Functionality (TC001-TC002)
- Single cell analysis
- Batch processing

### Advanced Analysis (TC003, TC006, TC009-TC010)
- Multi-modal analysis with literature
- Complex multi-step analysis
- Code generation
- Knowledge integration

### Edge Cases (TC004-TC005, TC013)
- Error handling
- Low confidence scenarios
- Poor quality images

### Performance Tests (TC011-TC012)
- Speed testing
- Scalability testing

### Interactive Features (TC014)
- Multi-turn conversations

### Comprehensive Evaluation (TC015)
- End-to-end system testing

## Supported Models

### OpenAI Models
- `gpt-4o` - Latest GPT-4 model
- `gpt-4o-mini` - Faster, more cost-effective option
- `gpt-4o-mini-2024-07-18` - Specific version
- `gpt-4o-2024-08-06` - Specific version

### Anthropic Models
- `claude-3-5-sonnet` - High-performance model
- `claude-3-5-haiku` - Fast, cost-effective option

### Google Models
- `gemini-1.5-pro` - Advanced multimodal model
- `gemini-1.5-flash` - Fast, efficient model

### Other Models
- `deepseek` - Alternative provider

## Model Categories

### Premium Tier
- High performance, higher cost
- Models: `gpt-4o`, `claude-3-5-sonnet`, `gemini-1.5-pro`
- Expected: >85% success rate, >8.0 quality score

### Standard Tier
- Balanced performance and cost
- Models: `gpt-4o-mini`, `claude-3-5-haiku`, `gemini-1.5-flash`
- Expected: >75% success rate, >7.0 quality score

### Budget Tier
- Cost-effective options
- Models: `gpt-4o-mini-2024-07-18`, `deepseek`
- Expected: >65% success rate, >6.0 quality score

## Evaluation Metrics

### Core Metrics
- **Success Rate**: Percentage of successfully completed tasks
- **Output Quality**: Expert-rated quality score (0-10)
- **Execution Time**: Average time per task completion
- **Cost**: Total cost per model
- **User Satisfaction**: Simulated user satisfaction score (0-5)

### Derived Metrics
- **Performance Score**: Weighted combination of all metrics (0-100)
- **Cost Efficiency**: Performance per dollar spent
- **Token Efficiency**: Tokens used per successful task
- **Error Rate**: Percentage of failed tasks

## Advanced Usage

### Parallel Evaluation

```bash
# Run evaluations in parallel for faster results
python run_evaluation.py \
    --test-suite test_suite.json \
    --output-dir results/ \
    --parallel \
    --max-workers 3
```

### Custom Model Configuration

Edit `model_configs.py` to add custom models or modify existing configurations:

```python
# Add custom model
MODEL_CONFIGS["my-custom-model"] = {
    "model_string": "custom-model-name",
    "temperature": 0.3,
    "max_tokens": 4000,
    "system_prompt": "Custom system prompt...",
    "expected_cost_per_1k_tokens": 0.001,
    "capabilities": ["multimodal", "tool_calling"],
    "api_key": os.getenv("CUSTOM_API_KEY"),
    "is_multimodal": True,
    "enable_cache": False
}
```

### Custom Test Cases

Add custom test cases to `test_suite.json`:

```json
{
  "id": "TC016",
  "name": "Custom Test Case",
  "description": "Description of the test case",
  "category": "custom",
  "difficulty": "medium",
  "input": {
    "image": "path/to/image.png",
    "query": "Test query"
  },
  "expected_output": {
    "required_field": "expected_value"
  },
  "success_criteria": {
    "must_include_field": true,
    "min_response_length": 50
  }
}
```

## Output Files

### Evaluation Results
- `evaluation_report.json` - Complete evaluation results
- `{model_name}_results.json` - Individual model results
- `model_comparison.json` - Model comparison analysis

### Visualizations
- `performance_dashboard.png` - Main performance dashboard
- `radar_chart.png` - Radar chart comparison
- `cost_analysis.png` - Cost analysis charts
- `test_case_analysis.png` - Test case performance
- `summary_statistics.png` - Summary statistics table
- `summary_statistics.csv` - Raw summary data

## Best Practices

### 1. Environment Setup
- Use virtual environments for dependency management
- Set up API keys securely
- Monitor API usage and costs

### 2. Evaluation Strategy
- Start with a subset of models for initial testing
- Use parallel evaluation for large-scale comparisons
- Save intermediate results for long evaluations

### 3. Result Analysis
- Review both quantitative metrics and qualitative outputs
- Consider cost-effectiveness alongside performance
- Validate results with domain experts

### 4. Continuous Evaluation
- Run evaluations regularly as models update
- Track performance trends over time
- Update test cases based on new requirements

## Troubleshooting

### Common Issues

1. **API Key Errors**
   ```
   Error: Missing required environment variables
   ```
   Solution: Ensure API keys are set in environment variables

2. **Model Initialization Errors**
   ```
   Error: Failed to initialize model
   ```
   Solution: Check model configuration and API access

3. **Test Case Failures**
   ```
   Error: Test case execution failed
   ```
   Solution: Verify test case format and input files

4. **Memory Issues**
   ```
   Error: Out of memory
   ```
   Solution: Reduce batch size or use fewer parallel workers

### Debug Mode

Enable verbose logging for detailed debugging:

```bash
python run_evaluation.py \
    --test-suite test_suite.json \
    --output-dir results/ \
    --verbose
```

## Contributing

To contribute to the evaluation framework:

1. **Add New Models**: Update `model_configs.py` with new model configurations
2. **Add Test Cases**: Create new test cases in `test_suite.json`
3. **Improve Metrics**: Enhance evaluation metrics in `evaluation_framework.py`
4. **Add Visualizations**: Create new visualization types in `visualize_results.py`

## License

This evaluation framework is part of the FBagent project and follows the same license terms.

## Support

For questions and support:
1. Check the troubleshooting section above
2. Review the code comments and documentation
3. Open an issue in the project repository
4. Contact the development team

---

**Note**: This evaluation framework is designed specifically for the FBagent system and fibroblast analysis tasks. Adaptations may be needed for other use cases. 
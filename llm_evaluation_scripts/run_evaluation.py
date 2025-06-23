#!/usr/bin/env python3
"""
Automated LLM Performance Evaluation Runner for FBagent
Main script to run comprehensive LLM evaluation across multiple models
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import logging

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_evaluation_scripts.evaluation_framework import LLMEvaluator
from llm_evaluation_scripts.model_configs import MODEL_CONFIGS, get_models_by_category

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def validate_environment():
    """Validate that required environment variables are set"""
    required_vars = ["OPENAI_API_KEY"]
    missing_vars = []
    
    for var in required_vars:
        if not Path(f"~/.env").expanduser().exists():
            logger.warning(f"Environment file not found. Please ensure {var} is set.")
            missing_vars.append(var)
    
    if missing_vars:
        logger.error(f"Missing required environment variables: {missing_vars}")
        return False
    
    return True

def get_models_to_evaluate(args) -> List[str]:
    """Determine which models to evaluate based on arguments"""
    if args.models:
        # Use explicitly specified models
        models_to_test = args.models
    elif args.category:
        # Use models from specified category
        try:
            models_to_test = get_models_by_category(args.category)
        except ValueError as e:
            logger.error(f"Invalid category: {e}")
            return []
    else:
        # Use all available models
        models_to_test = list(MODEL_CONFIGS.keys())
    
    # Filter by availability if specified
    if args.available_only:
        available_models = []
        for model in models_to_test:
            config = MODEL_CONFIGS.get(model, {})
            if config.get("api_key"):
                available_models.append(model)
            else:
                logger.warning(f"Skipping {model}: No API key available")
        models_to_test = available_models
    
    return models_to_test

def run_single_model_evaluation(evaluator: LLMEvaluator, model_name: str, 
                               model_config: Dict[str, Any], args) -> Dict[str, Any]:
    """Run evaluation for a single model"""
    logger.info(f"Starting evaluation for model: {model_name}")
    start_time = time.time()
    
    try:
        # Run evaluation
        results = evaluator.evaluate_model(model_name, model_config)
        
        # Calculate evaluation time
        evaluation_time = time.time() - start_time
        
        # Add metadata
        results["evaluation_metadata"] = {
            "evaluation_time": evaluation_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_suite_size": len(evaluator.test_suite),
            "parallel_execution": args.parallel
        }
        
        logger.info(f"Completed evaluation for {model_name} in {evaluation_time:.2f} seconds")
        return results
        
    except Exception as e:
        logger.error(f"Error evaluating {model_name}: {e}")
        return {
            "model_name": model_name,
            "error": str(e),
            "evaluation_metadata": {
                "evaluation_time": time.time() - start_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "error_occurred": True
            }
        }

def run_parallel_evaluation(evaluator: LLMEvaluator, models_to_test: List[str], args) -> Dict[str, Any]:
    """Run evaluation for multiple models in parallel"""
    import concurrent.futures
    
    results = {}
    
    with concurrent.futures.ThreadPoolExecutor(max_workers=args.max_workers) as executor:
        # Submit evaluation tasks
        future_to_model = {}
        for model_name in models_to_test:
            if model_name in MODEL_CONFIGS:
                config = MODEL_CONFIGS[model_name].copy()
                future = executor.submit(run_single_model_evaluation, evaluator, model_name, config, args)
                future_to_model[future] = model_name
        
        # Collect results
        for future in concurrent.futures.as_completed(future_to_model):
            model_name = future_to_model[future]
            try:
                result = future.result()
                results[model_name] = result
                logger.info(f"Completed parallel evaluation for {model_name}")
            except Exception as e:
                logger.error(f"Error in parallel evaluation for {model_name}: {e}")
                results[model_name] = {
                    "model_name": model_name,
                    "error": str(e),
                    "evaluation_metadata": {"error_occurred": True}
                }
    
    return results

def run_sequential_evaluation(evaluator: LLMEvaluator, models_to_test: List[str], args) -> Dict[str, Any]:
    """Run evaluation for multiple models sequentially"""
    results = {}
    
    for i, model_name in enumerate(models_to_test, 1):
        logger.info(f"Evaluating model {i}/{len(models_to_test)}: {model_name}")
        
        if model_name in MODEL_CONFIGS:
            config = MODEL_CONFIGS[model_name].copy()
            result = run_single_model_evaluation(evaluator, model_name, config, args)
            results[model_name] = result
        else:
            logger.warning(f"Model {model_name} not found in configurations, skipping")
    
    return results

def generate_evaluation_report(results: Dict[str, Any], output_dir: Path, args):
    """Generate comprehensive evaluation report"""
    report = {
        "evaluation_summary": {
            "total_models": len(results),
            "successful_evaluations": len([r for r in results.values() if "error" not in r]),
            "failed_evaluations": len([r for r in results.values() if "error" in r]),
            "evaluation_date": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_suite": args.test_suite,
            "parallel_execution": args.parallel
        },
        "model_results": results,
        "performance_comparison": {},
        "recommendations": {}
    }
    
    # Generate performance comparison
    successful_results = {k: v for k, v in results.items() if "error" not in v}
    if successful_results:
        evaluator = LLMEvaluator(args.test_suite, str(output_dir))
        comparison = evaluator.compare_models(successful_results)
        report["performance_comparison"] = comparison
    
    # Save report
    report_file = output_dir / "evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation report saved to {report_file}")
    
    # Print summary
    print_evaluation_summary(report)

def print_evaluation_summary(report: Dict[str, Any]):
    """Print a summary of evaluation results"""
    summary = report["evaluation_summary"]
    
    print("\n" + "="*60)
    print("LLM EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Models Evaluated: {summary['total_models']}")
    print(f"Successful Evaluations: {summary['successful_evaluations']}")
    print(f"Failed Evaluations: {summary['failed_evaluations']}")
    print(f"Evaluation Date: {summary['evaluation_date']}")
    print(f"Test Suite: {summary['test_suite']}")
    print(f"Parallel Execution: {summary['parallel_execution']}")
    
    # Performance rankings
    if "performance_comparison" in report and "overall_rankings" in report["performance_comparison"]:
        rankings = report["performance_comparison"]["overall_rankings"]
        print("\nTOP PERFORMING MODELS:")
        print("-" * 30)
        for i, (model, score) in enumerate(list(rankings.items())[:5], 1):
            print(f"{i}. {model}: {score:.3f}")
    
    # Recommendations
    if "performance_comparison" in report and "recommendations" in report["performance_comparison"]:
        recommendations = report["performance_comparison"]["recommendations"]
        print("\nRECOMMENDATIONS:")
        print("-" * 30)
        for key, value in recommendations.items():
            print(f"• {key.replace('_', ' ').title()}: {value}")
    
    print("="*60)

def main():
    parser = argparse.ArgumentParser(
        description="Run comprehensive LLM performance evaluation for FBagent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Evaluate all available models
  python run_evaluation.py --test-suite test_suite.json --output-dir results/
  
  # Evaluate specific models
  python run_evaluation.py --test-suite test_suite.json --output-dir results/ --models gpt-4o gpt-4o-mini
  
  # Evaluate models by category
  python run_evaluation.py --test-suite test_suite.json --output-dir results/ --category premium
  
  # Run parallel evaluation
  python run_evaluation.py --test-suite test_suite.json --output-dir results/ --parallel --max-workers 3
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--test-suite", 
        required=True, 
        help="Path to test suite JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        required=True, 
        help="Output directory for evaluation results"
    )
    
    # Model selection
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific models to evaluate (default: all available)"
    )
    parser.add_argument(
        "--category", 
        choices=["premium", "standard", "budget", "multimodal", "fast"],
        help="Evaluate models in specific category"
    )
    parser.add_argument(
        "--available-only", 
        action="store_true",
        help="Only evaluate models with available API keys"
    )
    
    # Execution options
    parser.add_argument(
        "--parallel", 
        action="store_true",
        help="Run evaluations in parallel"
    )
    parser.add_argument(
        "--max-workers", 
        type=int, 
        default=3,
        help="Maximum number of parallel workers (default: 3)"
    )
    
    # Output options
    parser.add_argument(
        "--save-intermediate", 
        action="store_true",
        help="Save intermediate results during evaluation"
    )
    parser.add_argument(
        "--verbose", 
        action="store_true",
        help="Enable verbose logging"
    )
    
    args = parser.parse_args()
    
    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Validate environment
    if not validate_environment():
        sys.exit(1)
    
    # Validate test suite file
    test_suite_path = Path(args.test_suite)
    if not test_suite_path.exists():
        logger.error(f"Test suite file not found: {test_suite_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get models to evaluate
    models_to_test = get_models_to_evaluate(args)
    if not models_to_test:
        logger.error("No models to evaluate")
        sys.exit(1)
    
    logger.info(f"Will evaluate {len(models_to_test)} models: {models_to_test}")
    
    # Initialize evaluator
    try:
        evaluator = LLMEvaluator(str(test_suite_path), str(output_dir))
        logger.info(f"Initialized evaluator with {len(evaluator.test_suite)} test cases")
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        sys.exit(1)
    
    # Run evaluation
    start_time = time.time()
    
    if args.parallel:
        logger.info("Running parallel evaluation")
        results = run_parallel_evaluation(evaluator, models_to_test, args)
    else:
        logger.info("Running sequential evaluation")
        results = run_sequential_evaluation(evaluator, models_to_test, args)
    
    total_time = time.time() - start_time
    
    # Generate report
    generate_evaluation_report(results, output_dir, args)
    
    logger.info(f"Evaluation completed in {total_time:.2f} seconds")
    logger.info(f"Results saved to: {output_dir}")

if __name__ == "__main__":
    main() 
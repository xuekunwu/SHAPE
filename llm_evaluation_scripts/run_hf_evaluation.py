#!/usr/bin/env python3
"""
Hugging Face LLM Performance Evaluation Runner for FBagent
Main script to run comprehensive evaluation of open-source LLMs
"""

import argparse
import json
import sys
import time
from pathlib import Path
from typing import List, Dict, Any
import logging
import torch

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

from llm_evaluation_scripts.hf_evaluation_framework import HuggingFaceEvaluator
from llm_evaluation_scripts.hf_model_configs import (
    HF_MODEL_CONFIGS, get_hf_models_by_category, 
    get_hf_models_by_hardware, get_optimal_models_for_hardware
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('hf_evaluation.log'),
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger(__name__)

def check_hardware_compatibility():
    """Check hardware compatibility and provide recommendations"""
    print("=== Hardware Compatibility Check ===")
    
    # Check CUDA availability
    if torch.cuda.is_available():
        gpu_count = torch.cuda.device_count()
        total_vram = 0
        
        for i in range(gpu_count):
            vram = torch.cuda.get_device_properties(i).total_memory / (1024**3)
            total_vram += vram
            print(f"GPU {i}: {torch.cuda.get_device_name(i)} ({vram:.1f}GB VRAM)")
        
        print(f"Total VRAM: {total_vram:.1f}GB")
        
        # Get model recommendations
        recommendations = get_optimal_models_for_hardware(int(total_vram))
        
        print("\nRecommended models for your hardware:")
        for category, models in recommendations.items():
            if models:
                print(f"  {category}: {', '.join(models)}")
        
        return int(total_vram)
    else:
        print("CUDA not available. Using CPU mode.")
        print("Note: CPU evaluation will be significantly slower.")
        return 0

def get_models_to_evaluate(args, available_vram: int) -> List[str]:
    """Determine which models to evaluate based on arguments and hardware"""
    if args.models:
        # Use explicitly specified models
        models_to_test = args.models
    elif args.category:
        # Use models from specified category
        try:
            models_to_test = get_hf_models_by_category(args.category)
        except ValueError as e:
            logger.error(f"Invalid category: {e}")
            return []
    elif args.hardware_optimized:
        # Use hardware-optimized selection
        recommendations = get_optimal_models_for_hardware(available_vram)
        models_to_test = []
        for category_models in recommendations.values():
            models_to_test.extend(category_models)
    else:
        # Use all available models that fit in memory
        models_to_test = get_hf_models_by_hardware(available_vram)
    
    # Filter by quantization if specified
    if args.quantization:
        filtered_models = []
        for model in models_to_test:
            if model in HF_MODEL_CONFIGS:
                config = HF_MODEL_CONFIGS[model]
                if args.quantization in config.get("quantization", []):
                    filtered_models.append(model)
        models_to_test = filtered_models
    
    return models_to_test

def run_single_model_evaluation(evaluator: HuggingFaceEvaluator, model_name: str, 
                               model_config: Dict[str, Any], quantization: str, args) -> Dict[str, Any]:
    """Run evaluation for a single Hugging Face model"""
    logger.info(f"Starting evaluation for model: {model_name} ({quantization})")
    start_time = time.time()
    
    try:
        # Run evaluation
        results = evaluator.evaluate_model(model_name, model_config, quantization)
        
        # Calculate evaluation time
        evaluation_time = time.time() - start_time
        
        # Add evaluation metadata
        results["evaluation_metadata"] = {
            "evaluation_time": evaluation_time,
            "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
            "test_suite_size": len(evaluator.test_suite),
            "quantization": quantization,
            "device": evaluator.device
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
                "error_occurred": True,
                "quantization": quantization
            }
        }

def run_sequential_evaluation(evaluator: HuggingFaceEvaluator, models_to_test: List[str], args) -> Dict[str, Any]:
    """Run evaluation for multiple models sequentially"""
    results = {}
    
    for i, model_name in enumerate(models_to_test, 1):
        logger.info(f"Evaluating model {i}/{len(models_to_test)}: {model_name}")
        
        if model_name in HF_MODEL_CONFIGS:
            config = HF_MODEL_CONFIGS[model_name].copy()
            result = run_single_model_evaluation(evaluator, model_name, config, args.quantization, args)
            results[model_name] = result
            
            # Clean up after each model to free memory
            if args.cleanup_after_model:
                evaluator.cleanup()
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
            "quantization": args.quantization,
            "device": "cuda" if torch.cuda.is_available() else "cpu"
        },
        "model_results": results,
        "performance_comparison": {},
        "recommendations": {}
    }
    
    # Generate performance comparison
    successful_results = {k: v for k, v in results.items() if "error" not in v}
    if successful_results:
        comparison = evaluator.compare_models(successful_results)
        report["performance_comparison"] = comparison
    
    # Save report
    report_file = output_dir / "hf_evaluation_report.json"
    with open(report_file, 'w', encoding='utf-8') as f:
        json.dump(report, f, indent=2, ensure_ascii=False)
    
    logger.info(f"Evaluation report saved to {report_file}")
    
    # Print summary
    print_evaluation_summary(report)

def print_evaluation_summary(report: Dict[str, Any]):
    """Print a summary of evaluation results"""
    summary = report["evaluation_summary"]
    
    print("\n" + "="*60)
    print("HUGGING FACE LLM EVALUATION SUMMARY")
    print("="*60)
    print(f"Total Models Evaluated: {summary['total_models']}")
    print(f"Successful Evaluations: {summary['successful_evaluations']}")
    print(f"Failed Evaluations: {summary['failed_evaluations']}")
    print(f"Evaluation Date: {summary['evaluation_date']}")
    print(f"Test Suite: {summary['test_suite']}")
    print(f"Quantization: {summary['quantization']}")
    print(f"Device: {summary['device']}")
    
    # Performance rankings
    if "performance_comparison" in report and "overall_rankings" in report["performance_comparison"]:
        rankings = report["performance_comparison"]["overall_rankings"]
        print("\nTOP PERFORMING MODELS:")
        print("-" * 30)
        for i, (model, score) in enumerate(list(rankings.items())[:5], 1):
            print(f"{i}. {model}: {score:.3f}")
    
    # Memory efficiency rankings
    if "performance_comparison" in report and "memory_efficiency" in report["performance_comparison"]:
        memory_eff = report["performance_comparison"]["memory_efficiency"]
        print("\nMOST MEMORY EFFICIENT MODELS:")
        print("-" * 30)
        sorted_memory = sorted(memory_eff.items(), key=lambda x: x[1], reverse=True)
        for i, (model, efficiency) in enumerate(sorted_memory[:3], 1):
            print(f"{i}. {model}: {efficiency:.3f} points/GB")
    
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
        description="Run comprehensive Hugging Face LLM performance evaluation for FBagent",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Check hardware compatibility
  python run_hf_evaluation.py --check-hardware
  
  # Evaluate lightweight models
  python run_hf_evaluation.py --test-suite test_suite.json --output-dir results/ --category lightweight
  
  # Evaluate specific models
  python run_hf_evaluation.py --test-suite test_suite.json --output-dir results/ --models llama-3-8b mistral-7b-instruct
  
  # Hardware-optimized evaluation
  python run_hf_evaluation.py --test-suite test_suite.json --output-dir results/ --hardware-optimized
  
  # Use 8-bit quantization
  python run_hf_evaluation.py --test-suite test_suite.json --output-dir results/ --quantization 8bit
        """
    )
    
    # Required arguments
    parser.add_argument(
        "--test-suite", 
        help="Path to test suite JSON file"
    )
    parser.add_argument(
        "--output-dir", 
        help="Output directory for evaluation results"
    )
    
    # Model selection
    parser.add_argument(
        "--models", 
        nargs="+", 
        help="Specific models to evaluate"
    )
    parser.add_argument(
        "--category", 
        choices=["lightweight", "medium", "heavyweight", "multimodal", "code_specialized", "chinese"],
        help="Evaluate models in specific category"
    )
    parser.add_argument(
        "--hardware-optimized", 
        action="store_true",
        help="Automatically select models based on available hardware"
    )
    
    # Quantization options
    parser.add_argument(
        "--quantization", 
        choices=["4bit", "8bit", "16bit"],
        default="4bit",
        help="Quantization level to use (default: 4bit)"
    )
    
    # Hardware check
    parser.add_argument(
        "--check-hardware", 
        action="store_true",
        help="Check hardware compatibility and exit"
    )
    
    # Execution options
    parser.add_argument(
        "--cleanup-after-model", 
        action="store_true",
        help="Clean up memory after each model evaluation"
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
    
    # Check hardware compatibility
    available_vram = check_hardware_compatibility()
    
    if args.check_hardware:
        return
    
    # Validate required arguments
    if not args.test_suite or not args.output_dir:
        parser.error("--test-suite and --output-dir are required")
    
    # Validate test suite file
    test_suite_path = Path(args.test_suite)
    if not test_suite_path.exists():
        logger.error(f"Test suite file not found: {test_suite_path}")
        sys.exit(1)
    
    # Create output directory
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Get models to evaluate
    models_to_test = get_models_to_evaluate(args, available_vram)
    if not models_to_test:
        logger.error("No models to evaluate")
        sys.exit(1)
    
    logger.info(f"Will evaluate {len(models_to_test)} models: {models_to_test}")
    
    # Initialize evaluator
    try:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        evaluator = HuggingFaceEvaluator(str(test_suite_path), str(output_dir), device)
        logger.info(f"Initialized evaluator with {len(evaluator.test_suite)} test cases")
    except Exception as e:
        logger.error(f"Failed to initialize evaluator: {e}")
        sys.exit(1)
    
    # Run evaluation
    start_time = time.time()
    
    logger.info("Running sequential evaluation")
    results = run_sequential_evaluation(evaluator, models_to_test, args)
    
    total_time = time.time() - start_time
    
    # Generate report
    generate_evaluation_report(results, output_dir, args)
    
    logger.info(f"Evaluation completed in {total_time:.2f} seconds")
    logger.info(f"Results saved to: {output_dir}")
    
    # Cleanup
    evaluator.cleanup()

if __name__ == "__main__":
    main() 
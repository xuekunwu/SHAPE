#!/usr/bin/env python3
"""
Quick Start Script for Hugging Face LLM Evaluation
Optimized for Hugging Face Spaces deployment
"""

import os
import sys
import json
import time
import torch
import argparse
from pathlib import Path
from typing import List, Dict, Any

# Add project root to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

try:
    from hf_evaluation_framework import HuggingFaceEvaluator
    from hf_model_configs import HF_MODEL_CONFIGS, get_hf_models_by_category
except ImportError:
    print("❌ HF evaluation framework not found. Please install dependencies first:")
    print("pip install -r llm_evaluation_scripts/hf_requirements_minimal.txt")
    sys.exit(1)

def check_hardware():
    """Check available hardware and recommend models"""
    print("🔍 Checking hardware...")
    
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name()
        vram_gb = torch.cuda.get_device_properties(0).total_memory / (1024**3)
        print(f"✅ GPU detected: {gpu_name}")
        print(f"✅ VRAM: {vram_gb:.1f}GB")
        
        # Recommend models based on VRAM
        if vram_gb >= 24:
            recommended = ["phi-3-mini", "llama-3-8b", "mixtral-8x7b"]
            print("🎯 Recommended: Heavyweight models available")
        elif vram_gb >= 16:
            recommended = ["phi-3-mini", "llama-3-8b", "mixtral-8x7b"]
            print("🎯 Recommended: Medium models available")
        elif vram_gb >= 8:
            recommended = ["phi-3-mini", "llama-3-8b"]
            print("🎯 Recommended: Lightweight models available")
        else:
            recommended = ["phi-3-mini"]
            print("⚠️  Limited VRAM: Only lightweight models recommended")
    else:
        print("⚠️  No GPU detected - using CPU mode")
        recommended = ["phi-3-mini"]
        print("🎯 Recommended: CPU-optimized models only")
    
    return recommended

def run_quick_evaluation(models: List[str], quantization: str = "4bit"):
    """Run quick evaluation with selected models"""
    
    print(f"\n🚀 Starting quick evaluation...")
    print(f"📋 Models: {', '.join(models)}")
    print(f"⚙️  Quantization: {quantization}")
    
    # Initialize evaluator
    test_suite_path = Path(__file__).parent / "test_suite.json"
    output_dir = Path("hf_evaluation_results")
    output_dir.mkdir(exist_ok=True)
    
    evaluator = HuggingFaceEvaluator(str(test_suite_path), str(output_dir))
    
    results = {}
    start_time = time.time()
    
    for i, model_name in enumerate(models, 1):
        print(f"\n📊 Evaluating {model_name} ({i}/{len(models)})...")
        
        try:
            if model_name not in HF_MODEL_CONFIGS:
                print(f"❌ Model {model_name} not found in configs")
                continue
            
            config = HF_MODEL_CONFIGS[model_name].copy()
            
            # Run evaluation
            result = evaluator.evaluate_model(model_name, config, quantization)
            results[model_name] = result
            
            # Print quick results
            if "performance_score" in result:
                print(f"✅ {model_name}: {result['performance_score']:.1f}/100")
                print(f"   Success Rate: {result.get('success_rate', 0):.1%}")
                print(f"   Memory: {result.get('memory_usage_gb', 0):.1f}GB")
            else:
                print(f"❌ {model_name}: Evaluation failed")
                
        except Exception as e:
            print(f"❌ Error evaluating {model_name}: {str(e)}")
            results[model_name] = {"error": str(e)}
        
        # Clear memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
    
    # Generate comparison
    if len(results) > 1:
        print(f"\n📈 Generating comparison...")
        try:
            comparison = evaluator.compare_models(results)
            
            # Save results
            with open(output_dir / "quick_evaluation_results.json", "w") as f:
                json.dump({
                    "results": results,
                    "comparison": comparison,
                    "evaluation_time": time.time() - start_time
                }, f, indent=2)
            
            # Print summary
            print(f"\n🎉 Quick evaluation completed!")
            print(f"⏱️  Total time: {time.time() - start_time:.1f}s")
            print(f"📁 Results saved to: {output_dir}")
            
            # Show top performers
            if "performance_scores" in comparison:
                scores = comparison["performance_scores"]
                sorted_models = sorted(scores.items(), key=lambda x: x[1], reverse=True)
                
                print(f"\n🏆 Top Performers:")
                for model, score in sorted_models[:3]:
                    print(f"   {model}: {score:.1f}/100")
            
        except Exception as e:
            print(f"❌ Error generating comparison: {str(e)}")
    
    return results

def main():
    """Main function for quick start"""
    
    parser = argparse.ArgumentParser(description="Quick HF LLM Evaluation")
    parser.add_argument("--models", nargs="+", help="Models to evaluate")
    parser.add_argument("--quantization", default="4bit", choices=["4bit", "8bit"], 
                       help="Quantization level")
    parser.add_argument("--auto", action="store_true", 
                       help="Auto-select models based on hardware")
    
    args = parser.parse_args()
    
    print("🤖 Hugging Face LLM Evaluation - Quick Start")
    print("=" * 50)
    
    # Check hardware
    recommended_models = check_hardware()
    
    # Determine models to evaluate
    if args.auto:
        models = recommended_models[:2]  # Limit to 2 models for quick evaluation
        print(f"\n🤖 Auto-selected models: {', '.join(models)}")
    elif args.models:
        models = args.models
        print(f"\n🎯 User-selected models: {', '.join(models)}")
    else:
        # Default to phi-3-mini for quick test
        models = ["phi-3-mini"]
        print(f"\n⚡ Default model: {', '.join(models)}")
    
    # Validate models
    available_models = list(HF_MODEL_CONFIGS.keys())
    invalid_models = [m for m in models if m not in available_models]
    
    if invalid_models:
        print(f"❌ Invalid models: {', '.join(invalid_models)}")
        print(f"✅ Available models: {', '.join(available_models[:5])}...")
        return
    
    # Run evaluation
    results = run_quick_evaluation(models, args.quantization)
    
    # Final recommendations
    print(f"\n💡 Next Steps:")
    print(f"   1. Check detailed results in hf_evaluation_results/")
    print(f"   2. Run full evaluation with more models")
    print(f"   3. Deploy best model to production")
    
    if torch.cuda.is_available():
        print(f"   4. Consider upgrading GPU for larger models")

if __name__ == "__main__":
    main() 
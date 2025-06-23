"""
Hugging Face LLM Evaluation Framework for FBagent
Specialized evaluation engine for open-source LLMs on Hugging Face
"""

import time
import json
import statistics
import asyncio
import torch
import gc
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
from pathlib import Path
import logging
from datetime import datetime

# Hugging Face imports
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoProcessor,
    BitsAndBytesConfig, pipeline
)
from accelerate import Accelerator
import huggingface_hub

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class HFModelInfo:
    """Information about a loaded Hugging Face model"""
    model_name: str
    model: Any
    tokenizer: Any
    processor: Optional[Any] = None
    device: str = "cuda"
    quantization: str = "4bit"
    memory_usage_gb: float = 0.0

@dataclass
class HFEvaluationResult:
    """Data class for storing Hugging Face model test results"""
    test_case_id: str
    llm_model: str
    success: bool
    execution_time: float
    output_quality: float
    memory_usage: float
    tokens_used: int
    errors: List[str]
    user_satisfaction: float
    timestamp: str
    detailed_output: Dict[str, Any]
    quantization: str

class HuggingFaceEvaluator:
    """Main evaluation engine for Hugging Face models"""
    
    def __init__(self, test_suite_path: str, output_dir: str, device: str = "auto"):
        """
        Initialize the Hugging Face evaluator
        
        Args:
            test_suite_path: Path to JSON file containing test cases
            output_dir: Directory to save evaluation results
            device: Device to use (auto, cuda, cpu)
        """
        self.test_suite = self.load_test_suite(test_suite_path)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.results = []
        
        # Device setup
        self.device = self._setup_device(device)
        self.accelerator = Accelerator()
        
        # Model cache
        self.loaded_models = {}
        
        # Performance weights for overall scoring
        self.weights = {
            "success_rate": 0.3,
            "output_quality": 0.25,
            "execution_time": 0.15,
            "memory_efficiency": 0.15,
            "user_satisfaction": 0.15
        }
    
    def _setup_device(self, device: str) -> str:
        """Setup the device for model inference"""
        if device == "auto":
            if torch.cuda.is_available():
                device = "cuda"
                logger.info(f"Using CUDA device: {torch.cuda.get_device_name()}")
            else:
                device = "cpu"
                logger.info("CUDA not available, using CPU")
        return device
    
    def load_test_suite(self, test_suite_path: str) -> List[Dict]:
        """Load test cases from JSON file"""
        try:
            with open(test_suite_path, 'r', encoding='utf-8') as f:
                test_suite = json.load(f)
            logger.info(f"Loaded {len(test_suite)} test cases from {test_suite_path}")
            return test_suite
        except Exception as e:
            logger.error(f"Failed to load test suite: {e}")
            raise
    
    def load_model(self, model_name: str, model_config: Dict[str, Any], 
                  quantization: str = "4bit") -> HFModelInfo:
        """
        Load a Hugging Face model with specified configuration
        
        Args:
            model_name: Name of the model
            model_config: Model configuration
            quantization: Quantization level (4bit, 8bit, 16bit)
            
        Returns:
            HFModelInfo object with loaded model
        """
        model_id = model_config["model_id"]
        cache_key = f"{model_name}_{quantization}"
        
        if cache_key in self.loaded_models:
            logger.info(f"Using cached model: {cache_key}")
            return self.loaded_models[cache_key]
        
        logger.info(f"Loading model: {model_id} with {quantization} quantization")
        
        try:
            # Setup quantization
            bnb_config = None
            if quantization == "4bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_quant_type="nf4",
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True
                )
            elif quantization == "8bit":
                bnb_config = BitsAndBytesConfig(
                    load_in_8bit=True
                )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                model_id,
                trust_remote_code=True,
                use_fast=False
            )
            
            # Add padding token if not present
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "trust_remote_code": True,
                "device_map": "auto" if self.device == "cuda" else None,
                "torch_dtype": torch.float16 if self.device == "cuda" else torch.float32
            }
            
            if bnb_config:
                model_kwargs["quantization_config"] = bnb_config
            
            model = AutoModelForCausalLM.from_pretrained(
                model_id,
                **model_kwargs
            )
            
            # Load processor for multimodal models
            processor = None
            if model_config.get("is_multimodal", False):
                try:
                    processor = AutoProcessor.from_pretrained(
                        model_id,
                        trust_remote_code=True
                    )
                except Exception as e:
                    logger.warning(f"Failed to load processor for {model_name}: {e}")
            
            # Move to device if not using device_map
            if self.device != "cuda" or "device_map" not in model_kwargs:
                model = model.to(self.device)
            
            # Estimate memory usage
            memory_usage = self._estimate_model_memory(model, quantization)
            
            model_info = HFModelInfo(
                model_name=model_name,
                model=model,
                tokenizer=tokenizer,
                processor=processor,
                device=self.device,
                quantization=quantization,
                memory_usage_gb=memory_usage
            )
            
            self.loaded_models[cache_key] = model_info
            logger.info(f"Successfully loaded {model_name} (Memory: {memory_usage:.2f}GB)")
            
            return model_info
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {e}")
            raise
    
    def _estimate_model_memory(self, model, quantization: str) -> float:
        """Estimate memory usage of loaded model"""
        try:
            if hasattr(model, 'get_memory_footprint'):
                memory_bytes = model.get_memory_footprint()
            else:
                # Rough estimation
                param_count = sum(p.numel() for p in model.parameters())
                if quantization == "4bit":
                    memory_bytes = param_count * 0.5  # 4 bits per parameter
                elif quantization == "8bit":
                    memory_bytes = param_count * 1.0  # 8 bits per parameter
                else:
                    memory_bytes = param_count * 2.0  # 16 bits per parameter
            
            return memory_bytes / (1024**3)  # Convert to GB
        except:
            return 0.0
    
    def generate_response(self, model_info: HFModelInfo, prompt: str, 
                         max_length: int = 2048) -> str:
        """
        Generate response using the loaded model
        
        Args:
            model_info: Loaded model information
            prompt: Input prompt
            max_length: Maximum generation length
            
        Returns:
            Generated response text
        """
        try:
            # Prepare input
            inputs = model_info.tokenizer(
                prompt,
                return_tensors="pt",
                truncation=True,
                max_length=model_info.model.config.max_position_embeddings - max_length
            )
            
            # Move to device
            inputs = {k: v.to(model_info.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model_info.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=model_info.tokenizer.eos_token_id,
                    eos_token_id=model_info.tokenizer.eos_token_id
                )
            
            # Decode response
            response = model_info.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating response: {e}")
            return f"Error: {str(e)}"
    
    def generate_multimodal_response(self, model_info: HFModelInfo, 
                                   prompt: str, image_path: str,
                                   max_length: int = 2048) -> str:
        """
        Generate multimodal response for image+text input
        
        Args:
            model_info: Loaded model information
            prompt: Text prompt
            image_path: Path to image file
            max_length: Maximum generation length
            
        Returns:
            Generated response text
        """
        try:
            if model_info.processor is None:
                return "Error: Model does not support multimodal input"
            
            # Load and process image
            from PIL import Image
            image = Image.open(image_path).convert('RGB')
            
            # Prepare multimodal input
            inputs = model_info.processor(
                text=prompt,
                images=image,
                return_tensors="pt"
            )
            
            # Move to device
            inputs = {k: v.to(model_info.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model_info.model.generate(
                    **inputs,
                    max_new_tokens=max_length,
                    temperature=0.3,
                    do_sample=True,
                    pad_token_id=model_info.tokenizer.eos_token_id,
                    eos_token_id=model_info.tokenizer.eos_token_id
                )
            
            # Decode response
            response = model_info.tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            )
            
            return response.strip()
            
        except Exception as e:
            logger.error(f"Error generating multimodal response: {e}")
            return f"Error: {str(e)}"
    
    def evaluate_model(self, model_name: str, model_config: Dict[str, Any],
                      quantization: str = "4bit") -> Dict[str, Any]:
        """
        Evaluate a single Hugging Face model across all test cases
        
        Args:
            model_name: Name of the model to evaluate
            model_config: Model configuration
            quantization: Quantization level
            
        Returns:
            Dictionary containing aggregated results and performance metrics
        """
        logger.info(f"Starting evaluation for model: {model_name} ({quantization})")
        start_time = time.time()
        
        try:
            # Load model
            model_info = self.load_model(model_name, model_config, quantization)
            
            # Run evaluation
            model_results = []
            for i, test_case in enumerate(self.test_suite):
                logger.info(f"Running test case {i+1}/{len(self.test_suite)}: {test_case['id']}")
                result = self.run_single_test(model_info, test_case)
                model_results.append(result)
                
                # Save intermediate results
                if (i + 1) % 5 == 0:
                    self.save_intermediate_results(model_name, model_results, quantization)
            
            # Aggregate results
            aggregated_results = self.aggregate_results(model_name, model_results, quantization)
            
            # Add evaluation metadata
            evaluation_time = time.time() - start_time
            aggregated_results["evaluation_metadata"] = {
                "evaluation_time": evaluation_time,
                "timestamp": time.strftime("%Y-%m-%d %H:%M:%S"),
                "test_suite_size": len(self.test_suite),
                "quantization": quantization,
                "device": self.device,
                "memory_usage_gb": model_info.memory_usage_gb
            }
            
            # Save final results
            self.save_model_results(model_name, aggregated_results, quantization)
            
            logger.info(f"Completed evaluation for {model_name} in {evaluation_time:.2f} seconds")
            return aggregated_results
            
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
    
    def run_single_test(self, model_info: HFModelInfo, test_case: Dict) -> HFEvaluationResult:
        """
        Execute a single test case with the specified model
        
        Args:
            model_info: Loaded model information
            test_case: Test case definition
            
        Returns:
            HFEvaluationResult object with test results
        """
        start_time = time.time()
        timestamp = datetime.now().isoformat()
        
        try:
            # Prepare input based on test case
            input_data = self.prepare_test_input(test_case)
            
            # Generate response
            if isinstance(input_data, tuple) and len(input_data) == 2:
                # Multimodal input
                prompt, image_path = input_data
                response = self.generate_multimodal_response(model_info, prompt, image_path)
            else:
                # Text-only input
                response = self.generate_response(model_info, input_data)
            
            # Evaluate results
            success = self.evaluate_output(response, test_case["expected_output"])
            quality_score = self.assess_output_quality(response, test_case)
            user_satisfaction = self.simulate_user_satisfaction(response, test_case)
            
            execution_time = time.time() - start_time
            
            return HFEvaluationResult(
                test_case_id=test_case["id"],
                llm_model=model_info.model_name,
                success=success,
                execution_time=execution_time,
                output_quality=quality_score,
                memory_usage=model_info.memory_usage_gb,
                tokens_used=len(model_info.tokenizer.encode(response)),
                errors=[],
                user_satisfaction=user_satisfaction,
                timestamp=timestamp,
                detailed_output={"response": response},
                quantization=model_info.quantization
            )
            
        except Exception as e:
            logger.error(f"Error in test case {test_case['id']}: {e}")
            return HFEvaluationResult(
                test_case_id=test_case["id"],
                llm_model=model_info.model_name,
                success=False,
                execution_time=time.time() - start_time,
                output_quality=0.0,
                memory_usage=model_info.memory_usage_gb,
                tokens_used=0,
                errors=[str(e)],
                user_satisfaction=0.0,
                timestamp=timestamp,
                detailed_output={},
                quantization=model_info.quantization
            )
    
    def prepare_test_input(self, test_case: Dict) -> Any:
        """Prepare input data for test case execution"""
        query = test_case["input"].get("query", "")
        image_path = test_case["input"].get("image", "")
        
        if image_path and Path(image_path).exists():
            return query, image_path
        else:
            return query
    
    def evaluate_output(self, output: str, expected_output: Dict) -> bool:
        """Evaluate if the model output meets expected criteria"""
        try:
            if not output or output.startswith("Error:"):
                return False
            
            # Check for expected keywords or patterns
            for key, expected_value in expected_output.items():
                if key == "cell_state":
                    if isinstance(expected_value, list):
                        if not any(state in output.lower() for state in expected_value):
                            return False
                    else:
                        if expected_value.lower() not in output.lower():
                            return False
                
                elif key == "explanation":
                    if len(output) < 50:
                        return False
            
            return True
            
        except Exception as e:
            logger.error(f"Error evaluating output: {e}")
            return False
    
    def assess_output_quality(self, output: str, test_case: Dict) -> float:
        """Assess the quality of model output (0-10 scale)"""
        try:
            if not output or output.startswith("Error:"):
                return 0.0
            
            score = 0.0
            
            # Length and completeness
            if len(output) > 100:
                score += 2.0
            elif len(output) > 50:
                score += 1.0
            
            # Relevance to biological context
            biological_keywords = ["fibroblast", "cell", "state", "activation", "myofibroblast"]
            if any(keyword in output.lower() for keyword in biological_keywords):
                score += 2.0
            
            # Structured output
            if any(marker in output for marker in ["classification:", "result:", "analysis:"]):
                score += 1.0
            
            # Confidence indication
            if "confidence" in output.lower():
                score += 1.0
            
            # Explanation quality
            if len(output.split()) > 20:
                score += 2.0
            
            # Technical accuracy
            if any(term in output.lower() for term in ["dead", "proliferative", "quiescent"]):
                score += 2.0
            
            return min(score, 10.0)
            
        except Exception as e:
            logger.error(f"Error assessing output quality: {e}")
            return 0.0
    
    def simulate_user_satisfaction(self, output: str, test_case: Dict) -> float:
        """Simulate user satisfaction score (0-5 scale)"""
        try:
            if not output or output.startswith("Error:"):
                return 0.0
            
            satisfaction = 0.0
            
            # Clear and understandable response
            if len(output) > 50 and len(output) < 1000:
                satisfaction += 1.0
            
            # Professional tone
            if not any(word in output.lower() for word in ["sorry", "cannot", "unable", "error"]):
                satisfaction += 1.0
            
            # Helpful information
            if any(word in output.lower() for word in ["analysis", "result", "classification", "explanation"]):
                satisfaction += 1.0
            
            # Confidence in response
            if "confidence" in output.lower() or "certain" in output.lower():
                satisfaction += 1.0
            
            # Actionable insights
            if any(word in output.lower() for word in ["recommend", "suggest", "next", "further"]):
                satisfaction += 1.0
            
            return min(satisfaction, 5.0)
            
        except Exception as e:
            logger.error(f"Error simulating user satisfaction: {e}")
            return 0.0
    
    def aggregate_results(self, model_name: str, results: List[HFEvaluationResult], 
                         quantization: str) -> Dict[str, Any]:
        """Aggregate individual test results into model performance metrics"""
        if not results:
            return {}
        
        # Calculate basic metrics
        total_tests = len(results)
        successful_tests = sum(1 for r in results if r.success)
        success_rate = successful_tests / total_tests if total_tests > 0 else 0
        
        execution_times = [r.execution_time for r in results if r.success]
        avg_execution_time = statistics.mean(execution_times) if execution_times else 0
        
        output_qualities = [r.output_quality for r in results]
        avg_output_quality = statistics.mean(output_qualities) if output_qualities else 0
        
        memory_usage = results[0].memory_usage if results else 0
        total_tokens = sum(r.tokens_used for r in results)
        
        error_count = sum(len(r.errors) for r in results)
        error_rate = error_count / total_tests if total_tests > 0 else 0
        
        user_satisfactions = [r.user_satisfaction for r in results]
        avg_user_satisfaction = statistics.mean(user_satisfactions) if user_satisfactions else 0
        
        # Calculate overall performance score
        performance_score = self.calculate_performance_score(
            success_rate, avg_output_quality, avg_execution_time, 
            memory_usage, avg_user_satisfaction
        )
        
        return {
            "model_name": model_name,
            "quantization": quantization,
            "total_tests": total_tests,
            "success_rate": success_rate,
            "avg_execution_time": avg_execution_time,
            "avg_output_quality": avg_output_quality,
            "memory_usage_gb": memory_usage,
            "total_tokens": total_tokens,
            "error_rate": error_rate,
            "avg_user_satisfaction": avg_user_satisfaction,
            "performance_score": performance_score,
            "test_results": [asdict(r) for r in results]
        }
    
    def calculate_performance_score(self, success_rate: float, output_quality: float, 
                                  execution_time: float, memory_usage: float, 
                                  user_satisfaction: float) -> float:
        """Calculate overall performance score (0-100)"""
        # Normalize metrics to 0-1 scale
        normalized_quality = output_quality / 10.0
        normalized_satisfaction = user_satisfaction / 5.0
        
        # Normalize execution time (lower is better, cap at 120 seconds)
        normalized_time = max(0, 1 - (execution_time / 120.0))
        
        # Normalize memory usage (lower is better, cap at 48GB)
        normalized_memory = max(0, 1 - (memory_usage / 48.0))
        
        # Calculate weighted score
        score = (
            self.weights["success_rate"] * success_rate +
            self.weights["output_quality"] * normalized_quality +
            self.weights["execution_time"] * normalized_time +
            self.weights["memory_efficiency"] * normalized_memory +
            self.weights["user_satisfaction"] * normalized_satisfaction
        )
        
        return score * 100  # Convert to 0-100 scale
    
    def compare_models(self, model_results: Dict[str, Dict]) -> Dict[str, Any]:
        """Compare performance across multiple models"""
        comparison = {
            "overall_rankings": {},
            "performance_by_category": {},
            "memory_efficiency": {},
            "recommendations": {}
        }
        
        # Calculate overall scores
        for model_name, results in model_results.items():
            if "performance_score" in results:
                comparison["overall_rankings"][model_name] = results["performance_score"]
        
        # Sort by overall score
        comparison["overall_rankings"] = dict(
            sorted(comparison["overall_rankings"].items(), 
                   key=lambda x: x[1], reverse=True)
        )
        
        # Performance by category
        categories = ["success_rate", "output_quality", "execution_time", "memory_usage_gb", "user_satisfaction"]
        for category in categories:
            comparison["performance_by_category"][category] = {}
            for model_name, results in model_results.items():
                if category in results:
                    comparison["performance_by_category"][category][model_name] = results[category]
        
        # Memory efficiency (performance per GB of memory)
        for model_name, results in model_results.items():
            if results.get("memory_usage_gb", 0) > 0:
                memory_efficiency = results.get("performance_score", 0) / results["memory_usage_gb"]
                comparison["memory_efficiency"][model_name] = memory_efficiency
        
        # Generate recommendations
        comparison["recommendations"] = self.generate_recommendations(model_results)
        
        return comparison
    
    def generate_recommendations(self, model_results: Dict[str, Dict]) -> Dict[str, str]:
        """Generate recommendations based on evaluation results"""
        recommendations = {}
        
        # Find best overall performer
        best_overall = max(model_results.items(), 
                          key=lambda x: x[1].get("performance_score", 0))
        recommendations["best_overall"] = f"Best overall performance: {best_overall[0]}"
        
        # Find most memory efficient
        memory_efficient = min(model_results.items(), 
                             key=lambda x: x[1].get("memory_usage_gb", float('inf')))
        recommendations["most_memory_efficient"] = f"Most memory efficient: {memory_efficient[0]}"
        
        # Find fastest
        fastest = min(model_results.items(), 
                     key=lambda x: x[1].get("avg_execution_time", float('inf')))
        recommendations["fastest"] = f"Fastest execution: {fastest[0]}"
        
        # Find highest quality
        highest_quality = max(model_results.items(), 
                             key=lambda x: x[1].get("avg_output_quality", 0))
        recommendations["highest_quality"] = f"Highest output quality: {highest_quality[0]}"
        
        return recommendations
    
    def save_model_results(self, model_name: str, results: Dict[str, Any], quantization: str):
        """Save results for a single model"""
        output_file = self.output_dir / f"{model_name}_{quantization}_results.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved results for {model_name} to {output_file}")
    
    def save_intermediate_results(self, model_name: str, results: List[HFEvaluationResult], quantization: str):
        """Save intermediate results during evaluation"""
        intermediate_file = self.output_dir / f"{model_name}_{quantization}_intermediate.json"
        with open(intermediate_file, 'w', encoding='utf-8') as f:
            json.dump([asdict(r) for r in results], f, indent=2, ensure_ascii=False)
    
    def save_comparison_results(self, comparison: Dict[str, Any]):
        """Save comparison results"""
        comparison_file = self.output_dir / "hf_model_comparison.json"
        with open(comparison_file, 'w', encoding='utf-8') as f:
            json.dump(comparison, f, indent=2, ensure_ascii=False)
        logger.info(f"Saved comparison results to {comparison_file}")
    
    def cleanup(self):
        """Clean up loaded models to free memory"""
        for model_info in self.loaded_models.values():
            del model_info.model
            del model_info.tokenizer
            if model_info.processor:
                del model_info.processor
        
        self.loaded_models.clear()
        gc.collect()
        
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        
        logger.info("Cleaned up loaded models") 
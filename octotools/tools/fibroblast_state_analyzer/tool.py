#!/usr/bin/env python3
"""
Fibroblast State Analyzer Tool - Analyzes cell state of individual fibroblast crops.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
from PIL import Image
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import matplotlib.pyplot as plt
from huggingface_hub import hf_hub_download

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from octotools.tools.base import BaseTool

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DinoV2Classifier(nn.Module):
    """DINOv2-based classifier for fibroblast state analysis."""
    
    def __init__(self, backbone, num_classes, feat_dim):
        super().__init__()
        self.backbone = backbone
        self.classifier = nn.Linear(feat_dim, num_classes)
        
    def forward(self, x):
        # Get features from backbone
        with torch.no_grad():
            # Use the correct DINOv2 API
            features = self.backbone(x)
            
            # Handle different DINOv2 output formats
            if isinstance(features, dict):
                # Handle different DINOv2 output formats
                if 'last_hidden_state' in features:
                    features = features['last_hidden_state']
                elif 'x_norm_clstoken' in features:
                    features = features['x_norm_clstoken']
                else:
                    # Take the first key if it's a dict
                    features = list(features.values())[0]
            
            # If features is a tensor with multiple dimensions, take the CLS token
            if len(features.shape) == 3:  # [batch_size, seq_len, feat_dim]
                features = features[:, 0, :]  # Take CLS token
            elif len(features.shape) == 2:  # [batch_size, feat_dim]
                features = features
            else:
                raise ValueError(f"Unexpected features shape: {features.shape}")
        
        return self.classifier(features)

class Fibroblast_State_Analyzer_Tool(BaseTool):
    """
    Analyzes fibroblast cell states using a pre-trained DINOv2-based classifier.
    Processes individual cell crops to determine their activation state.
    """
    
    def __init__(self, model_path=None, backbone_size="small", confidence_threshold=0.5):
        super().__init__(
            tool_name="Fibroblast_State_Analyzer_Tool",
            tool_description="Analyzes fibroblast cell states using deep learning to classify individual cells into different activation states.",
            tool_version="1.0.0",
            input_types={
                "cell_crops": "List[str] - Paths to individual cell crop images",
                "cell_metadata": "List[dict] - Metadata for each cell crop",
                "confidence_threshold": "float - Minimum confidence threshold for classification (default: 0.5)",
                "batch_size": "int - Batch size for processing (default: 16)",
                "query_cache_dir": "str - Directory for caching results"
            },
            output_type="dict - Analysis results with cell state classifications and statistics",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(cell_crops=["cell_0001.png", "cell_0002.png"], cell_metadata=[{"cell_id": 1}, {"cell_id": 2}])',
                    "description": "Analyze cell states for individual fibroblast crops"
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for optimal performance. Model accuracy depends on image quality and cell visibility. May struggle with very small or overlapping cells.",
                "best_practice": "Use with high-quality cell crops from Single_Cell_Cropper_Tool. Ensure cells are well-separated and clearly visible in crops.",
                "cell_states": "Classifies cells into: dead, np-MyoFb (non-proliferative myofibroblast), p-MyoFb (proliferative myofibroblast), proto-MyoFb (proto-myofibroblast), q-Fb (quiescent fibroblast)"
            }
        )
        
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Fibroblast_State_Analyzer_Tool: Using device: {self.device}")
        
        # Model configuration
        self.model_path = model_path
        self.backbone_size = backbone_size
        self.confidence_threshold = confidence_threshold
        
        # Cell state classes
        self.class_names = ["dead", "np-MyoFb", "p-MyoFb", "proto-MyoFb", "q-Fb"]
        self.class_descriptions = {
            "dead": "Dead or dying cells",
            "np-MyoFb": "Non-proliferative myofibroblasts",
            "p-MyoFb": "Proliferative myofibroblasts", 
            "proto-MyoFb": "Proto-myofibroblasts",
            "q-Fb": "Quiescent fibroblasts"
        }
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the DINOv2 model and classifier with finetuned weights."""
        try:
            logger.info("Initializing DINOv2 model...")
            
            # Import transformers here to avoid dependency issues
            from transformers import AutoModel
            
            # Load backbone
            backbone = AutoModel.from_pretrained(f"facebook/dinov2-{self.backbone_size}")
            backbone.eval()
            
            # Create classifier with correct feature dimension based on backbone size
            feat_dim_map = {
                "small": 384,
                "base": 768, 
                "large": 1024,
                "giant": 1536
            }
            feat_dim = feat_dim_map.get(self.backbone_size, 768)
            
            logger.info(f"Using {self.backbone_size} backbone with {feat_dim} feature dimensions")
            
            self.model = DinoV2Classifier(
                backbone=backbone,
                num_classes=len(self.class_names),
                feat_dim=feat_dim
            )
            
            # Load finetuned weights from HuggingFace Hub
            try:
                logger.info("Downloading finetuned weights from HuggingFace Hub...")
                
                # Download model weights from the specified repository with authentication
                model_weights_path = hf_hub_download(
                    repo_id="5xuekun/fb-classifier-model",
                    filename="model.pt",
                    cache_dir=None,  # Use default cache
                    token=os.getenv("HUGGINGFACE_TOKEN")  # Add authentication token
                )
                
                logger.info(f"Downloaded model weights to: {model_weights_path}")
                
                # Load the weights
                state_dict = torch.load(model_weights_path, map_location=self.device)
                
                # Handle different state dict formats
                if 'model_state_dict' in state_dict:
                    # If it's a checkpoint with model_state_dict
                    self.model.load_state_dict(state_dict['model_state_dict'])
                    logger.info("Loaded model_state_dict from checkpoint")
                elif 'state_dict' in state_dict:
                    # If it's a checkpoint with state_dict
                    self.model.load_state_dict(state_dict['state_dict'])
                    logger.info("Loaded state_dict from checkpoint")
                else:
                    # If it's just the state dict directly
                    self.model.load_state_dict(state_dict)
                    logger.info("Loaded state dict directly")
                
                logger.info("Successfully loaded finetuned weights from HuggingFace Hub")
                
            except Exception as e:
                logger.warning(f"Failed to load finetuned weights from HuggingFace Hub: {str(e)}")
                
                # Check if it's an authentication error
                if "401" in str(e) or "authentication" in str(e).lower() or "token" in str(e).lower():
                    logger.warning("Authentication error detected. Please set HUGGINGFACE_TOKEN environment variable.")
                    logger.warning("You can get a token from: https://huggingface.co/settings/tokens")
                    logger.warning("Example: export HUGGINGFACE_TOKEN=your_token_here")
                elif "Repository Not Found" in str(e):
                    logger.warning("Repository not found. The model may be private or the repository ID may be incorrect.")
                    logger.warning("Please check the repository access or contact the model owner.")
                
                logger.warning("Using untrained classifier - results may not be accurate")
                logger.warning("For best results, please provide a local model path or set up HuggingFace authentication")
                
                # Try to load from local path if provided
                if self.model_path and os.path.exists(self.model_path):
                    logger.info(f"Attempting to load from local path: {self.model_path}")
                    try:
                        state_dict = torch.load(self.model_path, map_location=self.device)
                        if 'model_state_dict' in state_dict:
                            self.model.load_state_dict(state_dict['model_state_dict'])
                        else:
                            self.model.load_state_dict(state_dict)
                        logger.info("Successfully loaded from local path")
                    except Exception as local_e:
                        logger.error(f"Failed to load from local path: {str(local_e)}")
                else:
                    logger.info("No local model path provided. Using untrained classifier.")
                    logger.info("To use a trained model, either:")
                    logger.info("1. Set HUGGINGFACE_TOKEN environment variable for remote access")
                    logger.info("2. Provide a local model_path parameter")
            
            self.model = self.model.to(self.device)
            self.model.eval()
            
            # Define transforms for preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])
            
            logger.info("Model initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _is_model_trained(self) -> bool:
        """Check if the model has been trained by examining classifier weights."""
        try:
            # Check if classifier weights are initialized with non-zero values
            classifier_weights = self.model.classifier.weight.data
            classifier_bias = self.model.classifier.bias.data
            
            # If weights are all close to zero, the model is likely untrained
            weights_norm = torch.norm(classifier_weights).item()
            bias_norm = torch.norm(classifier_bias).item()
            
            # Threshold for considering weights as "trained"
            threshold = 0.1
            return weights_norm > threshold or bias_norm > threshold
            
        except Exception as e:
            logger.warning(f"Could not determine if model is trained: {str(e)}")
            return False
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image for model input."""
        try:
            # Load and convert to RGB
            img = Image.open(image_path).convert("RGB")
            
            # Apply transforms
            img_tensor = self.transform(img).unsqueeze(0)
            return img_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def _classify_single_cell(self, image_path: str) -> Dict[str, Any]:
        """Classify a single cell image."""
        try:
            # Preprocess image
            img_tensor = self._preprocess_image(image_path)
            
            # Get predictions
            with torch.no_grad():
                logits = self.model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0][pred_idx].item()
            
            # Create result
            result = {
                "image_path": image_path,
                "predicted_class": self.class_names[pred_idx],
                "confidence": confidence,
                "all_probabilities": {
                    cls_name: prob.item() 
                    for cls_name, prob in zip(self.class_names, probs[0])
                }
            }
            
            return result
            
        except Exception as e:
            logger.error(f"Error classifying cell {image_path}: {str(e)}")
            return {
                "image_path": image_path,
                "predicted_class": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }
    
    def execute(self, cell_crops: List[str], cell_metadata: Optional[List[Dict]] = None, 
                confidence_threshold: Optional[float] = None, batch_size: int = 16, 
                query_cache_dir: Optional[str] = None) -> Dict[str, Any]:
        """
        Execute fibroblast state analysis on cell crops.
        
        Args:
            cell_crops: List of paths to cell crop images
            cell_metadata: Optional metadata for each cell
            confidence_threshold: Minimum confidence for classification
            batch_size: Batch size for processing
            query_cache_dir: Directory for caching results
            
        Returns:
            Dict containing analysis results and statistics
        """
        try:
            logger.info(f"Starting fibroblast state analysis for {len(cell_crops)} cells")
            
            # Check if model is trained and warn if not
            if not self._is_model_trained():
                logger.warning("⚠️  WARNING: Using untrained model. Results may not be accurate!")
                logger.warning("To get accurate results, please:")
                logger.warning("1. Set HUGGINGFACE_TOKEN environment variable for remote model access")
                logger.warning("2. Provide a local model_path parameter with trained weights")
                logger.warning("3. Contact the model owner for access to the trained model")
            
            # Use provided confidence threshold or default
            threshold = confidence_threshold if confidence_threshold is not None else self.confidence_threshold
            
            # Setup output directory
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
            os.makedirs(tool_cache_dir, exist_ok=True)
            
            # Process each cell
            results = []
            valid_results = []
            failed_cells = []
            
            for i, crop_path in enumerate(cell_crops):
                try:
                    # Validate file exists
                    if not os.path.exists(crop_path):
                        logger.warning(f"Cell crop not found: {crop_path}")
                        failed_cells.append({"path": crop_path, "error": "File not found"})
                        continue
                    
                    # Classify cell
                    result = self._classify_single_cell(crop_path)
                    
                    # Add metadata if available
                    if cell_metadata and i < len(cell_metadata):
                        result.update(cell_metadata[i])
                    
                    results.append(result)
                    
                    # Check confidence threshold
                    if result.get("confidence", 0) >= threshold:
                        valid_results.append(result)
                    else:
                        logger.warning(f"Low confidence ({result.get('confidence', 0):.3f}) for {crop_path}")
                    
                    # Progress logging
                    if (i + 1) % 50 == 0:
                        logger.info(f"Processed {i + 1}/{len(cell_crops)} cells")
                        
                except Exception as e:
                    logger.error(f"Error processing cell {crop_path}: {str(e)}")
                    failed_cells.append({"path": crop_path, "error": str(e)})
                    continue
            
            # Calculate statistics
            stats = self._calculate_statistics(valid_results)
            
            # Create visualizations
            viz_paths = self._create_visualizations(valid_results, stats, tool_cache_dir)
            
            # Save detailed results
            results_path = os.path.join(tool_cache_dir, f"fibroblast_analysis_results_{uuid4().hex[:8]}.json")
            with open(results_path, 'w') as f:
                json.dump({
                    "all_results": results,
                    "valid_results": valid_results,
                    "statistics": stats,
                    "failed_cells": failed_cells,
                    "parameters": {
                        "confidence_threshold": threshold,
                        "total_cells": len(cell_crops),
                        "valid_cells": len(valid_results),
                        "failed_cells": len(failed_cells)
                    }
                }, f, indent=2)
            
            # Clear CUDA cache if using GPU
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            return {
                "summary": f"Analyzed {len(valid_results)}/{len(cell_crops)} cells with confidence ≥ {threshold}",
                "total_cells": len(cell_crops),
                "valid_cells": len(valid_results),
                "failed_cells": len(failed_cells),
                "cell_state_distribution": stats["class_distribution"],
                "average_confidence": stats["average_confidence"],
                "visual_outputs": viz_paths + [results_path],
                "parameters": {
                    "confidence_threshold": threshold,
                    "backbone_size": self.backbone_size,
                    "model_path": self.model_path
                }
            }
            
        except Exception as e:
            logger.error(f"Error in fibroblast state analysis: {str(e)}")
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            return {
                "error": f"Error during fibroblast state analysis: {str(e)}",
                "summary": "Failed to analyze cell states"
            }
    
    def _calculate_statistics(self, results: List[Dict]) -> Dict[str, Any]:
        """Calculate statistics from classification results."""
        if not results:
            return {
                "class_distribution": {},
                "average_confidence": 0.0,
                "total_cells": 0
            }
        
        # Class distribution
        class_counts = {}
        total_confidence = 0.0
        
        for result in results:
            predicted_class = result.get("predicted_class", "unknown")
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
            total_confidence += result.get("confidence", 0.0)
        
        # Calculate percentages
        total_cells = len(results)
        class_distribution = {
            class_name: {
                "count": count,
                "percentage": (count / total_cells) * 100
            }
            for class_name, count in class_counts.items()
        }
        
        return {
            "class_distribution": class_distribution,
            "average_confidence": total_confidence / total_cells,
            "total_cells": total_cells
        }
    
    def _create_visualizations(self, results: List[Dict], stats: Dict, output_dir: str) -> List[str]:
        """Create visualizations of analysis results."""
        viz_paths = []
        
        try:
            # 1. Class distribution pie chart
            if stats["class_distribution"]:
                fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 6))
                
                # Pie chart
                classes = list(stats["class_distribution"].keys())
                counts = [stats["class_distribution"][cls]["count"] for cls in classes]
                percentages = [stats["class_distribution"][cls]["percentage"] for cls in classes]
                
                colors = plt.cm.Set3(np.linspace(0, 1, len(classes)))
                wedges, texts, autotexts = ax1.pie(counts, labels=classes, autopct='%1.1f%%', 
                                                   colors=colors, startangle=90)
                ax1.set_title("Cell State Distribution")
                
                # Bar chart
                ax2.bar(classes, counts, color=colors)
                ax2.set_title("Cell Count by State")
                ax2.set_ylabel("Number of Cells")
                ax2.tick_params(axis='x', rotation=45)
                
                # Add percentage labels on bars
                for i, (cls, count, pct) in enumerate(zip(classes, counts, percentages)):
                    ax2.text(i, count + 0.5, f'{pct:.1f}%', ha='center', va='bottom')
                
                plt.tight_layout()
                
                # Save visualization
                viz_path = os.path.join(output_dir, f"cell_state_distribution_{uuid4().hex[:8]}.png")
                plt.savefig(viz_path, bbox_inches='tight', dpi=150, format='png')
                plt.close()
                viz_paths.append(viz_path)
            
            # 2. Confidence distribution histogram
            if results:
                confidences = [r.get("confidence", 0) for r in results]
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', edgecolor='black')
                ax.set_xlabel("Confidence Score")
                ax.set_ylabel("Number of Cells")
                ax.set_title("Confidence Distribution")
                ax.axvline(self.confidence_threshold, color='red', linestyle='--', 
                          label=f'Threshold ({self.confidence_threshold})')
                ax.legend()
                ax.grid(True, alpha=0.3)
                
                # Save visualization
                conf_viz_path = os.path.join(output_dir, f"confidence_distribution_{uuid4().hex[:8]}.png")
                plt.savefig(conf_viz_path, bbox_inches='tight', dpi=150, format='png')
                plt.close()
                viz_paths.append(conf_viz_path)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
        
        return viz_paths
    
    def get_metadata(self):
        """Returns the metadata for the Fibroblast_State_Analyzer_Tool."""
        metadata = super().get_metadata()
        metadata.update({
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "is_model_trained": self._is_model_trained(),
            "backbone_size": self.backbone_size,
            "class_names": self.class_names,
            "class_descriptions": self.class_descriptions
        })
        return metadata


if __name__ == "__main__":
    # Test the tool
    print("Testing Fibroblast_State_Analyzer_Tool...")
    
    # Initialize tool
    tool = Fibroblast_State_Analyzer_Tool()
    
    # Get metadata
    metadata = tool.get_metadata()
    print("Tool Metadata:")
    print(json.dumps(metadata, indent=2))
    
    print("\nTool initialized successfully!")
    print("Example usage:")
    print("execution = tool.execute(cell_crops=['cell_0001.png', 'cell_0002.png'])") 
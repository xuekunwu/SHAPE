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
import argparse
import time
from sklearn.decomposition import PCA
import glob
import pandas as pd
from collections import Counter

# Configure logging first
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import anndata and scanpy for advanced visualizations
try:
    import anndata
    import scanpy as sc
    ANNDATA_AVAILABLE = True
    logger.info("anndata and scanpy are available for advanced visualizations")
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("anndata and scanpy not available. Using PCA for visualization only.")

# Add the project root to the Python path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from octotools.tools.base import BaseTool

class DinoV2Classifier(nn.Module):
    """
    Wrapper for the DINOv2 model with a custom classifier head.
    This matches the architecture used in the training notebook.
    """
    def __init__(self, backbone, num_classes):
        super().__init__()
        self.backbone = backbone
        self.num_classes = num_classes
        
    def forward(self, x):
        return self.backbone(x)

class Fibroblast_State_Analyzer_Tool(BaseTool):
    """
    Analyzes fibroblast cell states using a pre-trained DINOv2-based classifier.
    Processes individual cell crops to determine their activation state.
    """
    
    def __init__(self, model_path=None, backbone_size="large", confidence_threshold=0.5):
        super().__init__(
            tool_name="Fibroblast_State_Analyzer_Tool",
            tool_description="Classifies fibroblast cell states from single-cell images using a DINOv2 backbone and a custom classifier head. Provides detailed statistics and visualizations.",
            tool_version="1.0.0",
            input_types={
                "cell_crops": "List[str] - List of cell crop image paths or PIL Images.",
                "cell_metadata": "List[dict] - List of metadata dictionaries for each cell.",
                "batch_size": "int - Batch size for processing (default: 16).",
                "query_cache_dir": "str - Directory for caching results (default: 'solver_cache').",
                "visualization_type": "str - Type of visualization ('pca', 'umap', 'auto', 'all')."
            },
            output_type="dict - Analysis results with classifications and statistics.",
            user_metadata={
                "limitation": "Requires GPU for optimal performance. Model accuracy depends on image quality and cell visibility. May struggle with very small or overlapping cells.",
                "best_practice": "Use with high-quality cell crops from Single_Cell_Cropper_Tool. Ensure cells are well-separated and clearly visible in crops. For best results, use visualization_type='all' to get comprehensive visualizations including UMAP.",
                "cell_states": "Classifies cells into: dead, np-MyoFb (non-proliferative myofibroblast), p-MyoFb (proliferative myofibroblast), proto-MyoFb (proto-myofibroblast), q-Fb (quiescent fibroblast)",
                "visualization": "Supports comprehensive visualizations: UMAP (shows cell clustering and distribution), PCA (fast, interpretable), confidence distributions, and cell state bar charts. UMAP is particularly useful for understanding cell relationships and identifying clusters. Use visualization_type='all' for complete analysis.",
                "umap_benefits": "UMAP visualization provides superior insights into cell distribution patterns, helping identify clusters, outliers, and spatial relationships between different cell states in the high-dimensional feature space."
            }
        )
        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Fibroblast_State_Analyzer_Tool: Using device: {self.device}")
        # Model configuration
        self.model_path = model_path
        self.backbone_size = backbone_size
        self.confidence_threshold = confidence_threshold  # 保留参数但不再用于过滤
        
        # Backbone configuration constants
        self.backbone_archs = {
            "small": "vits14",
            "base": "vitb14", 
            "large": "vitl14",
            "giant": "vitg14",
        }
        self.feat_dim_map = {
            "vits14": 384,
            "vitb14": 768,
            "vitl14": 1024,
            "vitg14": 1536
        }
        
        # Cell state classes with specific colors
        self.class_names = ["dead", "np-MyoFb", "p-MyoFb", "proto-MyoFb", "q-Fb"]
        self.class_descriptions = {
            "dead": "Dead cells",
            "np-MyoFb": "Non-proliferative myofibroblasts",
            "p-MyoFb": "Proliferative myofibroblasts", 
            "proto-MyoFb": "Proto-myofibroblasts",
            "q-Fb": "Quiescent fibroblasts"
        }
        
        # Color mapping for visualizations
        self.color_map = {
            'dead': '#808080', 
            'np-MyoFb': '#A65A9F', 
            'p-MyoFb': '#D6B8D8', 
            'proto-MyoFb': '#F8BD6F', 
            'q-Fb': '#66B22F'
        }
        
        # Lazy initialization - don't load model until needed
        self.model = None
        self.transform = None
        self._model_initialized = False
        
    def _initialize_model(self):
        """Initialize the DINOv2 model and classifier with finetuned weights."""
        # Check if model is already initialized
        if self._model_initialized and self.model is not None:
            logger.info("Model already initialized, skipping...")
            return
            
        try:
            logger.info("Initializing DINOv2 model (torch.hub)...")
            
            # Define backbone architecture based on size
            backbone_arch = self.backbone_archs.get(self.backbone_size, "vitl14")
            backbone_name = f"dinov2_{backbone_arch}"

            # Load the DINOv2 backbone using torch.hub to match the training environment
            backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
            backbone_model.eval() # Keep backbone frozen
            
            # Create the model wrapper
            self.model = DinoV2Classifier(
                backbone=backbone_model,
                num_classes=len(self.class_names)
            ).to(self.device)

            # Set classifier head to Sequential structure as in training
            feat_dim = self.feat_dim_map[backbone_arch]
            hidden_dim = feat_dim // 2
            self.model.backbone.head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0),  # no drop for inference
                nn.Linear(hidden_dim, len(self.class_names))
            ).to(self.device)
            logger.info(f"Using {backbone_name} backbone with Sequential classifier head.")
            logger.info(f"Classifier head: {self.model.backbone.head}")
            logger.info(f"Expected output classes: {len(self.class_names)}")
            
            # Load finetuned weights from HuggingFace Hub or local path
            model_loaded = False
            
            # 只允许从 HuggingFace Hub 下载权重，不支持本地权重，也不允许无权重
            try:
                logger.info("Attempting to download model from HuggingFace Hub...")
                model_weights_path = hf_hub_download(
                    repo_id="5xuekun/fb-classifier-model",
                    filename="model.pt",
                    token=os.getenv("HUGGINGFACE_TOKEN")
                )
                logger.info(f"Downloaded model weights to: {model_weights_path}")

                checkpoint = torch.load(model_weights_path, map_location=self.device)
                if 'model_state_dict' in checkpoint:
                    self.model.load_state_dict(checkpoint['model_state_dict'])
                else:
                    self.model.load_state_dict(checkpoint)
                self.model.to(self.device)
                self.model.eval()
                logger.info("Successfully loaded finetuned weights from HuggingFace Hub")
                model_loaded = True

            except Exception as e:
                logger.error(f"Failed to load weights from HuggingFace Hub: {str(e)}")
                raise RuntimeError(f"Failed to load model weights from HuggingFace Hub: {e}")

            if not model_loaded:
                logger.info("Using untrained classifier head")

            # Verify model structure after loading
            if isinstance(self.model.backbone.head, nn.Sequential):
                output_dim = self.model.backbone.head[-1].out_features
            else:
                output_dim = self.model.backbone.head.out_features
            logger.info(f"Model loaded successfully. Classifier head output dimension: {output_dim}")
            logger.info(f"Expected number of classes: {len(self.class_names)}")
            
            # Test model with dummy input to verify output
            with torch.no_grad():
                dummy_input = torch.randn(1, 3, 224, 224).to(self.device)
                test_output = self.model(dummy_input)
                logger.info(f"Test output shape: {test_output.shape}")
                logger.info(f"Test output classes: {test_output.shape[1]}")
                
                if test_output.shape[1] != len(self.class_names):
                    logger.error(f"Model output dimension mismatch! Expected {len(self.class_names)}, got {test_output.shape[1]}")
                else:
                    logger.info("Model output dimension matches expected number of classes")
            
            # Define transforms for preprocessing - 与训练时完全一致
            self.transform = transforms.Compose([
                transforms.Resize(224, interpolation=transforms.InterpolationMode.BICUBIC),
                transforms.CenterCrop(224),  # Optional but recommended
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.4636, 0.5032, 0.5822], std=[0.2483, 0.2660, 0.2907]),
            ])
            
            # Mark model as initialized
            self._model_initialized = True
            logger.info("Model initialization completed successfully")
            
        except Exception as e:
            logger.error(f"Error initializing model: {str(e)}")
            raise
    
    def _is_model_trained(self) -> bool:
        """Check if the model has been trained by examining classifier weights."""
        try:
            # Check the norm of the last layer's weights
            if isinstance(self.model.backbone.head, nn.Sequential):
                final_layer_weights = self.model.backbone.head[-1].weight.data
            else:
                final_layer_weights = self.model.backbone.head.weight.data
            weights_norm = torch.norm(final_layer_weights).item()
            return weights_norm > 0.1 # A simple heuristic
        except Exception as e:
            logger.warning(f"Could not determine if model is trained: {str(e)}")
            return False
    
    def _preprocess_image(self, image_path: str) -> torch.Tensor:
        """Preprocess a single image for model input."""
        try:
            if image_path.lower().endswith('.tif') or image_path.lower().endswith('.tiff'):
                from skimage import io
                image = io.imread(image_path).astype(np.float32)
                
                if image.dtype == np.uint16:
                    image = image / 65535.0  # 16-bit normalization
                else:
                    image = image / 255.0    # 8-bit normalization
                
                if len(image.shape) == 2:
                    image = np.repeat(image[:, :, np.newaxis], 3, axis=-1)  # (H, W) → (H, W, 3)
                
                image = Image.fromarray((image * 255).astype(np.uint8)).convert("RGB")
            else:
                image = Image.open(image_path).convert("RGB")
            
            img_tensor = self.transform(image)
            
            img_tensor = torch.clamp(img_tensor, 0.0, 1.0)
            
            img_tensor = img_tensor.unsqueeze(0)
            return img_tensor.to(self.device)
            
        except Exception as e:
            logger.error(f"Error preprocessing image {image_path}: {str(e)}")
            raise
    
    def _classify_single_cell(self, image_path: str, model: nn.Module, confidence_threshold: float) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        """Classify a single cell image and return result with features."""
        try:
            # Preprocess image
            img_tensor = self._preprocess_image(image_path)
            # Get predictions and features
            with torch.no_grad():
                # Get logits from the model
                logits = model(img_tensor)
                # Extract features from backbone (before classifier head)
                backbone_output = model.backbone.forward_features(img_tensor)
                if isinstance(backbone_output, dict):
                    features = backbone_output['x_norm_clstoken']  # Shape: [1, feat_dim]
                else:
                    features = backbone_output
                    if features.dim() > 2:
                        features = features[:, 0, :]
                probs = torch.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0][pred_idx].item()
                # Debug: Log all probabilities for verification
                logger.debug(f"Classification for {os.path.basename(image_path)}:")
                logger.debug(f"Logits: {logits.squeeze().cpu().numpy()}")
                logger.debug(f"Probabilities: {probs.squeeze().cpu().numpy()}")
                logger.debug(f"Predicted class: {self.class_names[pred_idx]} (index: {pred_idx})")
                logger.debug(f"Confidence: {confidence:.4f}")
            # Create result
            result = {
                "image_path": image_path,
                "predicted_class": self.class_names[pred_idx],
                "confidence": confidence,
                "features": features.squeeze(0).cpu().numpy(),
                "all_probabilities": probs.squeeze(0).cpu().numpy().tolist(),
                "all_logits": logits.squeeze(0).cpu().numpy().tolist()
            }
            return result, features.squeeze(0)
        except Exception as e:
            logger.error(f"Error classifying cell {image_path}: {str(e)}")
            return {
                "image_path": image_path,
                "predicted_class": "unknown",
                "confidence": 0.0,
                "error": str(e)
            }, None
    
    def execute(self, cell_crops=None, cell_metadata=None, batch_size=16, query_cache_dir="solver_cache", visualization_type='auto'):
        """
        Execute fibroblast state analysis on cell crops.
        
        Args:
            cell_crops: List of cell crop image paths or PIL Images
            cell_metadata: List of metadata dictionaries for each cell
            batch_size: Batch size for processing
            query_cache_dir: Directory for caching results
            visualization_type: Type of visualization ('pca', 'umap', 'auto', 'all')
            
        Returns:
            dict: Analysis results with classifications and statistics
        """
        print(f"🚀 Fibroblast_State_Analyzer_Tool starting execution...")
        print(f"📊 Parameters: batch_size={batch_size}, visualization_type={visualization_type}")
        
        # Load cell data if not provided
        if cell_crops is None or cell_metadata is None:
            print(f"📁 Loading cell data from metadata in: {query_cache_dir}")
            cell_crops, cell_metadata = self._load_cell_data_from_metadata(query_cache_dir)
        
        if not cell_crops or len(cell_crops) == 0:
            return {"error": "No cell crops found for analysis", "status": "failed"}
        
        print(f"🔬 Processing {len(cell_crops)} cell crops...")
        
        try:
            # Ensure cache directory exists
            os.makedirs(query_cache_dir, exist_ok=True)
            
            # Normalize file paths to handle Windows/Unix path differences
            cell_crops = [os.path.normpath(crop_path) for crop_path in cell_crops]
            
            # Verify all crop files exist
            missing_files = [crop for crop in cell_crops if not os.path.exists(crop)]
            if missing_files:
                return {
                    "error": f"Missing crop files: {missing_files[:5]}... (showing first 5)",
                    "status": "failed",
                    "debug_info": {
                        "total_crops": len(cell_crops),
                        "missing_count": len(missing_files),
                        "cache_dir": query_cache_dir
                    }
                }
            
            print(f"Processing {len(cell_crops)} cell crops...")
            
            # Load model
            self._initialize_model()
            model = self.model
            if model is None:
                return {"error": "Failed to load model", "status": "failed"}
            
            # Process cells
            results = []
            features_list = []
            
            print(f"Starting to process {len(cell_crops)} cell crops...")
            
            # Batch processing loop
            for i in range(0, len(cell_crops), batch_size):
                batch_paths = cell_crops[i:i+batch_size]
                valid_paths_in_batch = []
                img_tensors = []
                for path in batch_paths:
                    try:
                        img_tensors.append(self._preprocess_image(path).squeeze(0))
                        valid_paths_in_batch.append(path)
                    except Exception as e:
                        logger.warning(f"Skipping problematic crop {path}: {e}")
                        continue
                if not img_tensors:
                    logger.warning(f"Skipping empty batch from index {i}")
                    continue
                batch_tensor = torch.stack(img_tensors)
                with torch.no_grad():
                    logits = self.model(batch_tensor)
                    probs = F.softmax(logits, dim=1)
                    confidences, predictions = torch.max(probs, dim=1)
                    backbone_output = self.model.backbone.forward_features(batch_tensor)
                    if isinstance(backbone_output, dict):
                        features = backbone_output['x_norm_clstoken']
                    else:
                        features = backbone_output
                        if features.dim() > 2:
                            features = features[:, 0, :]
                for j, path in enumerate(valid_paths_in_batch):
                    if j < len(predictions):
                        pred_index = predictions[j].item()
                        confidence = confidences[j].item()
                        feature_vector = features[j].cpu().numpy()
                        result = {
                            "image_path": path,
                            "predicted_class": self.class_names[pred_index],
                            "confidence": confidence,
                            "features": feature_vector
                        }
                        results.append(result)
                        features_list.append(feature_vector)

            print(f"Processing completed. Total results: {len(results)}, Total features: {len(features_list)}")
            
            if not results:
                return {"error": "No cells were successfully analyzed", "status": "failed"}
            
            # Calculate statistics
            summary = self._calculate_statistics(results)
            
            # Initialize visual_outputs and recommendations
            visual_outputs = []
            recommendations = {}
            
            # Generate final summary and visualizations
            if len(features_list) > 0:
                logger.info("Generating final summary and visualizations...")
                print(f"🔍 Debug: Initial features_list length: {len(features_list)}")
                
                # Safely extract features from results that definitely have them
                results_with_features = [r for r in results if 'features' in r]
                print(f"🔍 Debug: Results with features: {len(results_with_features)} out of {len(results)}")
                
                if len(results_with_features) == 0:
                    logger.warning("No features were successfully extracted, skipping visualizations.")
                    print("❌ No results have features, skipping visualizations")
                    # Return partial results without visualizations
                    return {
                        "summary": "Analysis completed, but no features were successfully extracted for visualization.",
                        "cell_count": len(results),
                        "visual_outputs": [],
                        "statistics": self._calculate_statistics(results)
                    }

                # Use the original features_list instead of recreating it
                print(f"🔍 Debug: Using original features_list with length: {len(features_list)}")
                summary = self._calculate_statistics(results)

                # Save results and get persistent output dir
                persistent_output_dir = self._save_results(
                    results, summary, query_cache_dir
                )
                print(f"📁 Saving all visualizations to: {persistent_output_dir}")
                
                # Convert features_list to numpy array for visualization
                features_array = np.array(features_list)
                print(f"🔍 Debug: Converted features_list to array with shape: {features_array.shape}")
                
                visual_outputs = self._create_visualizations(results, features_array, summary, persistent_output_dir)
                print(f"📊 Visualizations created: {len(visual_outputs)}")
                
                # Add recommendations and quality assessment
                recommendations = self._generate_recommendations(summary, len(cell_crops), len(results))
            else:
                print(f"❌ No features available for advanced visualizations. features_list length: {len(features_list) if features_list else 0}")
                print(f"🔍 Debug: results length = {len(results) if results else 0}")
                # Set default recommendations for cases without features
                recommendations = {"note": "No features available for advanced analysis"}
            
            # After all results are collected, print class distribution for debug
            print("Predicted class distribution:", Counter([r['predicted_class'] for r in results]))
            
            # Check for class distribution issues
            class_counts = Counter([r['predicted_class'] for r in results])
            unique_classes = len(class_counts)
            total_cells = len(results)
            
            logger.info(f"Classification summary: {unique_classes} unique classes out of {len(self.class_names)} possible classes")
            logger.info(f"Class distribution: {dict(class_counts)}")
            
            # Collect warnings for the final result
            warnings = []
            
            # Warn if not all classes are represented
            if unique_classes < len(self.class_names):
                missing_classes = set(self.class_names) - set(class_counts.keys())
                logger.warning(f"Not all classes are represented in predictions!")
                logger.warning(f"Missing classes: {missing_classes}")
                logger.warning(f"This might indicate: 1) Limited test data diversity, 2) Model bias, 3) Training data imbalance")
                
                # Add warning to warnings list
                warnings.append({
                    "type": "class_distribution",
                    "message": f"Only {unique_classes}/{len(self.class_names)} classes predicted. Missing: {missing_classes}",
                    "suggestion": "Consider using more diverse test data or checking model training balance"
                })
            
            # Check for extreme class imbalance
            if unique_classes > 1:
                max_count = max(class_counts.values())
                min_count = min(class_counts.values())
                imbalance_ratio = max_count / min_count if min_count > 0 else float('inf')
                
                if imbalance_ratio > 10:  # If most common class is 10x more frequent than least common
                    logger.warning(f"Extreme class imbalance detected: ratio = {imbalance_ratio:.1f}")
                    logger.warning(f"Most common: {max_count} cells, Least common: {min_count} cells")
                    
                    warnings.append({
                        "type": "class_imbalance",
                        "message": f"Extreme class imbalance detected: ratio = {imbalance_ratio:.1f}",
                        "suggestion": "Consider using class-balanced training or data augmentation"
                    })
            
            # Build AnnData object for downstream activation scoring (strictly follow user format)
            X = np.array(features_list)
            image_names = [os.path.basename(r['image_path']) for r in results]
            predicted_classes = [r['predicted_class'] for r in results]
            groups = [r.get('group', '') for r in results]
            cell_ids = [f"cell_{i}" for i in range(len(results))]
            obs_dict = {
                'predicted_class': predicted_classes,
                'image_name': image_names,
                'group': groups,
                'cell_id': cell_ids
            }
            import pandas as pd
            obs_df = pd.DataFrame(obs_dict)
            obs_df.index = cell_ids  # obs_names
            import anndata
            adata = anndata.AnnData(X=X, obs=obs_df)
            adata.obs_names = cell_ids
            try:
                from octotools.models.utils import VisualizationConfig
                output_dir = VisualizationConfig.get_output_dir(query_cache_dir)
                h5ad_path = os.path.join(output_dir, "fibroblast_state_analyzed.h5ad")
                adata.write(h5ad_path)
                print(f"💾 AnnData saved to: {h5ad_path}")
                tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
                os.makedirs(tool_cache_dir, exist_ok=True)
                tool_cache_h5ad_path = os.path.join(tool_cache_dir, "fibroblast_state_analyzed.h5ad")
                adata.write(tool_cache_h5ad_path)
                print(f"💾 AnnData also saved to tool cache: {tool_cache_h5ad_path}")
            except Exception as e:
                print(f"⚠️ Warning: Failed to save AnnData as h5ad file: {str(e)}")
                h5ad_path = None
            
            return {
                "summary": summary,
                "cell_state_distribution": self._get_state_distribution(results),
                "visual_outputs": visual_outputs,
                "parameters": {
                    "backbone_size": "large",
                    "model_path": self.model_path
                },
                "cell_state_descriptions": self.class_descriptions,
                "recommendations": recommendations,
                "metadata_info": {
                    "total_crops_processed": len(cell_crops),
                    "successful_analyses": len(results),
                    "metadata_files_used": self._list_metadata_files(query_cache_dir)
                },
                "adata": adata,
                "h5ad_path": h5ad_path,
                "warnings": warnings
            }
            
        except Exception as e:
            return {"error": f"Analysis failed: {str(e)}", "status": "failed"}

    def _load_cell_data_from_metadata(self, cache_dir):
        """
        Load cell crops and metadata from the most recent metadata file.
        Enhanced with file protection and backup functionality.
        
        Args:
            cache_dir: Directory containing metadata files
            
        Returns:
            Tuple of (cell_crops, cell_metadata)
        """
        try:
            # Find all metadata files in cache_dir and its subdirectories
            metadata_files = []
            
            # Search in the main cache directory
            metadata_files.extend(glob.glob(os.path.join(cache_dir, 'cell_crops_metadata_*.json')))
            
            # Search in tool_cache subdirectory (where single_cell_cropper saves files)
            tool_cache_dir = os.path.join(cache_dir, 'tool_cache')
            if os.path.exists(tool_cache_dir):
                metadata_files.extend(glob.glob(os.path.join(tool_cache_dir, 'cell_crops_metadata_*.json')))
            
            # Search recursively in all subdirectories
            for root, dirs, files in os.walk(cache_dir):
                for file in files:
                    if file.startswith('cell_crops_metadata_') and file.endswith('.json'):
                        metadata_files.append(os.path.join(root, file))
            
            if not metadata_files:
                print(f"No metadata files found in {cache_dir} or its subdirectories")
                return [], []
            
            # Sort by modification time (newest first)
            metadata_files.sort(key=os.path.getmtime, reverse=True)
            latest_file = metadata_files[0]
            
            print(f"Using metadata file: {latest_file}")
            
            # Create backup of the metadata file before processing
            backup_path = self._create_metadata_backup(latest_file)
            if backup_path:
                print(f"Created backup: {backup_path}")
            
            with open(latest_file, 'r') as f:
                metadata = json.load(f)
            
            # Handle different metadata formats
            if isinstance(metadata, list):
                cell_metadata_list = metadata
            elif isinstance(metadata, dict):
                # Try common keys first
                if 'cell_metadata' in metadata:
                    cell_metadata_list = metadata['cell_metadata']
                elif 'crops' in metadata:
                    cell_metadata_list = metadata['crops']
                elif 'cell_crops' in metadata:
                    cell_metadata_list = metadata['cell_crops']
                elif 'cell_crops_paths' in metadata:
                    # Handle single_cell_cropper format
                    cell_crops_paths = metadata['cell_crops_paths']
                    cell_metadata_list = metadata.get('cell_metadata', [])
                    
                    # If we have paths but no metadata, create basic metadata
                    if cell_crops_paths and not cell_metadata_list:
                        cell_metadata_list = [{'crop_path': path, 'cell_id': i} for i, path in enumerate(cell_crops_paths)]
                else:
                    # Try to find any list in the dictionary that contains crop data
                    cell_metadata_list = None
                    for key, value in metadata.items():
                        if isinstance(value, list) and len(value) > 0:
                            # Check if this list contains crop data
                            if isinstance(value[0], dict):
                                # Look for common crop-related keys
                                if any(k in value[0] for k in ['crop_path', 'cell_id', 'path', 'image_path']):
                                    cell_metadata_list = value
                                    break
                    
                    if cell_metadata_list is None:
                        # If still not found, try to use the entire metadata as a single item
                        if 'crop_path' in metadata or 'cell_id' in metadata:
                            cell_metadata_list = [metadata]
                        else:
                            raise ValueError(f"Could not find cell metadata list in dictionary format. Available keys: {list(metadata.keys())}")
            else:
                raise ValueError(f"Unexpected metadata format: {type(metadata)}. Expected list or dict, got {type(metadata)}")
            
            # Extract required data safely
            cell_crops = []
            cell_metadata = []
            
            for item in cell_metadata_list:
                if isinstance(item, dict):
                    # Handle different possible key names for crop path
                    crop_path = None
                    cell_id = None
                    
                    # Try different possible keys for crop path
                    for path_key in ['crop_path', 'path', 'image_path', 'file_path']:
                        if path_key in item:
                            crop_path = item[path_key]
                            break
                    
                    # Try different possible keys for cell ID
                    for id_key in ['cell_id', 'id', 'cell_id']:
                        if id_key in item:
                            cell_id = item[id_key]
                            break
                    
                    if crop_path:
                        # Normalize the crop path
                        crop_path = os.path.normpath(crop_path)
                        cell_crops.append(crop_path)
                        
                        # Create metadata entry
                        metadata_entry = {}
                        if cell_id is not None:
                            metadata_entry['cell_id'] = cell_id
                        cell_metadata.append(metadata_entry)
            
            print(f"Found {len(cell_crops)} valid cell crops from metadata")
            
            # Verify crop files exist
            existing_files, missing_files = self._verify_crop_files(cell_crops)
            if missing_files:
                print(f"Warning: {len(missing_files)} crop files are missing")
                print(f"Missing files: {missing_files[:3]}...")  # Show first 3
            
            return cell_crops, cell_metadata
            
        except Exception as e:
            print(f"Error loading metadata: {str(e)}")
            return [], []

    def _create_metadata_backup(self, file_path):
        """
        Create a backup of metadata file with timestamp.
        
        Args:
            file_path: Path to the metadata file
            
        Returns:
            Backup file path or None if failed
        """
        try:
            from datetime import datetime
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_path = file_path.replace('.json', f'_backup_{timestamp}.json')
            
            import shutil
            shutil.copy2(file_path, backup_path)
            return backup_path
        except Exception as e:
            print(f"Failed to create backup: {e}")
            return None

    def _verify_crop_files(self, cell_crops):
        """
        Verify that all crop files exist.
        
        Args:
            cell_crops: List of crop file paths
            
        Returns:
            Tuple of (existing_files, missing_files)
        """
        existing_files = []
        missing_files = []
        
        for crop_path in cell_crops:
            if os.path.exists(crop_path):
                existing_files.append(crop_path)
            else:
                missing_files.append(crop_path)
        
        return existing_files, missing_files

    def _list_metadata_files(self, cache_dir):
        """
        List all metadata files in the cache directory.
        
        Args:
            cache_dir: Directory to search
            
        Returns:
            List of metadata file information
        """
        try:
            metadata_files = glob.glob(os.path.join(cache_dir, 'cell_crops_metadata_*.json'))
            file_info = []
            
            for file_path in metadata_files:
                try:
                    stat = os.stat(file_path)
                    file_info.append({
                        "filename": os.path.basename(file_path),
                        "size": stat.st_size,
                        "modified": stat.st_mtime,
                        "path": file_path
                    })
                except Exception as e:
                    file_info.append({
                        "filename": os.path.basename(file_path),
                        "error": str(e)
                    })
            
            return file_info
        except Exception as e:
            return [{"error": f"Failed to list files: {str(e)}"}]

    def _calculate_statistics(self, results: list) -> dict:
        """Calculate statistics from classification results."""
        if not results:
            return {
                "class_distribution": {},
                "total_cells": 0
            }
        class_counts = {}
        for result in results:
            predicted_class = result.get("predicted_class", "unknown")
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
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
            "total_cells": total_cells
        }

    def _create_visualizations(self, results: list, features: np.ndarray, stats: dict, output_dir: str) -> list:
        vis_config = VisualizationConfig()
        output_paths = []
        # 1. Pie chart of cell state distribution
        try:
            fig, ax = vis_config.create_professional_figure(figsize=(10, 8))
            class_counts = Counter([r['predicted_class'] for r in results])
            labels = list(class_counts.keys())
            sizes = list(class_counts.values())
            color_list = [vis_config.get_professional_colors().get(label, '#cccccc') for label in labels]
            ax.pie(sizes, labels=labels, autopct='%1.1f%%', startangle=140, colors=color_list)
            ax.axis('equal')
            ax.set_title("Cell State Distribution", fontsize=18, fontweight='bold')
            pie_path = os.path.join(output_dir, "cell_state_distribution.png")
            vis_config.save_professional_figure(fig, pie_path)
            plt.close(fig)
            output_paths.append(pie_path)
            print(f"✅ Created pie chart: {pie_path}")
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            print(f"❌ Error creating pie chart: {str(e)}")
        # 2. Bar chart of cell states
        try:
            fig, ax = vis_config.create_professional_figure(figsize=(12, 8))
            class_counts = Counter([r['predicted_class'] for r in results])
            labels = list(class_counts.keys())
            sizes = list(class_counts.values())
            color_list = [vis_config.get_professional_colors().get(label, '#cccccc') for label in labels]
            bars = ax.bar(labels, sizes, color=color_list, edgecolor='black', linewidth=2, alpha=1.0)
            ax.set_title("Number of Each Cell State", fontsize=18, fontweight='bold')
            ax.set_xlabel("Cell States", fontsize=16, fontweight='bold')
            ax.set_ylabel("Number of Cells", fontsize=16, fontweight='bold')
            ax.grid(True, alpha=0.4, linewidth=1.0)
            ax.tick_params(axis='both', which='major', labelsize=14, width=2, length=6)
            ax.tick_params(axis='x', rotation=45, labelsize=14)
            bar_path = os.path.join(output_dir, "cell_state_bars.png")
            vis_config.save_professional_figure(fig, bar_path)
            plt.close(fig)
            output_paths.append(bar_path)
            print(f"✅ Created bar chart: {bar_path}")
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            print(f"❌ Error creating bar chart: {str(e)}")
        return output_paths

    def _save_results(self, results: list, summary: dict, query_cache_dir: str) -> str:
        """
        Save analysis results and create output directory for visualizations.
        
        Args:
            results: List of analysis results
            summary: Summary statistics
            query_cache_dir: Cache directory path
            
        Returns:
            Path to the output directory for visualizations
        """
        try:
            # Create output directory using VisualizationConfig
            from octotools.models.utils import VisualizationConfig
            output_dir = VisualizationConfig.get_output_dir(query_cache_dir)
            
            # Save results to JSON file
            results_file = os.path.join(output_dir, "analysis_results.json")
            with open(results_file, 'w') as f:
                json.dump({
                    "results": results,
                    "summary": summary,
                    "timestamp": time.strftime("%Y%m%d_%H%M%S")
                }, f, indent=2, default=str)
            
            print(f"💾 Analysis results saved to: {results_file}")
            return output_dir
            
        except Exception as e:
            print(f"❌ Error saving results: {str(e)}")
            # Fallback to default output directory
            return "output_visualizations"
    
    def get_metadata(self):
        """Returns the tool's metadata."""
        metadata = super().get_metadata()
        metadata.update({
            "device": str(self.device),
            "model_loaded": self.model is not None,
            "is_model_trained": self._is_model_trained(),
            "backbone_size": "large",
            "class_names": self.class_names,
            "class_descriptions": self.class_descriptions
        })
        return metadata

    def _generate_recommendations(self, stats: dict, total_cells: int, valid_cells: int) -> dict:
        recommendations = {}
        if valid_cells < 10:
            recommendations["data_quality"] = "Warning: Very few cells analyzed. Consider using more cell crops for reliable results."
        elif valid_cells < 50:
            recommendations["data_quality"] = "Moderate cell count. Results may be more reliable with additional cell crops."
        else:
            recommendations["data_quality"] = "Good cell count for analysis."
        class_dist = stats.get("class_distribution", {})
        if len(class_dist) < 2:
            recommendations["diversity"] = "Limited cell state diversity detected. This may indicate a homogeneous sample or classification issues."
        else:
            recommendations["diversity"] = "Good diversity of cell states detected."
        return recommendations
    
    def _get_state_distribution(self, results: list) -> dict:
        """Get cell state distribution from results."""
        if not results:
            return {}
        
        class_counts = {}
        for result in results:
            predicted_class = result.get("predicted_class", "unknown")
            class_counts[predicted_class] = class_counts.get(predicted_class, 0) + 1
        
        total_cells = len(results)
        return {
            class_name: {
                "count": count,
                "percentage": (count / total_cells) * 100
            }
            for class_name, count in class_counts.items()
        }
    
    def _assess_quality(self, results: list) -> dict:
        if not results:
            return {"overall_quality": "poor", "issues": ["No results to assess"]}
        if len(results) < 10:
            return {"overall_quality": "poor", "issues": ["Small sample size"]}
        return {"overall_quality": "good", "issues": []}


if __name__ == "__main__":
    # --- Command-Line Interface for Testing ---
    parser = argparse.ArgumentParser(
        description="Test the Fibroblast_State_Analyzer_Tool from the command line."
    )
    parser.add_argument(
        'cell_crops', 
        nargs='*',  # 0 or more arguments
        default=[],
        help="Paths to one or more cell crop images to analyze."
    )
    parser.add_argument(
        '--model_path',
        type=str,
        default=None,
        help="Optional path to a local model checkpoint file."
    )
    parser.add_argument(
        '--confidence',
        type=float,
        default=0.5,
        help="Confidence threshold for classification."
    )
    args = parser.parse_args()

    print("--- Initializing Fibroblast State Analyzer Tool ---")
    tool = Fibroblast_State_Analyzer_Tool(
        model_path=args.model_path,
        confidence_threshold=args.confidence
    )

    # --- Get and Print Tool Metadata ---
    print("\n--- Tool Metadata ---")
    metadata = tool.get_metadata()
    print(json.dumps(metadata, indent=2))

    # --- Execute Analysis if Crops are Provided ---
    if args.cell_crops:
        print(f"\n--- Analyzing {len(args.cell_crops)} Cell Crops ---")
        try:
            execution_result = tool.execute(cell_crops=args.cell_crops)
            print("\n--- Analysis Result ---")
            print(json.dumps(execution_result, indent=2))
        except Exception as e:
            print(f"\n--- An error occurred during execution ---")
            print(str(e))
    else:
        print("\n--- No cell crops provided for analysis. ---")
        print("You can test the tool by providing file paths as arguments.")
        print("Example: python tool.py path/to/cell1.png path/to/cell2.png")
        print("Or use the tool's demo commands in your code:")
        print("execution = tool.execute(cell_crops=['cell_0001.png', 'cell_0002.png'])")

    print("\n--- Test script finished. ---") 
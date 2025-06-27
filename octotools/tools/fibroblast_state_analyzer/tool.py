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
        
    def forward(self, x, return_features=False):
        """
        Correctly processes the input tensor, extracts the classification token,
        and returns logits. Optionally returns the feature tensor as well.
        """
        # DINOv2 backbone returns a dict. We need to extract the feature tensor.
        features_dict = self.backbone.forward_features(x)
        
        # The classification token is what is used for linear probing.
        features = features_dict['x_norm_clstoken']
        
        # Pass the feature tensor to our custom classifier head.
        logits = self.backbone.head(features)
        
        if return_features:
            return logits, features
        return logits

class Fibroblast_State_Analyzer_Tool(BaseTool):
    """
    Analyzes fibroblast cell states using a pre-trained DINOv2-based classifier.
    Processes individual cell crops to determine their activation state.
    """
    
    def __init__(self, model_path=None, backbone_size="base", confidence_threshold=0.5):
        super().__init__(
            tool_name="Fibroblast_State_Analyzer_Tool",
            tool_description="Analyzes fibroblast cell states using deep learning to classify individual cells into different activation states. Generates comprehensive visualizations including UMAP plots for cell distribution analysis, PCA plots, confidence distributions, and cell state bar charts. UMAP visualization is particularly useful for understanding cell clustering and spatial relationships in the feature space.",
            tool_version="1.0.0",
            input_types={
                "cell_crops": "List[str] - Paths to individual cell crop images",
                "cell_metadata": "List[dict] - Metadata for each cell crop",
                "batch_size": "int - Batch size for processing (default: 16)",
                "query_cache_dir": "str - Directory for caching results",
                "visualization_type": "str - Visualization method: 'pca', 'umap', 'auto', or 'all' (default: 'auto'). Use 'all' to generate all visualizations including UMAP for cell distribution analysis."
            },
            output_type="dict - Analysis results with cell state classifications, statistics, and comprehensive visualizations including UMAP plots",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(cell_crops=["cell_0001.png", "cell_0002.png"], cell_metadata=[{"cell_id": 1}, {"cell_id": 2}], visualization_type="all")',
                    "description": "Analyze cell states with all visualizations including UMAP for cell distribution analysis"
                },
                {
                    "command": 'execution = tool.execute(cell_crops=["cell_0001.png"], visualization_type="umap")',
                    "description": "Analyze with UMAP visualization to show cell clustering and distribution in feature space"
                },
                {
                    "command": 'execution = tool.execute(cell_crops=["cell_0001.png"], visualization_type="all")',
                    "description": "Generate comprehensive analysis with all visualizations: UMAP, PCA, confidence distributions, and cell state charts"
                }
            ],
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
        self.confidence_threshold = confidence_threshold
        
        # Cell state classes with specific colors
        self.class_names = ["dead", "np-MyoFb", "p-MyoFb", "proto-MyoFb", "q-Fb"]
        self.class_descriptions = {
            "dead": "Dead or dying cells",
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
            backbone_archs = {
                "small": "vits14", "base": "vitb14", "large": "vitl14", "giant": "vitg14"
            }
            feat_dim_map = {
                "vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536
            }
            backbone_arch = backbone_archs.get(self.backbone_size, "vitb14")
            backbone_name = f"dinov2_{backbone_arch}"

            # Load the DINOv2 backbone using torch.hub to match the training environment
            backbone_model = torch.hub.load(repo_or_dir="facebookresearch/dinov2", model=backbone_name)
            backbone_model.eval() # Keep backbone frozen
            
            # Create the model wrapper
            self.model = DinoV2Classifier(
                backbone=backbone_model,
                num_classes=len(self.class_names)
            ).to(self.device)

            # Re-create the classifier head exactly as in the training notebook
            feat_dim = feat_dim_map[backbone_arch]
            hidden_dim = feat_dim // 2
            self.model.backbone.head = nn.Sequential(
                nn.Linear(feat_dim, hidden_dim),
                nn.ReLU(inplace=True),
                nn.Dropout(p=0),
                nn.Linear(hidden_dim, len(self.class_names))
            ).to(self.device)
            
            logger.info(f"Using {backbone_name} backbone with a custom {hidden_dim}-hidden-dim head.")
            
            # Load finetuned weights from HuggingFace Hub or local path
            model_loaded = False
            
            # Try to load weights based on backbone size
            if self.model_path and os.path.exists(self.model_path):
                try:
                    logger.info(f"Attempting to load from local path: {self.model_path}")
                    state_dict = torch.load(self.model_path, map_location=self.device, weights_only=True)
                    if 'model_state_dict' in state_dict:
                        self.model.load_state_dict(state_dict['model_state_dict'])
                    else:
                        self.model.load_state_dict(state_dict)
                    logger.info("Successfully loaded model from local path")
                    model_loaded = True
                except Exception as local_e:
                    logger.error(f"Failed to load from local path: {str(local_e)}")
            
            if not model_loaded:
                try:
                    # Try to load the new model.pt (trained on base backbone)
                    logger.info("Attempting to download new model.pt (trained on base backbone) from HuggingFace Hub...")
                    model_weights_path = hf_hub_download(
                        repo_id="5xuekun/fb-classifier-model",
                        filename="model.pt",
                        token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                    logger.info(f"Downloaded model weights to: {model_weights_path}")
                    
                    # Load the state dict
                    checkpoint_data = torch.load(model_weights_path, map_location=self.device, weights_only=True)
                    if 'model_state_dict' in checkpoint_data:
                        self.model.load_state_dict(checkpoint_data['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint_data)
                    logger.info("Successfully loaded finetuned weights from new model.pt")
                    model_loaded = True

                except Exception as e:
                    logger.warning(f"Failed to load weights from HuggingFace Hub: {str(e)}")

            if not model_loaded:
                logger.info("Using untrained classifier head")

            self.model.eval()
            
            # Define transforms for preprocessing
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
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
            final_layer_weights = self.model.backbone.head[-1].weight.data
            weights_norm = torch.norm(final_layer_weights).item()
            return weights_norm > 0.1 # A simple heuristic
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
    
    def _classify_single_cell(self, image_path: str, model: nn.Module, confidence_threshold: float) -> Tuple[Dict[str, Any], Optional[torch.Tensor]]:
        """Classify a single cell image and return result with features."""
        try:
            # Preprocess image
            img_tensor = self._preprocess_image(image_path)
            
            # Get predictions and features
            with torch.no_grad():
                # Extract features from backbone (before classifier head)
                # DINOv2 backbone returns a dict with 'x_norm_clstoken' as the CLS token
                backbone_output = model.backbone(img_tensor)
                
                # Extract CLS token features (this is the embedding we want for UMAP)
                if isinstance(backbone_output, dict):
                    # DINOv2 returns dict with 'x_norm_clstoken' key
                    features = backbone_output['x_norm_clstoken']  # Shape: [1, feat_dim]
                else:
                    # Fallback: if backbone returns tensor directly, use it
                    features = backbone_output
                    if features.dim() > 2:
                        # If it's [1, seq_len, feat_dim], take the CLS token (first token)
                        features = features[:, 0, :]  # Shape: [1, feat_dim]
                
                # Get classification logits through the full model
                logits = model(img_tensor)
                probs = torch.softmax(logits, dim=1)
                pred_idx = probs.argmax(dim=1).item()
                confidence = probs[0][pred_idx].item()
            
            # Create result
            result = {
                "image_path": image_path,
                "predicted_class": self.class_names[pred_idx],
                "confidence": confidence,
                "features": features.squeeze(0).cpu().numpy()
            }
            
            return result, features.squeeze(0)  # Remove batch dimension
            
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
                
                # Get model output in no_grad context
                with torch.no_grad():
                    logits, features = self.model(batch_tensor, return_features=True)
                    probs = F.softmax(logits, dim=1)
                    confidences, predictions = torch.max(probs, dim=1)
                
                # Process batch results, ensuring we only use valid paths
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
                        features_list.append(feature_vector)  # Add features to the list

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
            
            # 构建AnnData对象用于下游激活评分
            obs_dict = {
                'image_path': [r['image_path'] for r in results],
                'predicted_class': [r['predicted_class'] for r in results],
                'confidence': [r['confidence'] for r in results],
            }
            
            # Fix for anndata compatibility - create AnnData with proper obs DataFrame
            obs_df = pd.DataFrame(obs_dict)
            X = np.array([r['features'] for r in results])
            adata = anndata.AnnData(X=X, obs=obs_df)
            
            # Save AnnData as h5ad file for downstream activation scoring
            try:
                # Create output directory using VisualizationConfig
                from octotools.models.utils import VisualizationConfig
                output_dir = VisualizationConfig.get_output_dir(query_cache_dir)
                
                # Save h5ad file
                h5ad_path = os.path.join(output_dir, "fibroblast_state_analyzed.h5ad")
                adata.write(h5ad_path)
                print(f"💾 AnnData saved to: {h5ad_path}")
                
                # Also save to tool cache directory for easier access
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
                "average_confidence": np.mean([r['confidence'] for r in results]),
                "visual_outputs": visual_outputs,
                "parameters": {
                    "confidence_threshold": self.confidence_threshold,
                    "backbone_size": "base",
                    "model_path": self.model_path
                },
                "cell_state_descriptions": self.class_descriptions,
                "analysis_quality": self._assess_quality(results),
                "recommendations": recommendations,
                "metadata_info": {
                    "total_crops_processed": len(cell_crops),
                    "successful_analyses": len(results),
                    "metadata_files_used": self._list_metadata_files(query_cache_dir)
                },
                "adata": adata,
                "h5ad_path": h5ad_path
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
            # Find all metadata files
            metadata_files = glob.glob(os.path.join(cache_dir, 'cell_crops_metadata_*.json'))
            
            if not metadata_files:
                print(f"No metadata files found in {cache_dir}")
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
    
    def _create_visualizations(self, results: List[Dict], features: np.ndarray, stats: Dict, output_dir: str) -> List[str]:
        """
        Creates all visualizations based on the analysis results and returns their paths.
        This function now contains all centralized plotting logic.
        """
        from octotools.models.utils import VisualizationConfig
        vis_config = VisualizationConfig()
        output_paths = []

        # 1. Cell State Distribution (Pie Chart)
        try:
            fig, ax = vis_config.create_professional_figure(figsize=(12, 8))
            class_distribution = stats.get('class_distribution', {})
            labels = list(class_distribution.keys())
            sizes = [class_distribution[label]['count'] for label in labels]
            colors = [vis_config.get_professional_colors().get(state, '#CCCCCC') for state in labels]
            
            wedges, texts, autotexts = ax.pie(
                sizes, labels=None, autopct='%1.1f%%', startangle=140, 
                colors=colors, wedgeprops=dict(edgecolor='w', linewidth=2)
            )
            plt.setp(autotexts, size=22, color="black", fontweight='bold')
            ax.axis('equal')
            vis_config.apply_professional_styling(ax, title="Cell State Composition")
            
            ax.legend(wedges, labels, title="Cell States", loc="center left",
                      bbox_to_anchor=(1, 0, 0.5, 1),
                      fontsize=vis_config.PROFESSIONAL_STYLE['legend.fontsize'],
                      title_fontsize=vis_config.PROFESSIONAL_STYLE['legend.title_fontsize'])
            
            fig.tight_layout()
            pie_path = os.path.join(output_dir, "cell_state_distribution.png")
            vis_config.save_professional_figure(fig, pie_path)
            plt.close(fig)
            output_paths.append(pie_path)
            print(f"✅ Created pie chart: {pie_path}")
        except Exception as e:
            logger.error(f"Error creating pie chart: {str(e)}")
            print(f"❌ Error creating pie chart: {str(e)}")

        # 3. Bar Chart of Cell States
        try:
            fig, ax = vis_config.create_professional_figure(figsize=(12, 8))
            class_distribution = stats.get('class_distribution', {})
            labels = list(class_distribution.keys())
            sizes = [class_distribution[label]['count'] for label in labels]
            colors = [vis_config.get_professional_colors().get(state, '#CCCCCC') for state in labels]
            
            ax.bar(labels, sizes, color=colors, edgecolor='black', linewidth=2)
            vis_config.apply_professional_styling(
                ax, title="Cell State Counts",
                xlabel="Cell State", ylabel="Number of Cells"
            )
            ax.tick_params(axis='x', rotation=45)
            fig.tight_layout()
            
            bar_path = os.path.join(output_dir, "cell_state_bars.png")
            vis_config.save_professional_figure(fig, bar_path)
            plt.close(fig)
            output_paths.append(bar_path)
            print(f"✅ Created bar chart: {bar_path}")
        except Exception as e:
            logger.error(f"Error creating bar chart: {str(e)}")
            print(f"❌ Error creating bar chart: {str(e)}")

        # 4. UMAP Visualization
        if ANNDATA_AVAILABLE:
            try:
                logger.info("Creating UMAP visualization...")
                print("🔍 Creating UMAP visualization...")
                adata = anndata.AnnData(X=features)
                adata.obs['predicted_class'] = [r['predicted_class'] for r in results]

                sc.pp.neighbors(adata, n_neighbors=min(10, len(adata) - 1), use_rep='X')
                sc.tl.umap(adata)

                fig, ax = plt.subplots(figsize=(10, 10), dpi=300)
                
                # Use a specific palette for cell states
                palette = vis_config.get_professional_colors()
                
                # Let scanpy handle the legend creation initially
                sc.pl.umap(adata, color='predicted_class', ax=ax, show=False, size=200,
                           palette=palette, legend_loc='on data')

                # Customize the existing legend
                legend = ax.get_legend()
                if legend:
                    legend.set_title("Cell States")
                    legend.get_title().set_fontsize(vis_config.PROFESSIONAL_STYLE['legend.title_fontsize'])
                    legend.get_title().set_fontweight('bold')
                    for text in legend.get_texts():
                        text.set_fontsize(vis_config.PROFESSIONAL_STYLE['legend.fontsize'])
                    legend.set_frame_on(True)
                    legend.set_bbox_to_anchor((1.05, 0.5))
                    legend.set_loc('center left')
                    legend.get_frame().set_edgecolor('black')
                    legend.get_frame().set_linewidth(1.5)
                    legend.get_frame().set_facecolor('#F0F0F0')
                
                # Remove default scanpy title if it exists, since we set a custom one
                if ax.get_title():
                    ax.set_title("")

                ax.set_title("UMAP Embedding", fontsize=vis_config.PROFESSIONAL_STYLE['axes.titlesize'], fontweight='bold', pad=20)
                ax.set_xlabel("UMAP 1", fontsize=vis_config.PROFESSIONAL_STYLE['axes.labelsize'], fontweight='bold')
                ax.set_ylabel("UMAP 2", fontsize=vis_config.PROFESSIONAL_STYLE['axes.labelsize'], fontweight='bold')
                ax.grid(False)
                ax.set_xticks([])
                ax.set_yticks([])
                ax.set_aspect('equal', adjustable='box')
                
                # Adjust layout to prevent legend from being cut off
                fig.tight_layout(rect=[0, 0, 0.85, 1])
                
                output_path = os.path.join(output_dir, "umap_cell_features.png")
                vis_config.save_professional_figure(fig, output_path)
                plt.close(fig)
                
                logger.info(f"UMAP visualization saved to {output_path}")
                print(f"✅ Created UMAP visualization: {output_path}")
                output_paths.append(output_path)
            except Exception as e:
                logger.error(f"Error creating UMAP visualization: {str(e)}")
                print(f"❌ Error creating UMAP visualization: {str(e)}")
                import traceback
                traceback.print_exc()

        print(f"📊 Total visualizations created: {len(output_paths)}")
        return output_paths
    
    def _save_results(self, results: List[Dict], summary: Dict, query_cache_dir: str) -> str:
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
            "backbone_size": "base",
            "class_names": self.class_names,
            "class_descriptions": self.class_descriptions
        })
        return metadata

    def _generate_recommendations(self, stats: Dict, total_cells: int, valid_cells: int) -> Dict[str, str]:
        """Generate recommendations based on analysis results."""
        recommendations = {}
        
        # Check data quality
        if valid_cells < 10:
            recommendations["data_quality"] = "Warning: Very few cells analyzed. Consider using more cell crops for reliable results."
        elif valid_cells < 50:
            recommendations["data_quality"] = "Moderate cell count. Results may be more reliable with additional cell crops."
        else:
            recommendations["data_quality"] = "Good cell count for analysis."
        
        # Check confidence distribution
        avg_confidence = stats.get("average_confidence", 0)
        if avg_confidence < 0.6:
            recommendations["confidence"] = "Low average confidence. Consider adjusting confidence threshold or improving image quality."
        elif avg_confidence < 0.8:
            recommendations["confidence"] = "Moderate confidence. Results are acceptable but could be improved."
        else:
            recommendations["confidence"] = "High confidence in classifications."
        
        # Check class distribution
        class_dist = stats.get("class_distribution", {})
        if len(class_dist) < 2:
            recommendations["diversity"] = "Limited cell state diversity detected. This may indicate a homogeneous sample or classification issues."
        else:
            recommendations["diversity"] = "Good diversity of cell states detected."
        
        return recommendations
    
    def _get_state_distribution(self, results: List[Dict]) -> Dict[str, Any]:
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
    
    def _assess_quality(self, results: List[Dict]) -> Dict[str, Any]:
        """Assess the quality of analysis results."""
        if not results:
            return {"overall_quality": "poor", "issues": ["No results to assess"]}
        
        # Calculate quality metrics
        confidences = [r.get("confidence", 0) for r in results]
        avg_confidence = np.mean(confidences)
        std_confidence = np.std(confidences)
        
        # Count high-confidence predictions
        high_conf_count = sum(1 for c in confidences if c >= 0.8)
        high_conf_ratio = high_conf_count / len(results)
        
        # Assess overall quality
        if avg_confidence >= 0.8 and high_conf_ratio >= 0.7:
            overall_quality = "excellent"
        elif avg_confidence >= 0.6 and high_conf_ratio >= 0.5:
            overall_quality = "good"
        elif avg_confidence >= 0.4:
            overall_quality = "fair"
        else:
            overall_quality = "poor"
        
        # Identify potential issues
        issues = []
        if avg_confidence < 0.6:
            issues.append("Low average confidence")
        if std_confidence > 0.3:
            issues.append("High confidence variability")
        if len(results) < 10:
            issues.append("Small sample size")
        
        return {
            "overall_quality": overall_quality,
            "average_confidence": avg_confidence,
            "confidence_std": std_confidence,
            "high_confidence_ratio": high_conf_ratio,
            "total_cells": len(results),
            "issues": issues
        }


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
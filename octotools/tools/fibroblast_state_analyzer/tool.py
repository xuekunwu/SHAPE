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
        # The forward pass now directly returns the output of the backbone's head
        return self.backbone(x)

class Fibroblast_State_Analyzer_Tool(BaseTool):
    """
    Analyzes fibroblast cell states using a pre-trained DINOv2-based classifier.
    Processes individual cell crops to determine their activation state.
    """
    
    def __init__(self, model_path=None, backbone_size="large", confidence_threshold=0.5):
        super().__init__(
            tool_name="Fibroblast_State_Analyzer_Tool",
            tool_description="Analyzes fibroblast cell states using deep learning to classify individual cells into different activation states. Supports both PCA and UMAP visualizations.",
            tool_version="1.0.0",
            input_types={
                "cell_crops": "List[str] - Paths to individual cell crop images",
                "cell_metadata": "List[dict] - Metadata for each cell crop",
                "confidence_threshold": "float - Minimum confidence threshold for classification (default: 0.5)",
                "batch_size": "int - Batch size for processing (default: 16)",
                "query_cache_dir": "str - Directory for caching results",
                "visualization_method": "str - Visualization method: 'pca', 'umap', or 'auto' (default: 'auto')"
            },
            output_type="dict - Analysis results with cell state classifications and statistics",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(cell_crops=["cell_0001.png", "cell_0002.png"], cell_metadata=[{"cell_id": 1}, {"cell_id": 2}])',
                    "description": "Analyze cell states for individual fibroblast crops"
                },
                {
                    "command": 'execution = tool.execute(cell_crops=["cell_0001.png"], visualization_method="umap")',
                    "description": "Analyze with UMAP visualization (requires anndata/scanpy)"
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for optimal performance. Model accuracy depends on image quality and cell visibility. May struggle with very small or overlapping cells.",
                "best_practice": "Use with high-quality cell crops from Single_Cell_Cropper_Tool. Ensure cells are well-separated and clearly visible in crops.",
                "cell_states": "Classifies cells into: dead, np-MyoFb (non-proliferative myofibroblast), p-MyoFb (proliferative myofibroblast), proto-MyoFb (proto-myofibroblast), q-Fb (quiescent fibroblast)",
                "visualization": "Supports PCA (fast, interpretable) and UMAP (advanced, requires anndata/scanpy). Auto-selects based on availability."
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
        
        # Initialize model
        self._initialize_model()
        
    def _initialize_model(self):
        """Initialize the DINOv2 model and classifier with finetuned weights."""
        try:
            logger.info("Initializing DINOv2 model (torch.hub)...")
            
            # Define backbone architecture based on size
            backbone_archs = {
                "small": "vits14", "base": "vitb14", "large": "vitl14", "giant": "vitg14"
            }
            feat_dim_map = {
                "vits14": 384, "vitb14": 768, "vitl14": 1024, "vitg14": 1536
            }
            backbone_arch = backbone_archs.get(self.backbone_size, "vitl14")
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
                    logger.info("Downloading finetuned weights from HuggingFace Hub...")
                    model_weights_path = hf_hub_download(
                        repo_id="5xuekun/fb-classifier-model",
                        filename="model.pt", # Or large_best_model.pth if the name is different
                        token=os.getenv("HUGGINGFACE_TOKEN")
                    )
                    logger.info(f"Downloaded model weights to: {model_weights_path}")
                    
                    # Load the state dict
                    checkpoint_data = torch.load(model_weights_path, map_location=self.device, weights_only=True)
                    if 'model_state_dict' in checkpoint_data:
                        self.model.load_state_dict(checkpoint_data['model_state_dict'])
                    else:
                        self.model.load_state_dict(checkpoint_data)
                    logger.info("Successfully loaded finetuned weights from HuggingFace Hub")
                    model_loaded = True

                except Exception as e:
                    logger.warning(f"Failed to load weights from HuggingFace Hub: {str(e)}")

            if not model_loaded:
                logger.warning("Could not load any trained weights. Using untrained classifier.")

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
                "all_probabilities": {
                    cls_name: prob.item() 
                    for cls_name, prob in zip(self.class_names, probs[0])
                }
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
    
    def execute(self, cell_crops=None, cell_metadata=None, confidence_threshold=0.5, 
                batch_size=16, query_cache_dir='solver_cache/temp/tool_cache/', 
                visualization_type='auto', **kwargs):
        """
        Execute fibroblast state analysis on cell crops.
        
        Args:
            cell_crops: List of paths to cell crop images
            cell_metadata: List of metadata dictionaries for each cell
            confidence_threshold: Minimum confidence for classification
            batch_size: Batch size for processing
            query_cache_dir: Directory for caching results
            visualization_type: 'pca', 'umap', or 'auto' for automatic selection
            **kwargs: Additional arguments
            
        Returns:
            Dictionary containing analysis results
        """
        try:
            # Ensure cache directory exists
            os.makedirs(query_cache_dir, exist_ok=True)
            
            # If cell_crops not provided, try to load from metadata
            if cell_crops is None or cell_metadata is None:
                print("Loading cell data from metadata files...")
                cell_crops, cell_metadata = self._load_cell_data_from_metadata(query_cache_dir)
                
                if not cell_crops:
                    # Get detailed metadata file information for debugging
                    metadata_files = self._list_metadata_files(query_cache_dir)
                    return {
                        "error": "No cell crops found. Please ensure metadata files exist and contain valid crop paths.",
                        "status": "failed",
                        "debug_info": {
                            "cache_dir": query_cache_dir,
                            "metadata_files_found": metadata_files,
                            "suggestion": "Check if metadata files were accidentally deleted or moved during task execution."
                        }
                    }
            
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
            
            for i in range(0, len(cell_crops), batch_size):
                batch_crops = cell_crops[i:i + batch_size]
                batch_metadata = cell_metadata[i:i + batch_size] if cell_metadata else [{}] * len(batch_crops)
                
                for crop_path, metadata in zip(batch_crops, batch_metadata):
                    try:
                        result, features = self._classify_single_cell(crop_path, model, confidence_threshold)
                        if result:
                            results.append(result)
                            if features is not None:
                                features_list.append(features)
                                if len(features_list) % 50 == 0:  # Print progress every 50 cells
                                    print(f"Processed {len(results)} cells, extracted {len(features_list)} features")
                    except Exception as e:
                        print(f"Error processing {crop_path}: {str(e)}")
                        continue
            
            print(f"Processing completed. Total results: {len(results)}, Total features: {len(features_list)}")
            
            if not results:
                return {"error": "No cells were successfully analyzed", "status": "failed"}
            
            # Calculate statistics
            summary = self._calculate_statistics(results)
            
            # Generate basic visualizations
            visual_outputs = self._create_visualizations(results, summary, query_cache_dir)
            
            # Generate advanced visualizations (PCA/UMAP) if features are available
            if features_list and len(features_list) > 0:
                try:
                    print(f"Generating advanced visualizations with {len(features_list)} features...")
                    # Stack features into a tensor
                    features_tensor = torch.stack(features_list)
                    print(f"Features tensor shape: {features_tensor.shape}")
                    
                    # Create PCA visualization
                    print("Creating PCA visualization...")
                    pca_viz_path = self._create_pca_visualization(results, features_tensor, query_cache_dir)
                    if pca_viz_path:
                        visual_outputs.append(pca_viz_path)
                        print(f"PCA visualization created: {pca_viz_path}")
                    else:
                        print("PCA visualization failed")
                    
                    # Create UMAP visualization if anndata is available
                    print("Creating UMAP visualization...")
                    umap_viz_path = self._create_umap_visualization(results, features_tensor, query_cache_dir)
                    if umap_viz_path:
                        visual_outputs.append(umap_viz_path)
                        print(f"UMAP visualization created: {umap_viz_path}")
                    else:
                        print("UMAP visualization failed")
                        
                except Exception as e:
                    print(f"Warning: Failed to create advanced visualizations: {str(e)}")
                    import traceback
                    traceback.print_exc()
            else:
                print(f"No features available for advanced visualizations. features_list length: {len(features_list) if features_list else 0}")
            
            return {
                "summary": summary,
                "cell_state_distribution": self._get_state_distribution(results),
                "average_confidence": np.mean([r['confidence'] for r in results]),
                "visual_outputs": visual_outputs,
                "parameters": {
                    "confidence_threshold": confidence_threshold,
                    "backbone_size": "large",
                    "model_path": None
                },
                "cell_state_descriptions": self.class_descriptions,
                "analysis_quality": self._assess_quality(results),
                "recommendations": self._generate_recommendations(summary, len(cell_crops), len(results)),
                "metadata_info": {
                    "total_crops_processed": len(cell_crops),
                    "successful_analyses": len(results),
                    "metadata_files_used": self._list_metadata_files(query_cache_dir)
                }
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
    
    def _create_visualizations(self, results: List[Dict], stats: Dict, output_dir: str) -> List[str]:
        """Create visualizations of analysis results."""
        viz_paths = []
        
        try:
            # 1. Class distribution pie chart with specific colors, legend, and external percentages
            if stats["class_distribution"]:
                fig, ax = plt.subplots(figsize=(12, 8)) # Adjusted for legend
                
                # Prepare data for pie chart
                labels = list(stats["class_distribution"].keys())
                sizes = [stats["class_distribution"][label]["count"] for label in labels]
                colors = [self.color_map.get(label, '#CCCCCC') for label in labels]

                # Explode smaller slices to prevent percentage overlap
                explode = [0.1 if (size / sum(sizes)) < 0.05 else 0 for size in sizes]

                wedges, texts, autotexts = ax.pie(
                    sizes, 
                    autopct='%1.1f%%', 
                    startangle=90,
                    colors=colors,
                    pctdistance=1.1,  # Move percentages outside the pie
                    explode=explode,
                    labels=None,       # Labels will be in the legend
                    labeldistance=1.2  # Adjust label line distance if labels were present
                )
                
                plt.setp(autotexts, size=12, color="black") # Percentages are outside, so use black text
                ax.set_title("Cell State Composition", fontsize=18)
                ax.axis('equal')  # Equal aspect ratio ensures that pie is drawn as a circle.
                
                # Add a legend to the side
                ax.legend(wedges, labels,
                          title="Cell States",
                          loc="center left",
                          bbox_to_anchor=(0.95, 0.5), # Anchor legend to the right
                          fontsize=12)

                fig.tight_layout()
                
                # Save visualization
                viz_path = os.path.join(output_dir, f"cell_state_distribution_{uuid4().hex[:8]}.png")
                plt.savefig(viz_path, bbox_inches='tight', dpi=150, format='png')
                plt.close(fig)
                viz_paths.append(viz_path)
            
            # 2. Confidence distribution histogram with improved styling
            if results:
                confidences = [r.get("confidence", 0) for r in results]
                fig, ax = plt.subplots(figsize=(10, 6))
                
                # Create histogram with better styling
                n, bins, patches = ax.hist(confidences, bins=20, alpha=0.7, color='skyblue', 
                                         edgecolor='black', linewidth=1.2)
                
                # Add threshold line
                ax.axvline(self.confidence_threshold, color='red', linestyle='--', 
                          linewidth=2, label=f'Threshold ({self.confidence_threshold})')
                
                # Styling
                ax.set_xlabel("Confidence Score", fontsize=12)
                ax.set_ylabel("Number of Cells", fontsize=12)
                ax.set_title("Confidence Distribution", fontsize=14, fontweight='bold')
                ax.legend(fontsize=10)
                ax.grid(True, alpha=0.3)
                
                # Add statistics text
                mean_conf = np.mean(confidences)
                std_conf = np.std(confidences)
                ax.text(0.02, 0.98, f'Mean: {mean_conf:.3f}\nStd: {std_conf:.3f}', 
                       transform=ax.transAxes, verticalalignment='top',
                       bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
                
                # Save visualization
                conf_viz_path = os.path.join(output_dir, f"confidence_distribution_{uuid4().hex[:8]}.png")
                plt.savefig(conf_viz_path, bbox_inches='tight', dpi=150, format='png')
                plt.close()
                viz_paths.append(conf_viz_path)
            
            # 3. Bar chart of cell state distribution
            if stats["class_distribution"]:
                fig, ax = plt.subplots(figsize=(10, 6))
                
                labels = list(stats["class_distribution"].keys())
                counts = [stats["class_distribution"][label]["count"] for label in labels]
                colors = [self.color_map.get(label, '#CCCCCC') for label in labels]
                
                bars = ax.bar(labels, counts, color=colors, alpha=0.8, edgecolor='black', linewidth=1)
                
                # Add value labels on bars
                for bar, count in zip(bars, counts):
                    height = bar.get_height()
                    ax.text(bar.get_x() + bar.get_width()/2., height + 0.5,
                           f'{count}', ha='center', va='bottom', fontweight='bold')
                
                ax.set_xlabel("Cell States", fontsize=12)
                ax.set_ylabel("Number of Cells", fontsize=12)
                ax.set_title("Cell State Distribution", fontsize=14, fontweight='bold')
                ax.tick_params(axis='x', rotation=45)
                ax.grid(True, alpha=0.3, axis='y')
                
                # Save visualization
                bar_viz_path = os.path.join(output_dir, f"cell_state_bars_{uuid4().hex[:8]}.png")
                plt.savefig(bar_viz_path, bbox_inches='tight', dpi=150, format='png')
                plt.close()
                viz_paths.append(bar_viz_path)
            
        except Exception as e:
            logger.error(f"Error creating visualizations: {str(e)}")
        
        return viz_paths
    
    def _create_pca_visualization(self, results: List[Dict], features: torch.Tensor, output_dir: str) -> Optional[str]:
        """Generate a PCA visualization from extracted features."""
        try:
            logger.info("Generating PCA visualization...")
            
            # Convert features to numpy array
            features_np = features.numpy()
            
            # Apply PCA
            pca = PCA(n_components=2)
            features_2d = pca.fit_transform(features_np)
            
            # Create visualization
            fig, ax = plt.subplots(figsize=(10, 8))
            
            # Plot each class with its specific color
            for class_name in self.class_names:
                class_indices = [i for i, r in enumerate(results) if r['predicted_class'] == class_name]
                if class_indices:
                    class_features = features_2d[class_indices]
                    ax.scatter(class_features[:, 0], class_features[:, 1], 
                             c=self.color_map.get(class_name, '#CCCCCC'), 
                             label=class_name, alpha=0.7, s=60, edgecolors='black', linewidth=0.5)
            
            ax.set_xlabel(f'PC1 ({pca.explained_variance_ratio_[0]:.1%} variance)', fontsize=12)
            ax.set_ylabel(f'PC2 ({pca.explained_variance_ratio_[1]:.1%} variance)', fontsize=12)
            ax.set_title("Cell Feature Space (PCA)", fontsize=14, fontweight='bold')
            ax.legend(title="Cell States", fontsize=10)
            ax.grid(True, alpha=0.3)
            
            # Add explained variance information
            total_variance = pca.explained_variance_ratio_.sum()
            ax.text(0.02, 0.98, f'Total explained variance: {total_variance:.1%}', 
                   transform=ax.transAxes, verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
            
            plt.tight_layout()
            
            # Save the plot
            save_path = os.path.join(output_dir, f"pca_cell_features_{uuid4().hex[:8]}.png")
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"PCA visualization saved to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Error creating PCA visualization: {e}")
            return None
    
    def _create_umap_visualization(self, results: List[Dict], features: torch.Tensor, output_dir: str) -> Optional[str]:
        """Generate a UMAP visualization from extracted features using anndata and scanpy."""
        if not ANNDATA_AVAILABLE:
            logger.warning("UMAP visualization requested but anndata/scanpy not available")
            return None
            
        try:
            logger.info("Generating UMAP visualization...")
            
            # Create AnnData object
            adata = anndata.AnnData(X=features.numpy())
            adata.obs['predicted_class'] = [r['predicted_class'] for r in results]
            adata.obs['image_name'] = [Path(r['image_path']).name for r in results]

            # Run Scanpy workflow with adaptive PCA components
            n_samples = features.shape[0]
            n_pcs = min(50, n_samples - 1)  # Ensure n_pcs <= n_samples - 1
            
            if n_pcs < 2:
                logger.warning(f"Not enough samples ({n_samples}) for UMAP visualization. Need at least 3 samples.")
                return None
            
            sc.tl.pca(adata, n_comps=n_pcs)
            sc.pp.neighbors(adata, n_neighbors=min(15, n_samples - 1), n_pcs=n_pcs)
            sc.tl.umap(adata, min_dist=0.5, spread=2.5)
            
            # Plot UMAP
            fig, ax = plt.subplots(figsize=(10, 8))
            sc.pl.umap(adata, color=['predicted_class'], show=False, 
                      palette=self.color_map, size=120, alpha=0.7, title="", ax=ax)

            if ax.get_legend() is not None:
                ax.get_legend().remove()
            
            ax.set_xticks([]); ax.set_yticks([]); ax.set_xlabel(""); ax.set_ylabel("")
            plt.grid(False)
            for spine in ax.spines.values():
                spine.set_visible(False)
            
            # Add title and legend
            ax.set_title("Cell Feature Space (UMAP)", fontsize=14, fontweight='bold')
            
            # Create custom legend
            from matplotlib.patches import Patch
            legend_elements = [Patch(facecolor=color, label=state) 
                             for state, color in self.color_map.items()]
            ax.legend(handles=legend_elements, title="Cell States", 
                     loc='upper right', fontsize=10)
            
            plt.tight_layout()

            # Save the plot
            save_path = os.path.join(output_dir, f"umap_cell_features_{uuid4().hex[:8]}.png")
            plt.savefig(save_path, format='png', dpi=300, bbox_inches='tight')
            plt.close(fig)
            
            logger.info(f"UMAP visualization saved to: {save_path}")
            return save_path

        except Exception as e:
            logger.error(f"Error creating UMAP visualization: {e}")
            return None
    
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
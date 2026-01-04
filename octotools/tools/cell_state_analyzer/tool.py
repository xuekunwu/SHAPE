#!/usr/bin/env python3
"""
Cell State Analyzer Tool - Self-supervised learning for cell state analysis.
Performs contrastive learning on single-cell crops and generates UMAP visualizations with clustering.
"""

import os
import sys
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms
from torch.cuda.amp import GradScaler, autocast
from PIL import Image
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
from uuid import uuid4
import matplotlib.pyplot as plt
import seaborn as sns
# Import hf_hub_download for loading PyTorch model weights
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Model loading may fail.")
import glob
import pandas as pd
from tqdm import tqdm
from collections import Counter

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Try to import anndata and scanpy
try:
    import anndata as ad
    import scanpy as sc
    ANNDATA_AVAILABLE = True
    logger.info("anndata and scanpy are available")
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("anndata and scanpy not available. Install them for advanced visualizations.")

# Try to import skimage for image reading
try:
    from skimage import io
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False
    logger.warning("skimage not available. Using PIL only.")

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from octotools.tools.base import BaseTool
from octotools.models.utils import VisualizationConfig


class CellCropDataset(Dataset):
    """Dataset for single-cell crop images."""
    def __init__(self, image_paths, groups=None, transform=None):
        self.image_paths = image_paths
        self.groups = groups if groups else ["default"] * len(image_paths)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        group = self.groups[idx]
        
        # Load image
        if SKIMAGE_AVAILABLE and (path.lower().endswith('.tif') or path.lower().endswith('.tiff')):
            img = io.imread(path).astype(np.float32)
            if img.dtype == np.uint16:
                img = img / 65535.0
            else:
                img = img / 255.0
            if img.ndim == 2:
                img = np.repeat(img[..., None], 3, axis=-1)
            img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
        else:
            img = Image.open(path).convert("RGB")
        
        # Apply transforms (two augmented views for contrastive learning)
        if self.transform:
            v1 = self.transform(img)
            v2 = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
            v1 = v2 = tensor
        
        image_name = os.path.splitext(os.path.basename(path))[0]
        return v1, v2, image_name, group


class DinoV3Projector(nn.Module):
    """DINOv3 model with projection head for contrastive learning."""
    def __init__(self, backbone_name="dinov3_vits16", proj_dim=256):
        super().__init__()
        
        feat_dim_map = {
            "dinov3_vits16": 384,
            "dinov3_vits16plus": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
            "dinov3_vith16plus": 1280,
            "dinov3_vit7b16": 4096,
        }
        
        # Load DINOv3 backbone from Hugging Face Hub (PyTorch weights file)
        # Using smaller vits16 model (384 dims) to reduce GPU memory usage
        custom_repo_id = "5xuekun/dinov3_vits16"
        model_filename = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        # Use dinov2-small as architecture base since the weights are for ViT-S/16 (384 dimensions)
        architecture_repo_id = "facebook/dinov2-small"
        fallback_repo_id = "facebook/dinov2-small"  # Official DINOv2-small model as fallback
        
        logger.info(f"Attempting to load DINOv3 model weights from Hugging Face Hub: {custom_repo_id}/{model_filename}")
        
        self.backbone = None
        from transformers import AutoModel
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Try custom DINOv3 model first (download .pth weights and load into DINOv2 architecture)
        try:
            if not HF_HUB_AVAILABLE:
                raise ImportError("huggingface_hub not available")
            
            # Download DINOv3 weights file
            logger.info(f"Downloading DINOv3 weights: {model_filename}")
            weights_path = hf_hub_download(
                repo_id=custom_repo_id,
                filename=model_filename,
                token=hf_token
            )
            logger.info(f"DINOv3 weights downloaded to: {weights_path}")
            
            # Load DINOv3 weights into DINOv2 architecture (same ViT architecture)
            # Use dinov2-large architecture since the weights are for ViT-L/16
            logger.info(f"Loading DINOv3 weights into {architecture_repo_id} architecture...")
            base_model = AutoModel.from_pretrained(
                architecture_repo_id,
                token=hf_token,
                trust_remote_code=True
            )
            
            # Load DINOv3 state dict
            logger.info("Loading PyTorch state dict from weights file...")
            state_dict = torch.load(weights_path, map_location='cpu')
            
            # Handle different state dict formats (e.g., {"teacher": {...}} or direct state dict)
            if isinstance(state_dict, dict) and "teacher" in state_dict:
                logger.info("Found 'teacher' key in state dict, extracting teacher weights...")
                state_dict = state_dict["teacher"]
            elif isinstance(state_dict, dict) and "model" in state_dict:
                logger.info("Found 'model' key in state dict, extracting model weights...")
                state_dict = state_dict["model"]
            
            # Filter out keys that don't match (e.g., if DINOv3 has slightly different keys)
            model_state_dict = base_model.state_dict()
            filtered_state_dict = {}
            skipped_keys = []
            for k, v in state_dict.items():
                if k in model_state_dict and v.shape == model_state_dict[k].shape:
                    filtered_state_dict[k] = v
                else:
                    skipped_keys.append(k)
            
            if skipped_keys:
                logger.debug(f"Skipped {len(skipped_keys)} keys that don't match model architecture")
            
            # Load filtered weights (missing keys will use default initialization)
            missing_keys, unexpected_keys = base_model.load_state_dict(filtered_state_dict, strict=False)
            if missing_keys:
                logger.warning(f"Some model keys were not initialized from weights: {len(missing_keys)} keys")
            if unexpected_keys:
                logger.warning(f"Some weight keys were not used: {len(unexpected_keys)} keys")
            
            self.backbone = base_model
            logger.info(f"✅ Successfully loaded DINOv3 model weights from {model_filename}")
            
            # Update feat_dim for DINOv3 ViT-S/16 (384 dimensions)
            # Note: model filename may say vitb16 but it's actually vits16 based on repo
            if 'vits16' in backbone_name.lower() or custom_repo_id.endswith('dinov3_vits16'):
                feat_dim_map[backbone_name] = 384
                logger.info("Detected DINOv3 ViT-S/16 model (384 dimensions) - using for reduced GPU memory")
            elif 'vitl16' in model_filename.lower() or custom_repo_id.endswith('dinov3_vitl16'):
                feat_dim_map[backbone_name] = 1024
                logger.info("Detected DINOv3 ViT-L/16 model (1024 dimensions)")
            
        except Exception as e:
            logger.warning(f"Failed to load custom DINOv3 model ({custom_repo_id}/{model_filename}): {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.info(f"Falling back to official DINOv2 model: {fallback_repo_id}")
            
            # Fallback to official DINOv2 model
            try:
                self.backbone = AutoModel.from_pretrained(
                    fallback_repo_id,
                    token=hf_token,
                    trust_remote_code=True
                )
                logger.info(f"✅ Loaded DINOv2 model from Hugging Face Hub: {fallback_repo_id}")
                # Update feat_dim for DINOv2-small (384 dimensions)
                feat_dim_map[backbone_name] = 384
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback DINOv2 model: {fallback_e}")
                raise ValueError(
                    f"Model loading failed. Tried:\n"
                    f"1. Custom DINOv3: {custom_repo_id}/{model_filename} - {str(e)}\n"
                    f"2. Official DINOv2: {fallback_repo_id} - {str(fallback_e)}\n"
                    f"Please check your internet connection and Hugging Face Hub access."
                )
        
        # Get feature dimension - use map value as default, will be adjusted if needed
        # Default to 384 for vits16 to reduce GPU memory usage
        feat_dim = feat_dim_map.get(backbone_name, 384)
        
        # Try to infer actual feat_dim from loaded model if available
        if self.backbone is not None:
            try:
                # DINOv2/DINOv3 models typically have a config attribute
                if hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'hidden_size'):
                    actual_feat_dim = self.backbone.config.hidden_size
                    logger.info(f"Model hidden_size: {actual_feat_dim}, using for projection head")
                    feat_dim = actual_feat_dim
            except Exception as e:
                logger.debug(f"Could not infer feat_dim from model config: {e}, using default: {feat_dim}")
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )
    
    def forward(self, x):
        # DINOv3 backbone directly returns CLS token as tensor (matching training script)
        z = self.backbone(x)
        
        # Handle tensor output - if multi-dimensional, extract CLS token
        if isinstance(z, torch.Tensor):
            if z.dim() > 2:
                # Shape is [B, N, D], take first token (CLS)
                z = z[:, 0, :]
            # Otherwise it's already CLS token [B, D]
        else:
            # Fallback for other formats (shouldn't happen with DINOv3)
            if hasattr(z, 'last_hidden_state'):
                z = z.last_hidden_state[:, 0]
            elif hasattr(z, 'pooler_output'):
                z = z.pooler_output
        
        # Project and normalize
        z = self.projector(z)
        return F.normalize(z, dim=-1)


def contrastive_loss(z1, z2, temperature=0.1):
    """Compute contrastive loss for self-supervised learning."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return nn.CrossEntropyLoss()(logits, labels)


class Cell_State_Analyzer_Tool(BaseTool):
    """
    Analyzes cell states using self-supervised learning (contrastive learning).
    Performs training on single-cell crops and generates UMAP visualizations with clustering.
    """
    
    def __init__(self):
        super().__init__(
            tool_name="Cell_State_Analyzer_Tool",
            tool_description="Performs self-supervised learning (contrastive learning) on single-cell crops to analyze cell states. Trains a DINOv3 model and generates UMAP visualizations with clustering. Supports multi-group analysis.",
            tool_version="1.0.0",
            input_types={
                "cell_crops": "List[str] - List of cell crop image paths from Single_Cell_Cropper_Tool output.",
                "cell_metadata": "List[dict] - List of metadata dictionaries for each cell (must include 'group' field for multi-group analysis).",
                "max_epochs": "int - Maximum number of training epochs (default: 100).",
                "early_stop_loss": "float - Early stopping threshold for loss (default: 0.5). Training stops if loss <= this value.",
                "batch_size": "int - Batch size for training (default: 16).",
                "learning_rate": "float - Learning rate for optimizer (default: 3e-5).",
                "cluster_resolution": "float - Resolution for Leiden clustering (default: 0.5).",
                "query_cache_dir": "str - Directory for caching results.",
            },
            output_type="dict - Analysis results with training history, UMAP visualizations, and cluster analysis.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(cell_crops=cell_crops, cell_metadata=cell_metadata)',
                    "description": "Train model and generate UMAP visualizations with default parameters."
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for training. Requires anndata and scanpy for advanced visualizations. Training time depends on number of cells and epochs.",
                "best_practice": "Use with output from Single_Cell_Cropper_Tool. Include 'group' field in cell_metadata for multi-group cluster composition analysis. Adjust cluster_resolution based on your data granularity needs."
            },
            output_dir="output_visualizations"
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Cell_State_Analyzer_Tool: Using device: {self.device}")
    
    def _get_augmentation_transform(self):
        """Get augmentation transform for training."""
        return transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.3, 1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(60),
            transforms.RandomApply([transforms.ColorJitter(0.6, 0.6, 0.6, 0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1, 2.0))], p=0.8),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
            transforms.Normalize([0.4636, 0.5032, 0.5822], [0.2483, 0.2660, 0.2907])
        ])
    
    def _get_eval_transform(self):
        """Get evaluation transform (no augmentation)."""
        return transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize([0.4636, 0.5032, 0.5822], [0.2483, 0.2660, 0.2907])
        ])
    
    def _extract_features(self, model, dataloader, device):
        """Extract CLS features from the model backbone.
        
        Following the training script pattern: DINOv3 backbone directly returns CLS token as tensor.
        """
        model.eval()
        feats, img_names, groups = [], [], []
        proj_backup = model.projector
        model.projector = torch.nn.Identity()
        
        with torch.no_grad():
            for v1, _, names, grps in tqdm(dataloader, desc="Extracting features"):
                v1 = v1.to(device)
                # DINOv3 backbone directly returns CLS token (matching training script pattern)
                out = model.backbone(v1)
                
                # Handle tensor output (DINOv3 returns tensor directly)
                if isinstance(out, torch.Tensor):
                    # If shape is [B, N, D], take first token (CLS)
                    # Otherwise it's already CLS token [B, D]
                    if out.dim() > 2:
                        out = out[:, 0, :]
                else:
                    # Fallback for other output formats (shouldn't happen with DINOv3)
                    logger.warning(f"Unexpected output type from backbone: {type(out)}, trying to extract CLS token...")
                    if hasattr(out, 'last_hidden_state'):
                        out = out.last_hidden_state[:, 0]
                    elif hasattr(out, 'pooler_output'):
                        out = out.pooler_output
                    elif isinstance(out, dict):
                        # Try common keys
                        if 'x_norm_clstoken' in out:
                            out = out['x_norm_clstoken']
                        elif 'last_hidden_state' in out:
                            out = out['last_hidden_state'][:, 0]
                        else:
                            raise ValueError(f"Could not extract features from output dict. Available keys: {list(out.keys())}")
                    else:
                        raise ValueError(f"Unexpected output type from backbone: {type(out)}")
                
                feats.append(out.cpu())
                img_names.extend(names)
                groups.extend(grps)
        
        model.projector = proj_backup
        feats_all = torch.cat(feats, dim=0).numpy()
        logger.info(f"✅ Extracted CLS features: {feats_all.shape}")
        return feats_all, img_names, groups
    
    def _train_model(self, model, train_loader, max_epochs, early_stop_loss, learning_rate, output_dir, training_logs=None):
        """Train the model with contrastive learning."""
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = GradScaler()
        
        history = []
        best_loss = float("inf")
        best_model_path = os.path.join(output_dir, "best_model.pth")
        
        # Initialize training logs list if not provided
        if training_logs is None:
            training_logs = []
        
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            num_batches = 0
            
            for v1, v2, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
                v1, v2 = v1.to(self.device), v2.to(self.device)
                optimizer.zero_grad()
                
                with autocast():
                    z1, z2 = model(v1), model(v2)
                    loss = contrastive_loss(z1, z2)
                
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                
                total_loss += loss.item()
                num_batches += 1
            
            avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
            history.append(avg_loss)
            
            # Log epoch progress
            epoch_log = f"Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}"
            logger.info(epoch_log)
            training_logs.append(epoch_log)
            
            # Save best model
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                best_model_log = f"🔥 New best model saved (Loss = {best_loss:.4f})"
                logger.info(best_model_log)
                training_logs.append(best_model_log)
            
            # Early stopping
            if avg_loss <= early_stop_loss:
                early_stop_log = f"✅ Early stopping triggered: Loss = {avg_loss:.4f} <= {early_stop_loss}"
                logger.info(early_stop_log)
                training_logs.append(early_stop_log)
                break
        
        return history, best_model_path, training_logs
    
    def _create_loss_curve(self, history, output_dir):
        """Create and save loss curve plot."""
        fig, ax = VisualizationConfig.create_professional_figure(figsize=(10, 6))
        ax.plot(range(1, len(history) + 1), history, 'b-', linewidth=2)
        ax.set_xlabel("Epoch", fontsize=12)
        ax.set_ylabel("Contrastive Loss", fontsize=12)
        VisualizationConfig.apply_professional_styling(ax, title="Training Loss Curve")
        ax.grid(True, alpha=0.3)
        
        output_path = os.path.join(output_dir, "loss_curve.png")
        VisualizationConfig.save_professional_figure(fig, output_path)
        plt.close(fig)
        logger.info(f"✅ Loss curve saved to {output_path}")
        return output_path
    
    def _compute_umap_and_clustering(self, adata, resolution, groups=None):
        """Compute UMAP coordinates and perform clustering (no visualization)."""
        if not ANNDATA_AVAILABLE:
            logger.warning("anndata/scanpy not available. Cannot compute UMAP and clustering.")
            return None
        
        # Compute neighbors and UMAP
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=20, metric="cosine")
        sc.tl.umap(adata, min_dist=0.05, spread=1, random_state=42)
        
        # Perform clustering
        cluster_key = f"leiden_{resolution}"
        sc.tl.leiden(adata, resolution=resolution, key_added=cluster_key)
        
        logger.info(f"✅ Computed UMAP and Leiden clustering (resolution={resolution})")
        logger.info(f"✅ Found {len(adata.obs[cluster_key].unique())} clusters")
        
        return cluster_key
    
    def _load_cell_data_from_metadata(self, query_cache_dir):
        """Load cell crops and metadata from metadata files."""
        metadata_dir = os.path.join(query_cache_dir, "tool_cache")
        metadata_files = glob.glob(os.path.join(metadata_dir, 'cell_crops_metadata_*.json'))
        
        if not metadata_files:
            raise ValueError(f"No metadata files found in {metadata_dir}")
        
        latest_metadata_file = max(metadata_files, key=os.path.getctime)
        logger.info(f"Loading metadata from: {latest_metadata_file}")
        
        with open(latest_metadata_file, 'r') as f:
            data = json.load(f)
        
        cell_crops = data.get('cell_crops_paths', [])
        cell_metadata = data.get('cell_metadata', [])
        
        # Normalize paths
        cell_crops = [os.path.normpath(path) for path in cell_crops]
        
        return cell_crops, cell_metadata
    
    def execute(self, cell_crops=None, cell_metadata=None, max_epochs=100, early_stop_loss=0.5,
                batch_size=16, learning_rate=3e-5, cluster_resolution=0.5, query_cache_dir=None):
        """
        Execute self-supervised learning training and analysis.
        
        Args:
            cell_crops: List of cell crop image paths
            cell_metadata: List of metadata dictionaries (should include 'group' field)
            max_epochs: Maximum training epochs (default: 100)
            early_stop_loss: Early stopping loss threshold (default: 0.5)
            batch_size: Training batch size (default: 16)
            learning_rate: Learning rate (default: 3e-5)
            cluster_resolution: Leiden clustering resolution (default: 0.5)
            query_cache_dir: Directory for outputs
            
        Returns:
            dict: Analysis results with visualizations and AnnData object
        """
        logger.info("🚀 Cell_State_Analyzer_Tool starting execution...")
        
        # Load cell data if not provided
        if cell_crops is None or cell_metadata is None:
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            logger.info(f"Loading cell data from metadata in: {query_cache_dir}")
            cell_crops, cell_metadata = self._load_cell_data_from_metadata(query_cache_dir)
        
        if not cell_crops or len(cell_crops) == 0:
            return {"error": "No cell crops found for analysis", "status": "failed"}
        
        logger.info(f"🔬 Processing {len(cell_crops)} cell crops...")
        
        # Setup output directory
        if query_cache_dir is None:
            query_cache_dir = "solver_cache/temp"
        output_dir = os.path.join(query_cache_dir, "cell_state_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract groups from metadata
        groups = []
        for meta in cell_metadata:
            if isinstance(meta, dict) and 'group' in meta:
                groups.append(meta['group'])
            else:
                groups.append("default")
        
        # Create datasets
        train_transform = self._get_augmentation_transform()
        eval_transform = self._get_eval_transform()
        
        train_dataset = CellCropDataset(cell_crops, groups, transform=train_transform)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                 num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        
        eval_dataset = CellCropDataset(cell_crops, groups, transform=eval_transform)
        eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
        
        logger.info(f"✅ Loaded {len(train_dataset)} images")
        
        # Initialize model
        model = DinoV3Projector(backbone_name="dinov3_vits16", proj_dim=256).to(self.device)
        
        # Train model
        logger.info("🎯 Starting training...")
        training_logs = []  # Collect training progress logs
        history, best_model_path, training_logs = self._train_model(
            model, train_loader, max_epochs, early_stop_loss, learning_rate, output_dir, training_logs
        )
        
        # Load best model
        model.load_state_dict(torch.load(best_model_path))
        logger.info(f"✅ Loaded best model from {best_model_path}")
        
        # Extract features
        logger.info("🔍 Extracting features...")
        feats, img_names, groups_extracted = self._extract_features(model, eval_loader, self.device)
        
        # Create loss curve
        loss_curve_path = self._create_loss_curve(history, output_dir)
        
        # Create AnnData object
        if not ANNDATA_AVAILABLE:
            return {
                "error": "anndata/scanpy not available. Please install them for visualization.",
                "loss_curve": loss_curve_path,
                "training_history": history
            }
        
        adata = ad.AnnData(X=feats)
        adata.obs["image_name"] = img_names
        adata.obs["group"] = groups_extracted
        adata.obs.index = [f"cell_{i}" for i in range(adata.n_obs)]
        
        # Compute UMAP and clustering (no visualization - handled by Analysis_Visualizer_Tool)
        cluster_key = self._compute_umap_and_clustering(adata, cluster_resolution, groups_extracted)
        
        # Save AnnData (contains UMAP coordinates and cluster assignments)
        adata_path = os.path.join(output_dir, "cell_state_analyzed.h5ad")
        adata.write(adata_path)
        logger.info(f"✅ AnnData saved to {adata_path}")
        
        # Prepare output - visualization will be handled by Analysis_Visualizer_Tool
        visual_outputs = [loss_curve_path]
        
        # Format training logs for display
        training_logs_text = "\n".join(training_logs) if training_logs else ""
        
        summary = f"Training completed. Processed {len(cell_crops)} cells in {len(history)} epochs. Final loss: {history[-1]:.4f}"
        if training_logs_text:
            summary = f"{summary}\n\n**Training Progress:**\n```\n{training_logs_text}\n```"
        
        return {
            "summary": summary,
            "cell_count": len(cell_crops),
            "epochs_trained": len(history),
            "final_loss": history[-1],
            "best_loss": min(history),
            "loss_curve": loss_curve_path,
            "adata_path": adata_path,  # AnnData file path for Analysis_Visualizer_Tool
            "visual_outputs": visual_outputs,
            "training_history": history,
            "training_logs": training_logs_text,  # Include training logs for display
            "cluster_key": cluster_key,  # Cluster column name (e.g., "leiden_0.5")
            "cluster_resolution": cluster_resolution,  # Resolution used for clustering
            "analysis_type": "cell_state_analysis"  # Flag for Analysis_Visualizer_Tool to detect this output
        }


if __name__ == "__main__":
    # Test script
    tool = Cell_State_Analyzer_Tool()
    print("Tool initialized successfully")

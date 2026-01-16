#!/usr/bin/env python3
"""
Cell State Analyzer Multi-Channel Tool - Self-supervised learning for multi-channel cell state analysis.
Supports arbitrary number of channels (2, 3, 4, 5+).
"""

import os
import sys
import random
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader
from torch.cuda.amp import GradScaler, autocast
import numpy as np
import json
import logging
from pathlib import Path
from typing import List, Dict, Optional
from tqdm import tqdm
import glob

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

from huggingface_hub import hf_hub_download
import anndata as ad
import scanpy as sc
from skimage import io
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)
from octotools.tools.base import BaseTool
from octotools.models.utils import VisualizationConfig
from octotools.utils.image_processor import ImageProcessor
import matplotlib.pyplot as plt

# ====================== Normalization ======================
def channel_norm(x, eps=1e-6):
    x = x.clone()
    for c in range(x.shape[0]):
        v = x[c]
        p1, p99 = torch.quantile(v, 0.01), torch.quantile(v, 0.99)
        x[c] = torch.clamp((v - p1) / (p99 - p1 + eps), 0, 1)
    return x

# ====================== Dataset ======================
class MultiChannelCellCropDataset(Dataset):
    def __init__(self, image_paths, groups=None, transform=None, selected_channels=None):
        self.image_paths = image_paths
        self.groups = groups if groups else ["default"] * len(image_paths)
        self.transform = transform
        self.selected_channels = selected_channels
        self.detected_channels = None  # Will be set when first image is loaded
        
    def __len__(self):
        return len(self.image_paths)
    
    def get_detected_channels(self):
        """Get detected number of channels from first image."""
        if self.detected_channels is None:
            if not self.image_paths:
                raise ValueError("No images in dataset")
            
            img = io.imread(self.image_paths[0])
            if img.ndim == 2:
                raise ValueError(
                    f"Single-channel image detected at {self.image_paths[0]}. "
                    f"Please use Cell_State_Analyzer_Single_Tool for single-channel images."
                )
            elif img.ndim == 3:
                # Image format is fixed as (C, H, W) - channels in first dimension
                self.detected_channels = img.shape[0]
        return self.detected_channels
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        group = self.groups[idx]
        
        # Load image using io.imread (works for TIFF, PNG, JPEG)
        img = io.imread(path)
        
        if img.ndim == 2:
            raise ValueError(
                f"Single-channel image detected at {path}. "
                f"Please use Cell_State_Analyzer_Single_Tool for single-channel images."
            )
        # io.imread returns (C, H, W) format for TIFF files saved by Single_Cell_Cropper_Tool
        
        # Apply channel selection
        if self.selected_channels is not None:
            img = img[self.selected_channels]

        # Normalize
        img = torch.from_numpy(img).float()
        img = channel_norm(img)
        img = (img - 0.5) / 0.5

        # Apply transforms
        if self.transform:
            v1 = self.transform(img)
            v2 = self.transform(img)
        else:
            v1 = v2 = img

        return v1, v2, Path(path).stem, group


# ====================== Transform ======================
class MultiChannelTransform(nn.Module):
    """Multi-channel transform for augmentation (works on (C, H, W) tensors)."""
    def __init__(self, out_size=224, train=True, hflip_p=0.5, vflip_p=0.2, erase_p=0.3):
        super().__init__()
        self.out_size = out_size
        self.train = train
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.erase_p = erase_p
    
    def forward(self, x):
        # Resize to target size
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)
        
        if self.train:
            if random.random() < self.hflip_p:
                x = torch.flip(x, dims=[2])
            if random.random() < self.vflip_p:
                x = torch.flip(x, dims=[1])
            if self.erase_p > 0 and random.random() < self.erase_p:
                C, H, W = x.shape
                eh, ew = int(H * random.uniform(0.02, 0.2)), int(W * random.uniform(0.02, 0.2))
                y, x0 = random.randint(0, max(1, H - eh)), random.randint(0, max(1, W - ew))
                x[:, y:y+eh, x0:x0+ew] = 0
        
        return x

# ====================== Model ======================
class DinoV3Projector(nn.Module):
    def __init__(self, backbone_name="dinov3_vitb16", proj_dim=256, in_channels=None,
                 freeze_patch_embed=False, freeze_blocks=0):
        super().__init__()
        if in_channels is None:
            raise ValueError("in_channels must be specified (cannot be None). Channel count should be detected from input images.")
        self.in_channels = in_channels
        
        # Download weights from Hugging Face Hub
        custom_repo_id = "5xuekun/dinov3_vits16"
        model_filename = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        logger.info(f"Downloading weights from Hugging Face Hub: {custom_repo_id}/{model_filename}")
        ckpt_path = hf_hub_download(
            repo_id=custom_repo_id,
            filename=model_filename,
            token=hf_token
        )
        
        # Load model architecture from GitHub (facebookresearch/dinov3 contains hubconf.py)
        hub_repo = "facebookresearch/dinov3"
        logger.info(f"Loading model architecture from GitHub: {hub_repo}")
        self.backbone = torch.hub.load(hub_repo, backbone_name, pretrained=False, source="github")
        
        # Load weights
        logger.info(f"Loading weights from local path: {ckpt_path}")
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        self.backbone.load_state_dict(state_dict, strict=False)
        
        # Adapt patch embedding for multi-channel input
        self._adapt_patch_embedding(in_channels)
        
        # Freeze layers if requested
        if freeze_patch_embed:
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False
        
        if freeze_blocks > 0:
            for blk in self.backbone.blocks[:freeze_blocks]:
                for p in blk.parameters():
                    p.requires_grad = False
        
        # Projection head - use feat_dim_map like reference code
        feat_dim_map = {
            "dinov3_vits16": 384,
            "dinov3_vits16plus": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
            "dinov3_vith16plus": 1280,
            "dinov3_vit7b16": 4096,
        }
        feat_dim = feat_dim_map[backbone_name]
        
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
    
    def _adapt_patch_embedding(self, in_channels):
        """Adapt patch embedding layer for multi-channel input."""
        patch_embed = self.backbone.patch_embed
        old_proj = patch_embed.proj

        if old_proj.in_channels != in_channels:
            new_proj = nn.Conv2d(
                in_channels=in_channels,
                out_channels=old_proj.out_channels,
                kernel_size=old_proj.kernel_size,
                stride=old_proj.stride,
                padding=old_proj.padding,
                bias=old_proj.bias is not None,
            )

            with torch.no_grad():
                # inherit as much as possible from RGB weights
                c = min(old_proj.in_channels, in_channels)
                new_proj.weight[:, :c] = old_proj.weight[:, :c]
                if old_proj.bias is not None:
                    new_proj.bias.copy_(old_proj.bias)

            patch_embed.proj = new_proj
            logger.info(f"✅ Adapted patch embedding: {old_proj.in_channels} -> {in_channels} channels")
    
    def forward(self, x):
        feats = self.backbone(x)  # CLS token
        z = self.projector(feats)
        return F.normalize(z, dim=-1)


def contrastive_loss(z1, z2, temperature=0.1):
    """Compute contrastive loss."""
    z1, z2 = F.normalize(z1, dim=1), F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    return nn.CrossEntropyLoss()(logits, labels)


# ====================== Tool ======================
class Cell_State_Analyzer_Multi_Tool(BaseTool):
    """Cell state analyzer for multi-channel images. Supports arbitrary number of channels."""
    
    def __init__(self):
        super().__init__(
            tool_name="Cell_State_Analyzer_Multi_Tool",
            tool_description="Performs self-supervised learning (contrastive learning) on individual cell/organoid crops to analyze cell states. Analyzes pre-cropped single-cell/organoid images from Single_Cell_Cropper_Tool. Performs feature extraction, UMAP embedding, and clustering. Supports 2+ channel multi-channel images. REQUIRES: Single_Cell_Cropper_Tool must be executed first to generate individual crop images.",
            tool_version="2.0.0",
            input_types={
                "cell_crops": "List[str] - REQUIRED: Individual cell/organoid crop image paths from Single_Cell_Cropper_Tool output. Each path should be a cropped image of a single cell/organoid.",
                "cell_metadata": "List[dict] - Metadata with 'group' field for each crop (must match cell_crops length)",
                "max_epochs": "int - Max training epochs (default: 25)",
                "early_stop_loss": "float - Early stopping threshold (default: 0.5)",
                "batch_size": "int - Batch size (default: 16)",
                "learning_rate": "float - Learning rate (default: 3e-5)",
                "cluster_resolution": "float - Clustering resolution (default: 0.5)",
                "query_cache_dir": "str - Output directory",
            },
            output_type="dict - Analysis results with AnnData object containing UMAP coordinates and cluster assignments",
            demo_commands=[{
                "command": "tool.execute(cell_crops=crop_paths, cell_metadata=metadata)",
                "description": "Analyze pre-cropped cell/organoid images from Single_Cell_Cropper_Tool"
            }],
            user_metadata={
                "limitation": "Requires GPU for training. Supports 2+ channels. REQUIRES Single_Cell_Cropper_Tool output as input - cannot process raw images or segmentation masks.",
                "best_practice": "MUST use output from Single_Cell_Cropper_Tool. Input should be individual crop images, not original images or masks. Include 'group' field in cell_metadata for multi-group analysis."
            },
            output_dir="output_visualizations"
        )
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    def _get_transform(self, train=True):
        """Get transform for training or evaluation."""
        return MultiChannelTransform(out_size=224, train=train, hflip_p=0.5 if train else 0.0,
                                   vflip_p=0.2 if train else 0.0, erase_p=0.3 if train else 0.0)
    
    def _extract_features(self, model, dataloader):
        """Extract CLS features from model."""
        model.eval()
        feats, names, groups = [], [], []
        proj_backup = model.projector
        model.projector = nn.Identity()
        
        with torch.no_grad():
            for v1, _, n, g in tqdm(dataloader, desc="Extracting features"):
                v1 = v1.to(self.device)
                out = model.backbone(v1)  # CLS token
                feats.append(out.cpu())
                names.extend(n)
                groups.extend(g)
        
        model.projector = proj_backup
        return torch.cat(feats, dim=0).numpy(), names, groups
    
    def _train_model(self, model, train_loader, max_epochs, early_stop_loss, lr, output_dir, patience=5):
        """Train model with contrastive learning."""
        optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
        scaler = GradScaler()
        history, best_loss = [], float("inf")
        best_path = os.path.join(output_dir, "best_model.pth")
        no_improve = 0
        
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            for v1, v2, _, _ in tqdm(train_loader, desc=f"Epoch {epoch+1}/{max_epochs}"):
                v1, v2 = v1.to(self.device), v2.to(self.device)
                optimizer.zero_grad()
                with autocast():
                    loss = contrastive_loss(model(v1), model(v2))
                scaler.scale(loss).backward()
                scaler.step(optimizer)
                scaler.update()
                total_loss += loss.item()
            
            avg_loss = total_loss / len(train_loader)
            history.append(avg_loss)
            logger.info(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_path)
                no_improve = 0
            else:
                no_improve += 1
            
            if avg_loss <= early_stop_loss or no_improve >= patience:
                break
            
        return history, best_path
    
    def _load_cell_data_from_metadata(self, query_cache_dir):
        """Load cell crops and metadata from metadata files."""
        # Normalize query_cache_dir (remove 'tool_cache' if present)
        if query_cache_dir.endswith('tool_cache') or query_cache_dir.endswith(os.path.sep + 'tool_cache'):
            query_cache_dir = os.path.dirname(query_cache_dir.rstrip(os.sep))
        
        tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
        metadata_files = glob.glob(os.path.join(tool_cache_dir, 'cell_crops_metadata_*.json'))
        
        if not metadata_files:
            raise ValueError(f"No metadata files found in {tool_cache_dir}")
        
        # Deduplicate by source_image_id
        metadata_by_image = {}
        for f in metadata_files:
            try:
                with open(f, 'r') as fp:
                    data = json.load(fp)
                img_id = data.get('source_image_id', f"unknown_{len(metadata_by_image)}")
                if img_id not in metadata_by_image:
                    metadata_by_image[img_id] = data
            except Exception as e:
                logger.warning(f"Error reading {f}: {e}")
        
        # Collect all crops and metadata
        # Filter to only include TIFF format files (multi-channel crops from Single_Cell_Cropper_Tool)
        # Exclude PNG/JPEG files (overlays, visualizations, or single-channel crops)
        all_crops, all_metadata, skipped = [], [], []
        for img_id, data in metadata_by_image.items():
            if data.get('execution_status') == 'no_crops_generated':
                skipped.append({'image': img_id})
                continue
                
            crops = data.get('cell_crops_paths', [])
            metadata = data.get('cell_metadata', [])
            group = data.get('group', 'default')
            
            # Filter crops to only include TIFF format files
            tiff_crops = []
            tiff_indices = []
            for idx, crop_path in enumerate(crops):
                if isinstance(crop_path, str):
                    crop_path_lower = crop_path.lower()
                    # Only include TIFF format files (multi-channel crops)
                    if crop_path_lower.endswith('.tif') or crop_path_lower.endswith('.tiff'):
                        tiff_crops.append(crop_path)
                        tiff_indices.append(idx)
            
            if not tiff_crops:
                logger.warning(f"No TIFF format crops found for image {img_id}. Skipping.")
                skipped.append({'image': img_id, 'reason': 'no_tiff_crops'})
                continue
            
            all_crops.extend(tiff_crops)
            # Match metadata to filtered crops
            filtered_metadata = []
            for idx in tiff_indices:
                if idx < len(metadata) and isinstance(metadata[idx], dict):
                    m = metadata[idx].copy()
                    m.setdefault('group', group)
                    m.setdefault('image_name', img_id)
                    filtered_metadata.append(m)
                else:
                    filtered_metadata.append({'group': group, 'image_name': img_id})
            
            all_metadata.extend(filtered_metadata)
        
        if not all_crops:
            raise ValueError("No TIFF format crop files found in metadata. Cell_State_Analyzer_Multi_Tool requires multi-channel TIFF crops from Single_Cell_Cropper_Tool.")
        
        return all_crops, all_metadata, skipped
    
    def execute(self, cell_crops=None, cell_metadata=None, max_epochs=25, early_stop_loss=0.5,
                batch_size=16, learning_rate=3e-5, cluster_resolution=0.5, query_cache_dir=None,
                freeze_patch_embed=False, freeze_blocks=0):
        """Execute analysis."""
        logger.info("🚀 Starting cell state analysis...")
        
        # Load data
        if cell_crops is None or cell_metadata is None:
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            cell_crops, cell_metadata, skipped = self._load_cell_data_from_metadata(query_cache_dir)
        
        if not cell_crops:
            return {"error": "No cell crops found", "status": "failed"}
        
        num_crops = len(cell_crops)
        logger.info(f"Processing {num_crops} cell crops")
        
        # Setup
        use_zero_shot = num_crops < 50
        output_dir = os.path.join(query_cache_dir or "solver_cache/temp", "cell_state_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare groups
        groups = [m.get('group', 'default') if isinstance(m, dict) else 'default' for m in cell_metadata]
        unique_groups = set(groups)
        is_single_group = len(unique_groups) == 1
        
        # Create dataset and detect channels from first image
        eval_dataset = MultiChannelCellCropDataset(
            cell_crops, groups, 
            transform=self._get_transform(train=False),
            selected_channels=None  # Will be set after detection
        )
        
        # Detect channels using dataset's method
        detected_channels = eval_dataset.get_detected_channels()
        logger.info(f"Auto-detected {detected_channels} channels from first crop")
        
        # Use all detected channels
        selected_channels = list(range(detected_channels))
        eval_dataset.selected_channels = selected_channels
        eval_loader = DataLoader(eval_dataset, batch_size=min(batch_size, num_crops), shuffle=False, num_workers=0)
        
        # Initialize model with detected channels
        model = DinoV3Projector(
            backbone_name="dinov3_vitb16",
            proj_dim=256,
            in_channels=detected_channels,
            freeze_patch_embed=freeze_patch_embed,
            freeze_blocks=freeze_blocks
        ).to(self.device)
        
        # Train or zero-shot
        if use_zero_shot:
            logger.info("Using zero-shot inference (no training)")
            history, loss_path = [], None
        else:
            logger.info("Starting training...")
            train_dataset = MultiChannelCellCropDataset(
                cell_crops, groups,
                transform=self._get_transform(train=True),
                selected_channels=selected_channels
            )
            train_dataset.detected_channels = detected_channels
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            history, best_path = self._train_model(model, train_loader, max_epochs, early_stop_loss,
                                                  learning_rate, output_dir)
            model.load_state_dict(torch.load(best_path))
            
            # Save loss curve
            if history:
                fig, ax = plt.subplots(figsize=(10, 6))
                ax.plot(range(1, len(history) + 1), history)
                ax.set_xlabel("Epoch")
                ax.set_ylabel("Loss")
                ax.set_title("Training Loss")
                loss_path = os.path.join(output_dir, "loss_curve.png")
                plt.savefig(loss_path)
                plt.close()
            
            # Extract features
        feats, names, groups_extracted = self._extract_features(model, eval_loader)
        
        # Create AnnData
        adata = ad.AnnData(X=feats)
        adata.obs["group"] = groups_extracted if len(groups_extracted) == adata.n_obs else groups[:adata.n_obs]
        adata.obs["image_name"] = names if len(names) == adata.n_obs else ['unknown'] * adata.n_obs
        adata.obs.index = [f"cell_{i}" for i in range(adata.n_obs)]
        
        # Compute UMAP and clustering
        sc.pp.neighbors(adata, use_rep='X', n_neighbors=20, metric="cosine")
        sc.tl.umap(adata, min_dist=0.05, spread=1, random_state=42)
        cluster_key = f"leiden_{cluster_resolution}"
        sc.tl.leiden(adata, resolution=cluster_resolution, key_added=cluster_key)
        
        # Save
        adata_path = os.path.join(output_dir, "cell_state_analyzed.h5ad")
        adata.write(adata_path)
        
        # Generate summary based on analysis type
        # Check if results are sufficient to answer group comparison queries
        insufficient_for_comparison = False
        comparison_limitations = []
        termination_recommended = False
        termination_reason = None
        
        if is_single_group:
            insufficient_for_comparison = True
            comparison_limitations.append("All cells are labeled as 'default' - no group labels available for comparison")
            # If query requires group comparison, recommend termination
            termination_recommended = True
            termination_reason = (
                "**Cannot perform group comparison analysis:** All cells are in a single group ('default'). "
                "Group comparison queries cannot be answered with the current data.\n\n"
                "**Recommendation:** Terminate analysis and inform user that group labels are required for comparison queries.\n\n"
                "**Solution:** Provide treatment/group labels when uploading images to enable group comparison analysis."
            )
        
        if num_crops < 10:
            insufficient_for_comparison = True
            comparison_limitations.append(f"Very small sample size ({num_crops} cells) - may be insufficient for robust statistical comparisons")
        
        if use_zero_shot and not history:
            comparison_limitations.append("Zero-shot inference mode (no training) - features may not be optimal for comparison")
        
        if is_single_group:
            summary = f"Single-group analysis completed. {num_crops} cells analyzed. "
            summary += f"{len(history)} epochs trained" if history else "Zero-shot inference (no training)"
            summary += "\n\n⚠️ **Limitations for Group Comparison Queries:**\n"
            summary += "- All cells are in a single group ('default') - group comparison analysis cannot be performed\n"
            summary += "- To enable group comparison, provide treatment/group labels when uploading images\n"
            summary += "- Current analysis provides single-group clustering and UMAP visualization only"
            if termination_recommended:
                summary += f"\n\n🛑 **Termination Recommendation:**\n{termination_reason}"
        else:
            summary = f"Multi-group analysis completed. {num_crops} cells across {len(unique_groups)} groups. "
            summary += f"{len(history)} epochs trained" if history else "Zero-shot inference (no training)"
            summary += f". Groups: {', '.join(sorted(unique_groups))}"
            if comparison_limitations:
                summary += "\n\n⚠️ **Note:** " + "; ".join(comparison_limitations)
        
        return {
            "summary": summary,
            "cell_count": num_crops,
            "mode": "training" if not use_zero_shot else "zero-shot",
            "epochs_trained": len(history),
            "final_loss": history[-1] if history else None,
            "adata_path": adata_path,
            "deliverables": [adata_path] + ([loss_path] if history and loss_path else []),
            "cluster_key": cluster_key,
            "cluster_resolution": cluster_resolution,
            "analysis_type": "cell_state_analysis",
            "num_groups": len(unique_groups),
            "groups": sorted(unique_groups) if not is_single_group else ["default"],
            "insufficient_for_comparison": insufficient_for_comparison,
            "comparison_limitations": comparison_limitations if comparison_limitations else None,
            "termination_recommended": termination_recommended,
            "termination_reason": termination_reason
        }


if __name__ == "__main__":
    tool = Cell_State_Analyzer_Multi_Tool()
    print("Tool initialized successfully")

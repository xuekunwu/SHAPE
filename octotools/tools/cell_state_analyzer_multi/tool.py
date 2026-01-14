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

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Optional dependencies
try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available")

try:
    import anndata as ad
    import scanpy as sc
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("anndata/scanpy not available")

try:
    from skimage import io
    SKIMAGE_AVAILABLE = True
except ImportError:
    SKIMAGE_AVAILABLE = False

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from octotools.tools.base import BaseTool
from octotools.models.utils import VisualizationConfig
from octotools.utils.image_processor import ImageProcessor
import matplotlib.pyplot as plt


# ====================== Dataset ======================
class MultiChannelCellCropDataset(Dataset):
    """Dataset for multi-channel cell crop images. Supports arbitrary number of channels."""
    def __init__(self, image_paths, groups=None, transform=None, selected_channels=None):
        self.image_paths = image_paths
        self.groups = groups if groups else ["default"] * len(image_paths)
        self.transform = transform
        self.selected_channels = selected_channels
    
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        group = self.groups[idx]
        
        # Load image using ImageProcessor (handles all formats)
        try:
            img_data = ImageProcessor.load_image(path)
            img = img_data.data  # (H, W, C)
            if img.ndim == 2:
                img = img[:, :, np.newaxis]
            # Convert to (C, H, W) for PyTorch
            img = np.transpose(img, (2, 0, 1)) if img.ndim == 3 else img
        except Exception as e:
            if SKIMAGE_AVAILABLE:
                img = io.imread(path)
                if img.ndim == 2:
                    img = img[None, ...]
                elif img.ndim == 3:
                    h, w, c = img.shape
                    img = np.transpose(img, (2, 0, 1)) if (c <= 4 and h > c) else img
            else:
                raise ValueError(f"Failed to load {path}: {e}")
        
        # Select channels if specified
        if self.selected_channels is not None:
            img = img[self.selected_channels]
        
        # Normalize to [-1, 1]
        img = torch.from_numpy(img).float()
        if img.max() > 1:
            img = img / 255.0
        img = (img - 0.5) / 0.5
        
        # Apply transforms (two views for contrastive learning)
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
    """DINOv3 model with projection head. Supports arbitrary number of input channels."""
    def __init__(self, backbone_name="dinov3_vits16", proj_dim=256, in_channels=3,
                 freeze_patch_embed=False, freeze_blocks=0):
        super().__init__()
        self.in_channels = in_channels
        
        feat_dim_map = {
            "dinov3_vits16": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
        }
        
        # Load backbone
        self.backbone = self._load_backbone(backbone_name)
        
        # CRITICAL: Update config immediately after loading to prevent validation errors
        # Transformers may validate input channels during forward, so config must match
        if hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'num_channels'):
            if self.backbone.config.num_channels != in_channels:
                logger.info(f"Updating model config.num_channels before adaptation: {self.backbone.config.num_channels} -> {in_channels}")
                self.backbone.config.num_channels = in_channels
        
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
        
        # Projection head
        feat_dim = feat_dim_map.get(backbone_name, 384)
        if hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'hidden_size'):
            feat_dim = self.backbone.config.hidden_size
        
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, feat_dim),
            nn.GELU(),
            nn.Linear(feat_dim, proj_dim),
        )
    
    def _load_backbone(self, backbone_name):
        """Load DINOv3 backbone from Hugging Face Hub."""
        from transformers import AutoModel
        
        custom_repo_id = "5xuekun/dinov3_vits16"
        model_filename = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        architecture_repo_id = "facebook/dinov2-small"
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        try:
            if HF_HUB_AVAILABLE:
                weights_path = hf_hub_download(custom_repo_id, model_filename, token=hf_token)
                base_model = AutoModel.from_pretrained(architecture_repo_id, token=hf_token, trust_remote_code=True)
                
                state_dict = torch.load(weights_path, map_location='cpu')
                if isinstance(state_dict, dict):
                    state_dict = state_dict.get("teacher") or state_dict.get("model") or state_dict
                
                # Filter matching keys
                model_dict = base_model.state_dict()
                filtered_dict = {k: v for k, v in state_dict.items() 
                               if k in model_dict and v.shape == model_dict[k].shape}
                
                base_model.load_state_dict(filtered_dict, strict=False)
                logger.info(f"✅ Loaded DINOv3 weights from {model_filename}")
                return base_model
        except Exception as e:
            logger.warning(f"Failed to load custom model: {e}, using fallback")
        
        # Fallback
        return AutoModel.from_pretrained(architecture_repo_id, token=hf_token, trust_remote_code=True)
    
    def _adapt_patch_embedding(self, in_channels):
        """Adapt patch embedding layer for multi-channel input."""
        # CRITICAL: Update config FIRST before any operations
        # This prevents transformers from validating with wrong channel count
        if hasattr(self.backbone, 'config'):
            if hasattr(self.backbone.config, 'num_channels'):
                original_channels = self.backbone.config.num_channels
                self.backbone.config.num_channels = in_channels
                logger.info(f"Updated backbone.config.num_channels: {original_channels} -> {in_channels}")
        
        # Find patch embedding
        patch_embed = (self.backbone.patch_embed if hasattr(self.backbone, 'patch_embed') else
                      (self.backbone.embeddings.patch_embeddings if hasattr(self.backbone, 'embeddings') else None))
        
        if patch_embed is None:
            logger.warning("Could not find patch embedding layer")
            return
        
        old_proj = getattr(patch_embed, 'proj', None) or getattr(patch_embed, 'projection', None)
        if old_proj is None:
            logger.warning("Could not find projection layer in patch embedding")
            return
        
        if old_proj.in_channels == in_channels:
            logger.info(f"Patch embedding already has {in_channels} channels, no adaptation needed")
            return
        
        # Create new projection layer
        new_proj = nn.Conv2d(
            in_channels=in_channels,
            out_channels=old_proj.out_channels,
            kernel_size=old_proj.kernel_size,
            stride=old_proj.stride,
            padding=old_proj.padding,
            bias=old_proj.bias is not None,
        )
        
        # Inherit weights
        with torch.no_grad():
            c = min(old_proj.in_channels, in_channels)
            new_proj.weight[:, :c] = old_proj.weight[:, :c]
            if old_proj.bias is not None:
                new_proj.bias.copy_(old_proj.bias)
        
        # Replace layer
        if hasattr(patch_embed, 'proj'):
            patch_embed.proj = new_proj
        elif hasattr(patch_embed, 'projection'):
            patch_embed.projection = new_proj
        else:
            logger.warning("Could not find proj or projection attribute in patch_embed")
            return
        
        # Ensure config is still updated (redundant but safe)
        if hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'num_channels'):
            self.backbone.config.num_channels = in_channels
        
        logger.info(f"✅ Adapted patch embedding: {old_proj.in_channels} -> {in_channels} channels")
    
    def forward(self, x):
        """Forward pass: (B, C, H, W) -> (B, proj_dim)"""
        if x.dim() == 3:
            x = x.unsqueeze(0)
        
        if x.shape[1] != self.in_channels:
            raise ValueError(f"Channel mismatch: expected {self.in_channels}, got {x.shape[1]}")
        
        # Forward through backbone - handle transformers models that may validate channels
        try:
            out = self.backbone(x)
        except (ValueError, RuntimeError) as e:
            error_msg = str(e).lower()
            if 'channel' in error_msg or ('expected' in error_msg and 'got' in error_msg):
                # Transformers model may still be checking config - try with pixel_values
                if hasattr(self.backbone, 'forward'):
                    logger.warning(f"Direct call failed ({e}), trying with pixel_values parameter")
                    # Update config one more time before retry
                    if hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'num_channels'):
                        self.backbone.config.num_channels = self.in_channels
                    try:
                        out = self.backbone(pixel_values=x, output_hidden_states=False)
                    except Exception as e2:
                        logger.error(f"Both methods failed: {e2}")
                        raise ValueError(f"Channel adaptation failed: {e}. Tried both direct call and pixel_values.")
                else:
                    raise
            else:
                raise
        
        # Extract CLS token
        if isinstance(out, torch.Tensor):
            feats = out[:, 0, :] if out.dim() == 3 else out
        elif isinstance(out, dict):
            feats = out.get('last_hidden_state', out.get('pooler_output'))
            if feats is not None and feats.dim() == 3:
                feats = feats[:, 0, :]
        elif hasattr(out, 'last_hidden_state'):
            feats = out.last_hidden_state[:, 0]
        elif hasattr(out, 'pooler_output'):
            feats = out.pooler_output
        else:
            raise ValueError(f"Unexpected output format: {type(out)}")
        
        # Project and normalize
        return F.normalize(self.projector(feats), dim=-1)


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
                "in_channels": "int - Number of input channels (auto-detect from first crop if None)",
                "selected_channels": "List[int] - Channel indices to use (all channels if None)",
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
                out = model.backbone(v1)
                
                # Extract CLS token
                if isinstance(out, torch.Tensor):
                    out = out[:, 0, :] if out.dim() == 3 else out
                elif hasattr(out, 'last_hidden_state'):
                    out = out.last_hidden_state[:, 0]
                
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
        
        # Merge all crops
        all_crops, all_metadata, skipped = [], [], []
        for img_id, data in metadata_by_image.items():
            if data.get('execution_status') == 'no_crops_generated':
                skipped.append({'image': img_id})
                continue
            
            crops = data.get('cell_crops_paths', [])
            metadata = data.get('cell_metadata', [])
            group = data.get('group', 'default')
            
            all_crops.extend(crops)
            for m in metadata:
                if isinstance(m, dict):
                    m.setdefault('group', group)
                    m.setdefault('image_name', img_id)
            all_metadata.extend(metadata if metadata else [{'group': group, 'image_name': img_id}] * len(crops))
        
        return all_crops, all_metadata, skipped
    
    def execute(self, cell_crops=None, cell_metadata=None, max_epochs=25, early_stop_loss=0.5,
                batch_size=16, learning_rate=3e-5, cluster_resolution=0.5, query_cache_dir=None,
                in_channels=None, selected_channels=None, freeze_patch_embed=False, freeze_blocks=0):
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
        
        # Detect channels
        if in_channels is None:
            try:
                img_data = ImageProcessor.load_image(cell_crops[0])
                in_channels = img_data.num_channels
                logger.info(f"Auto-detected {in_channels} channels from first crop: {cell_crops[0]}")
                if in_channels <= 0:
                    raise ValueError(f"Invalid channel count: {in_channels}")
            except Exception as e:
                logger.error(f"Failed to detect channels from {cell_crops[0] if cell_crops else 'N/A'}: {e}")
                in_channels = 2
                logger.warning(f"Defaulting to {in_channels} channels. If this is incorrect, specify in_channels parameter.")
        
        # Set selected_channels
        if selected_channels is None:
            selected_channels = list(range(in_channels))
        else:
            selected_channels = [ch for ch in selected_channels if 0 <= ch < in_channels]
            if not selected_channels:
                raise ValueError(f"Invalid selected_channels for {in_channels} channels")
        
        actual_in_channels = len(selected_channels)
        logger.info(f"Using {actual_in_channels} channels: {selected_channels}")
        
        # Setup
        use_zero_shot = num_crops < 50
        output_dir = os.path.join(query_cache_dir or "solver_cache/temp", "cell_state_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Prepare groups
        groups = [m.get('group', 'default') if isinstance(m, dict) else 'default' for m in cell_metadata]
        
        # Create datasets
        eval_dataset = MultiChannelCellCropDataset(cell_crops, groups, 
                                                  transform=self._get_transform(train=False),
                                                  selected_channels=selected_channels)
        eval_loader = DataLoader(eval_dataset, batch_size=min(batch_size, num_crops), shuffle=False, num_workers=0)
        
        # Initialize model
        model = DinoV3Projector(
            backbone_name="dinov3_vits16",
            proj_dim=256,
            in_channels=actual_in_channels,
            freeze_patch_embed=freeze_patch_embed,
            freeze_blocks=freeze_blocks
        ).to(self.device)
        
        # Train or zero-shot
        if use_zero_shot:
            logger.info("Using zero-shot inference (no training)")
            history, loss_path = [], None
        else:
            logger.info("Starting training...")
            train_dataset = MultiChannelCellCropDataset(cell_crops, groups,
                                                       transform=self._get_transform(train=True),
                                                       selected_channels=selected_channels)
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
        if not ANNDATA_AVAILABLE:
            return {"error": "anndata/scanpy not available"}
        
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
        
        return {
            "summary": f"Analysis completed. {num_crops} cells, {len(history)} epochs" if history else f"Zero-shot: {num_crops} cells",
            "cell_count": num_crops,
            "mode": "training" if not use_zero_shot else "zero-shot",
            "epochs_trained": len(history),
            "final_loss": history[-1] if history else None,
            "adata_path": adata_path,
            "deliverables": [adata_path] + ([loss_path] if history and loss_path else []),
            "cluster_key": cluster_key,
            "cluster_resolution": cluster_resolution,
            "analysis_type": "cell_state_analysis"
        }


if __name__ == "__main__":
    tool = Cell_State_Analyzer_Multi_Tool()
    print("Tool initialized successfully")

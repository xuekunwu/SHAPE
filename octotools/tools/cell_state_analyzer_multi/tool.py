#!/usr/bin/env python3
"""
Cell State Analyzer Multi-Channel Tool - Self-supervised learning for multi-channel cell state analysis.
Performs contrastive learning on multi-channel single-cell crops and generates UMAP visualizations with clustering.
Designed for multi-channel images (e.g., 2-channel BF+GFP).
"""

import os
import sys
import random
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

# Set up logger
logger = logging.getLogger(__name__)

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
from octotools.models.image_data import ImageData
from octotools.utils.image_processor import ImageProcessor


class MultiChannelCellCropDataset(Dataset):
    """Dataset for multi-channel single-cell crop images. Directly handles multi-channel data as (C, H, W) tensors."""
    def __init__(self, image_paths, groups=None, transform=None, selected_channels=None):
        self.image_paths = image_paths
        self.groups = groups if groups else ["default"] * len(image_paths)
        self.transform = transform
        self.selected_channels = selected_channels  # e.g., [0, 1] for first 2 channels
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        group = self.groups[idx]
        
        # Load image using unified ImageProcessor (optimal solution: unified interface)
        # ImageProcessor.load_image() returns ImageData with guaranteed (H, W, C) format
        try:
            img_data = ImageProcessor.load_image(path)
            # ImageData.data is always (H, W, C) format where C >= 1
            # - Single-channel: (H, W, 1)
            # - Multi-channel: (H, W, C) where C > 1
            img = img_data.data  # (H, W, C) numpy array
            
            # Convert to (C, H, W) format for PyTorch (matching reference implementation)
            # Reference: 260113_Training_dinov3_CO_screen.py line 42-46
            # Since ImageData always returns (H, W, C), we can directly transpose
            img = np.transpose(img, (2, 0, 1))  # (H, W, C) -> (C, H, W)
            
        except Exception as load_error:
            # Fallback to skimage for direct TIFF loading (legacy support)
            logger.warning(f"Failed to load with ImageProcessor: {load_error}, using skimage fallback")
            if SKIMAGE_AVAILABLE:
                img = io.imread(path)  # May be (H, W, C) or (C, H, W) or (H, W)
                
                # Normalize to (C, H, W) format (matching reference implementation)
                if img.ndim == 2:
                    # (H, W) -> (1, H, W)
                    img = img[None, ...]
                elif img.shape[0] not in (1, 2, 3, 4):
                    # First dimension is large (likely H), assume (H, W, C) -> (C, H, W)
                    img = np.transpose(img, (2, 0, 1))
                # else: already (C, H, W) format
            else:
                raise ValueError(f"Failed to load image {path}: {load_error}. skimage not available.")
        
        # Select channels if specified
        if self.selected_channels is not None:
            img = img[self.selected_channels]
        
        # Convert to float tensor and normalize (matching reference implementation)
        # Reference: 260113_Training_dinov3_CO_screen.py line 51-56
        img = torch.from_numpy(img).float()
        
        # Normalize to [-1, 1] range
        if img.max() > 1:
            img = img / 255.0
        img = (img - 0.5) / 0.5
        
        # Apply transforms (two augmented views for contrastive learning)
        if self.transform:
            v1 = self.transform(img)
            v2 = self.transform(img)
        else:
            v1 = v2 = img
        
        image_name = os.path.splitext(os.path.basename(path))[0]
        return v1, v2, image_name, group


# ====================== Multi-Channel Transform ======================
class MultiChannelTransform(nn.Module):
    """Multi-channel transform for tensor augmentation (works on (C, H, W) tensors)."""
    def __init__(self, out_size=224, train=True, hflip_p=0.5, vflip_p=0.2, erase_p=0.3):
        super().__init__()
        self.out_size = out_size
        self.train = train
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.erase_p = erase_p
    
    def forward(self, x):
        """
        x: (C, H, W) tensor
        Returns: (C, out_size, out_size) tensor
        """
        # Resize (always)
        x = F.interpolate(
            x.unsqueeze(0),  # (C, H, W) -> (1, C, H, W)
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)  # (1, C, out_size, out_size) -> (C, out_size, out_size)
        
        if self.train:
            # Random flips
            if random.random() < self.hflip_p:
                x = torch.flip(x, dims=[2])  # Flip width dimension
            if random.random() < self.vflip_p:
                x = torch.flip(x, dims=[1])  # Flip height dimension
            
            # Optional random erase
            if self.erase_p > 0 and random.random() < self.erase_p:
                C, H, W = x.shape
                eh = int(H * random.uniform(0.02, 0.2))
                ew = int(W * random.uniform(0.02, 0.2))
                y = random.randint(0, max(1, H - eh))
                x0 = random.randint(0, max(1, W - ew))
                x[:, y:y+eh, x0:x0+ew] = 0
        
        return x


class DinoV3Projector(nn.Module):
    """DINOv3 model with projection head for contrastive learning. Supports multi-channel input."""
    def __init__(self, backbone_name="dinov3_vits16", proj_dim=256, in_channels=3, freeze_patch_embed=False, freeze_blocks=0):
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
        # Using dinov3-small (ViT-S/16, 384 dimensions)
        custom_repo_id = "5xuekun/dinov3_vits16"
        model_filename = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        # Architecture base (ViT-S/16, 384 dimensions for dinov3-small)
        architecture_repo_id = "facebook/dinov2-small"
        fallback_repo_id = "facebook/dinov2-small"  # Fallback model
        
        logger.info(f"Attempting to load DINOv3 model weights from Hugging Face Hub: {custom_repo_id}/{model_filename}")
        
        # Log in_channels for debugging
        logger.info(f"🔧 DinoV3Projector.__init__: in_channels={in_channels}, freeze_patch_embed={freeze_patch_embed}, freeze_blocks={freeze_blocks}")
        
        self.backbone = None
        from transformers import AutoModel
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        
        # Try custom DINOv3 model first (download .pth weights and load into model architecture)
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
            
            # Load DINOv3 weights into model architecture (ViT-S/16, 384 dimensions)
            logger.info("Loading DINOv3 model architecture...")
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
                logger.debug(f"Missing keys (first 10): {missing_keys[:10]}")
            if unexpected_keys:
                logger.warning(f"Some weight keys were not used: {len(unexpected_keys)} keys")
                logger.debug(f"Unexpected keys (first 10): {unexpected_keys[:10]}")
            if not missing_keys and not unexpected_keys:
                logger.info(f"✅ Perfect match: All {len(filtered_state_dict)} weight keys matched model architecture")
            elif len(filtered_state_dict) > 0:
                matched_ratio = len(filtered_state_dict) / (len(filtered_state_dict) + len(missing_keys) + len(unexpected_keys)) * 100
                logger.info(f"✅ Loaded {len(filtered_state_dict)}/{len(filtered_state_dict) + len(missing_keys) + len(unexpected_keys)} weight keys ({matched_ratio:.1f}% match)")
            
            self.backbone = base_model
            logger.info(f"✅ Successfully loaded DINOv3 model weights from {model_filename}")
            
            # Update config.num_channels FIRST (before adapting patch embedding)
            # This is critical for transformers library compatibility
            if hasattr(self.backbone, 'config'):
                if hasattr(self.backbone.config, 'num_channels'):
                    logger.info(f"🔧 Updating config.num_channels from {self.backbone.config.num_channels} to {in_channels}")
                    self.backbone.config.num_channels = in_channels
                else:
                    logger.warning(f"⚠️ Model config does not have num_channels attribute")
            
            # Adapt patch embedding for multi-channel input (if in_channels != 3)
            # IMPORTANT: Do this AFTER loading weights (matching reference implementation)
            # Reference: 260113_Training_dinov3_CO_screen.py line 141-164
            if in_channels != 3:
                try:
                    # Reference implementation uses: self.backbone.patch_embed.proj
                    # Try this first (torch.hub style)
                    if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'proj'):
                        patch_embed = self.backbone.patch_embed
                        old_proj = patch_embed.proj
                        
                        logger.info(f"🔍 Checking patch embedding: old_proj.in_channels={old_proj.in_channels}, target in_channels={in_channels}")
                        if old_proj.in_channels != in_channels:
                            logger.info(f"🔧 Adapting patch embedding from {old_proj.in_channels} to {in_channels} channels")
                            new_proj = nn.Conv2d(
                                in_channels=in_channels,
                                out_channels=old_proj.out_channels,
                                kernel_size=old_proj.kernel_size,
                                stride=old_proj.stride,
                                padding=old_proj.padding,
                                bias=old_proj.bias is not None,
                            )
                            
                            with torch.no_grad():
                                # Inherit as much as possible from RGB weights (matching reference)
                                c = min(old_proj.in_channels, in_channels)
                                new_proj.weight[:, :c] = old_proj.weight[:, :c]
                                if old_proj.bias is not None:
                                    new_proj.bias.copy_(old_proj.bias)
                            
                            patch_embed.proj = new_proj
                            
                            # Update config if available (for transformers compatibility)
                            if hasattr(self.backbone, 'config'):
                                if hasattr(self.backbone.config, 'num_channels'):
                                    self.backbone.config.num_channels = in_channels
                                    logger.info(f"✅ Updated config.num_channels to {in_channels}")
                                # Also check for image_size and other relevant config fields
                                if hasattr(self.backbone.config, 'image_size'):
                                    logger.debug(f"Model config.image_size: {self.backbone.config.image_size}")
                            
                            logger.info(f"✅ Patch embedding adapted to {in_channels} channels")
                    else:
                        # Fallback: try transformers style structure
                        logger.warning(f"⚠️ Model does not have patch_embed.proj, trying transformers structure...")
                        if hasattr(self.backbone, 'embeddings') and hasattr(self.backbone.embeddings, 'patch_embeddings'):
                            if hasattr(self.backbone.embeddings.patch_embeddings, 'projection'):
                                old_proj = self.backbone.embeddings.patch_embeddings.projection
                                if old_proj.in_channels != in_channels:
                                    logger.info(f"🔧 Adapting transformers patch embedding from {old_proj.in_channels} to {in_channels} channels")
                                    new_proj = nn.Conv2d(
                                        in_channels=in_channels,
                                        out_channels=old_proj.out_channels,
                                        kernel_size=old_proj.kernel_size,
                                        stride=old_proj.stride,
                                        padding=old_proj.padding,
                                        bias=old_proj.bias is not None,
                                    )
                                    
                                    with torch.no_grad():
                                        c = min(old_proj.in_channels, in_channels)
                                        new_proj.weight[:, :c] = old_proj.weight[:, :c]
                                        if old_proj.bias is not None:
                                            new_proj.bias.copy_(old_proj.bias)
                                    
                                    self.backbone.embeddings.patch_embeddings.projection = new_proj
                                    
                                    # Also update config if available
                                    if hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'num_channels'):
                                        self.backbone.config.num_channels = in_channels
                                    
                                    logger.info(f"✅ Transformers patch embedding adapted to {in_channels} channels")
                        else:
                            logger.error(f"❌ Could not find patch embedding layer. Model type: {type(self.backbone)}")
                            logger.error(f"   Available attributes: {[k for k in dir(self.backbone) if not k.startswith('_')][:20]}")
                            raise ValueError(f"Cannot adapt patch embedding: model structure not recognized")
                except Exception as e:
                    logger.error(f"❌ Failed to adapt patch embedding for multi-channel input: {e}")
                    import traceback
                    logger.error(traceback.format_exc())
                    raise ValueError(f"Failed to adapt model for {in_channels} channels: {e}")
            
            # Freeze patch embedding if requested
            if freeze_patch_embed:
                try:
                    if hasattr(self.backbone, 'patch_embed'):
                        for p in self.backbone.patch_embed.parameters():
                            p.requires_grad = False
                        logger.info("🔒 Frozen patch embedding layer")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to freeze patch embedding: {e}")
            
            # Freeze transformer blocks if requested
            if freeze_blocks > 0:
                try:
                    if hasattr(self.backbone, 'blocks'):
                        for blk in self.backbone.blocks[:freeze_blocks]:
                            for p in blk.parameters():
                                p.requires_grad = False
                        logger.info(f"🔒 Frozen first {freeze_blocks} transformer blocks")
                except Exception as e:
                    logger.warning(f"⚠️ Failed to freeze transformer blocks: {e}")
            
            # Update feat_dim based on model architecture
            # Using dinov3-small (ViT-S/16, 384 dimensions) as specified
            if 'vits16' in backbone_name.lower() or custom_repo_id.endswith('dinov3_vits16'):
                feat_dim_map[backbone_name] = 384
                logger.info("✅ DINOv3-small model loaded (ViT-S/16, 384 dimensions)")
            elif 'vitb16' in model_filename.lower():
                feat_dim_map[backbone_name] = 768
                logger.info("Detected DINOv3 ViT-B/16 model (768 dimensions)")
            elif 'vitl16' in model_filename.lower() or custom_repo_id.endswith('dinov3_vitl16'):
                feat_dim_map[backbone_name] = 1024
                logger.info("Detected DINOv3 ViT-L/16 model (1024 dimensions)")
            
        except Exception as e:
            logger.warning(f"Failed to load custom DINOv3 model ({custom_repo_id}/{model_filename}): {e}")
            import traceback
            logger.debug(traceback.format_exc())
            logger.info(f"Falling back to fallback model: {fallback_repo_id}")
            
            # Fallback model
            try:
                self.backbone = AutoModel.from_pretrained(
                    fallback_repo_id,
                    token=hf_token,
                    trust_remote_code=True
                )
                logger.info(f"✅ Loaded fallback model from Hugging Face Hub (384 dimensions)")
                # Update feat_dim for small model (384 dimensions)
                feat_dim_map[backbone_name] = 384
                
                # Update config.num_channels FIRST (before adapting patch embedding)
                # This is critical for transformers library compatibility
                if hasattr(self.backbone, 'config'):
                    if hasattr(self.backbone.config, 'num_channels'):
                        logger.info(f"🔧 Updating fallback config.num_channels from {self.backbone.config.num_channels} to {in_channels}")
                        self.backbone.config.num_channels = in_channels
                    else:
                        logger.warning(f"⚠️ Fallback model config does not have num_channels attribute")
                
                # Adapt patch embedding for multi-channel input (if in_channels != 3)
                # Use same logic as main model (matching reference implementation)
                if in_channels != 3:
                    try:
                        # Reference implementation uses: self.backbone.patch_embed.proj
                        if hasattr(self.backbone, 'patch_embed') and hasattr(self.backbone.patch_embed, 'proj'):
                            patch_embed = self.backbone.patch_embed
                            old_proj = patch_embed.proj
                            
                            if old_proj.in_channels != in_channels:
                                logger.info(f"🔧 Adapting fallback model patch embedding from {old_proj.in_channels} to {in_channels} channels")
                                new_proj = nn.Conv2d(
                                    in_channels=in_channels,
                                    out_channels=old_proj.out_channels,
                                    kernel_size=old_proj.kernel_size,
                                    stride=old_proj.stride,
                                    padding=old_proj.padding,
                                    bias=old_proj.bias is not None,
                                )
                                
                                with torch.no_grad():
                                    c = min(old_proj.in_channels, in_channels)
                                    new_proj.weight[:, :c] = old_proj.weight[:, :c]
                                    if old_proj.bias is not None:
                                        new_proj.bias.copy_(old_proj.bias)
                                
                                patch_embed.proj = new_proj
                                
                                # Update config if available (for transformers compatibility)
                                if hasattr(self.backbone, 'config'):
                                    if hasattr(self.backbone.config, 'num_channels'):
                                        self.backbone.config.num_channels = in_channels
                                        logger.info(f"✅ Updated fallback config.num_channels to {in_channels}")
                                    # Also check for image_size and other relevant config fields
                                    if hasattr(self.backbone.config, 'image_size'):
                                        logger.debug(f"Fallback model config.image_size: {self.backbone.config.image_size}")
                                
                                logger.info(f"✅ Fallback model patch embedding adapted to {in_channels} channels")
                        else:
                            # Try transformers style
                            if hasattr(self.backbone, 'embeddings') and hasattr(self.backbone.embeddings, 'patch_embeddings'):
                                if hasattr(self.backbone.embeddings.patch_embeddings, 'projection'):
                                    old_proj = self.backbone.embeddings.patch_embeddings.projection
                                    if old_proj.in_channels != in_channels:
                                        logger.info(f"🔧 Adapting fallback transformers patch embedding from {old_proj.in_channels} to {in_channels} channels")
                                        new_proj = nn.Conv2d(
                                            in_channels=in_channels,
                                            out_channels=old_proj.out_channels,
                                            kernel_size=old_proj.kernel_size,
                                            stride=old_proj.stride,
                                            padding=old_proj.padding,
                                            bias=old_proj.bias is not None,
                                        )
                                        
                                        with torch.no_grad():
                                            c = min(old_proj.in_channels, in_channels)
                                            new_proj.weight[:, :c] = old_proj.weight[:, :c]
                                            if old_proj.bias is not None:
                                                new_proj.bias.copy_(old_proj.bias)
                                        
                                        self.backbone.embeddings.patch_embeddings.projection = new_proj
                                        
                                        if hasattr(self.backbone, 'config') and hasattr(self.backbone.config, 'num_channels'):
                                            self.backbone.config.num_channels = in_channels
                                        
                                        logger.info(f"✅ Fallback transformers patch embedding adapted to {in_channels} channels")
                            else:
                                logger.error(f"❌ Could not find patch embedding in fallback model")
                                raise ValueError(f"Cannot adapt fallback model patch embedding: model structure not recognized")
                    except Exception as e:
                        logger.error(f"❌ Failed to adapt fallback model patch embedding: {e}")
                        import traceback
                        logger.error(traceback.format_exc())
                        raise ValueError(f"Failed to adapt fallback model for {in_channels} channels: {e}")
                
                # Freeze layers if requested
                if freeze_patch_embed:
                    try:
                        if hasattr(self.backbone, 'patch_embed'):
                            for p in self.backbone.patch_embed.parameters():
                                p.requires_grad = False
                    except Exception:
                        pass
                
                if freeze_blocks > 0:
                    try:
                        if hasattr(self.backbone, 'blocks'):
                            for blk in self.backbone.blocks[:freeze_blocks]:
                                for p in blk.parameters():
                                    p.requires_grad = False
                    except Exception:
                        pass
            except Exception as fallback_e:
                logger.error(f"Failed to load fallback model: {fallback_e}")
                raise ValueError(
                    f"Model loading failed. Tried:\n"
                    f"1. Custom DINOv3: {custom_repo_id}/{model_filename} - {str(e)}\n"
                    f"2. Fallback model: {fallback_repo_id} - {str(fallback_e)}\n"
                    f"Please check your internet connection and Hugging Face Hub access."
                )
        
        # Get feature dimension - use map value as default, will be adjusted if needed
        # Default to 384 for vits16 to reduce GPU memory usage
        feat_dim = feat_dim_map.get(backbone_name, 384)
        
        # Try to infer actual feat_dim from loaded model if available
        if self.backbone is not None:
            try:
                # DINOv3 models typically have a config attribute
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


class Cell_State_Analyzer_Multi_Tool(BaseTool):
    """
    Analyzes cell states using self-supervised learning (contrastive learning) for multi-channel images.
    Performs training on multi-channel single-cell crops and generates UMAP visualizations with clustering.
    Designed for multi-channel images (e.g., 2-channel BF+GFP).
    """
    
    def __init__(self):
        super().__init__(
            tool_name="Cell_State_Analyzer_Multi_Tool",
            tool_description="Performs self-supervised learning (contrastive learning) on multi-channel single-cell crops to analyze cell states. Trains a DINOv3 model adapted for multi-channel input and generates UMAP visualizations with clustering. Designed for multi-channel images (e.g., 2-channel BF+GFP). Supports multi-group analysis.",
            tool_version="1.0.0",
            input_types={
                "cell_crops": "List[str] - List of cell crop image paths from Single_Cell_Cropper_Tool output.",
                "cell_metadata": "List[dict] - List of metadata dictionaries for each cell (must include 'group' field for multi-group analysis).",
                "max_epochs": "int - Maximum number of training epochs (default: 25).",
                "early_stop_loss": "float - Early stopping threshold for loss (default: 0.5). Training stops if loss <= this value.",
                "batch_size": "int - Batch size for training (default: 16).",
                "learning_rate": "float - Learning rate for optimizer (default: 3e-5).",
                "cluster_resolution": "float - Resolution for Leiden clustering (default: 0.5).",
                "query_cache_dir": "str - Directory for caching results.",
                "in_channels": "int - Number of input channels (default: 2 for BF+GFP).",
            },
            output_type="dict - Analysis results with training history, UMAP visualizations, and cluster analysis.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(cell_crops=cell_crops, cell_metadata=cell_metadata)',
                    "description": "Train model and generate UMAP visualizations with default parameters."
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for training. Requires anndata and scanpy for advanced visualizations. Training time depends on number of cells and epochs. Designed for multi-channel images (typically 2 channels: BF+GFP).",
                "best_practice": "Use with output from Single_Cell_Cropper_Tool. Use with multi-channel crop images (e.g., 2-channel BF+GFP TIFF files). Include 'group' field in cell_metadata for multi-group cluster composition analysis. Adjust cluster_resolution based on your data granularity needs."
            },
            output_dir="output_visualizations"
        )
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        logger.info(f"Cell_State_Analyzer_Multi_Tool: Using device: {self.device}")
    
    def _get_augmentation_transform(self):
        """Get augmentation transform for training (multi-channel tensor transform)."""
        return MultiChannelTransform(out_size=224, train=True, hflip_p=0.5, vflip_p=0.2, erase_p=0.3)
    
    def _get_eval_transform(self):
        """Get evaluation transform (no augmentation, multi-channel tensor transform)."""
        return MultiChannelTransform(out_size=224, train=False, hflip_p=0.0, vflip_p=0.0, erase_p=0.0)
    
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
                # Backbone may return tensor directly or transformers output object
                out = model.backbone(v1)
                
                # Handle different output formats
                if isinstance(out, torch.Tensor):
                    # Direct tensor output: if shape is [B, N, D], take first token (CLS)
                    # Otherwise it's already CLS token [B, D]
                    if out.dim() > 2:
                        out = out[:, 0, :]
                elif hasattr(out, 'last_hidden_state'):
                    # Transformers BaseModelOutput or similar: extract CLS token from last_hidden_state
                    out = out.last_hidden_state[:, 0]
                elif hasattr(out, 'pooler_output'):
                    # Some models return pooler_output directly
                    out = out.pooler_output
                elif isinstance(out, dict):
                    # Dictionary output: try common keys
                    if 'x_norm_clstoken' in out:
                        out = out['x_norm_clstoken']
                    elif 'last_hidden_state' in out:
                        out = out['last_hidden_state'][:, 0]
                    else:
                        raise ValueError(f"Could not extract features from output dict. Available keys: {list(out.keys())}")
                else:
                    # Truly unexpected output type
                    logger.warning(f"Unexpected output type from backbone: {type(out)}, trying to extract CLS token...")
                    raise ValueError(f"Unexpected output type from backbone: {type(out)}")
                
                feats.append(out.cpu())
                img_names.extend(names)
                groups.extend(grps)
        
        model.projector = proj_backup
        feats_all = torch.cat(feats, dim=0).numpy()
        logger.info(f"✅ Extracted CLS features: {feats_all.shape}")
        return feats_all, img_names, groups
    
    def _train_model(self, model, train_loader, max_epochs, early_stop_loss, learning_rate, output_dir, training_logs=None, patience=5):
        """Train the model with contrastive learning.
        
        Args:
            patience: Number of epochs without improvement before stopping (default: 5)
        """
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = GradScaler()
        
        history = []
        best_loss = float("inf")
        best_model_path = os.path.join(output_dir, "best_model.pth")
        epochs_without_improvement = 0
        
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
            
            # Save best model and reset patience counter
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                best_model_log = f"🔥 New best model saved (Loss = {best_loss:.4f})"
                logger.info(best_model_log)
                training_logs.append(best_model_log)
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            # Early stopping: loss threshold
            if avg_loss <= early_stop_loss:
                early_stop_log = f"✅ Early stopping triggered: Loss = {avg_loss:.4f} <= {early_stop_loss}"
                logger.info(early_stop_log)
                training_logs.append(early_stop_log)
                break
            
            # Early stopping: patience (no improvement for >patience epochs)
            if epochs_without_improvement >= patience:
                patience_stop_log = f"✅ Early stopping triggered: No improvement for {patience} epochs (best loss: {best_loss:.4f})"
                logger.info(patience_stop_log)
                training_logs.append(patience_stop_log)
                break
        
        return history, best_model_path, training_logs
    
    def _create_loss_curve(self, history, output_dir):
        """Create and save loss curve plot."""
        if not history or len(history) == 0:
            logger.warning("No training history available, skipping loss curve creation")
            return None
        
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
        """Load cell crops and metadata from metadata files. Merges all metadata files for multi-image processing."""
        logger.info(f"Cell_State_Analyzer_Tool: Loading metadata from query_cache_dir: {query_cache_dir}")
        
        # Standard tool_cache directory
        tool_cache_dir = os.path.join(query_cache_dir, "tool_cache")
        
        # Try multiple possible paths in priority order:
        # 1. Directly in tool_cache_dir (where Single_Cell_Cropper_Tool saves metadata) - HIGHEST PRIORITY
        # 2. In group subdirectories (e.g., tool_cache_dir/default/, tool_cache_dir/control/, etc.) - FALLBACK
        # Use dict.fromkeys() to remove duplicates while preserving insertion order (Python 3.7+)
        possible_dirs = []
        
        # Priority 1: Root tool_cache_dir (where metadata files are actually saved)
        if tool_cache_dir:
            possible_dirs.append(tool_cache_dir)
        
        # Priority 2: Check if query_cache_dir already contains tool_cache (edge case)
        if "tool_cache" in query_cache_dir and query_cache_dir not in possible_dirs:
            possible_dirs.append(query_cache_dir)
        
        # Priority 3: Group subdirectories (for backward compatibility, but metadata should be in root)
        if os.path.exists(tool_cache_dir):
            try:
                subdirs = [d for d in os.listdir(tool_cache_dir) 
                          if os.path.isdir(os.path.join(tool_cache_dir, d))]
                if subdirs:
                    logger.info(f"Cell_State_Analyzer_Tool: Found subdirectories in tool_cache: {subdirs}")
                    for subdir in subdirs:
                        subdir_path = os.path.join(tool_cache_dir, subdir)
                        if subdir_path not in possible_dirs:
                            possible_dirs.append(subdir_path)
            except Exception as e:
                logger.warning(f"Cell_State_Analyzer_Tool: Error listing subdirectories: {e}")
        
        # Remove None values and duplicates while preserving order
        possible_dirs = [d for d in dict.fromkeys(possible_dirs) if d is not None]
        
        logger.info(f"Cell_State_Analyzer_Tool: Searching in possible directories (priority order): {possible_dirs}")
        
        metadata_files = []
        for metadata_dir in possible_dirs:
            logger.info(f"Cell_State_Analyzer_Tool: Checking directory: {metadata_dir}, exists={os.path.exists(metadata_dir)}")
            if os.path.exists(metadata_dir):
                search_pattern = os.path.join(metadata_dir, 'cell_crops_metadata_*.json')
                logger.info(f"Cell_State_Analyzer_Tool: Searching for pattern: {search_pattern}")
                found_files = glob.glob(search_pattern)
                logger.info(f"Cell_State_Analyzer_Tool: Found {len(found_files)} file(s) matching pattern")
                if found_files:
                    metadata_files.extend(found_files)
                    logger.info(f"Found {len(found_files)} metadata file(s) in {metadata_dir}")
                    for f in found_files:
                        logger.info(f"  - {f}")
            else:
                logger.warning(f"Cell_State_Analyzer_Tool: Directory does not exist: {metadata_dir}")
        
        # Remove duplicates
        metadata_files = list(set(metadata_files))
        
        if not metadata_files:
            # Provide detailed error message with all searched paths
            searched_paths = ", ".join(possible_dirs)
            # Also list all files in the directories for debugging
            debug_info = []
            for metadata_dir in possible_dirs:
                if os.path.exists(metadata_dir):
                    try:
                        all_items = os.listdir(metadata_dir)
                        debug_info.append(f"Directory {metadata_dir} contains: {all_items}")
                    except Exception as e:
                        debug_info.append(f"Directory {metadata_dir} exists but cannot list contents: {e}")
                else:
                    debug_info.append(f"Directory {metadata_dir} does not exist")
            error_msg = f"No metadata files found. Searched in: {searched_paths}. Please ensure Single_Cell_Cropper_Tool has been executed successfully."
            if debug_info:
                error_msg += f"\nDebug info: {'; '.join(debug_info)}"
            raise ValueError(error_msg)
        
        # Merge all metadata files (for multi-image processing)
        all_cell_crops = []
        all_cell_metadata = []
        all_cell_crop_objects = []
        execution_statuses = []
        
        logger.info(f"Found {len(metadata_files)} metadata file(s), merging all files for multi-image processing")
        
        # Deduplicate metadata files by source_image_id: keep only the most recent file for each image
        # This prevents duplicate processing when Single_Cell_Cropper_Tool is called multiple times for the same image
        metadata_by_image = {}  # {source_image_id: (file_path, mtime, data)}
        for metadata_file in metadata_files:
            try:
                with open(metadata_file, 'r') as f:
                    data = json.load(f)
                
                source_image_id = data.get('source_image_id')
                if not source_image_id:
                    # Fallback: try to extract from cell_crop_objects if available
                    cell_crop_objects = data.get('cell_crop_objects', [])
                    if cell_crop_objects and len(cell_crop_objects) > 0:
                        first_crop = cell_crop_objects[0]
                        if isinstance(first_crop, dict):
                            source_image_id = first_crop.get('source_image_id')
                
                # Use file modification time to determine which is most recent
                mtime = os.path.getmtime(metadata_file)
                
                if source_image_id:
                    # If we already have a metadata file for this image, keep the most recent one
                    if source_image_id in metadata_by_image:
                        existing_mtime = metadata_by_image[source_image_id][1]
                        if mtime > existing_mtime:
                            logger.info(f"Replacing older metadata for image {source_image_id} with newer file: {os.path.basename(metadata_file)}")
                            metadata_by_image[source_image_id] = (metadata_file, mtime, data)
                        else:
                            logger.info(f"Skipping duplicate metadata for image {source_image_id}: {os.path.basename(metadata_file)} (older than existing)")
                    else:
                        metadata_by_image[source_image_id] = (metadata_file, mtime, data)
                else:
                    # If no source_image_id, include it anyway (backward compatibility)
                    logger.warning(f"Metadata file {os.path.basename(metadata_file)} has no source_image_id, including anyway")
                    metadata_by_image[f"unknown_{len(metadata_by_image)}"] = (metadata_file, mtime, data)
            except Exception as e:
                logger.warning(f"Error reading metadata file {metadata_file}: {e}, skipping")
                continue
        
        logger.info(f"After deduplication: {len(metadata_by_image)} unique image(s) from {len(metadata_files)} metadata file(s)")
        
        # Process deduplicated metadata files
        # Track which cells belong to which metadata file for group assignment
        cell_to_metadata_file = {}  # {cell_index: (file_group, file_image_name)}
        cell_index = 0
        
        skipped_images = []  # Track skipped images for summary
        
        for source_image_id, (metadata_file, mtime, data) in metadata_by_image.items():
            try:
                execution_status = data.get('execution_status', 'unknown')
                
                cell_crops = data.get('cell_crops_paths', [])
                cell_metadata = data.get('cell_metadata', [])
                cell_crop_objects = data.get('cell_crop_objects', [])
                
                # Get top-level group and image_name from metadata file (saved by Single_Cell_Cropper_Tool at line 228)
                file_group = data.get('group', None)
                file_image_name = data.get('source_image_id', source_image_id)
                
                # Check if this metadata file indicates no crops were generated
                if execution_status == 'no_crops_generated':
                    logger.warning(f"Metadata file {os.path.basename(metadata_file)} (image: {file_image_name}, group: {file_group}) indicates no crops were generated by Single_Cell_Cropper_Tool. Skipping this image.")
                    skipped_images.append({'image': file_image_name, 'group': file_group, 'status': 'no_crops_generated'})
                    # Skip this metadata file - don't add to all_cell_crops
                    continue
                
                # Only track non-skipped statuses
                execution_statuses.append(execution_status)
                
                # Normalize paths
                cell_crops = [os.path.normpath(path) for path in cell_crops]
                
                # Track which cells belong to this metadata file
                for _ in range(len(cell_crops)):
                    cell_to_metadata_file[cell_index] = (file_group, file_image_name)
                    cell_index += 1
                
                # Append to merged lists
                all_cell_crops.extend(cell_crops)
                all_cell_metadata.extend(cell_metadata)
                all_cell_crop_objects.extend(cell_crop_objects)
                
                logger.info(f"Loaded {len(cell_crops)} cells from {os.path.basename(metadata_file)} (image: {source_image_id}, group: {file_group}, status: {execution_status})")
            except Exception as e:
                logger.warning(f"Error loading metadata file {metadata_file}: {e}, skipping")
                continue
        
        logger.info(f"Total merged: {len(all_cell_crops)} cells from {len(metadata_files)} metadata file(s)")
        
        # Log skipped images if any
        if skipped_images:
            logger.info(f"Skipped {len(skipped_images)} image(s) with no crops generated: {[img['image'] for img in skipped_images]}")
        
        # Check execution statuses - only fail if ALL remaining images had errors
        # (no_crops_generated images are already skipped, so they don't affect this check)
        if execution_statuses:
            if all(s == 'error' for s in execution_statuses):
                raise ValueError(
                    f"Single_Cell_Cropper_Tool execution failed with errors for all images. "
                    f"Cannot proceed with Cell_State_Analyzer_Tool. "
                    f"Please check Single_Cell_Cropper_Tool execution logs for details. "
                    f"Execution statuses: {execution_statuses}"
                )
        
        # Check if we have any crops to process
        if not all_cell_crops:
            raise ValueError(
                f"No cell crops found in metadata files despite successful execution status. "
                f"This is unexpected. Please check Single_Cell_Cropper_Tool execution. "
                f"Execution statuses: {execution_statuses if execution_statuses else 'unknown'}"
            )
        
        # Ensure all cell_metadata entries have group and image_name from Single_Cell_Cropper_Tool
        # Priority order: 1) cell_metadata (saved by Single_Cell_Cropper_Tool at line 529)
        #                 2) cell_crop_objects (serialized CellCrop objects with group field)
        #                 3) Top-level group from metadata file (saved at line 228)
        for i in range(len(all_cell_crops)):
            # Initialize metadata entry if needed
            if i >= len(all_cell_metadata):
                all_cell_metadata.append({})
            if not isinstance(all_cell_metadata[i], dict):
                all_cell_metadata[i] = {}
            
            # Priority 1: Get from cell_metadata (direct path from Single_Cell_Cropper_Tool)
            if 'group' not in all_cell_metadata[i] or all_cell_metadata[i].get('group') == 'default':
                # Priority 2: Get from cell_crop_objects if available
                if all_cell_crop_objects and i < len(all_cell_crop_objects):
                    crop_obj = all_cell_crop_objects[i]
                    if isinstance(crop_obj, dict):
                        crop_group = crop_obj.get('group')
                        if crop_group and crop_group != 'default':
                            all_cell_metadata[i]['group'] = crop_group
                            all_cell_metadata[i]['image_name'] = crop_obj.get('source_image_id', 'unknown')
                
                # Priority 3: Get from metadata file's top-level group field
                if ('group' not in all_cell_metadata[i] or all_cell_metadata[i].get('group') == 'default') and i in cell_to_metadata_file:
                    file_group, file_image_name = cell_to_metadata_file[i]
                    if file_group and file_group != 'default':
                        all_cell_metadata[i]['group'] = file_group
                    if 'image_name' not in all_cell_metadata[i] or all_cell_metadata[i].get('image_name') == 'unknown':
                        all_cell_metadata[i]['image_name'] = file_image_name
            
            # Final fallback: ensure required fields exist (only if still missing)
            if 'group' not in all_cell_metadata[i] or all_cell_metadata[i].get('group') == 'default':
                # Only use 'default' if we truly have no group information
                if i in cell_to_metadata_file:
                    file_group, _ = cell_to_metadata_file[i]
                    all_cell_metadata[i]['group'] = file_group if file_group else 'default'
                else:
                    all_cell_metadata[i]['group'] = 'default'
            if 'image_name' not in all_cell_metadata[i] or all_cell_metadata[i].get('image_name') == 'unknown':
                if i in cell_to_metadata_file:
                    _, file_image_name = cell_to_metadata_file[i]
                    all_cell_metadata[i]['image_name'] = file_image_name if file_image_name else 'unknown'
                else:
                    all_cell_metadata[i]['image_name'] = 'unknown'
        
        return all_cell_crops, all_cell_metadata, skipped_images
    
    def _recommend_hyperparameters(self, num_crops: int, batch_size: int = None, 
                                   learning_rate: float = None, max_epochs: int = None) -> dict:
        """
        Intelligently recommend hyperparameters based on number of crops.
        
        Strategy:
        - Very few crops (<10): Lower LR, smaller batch, more epochs
        - Few crops (10-50): Moderate LR, adjust batch size
        - Medium crops (50-200): Default settings
        - Many crops (>200): Can use higher LR, larger batch
        
        Args:
            num_crops: Number of cell crops
            batch_size: User-specified batch size (None to auto-recommend)
            learning_rate: User-specified learning rate (None to auto-recommend)
            max_epochs: User-specified max epochs (None to auto-recommend)
            
        Returns:
            dict with recommended batch_size, learning_rate, max_epochs
        """
        recommendations = {}
        
        # Recommend batch_size based on crop count
        if batch_size is None:
            if num_crops < 10:
                # Very few crops: use batch_size = num_crops (or 2 if num_crops < 2)
                recommendations['batch_size'] = max(2, min(num_crops, 4))
            elif num_crops < 50:
                # Few crops: use smaller batch size
                recommendations['batch_size'] = min(8, num_crops // 2)
            elif num_crops < 200:
                # Medium crops: default batch size
                recommendations['batch_size'] = 16
            else:
                # Many crops: can use larger batch
                recommendations['batch_size'] = 32
        else:
            # User specified, but ensure it's not larger than num_crops
            recommendations['batch_size'] = min(batch_size, num_crops)
        
        # Recommend learning_rate based on crop count
        if learning_rate is None:
            if num_crops < 5:
                # Very few crops (<5): Use very small LR to prevent overfitting
                recommendations['learning_rate'] = 1e-6
            elif num_crops < 10:
                # Few crops (5-9): Small LR
                recommendations['learning_rate'] = 5e-6
            elif num_crops < 50:
                # Moderate crops (10-49): Reduced LR
                recommendations['learning_rate'] = 1e-5
            elif num_crops < 200:
                # Medium crops (50-199): Default LR
                recommendations['learning_rate'] = 3e-5
            else:
                # Many crops (>=200): Can use slightly higher LR
                recommendations['learning_rate'] = 5e-5
        else:
            recommendations['learning_rate'] = learning_rate
        
        # Recommend max_epochs based on crop count
        if max_epochs is None:
            if num_crops < 10:
                # Very few crops: More epochs needed for learning
                recommendations['max_epochs'] = 50
            elif num_crops < 50:
                # Few crops: Moderate epochs
                recommendations['max_epochs'] = 35
            else:
                # Medium/many crops: Default epochs
                recommendations['max_epochs'] = 25
        else:
            recommendations['max_epochs'] = max_epochs
        
        return recommendations
    
    def execute(self, cell_crops=None, cell_metadata=None, max_epochs=25, early_stop_loss=0.5,
                batch_size=16, learning_rate=3e-5, cluster_resolution=0.5, query_cache_dir=None,
                in_channels=None, selected_channels=None, freeze_patch_embed=False, freeze_blocks=0):
        """
        Execute self-supervised learning training and analysis.
        
        Args:
            cell_crops: List of cell crop image paths
            cell_metadata: List of metadata dictionaries (should include 'group' field)
            max_epochs: Maximum training epochs (default: 25, auto-adjusted based on crop count)
            early_stop_loss: Early stopping loss threshold (default: 0.5)
            batch_size: Training batch size (default: 16, auto-adjusted based on crop count)
            learning_rate: Learning rate (default: 3e-5, auto-adjusted based on crop count)
            cluster_resolution: Leiden clustering resolution (default: 0.5)
            query_cache_dir: Directory for outputs
            in_channels: Number of input channels (default: None, auto-detect from first image)
            selected_channels: List of channel indices to use (e.g., [0, 1] for BF+GFP). If None and in_channels is set, uses [0, ..., in_channels-1]
            freeze_patch_embed: Whether to freeze patch embedding layer (default: False)
            freeze_blocks: Number of transformer blocks to freeze (default: 0)
            
        Returns:
            dict: Analysis results with visualizations and AnnData object
        """
        logger.info("🚀 Cell_State_Analyzer_Multi_Tool starting execution...")
        
        # Load cell data if not provided
        skipped_images = []  # Track skipped images for summary
        if cell_crops is None or cell_metadata is None:
            if query_cache_dir is None:
                query_cache_dir = "solver_cache/temp"
            logger.info(f"Loading cell data from metadata in: {query_cache_dir}")
            cell_crops, cell_metadata, skipped_images = self._load_cell_data_from_metadata(query_cache_dir)
        
        if not cell_crops or len(cell_crops) == 0:
            return {"error": "No cell crops found for analysis", "status": "failed"}
        
        num_crops = len(cell_crops)
        logger.info(f"🔬 Processing {num_crops} cell crops...")
        
        # Auto-detect in_channels if not provided
        if in_channels is None:
            # Load first image to detect channel count
            try:
                img_data = ImageProcessor.load_image(cell_crops[0])
                in_channels = img_data.num_channels
                logger.info(f"🔍 Auto-detected {in_channels} channels from first image: {cell_crops[0]}")
            except Exception as e:
                logger.warning(f"Failed to auto-detect channels: {e}, defaulting to 2 channels")
                in_channels = 2
        
        # Set selected_channels if not provided
        if selected_channels is None:
            selected_channels = list(range(in_channels))
            logger.info(f"📊 Using channels: {selected_channels} (all {in_channels} channels)")
        else:
            logger.info(f"📊 Using selected channels: {selected_channels}")
        
        # Strategy: Use zero-shot inference for small datasets (<50 crops), training for larger datasets
        use_zero_shot = num_crops < 50
        
        if use_zero_shot:
            logger.info(f"🎯 Using zero-shot inference mode (crops: {num_crops} < 50). Training is skipped for small datasets.")
            # For zero-shot, we only need eval transform and batch size
            final_batch_size = min(batch_size, num_crops) if batch_size is not None else min(16, num_crops)
            logger.info(f"   Batch size: {final_batch_size} (for feature extraction)")
        else:
            logger.info(f"🎯 Using training mode (crops: {num_crops} >= 50). Model will be fine-tuned.")
            # Get intelligent hyperparameter recommendations for training
            recommendations = self._recommend_hyperparameters(
                num_crops, 
                batch_size=None,  # Always get recommendation, then use user value if provided
                learning_rate=None,  # Always get recommendation, then use user value if provided
                max_epochs=None  # Always get recommendation, then use user value if provided
            )
            
            # Use user-specified values if provided (and valid), otherwise use recommendations
            # For batch_size: ensure it doesn't exceed num_crops
            final_batch_size = min(batch_size, num_crops) if batch_size is not None else recommendations['batch_size']
            final_learning_rate = learning_rate if learning_rate is not None else recommendations['learning_rate']
            final_max_epochs = max_epochs if max_epochs is not None else recommendations['max_epochs']
            
            # Log recommendations and final values
            logger.info(f"📊 Hyperparameter recommendations for {num_crops} crops:")
            if batch_size is not None and batch_size != recommendations['batch_size']:
                logger.info(f"   Batch size: {final_batch_size} (user-specified: {batch_size}, recommended: {recommendations['batch_size']})")
            else:
                logger.info(f"   Batch size: {final_batch_size} (auto-recommended)")
            
            if learning_rate is not None and abs(learning_rate - recommendations['learning_rate']) > 1e-8:
                logger.info(f"   Learning rate: {final_learning_rate:.2e} (user-specified: {learning_rate:.2e}, recommended: {recommendations['learning_rate']:.2e})")
            else:
                logger.info(f"   Learning rate: {final_learning_rate:.2e} (auto-recommended)")
            
            if max_epochs is not None and max_epochs != recommendations['max_epochs']:
                logger.info(f"   Max epochs: {final_max_epochs} (user-specified: {max_epochs}, recommended: {recommendations['max_epochs']})")
            else:
                logger.info(f"   Max epochs: {final_max_epochs} (auto-recommended)")
            
            # Update variables for use in training
            batch_size = final_batch_size
            learning_rate = final_learning_rate
            max_epochs = final_max_epochs
        
        # Setup output directory
        if query_cache_dir is None:
            query_cache_dir = "solver_cache/temp"
        output_dir = os.path.join(query_cache_dir, "cell_state_analysis")
        os.makedirs(output_dir, exist_ok=True)
        
        # Extract groups and image_names from metadata - direct path, no fallbacks
        groups = []
        image_names = []
        for meta in cell_metadata:
            if isinstance(meta, dict):
                # Direct extraction: group and image_name should be in metadata from Single_Cell_Cropper_Tool
                group = meta.get('group', 'default')
                image_name = meta.get('image_name', '')
                groups.append(group)
                image_names.append(image_name if image_name else 'unknown')
            else:
                groups.append("default")
                image_names.append("unknown")
        
        # Create datasets
        eval_transform = self._get_eval_transform()
        
        if use_zero_shot:
            # Zero-shot mode: only need eval dataset for feature extraction
            eval_dataset = MultiChannelCellCropDataset(cell_crops, groups, transform=eval_transform, selected_channels=selected_channels)
            eval_loader = DataLoader(eval_dataset, batch_size=final_batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
            logger.info(f"✅ Loaded {len(eval_dataset)} images for zero-shot inference")
            
            # Initialize model (will use pretrained weights)
            # Use len(selected_channels) as in_channels to match the actual input channels
            actual_in_channels = len(selected_channels)
            logger.info(f"🔧 Initializing model with in_channels={actual_in_channels} (from selected_channels={selected_channels})")
            model = DinoV3Projector(backbone_name="dinov3_vits16", proj_dim=256, in_channels=actual_in_channels, freeze_patch_embed=freeze_patch_embed, freeze_blocks=freeze_blocks).to(self.device)
            logger.info("✅ Using pretrained DINOv3 model (zero-shot mode, no training)")
            
            # Extract features directly with pretrained model
            logger.info("🔍 Extracting features with pretrained model (zero-shot inference)...")
            feats, img_names, groups_extracted = self._extract_features(model, eval_loader, self.device)
            
            # No training history for zero-shot
            history = []
            training_logs = []
            loss_curve_path = None
        else:
            # Training mode: create train and eval datasets
            train_transform = self._get_augmentation_transform()
            
            train_dataset = MultiChannelCellCropDataset(cell_crops, groups, transform=train_transform, selected_channels=selected_channels)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, 
                                     num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
            
            eval_dataset = MultiChannelCellCropDataset(cell_crops, groups, transform=eval_transform, selected_channels=selected_channels)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False,
                                    num_workers=0, pin_memory=True if torch.cuda.is_available() else False)
            
            logger.info(f"✅ Loaded {len(train_dataset)} images")
            
            # Initialize model with multi-channel support
            # Use len(selected_channels) as in_channels to match the actual input channels
            actual_in_channels = len(selected_channels)
            logger.info(f"🔧 Initializing model with in_channels={actual_in_channels} (from selected_channels={selected_channels})")
            model = DinoV3Projector(backbone_name="dinov3_vits16", proj_dim=256, in_channels=actual_in_channels, freeze_patch_embed=freeze_patch_embed, freeze_blocks=freeze_blocks).to(self.device)
            
            # Train model
            logger.info("🎯 Starting training...")
            training_logs = []  # Collect training progress logs
            history, best_model_path, training_logs = self._train_model(
                model, train_loader, max_epochs, early_stop_loss, learning_rate, output_dir, training_logs, patience=5
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
        # Use groups_extracted and img_names from _extract_features to ensure order matches features
        # These are extracted from the dataloader in the same order as features
        # Fallback to groups/image_names from metadata if lengths don't match
        if len(groups_extracted) == adata.n_obs:
            adata.obs["group"] = groups_extracted
        else:
            logger.warning(f"groups_extracted length ({len(groups_extracted)}) doesn't match adata.n_obs ({adata.n_obs}), using groups from metadata")
            adata.obs["group"] = groups[:adata.n_obs] if len(groups) >= adata.n_obs else (groups + ['default'] * (adata.n_obs - len(groups)))
        
        if len(img_names) == adata.n_obs:
            adata.obs["image_name"] = img_names
        else:
            logger.warning(f"img_names length ({len(img_names)}) doesn't match adata.n_obs ({adata.n_obs}), using image_names from metadata")
            adata.obs["image_name"] = image_names[:adata.n_obs] if len(image_names) >= adata.n_obs else (image_names + ['unknown'] * (adata.n_obs - len(image_names)))
        # Store full cell crop paths for visualization
        adata.obs["crop_path"] = cell_crops
        # Use cell_id from metadata as index if available, otherwise use default
        if cell_metadata and len(cell_metadata) == adata.n_obs:
            cell_ids = []
            for meta in cell_metadata:
                if isinstance(meta, dict) and 'cell_id' in meta:
                    cell_ids.append(meta['cell_id'])
                else:
                    cell_ids.append(f"cell_{len(cell_ids)}")
            adata.obs.index = cell_ids
        else:
            adata.obs.index = [f"cell_{i}" for i in range(adata.n_obs)]
        
        # Compute UMAP and clustering (no visualization - handled by Analysis_Visualizer_Tool)
        cluster_key = self._compute_umap_and_clustering(adata, cluster_resolution, groups if len(groups) == adata.n_obs else groups_extracted)
        
        # Save AnnData (contains UMAP coordinates and cluster assignments)
        # Use absolute path to ensure clarity and reliability
        adata_path = os.path.join(output_dir, "cell_state_analyzed.h5ad")
        adata_path = os.path.abspath(adata_path)  # Ensure absolute path for clarity
        adata.write(adata_path)
        logger.info(f"✅ AnnData saved to {adata_path}")
        
        # Prepare deliverables - includes visualizations and data files
        deliverables = [adata_path]  # Always include AnnData file
        if loss_curve_path is not None:
            deliverables.append(loss_curve_path)  # Only include loss curve if training was performed
        
        # Format training logs for display
        training_logs_text = "\n".join(training_logs) if training_logs else ""
        
        if use_zero_shot:
            summary = f"Zero-shot inference completed. Processed {len(cell_crops)} cells using pretrained DINOv3 model (no training)."
            if skipped_images:
                skipped_info = f"Note: {len(skipped_images)} image(s) were skipped because no crops were generated by Single_Cell_Cropper_Tool: {', '.join([img.get('image', 'unknown') for img in skipped_images])}."
                summary = f"{summary}\n\n{skipped_info}"
            if training_logs_text:
                summary = f"{summary}\n\n**Processing Log:**\n```\n{training_logs_text}\n```"
            
            return {
                "summary": summary,
                "cell_count": len(cell_crops),
                "mode": "zero-shot",
                "epochs_trained": 0,
                "final_loss": None,
                "best_loss": None,
                "loss_curve": None,
                "adata_path": adata_path,  # AnnData file path for Analysis_Visualizer_Tool
                "deliverables": deliverables,
                "visual_outputs": deliverables,  # Keep for backward compatibility
                "training_history": [],
                "training_logs": training_logs_text,
                "cluster_key": cluster_key,  # Cluster column name (e.g., "leiden_0.5")
                "cluster_resolution": cluster_resolution,  # Resolution used for clustering
                "analysis_type": "cell_state_analysis"  # Flag for Analysis_Visualizer_Tool to detect this output
            }
        else:
            summary = f"Training completed. Processed {len(cell_crops)} cells in {len(history)} epochs. Final loss: {history[-1]:.4f}"
            if skipped_images:
                skipped_info = f"Note: {len(skipped_images)} image(s) were skipped because no crops were generated by Single_Cell_Cropper_Tool: {', '.join([img.get('image', 'unknown') for img in skipped_images])}."
                summary = f"{summary}\n\n{skipped_info}"
            if training_logs_text:
                summary = f"{summary}\n\n**Training Progress:**\n```\n{training_logs_text}\n```"
            
            return {
                "summary": summary,
                "cell_count": len(cell_crops),
                "mode": "training",
                "epochs_trained": len(history),
                "final_loss": history[-1],
                "best_loss": min(history),
                "loss_curve": loss_curve_path,
                "adata_path": adata_path,  # AnnData file path for Analysis_Visualizer_Tool
                "deliverables": deliverables,
                "visual_outputs": deliverables,  # Keep for backward compatibility
                "training_history": history,
                "training_logs": training_logs_text,  # Include training logs for display
                "cluster_key": cluster_key,  # Cluster column name (e.g., "leiden_0.5")
                "cluster_resolution": cluster_resolution,  # Resolution used for clustering
                "analysis_type": "cell_state_analysis"  # Flag for Analysis_Visualizer_Tool to detect this output
            }


if __name__ == "__main__":
    # Test script
    tool = Cell_State_Analyzer_Multi_Tool()
    print("Tool initialized successfully")

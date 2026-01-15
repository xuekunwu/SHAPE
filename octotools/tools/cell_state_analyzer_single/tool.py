#!/usr/bin/env python3
"""
Cell State Analyzer Single-Channel Tool - Self-supervised learning for single-channel cell state analysis.
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
import matplotlib.pyplot as plt
import glob
from tqdm import tqdm

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

try:
    from huggingface_hub import hf_hub_download
    HF_HUB_AVAILABLE = True
except ImportError:
    HF_HUB_AVAILABLE = False
    logger.warning("huggingface_hub not available. Some features may not work.")

try:
    import anndata as ad
    import scanpy as sc
    ANNDATA_AVAILABLE = True
except ImportError:
    ANNDATA_AVAILABLE = False
    logger.warning("anndata/scanpy not available. Install them for visualizations.")

try:
    from skimage import io
except ImportError:
    raise ImportError("skimage is required for image loading")

# Add project root to path
current_dir = os.path.dirname(os.path.abspath(__file__))
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.dirname(current_dir))))
sys.path.insert(0, project_root)

from octotools.tools.base import BaseTool
from octotools.models.utils import VisualizationConfig
from octotools.models.image_data import ImageData
from octotools.utils.image_processor import ImageProcessor


class SingleChannelCellCropDataset(Dataset):
    """Dataset for single-channel cell crop images from Single_Cell_Cropper_Tool output.
    
    This dataset handles single-channel images from cropped cell images and converts them to RGB format
    for DINOv3 model input (which expects 3-channel RGB images).
    Uses images already saved by Single_Cell_Cropper_Tool, not original images.
    """
    def __init__(self, image_paths, groups=None, transform=None):
        self.image_paths = image_paths
        self.groups = groups if groups else ["default"] * len(image_paths)
        self.transform = transform
        
    def __len__(self):
        return len(self.image_paths)
    
    def __getitem__(self, idx):
        path = self.image_paths[idx]
        group = self.groups[idx]
        
        # Load cropped cell image from Single_Cell_Cropper_Tool output (already saved as single-channel)
        # Use skimage.io to read the saved crop file (aligned with reference implementation)
        img = io.imread(path).astype(np.float32)
        
        # Normalize based on dtype (aligned with reference)
        # Check max value to determine if uint16 or uint8
        img = img / 65535.0 if img.max() > 255 else img / 255.0
        
        # Convert single channel to RGB by repeating channel
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        elif img.ndim == 3:
            # Multi-channel: use first channel only for single-channel dataset
            img = img[:, :, 0:1]  # Keep dimension
            img = np.repeat(img, 3, axis=-1)
        
        # Convert to PIL Image
        img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
        
        # Apply transforms (two augmented views for contrastive learning)
        if self.transform:
            v1 = self.transform(img)
            v2 = self.transform(img)
        else:
            v1 = v2 = transforms.ToTensor()(img)
        
        image_name = os.path.splitext(os.path.basename(path))[0]
        return v1, v2, image_name, group


class DinoV3Projector(nn.Module):
    """DINOv3 model with projection head for contrastive learning."""
    def __init__(self, backbone_name="dinov3_vitb16", proj_dim=256):
        super().__init__()
        
        # Load DINOv3 backbone from Hugging Face Hub
        from transformers import AutoModel
        hf_token = os.getenv("HUGGINGFACE_TOKEN")
        architecture_repo_id = "facebook/dinov2-base"  # Base model for ViT-B/16
        
        # Download and load weights from Hugging Face Hub
        custom_repo_id = "5xuekun/dinov3_vits16"
        model_filename = "dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"
        
        if HF_HUB_AVAILABLE:
            weights_path = hf_hub_download(
                repo_id=custom_repo_id,
                filename=model_filename,
                token=hf_token
            )
            state_dict = torch.load(weights_path, map_location="cpu")
            if "teacher" in state_dict:
                state_dict = state_dict["teacher"]
        else:
            raise ImportError("huggingface_hub not available")
        
        # Load model architecture and weights
        self.backbone = AutoModel.from_pretrained(
            architecture_repo_id,
            token=hf_token,
            trust_remote_code=True
        )
        self.backbone.load_state_dict(state_dict, strict=False)
        
        feat_dim_map = {
            "dinov3_vits16": 384,
            "dinov3_vits16plus": 384,
            "dinov3_vitb16": 768,
            "dinov3_vitl16": 1024,
            "dinov3_vith16plus": 1280,
            "dinov3_vit7b16": 4096,
        }
        feat_dim = feat_dim_map.get(backbone_name)
        
        self.projector = nn.Sequential(
            nn.Linear(feat_dim, 512),
            nn.ReLU(inplace=True),
            nn.Linear(512, proj_dim)
        )
    
    def forward(self, x):
        """Forward pass: extract CLS token and project."""
        out = self.backbone(x)
        
        # Extract CLS token from backbone output
        if isinstance(out, torch.Tensor):
            # If tensor, check if it's [B, N, D] or [B, D]
            z = out[:, 0, :] if out.dim() == 3 else out
        elif hasattr(out, 'last_hidden_state'):
            z = out.last_hidden_state[:, 0]
        else:
            # Fallback: assume it's already CLS token
            z = out
        
        z = self.projector(z)
        return F.normalize(z, dim=-1)


def contrastive_loss(z1, z2, temperature=0.1):
    """Compute contrastive loss for self-supervised learning."""
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return nn.CrossEntropyLoss()(logits, labels)


class Cell_State_Analyzer_Single_Tool(BaseTool):
    """
    Analyzes cell states using self-supervised learning (contrastive learning).
    Performs training on single-cell crops and generates UMAP visualizations with clustering.
    Designed for single-channel images (converted to RGB for DINOv3).
    """
    
    def __init__(self):
        super().__init__(
            tool_name="Cell_State_Analyzer_Single_Tool",
            tool_description="Performs self-supervised learning (contrastive learning) on individual cell/organoid crops to analyze cell states. Analyzes pre-cropped single-cell/organoid images from Single_Cell_Cropper_Tool. Performs feature extraction, UMAP embedding, and clustering. Designed for single-channel images (converted to RGB). Supports multi-group analysis. REQUIRES: Single_Cell_Cropper_Tool must be executed first to generate individual crop images.",
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
            },
            output_type="dict - Analysis results with training history, UMAP visualizations, and cluster analysis.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(cell_crops=cell_crops, cell_metadata=cell_metadata)',
                    "description": "Train model and generate UMAP visualizations with default parameters."
                }
            ],
            user_metadata={
                "limitation": "Requires GPU for training. Requires anndata and scanpy for advanced visualizations. REQUIRES Single_Cell_Cropper_Tool output as input - cannot process raw images or segmentation masks.",
                "best_practice": "MUST use output from Single_Cell_Cropper_Tool. Input should be individual crop images, not original images or masks. Include 'group' field in cell_metadata for multi-group cluster composition analysis."
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
        """Extract CLS features from the model backbone."""
        model.eval()
        feats, img_names, groups = [], [], []
        has_proj = hasattr(model, "projector")
        if has_proj:
            proj_backup = model.projector
            model.projector = torch.nn.Identity()
        
        with torch.no_grad():
            for v1, _, names, grps in tqdm(dataloader, desc="Extracting CLS features"):
                v1 = v1.to(device)
                out = model.backbone(v1)
                feats.append(out.cpu())
                img_names.extend(names)
                groups.extend(grps)
        
        if has_proj:
            model.projector = proj_backup
        feats_all = torch.cat(feats, dim=0).numpy()
        logger.info(f"✅ Extracted CLS features: {feats_all.shape}")
        return feats_all, img_names, groups
    
    def _train_model(self, model, train_loader, max_epochs, early_stop_loss, learning_rate, output_dir, training_logs=None, patience=5):
        """Train the model with contrastive learning."""
        optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=1e-4)
        scaler = GradScaler()
        
        history = []
        best_loss = float("inf")
        best_model_path = os.path.join(output_dir, "best_model.pth")
        epochs_without_improvement = 0
        
        if training_logs is None:
            training_logs = []
        
        for epoch in range(max_epochs):
            model.train()
            total_loss = 0
            
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
            
            avg_loss = total_loss / len(train_loader)
            history.append(avg_loss)
            logger.info(f"Epoch {epoch+1}/{max_epochs}: Loss = {avg_loss:.4f}")
            
            if avg_loss < best_loss:
                best_loss = avg_loss
                torch.save(model.state_dict(), best_model_path)
                logger.info(f"🔥 New best model saved (Loss = {best_loss:.4f})")
                epochs_without_improvement = 0
            else:
                epochs_without_improvement += 1
            
            if avg_loss <= early_stop_loss:
                logger.info(f"✅ Early stopping: Loss = {avg_loss:.4f} <= {early_stop_loss}")
                break
            
            if epochs_without_improvement >= patience:
                logger.info(f"✅ Early stopping: No improvement for {patience} epochs")
                break
        
        return history, best_model_path, training_logs
    
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
        
        if not all_crops:
            raise ValueError(f"No cell crops found in metadata files")
        
        return all_crops, all_metadata, skipped
    
    
    def execute(self, cell_crops=None, cell_metadata=None, max_epochs=25, early_stop_loss=0.5,
                batch_size=16, learning_rate=3e-5, cluster_resolution=0.5, query_cache_dir=None):
        """Execute self-supervised learning training and analysis."""
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
        
        # Create datasets
        eval_transform = self._get_eval_transform()
        eval_dataset = SingleChannelCellCropDataset(cell_crops, groups, transform=eval_transform)
        eval_loader = DataLoader(eval_dataset, batch_size=min(batch_size, num_crops), shuffle=False, num_workers=0)
        
        # Initialize model
        model = DinoV3Projector(backbone_name="dinov3_vits16", proj_dim=256).to(self.device)
        
        # Train or zero-shot
        if use_zero_shot:
            logger.info("Using zero-shot inference (no training)")
            history, loss_path = [], None
        else:
            logger.info("Starting training...")
            train_transform = self._get_augmentation_transform()
            train_dataset = SingleChannelCellCropDataset(cell_crops, groups, transform=train_transform)
            train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=0)
            
            history, best_path, _ = self._train_model(model, train_loader, max_epochs, early_stop_loss,
                                                      learning_rate, output_dir, [], patience=5)
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
        feats, names, groups_extracted = self._extract_features(model, eval_loader, self.device)
        
        # Create AnnData
        if not ANNDATA_AVAILABLE:
            return {"error": "anndata/scanpy not available"}
        
        adata = ad.AnnData(X=feats)
        adata.obs["group"] = groups_extracted if len(groups_extracted) == adata.n_obs else groups[:adata.n_obs]
        adata.obs["image_name"] = names if len(names) == adata.n_obs else ['unknown'] * adata.n_obs
        adata.obs.index = [f"cell_{i}" for i in range(adata.n_obs)]
        
        # Compute UMAP and clustering
        cluster_key = self._compute_umap_and_clustering(adata, cluster_resolution, groups if len(groups) == adata.n_obs else groups_extracted)
        
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
    # Test script
    tool = Cell_State_Analyzer_Single_Tool()
    print("Tool initialized successfully")

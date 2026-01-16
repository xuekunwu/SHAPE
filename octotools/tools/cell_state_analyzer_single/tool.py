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
except ImportError:
    raise ImportError("huggingface_hub is required for model weight download")

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
    """DINOv3 model with projection head for contrastive learning. Uses same loading strategy as multi-channel version."""
    def __init__(self, backbone_name="dinov3_vitb16", proj_dim=256):
        super().__init__()
        
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
        
        # Projection head - use feat_dim_map like multi-channel version
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
    
    def forward(self, x):
        feats = self.backbone(x)  # CLS token
        z = self.projector(feats)
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
    
    def _extract_features(self, model, dataloader):
        """Extract CLS features from model."""
        model.eval()
        feats, names, groups = [], [], []
        proj_backup = model.projector
        model.projector = nn.Identity()
        
        with torch.no_grad():
            for v1, _, n, g in tqdm(dataloader, desc="Extracting features"):
                v1 = v1.to(self.device)
                
                # Forward through backbone
                backbone_output = model.backbone(v1)
                
                # Extract CLS token
                out = backbone_output  # Assuming backbone directly returns CLS token
                feats.append(out.cpu())
                names.extend(n)
                groups.extend(g)
        
        model.projector = proj_backup
        return torch.cat(feats, dim=0).numpy(), names, groups
    
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
        # Filter to exclude overlay and visualization files (only include actual crop files)
        all_crops, all_metadata, skipped = [], [], []
        for img_id, data in metadata_by_image.items():
            if data.get('execution_status') == 'no_crops_generated':
                skipped.append({'image': img_id})
                continue
            
            crops = data.get('cell_crops_paths', [])
            metadata = data.get('cell_metadata', [])
            group = data.get('group', 'default')
            
            # Filter crops to exclude overlay and visualization files
            # Crop files should contain 'crop' in filename and not contain 'overlay', 'viz', 'mask_viz'
            filtered_crops = []
            filtered_indices = []
            for idx, crop_path in enumerate(crops):
                if isinstance(crop_path, str):
                    crop_filename_lower = os.path.basename(crop_path).lower()
                    # Exclude overlay, visualization, and mask files
                    if ('overlay' in crop_filename_lower or 
                        'viz' in crop_filename_lower or 
                        'mask_viz' in crop_filename_lower or
                        'summary' in crop_filename_lower):
                        continue
                    # Include files that contain 'crop' in filename (actual crop files)
                    if 'crop' in crop_filename_lower:
                        filtered_crops.append(crop_path)
                        filtered_indices.append(idx)
            
            if not filtered_crops:
                logger.warning(f"No valid crop files found for image {img_id} (all files appear to be overlays/visualizations). Skipping.")
                skipped.append({'image': img_id, 'reason': 'no_valid_crops'})
                continue
            
            all_crops.extend(filtered_crops)
            # Match metadata to filtered crops
            filtered_metadata = []
            for idx in filtered_indices:
                if idx < len(metadata) and isinstance(metadata[idx], dict):
                    m = metadata[idx].copy()
                    m.setdefault('group', group)
                    m.setdefault('image_name', img_id)
                    filtered_metadata.append(m)
                else:
                    filtered_metadata.append({'group': group, 'image_name': img_id})
            
            all_metadata.extend(filtered_metadata)
        
        if not all_crops:
            raise ValueError(f"No valid crop files found in metadata files (only overlays/visualizations found). Cell_State_Analyzer_Single_Tool requires actual crop files from Single_Cell_Cropper_Tool.")
        
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
        
        # Initialize model - uses dinov3_vitb16 (768 dim) to match weights
        model = DinoV3Projector(backbone_name="dinov3_vitb16", proj_dim=256).to(self.device)
        
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
        feats, names, groups_extracted = self._extract_features(model, eval_loader)
        
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
        
        # Check if results are sufficient for group comparison queries
        unique_groups = set(groups_extracted if len(groups_extracted) == adata.n_obs else groups[:adata.n_obs])
        is_single_group = len(unique_groups) == 1
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
        
        # Generate summary
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
            "can_terminate_after_chain": termination_recommended,  # Can terminate after tool chain completes
            "termination_reason": termination_reason,
            "required_next_tools": ["Analysis_Visualizer_Tool"]  # This tool requires Analysis_Visualizer_Tool to complete the workflow
        }


if __name__ == "__main__":
    # Test script
    tool = Cell_State_Analyzer_Single_Tool()
    print("Tool initialized successfully")

import os, random, torch, torch.nn as nn, torch.optim as optim
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
from skimage import io
from PIL import Image
import numpy as np
from tqdm import tqdm
from torch.cuda.amp import GradScaler, autocast
import pandas as pd
from datetime import datetime
import matplotlib.pyplot as plt
import anndata as ad
from pathlib import Path
import torch.nn.functional as F
import random
import scanpy as sc
# ====================== Dataset ======================
class MultiChannelTIFFDataset(Dataset):
    def __init__(self, root_dir, selected_channels=None, transform=None):
        self.image_paths = []
        self.groups = []
        self.selected_channels = selected_channels
        self.transform = transform

        root_dir = Path(root_dir)
        for subdir in sorted(p for p in root_dir.iterdir() if p.is_dir()):
            group = subdir.name
            for f in subdir.glob("*.tiff"):
                self.image_paths.append(str(f))
                self.groups.append(group)

        if len(self.image_paths) == 0:
            raise ValueError(f"No valid TIFF images found in {root_dir}")

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = io.imread(path)

        # (H,W,C) or (C,H,W) → (C,H,W)
        if img.ndim == 2:
            img = img[None, ...]
        elif img.shape[0] not in (1,2,3):
            img = img.permute(2, 0, 1)

        if self.selected_channels is not None:
            img = img[self.selected_channels]

        img = torch.from_numpy(img).float()

        # normalize
        if img.max() > 1:
            img = img / 255.0
        img = (img - 0.5) / 0.5

        if self.transform:
            v1 = self.transform(img)
            v2 = self.transform(img)
        else:
            v1 = v2 = img

        name = Path(path).stem
        group = self.groups[idx]

        return v1, v2, name, group


# ====================== Augmentation ======================
class MultiChannelTransform(nn.Module):
    def __init__(
        self,
        out_size=224,
        train=True,
        hflip_p=0.5,
        vflip_p=0.2,
        erase_p=0.0,   # eval 时默认 0
        eps=1e-6
    ):
        super().__init__()
        self.out_size = out_size
        self.train = train
        self.hflip_p = hflip_p
        self.vflip_p = vflip_p
        self.erase_p = erase_p
        self.eps = eps

    def forward(self, x):
        # ---- resize (always) ----
        x = F.interpolate(
            x.unsqueeze(0),
            size=(self.out_size, self.out_size),
            mode="bilinear",
            align_corners=False
        ).squeeze(0)

        if self.train:
            # ---- random flips ----
            if random.random() < self.hflip_p:
                x = torch.flip(x, dims=[2])
            if random.random() < self.vflip_p:
                x = torch.flip(x, dims=[1])

            # ---- optional random erase ----
            if self.erase_p > 0 and random.random() < self.erase_p:
                C, H, W = x.shape
                eh = int(H * random.uniform(0.02, 0.2))
                ew = int(W * random.uniform(0.02, 0.2))
                y = random.randint(0, H - eh)
                x0 = random.randint(0, W - ew)
                x[:, y:y+eh, x0:x0+ew] = 0

        return x
    
# ====================== Model ======================
class DinoV3Projector(nn.Module):
    def __init__(
        self,
        backbone_name="dinov3_vitb16",
        proj_dim=256,
        in_channels=2,                 # ⭐ 多通道入口
        freeze_patch_embed=False,       # ⭐ 控制复杂度
        freeze_blocks=0,                # ⭐ 冻结前 N 个 transformer blocks
        local_repo="/scratch/groups/joewu/xuekunwu/model_training/dinov3/dinov3_code",
        ckpt_path="/scratch/groups/joewu/xuekunwu/model_training/dinov3/dinov3_backbone/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth",
    ):
        super().__init__()
        self.backbone = torch.hub.load(
            local_repo,
            backbone_name,
            source="local",
            pretrained=False
        )

        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "teacher" in state_dict:
            state_dict = state_dict["teacher"]
        self.backbone.load_state_dict(state_dict, strict=False)

        # -------------------------------------------------
        # 2. Adapt patch embedding for multi-channel input
        # -------------------------------------------------
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

        if freeze_patch_embed:
            for p in self.backbone.patch_embed.parameters():
                p.requires_grad = False

        if freeze_blocks > 0:
            for blk in self.backbone.blocks[:freeze_blocks]:
                for p in blk.parameters():
                    p.requires_grad = False
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
        """
        x: (B, C, H, W)
        return: normalized embedding (B, proj_dim)
        """
        feats = self.backbone(x)              # CLS token
        z = self.projector(feats)
        return F.normalize(z, dim=-1)


# ====================== Contrastive loss ======================
def contrastive_loss(z1, z2, temperature=0.1):
    z1 = F.normalize(z1, dim=1)
    z2 = F.normalize(z2, dim=1)
    logits = z1 @ z2.T / temperature
    labels = torch.arange(z1.size(0), device=z1.device)
    loss_12 = F.cross_entropy(logits, labels)
    loss_21 = F.cross_entropy(logits.T, labels)
    return 0.5 * (loss_12 + loss_21)

# ====================== Feature Extraction ======================
def extract_cls_features(model, dataloader, device):
    model.eval()
    feats, img_names, celltypes = [], [], []
    has_proj = hasattr(model, "projector")
    if has_proj:
        proj_backup = model.projector
        model.projector = torch.nn.Identity()
    with torch.no_grad():
        for v1, _, names, cts in tqdm(dataloader, desc="Extracting CLS features"):
            v1 = v1.to(device)
            out = model.backbone(v1)
            feats.append(out.cpu())
            img_names.extend(names)
            celltypes.extend(cts)
    if has_proj:
        model.projector = proj_backup
    feats_all = torch.cat(feats, dim=0).numpy()
    print(f"✅ Extracted CLS features: {feats_all.shape}")
    return feats_all, img_names, celltypes

def print_gpu_usage():
    if torch.cuda.is_available():
        gpu_name = torch.cuda.get_device_name(0)
        total_mem = torch.cuda.get_device_properties(0).total_memory / 1024**3
        allocated = torch.cuda.memory_allocated(0) / 1024**3
        reserved = torch.cuda.memory_reserved(0) / 1024**3
        print(f"GPU: {gpu_name} | Total: {total_mem:.2f} GB | "
              f"Allocated: {allocated:.2f} GB | Reserved: {reserved:.2f} GB")
        
# ====================== Training ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = "/scratch/users/xuekunwu/data/Cellimage/250429_96well_CO_DEFINE candidates"
    date_prefix = datetime.now().strftime("%y%m%d")
    save_dir = f"/scratch/users/xuekunwu/model/CO_screen_{date_prefix}"
    os.makedirs(save_dir, exist_ok=True)

    epochs, batch_size, lr = 100, 128, 3e-5
    backbone_name = "dinov3_vitb16"
    train_transform = MultiChannelTransform(out_size=224,train=True,hflip_p=0.5,vflip_p=0.2,erase_p=0.3)
    dataset = MultiChannelTIFFDataset(train_dir, selected_channels=[0,1], transform=train_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f"✅ Loaded {len(dataset)} images")

    model = DinoV3Projector(backbone_name, proj_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()

    history, best_loss = [], float("inf")
    best_model_path = os.path.join(save_dir, "best_model.pth")

    for epoch in range(epochs):
        model.train(); total_loss = 0
        for v1, v2, _, _ in tqdm(loader, desc=f"Epoch {epoch+1}/{epochs}"):
            v1, v2 = v1.to(device), v2.to(device)
            optimizer.zero_grad()
            with autocast():
                z1, z2 = model(v1), model(v2)
                loss = contrastive_loss(z1, z2)
            scaler.scale(loss).backward()
            scaler.step(optimizer); scaler.update()
            total_loss += loss.item()

        avg_loss = total_loss / len(loader)
        history.append(avg_loss)
        print(f"Epoch {epoch+1}: Loss = {avg_loss:.4f}")
        print_gpu_usage()  # ✅ 打印 GPU 使用情况
        if avg_loss < best_loss:
            best_loss = avg_loss
            torch.save(model.state_dict(), best_model_path)
            print(f"🔥 New best model saved (Loss = {best_loss:.4f})")

        # === 每 10 epoch 保存一次特征 ===
        if (epoch + 1) % 20 == 0 or (epoch + 1) == epochs:
            print(f"\n🔍 Extracting CLS features at epoch {epoch + 1}...")
            model.eval()
            eval_transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize([0.4636,0.5032,0.5822],[0.2483,0.2660,0.2907])
            ])
            eval_transform = MultiChannelTransform(out_size=224,train=False)
            eval_dataset = MultiChannelTIFFDataset(train_dir, selected_channels=[0, 1], transform=eval_transform)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            feats, names, celltypes = extract_cls_features(model, eval_loader, device)
            adata = ad.AnnData(X=feats)
            adata.obs["image_name"] = names
            adata.obs["celltype"] = celltypes
            adata.obs.index = [f"cell_{i}" for i in range(adata.n_obs)]
            sc.pp.neighbors(adata, use_rep='X', n_neighbors=20, metric="cosine")
            sc.tl.umap(adata, min_dist=0.05, spread=1, random_state=42)
            resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1]
            for r in resolutions:
                key = f"leiden{r}"
                sc.tl.leiden(adata, resolution=r, key_added=key)
            save_path = os.path.join(save_dir, f"CLS_features_epoch_{epoch+1}.h5ad")
            adata.write(save_path)
            print(f"✅ CLS features saved to {save_path}")

            model.train()

    pd.DataFrame({"Epoch": range(1, epochs+1), "Loss": history}).to_excel(os.path.join(save_dir, "training_history.xlsx"), index=False)
    plt.plot(range(1, len(history)+1), history)
    plt.xlabel("Epoch"); plt.ylabel("Contrastive Loss"); plt.title("Training Loss")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))

    print("✅ Training completed.")

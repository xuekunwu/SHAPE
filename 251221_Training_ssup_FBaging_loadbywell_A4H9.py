import os, random,  torch, torch.nn as nn, torch.optim as optim
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
import seaborn as sns
import umap
import anndata as ad
from pathlib import Path
import subprocess
import scanpy as sc
# ====================== Dataset ======================
class UnsupervisedTIFFDataset(Dataset):
    def __init__(self, dir_list, wells=None, transform=None):
        if isinstance(dir_list, str):
            dir_list = [dir_list]
        if wells is not None:
            if isinstance(wells, str):
                wells = [wells]
            self.allowed_wells = set(w.upper() for w in wells)
        else:
            self.allowed_wells = None
        self.image_paths = []
        self.celltypes = []
        self.transform = transform

        for root_dir in dir_list:
            subfolders = [f for f in Path(root_dir).iterdir() if f.is_dir()]
            for sub in subfolders:
                files = list(sub.rglob("*.tif"))
                kept = 0
                for f in files:
                    well = self._parse_well_from_name(f.name)
                    if well is None:
                        continue
                    if self.allowed_wells is None or well in self.allowed_wells:
                        self.image_paths.append(str(f))
                        self.celltypes.append("unknown")
                        kept += 1
                print(f"Loaded {kept}/{len(files)} images from folder: {sub.name}")
        
    @staticmethod
    def _parse_well_from_name(filename):
        parts = filename.split("_")
        if len(parts) < 3:
            return None
        return parts[1].upper()

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        path = self.image_paths[idx]
        img = io.imread(path).astype(np.float32)
        if img.dtype == np.uint16:
            img = img / 65535.0
        else:
            img = img / 255.0
        if img.ndim == 2:
            img = np.repeat(img[..., None], 3, axis=-1)
        img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
        if self.transform:
            v1 = self.transform(img)
            v2 = self.transform(img)
        else:
            tensor = transforms.ToTensor()(img)
            v1 = v2 = tensor
        image_name = os.path.splitext(os.path.basename(path))[0]
        celltype = self.celltypes[idx]
        return v1, v2, image_name, celltype

# ====================== Augmentation ======================
augmentation_transform = transforms.Compose([
            transforms.RandomResizedCrop(224, scale=(0.3,1.0)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.2),
            transforms.RandomRotation(60),
            transforms.RandomApply([transforms.ColorJitter(0.6,0.6,0.6,0.1)], p=0.8),
            transforms.RandomGrayscale(p=0.2),
            transforms.RandomApply([transforms.GaussianBlur(kernel_size=23, sigma=(0.1,2.0))], p=0.8),
            transforms.ToTensor(),
            transforms.RandomErasing(p=0.3, scale=(0.02,0.2), ratio=(0.3,3.3)),
            transforms.Normalize([0.4636,0.5032,0.5822],[0.2483,0.2660,0.2907])
        ])

# ====================== Model ======================
class DinoV3Projector(nn.Module):
    def __init__(self,
                 backbone_name="dinov3_vitb16",
                 proj_dim=256,
                 local_repo="/scratch/groups/joewu/xuekunwu/model_training/dinov3/dinov3_code",
                 ckpt_path="/scratch/groups/joewu/xuekunwu/model_training/dinov3/dinov3_backbone/dinov3_vitb16_pretrain_lvd1689m-73cec8be.pth"):
        super().__init__()

        self.backbone = torch.hub.load(local_repo, backbone_name, source="local", pretrained=False)
        state_dict = torch.load(ckpt_path, map_location="cpu")
        if "teacher" in state_dict:
              state_dict = state_dict["teacher"]
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
        z = self.backbone(x)
        z = self.projector(z)
        return nn.functional.normalize(z, dim=-1)

# ====================== Contrastive loss ======================
def contrastive_loss(z1, z2, temperature=0.1):
    z1 = nn.functional.normalize(z1, dim=1)
    z2 = nn.functional.normalize(z2, dim=1)
    logits = torch.mm(z1, z2.t()) / temperature
    labels = torch.arange(z1.size(0)).to(z1.device)
    return nn.CrossEntropyLoss()(logits, labels)

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

def print_nvidia_smi():
    result = subprocess.run(
        ["nvidia-smi", "--query-gpu=memory.used,memory.total", "--format=csv,noheader,nounits"],
        stdout=subprocess.PIPE,
        text=True
    )
    used, total = result.stdout.strip().split(", ")
    print(f"nvidia-smi: {used} MB / {total} MB")
        
# ====================== Training ======================
if __name__ == "__main__":
    device = "cuda" if torch.cuda.is_available() else "cpu"
    train_dir = "/scratch/users/xuekunwu/data/Cellimage/SkinFB_aging_timecourse"
    date_prefix = datetime.now().strftime("%y%m%d")
    save_dir = f"/scratch/users/xuekunwu/model/FBaging_training_{date_prefix}"
    os.makedirs(save_dir, exist_ok=True)

    epochs, batch_size, lr = 100, 128, 3e-5
    backbone_name = "dinov3_vitb16"
    well_select =["A4", "H9"]

    dataset = UnsupervisedTIFFDataset(train_dir, wells=well_select, transform=augmentation_transform)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True, num_workers=0, pin_memory=True)
    print(f"✅ Loaded {len(dataset)} images")

    model = DinoV3Projector(backbone_name, proj_dim=256).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=1e-4)
    scaler = GradScaler()

    history, best_loss = [], float("inf")
    best_model_path = os.path.join(save_dir, "A4_H9_best_model.pth")

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
            eval_dataset = UnsupervisedTIFFDataset(train_dir, wells=well_select, transform=eval_transform)
            eval_loader = DataLoader(eval_dataset, batch_size=batch_size, shuffle=False, num_workers=2)
            feats, names, celltypes = extract_cls_features(model, eval_loader, device)
            adata = ad.AnnData(X=feats)
            adata.obs["image_name"] = names
            adata.obs["celltype"] = celltypes
            adata.obs.index = [f"cell_{i}" for i in range(adata.n_obs)]
            sc.pp.neighbors(adata, use_rep='X', n_neighbors=20, metric="cosine")  # 使用余弦距离更适合图像特征
            sc.tl.umap(adata, min_dist=0.05, spread=1, random_state=42)
            resolutions = [0.1, 0.2, 0.3, 0.4, 0.5, 0.7, 0.9, 1.1]
            for r in resolutions:
                key = f"leiden{r}"
                sc.tl.leiden(adata, resolution=r, key_added=key)
            save_path = os.path.join(save_dir, f"A4_H9_epoch_{epoch+1}.h5ad")
            adata.write(save_path)
            print(f"✅ CLS features saved to {save_path}")

            model.train()

    pd.DataFrame({"Epoch": range(1, epochs+1), "Loss": history}).to_excel(os.path.join(save_dir, "A4_H9_training_history.xlsx"), index=False)
    plt.plot(range(1, len(history)+1), history)
    plt.xlabel("Epoch"); plt.ylabel("Contrastive Loss"); plt.title("Training Loss")
    plt.savefig(os.path.join(save_dir, "loss_curve.png"))

    print("✅ Training completed.")
import cv2
import numpy as np
from skimage.measure import label, regionprops
from pathlib import Path
from tqdm import tqdm
from PIL import Image
import os
from cellpose import models
from concurrent.futures import ThreadPoolExecutor, as_completed

# ---- 运行 Cellpose 模型并生成 mask ----
def run_model_mask(image_path, model, flow_threshold=0.6, cellprob_threshold=0):
    img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if img is None:
        raise ValueError(f"Cannot read {image_path}")
    img = img.astype(np.float32)
    masks, flows, styles = model.eval(
        [img],
        diameter=None,
        flow_threshold=flow_threshold,
        cellprob_threshold=cellprob_threshold,
        progress=False
    )
    return img, masks[0].astype(np.uint8)

# ---- 生成单细胞裁剪 ----
def generate_multichannel_crops(base_name, mask, trans_img, fitc_img, dapi_img, output_dir, min_area=200, margin=25, output_format='tiff'):
    labeled_mask = label(mask)
    regions = regionprops(labeled_mask)
    cell_idx = 0
    for r in regions:
        if r.area < min_area:
            continue
        minr, minc, maxr, maxc = r.bbox
        h, w = maxr - minr, maxc - minc
        center_r, center_c = (minr + maxr)//2, (minc + maxc)//2
        half_side = max(h//2, w//2) + margin
        new_minr = max(center_r - half_side, 0)
        new_maxr = min(center_r + half_side, trans_img.shape[0])
        new_minc = max(center_c - half_side, 0)
        new_maxc = min(center_c + half_side, trans_img.shape[1])

        crop_trans = trans_img[new_minr:new_maxr, new_minc:new_maxc]
        crop_fitc  = fitc_img[new_minr:new_maxr, new_minc:new_maxc]
        crop_dapi  = dapi_img[new_minr:new_maxr, new_minc:new_maxc]

        # 确保三个通道尺寸一致
        min_h = min(crop_trans.shape[0], crop_fitc.shape[0], crop_dapi.shape[0])
        min_w = min(crop_trans.shape[1], crop_fitc.shape[1], crop_dapi.shape[1])

        crop_trans = crop_trans[:min_h, :min_w]
        crop_fitc  = crop_fitc[:min_h, :min_w]
        crop_dapi  = crop_dapi[:min_h, :min_w]

        # 合成三通道：RGB 顺序可自定义 (FITC, DAPI, Trans)
        merged = cv2.merge([
            crop_fitc[:, :, 1],   # green channel from FITC
            crop_dapi[:, :, 0],   # blue channel
            crop_trans[:, :, 0]   # gray/bright channel
        ])

        crop_name = f"{base_name}_organoid{cell_idx:03d}.{output_format}"
        out_path = os.path.join(output_dir, crop_name)
        Image.fromarray(merged).save(out_path)
        cell_idx += 1

    return cell_idx

# ---- 单图处理 ----
def process_single_image_group(trans_path, model, output_root, created_dirs, min_area, margin, flow_threshold, cellprob_threshold):
    base_name = trans_path.stem.replace("_trans", "")
    parent_dir = trans_path.parent
    fitc_path = parent_dir / f"{base_name}_fitc.tiff"
    dapi_path = parent_dir / f"{base_name}_dapi.tiff"

    if not fitc_path.exists() or not dapi_path.exists():
        return f"⚠️ Missing FITC or DAPI for {base_name}"

    # ---- Load all channels ----
    trans_img = cv2.imread(str(trans_path))
    fitc_img  = cv2.imread(str(fitc_path))
    dapi_img  = cv2.imread(str(dapi_path))

    # ---- Run segmentation on brightfield ----
    _, mask = run_model_mask(str(trans_path), model, flow_threshold, cellprob_threshold)

    # ---- Determine output directory ----
    cell_type = trans_path.parent.name
    output_dir = output_root / cell_type
    if cell_type not in created_dirs:
        output_dir.mkdir(parents=True, exist_ok=True)
        created_dirs.add(cell_type)

    # ---- Generate multi-channel crops ----
    n_crops = generate_multichannel_crops(
        base_name, mask, trans_img, fitc_img, dapi_img, output_dir,
        min_area=min_area, margin=margin
    )
    return f"{base_name}: {n_crops} crops"

# ---- 主函数 ----
def run_organoid_crop(image_folder, model, output_root,
                      min_area=200, margin=10, flow_threshold=0.6,
                      cellprob_threshold=0, max_workers=4):
    image_folder = Path(image_folder)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trans_paths = list(image_folder.glob("*_trans.tiff"))
    print(f"📸 Found {len(trans_paths)} brightfield images under {image_folder}")
    created_dirs = set()

    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_image_group, p, model, output_root, created_dirs,
                                min_area, margin, flow_threshold, cellprob_threshold)
                for p in trans_paths
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Processing groups"):
                print(f.result())
    else:
        for p in tqdm(trans_paths, desc="Processing groups"):
            print(process_single_image_group(p, model, output_root, created_dirs,
                                             min_area, margin, flow_threshold, cellprob_threshold))

# ---- 执行入口 ----
if __name__ == "__main__":
    model_path = "/scratch/groups/joewu/xuekunwu/CO_4x_V2"
    input_path = "/scratch/groups/joewu/xuekunwu/data/XW-CO-Mdf-251029"
    output_path = "/scratch/groups/joewu/xuekunwu/XW-CO-Mdf-251029/crops"

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)

    run_organoid_crop(
        image_folder=input_path,
        model=model,
        output_root=output_path,
        min_area=200, margin=10,
        flow_threshold=0.6, cellprob_threshold=0,
        max_workers=8
    )

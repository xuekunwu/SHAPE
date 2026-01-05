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
def generate_cell_crops(original_img, mask, output_dir, base_name, min_area=50, margin=25, output_format='tif'):
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
        new_maxr = min(center_r + half_side, original_img.shape[0])
        new_minc = max(center_c - half_side, 0)
        new_maxc = min(center_c + half_side, original_img.shape[1])
        side = min(new_maxr - new_minr, new_maxc - new_minc)
        new_maxr = new_minr + side
        new_maxc = new_minc + side
        if (new_maxr - new_minr) != (new_maxc - new_minc):
            continue
        crop = original_img[new_minr:new_maxr, new_minc:new_maxc]
        if crop.size == 0:
            continue
        crop = np.clip(crop, 0, 255).astype(np.uint8)
        crop_name = f"{base_name}_cell{cell_idx:04d}.{output_format}"
        Image.fromarray(crop).save(os.path.join(output_dir, crop_name))
        cell_idx += 1
    return cell_idx

# ---- 单图处理 ----
def process_single_image(img_path, model, output_root, created_dirs, min_area, margin, flow_threshold, cellprob_threshold):
    try:
        img, mask = run_model_mask(str(img_path), model, flow_threshold, cellprob_threshold)
        parts = img_path.stem.split("_")
        cell_type = parts[-1] if len(parts) > 1 else img_path.parent.name
        if cell_type not in created_dirs:
            output_dir = output_root / cell_type
            output_dir.mkdir(parents=True, exist_ok=True)
            created_dirs.add(cell_type)
        else:
            output_dir = output_root / cell_type
        n_crops = generate_cell_crops(img, mask, output_dir, img_path.stem, min_area, margin)
        return f"{img_path.name}: {n_crops} crops"
    except Exception as e:
        return f"❌ Error processing {img_path.name}: {e}"

# ---- 主函数 ----
def run_single_cell_crop(image_folder, model, output_root, min_area=50, margin=25,
                         flow_threshold=0.6, cellprob_threshold=0, max_workers=4):
    image_folder = Path(image_folder)
    output_root = Path(output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    image_paths = list(image_folder.glob("*.jpg"))
    print(f"📸 Found {len(image_paths)} images under {image_folder}")
    created_dirs = set()
    if max_workers > 1:
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = [
                executor.submit(process_single_image, img_path, model, output_root, created_dirs,
                                min_area, margin, flow_threshold, cellprob_threshold)
                for img_path in image_paths
            ]
            for f in tqdm(as_completed(futures), total=len(futures), desc="Processing images"):
                print(f.result())
    else:
        for img_path in tqdm(image_paths, desc="Processing images"):
            print(process_single_image(img_path, model, output_root, created_dirs,
                                       min_area, margin, flow_threshold, cellprob_threshold))

# ---- 执行入口 ----
if __name__ == "__main__":
    model_path = "/home/users/xuekunwu/cellpose_model/cpsam_lr_1e-04"
    input_path = "/scratch/groups/joewu/xuekunwu/EVICAN"
    output_path = "/scratch/groups/joewu/xuekunwu/EVICAN crops"

    model = models.CellposeModel(gpu=True, pretrained_model=model_path)
    run_single_cell_crop(input_path, model, output_path,
                         min_area=50, margin=25,
                         flow_threshold=0.6, cellprob_threshold=0,
                         max_workers=8)

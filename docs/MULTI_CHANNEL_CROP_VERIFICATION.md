# Multi-Channel Crop Pipeline Verification

## 问题检查

### 当前Pipeline状态

#### 1. Single_Cell_Cropper_Tool ✅
**状态**: 正确保留多通道信息

**验证点**:
- ✅ 使用 `ImageProcessor.load_image()` 加载图像，检测 `is_multi_channel`
- ✅ 在cropping时使用 `original_img[new_minr:new_maxr, new_minc:new_maxc, :]` 保留所有通道
- ✅ 多通道图像保存为TIFF格式: `tifffile.imwrite(crop_path, cell_crop)`
- ✅ 文件名包含image_name: `{image_name}_cell_{idx}_crop.tiff`

**代码位置**: `octotools/tools/single_cell_cropper/tool.py`
- Lines 119-127: 检测多通道
- Lines 527-530: 保留所有通道的cropping
- Lines 557-567: 保存为多通道TIFF

#### 2. Cell_State_Analyzer_Tool ⚠️
**状态**: 可能丢失多通道信息

**问题点**:
- ⚠️ `CellCropDataset.__getitem__()` 使用 `Image.fromarray(...).convert("RGB")`
- ⚠️ 对于2通道图像（bright-field + GFP），转换为RGB时可能丢失GFP通道信息
- ⚠️ 如果图像是2通道的，PIL的 `convert("RGB")` 可能只使用第一个通道

**代码位置**: `octotools/tools/cell_state_analyzer/tool.py`
- Lines 87-97: 图像加载和转换逻辑

**当前逻辑**:
```python
if SKIMAGE_AVAILABLE and (path.lower().endswith('.tif') or path.lower().endswith('.tiff')):
    img = io.imread(path).astype(np.float32)
    if img.dtype == np.uint16:
        img = img / 65535.0
    else:
        img = img / 255.0
    if img.ndim == 2:
        img = np.repeat(img[..., None], 3, axis=-1)
    img = Image.fromarray((img * 255).astype(np.uint8)).convert("RGB")
```

**问题**:
- 如果 `img.ndim == 3` 且 `img.shape[2] == 2`（2通道图像），直接转换为RGB会丢失信息
- 需要将2通道图像正确映射到RGB（例如：BF→R/G/B, GFP→G）

## 修复方案

### 方案1: 智能多通道到RGB转换（推荐）

对于多通道图像，应该：
1. **2通道** (bright-field + GFP): 
   - BF → R, G, B (grayscale)
   - GFP → G (green channel)
   - 合并: R=BF, G=BF+GFP, B=BF

2. **3通道** (例如: BF + GFP + DAPI):
   - 直接使用前3个通道作为RGB

3. **>3通道**:
   - 使用前3个通道，或创建合并视图

### 方案2: 使用ImageProcessor统一处理

使用 `ImageProcessor.create_merged_rgb_for_display()` 来创建RGB视图，确保所有通道信息都被利用。

## 验证检查清单

- [ ] Single_Cell_Cropper_Tool 正确保存多通道TIFF ✅
- [ ] Cell_State_Analyzer_Tool 正确加载多通道TIFF ⚠️ 需要修复
- [ ] 多通道信息在整个pipeline中正确传递 ⚠️ 需要修复
- [ ] Analysis_Visualizer_Tool 正确显示多通道exemplars ✅ (已修复)


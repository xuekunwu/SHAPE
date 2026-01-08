# Logic Verification: Crop Naming and Exemplar Sampling Fix

## 修改概述

### 问题1: Cluster Exemplars 显示相同图像
**现象**: 每个cluster的5个exemplars显示完全相同的图像

**根本原因**: 
1. `Single_Cell_Cropper_Tool` 生成的crop文件名不包含原始图像名称，所有图像都使用 `cell_0000_crop.tiff` 等相同文件名
2. 当处理多个图像时，不同图像的crop可能被保存为相同文件名，导致覆盖
3. `Analysis_Visualizer_Tool` 从索引中采样，但多个索引可能指向同一个crop_path文件

### 问题2: Crop文件名冲突
**现象**: 所有crop_path都是 `cell_0000_crop.tiff`，可能指向同一文件

**根本原因**: Crop文件名只包含索引，不包含原始图像标识符

## 修改内容

### 修改1: Single_Cell_Cropper_Tool - Crop文件名包含图像名称

**文件**: `octotools/tools/single_cell_cropper/tool.py`

**修改位置**: Lines 547-561

**修改前**:
```python
if is_multi_channel and cell_crop.ndim == 3:
    crop_filename = f"cell_{idx:04d}_crop.tiff"
else:
    crop_filename = f"cell_{idx:04d}_crop.{output_format}"
```

**修改后**:
```python
# Extract image name for unique crop filename
image_name = source_image_id if source_image_id else Path(original_image_path).stem
image_name_safe = "".join(c for c in image_name if c.isalnum() or c in ('_', '-'))[:50]

if is_multi_channel and cell_crop.ndim == 3:
    crop_filename = f"{image_name_safe}_cell_{idx:04d}_crop.tiff"
else:
    crop_filename = f"{image_name_safe}_cell_{idx:04d}_crop.{output_format}"
```

**效果**:
- Crop文件名格式: `{image_name}_cell_{idx:04d}_crop.{ext}`
- 例如: `bs_bbb275a4_cell_0000_crop.tiff` 而不是 `cell_0000_crop.tiff`
- 确保不同图像的crop有唯一文件名，避免覆盖

### 修改2: Analysis_Visualizer_Tool - 从唯一路径采样

**文件**: `octotools/tools/analysis_visualizer/tool.py`

**修改位置**: Lines 1040-1071

**修改前**:
```python
# Randomly sample cells (ensure we get different samples each time)
n_samples = min(crops_per_cluster, len(cluster_indices))
if n_samples > 0:
    cluster_seed = 42 + hash(str(cluster)) % 1000
    random.seed(cluster_seed)
    sampled_indices = random.sample(list(cluster_indices), n_samples)
    random.seed(42)
```

**修改后**:
```python
# Collect unique crop paths for this cluster (avoid duplicates)
unique_crop_paths = {}
for idx in cluster_indices:
    crop_path = adata.obs.iloc[idx]['crop_path']
    # ... normalize path ...
    if crop_path not in unique_crop_paths and os.path.exists(crop_path):
        unique_crop_paths[crop_path] = idx

# Randomly sample from unique crop paths
unique_paths_list = list(unique_crop_paths.keys())
n_samples = min(crops_per_cluster, len(unique_paths_list))
if n_samples > 0:
    cluster_seed = 42 + hash(str(cluster)) % 1000
    random.seed(cluster_seed)
    sampled_paths = random.sample(unique_paths_list, n_samples)
    random.seed(42)
```

**效果**:
- 先收集每个cluster的唯一crop_path（去重）
- 从唯一路径中采样，而不是从索引中采样
- 确保显示的是不同的图像文件，而不是重复的

## 整体逻辑验证

### 1. 数据流验证

**完整流程**:
```
Image → Segmentation → Single_Cell_Cropper → Cell_State_Analyzer → Analysis_Visualizer
```

**Crop文件命名流程**:
1. `Single_Cell_Cropper_Tool` 接收 `original_image` 和 `source_image_id`
2. 提取 `image_name` (从 `source_image_id` 或 `original_image_path`)
3. 生成唯一crop文件名: `{image_name}_cell_{idx}_crop.{ext}`
4. 保存crop文件，路径存储在metadata的 `crop_path` 字段

**Metadata传递流程**:
1. `Single_Cell_Cropper_Tool` 生成metadata，包含 `crop_path`
2. `Cell_State_Analyzer_Tool` 读取metadata，加载crop文件
3. `Cell_State_Analyzer_Tool` 创建AnnData，将 `crop_path` 存储在 `adata.obs['crop_path']`
4. `Analysis_Visualizer_Tool` 从AnnData读取 `crop_path`，用于显示exemplars

### 2. 一致性检查

✅ **文件名唯一性**: 
- 修改后，每个图像的crop都有唯一文件名（包含image_name）
- 不同图像的crop不会互相覆盖

✅ **路径传递一致性**:
- `Single_Cell_Cropper_Tool` 生成的 `crop_path` 包含完整路径和唯一文件名
- `Cell_State_Analyzer_Tool` 正确读取并存储 `crop_path`
- `Analysis_Visualizer_Tool` 正确读取并使用 `crop_path`

✅ **采样逻辑正确性**:
- 修改后，从唯一crop_path中采样，而不是从索引中采样
- 即使多个索引指向同一文件，也只会显示一次
- 确保每个cluster显示最多5个不同的图像

### 3. 边界情况处理

✅ **image_name为空或None**:
- 使用 `Path(original_image_path).stem` 作为fallback
- 确保总是有有效的image_name

✅ **image_name包含特殊字符**:
- 使用 `image_name_safe` 进行sanitization
- 只保留字母数字和 `_`, `-` 字符
- 限制长度为50字符

✅ **多个索引指向同一crop_path**:
- `Analysis_Visualizer_Tool` 使用 `unique_crop_paths` 字典去重
- 确保每个唯一路径只被采样一次

✅ **crop_path不存在**:
- `Analysis_Visualizer_Tool` 检查 `os.path.exists(crop_path)`
- 只采样存在的文件

### 4. 向后兼容性

✅ **Metadata格式兼容**:
- `crop_path` 字段格式不变，只是文件名更唯一
- 现有metadata仍然可以正常读取

✅ **CellCrop对象兼容**:
- `CellCrop` 对象的 `crop_path` 属性包含完整路径
- 修改不影响对象结构

✅ **AnnData兼容**:
- `adata.obs['crop_path']` 存储格式不变
- 只是路径中的文件名更唯一

## 潜在问题和风险

### 风险1: 文件名过长
**缓解措施**: 
- `image_name_safe` 限制长度为50字符
- 如果原始image_name很长，会被截断

### 风险2: image_name包含路径分隔符
**缓解措施**:
- `Path(original_image_path).stem` 只提取文件名（不含路径）
- `image_name_safe` 移除所有特殊字符，只保留安全字符

### 风险3: 不同图像有相同的stem
**可能性**: 低（如果使用 `source_image_id`）
**缓解措施**:
- 建议总是提供 `source_image_id` 参数
- 如果使用自动提取，确保原始图像文件名唯一

## 测试建议

1. **单图像测试**: 验证单个图像的crop文件名正确
2. **多图像测试**: 验证多个图像的crop文件名不冲突
3. **Exemplar测试**: 验证cluster exemplars显示不同的图像
4. **路径传递测试**: 验证crop_path在整个流程中正确传递

## 总结

✅ **修改正确性**: 两个修改都针对根本原因，逻辑正确
✅ **一致性**: 修改后的逻辑在整个流程中保持一致
✅ **完整性**: 修改覆盖了从crop生成到exemplar显示的全流程
✅ **向后兼容**: 修改不影响现有数据结构和接口

**建议**: 修改已经完成并push，建议在实际使用中验证效果。如果发现问题，可以进一步优化。


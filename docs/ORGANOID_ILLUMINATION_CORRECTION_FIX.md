# Organoid Illumination Correction Fix

## 修改概述

对于organoid图像，不需要进行illumination correction，只需要调整亮度。

## 修改内容

### Image_Preprocessor_Tool

**文件**: `octotools/tools/image_preprocessor/tool.py`

**修改1: 添加参数**
- 在 `execute` 方法中添加 `skip_illumination_correction` 参数（默认: False）
- 当 `skip_illumination_correction=True` 时，跳过illumination correction，只调整亮度

**修改2: 单通道处理逻辑**
```python
# For organoid images: skip illumination correction, only adjust brightness
if skip_illumination_correction:
    print(f"⚠️ Skipping illumination correction (organoid mode). Only adjusting brightness...")
    corrected_image = original_image  # Use original image without illumination correction
    normalized_image = self.adjust_brightness(corrected_image, target_brightness)
else:
    # Standard preprocessing: illumination correction + brightness adjustment
    corrected_image = self.global_illumination_correction(original_image, gaussian_kernel_size)
    normalized_image = self.adjust_brightness(corrected_image, target_brightness)
```

**修改3: 多通道处理逻辑**
- 对于bright-field通道，如果 `skip_illumination_correction=True`，跳过illumination correction
- 只调整亮度，保留原始图像特征

## 使用方式

### 对于Organoid图像

当调用 `Image_Preprocessor_Tool` 处理organoid图像时，设置 `skip_illumination_correction=True`:

```python
execution = tool.execute(
    image=image_path,
    target_brightness=120,
    skip_illumination_correction=True  # Skip illumination correction for organoids
)
```

### 对于其他图像

保持默认行为（`skip_illumination_correction=False`），进行标准的illumination correction和亮度调整。

## 验证

✅ **单通道图像**: 跳过illumination correction，只调整亮度
✅ **多通道图像**: bright-field通道跳过illumination correction，只调整亮度
✅ **向后兼容**: 默认行为不变（`skip_illumination_correction=False`）

## 与Organoid_Segmenter_Tool的配合

`Organoid_Segmenter_Tool` 已经实现了只调整亮度的逻辑（`check_brightness_only` 和 `adjust_brightness_only`）。

如果 `Organoid_Segmenter_Tool` 需要调用 `Image_Preprocessor_Tool`，应该传递 `skip_illumination_correction=True`。

## 总结

✅ **修改完成**: `Image_Preprocessor_Tool` 现在支持跳过illumination correction
✅ **Organoid支持**: 对于organoid图像，只调整亮度，不进行illumination correction
✅ **向后兼容**: 默认行为保持不变


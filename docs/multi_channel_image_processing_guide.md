# 多通道图像处理思维链指南

## 概述
本文档总结了处理多通道 TIFF 图像（如 bright-field 和 GFP 荧光通道）的完整思维链和最佳实践。

## 思维链步骤

### 1. 加载和检查图像结构

**关键操作：**
```python
import tifffile
import numpy as np

# 加载图像
img_full = tifffile.imread(image_path)

# 检查基本属性
print(f"Image shape: {img_full.shape}")
print(f"Image dtype: {img_full.dtype}")
print(f"Image min/max values: {img_full.min()} / {img_full.max()}")

# 识别维度结构
if img_full.ndim == 3:
    if img_full.shape[0] <= 4:  # (C, H, W) 格式
        num_channels = img_full.shape[0]
        height, width = img_full.shape[1], img_full.shape[2]
    elif img_full.shape[2] <= 4:  # (H, W, C) 格式
        num_channels = img_full.shape[2]
        height, width = img_full.shape[0], img_full.shape[1]
```

**检查清单：**
- [ ] 图像形状（shape）
- [ ] 数据类型（dtype，通常是 uint16 或 uint8）
- [ ] 值范围（min/max）
- [ ] 维度结构（2D, 3D, 4D）
- [ ] 通道数量

### 2. 分离和分析通道

**关键操作：**
```python
# 提取每个通道
channels = []
for ch_idx in range(num_channels):
    if img_full.shape[0] <= 4:  # (C, H, W)
        channel = img_full[ch_idx, :, :]
    else:  # (H, W, C)
        channel = img_full[:, :, ch_idx]
    channels.append(channel)

# 分析每个通道的统计特征
channel_stats = []
for ch_idx, channel in enumerate(channels):
    stats = {
        'channel_index': ch_idx,
        'shape': channel.shape,
        'dtype': channel.dtype,
        'min': channel.min(),
        'max': channel.max(),
        'mean': channel.mean(),
        'std': channel.std(),
        'median': np.median(channel)
    }
    channel_stats.append(stats)
    print(f"Channel {ch_idx}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
```

**分析指标：**
- **均值（Mean）**：通道的平均亮度
- **标准差（Std）**：对比度/纹理丰富度
- **最小值/最大值**：动态范围
- **中位数（Median）**：不受异常值影响的中心趋势

### 3. 智能通道识别

**基于统计特征的推断规则：**

```python
def identify_channels(channel_stats):
    """
    基于统计特征识别通道类型
    
    规则：
    - Bright-field: 通常有更高的对比度（更大的标准差）
    - GFP: 通常有更均匀的强度分布（较小的标准差）
    - 如果均值差异大，均值高的可能是 bright-field
    """
    # 计算对比度指标（标准差）
    contrast_scores = [stats['std'] for stats in channel_stats]
    
    # 计算亮度指标（均值）
    brightness_scores = [stats['mean'] for stats in channel_stats]
    
    # 综合指标：对比度 × 亮度
    combined_scores = [c * b for c, b in zip(contrast_scores, brightness_scores)]
    
    # 识别 bright-field（通常对比度最高）
    bf_idx = np.argmax(combined_scores)
    
    # 识别 GFP（通常是第二个通道，或对比度较低的通道）
    if len(channel_stats) >= 2:
        # 排除 bright-field 后，选择对比度第二高的
        remaining_indices = [i for i in range(len(channel_stats)) if i != bf_idx]
        gfp_idx = remaining_indices[np.argmax([contrast_scores[i] for i in remaining_indices])]
    else:
        gfp_idx = None
    
    return {
        'bright_field': bf_idx,
        'gfp': gfp_idx,
        'confidence': 'high' if len(channel_stats) == 2 else 'medium'
    }
```

**识别特征：**
- **Bright-field**：
  - 更高的标准差（更多纹理和对比度）
  - 通常有更宽的动态范围
  - 可能显示细胞结构、边界等细节
  
- **GFP 荧光**：
  - 较小的标准差（更均匀的强度分布）
  - 通常显示特定的荧光信号区域
  - 背景通常较暗，信号区域较亮

### 4. 归一化和预处理

**关键操作：**
```python
def normalize_channel(channel, method='minmax'):
    """
    归一化通道到 [0, 1] 范围
    
    Args:
        channel: 输入通道（numpy array）
        method: 归一化方法
            - 'minmax': 线性归一化到 [0, 1]
            - 'percentile': 使用百分位数裁剪后归一化
            - 'zscore': Z-score 归一化
    """
    if method == 'minmax':
        # 线性归一化
        channel_min = channel.min()
        channel_max = channel.max()
        if channel_max > channel_min:
            normalized = (channel - channel_min) / (channel_max - channel_min)
        else:
            normalized = channel.astype(np.float32)
    
    elif method == 'percentile':
        # 使用 1st 和 99th 百分位数裁剪异常值
        p1, p99 = np.percentile(channel, [1, 99])
        normalized = np.clip(channel, p1, p99)
        normalized = (normalized - p1) / (p99 - p1 + 1e-12)
    
    elif method == 'zscore':
        # Z-score 归一化
        mean = channel.mean()
        std = channel.std()
        normalized = (channel - mean) / (std + 1e-12)
        # 转换到 [0, 1] 范围
        normalized = (normalized - normalized.min()) / (normalized.max() - normalized.min() + 1e-12)
    
    return normalized.astype(np.float32)
```

**归一化方法选择：**
- **minmax**：适用于动态范围已知的图像
- **percentile**：适用于有异常值或极端值的图像
- **zscore**：适用于需要标准化分布的统计分析

### 5. 创建可视化

**多通道可视化模板：**

```python
import matplotlib.pyplot as plt
from matplotlib.gridspec import GridSpec

def create_multi_channel_visualization(channels, channel_names, normalized_channels=None):
    """
    创建多通道可视化
    
    Args:
        channels: 原始通道列表
        channel_names: 通道名称列表
        normalized_channels: 归一化后的通道列表（可选）
    """
    num_channels = len(channels)
    
    # 创建图形布局
    fig = plt.figure(figsize=(6 * num_channels, 6))
    gs = GridSpec(2, num_channels, figure=fig, hspace=0.3, wspace=0.3)
    
    for ch_idx, (channel, name) in enumerate(zip(channels, channel_names)):
        # 原始通道
        ax_orig = fig.add_subplot(gs[0, ch_idx])
        ax_orig.imshow(channel, cmap='gray')
        ax_orig.set_title(f'{name} (Original)', fontsize=12, fontweight='bold')
        ax_orig.axis('off')
        
        # 归一化通道（如果提供）
        if normalized_channels:
            ax_norm = fig.add_subplot(gs[1, ch_idx])
            ax_norm.imshow(normalized_channels[ch_idx], cmap='gray')
            ax_norm.set_title(f'{name} (Normalized)', fontsize=12, fontweight='bold')
            ax_norm.axis('off')
    
    plt.suptitle('Multi-Channel Image Analysis', fontsize=16, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    return fig
```

### 6. 保存结果

**文件命名规范：**
```python
def save_channel_outputs(channels, channel_names, base_name, output_dir):
    """
    保存每个通道为单独的图像文件
    
    文件命名格式：
    - {base_name}_bright-field.png
    - {base_name}_gfp.png
    - {base_name}_channel_3.png
    """
    saved_paths = []
    
    for channel, name in zip(channels, channel_names):
        # 清理通道名称用于文件名
        safe_name = name.replace(' ', '_').replace('-', '_').lower()
        filename = f"{base_name}_{safe_name}.png"
        filepath = os.path.join(output_dir, filename)
        
        # 转换为 uint8 并保存
        if channel.dtype != np.uint8:
            if channel.dtype == np.uint16:
                channel_uint8 = (channel / 65535.0 * 255).astype(np.uint8)
            else:
                channel_uint8 = np.clip(channel, 0, 255).astype(np.uint8)
        else:
            channel_uint8 = channel
        
        Image.fromarray(channel_uint8, mode='L').save(filepath)
        saved_paths.append(filepath)
        print(f"Saved {name} channel to: {filepath}")
    
    return saved_paths
```

## 完整工作流程示例

```python
import tifffile
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt
import os

def process_multi_channel_tiff(image_path, output_dir):
    """
    完整的多通道 TIFF 处理流程
    """
    # 1. 加载和检查
    img_full = tifffile.imread(image_path)
    print(f"Image shape: {img_full.shape}, dtype: {img_full.dtype}")
    
    # 2. 分离通道
    if img_full.shape[0] <= 4:  # (C, H, W)
        num_channels = img_full.shape[0]
        channels = [img_full[i, :, :] for i in range(num_channels)]
    else:  # (H, W, C)
        num_channels = img_full.shape[2]
        channels = [img_full[:, :, i] for i in range(num_channels)]
    
    # 3. 分析通道
    channel_stats = []
    for ch_idx, channel in enumerate(channels):
        stats = {
            'index': ch_idx,
            'mean': channel.mean(),
            'std': channel.std(),
            'min': channel.min(),
            'max': channel.max()
        }
        channel_stats.append(stats)
        print(f"Channel {ch_idx}: Mean={stats['mean']:.2f}, Std={stats['std']:.2f}")
    
    # 4. 识别通道类型
    contrast_scores = [s['std'] for s in channel_stats]
    bf_idx = np.argmax(contrast_scores)
    channel_names = []
    for i in range(num_channels):
        if i == bf_idx:
            channel_names.append("bright-field")
        elif i == 1 - bf_idx and num_channels >= 2:
            channel_names.append("GFP")
        else:
            channel_names.append(f"Channel_{i+1}")
    
    # 5. 归一化
    normalized_channels = [normalize_channel(ch) for ch in channels]
    
    # 6. 可视化
    fig = create_multi_channel_visualization(channels, channel_names, normalized_channels)
    vis_path = os.path.join(output_dir, "multi_channel_visualization.png")
    fig.savefig(vis_path, dpi=150, bbox_inches='tight')
    plt.close(fig)
    
    # 7. 保存通道
    base_name = os.path.splitext(os.path.basename(image_path))[0]
    saved_paths = save_channel_outputs(normalized_channels, channel_names, base_name, output_dir)
    
    return {
        'channels': channels,
        'normalized_channels': normalized_channels,
        'channel_names': channel_names,
        'channel_stats': channel_stats,
        'visualization_path': vis_path,
        'saved_paths': saved_paths
    }
```

## 最佳实践

1. **总是检查图像维度结构**：不同的 TIFF 格式可能使用不同的维度顺序
2. **分析统计特征**：使用均值、标准差等指标帮助识别通道类型
3. **归一化处理**：根据应用场景选择合适的归一化方法
4. **保存中间结果**：保存原始和归一化后的通道，便于后续分析
5. **清晰的命名**：使用描述性的文件名，包含通道类型信息
6. **错误处理**：处理可能的异常情况（单通道、异常维度等）

## 与现有工具的集成

现有的 `Image_Preprocessor_Tool` 已经实现了部分功能：
- ✅ 多通道检测和提取
- ✅ 通道分离和保存
- ✅ 预处理（光照校正、亮度调整）

可以增强的功能：
- 🔄 智能通道识别（基于统计特征）
- 🔄 更灵活的归一化选项
- 🔄 更丰富的可视化选项

## 参考

- TIFF 格式规范：https://www.loc.gov/preservation/digital/formats/fdd/fdd000022.shtml
- NumPy 数组操作：https://numpy.org/doc/stable/reference/arrays.html
- Matplotlib 可视化：https://matplotlib.org/stable/tutorials/index.html


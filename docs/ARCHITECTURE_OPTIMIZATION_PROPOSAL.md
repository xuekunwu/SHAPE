# 全局架构优化方案 (Global Architecture Optimization Proposal)

## 一、当前问题分析 (Current Issues)

### 1.1 多通道图像处理的重复逻辑
- **问题**：每个工具（`Image_Preprocessor_Tool`, `Cell_Segmenter_Tool`, `Nuclei_Segmenter_Tool`, `Organoid_Segmenter_Tool`, `Single_Cell_Cropper_Tool`, `Analysis_Visualizer_Tool`）都有各自的 multi-channel 检测和处理逻辑
- **重复代码**：
  - 检测图像维度：`img_full.ndim == 3` 检查
  - 判断通道位置：`shape[2] <= 4` vs `shape[0] <= 4`
  - 提取通道：`img_full[:, :, c]` vs `img_full[c, :, :]`
  - 归一化处理：重复的 dtype 转换逻辑

### 1.2 缺少统一的图像数据抽象
- **问题**：没有统一的图像数据结构来表示单通道/多通道图像
- **影响**：工具间传递图像数据时，需要重复检测和处理

### 1.3 Planning 策略可以优化
- **当前**：基于工具优先级和依赖关系的 step-by-step planning
- **可以改进**：参考 Biomni 的模块化设计，引入更高层次的 task decomposition

## 二、优化方案 (Optimization Proposal)

### 2.1 创建统一的图像处理抽象层 (Unified Image Processing Abstraction)

#### 2.1.1 创建 `ImageData` 类
```python
# octotools/models/image_data.py
from dataclasses import dataclass
from typing import Optional, List, Tuple
import numpy as np
import tifffile

@dataclass
class ImageData:
    """
    统一的图像数据表示：单通道是多通道的特殊形式（1 channel）
    
    核心原则：
    - 所有图像都表示为 (H, W, C) 格式，C >= 1
    - 单通道图像：C=1
    - 多通道图像：C>1
    """
    data: np.ndarray  # 始终为 (H, W, C) 格式
    dtype: np.dtype
    channel_names: Optional[List[str]] = None  # ['bright-field', 'GFP', 'DAPI']
    source_path: Optional[str] = None
    
    @classmethod
    def from_path(cls, path: str, channel_names: Optional[List[str]] = None) -> 'ImageData':
        """
        从文件路径加载图像，自动检测和处理多通道
        
        统一逻辑：
        1. 加载图像（TIFF/PIL）
        2. 标准化为 (H, W, C) 格式
        3. 检测通道数量
        4. 推断通道名称（如果未提供）
        """
        # 实现统一的加载和标准化逻辑
    
    def get_channel(self, idx: int) -> np.ndarray:
        """获取单个通道（返回 2D 数组）"""
        return self.data[:, :, idx]
    
    def get_channels(self, indices: Optional[List[int]] = None) -> List[np.ndarray]:
        """获取多个通道"""
        if indices is None:
            indices = list(range(self.num_channels))
        return [self.get_channel(i) for i in indices]
    
    @property
    def num_channels(self) -> int:
        return self.data.shape[2]
    
    @property
    def is_multi_channel(self) -> bool:
        """是否为多通道（>1 channel）"""
        return self.num_channels > 1
    
    @property
    def is_single_channel(self) -> bool:
        """是否为单通道（1 channel）"""
        return self.num_channels == 1
    
    def to_segmentation_input(self, channel_idx: int = 0) -> np.ndarray:
        """
        转换为分割工具需要的格式（单通道，float32）
        默认使用第一个通道
        """
        channel = self.get_channel(channel_idx)
        if channel.dtype != np.float32:
            channel = channel.astype(np.float32)
        return channel
    
    def create_merged_rgb(self, channel_mapping: Optional[Dict[int, str]] = None) -> np.ndarray:
        """
        创建合并的 RGB 视图用于可视化
        
        统一逻辑：
        - 单通道：grayscale -> RGB
        - 多通道：按 channel_mapping 合并
        """
        # 实现统一的合并逻辑
```

#### 2.1.2 创建 `ImageProcessor` 工具类
```python
# octotools/utils/image_processor.py
class ImageProcessor:
    """
    统一的图像处理工具类
    所有工具都应该使用这个类来处理图像
    """
    
    @staticmethod
    def load_image(path: str) -> ImageData:
        """统一的图像加载接口"""
        return ImageData.from_path(path)
    
    @staticmethod
    def normalize_for_display(img_data: ImageData, channel_idx: int = 0) -> np.ndarray:
        """标准化单个通道用于显示（uint8）"""
        # 统一的正规化逻辑
    
    @staticmethod
    def create_multi_channel_visualization(
        img_data: ImageData, 
        output_path: str,
        vis_config: VisualizationConfig
    ) -> str:
        """创建多通道可视化（统一逻辑）"""
        # 统一的可视化创建逻辑
```

### 2.2 重构工具以使用统一抽象 (Refactor Tools to Use Unified Abstraction)

#### 2.2.1 修改工具基类
```python
# octotools/tools/base.py
from octotools.utils.image_processor import ImageProcessor
from octotools.models.image_data import ImageData

class BaseTool:
    # ... 现有代码 ...
    
    def _load_image_unified(self, image_path: str) -> ImageData:
        """
        统一的图像加载方法
        所有工具都应该使用这个方法而不是各自实现
        """
        return ImageProcessor.load_image(image_path)
```

#### 2.2.2 重构各个工具
- **Image_Preprocessor_Tool**: 使用 `ImageData` 和 `ImageProcessor`
- **Cell_Segmenter_Tool**: 使用 `img_data.to_segmentation_input()`
- **Nuclei_Segmenter_Tool**: 使用 `img_data.to_segmentation_input()`
- **Organoid_Segmenter_Tool**: 使用 `img_data.to_segmentation_input()`
- **Single_Cell_Cropper_Tool**: 使用 `ImageData` 保存多通道 crops
- **Analysis_Visualizer_Tool**: 使用 `img_data.create_merged_rgb()`

### 2.3 优化 Planning 策略 (Enhanced Planning Strategy)

#### 2.3.1 引入 Task Decomposition
参考 Biomni 的策略，在 Planner 中引入更高层次的任务分解：

```python
# octotools/models/planner.py
class Planner:
    def decompose_task(self, query: str, image_context: ImageContext) -> List[Task]:
        """
        将查询分解为子任务（参考 Biomni 的模块化策略）
        
        例如：
        "What changes of organoid among different groups?"
        -> Task 1: Segment all organoids in all images
        -> Task 2: Extract features from all organoids
        -> Task 3: Compare groups statistically
        -> Task 4: Visualize results
        """
        # 使用 LLM 进行任务分解
        pass
```

#### 2.3.2 引入任务优先级和并行化
- **独立任务**：可以并行执行（如多图像分割）
- **依赖任务**：按依赖顺序执行
- **合并任务**：在合适的时机合并结果

### 2.4 统一多图处理逻辑 (Unified Multi-Image Processing)

#### 2.4.1 增强 BatchImage 数据结构
```python
# octotools/models/task_state.py
@dataclass
class BatchImage:
    group: str
    image_id: str
    image_path: str
    image_name: str
    image_data: Optional[ImageData] = None  # 添加统一的图像数据
    metadata: Dict[str, Any] = field(default_factory=dict)
```

#### 2.4.2 统一批处理接口
所有工具都应该支持批处理：
```python
def execute(self, images: Union[str, List[str]], ...) -> Union[Dict, Dict[str, Dict]]:
    """
    统一的批处理接口
    - 单个图像：返回单个结果
    - 多个图像：返回 per_image 结构
    """
    if isinstance(images, str):
        images = [images]
    
    results = []
    for img in images:
        img_data = self._load_image_unified(img)
        result = self._process_single_image(img_data, ...)
        results.append(result)
    
    return self._format_results(results)
```

## 三、实施计划 (Implementation Plan)

### Phase 1: 创建统一抽象层
1. ✅ 创建 `ImageData` 类 (`octotools/models/image_data.py`)
2. ✅ 创建 `ImageProcessor` 工具类 (`octotools/utils/image_processor.py`)
3. ✅ 编写单元测试

### Phase 2: 重构核心工具（向后兼容）
1. ✅ 重构 `Image_Preprocessor_Tool`
2. ✅ 重构 `Cell_Segmenter_Tool`, `Nuclei_Segmenter_Tool`, `Organoid_Segmenter_Tool`
3. ✅ 重构 `Single_Cell_Cropper_Tool`
4. ✅ 重构 `Analysis_Visualizer_Tool`

### Phase 3: 优化 Planning 策略
1. ✅ 增强图像上下文感知（使用 ImageProcessor 获取元数据）
2. ✅ 添加规则化决策系统（减少 LLM 调用）
3. ✅ 优化规划效率（规则匹配跳过 LLM 调用）

### Phase 4: 清理和文档
1. ✅ 移除重复代码（~400+ 行）
2. ✅ 更新文档（架构重构总结、Phase 3 规划优化指南）
3. ✅ 全面测试（Phase 1: 10/10, Phase 2: 15/15 通过）

## 四、关键原则 (Key Principles)

1. **统一性**：单通道是多通道的特殊形式（C=1），使用统一的处理逻辑
2. **向后兼容**：所有修改都应该保持向后兼容，不破坏现有功能
3. **简单性**：避免过度设计，保持代码简洁
4. **全局优化**：不是局部修补，而是系统性的重构
5. **非破坏性**：不拆分 `app.py`，不进行破坏性修改

## 五、预期收益 (Expected Benefits)

1. **代码简化**：减少重复代码 50%+
2. **可维护性**：统一的抽象层便于维护和扩展
3. **正确性**：统一逻辑减少 bug
4. **性能**：避免重复的图像加载和处理
5. **可扩展性**：更容易添加新的通道处理逻辑

## 六、风险评估 (Risk Assessment)

1. **向后兼容风险**：需要确保所有现有工具仍然正常工作
   - **缓解**：渐进式重构，保持旧接口可用
2. **性能风险**：新的抽象层可能引入性能开销
   - **缓解**：优化关键路径，必要时缓存
3. **测试风险**：需要全面测试所有工具
   - **缓解**：分阶段实施，每阶段充分测试


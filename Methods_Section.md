# Methods Section: SHAPE Framework

## Model Architecture

SHAPE (Single-cell Hierarchical Analysis Platform for Exploration) is an augmented agentic framework designed for resolving morphological cell states across biological scales. The framework integrates three core technical innovations: multi-turn conversational planning, multimodal output generation, and GPU-accelerated deep learning tool modules.

### Image Analysis Pipeline and Workflow Logic

SHAPE implements a hierarchical image analysis pipeline that adaptively selects tools based on query complexity. The pipeline supports three analysis levels, each with distinct tool chains and parameter configurations:

**Level 1: Simple Counting Queries** (minimal analysis, 1-2 steps)
- **Query patterns**: "how many cells", "count organoids", "number of cells"
- **Tool chain**: `Image_Preprocessor_Tool` (optional) → `[Cell_Segmenter_Tool | Nuclei_Segmenter_Tool | Organoid_Segmenter_Tool]` → STOP
- **Output**: Cell count extracted from segmentation masks (count = `len(np.unique(mask)) - 1`, excluding background label 0)
- **Termination condition**: Segmentation tool execution completes; count is available in result dictionary (`cell_count` key)

**Level 2: Basic Morphology Analysis** (intermediate analysis, 2-3 steps)
- **Query patterns**: "cell area", "organoid size distribution", "morphological measurements"
- **Tool chain**: `Image_Preprocessor_Tool` (optional) → `Segmenter_Tool` → `Analysis_Visualizer_Tool`
- **Output**: Statistical distributions and visualizations of morphological features (area, perimeter, circularity) extracted from segmentation masks
- **Note**: May skip `Single_Cell_Cropper_Tool` if individual cell analysis is not required

**Level 3: Full Cell State Analysis** (complete pipeline, 4-5 steps)
- **Query patterns**: "what cell states", "analyze cell states", "compare groups", "clustering", "UMAP", "cell types"
- **Tool chain**: `Image_Preprocessor_Tool` (optional) → `Segmenter_Tool` → `Single_Cell_Cropper_Tool` (MANDATORY) → `Cell_State_Analyzer_[Single|Multi]_Tool` (MANDATORY) → `Analysis_Visualizer_Tool` (MANDATORY)
- **Output**: Complete analysis with self-supervised feature extraction, UMAP embeddings, Leiden clustering, and multi-group statistical comparisons
- **Critical dependencies**: 
  - `Single_Cell_Cropper_Tool` requires segmentation mask from `Segmenter_Tool` (dependency enforced in `octotools/models/tool_priority.py`, lines 75-77)
  - `Cell_State_Analyzer_*_Tool` requires individual crop images from `Single_Cell_Cropper_Tool` (automatically loads from metadata files in `query_cache_dir/tool_cache/`)
  - `Analysis_Visualizer_Tool` requires AnnData object from `Cell_State_Analyzer_*_Tool`

**Query Type Detection**: The planner (`octotools/models/planner.py`) classifies queries using keyword matching in `_requires_full_cell_state_analysis()` method. Keywords triggering Level 3 analysis include: 'cell state', 'cluster', 'umap', 'compare', 'group', 'treatment'. Keywords indicating Level 1 include: 'how many', 'count', 'number of'. The planner uses LLM-based reasoning for ambiguous queries, with rule-based shortcuts for clear cases (lines 569-632).

**Tool Selection Logic**: At each step, the planner (`generate_next_step()`, lines 240-377) receives: (1) conversation context (formatted history), (2) current memory state (results from previous tools), (3) query analysis (structured breakdown of requirements), and (4) available tools with priority rankings. The planner generates a `NextStep` object containing tool name, sub-goal, and context. Tool dependencies are checked via `TOOL_DEPENDENCIES` dictionary, and the system prevents skipping mandatory steps (e.g., cannot call `Cell_State_Analyzer_Tool` without `Single_Cell_Cropper_Tool` output).

### Multi-turn Conversational Planning

SHAPE supports iterative multi-turn dialogue to handle complex image analysis queries that require sequential reasoning and adaptive tool selection. The conversational planning mechanism is implemented through a stateful `Solver` class that maintains conversation context across multiple analysis steps.

**Conversation History Management**: The system maintains a persistent conversation history stored in `agent_state.conversation` (list of `ChatMessage` objects). At each planning step, the conversation history is formatted into a plain-text transcript using the `_format_conversation_history()` method, which serializes all previous user queries and system responses into a structured text format:

```python
def _format_conversation_history(self) -> str:
    history = self.agent_state.conversation or []
    lines = []
    for msg in history:
        role = getattr(msg, "role", "assistant")
        content = getattr(msg, "content", "")
        lines.append(f"{role}: {content}")
    return "\n".join(lines)
```

**Context Propagation**: The formatted conversation context (`conversation_text`) is propagated to all planning stages: (1) initial query analysis via `planner.analyze_query()`, which determines required tools and skills; (2) step-by-step tool selection via `planner.generate_next_step()`, which selects the next tool based on current state and conversation history; (3) completion verification via `planner.verificate_memory()`; and (4) final output generation via `planner.generate_final_output()`. This ensures that each planning decision incorporates the full dialogue context, enabling the system to handle follow-up questions, refine previous analyses, and adapt to user feedback.

**Iterative Execution Loop**: The execution loop implements a maximum of `max_steps` (default: 10) iterations. At each step `i`, the planner receives the conversation context (formatted via `_format_conversation_history()`), current memory state (containing results from previous tool executions stored in `self.memory`), and the original user query. The planner generates a `NextStep` object via `planner.generate_next_step()` containing the selected tool name, sub-goal, and context. After tool execution via `executor.execute_tool()`, results are stored in memory using `memory.add_action()` and the conversation is updated with the tool's output through `messages.append(ChatMessage(...))`. This iterative process continues until either the verification step (`planner.verificate_memory()`) signals completion (`stop_signal='STOP'`) or the maximum step count is reached. The loop also includes loop detection mechanisms to prevent infinite execution when the same tool is selected consecutively three or more times.

### Multimodal Output Generation

SHAPE generates comprehensive multimodal outputs that include processed images, cell/organoid crops, segmentation overlays, and structured analyzed data (AnnData objects), enhancing analysis transparency and enabling downstream computational analysis.

**Visual Output Collection**: The `_collect_visual_outputs()` function (lines 949-1230) implements a unified collection mechanism that aggregates visual outputs from tool execution results. Tools return structured dictionaries containing multiple output types:

- **Segmentation overlays**: Tools such as `Cell_Segmenter_Tool`, `Nuclei_Segmenter_Tool`, and `Organoid_Segmenter_Tool` generate overlay visualizations by combining the original image with segmentation masks using `plot.mask_overlay()` (e.g., `cell_segmenter/tool.py`, line 388). Overlays are saved as PNG files with professional styling via `VisualizationConfig.create_professional_figure()` and `VisualizationConfig.save_professional_figure()`.

- **Individual cell/organoid crops**: The `Single_Cell_Cropper_Tool` extracts individual cell regions based on segmentation masks and saves each crop as a multi-channel TIFF file (format: `(C, H, W)`). Crops are organized into ZIP archives (`crops_zip_path`) for batch download, with metadata stored in JSON files (`cell_crops_metadata_*.json`) containing crop paths, cell IDs, and group labels.

- **Segmentation masks**: Masks are saved in multiple formats: (1) 16-bit TIFF files (`.tif`) for visual display, preserving all label values up to 65,535 cells; (2) NumPy arrays (`.npy`) for downstream computational analysis; and (3) PNG visualizations with colormap rendering for quick inspection.

**Structured Data Output (AnnData)**: The `Cell_State_Analyzer_Single_Tool` and `Cell_State_Analyzer_Multi_Tool` generate structured analyzed data using the AnnData format (compatible with Scanpy ecosystem). The analysis pipeline (e.g., `cell_state_analyzer_multi/tool.py`, lines 412-573) performs the following steps:

1. **Feature extraction**: DINOv3 Vision Transformer (ViT-B/16) extracts morphological features from cell crops. For multi-channel images, the patch embedding layer is adapted to accept arbitrary channel counts (lines 206-229) by initializing new convolutional weights while inheriting RGB weights for the first three channels.

2. **Self-supervised learning**: When the number of crops ≥ 50, contrastive learning is performed using a temperature-scaled cross-entropy loss (line 237-242). The model is trained for up to 25 epochs with early stopping (patience=5) and mixed-precision training via `torch.cuda.amp.autocast()`.

3. **Dimensionality reduction and clustering**: Extracted features are stored in an AnnData object (`ad.AnnData(X=feats)`, line 498). UMAP embedding is computed using `sc.tl.umap()` with cosine distance metric (line 505), and Leiden clustering is performed with configurable resolution (default: 0.5, line 507).

4. **Data persistence**: The AnnData object is saved as an HDF5 file (`.h5ad` format, line 510) using `adata.write()`, containing:
   - `X`: Feature matrix (n_cells × feature_dim)
   - `obs`: Cell metadata including `group`, `image_name`, and cluster assignments (`leiden_{resolution}`)
   - `obsm`: UMAP coordinates (`X_umap`)
   - `uns`: Analysis parameters and metadata

The AnnData file enables downstream analysis in Python/R using Scanpy, Seurat, or custom scripts, facilitating integration with existing single-cell analysis workflows.

**Output Aggregation**: Visual outputs are collected and processed by the `_collect_visual_outputs()` function, which extracts visual files from multiple dictionary keys: `deliverables` (preferred), `visual_outputs`, `overlay_path`, `output_path`, and `mask_path`. Structured data files (`.h5ad` AnnData files, `.zip` archives containing crops) are added to `downloadable_files` for batch download. For multi-image analyses, the system automatically creates unified ZIP archives via `_create_unified_crops_zip()` and `_create_unified_masks_zip()`, organizing crops and masks by group and image ID in a hierarchical directory structure within the ZIP file.

### GPU-Compatible Deep Learning Tool Modules

SHAPE integrates state-of-the-art deep learning models through GPU-compatible tool modules. The tool modules act as lightweight wrappers that load, configure, and execute pre-trained models without modifying their core architectures, enabling seamless integration of the latest advances in computer vision (e.g., Vision Transformers, Cellpose-SAM) into the agentic framework.

**GPU Detection and Fallback**: All tool modules implement a consistent GPU detection pattern using `torch.cuda.is_available()` (e.g., `cell_segmenter/tool.py`, line 150). If GPU initialization fails (e.g., due to NVML issues or CUDA version mismatches), tools automatically fall back to CPU execution with error logging (lines 166-185). After GPU operations, CUDA cache is cleared using `torch.cuda.empty_cache()` to prevent memory leaks (lines 451-453).

**Cellpose-SAM Integration**: All segmentation tools integrate the Cellpose model architecture via the `cellpose` Python package, but differ in model selection and initialization strategies:

**Cell_Segmenter_Tool** (`cell_segmenter/tool.py`): Initializes with dual-model support. Primary model is a custom CPSAM (Cellpose-SAM) model downloaded from Hugging Face Hub (`5xuekun/cell-segmenter-cpsam-model`, filename: `cpsam`). If CPSAM download fails, automatically falls back to Cellpose's pre-trained `'cyto'` model, which is optimized for whole cell segmentation in phase-contrast images. Model initialization (lines 146-185):

```python
if model_path:
    self.model = models.CellposeModel(
        gpu=torch.cuda.is_available(), 
        pretrained_model=model_path  # CPSAM model
    )
    self.model_type = "cpsam"
else:
    self.model = models.CellposeModel(
        gpu=torch.cuda.is_available(),
        model_type='cyto'  # Fallback for phase-contrast whole cells
    )
    self.model_type = "cyto"
```

**Organoid_Segmenter_Tool** (`organoid_segmenter/tool.py`): Uses lazy loading strategy to avoid CP3/CP4 compatibility issues. Model is loaded on-demand during first execution via `_load_model()` method (lines 229-318), rather than at initialization. REQUIRES a specialized organoid model (`5xuekun/organoid-segmenter-model`, filename: `cpsam_CO_4x_260105`). Standard Cellpose models (`cyto`, `cyto2`) are explicitly not suitable for organoids. Model loading includes special error handling for version compatibility:

```python
self.model = models.CellposeModel(
    gpu=torch.cuda.is_available(), 
    pretrained_model=model_to_load  # Custom organoid model
)
```

**Nuclei_Segmenter_Tool** (`nuclei_segmenter/tool.py`): Uses custom CPSAM model downloaded from Hugging Face Hub (`5xuekun/nuclei-segmenter-model`, filename: `cpsam_lr_1e-04`, lines 126-130). Model is loaded at initialization with GPU support.

All segmentation tools perform segmentation via `model.eval()` with configurable parameters (`diameter`, `flow_threshold`, `cellprob_threshold`). The tools use grayscale channel configuration `[0, 0]` for phase-contrast images, where the first channel (bright-field/phase contrast) is used for both channel inputs required by Cellpose.

**Vision Transformer (DINOv3) Integration**: The cell state analysis tools use DINOv3 (ViT-B/16) as the backbone for morphological feature extraction. The model architecture is loaded from the official GitHub repository (`facebookresearch/dinov3`) via `torch.hub.load()` (e.g., `cell_state_analyzer_multi/tool.py`, line 167), while pre-trained weights are downloaded from Hugging Face Hub (`5xuekun/dinov3_vits16`, line 158-162). The ViT backbone processes cell crops through:

1. **Patch embedding**: Input images are divided into 16×16 patches and embedded into 768-dimensional vectors (ViT-B/16 configuration).

2. **Transformer encoding**: 12 transformer blocks process the patch embeddings, with the [CLS] token representation used as the global feature vector.

3. **Projection head**: A two-layer MLP (768 → 768 → 256) projects features to a lower-dimensional space for contrastive learning (lines 200-204).

The model is moved to GPU via `.to(self.device)` (line 463), and training/inference uses mixed-precision arithmetic (`torch.cuda.amp.autocast()`, line 314) for efficiency.

**Model Loading Strategy**: To optimize memory usage, models are loaded on-demand when tools are first instantiated, rather than pre-loading all models at startup. Model weights are downloaded from Hugging Face Hub using `hf_hub_download()` and cached in the Hugging Face cache directory (`~/.cache/huggingface/hub/`) to avoid redundant downloads across sessions. This caching mechanism, combined with the on-demand loading strategy, enables efficient resource utilization. For multi-channel analysis, the patch embedding layer is dynamically adapted to match the input channel count (2, 3, 4, or more channels) by creating a new `nn.Conv2d` layer and initializing it with inherited weights from the RGB channels (lines 211-228), allowing the same ViT architecture to process images with arbitrary channel configurations without retraining.

**Tool Module Architecture**: Each tool module inherits from a `BaseTool` class (defined in `octotools/tools/base.py`) that provides standardized interfaces for tool registration, metadata management, and execution tracking. Tools expose their capabilities through structured metadata dictionaries passed to `super().__init__()` containing: `input_types` (parameter descriptions), `output_type` (return value description), `demo_commands` (usage examples), and `user_metadata` (limitations and best practices). This metadata is used by the planner (`octotools/models/planner.py`) for intelligent tool selection during query analysis and step generation. The modular design allows new deep learning models to be integrated by creating new tool classes that: (1) inherit from `BaseTool`, (2) initialize the model in `__init__()` with GPU detection, (3) implement an `execute()` method that processes inputs and returns structured dictionaries, and (4) register themselves in the tool discovery system, which dynamically scans the `octotools/tools/` directory. This architecture enables SHAPE to incorporate new state-of-the-art models (e.g., Segment Anything Model, newer ViT variants) without modifying the core agentic framework, ensuring the system remains current with rapid advances in computer vision.

### Tool Technical Parameters and Configurations

Each tool in SHAPE exposes configurable parameters that control analysis behavior. Default values are optimized for typical microscopy images, but can be adjusted based on image characteristics and analysis requirements.

**Image Preprocessing Tool** (`Image_Preprocessor_Tool`):
- **`target_brightness`** (int, default: 120, range: 0-255): Target mean brightness level for normalization. Applied via `cv2.convertScaleAbs()` with adjustment factor `target_brightness / current_brightness` (`image_preprocessor/tool.py`, lines 91-108).
- **`gaussian_kernel_size`** (int, default: 151, must be odd): Size of Gaussian kernel for illumination correction. Applied via `cv2.GaussianBlur()` to estimate illumination pattern, then corrected by division (`image_preprocessor/tool.py`, lines 53-77). Kernel size is automatically adjusted to odd if even (line 65-66).
- **`skip_illumination_correction`** (bool, default: False): If True, skips Gaussian-based illumination correction and only adjusts brightness. Recommended for organoid images where illumination correction may introduce artifacts.
- **Output format**: Always PNG for Visual Outputs display compatibility, regardless of input format.

**Segmentation Tools** (`Cell_Segmenter_Tool`, `Nuclei_Segmenter_Tool`, `Organoid_Segmenter_Tool`):

All segmentation tools use the Cellpose architecture via the `cellpose` Python package, but differ in model selection, default parameters, and target applications:

**Cell_Segmenter_Tool** (for whole cell segmentation in phase-contrast images):
- **Purpose**: Segments whole cells (not just nuclei) in phase-contrast microscopy images. Designed for general cell types where the entire cell boundary is visible.
- **Model selection** (`cell_segmenter/tool.py`, lines 94-185):
  - **Primary model**: Custom CPSAM (Cellpose-SAM) model downloaded from Hugging Face Hub (`repo_id="5xuekun/cell-segmenter-cpsam-model"`, `filename="cpsam"`). This is the default when `model_type="cpsam"` (line 187).
  - **Fallback model**: If CPSAM download fails, automatically falls back to Cellpose `'cyto'` model (`model_type='cyto'`), which is pre-trained for whole cell segmentation in phase-contrast images (lines 155-160).
  - **Model switching**: The tool supports dynamic model switching during execution. If `model_type` parameter differs from the initialized model, it reloads the CPSAM model on-demand (lines 204-220).
- **Default parameters**: `diameter=30` (typical for cells), `flow_threshold=0.4`, `cellprob_threshold=0` (line 187).
- **Image processing**: Extracts first channel from multi-channel images for segmentation (uses `ImageProcessor.load_image()` and `to_segmentation_input(channel_idx=0)`, line 249). For phase-contrast images, uses grayscale channels `[0, 0]` in Cellpose evaluation (line 382).
- **Use case**: Phase-contrast microscopy images where whole cell boundaries are visible, such as live cell imaging or adherent cell cultures.

**Organoid_Segmenter_Tool** (for organoid segmentation):
- **Purpose**: Segments organoids in microscopy images. Organoids are 3D structures that appear as larger, more complex objects compared to individual cells.
- **Model requirements** (`organoid_segmenter/tool.py`, lines 165-325):
  - **Mandatory custom model**: REQUIRES a specialized organoid segmentation model. Standard Cellpose models (`cyto`, `cyto2`) are explicitly NOT suitable for organoids and will not work (documented in tool metadata, line 194).
  - **Model source**: Custom organoid model downloaded from Hugging Face Hub (`repo_id="5xuekun/organoid-segmenter-model"`, `filename="cpsam_CO_4x_260105"`, lines 219-224). The model filename indicates it was trained on 4× magnification organoid images.
  - **Lazy loading strategy**: Model is loaded on-demand during first execution (`_load_model()` method, lines 229-318) rather than at initialization, to avoid CP3/CP4 compatibility issues. This allows the tool to initialize even if model download fails initially.
  - **CP3/CP4 compatibility handling**: Includes special error handling for Cellpose version compatibility issues. If GPU loading fails due to CP3/CP4 incompatibility, automatically retries on CPU (lines 272-300). Provides detailed error messages guiding users to convert models or use compatible Cellpose versions.
- **Default parameters**: `diameter=100` (optimized for organoid sizes, which are typically 50-200 pixels), `flow_threshold=0.4`, `cellprob_threshold=0` (line 325).
- **Image processing**: 
  - Uses brightness adjustment only (no illumination correction) via `check_brightness_only()` and `adjust_brightness_only()` functions (lines 433-443). This is because organoid images may have artifacts from illumination correction.
  - Extracts first channel (phase contrast) from multi-channel images for segmentation (lines 388-427).
  - Uses grayscale channels `[0, 0]` in Cellpose evaluation (line 490).
- **Use case**: Organoid cultures in microscopy images, typically at lower magnifications (4×, 10×) where entire organoid structures are visible.

**Common segmentation parameters** (applied to all segmentation tools):
- **`diameter`** (float, default: None for auto-detect, typical ranges: 25-30 for nuclei, 30-50 for cells, 50-200 for organoids): Expected object diameter in pixels. When `None`, Cellpose automatically estimates diameter from image statistics using image size and intensity distribution. Auto-detection logic handles string inputs ('auto') and type conversions (`cell_segmenter/tool.py`, lines 363-375; `organoid_segmenter/tool.py`, lines 466-478).
- **`flow_threshold`** (float, default: 0.4 for cells/organoids, 0.6 for nuclei): Flow field threshold for object boundary detection in Cellpose's flow-based segmentation algorithm. Higher values (0.5-0.8) reduce false positives but may miss weak boundaries. Lower values (0.2-0.4) detect more objects but may include noise. Passed to `model.eval()` as a parameter (`cell_segmenter/tool.py`, line 381; `organoid_segmenter/tool.py`, line 497).
- **`cellprob_threshold`** (float, default: 0): Cell probability threshold. Negative values (-6 to 0) are more permissive (detect more objects), positive values (0 to 6) are more restrictive (detect fewer, higher-confidence objects). Default 0 balances sensitivity and specificity.

**Output format**: All segmentation tools generate:
- Segmentation masks in multiple formats: 16-bit TIFF (`.tif`) preserving label values up to 65,535 objects, NumPy arrays (`.npy`) for computational analysis, and PNG visualizations with colormap rendering.
- Overlay visualizations combining original images with segmentation masks using `plot.mask_overlay()`.
- Cell/organoid counts extracted from masks: `count = len(np.unique(mask)) - 1` (excluding background label 0).

**Single Cell Cropper Tool** (`Single_Cell_Cropper_Tool`):
- **`min_area`** (int, auto-detected from mask type): Minimum pixel area threshold for valid objects. Auto-detected defaults: 50 for cell/nuclei masks, 1000 for organoid masks (`single_cell_cropper/tool.py`, lines 81-103). Objects below this threshold are filtered out to remove noise and debris.
- **`margin`** (int, auto-detected from mask type): Margin in pixels around each object bounding box. Auto-detected defaults: 1 for cell masks, 25 for nuclei masks, 10 for organoid masks. The crop size is calculated as `half_side = max(height//2, width//2) + margin`, ensuring the entire mask is included with padding (`single_cell_cropper/tool.py`, lines 556-569).
- **`output_format`** (str, default: 'png'): Output format for single-channel crops ('png', 'tif', 'jpg'). Multi-channel crops are always saved as TIFF to preserve all channel information (format: `(C, H, W)` for ImageJ compatibility, line 637-638).
- **Filtering statistics**: The tool tracks filtering metrics: `filtered_by_area` (objects below `min_area`), `filtered_by_border` (objects where crop boundaries cannot fully contain mask due to image borders), and `invalid_crop_data` (crops with NaN/Inf values or zero size). These statistics are included in the output for quality assessment.

**Cell State Analyzer Tools** (`Cell_State_Analyzer_Single_Tool`, `Cell_State_Analyzer_Multi_Tool`):
- **`max_epochs`** (int, default: 25): Maximum number of training epochs for contrastive learning. Training uses early stopping with patience=5 (no improvement for 5 consecutive epochs triggers termination, `cell_state_analyzer_multi/tool.py`, lines 300-338).
- **`early_stop_loss`** (float, default: 0.5): Early stopping threshold. If training loss drops below this value, training terminates early. Not actively used in current implementation (patience-based stopping is primary mechanism).
- **`batch_size`** (int, default: 16): Batch size for training and inference. For inference, batch size is automatically adjusted to `min(batch_size, num_crops)` to handle small datasets (line 454). Training uses full `batch_size` with shuffling (line 477).
- **`learning_rate`** (float, default: 3e-5): Learning rate for AdamW optimizer. Weight decay is set to 1e-4. Mixed-precision training via `torch.cuda.amp.autocast()` and `GradScaler()` for numerical stability (lines 302-318).
- **`cluster_resolution`** (float, default: 0.5): Leiden clustering resolution parameter. Higher values (0.5-2.0) produce more clusters (finer granularity), lower values (0.1-0.5) produce fewer clusters (coarser granularity). Clustering is performed via `sc.tl.leiden()` with cosine distance metric (line 507).
- **Zero-shot mode**: When `num_crops < 50`, the tool skips training and uses pre-trained DINOv3 features directly (line 431). This enables rapid analysis for small datasets but may have reduced feature quality compared to fine-tuned models.
- **UMAP parameters**: UMAP embedding uses `min_dist=0.05`, `spread=1.0`, `random_state=42`, and cosine distance metric via `sc.tl.umap()` (line 505). Neighbor graph is constructed with `n_neighbors=20` using cosine similarity (line 504).

**Analysis Visualizer Tool** (`Analysis_Visualizer_Tool`):
- **`chart_type`** (str, default: 'auto'): Visualization type. Options: 'auto' (automatic selection based on data characteristics), 'bar' (bar charts for counts), 'pie' (pie charts for proportions), 'box' (box plots for continuous metrics), 'violin' (violin plots for distributions), 'scatter' (scatter plots for correlations), 'line' (line plots for trends). Auto-selection logic (`analysis_visualizer/tool.py`, lines 161-185): 'bar' for count metrics, 'pie' for distribution/proportion metrics, 'box'/'violin' for confidence scores.
- **`comparison_metric`** (str, default: 'cell_count'): Metric to compare across groups. Common values: 'cell_count', 'cell_type_distribution', 'confidence_mean', 'area_mean', 'perimeter_mean'.
- **`group_column`** (str, default: 'group'): Column name in AnnData `obs` or result dictionary containing group labels.
- **`figure_size`** (tuple, default: (10, 6)): Figure dimensions in inches (width, height).
- **`dpi`** (int, default: 300): Resolution for saved figures in dots per inch.
- **Statistical testing**: Automatically performs appropriate tests based on number of groups: two-sample t-test (independent samples) for 2 groups (`scipy.stats.ttest_ind()`, lines 125-136), one-way ANOVA for 3+ groups (`scipy.stats.f_oneway()`, lines 142-154). Results include p-value, test statistic, and significance interpretation (p < 0.05 threshold).
- **Color schemes**: Uses colorblindness-adjusted palettes from `VisualizationConfig.get_color_palette()`: `vega_10_scanpy` (≤10 groups), `vega_20_scanpy` (11-20 groups), `default_26` (21-26 groups), `default_64` (>26 groups).

**Multi-image and Group Comparison**: For multi-image analyses, tools that merge all images (`Cell_State_Analyzer_*_Tool`, `Analysis_Visualizer_Tool`) execute once, processing all images together for unified analysis. Tools that process images independently (`Segmenter_Tool`, `Single_Cell_Cropper_Tool`) execute per-image in sequential loops. Group labels are propagated through the pipeline: crops inherit group labels from source images, and AnnData objects include group information in `obs['group']` for downstream statistical comparisons.


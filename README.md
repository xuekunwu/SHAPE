# SHAPE: Self-supervised morpHology Agent for cellular PhenotypE

**SHAPE** is an augmented agentic framework for morphological cell state inference across biological scales. Unlike traditional sequencing-based approaches that require months and thousands of dollars, SHAPE enables real-time morphological analysis in 5-30 minutes at less than $5, using live-cell imaging and LLM-driven planning.

## Why SHAPE?

Traditional sequencing-based methods for cell state analysis require:
- ⏱️ **3-6 months** of processing time
- 💰 **> $5,000** per experiment
- 🔬 **Destructive sampling** (no live-cell compatibility)

SHAPE offers a revolutionary alternative:
- ⚡ **5-30 minutes** for complete analysis
- 💵 **< $5** per experiment
- 🔬 **Live-cell compatible** - works with real-time imaging
- 🤖 **LLM-driven planning** - intelligent workflow generation
- 🔧 **Extensible tools** - easy integration of new methods
- 🚀 **GPU-accelerated** - leverages modern deep learning models

## Architecture

SHAPE follows a four-stage pipeline orchestrated by an intelligent agent:

```
Raw Images → Image Preprocessor → Cell/Organoid Segmenter → 
Single-cell Cropper → Cell-state Analyzer → Morphology Clusters
```

### Core Components

**SHAPE Agent** consists of two main components:

1. **Planner**:
   - Query analyzer: Understands user queries (e.g., "How many cell states in the image?")
   - Action predictor: Generates tool selection and execution plans
   - Context verifier: Validates intermediate results and decides when to stop
   - Solution generator: Synthesizes final answers from execution history

2. **Executor**:
   - Command generator: Converts planner actions into tool-specific commands
   - Command executor: Executes tools and collects results
   - Tool calling: Interfaces with extensible GPU-compatible tools

### Extensible GPU-Compatible Tools

SHAPE integrates with a wide range of tools and libraries:
- **Deep Learning**: PyTorch, DINO v3, Cellpose
- **Image Processing**: NumPy, Scikit-image, OpenCV
- **Single-cell Analysis**: Scanpy, Scvi-tools
- **Visualization**: Matplotlib, Seaborn
- **Data Processing**: Pandas

### Processing Pipeline

1. **Image Preprocessor**: Enhances and normalizes raw microscopy images
2. **Cell/Organoid Segmenter**: Performs instance segmentation to generate individual masks
3. **Single-cell Cropper**: Extracts individual cell crops from segmented masks
4. **Cell-state Analyzer**: Applies self-supervised learning to identify morphological cell states

### Key Features

- **Real-time Analysis**: 5-30 minutes vs. 3-6 months for sequencing-based methods
- **Cost-effective**: < $5 vs. > $5,000 for traditional approaches
- **Live-cell Compatible**: Works with live-cell imaging without sample destruction
- **LLM-driven Planning**: Intelligent tool selection and workflow orchestration
- **GPU-accelerated**: Leverages GPU computing for deep learning models
- **Extensible Tools**: Easy integration of new tools and methods
- **Multi-channel Support**: Handles multi-channel microscopy images
- **Multi-group Comparison**: Enables comparative analysis across experimental groups

## Biological Applications

SHAPE supports diverse applications in biological research:

- **iPSC Differentiation Purity**: Assess differentiation efficiency and purity in induced pluripotent stem cell cultures
- **Cell-state Trajectory**: Track cellular state transitions and developmental trajectories
- **Organoid Activities**: Analyze organoid morphology and activity patterns
- **Spatial Annotation**: Integrate morphological features with spatial context
- **Phenotypic Screening**: High-throughput screening of cellular phenotypes

## Installation

```bash
# Clone the repository
git clone https://github.com/xuekunwu/SHAPE.git
cd SHAPE

# Install dependencies
pip install -r requirements.txt

# Set up environment variables
export OPENAI_API_KEY="<your-api-key-here>"
```

### Verify Installation

After installation, verify that everything is set up correctly:

```bash
python test_installation.py
```

This will test imports, dependencies, and tool discovery.

## Quick Start

```python
import os
from shape import solve

# Set your API key
os.environ["OPENAI_API_KEY"] = "your-api-key-here"

# Solve a problem
result = solve(
    question="What cell states are present in this image?",
    image_path="path/to/image.tif",
    llm_engine_name="gpt-4o"
)

print(result["direct_output"])
```

For more control, you can use `construct_solver` to get individual components:

```python
from shape import construct_solver

# Construct solver with all components
solver = construct_solver(
    llm_engine_name="gpt-4o",
    api_key=os.getenv("OPENAI_API_KEY")
)

planner = solver["planner"]
executor = solver["executor"]
memory = solver["memory"]

# Use components as needed...
```

## Repository Structure

```
SHAPE/
├── README.md                 # This file
├── shape/                    # Core framework
│   ├── agent/               # Agent logic (planner, executor, memory)
│   ├── schemas/             # Tool and observation schemas
│   └── solver.py            # Main entry point (solve, construct_solver)
├── octotools/               # Tool implementations and utilities
│   ├── tools/              # Modular tool implementations
│   ├── models/             # Core models (planner, executor, memory)
│   ├── engine/             # LLM engine integrations
│   └── utils/              # Utility functions
└── examples/                # Reproducible analysis notebooks
```

## Citation

If you use SHAPE in your research, please cite:

```bibtex
@article{wu2026shape,
    title={Augmented agentic inference of morphological cell states across biological scales},
    author={Xuekun Wu, Pan Lu, Di Yin, Xiu Liu, Anisa Subbiah, Zehra Yildirim, Xin Zhou, Le Cong, James Zou, Joseph Wu},
    journal = {not available yet},
    year={2026}
}
```

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please see our contributing guidelines for details on code style, testing, and the pull request process.

---


# SHAPE: Self-supervised morpHology Agent for cellular Phenotype

**SHAPE** is an agentic framework that treats cell morphology as an executable layer for biological reasoning. Unlike traditional image analysis pipelines, SHAPE uses an LLM-driven agent to dynamically plan and execute morphological analysis workflows, enabling adaptive reasoning across biological scales.

## Core Philosophy

SHAPE is built on three foundational principles:

1. **Morphology-First Reasoning**: Morphological observations are first-class objects in the reasoning process, not just preprocessing steps for downstream analysis.

2. **Agentic Planning**: An LLM-driven planner dynamically selects and orchestrates tools based on query intent, avoiding hard-coded pipelines and enabling flexible, context-aware analysis.

3. **Modular Tools**: Tools are composable and swappable via unified interfaces, allowing the framework to adapt to new biological questions without core modifications.

## Architecture

```
┌─────────────────────────────────────────────────────────────┐
│                      User Query                              │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Agent (Planner)                          │
│  • Query analysis & intent understanding                    │
│  • Dynamic tool selection                                   │
│  • Multi-step planning with memory                          │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│                    Tool Registry                            │
│  ┌──────────────┐  ┌──────────────┐  ┌──────────────┐    │
│  │ Vision       │  │ Analysis     │  │ Knowledge    │    │
│  │ Tools        │  │ Tools        │  │ Tools       │    │
│  └──────────────┘  └──────────────┘  └──────────────┘    │
└──────────────────────┬──────────────────────────────────────┘
                       │
                       ▼
┌─────────────────────────────────────────────────────────────┐
│              Morphological Observations                      │
│  • Segmentation masks                                       │
│  • Cell embeddings                                          │
│  • Spatial graphs                                           │
│  • State classifications                                    │
└─────────────────────────────────────────────────────────────┘
```

### Key Components

- **`shape/agent/`**: Core agent logic including the planner, executor, and memory system
- **`shape/schemas/`**: Unified schemas for tool inputs/outputs and morphological observations
- **`octotools/tools/`**: Modular tool implementations including:
  - **Vision tools**: Image preprocessing, segmentation (cell/nuclei/organoid), single-cell cropping, feature extraction, clustering, visualization
  - **Analysis tools**: Cell state analysis, functional hypothesis generation
  - **Knowledge tools**: PubMed search, ArXiv search, Wikipedia search, Google search
  - **Utility tools**: Python code generation, text extraction, image captioning

## Biological Applications

SHAPE enables a wide range of morphological reasoning tasks:

- **Cell State Discovery**: Self-supervised learning on morphological features to identify distinct cellular states
- **Phenotypic Comparison**: Quantitative comparison of morphological phenotypes across conditions
- **Spatial Context Analysis**: Integration of morphological features with spatial neighborhood information
- **Multi-Scale Reasoning**: From single-cell morphology to tissue-level patterns

## Design Principles

- **No Hard-Coded Pipelines**: The agent selects tools dynamically based on query requirements
- **Task-Agnostic Core**: The agent makes minimal assumptions about specific biological tasks
- **Composable Tools**: Tools expose unified interfaces and can be combined in novel ways
- **Morphology as Data**: Morphological observations are structured, queryable objects

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

## Quick Start

```python
from shape.agent.planner import Planner
from shape.agent.executor import Executor

# Initialize agent components
planner = Planner(
    llm_engine_name="gpt-4",
    available_tools=get_available_tools(),
    api_key=os.getenv("OPENAI_API_KEY")
)

executor = Executor(planner=planner)

# Execute a query
result = executor.execute(
    question="What cell states are present in this image?",
    image_path="path/to/image.tif"
)
```

## Repository Structure

```
SHAPE/
├── README.md                 # This file
├── shape/                    # Core framework
│   ├── agent/               # Agent logic (planner, executor, memory)
│   └── schemas/             # Tool and observation schemas
├── octotools/               # Tool implementations and utilities
│   ├── tools/              # Modular tool implementations
│   ├── models/             # Core models (planner, executor, memory)
│   ├── engine/             # LLM engine integrations
│   └── utils/              # Utility functions
├── examples/                # Reproducible analysis notebooks
└── docs/                    # Design documents and methodology
```

## Citation

If you use SHAPE in your research, please cite:

```bibtex
@software{shape2024,
  title={SHAPE: Self-supervised morpHology Agent for cellular Phenotype},
  author={Wu, Xuekun},
  year={2024},
  url={https://github.com/xuekunwu/SHAPE}
}
```

## Contributors

| [xuekunwu](https://github.com/xuekunwu) |
|----------------------------------------|
| **Xuekun Wu**                          |

## License

MIT License - see [LICENSE](LICENSE) for details.

## Contributing

We welcome contributions! Please see our contributing guidelines for details on code style, testing, and the pull request process.

---


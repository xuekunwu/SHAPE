# SHAPE Design Philosophy

## Core Principles

### 1. Morphology-First Reasoning

Cell morphology is not just a preprocessing step—it is a first-class reasoning object. Morphological observations (segmentation masks, embeddings, clusters) are structured data that flow through the system and inform decision-making.

**Implications:**
- Tools produce structured observations, not just file paths
- Observations are queryable and composable
- The agent reasons about morphological features, not just executes pipelines

### 2. Agentic Planning

The agent dynamically plans analysis workflows based on query intent. There are no hard-coded pipelines.

**Implications:**
- Query analysis determines tool selection
- Tools are selected based on capabilities, not fixed sequences
- The agent adapts to novel queries without code changes

### 3. Modular Tools

Tools are self-contained, composable units with unified interfaces.

**Implications:**
- Tools declare their dependencies explicitly
- Tools can be swapped or extended without core changes
- Tool combinations emerge from query requirements, not design-time decisions

### 4. Human-Agent Collaboration

The system supports interactive, interpretable workflows where humans and agents collaborate.

**Implications:**
- Planning decisions are explainable
- Intermediate results are accessible
- Users can guide or override agent decisions

## What SHAPE Is Not

### Not a Toolbox

SHAPE is not a collection of independent tools. It is a framework where tools are orchestrated by an intelligent agent.

### Not a Pipeline

SHAPE does not enforce fixed analysis pipelines. Workflows emerge from query requirements.

### Not Task-Specific

SHAPE does not assume specific biological tasks. The agent adapts to diverse queries.

### Not a Demo

The Hugging Face Space is a showcase, not the core system. The framework is designed for programmatic use and research.

## Abstraction Levels

### Framework Level (`shape/`)

- Agent architecture
- Tool interfaces
- Observation schemas
- Planning logic

### Implementation Level (`octotools/`)

- Tool implementations
- LLM integration
- Image processing utilities
- Caching and persistence

### Application Level (`hf_space/`, `examples/`)

- User interfaces
- Example workflows
- Demonstrations

## Research Focus

SHAPE prioritizes:

1. **Methodology**: How to reason about morphology agentically
2. **Architecture**: How to design composable, extensible frameworks
3. **Abstraction**: How to separate concerns cleanly

SHAPE does not prioritize:

1. **UI polish**: The demo is functional, not polished
2. **Performance optimization**: Efficiency is important but not the primary focus
3. **Biological claims**: We report what morphology supports, not beyond

## Design Trade-offs

### Flexibility vs. Efficiency

SHAPE prioritizes flexibility. The agent can handle diverse queries but may be less efficient than task-specific pipelines.

### Generality vs. Specificity

SHAPE prioritizes generality. The framework works across biological domains but may require more configuration than domain-specific tools.

### Abstraction vs. Control

SHAPE prioritizes abstraction. Users interact with queries, not tool parameters, but may have less fine-grained control.

## Future Directions

1. **Spatial Reasoning**: Enhanced spatial analysis tools
2. **Multi-Modal Integration**: Better integration with omics data
3. **Learning**: Agent learns from user feedback
4. **Composition**: More sophisticated tool composition patterns


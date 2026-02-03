# SHAPE Architecture

## Overview

SHAPE is an agentic framework for morphological reasoning in biological image analysis. The architecture emphasizes:

1. **Agent-driven planning**: LLM-based planner dynamically selects tools
2. **Modular tools**: Composable tools with unified interfaces
3. **Morphology-first reasoning**: Morphological observations as first-class objects
4. **Human-agent collaboration**: Interactive, interpretable workflows

## Core Components

### Agent Layer

The agent layer (`shape/agent/`) consists of three main components:

- **Planner** (`planner.py`): Analyzes queries and generates step-by-step plans
- **Executor** (`executor.py`): Executes planned steps using available tools
- **Memory** (`memory.py`): Maintains execution history and context

### Tool Layer

Tools are organized by domain (`shape/tools/`):

- **Vision** (`vision/`): Segmentation, embedding, clustering
- **Spatial** (`spatial/`): Neighborhood graphs, niche detection
- **Omics** (`omics/`): GRN inference, ligand-receptor analysis

### Schema Layer

Schemas (`shape/schemas/`) define:

- **Formatters**: Communication formats between agent components
- **Observations**: Structured morphological observations

## Design Principles

### No Hard-Coded Pipelines

The agent selects tools dynamically based on query requirements. There are no fixed analysis pipelines.

### Task-Agnostic Core

The agent makes minimal assumptions about specific biological tasks. Tool selection is driven by query intent, not predefined workflows.

### Composable Tools

Tools expose unified interfaces and can be combined in novel ways. Each tool is self-contained and declares its dependencies.

### Morphology as Data

Morphological observations are structured, queryable objects. They flow through the system as first-class data structures.

## Data Flow

```
User Query
    ↓
Planner (analyzes intent)
    ↓
NextStep (tool selection)
    ↓
Executor (executes tool)
    ↓
Tool (produces observation)
    ↓
Memory (stores observation)
    ↓
Planner (uses memory for next step)
    ↓
... (repeat until query answered)
    ↓
Final Answer
```

## Tool Execution Model

1. **Query Analysis**: Planner analyzes query to understand intent
2. **Step Generation**: Planner selects next tool based on:
   - Query requirements
   - Available tools and their capabilities
   - Previous execution results (memory)
   - Tool dependencies
3. **Command Generation**: Executor generates tool command with proper context
4. **Tool Execution**: Tool processes inputs and produces observations
5. **Memory Update**: Observations are stored in memory
6. **Verification**: Planner verifies if query can be answered
7. **Termination or Continuation**: Either generate final answer or continue to next step

## Extension Points

### Adding New Tools

1. Create tool class inheriting from `BaseTool`
2. Implement `execute()` method
3. Declare tool metadata (name, description, input/output types)
4. Register tool in tool registry
5. Tool is automatically available to planner

### Customizing Planner

The planner can be customized by:
- Modifying tool priority mappings
- Adjusting query analysis prompts
- Adding domain-specific heuristics

### Observation Schemas

New observation types can be added to `shape/schemas/observations.py` to support novel analysis workflows.

## Implementation Notes

Currently, SHAPE uses `octotools` as the implementation layer. The `shape/` package provides a clean interface that will eventually replace `octotools` as the core implementation.

This separation allows:
- Clean research-focused API in `shape/`
- Gradual migration from `octotools` to `shape/`
- Backward compatibility during transition


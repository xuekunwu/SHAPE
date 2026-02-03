# SHAPE Examples

This directory contains example notebooks and scripts demonstrating SHAPE's capabilities.

## Basic Usage

`basic_usage.py` shows how to use SHAPE programmatically for morphological analysis.

## Example Workflows

### Cell State Analysis

Analyze cell states in microscopy images:

```python
from shape.agent import Planner, Executor, Memory

# Initialize agent
planner = Planner(...)
executor = Executor(...)
memory = Memory()

# Query
query = "What cell states are present in this image?"
result = analyze_with_shape(image_path, query, planner, executor, memory)
```

### Morphological Comparison

Compare morphological features across conditions:

```python
query = "Compare cell morphology between treatment and control groups"
# Agent automatically handles multi-image processing
```

### Spatial Analysis

Analyze spatial patterns in cell distributions:

```python
query = "What spatial niches are present in this tissue?"
# Uses spatial tools for neighborhood analysis
```

## Notebooks

TODO: Add Jupyter notebooks demonstrating:
- Interactive analysis workflows
- Custom tool development
- Observation schema usage
- Multi-modal integration


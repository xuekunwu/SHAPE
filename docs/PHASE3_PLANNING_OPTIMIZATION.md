# Phase 3: Planning Strategy Optimization

## Overview

This document outlines the optimization strategy for the Planning component, inspired by Biomni's modular and efficient planning approach.

## Current State Analysis

### Strengths
1. **Tool Priority System**: Already implemented with HIGH/MEDIUM/LOW priorities
2. **Dependency Management**: Tool dependencies are tracked and validated
3. **Domain Detection**: Automatic task domain detection (bioimage, search, etc.)
4. **Query Type Detection**: Distinguishes between counting vs. full analysis queries

### Areas for Improvement
1. **Task Decomposition**: Currently step-by-step planning, could benefit from high-level task decomposition
2. **Multi-Image Processing**: Could be more efficient with parallel task identification
3. **Context Awareness**: Could better leverage image properties and task context
4. **Planning Efficiency**: Reduce redundant LLM calls and improve decision-making speed

## Optimization Strategy

### 1. Enhanced Task Decomposition

**Goal**: Break down complex queries into sub-tasks upfront, enabling better planning.

**Implementation**:
- Add `decompose_task()` method to Planner
- Identify independent vs. dependent sub-tasks
- Plan execution order based on dependencies

**Example**:
```
Query: "What changes of organoid among different groups?"
Decomposition:
  - Task 1: Segment organoids in all images (parallelizable)
  - Task 2: Extract features from all organoids (depends on Task 1)
  - Task 3: Compare groups statistically (depends on Task 2)
  - Task 4: Visualize results (depends on Task 3)
```

### 2. Parallel Task Identification

**Goal**: Identify tasks that can be executed in parallel for multi-image scenarios.

**Implementation**:
- Detect when multiple images need the same tool
- Group images by tool requirements
- Execute in parallel where possible

**Example**:
```
16 images, all need Organoid_Segmenter_Tool
-> Execute segmentation in parallel batches
```

### 3. Context-Aware Planning

**Goal**: Better utilize image properties and task context for smarter planning.

**Implementation**:
- Extract image metadata (channels, dimensions, format)
- Use metadata to pre-select appropriate tools
- Adjust planning based on image characteristics

**Example**:
```
Multi-channel TIFF detected
-> Pre-select tools that handle multi-channel images
-> Skip tools that don't support multi-channel
```

### 4. Planning Efficiency Improvements

**Goal**: Reduce LLM calls and improve decision-making speed.

**Implementation**:
- Cache common planning patterns
- Use rule-based decisions for common scenarios
- Only use LLM for complex/ambiguous cases

**Example**:
```
If query = "how many cells" AND image exists:
  -> Rule-based: Image_Preprocessor_Tool → Segmenter → STOP
  -> No LLM call needed
```

## Implementation Plan

### Step 1: Add Task Decomposition
- [ ] Implement `decompose_task()` method
- [ ] Add task dependency tracking
- [ ] Integrate with existing planning flow

### Step 2: Parallel Task Support
- [ ] Identify parallelizable tasks
- [ ] Add batch execution support
- [ ] Update executor to handle parallel tasks

### Step 3: Context-Aware Planning
- [ ] Enhance image metadata extraction
- [ ] Add metadata-based tool pre-selection
- [ ] Update planning prompts with context

### Step 4: Efficiency Improvements
- [ ] Add planning pattern cache
- [ ] Implement rule-based shortcuts
- [ ] Optimize LLM prompt sizes

## Success Metrics

1. **Planning Speed**: Reduce average planning time by 30%
2. **Multi-Image Efficiency**: 2-4x speedup for multi-image tasks
3. **Tool Selection Accuracy**: Maintain or improve current accuracy
4. **Code Maintainability**: Keep code clean and well-documented

## Notes

- All changes must maintain backward compatibility
- Existing functionality should not be broken
- Gradual rollout with testing at each step


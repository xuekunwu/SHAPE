# Tool Priority System for LLM-Orchestrated Single-Cell Bioimage Analysis

## Overview

This document describes the tool priority system implemented to ensure that the LLM-Orchestrated Single-Cell Bioimage Analysis System selects the most appropriate tools for each task, avoiding irrelevant tool calls and errors.

## Architecture

### 1. Tool Priority Configuration (`octotools/models/tool_priority.py`)

A comprehensive priority system that categorizes tools based on their relevance to bioimage analysis tasks.

#### Priority Levels

- **HIGH (1)**: High priority tools for bioimage analysis
  - Core image processing and segmentation: `Image_Preprocessor_Tool`, `Nuclei_Segmenter_Tool`, `Single_Cell_Cropper_Tool`
  - Specialized analysis tools: `Fibroblast_State_Analyzer_Tool`, `Fibroblast_Activation_Scorer_Tool`, `Analysis_Visualizer_Tool`

- **MEDIUM (2)**: Medium priority tools
  - General-purpose image analysis tools: `Object_Detector_Tool`, `Advanced_Object_Detector_Tool`, `Image_Captioner_Tool`

- **LOW (3)**: Low priority tools (use sparingly)
  - Utility tools: `Text_Detector_Tool`
  - Code generation tools: `Python_Code_Generator_Tool`, `Generalist_Solution_Generator_Tool`

- **EXCLUDED (99)**: Tools not relevant for bioimage analysis
  - All search tools (Google_Search_Tool, Pubmed_Search_Tool, etc.)
  - Text extraction tools (Url_Text_Extractor_Tool)
  - `Relevant_Patch_Zoomer_Tool` (not suitable for bioimages)

#### Key Features

1. **Domain Detection**: Automatically detects task domain (bioimage, search, text_extraction, general)
2. **Tool Filtering**: Filters out irrelevant tools based on detected domain
3. **Dependency Management**: Tracks tool dependencies (e.g., Single_Cell_Cropper_Tool requires Nuclei_Segmenter_Tool)
4. **Priority-Based Recommendations**: Suggests next tools based on priorities and dependencies

### 2. Planner Integration (`octotools/models/planner.py`)

The Planner class has been enhanced to use the priority system:

#### Changes Made

1. **Tool Priority Manager Integration**:
   - Initializes `ToolPriorityManager` in `__init__`
   - Tracks detected task domain
   
2. **Enhanced Query Analysis** (`analyze_query`):
   - Detects task domain automatically
   - Filters tools based on domain
   - Excludes irrelevant tools from consideration

3. **Improved Next Step Generation** (`generate_next_step`):
   - Filters available tools based on domain
   - Groups tools by priority in prompt
   - Provides recommended next tools considering dependencies
   - Enhanced prompt with explicit priority guidance

4. **Tool Selection Validation** (`_validate_tool_selection`):
   - Validates that selected tool is in available tools list
   - Checks if tool is excluded for the domain
   - Verifies tool dependencies are satisfied
   - Warns if LOW priority code generation tools are selected when higher-priority alternatives exist

## Usage

The system works automatically. When the Planner analyzes a query:

1. **Domain Detection**: The system detects if the query is related to:
   - Bioimage analysis (default for this system)
   - Search tasks
   - Text extraction
   - General tasks

2. **Tool Filtering**: Based on the detected domain:
   - Bioimage tasks: Excludes search/text tools, includes all bioimage-relevant tools
   - Other domains: Applies appropriate filtering

3. **Priority-Based Selection**: The LLM receives:
   - Tools organized by priority level
   - Recommended next tools based on dependencies
   - Clear guidance on which tools to prefer

4. **Validation**: After tool selection:
   - Validates the tool is appropriate
   - Checks dependencies are satisfied
   - Provides warnings for suboptimal selections

## Benefits

1. **Reduced Errors**: Prevents calling irrelevant tools (e.g., search tools for image analysis)
2. **Better Tool Selection**: Prioritizes tools most relevant to the task
3. **Dependency Management**: Ensures tools are called in the correct order
4. **Extensibility**: Easy to add new tools and adjust priorities
5. **Maintainability**: Centralized priority configuration

## Configuration

To modify tool priorities or add new tools, edit `octotools/models/tool_priority.py`:

1. Update `BIOIMAGE_TOOL_PRIORITIES` dictionary to set priorities
2. Add tool dependencies to `TOOL_DEPENDENCIES` dictionary
3. Update keywords in domain detection sets if needed

## Example

For a bioimage analysis query like "segment the cells in this image":

1. Domain detected: `bioimage`
2. Tools filtered: Search/text tools excluded
3. Available tools prioritized:
   - HIGH: Image_Preprocessor_Tool, Nuclei_Segmenter_Tool, Single_Cell_Cropper_Tool, Fibroblast_State_Analyzer_Tool, Analysis_Visualizer_Tool
   - MEDIUM: Object_Detector_Tool, Image_Captioner_Tool, etc.
   - LOW: Text_Detector_Tool, Python_Code_Generator_Tool, Generalist_Solution_Generator_Tool
4. LLM guided to select from HIGH priority tools first
5. Tool selection validated before execution

## Files Modified

- `octotools/models/tool_priority.py` (NEW): Priority system implementation
- `octotools/models/planner.py` (MODIFIED): Integrated priority system

## Future Enhancements

- Dynamic priority adjustment based on query complexity
- Learning from successful tool sequences
- More sophisticated dependency resolution
- Integration with tool performance metrics
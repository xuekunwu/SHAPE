# Planner Logic Overview

## 1. Overall Architecture

The Planner is responsible for orchestrating the tool execution chain in the LLM-Orchestrated Bioimage Analysis System. It uses a combination of:
- **Rule-based decisions** (for efficiency)
- **LLM-based planning** (for complex scenarios)
- **Code-level enforcement** (for critical tool chains)

## 2. Query Type Detection

### 2.1 Query Classification

The planner classifies queries into two main types:

1. **Simple Counting Queries**
   - Examples: "how many cells", "count organoids", "number of cells"
   - Required tools: `Image_Preprocessor_Tool` (optional) → `Segmenter` → STOP
   - Result: Cell count is available in segmentation result

2. **Full Cell State Analysis Queries**
   - Examples: "what cell states", "analyze cell states", "compare groups", "what changes"
   - Required tools: Full pipeline (see Section 3)
   - Result: Complete analysis with clustering, UMAP, and visualization

### 2.2 Detection Method

The `_requires_full_cell_state_analysis(question: str)` method determines if a query requires full analysis:

```python
def _requires_full_cell_state_analysis(self, question: str) -> bool:
    """
    Returns True if query requires full cell state analysis pipeline.
    Returns False for simple counting queries.
    """
```

**Keywords that trigger full analysis:**
- Cell state keywords: 'cell state', 'cell states', 'cluster', 'clustering', 'umap', 'embedding'
- Comparison keywords: 'compare', 'comparison', 'difference', 'change', 'group', 'treatment', 'control'
- Analysis keywords: 'analyze', 'phenotype', 'morphological state'

**Keywords that indicate simple counting:**
- 'how many', 'count', 'number of', 'quantity', 'total cells'

**Priority order:**
1. Check comparison keywords first (typically need full analysis)
2. Check counting keywords (if only counting, return False)
3. Check state analysis keywords
4. Default: False (let LLM decide)

## 3. Bioimage Analysis Tool Chain

### 3.1 Complete Chain (for full cell state analysis)

```
Image_Preprocessor_Tool (optional)
    ↓
[Cell_Segmenter_Tool | Nuclei_Segmenter_Tool | Organoid_Segmenter_Tool]
    ↓
Single_Cell_Cropper_Tool (MANDATORY)
    ↓
Cell_State_Analyzer_Tool (MANDATORY - performs self-supervised learning)
    ↓
Analysis_Visualizer_Tool (MANDATORY)
    ↓
Image_Captioner_Tool (RECOMMENDED, not mandatory)
```

### 3.2 Tool Dependencies

Defined in `octotools/models/tool_priority.py`:

```python
TOOL_DEPENDENCIES = {
    "Single_Cell_Cropper_Tool": ["Nuclei_Segmenter_Tool", "Cell_Segmenter_Tool", "Organoid_Segmenter_Tool"],
    "Cell_State_Analyzer_Tool": ["Single_Cell_Cropper_Tool"],
    "Analysis_Visualizer_Tool": [],  # Can work with any analysis output
}
```

### 3.3 Critical Tool: Cell_State_Analyzer_Tool

**Why it's essential:**
- Performs self-supervised learning (contrastive learning) using DINOv3
- Extracts morphological features from cell crops
- Generates UMAP embeddings
- Performs clustering for cell state determination
- **Cannot be skipped** for cell state analysis queries

## 4. Planning Flow

### 4.1 Rule-Based Decisions (First Pass)

The `_try_rule_based_decision()` method handles simple cases without LLM calls:

**Rule 1: Simple counting query with no steps**
- Pattern: "how many cells" + no tools used
- Action: Select appropriate segmenter (Cell/Nuclei/Organoid)

**Rule 2: Counting query after segmentation**
- Pattern: "how many" + segmentation tool used
- Action: Return None (signal completion)

**Rule 3: Multi-channel image detection**
- Pattern: Multi-channel image detected
- Action: Log info (informational, not a decision)

### 4.2 Pre-LLM Enforcement (Second Pass)

**Note:** We do NOT force tools before LLM selection. The LLM intelligently decides the analysis depth based on the query requirements. This allows for flexible analysis chains:
- Simple counting: Segmentation → STOP
- Basic morphology: Segmentation → (optional cropping) → Analysis_Visualizer
- Cell state analysis: Segmentation → Cropping → Cell_State_Analyzer → Analysis_Visualizer

### 4.3 LLM-Based Planning (Third Pass)

If no rule matches and no pre-LLM enforcement applies, use LLM to select next tool.

The prompt includes:
- Query type identification
- Tool chain priority
- Critical enforcement rules
- Tool dependencies
- Previous steps context

### 4.4 Post-LLM Enforcement (Fourth Pass)

After LLM selection, verify and enforce critical dependency rules:

**Note:** We do NOT enforce Segmentation → Single_Cell_Cropper. The LLM decides based on query requirements.

**Enforcement 1: Single_Cell_Cropper → Cell_State_Analyzer**
- Condition: Last tool was `Single_Cell_Cropper_Tool` AND query requires full analysis
- Action: Override LLM selection if not `Cell_State_Analyzer_Tool`
- Location: Lines 744-775 in `planner.py`
- **This is the critical fix for the reported issue**

**Enforcement 2: Cell_State_Analyzer → Analysis_Visualizer**
- Condition: Last tool was `Cell_State_Analyzer_Tool`
- Action: Override LLM selection if not `Analysis_Visualizer_Tool`
- Location: Lines 780-804 in `planner.py`

**Enforcement 3: Analysis_Visualizer → Image_Captioner (Recommendation)**
- Condition: Last tool was `Analysis_Visualizer_Tool` AND `Image_Captioner_Tool` not used
- Action: Override LLM selection (recommendation, not mandatory)
- Location: Lines 806-834 in `planner.py`

## 5. Enforcement Logic Consistency

### 5.1 Enforcement Points

All enforcement happens in `generate_next_step()`:

1. **Pre-LLM**: None (removed - let LLM decide analysis depth)
2. **Post-LLM** (lines 747-778): **Force `Cell_State_Analyzer_Tool` after `Single_Cell_Cropper_Tool`** ⭐
   - Rationale: If cropping was done, individual cell analysis is needed, which requires self-supervised learning
3. **Post-LLM** (lines 780-804): Force `Analysis_Visualizer_Tool` after `Cell_State_Analyzer_Tool`
4. **Post-LLM** (lines 806-834): Recommend `Image_Captioner_Tool` after `Analysis_Visualizer_Tool`

### 5.2 Condition Checking

All enforcements check:
- `last_tool` matches expected previous tool
- Tool is in `available_tools`
- Override LLM selection if it doesn't match

**Note:** We removed the `requires_full_analysis` check for `Segmentation → Single_Cell_Cropper` enforcement. The LLM now intelligently decides whether cropping is needed based on the query.

### 5.3 Logging

Each enforcement logs:
- Warning when overriding LLM selection
- Justification for the override
- Context and sub-goal for the forced tool

## 6. Prompt Engineering

### 6.1 Query Analysis Prompt

Located in `analyze_query()` method (lines 144-248):
- Emphasizes MINIMUM necessary tools
- Distinguishes counting vs. full analysis queries
- Prevents over-extension

### 6.2 Next Step Generation Prompt

Located in `generate_next_step()` method (lines 497-695):
- Includes complete tool chain documentation
- Emphasizes `Cell_State_Analyzer_Tool` as ESSENTIAL
- Provides critical enforcement rules
- Lists tool dependencies

### 6.3 Memory Verification Prompt

Located in `verificate_memory()` method (lines 916-1109):
- Checks if query is satisfied
- Distinguishes counting vs. full analysis completion
- Prevents unnecessary continuation

## 7. Edge Cases and Special Handling

### 7.1 Counting Queries After Segmentation

- If query is "how many cells" and segmentation is done → STOP
- Count is available in segmentation result
- Do NOT force `Single_Cell_Cropper_Tool`

### 7.2 Comparison Queries

- Comparison queries (e.g., "compare groups", "what changes") typically require full analysis
- `_requires_full_cell_state_analysis()` returns True for comparison keywords
- Full chain is enforced

### 7.3 Multi-Channel Images

- Detected via `get_image_info()` using `ImageProcessor`
- Logged for informational purposes
- Tools should handle multi-channel images automatically

## 8. Validation and Error Handling

### 8.1 Tool Selection Validation

The `_validate_tool_selection()` method (lines 845-914) checks:
- Tool is in available tools list
- Tool is not EXCLUDED for the domain
- Tool dependencies are satisfied
- Warns if LOW priority tool is selected when HIGH priority alternatives exist

### 8.2 Error Recovery

- If LLM returns invalid tool name, validation warns but doesn't block
- Enforcement overrides take precedence over LLM selection
- Fallback to rule-based decisions when possible

## 9. Summary of Critical Fixes

### 9.1 Issue: Missing Cell_State_Analyzer_Tool

**Problem:** After `Single_Cell_Cropper_Tool`, the system was jumping directly to `Analysis_Visualizer_Tool`, skipping the critical self-supervised learning step.

**Solution:** Added code-level enforcement (lines 747-778) to force `Cell_State_Analyzer_Tool` after `Single_Cell_Cropper_Tool`.

**Key Changes:**
1. Added enforcement check for `Single_Cell_Cropper_Tool` → `Cell_State_Analyzer_Tool`
2. Updated prompt to emphasize `Cell_State_Analyzer_Tool` as ESSENTIAL
3. Enhanced `_requires_full_cell_state_analysis()` to detect comparison queries
4. Updated tool chain documentation in prompts

### 9.2 Issue: Overly Rigid Enforcement

**Problem:** The system was forcing `Single_Cell_Cropper_Tool` after segmentation, even when the query might not need individual cell analysis.

**Solution:** Removed the `Segmentation → Single_Cell_Cropper` enforcement. The LLM now intelligently decides the analysis depth based on query requirements.

**Key Changes:**
1. Removed Pre-LLM enforcement for `Segmentation → Single_Cell_Cropper`
2. Removed Post-LLM enforcement for `Segmentation → Single_Cell_Cropper`
3. Updated prompts to guide LLM on different analysis levels (Level 1: counting, Level 2: basic morphology, Level 3: full cell state analysis)
4. Kept dependency enforcement: If cropping is done, then `Cell_State_Analyzer_Tool` is mandatory

### 9.2 Consistency Checks

All enforcement points now follow the same pattern:
1. Check `last_tool`
2. Check `requires_full_analysis` (if applicable)
3. Extract `selected_tool` from LLM response
4. Override if needed
5. Log warning with justification

## 10. Testing

The test suite `tests/test_cell_state_analyzer_enforcement.py` verifies:
1. Full analysis detection works correctly
2. Tool chain enforcement logic is in place
3. Bioimage chain order is documented
4. Code-level enforcement exists

All tests pass, confirming the fix is properly implemented.


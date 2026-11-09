"""
Intelligent Planner - Goal-Oriented Planning for Bioimage Analysis

Design Philosophy (inspired by Biomni):
- Goal-Oriented Action Planning (GOAP)
- LLM-driven intelligent decision making
- Minimal hardcoded rules, maximum flexibility
- Rich context provision for LLM reasoning
- Essential bioimage analysis chain awareness
- Multi-image and group comparison support
"""

import os
from typing import Dict, Any, List, Optional, Tuple
from PIL import Image
from io import BytesIO

from shape.engine.factory import create_engine
from shape.agent.memory import Memory
from shape.schemas.formatters import QueryAnalysis, NextStep, MemoryVerification
from shape.tools.tool_priority import ToolPriorityManager
from shape.utils import logger, ResponseParser
from shape.utils.image_processor import ImageProcessor

_FIGURE_EXTS = (".png", ".jpg", ".jpeg")


def _iter_fig_paths(obj):
    """Yield every string in a (possibly nested) tool result."""
    if isinstance(obj, str):
        yield obj
    elif isinstance(obj, dict):
        for v in obj.values():
            yield from _iter_fig_paths(v)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            yield from _iter_fig_paths(v)


def collect_output_figures(memory: Memory, max_images: int = 2) -> List[str]:
    """Most-recent-first deliverable FIGURES the pipeline produced (overlay, UMAP,
    cluster plots) -- the artifacts that carry the results, so the reasoning layer
    can visually check them. Walks newest steps first; dedups; existing PNG/JPG only
    (mask .tif/.npy excluded); capped at `max_images` to bound token cost.

    Cap is 2 (was 4): each figure is sent at detail='high' and is attached to BOTH the
    planner's next-step call and the task-completion verifier, so it is a large, repeated
    token cost. 2 keeps the current + previous attempt visible (enough for the planner to
    judge a single-tool result); multi-stage chains see fewer prior-stage outputs, a minor
    loss of visual context for a big token saving. Env SHAPE_PLANNER_MAX_FIGURES overrides."""
    try:
        max_images = int(os.environ.get("SHAPE_PLANNER_MAX_FIGURES", str(max_images)))
    except ValueError:
        pass
    if memory is None or not hasattr(memory, "get_actions"):
        return []
    seen, figs = set(), []
    for action in reversed(memory.get_actions(llm_safe=False)):
        for path in _iter_fig_paths(action.get("result")):
            if not isinstance(path, str) or path in seen:
                continue
            seen.add(path)
            if path.lower().endswith(_FIGURE_EXTS) and os.path.exists(path):
                figs.append(path)
                if len(figs) >= max_images:
                    return figs
    return figs


def _append_figure_bytes(input_data: list, memory: Memory, max_images: int = 2) -> List[str]:
    """Read the pipeline's output figures and append their bytes to a multimodal
    input list (the engine renders bytes as images). Returns the attached paths."""
    attached = []
    for path in collect_output_figures(memory, max_images):
        try:
            with open(path, "rb") as f:
                input_data.append(f.read())
            attached.append(path)
        except Exception:
            pass
    return attached


class Planner:
    """
    Intelligent planner for bioimage analysis system.
    
    Core principles:
    - Trust LLM intelligence for tool selection
    - Provide rich context (query, memory, tools, metadata)
    - Guide through prompts, not hardcoded rules
    - Retain essential bioimage analysis chain awareness
    - Support multi-image and group comparison scenarios
    """
    
    def __init__(self, llm_engine_name: str, toolbox_metadata: dict = None,
                 available_tools: List = None, api_key: str = None, initializer=None):
        self.toolbox_metadata = toolbox_metadata or {}
        self.available_tools = available_tools or []
        self.api_key = api_key
        # Initializer is used to load tool metadata lazily (only for the tools the
        # planner actually selects), instead of reading all tools up front.
        self.initializer = initializer
        self.recommended_params = None
        
        # Initialize LLM engines
        self.llm_engine = create_engine(
            model_string=llm_engine_name, 
            is_multimodal=False, 
            api_key=api_key
        )
        self.llm_engine_mm = create_engine(
            model_string=llm_engine_name, 
            is_multimodal=True, 
            api_key=api_key
        )
        
        # Tool priority manager
        self.priority_manager = ToolPriorityManager()
        
        # State tracking
        self.query_analysis = None
        self.task_spec = None            # 045 P3 Fix 0: query stage-boundary (set by analyze_query)
        self.last_usage = {}
        self.base_response = None
        self.detected_domain = 'general'
    
    def get_image_info(self, image_path: str) -> Dict[str, Any]:
        """Extract comprehensive image metadata for context."""
        if not image_path or not os.path.isfile(image_path):
            return {}
        
        try:
            img_data = ImageProcessor.load_image(image_path)
            return {
                "width": img_data.shape[1],
                "height": img_data.shape[0],
                "num_channels": img_data.num_channels,
                "is_multi_channel": img_data.is_multi_channel,
                "channel_names": getattr(img_data, 'channel_names', []),
                "dtype": str(img_data.dtype),
                "format": "multi-channel TIFF" if img_data.is_multi_channel else "single-channel"
            }
        except Exception as e:
            logger.debug(f"ImageProcessor failed, using PIL fallback: {e}")
            try:
                with Image.open(image_path) as img:
                    width, height = img.size
                return {
                    "width": width,
                    "height": height,
                    "num_channels": 1,
                    "is_multi_channel": False
                }
            except Exception:
                return {}
    
    def get_image_info_bytes(self, bytes: str) -> Dict[str, Any]:
        """Extract image info from bytes (for compatibility)."""
        try:
            with Image.open(BytesIO(bytes)) as img:
                width, height = img.size
            return {
                "width": width,
                "height": height,
                "num_channels": 1,
                "is_multi_channel": False
            }
        except Exception:
            return {}
    
    def _format_memory(self, memory: Memory, max_actions: int = 10) -> str:
        """Format memory for prompt - concise and relevant."""
        actions = memory.get_actions(llm_safe=True)
        if not actions:
            return "No previous steps."
        
        recent = actions[-max_actions:]
        formatted = []
        for i, action in enumerate(recent, 1):
            tool = action.get('tool_name', 'Unknown')
            goal = action.get('sub_goal', '')
            result = action.get('result', {})
            
            # Truncate long results
            result_str = str(result)
            if len(result_str) > 500:
                result_str = result_str[:500] + "... (truncated)"

            entry = (
                f"Step {i}: {tool}\n"
                f"  Goal: {goal[:200] if len(goal) > 200 else goal}\n"
                f"  Result: {result_str}"
            )
            qc = self._format_qc(action.get('verification'))
            if qc:
                entry += f"\n  QC: {qc}"
            image_context = self._format_image_context_profile(result)
            if image_context:
                entry += f"\n  ImageContext: {image_context}"
            formatted.append(entry)
        
        if len(actions) > max_actions:
            formatted.insert(0, f"[Showing last {max_actions} of {len(actions)} steps. Earlier steps omitted to save context.]")
        
        return "\n\n".join(formatted)
    
    def _format_qc(self, verification) -> str:
        """One-line QC digest of a step's VLRR verdict for the reasoning prompt.

        Surfaces the verdict (decision/action/status) for every stage and the full
        numeric metrics only for the concerning ones (WARNING/FAIL or a `conceded`
        stage), so the LLM can see a failed/conceded step and weak metrics (e.g.
        negative group silhouette) instead of silently over-claiming.

        `action == "conceded"` (task 032; was `escalate_upstream`/`escalate_human`)
        marks a stage whose tuning loop gave up on a fault it could not fix locally.
        When it carries an `upstream_target` (top-level, below), the digest renders
        it WITH an explicit re-run cue -- deterministic, no LLM needed to produce it
        (the LLM only needs to read + act on it, per the ESCALATION HANDLING guidance
        below). A `conceded` with no upstream_target (a `fault=ceiling`: the tool/data
        itself can't do better here) or a bare `overall=conceded` with the per-stage
        action still `retune_here` (StepVerifier's own tuning ceiling) render with no
        such cue -- see the ESCALATION HANDLING prose.

        For a concerning stage, the per-stage `fault` (ceiling | upstream -- task 045:
        the old third value `human` is retired, folded into `ceiling` -- already
        computed by the VRR verifier, `compact_verification` in memory.py)
        is also rendered verbatim as `fault=<value>` (task 045 Phase-2 refinement).
        This is a deterministic text surfacing of an existing field, not a new
        judgment call by this function: the reasoning prompts that read this digest
        (`verificate_memory`'s completion check, `generate_next_step`'s ESCALATION
        HANDLING) treat it as EVIDENCE for their OWN termination/selection reasoning
        -- they weigh it, they are not meant to obey it mechanically.

        Also handles the batch/multi-image shape `{"tool":.., "per_image": [record, ...]}`
        (task 047): each record is a flat `{"status":.., "stages":[...]}` record like the
        single-image shape, unwrapped and rendered with the same per-stage logic. A
        single-record batch (n=1, the common case) renders IDENTICALLY to the equivalent
        flat record; N>1 renders one digest per image, prefixed by its index.
        """
        if not isinstance(verification, dict):
            return ""

        def _digest_one(v: Dict[str, Any]) -> str:
            parts = []
            for s in v.get("stages", []) or []:
                if not isinstance(s, dict):
                    continue
                decision, action = s.get("decision"), s.get("action")
                seg = f"{s.get('stage')}={decision}/{action}"
                concerning = decision in ("WARNING", "FAIL") or action == "conceded"
                if concerning:
                    if s.get("fault"):
                        # ceiling | upstream -- surfaced as plain evidence for the
                        # reasoning prompts to weigh, see docstring above.
                        seg += f" fault={s['fault']}"
                    if s.get("no_output"):
                        seg += f" NO OUTPUT produced ({s['no_output']})"
                    if s.get("kept_input"):
                        seg += f" INPUT KEPT unchanged, deliberately ({s['kept_input']})"
                    if s.get("notes"):
                        seg += f" ({s['notes']})"
                    # [045 P3 Fix 4] the lever ledger (reconcile diagnostic): which
                    # retunes actually moved the metric (effective) vs did nothing
                    # (inert) or hit no headroom -- so the planner SEES "re-tuning isn't
                    # helping" as evidence for its stop/continue call (was computed by
                    # the VRR loop but dropped before the digest). "untried" is omitted
                    # (uninformative: a lever set but never reconciled).
                    ledger = {k: v for k, v in (s.get("ledger") or {}).items()
                              if v in ("inert", "no-headroom", "effective")}
                    if ledger:
                        seg += f" ledger={ledger}"
                    metrics = s.get("metrics") or {}
                    if metrics:
                        seg += f" metrics={metrics}"
                # the vision judge's verdict, even when a later deterministic trial
                # became the final one (otherwise the visual reasoning is lost).
                if s.get("vision_verdict"):
                    seg += f" | {s['vision_verdict'][:200]}"
                parts.append(seg)
            status = v.get("status")
            if status and status != "satisfied":
                parts.append(f"overall={status}")
            if v.get("upstream_target"):
                parts.append(f"upstream_target={v['upstream_target']} "
                            f"(re-run this stage before continuing)")
            return "; ".join(parts)

        per_image = verification.get("per_image")
        if isinstance(per_image, list) and per_image:
            digests = [_digest_one(r) for r in per_image if isinstance(r, dict)]
            digests = [d for d in digests if d]
            if not digests:
                return ""
            if len(digests) == 1:
                return digests[0]
            return "; ".join(f"img{i}: {d}" for i, d in enumerate(digests))

        return _digest_one(verification)

    def _format_image_context_profile(self, result) -> str:
        """Task 057 M4: a dedicated digest line for a tool result's
        `ImageContextProfile`, so it reliably reaches the planner instead of riding
        `_format_memory`'s generic 500-char `result_str` truncation (which can cut it
        off on a busy multi-channel result -- see CODING_LOG.md). Reads no instance
        state (mirrors `_format_qc`), handles the flat and `per_image` batch result
        shapes, and returns "" when no profile is present. `provisional_target_hint`
        is rendered as TEXT ONLY for the LLM to weigh qualitatively -- never a
        deterministic branch (BRIEF M1: no runtime consumer in this task).
        """
        if not isinstance(result, dict):
            return ""
        profile = result.get('image_context_profile')
        if profile is None:
            per_image = result.get('per_image')
            if isinstance(per_image, list) and per_image and isinstance(per_image[0], dict):
                profile = per_image[0].get('image_context_profile')
        if not isinstance(profile, dict):
            return ""
        parts = [f"modality={profile.get('modality', 'unknown')}",
                f"specimen_context={profile.get('specimen_context', 'unknown')}",
                f"confidence={profile.get('confidence', 'low')}"]
        classes = profile.get('visible_object_classes') or []
        if classes:
            parts.append(f"visible_object_classes={classes}")
        hint = profile.get('provisional_target_hint')
        if hint and hint != 'unknown':
            parts.append(f"provisional_target_hint={hint} (advisory only, not enforced)")
        limitations = profile.get('limitations') or []
        if limitations:
            parts.append(f"limitations={limitations}")
        return "; ".join(parts)

    def _format_memory_for_prompt(self, memory: Memory, max_recent_actions: int = 10) -> str:
        """Alias for _format_memory (for compatibility)."""
        return self._format_memory(memory, max_recent_actions)
    
    def analyze_query(self, question: str, image: str, bytes_mode: bool = False,
                     conversation_context: str = "", group_images: List[Dict[str, Any]] = None,
                     **kwargs) -> str:
        """
        Analyze query to understand intent and requirements.
        Returns structured query analysis for display and planning.
        """
        image_info = self.get_image_info(image) if not bytes_mode else {}
        
        # Detect domain and filter tools
        self.detected_domain = self.priority_manager.detect_task_domain(question, "")
        available_tools, excluded_tools = self.priority_manager.filter_tools_for_domain(
            self.available_tools, self.detected_domain, exclude_excluded=True
        )
        if self.detected_domain == "bioimage":
            available_tools, modality_excluded = self.priority_manager.filter_segmentation_tools(
                available_tools, question, image_info=image_info
            )
            excluded_tools.extend(modality_excluded)
        
        # Multi-image and group context
        num_images = len(group_images) if group_images else 1
        groups = set()
        if group_images:
            groups = {img.get("group", "default") for img in group_images if img.get("group")}
        is_group_comparison = len(groups) > 1 if groups else False
        
        # Filter metadata
        toolbox_metadata = self._metadata_for(available_tools)
        
        # Build multi-image context
        multi_image_context = ""
        if num_images > 1:
            groups_summary = ", ".join(sorted(groups)) if groups else "default"
            multi_image_context = f"""
MULTI-IMAGE CONTEXT:
- Number of images: {num_images}
- Groups detected: {len(groups)} ({groups_summary})
- Group comparison: {'YES' if is_group_comparison else 'NO (all images in same group)'}

MULTI-IMAGE PROCESSING:
- Per-image tools: Process each image independently (segmenters, croppers)
- Merge-all tools: Execute once, merging all images with group labels (analyzers, visualizers)
- For comparison: Analysis tool extracts features -> Visualization tool displays results
"""
        
        # Enhanced prompt with structured output
        prompt = f"""Analyze this bioimage analysis query and determine the minimum necessary skills and tools.

Conversation so far:
{conversation_context}

Query: {question}
Image info: {image_info}
{multi_image_context}

Available tools: {available_tools}
Tool metadata: {toolbox_metadata}

CRITICAL INSTRUCTIONS:
1. Understand the query precisely - identify EXACTLY what is being asked
2. Determine the SINGLE main objective:
   - "How many cells/organoids" -> Only counting needed (segmentation + counting, may visualize counts)
   - "What cell states" or "analyze cell states" -> Full cell state analysis (image loading -> segmentation -> cropping -> analysis -> visualization)
   - "Compare" or "morphology" -> Full pipeline (analysis tool extracts features, visualizer displays)
   - Analysis_Visualizer_Tool requires pre-computed results from Cell_State_Analyzer_Tool
3. Select minimum necessary tools based on query requirements
4. Do not skip analysis tool for morphology/comparison queries
5. Use intelligent planning: Consider whether additional context (e.g., image type analysis) would help, but prioritize efficiency - if the task can be accomplished directly with specialized tools, use them directly

Provide your analysis in structured format:
- Concise Summary: Brief summary of what the query asks
- Query Type: One of [Simple Counting / Basic Morphology / Cell State Analysis / Comparison]
- Required Analysis Pipeline: Describe steps needed (e.g., "Image loading -> Segmentation -> Count extraction")
- Key Requirements: Specific requirements, constraints, considerations
- Expected Tools: Tools likely needed in order

Provide a clear, actionable analysis that helps plan the bioimage analysis workflow."""
        
        # Try structured format first, fallback to string
        try:
            response = self.llm_engine_mm.generate([prompt], response_format=QueryAnalysis)
            if isinstance(response, dict) and 'content' in response:
                self.query_analysis = response['content']
                self.last_usage = response.get('usage', {})
            else:
                self.query_analysis = str(response)
        except Exception as e:
            logger.debug(f"Structured format failed, using text: {e}")
            response = self.llm_engine.generate(prompt, max_tokens=500)
            if isinstance(response, dict) and 'content' in response:
                self.query_analysis = response['content']
            elif isinstance(response, str):
                self.query_analysis = response
            else:
                self.query_analysis = str(response)

        # 045 Phase 3 (Fix 0): derive the query's stage boundary ONCE, up front, from
        # the (structured) query type -- the planner then plans + stops within it, and
        # the plan-gate blocks out-of-scope tools. A str fallback has no query_type ->
        # fail-open (no boundary enforced).
        from shape.agent.verification.stage_map import task_spec_for
        self.task_spec = task_spec_for(getattr(self.query_analysis, "query_type", "") or "")

        return str(self.query_analysis)
    
    def generate_next_step(self, question: str, image: str, query_analysis: str,
                           memory: Memory, step_count: int, max_step_count: int,
                           bytes_mode: bool = False, conversation_context: str = "",
                           group_images: List[Dict[str, Any]] = None, **kwargs) -> NextStep:
        """
        Generate next step using intelligent LLM-based planning.
        
        Core approach:
        - Provide rich context (query, memory, tools, metadata)
        - Guide through prompts with bioimage analysis chain awareness
        - Trust LLM's intelligence for tool selection
        - Support multi-image and group comparison scenarios
        """
        # Ensure query_analysis is a string
        if not isinstance(query_analysis, str):
            query_analysis = str(query_analysis) if query_analysis else ""

        # Get context
        image_info = self.get_image_info(image) if not bytes_mode else {}
        detected_domain = self.priority_manager.detect_task_domain(question, query_analysis)
        available_tools, excluded_tools = self.priority_manager.filter_tools_for_domain(
            self.available_tools, detected_domain, exclude_excluded=True
        )
        if detected_domain == "bioimage":
            available_tools, modality_excluded = self.priority_manager.filter_segmentation_tools(
                available_tools, question, query_analysis, image_info
            )
            excluded_tools.extend(modality_excluded)
        
        # Get used tools
        actions = memory.get_actions(llm_safe=True)
        used_tools = [a.get('tool_name', '') for a in actions if 'tool_name' in a]

        # Filter metadata
        toolbox_metadata = self._metadata_for(available_tools)
        
        # Multi-image context
        num_images = len(group_images) if group_images else 1
        groups = set()
        if group_images:
            groups = {img.get("group", "default") for img in group_images if img.get("group")}
        is_group_comparison = len(groups) > 1 if groups else False
        
        # Build multi-image prompt section
        multi_image_prompt = ""
        if num_images > 1:
            groups_summary = ", ".join(sorted(groups)) if groups else "default"
            multi_image_prompt = f"""
MULTI-IMAGE PROCESSING CONTEXT:
- Processing {num_images} image(s) across {len(groups)} group(s): {groups_summary}
- Group comparison enabled: {'YES - tools should compare groups statistically' if is_group_comparison else 'NO'}
- Tool execution modes:
  * Per-image tools (segmenters, croppers): Process each of {num_images} images independently
  * Merge-all tools (Cell_State_Analyzer_Tool, Analysis_Visualizer_Tool): Execute ONCE, merging all crops from all {num_images} images with group labels
- Cell_State_Analyzer_Tool automatically loads all crop images from Single_Cell_Cropper_Tool metadata (from all {num_images} images)
"""
        
        # Get tools by priority
        tools_by_priority = self.priority_manager.format_tools_by_priority(available_tools)
        
        # Get recommended next tools
        recommended_tools = self.priority_manager.get_recommended_next_tools(
            available_tools, used_tools, detected_domain, question, query_analysis
        )
        recommended_str = ", ".join(recommended_tools[:5]) if recommended_tools else "None"

        # 045 P3 (Fix 0): surface the query's STAGE BOUNDARY so the planner selects
        # in-scope tools (the plan-gate scope guard is the deterministic backstop). Only
        # rendered when a boundary is actually enforced (a segment/count query); an
        # unclassified/full-pipeline query adds nothing (fail-open).
        scope_prompt = ""
        ts = getattr(self, "task_spec", None)
        if ts is not None and getattr(ts, "out_of_scope_stages", None):
            from shape.agent.verification.stage_map import STAGES
            in_tools = sorted({t for s in STAGES if s.stage in ts.in_scope_stages
                               for t in s.tool_names})
            out_tools = sorted({t for s in STAGES if s.stage in ts.out_of_scope_stages
                                for t in s.tool_names})
            scope_prompt = f"""
TASK BOUNDARY (this query's task is ANSWERED at the '{ts.required_terminal_stage}' stage):
- IN SCOPE -- select only these managed tools (plus any unmanaged utility tool): {in_tools}
- OUT OF SCOPE -- do NOT select these; they will be rejected by the plan gate: {out_tools}
Advance toward the terminal stage; do not run downstream tools beyond what the query asks.
"""

        # Build intelligent prompt with bioimage analysis chain awareness
        prompt = f"""You are an intelligent planner for a bioimage analysis system specializing in SINGLE-CELL level analysis.
Your goal: Select the optimal next tool to progress toward answering the query.

QUERY: {question}
QUERY ANALYSIS: {query_analysis}
IMAGE INFO: {image_info}
STEP: {step_count}/{max_step_count}
{multi_image_prompt}
{scope_prompt}
PREVIOUS STEPS:
{self._format_memory(memory)}

AVAILABLE TOOLS (grouped by category; choose by REASONING over each tool's when-to-use, not by position):
{tools_by_priority}

RECOMMENDED NEXT TOOLS: {recommended_str}

TOOL METADATA (includes dependencies and capabilities):
{self._format_tool_metadata(toolbox_metadata)}

CRITICAL: Bioimage Analysis Chain for Single-Cell Analysis

This system performs SINGLE-CELL level analysis. For queries involving cell states, cell counts, or cell-level comparisons:

**Tool Role & Pipeline:**
- Analysis Tools: Cell_State_Analyzer_Tool analyzes individual cell/organoid crops - REQUIRES Single_Cell_Cropper_Tool output
- Visualization Tool: Analysis_Visualizer_Tool ONLY visualizes pre-computed analysis results

**Pipeline Flow:**
1. Segmentation -> [Cell/Nuclei/Organoid_Segmenter_Tool] -> produces masks
2. Cropping -> Single_Cell_Cropper_Tool -> REQUIRED: produces individual crop images (one per cell/organoid)
3. Analysis -> Cell_State_Analyzer_Tool (auto-detects and validates the channel count)
   - Input: Individual crop images from step 2 (NOT original images or masks)
   - Processes each crop to extract features, perform clustering
4. Visualization -> Analysis_Visualizer_Tool

**CRITICAL Input Requirements:**
- Cell_State_Analyzer_Tool REQUIRES cell_crops: individual crop paths from Single_Cell_Cropper_Tool
- Input format: cell_crops=["path/to/crop1.tiff", "path/to/crop2.tiff", ...] (individual crop images)
- Single_Cell_Cropper_Tool MUST run first; all crops in one analyzer run must have the same channel count
- When Single_Cell_Cropper_Tool completes, it saves metadata in query_cache_dir/tool_cache/ - Cell_State_Analyzer_Tool automatically loads crops from this metadata

**Query -> Pipeline Mapping:**
- Counting: Segmentation -> Count (visualize if needed)
- Morphology/Comparison/States: Full pipeline (Segmentation -> Cropping -> Analysis -> Visualization)
- Analysis tools analyze individual crops - cannot skip cropping step
- A non-terminal stage being satisfied/PASS only means "proceed
  to the next required stage" -- it does NOT itself answer a segment/count/cell-state
  query. Keep advancing toward the query's REQUIRED TERMINAL stage (e.g. segmentation
  for a count/segment ask; the full pipeline's last required stage for a cell-state
  ask) rather than treating an earlier stage's completion as the finish line.

**Intelligent Planning Guidelines:**
- ESCALATION HANDLING: a `conceded` stage's QC falls into exactly two buckets. (1)
  NO `upstream_target` (`fault=ceiling` -- whether the tool exhausted its tunable
  parameters, or judged the tool/data itself unfit; both are now the SAME bucket) --
  the tool has already done all it can here; do NOT repeat it. Its best-effort
  output is carried forward, so proceed as you would after any completed step --
  unless its QC says "NO OUTPUT produced (<reason>)": then the stage produced
  nothing (e.g. its call failed; the stage's notes give the error), no output is
  carried forward, and a later step that needs this stage's output cannot use it.
  (2) `upstream_target=<stage>` (`fault=upstream`) -- the stage is blaming a
  NAMED prior stage's input. Treat this SKEPTICALLY by default: it is a bounded,
  VERIFICATION-oriented look-back, NOT a from-scratch re-run -- re-examine the named
  upstream stage in light of THIS downstream stage's specific complaint, using the
  upstream's prior output, only when there is a specific, evidence-backed reason it
  can do better for this complaint (both stages already optimized to their own best,
  so a blind re-run rarely helps). Strictly bounded to avoid a vicious cycle -- at
  MOST once per upstream stage: check PREVIOUS STEPS for whether the named upstream
  tool already ran again since this concession. If it has, do NOT re-run it again --
  proceed as a completed step instead of looping.
- For counting queries: Segmentation tools (Cell_Segmenter_Tool, Organoid_Segmenter_Tool) are designed to handle counting directly. Consider whether additional image analysis would add value or if direct segmentation is more efficient.
- For ambiguous image types: Segmentation tools have built-in detection capabilities. Consider whether explicit image analysis would significantly improve results or if direct segmentation with auto-detection is sufficient.
- Tool selection should balance: (1) Task efficiency - can the task be accomplished directly? (2) Result quality - would additional context improve outcomes? (3) User experience - minimize unnecessary steps while ensuring accuracy.

TASK DOMAIN AWARENESS:

**Knowledge Domain (Non-Image Analysis):**
The system has detected this as a knowledge-based task (literature mining, gene annotation, pathway enrichment, functional analysis, etc.).

**CRITICAL: For cell cluster functional annotation tasks, use the specialized tool:**
- **If the query involves cell clusters with regulator genes, functional state annotation, or cluster classification:**
  - **USE Cell_Cluster_Functional_Hypothesis_Tool FIRST** - This specialized tool internally handles:
    * Literature search (PubMed) for all genes in batch
    * Evidence collection and evaluation
    * Functional hypothesis generation with confidence scores
    * Structured output with citations
  - **DO NOT manually call Pubmed_Searcher_Tool + Generalist_Solution_Generator_Tool** - The specialized tool replaces this workflow
  - Input format: clusters=[{{"cluster_id": "...", "regulator_genes": ["GENE1", "GENE2"], "tissue_context": "...", "disease_context": "...", "target_lineage": "..."}}]

**For other knowledge tasks (general literature mining, gene function lookup, etc.):**
- **Use search tools FIRST** to retrieve real knowledge from databases:
  - For literature mining: Use Pubmed_Searcher_Tool to search PubMed for relevant articles
  - For general knowledge: Use Google_Searcher_Tool to find current information
  - For academic papers: Use ArXiv_Paper_Searcher_Tool for preprints
  - These tools provide REAL, VERIFIABLE information with actual citations
- **Then use Generalist_Solution_Generator_Tool** to synthesize the search results into comprehensive answers
- Workflow: Search tools (Pubmed_Searcher_Tool, Google_Searcher_Tool) -> Generalist_Solution_Generator_Tool (synthesis with references)

**Bioimage Domain (Image Analysis):**
For image analysis tasks, follow the bioimage analysis chain below.

INSTRUCTIONS:
1. Review the query analysis and detected domain to understand what type of analysis is needed
2. For knowledge domain: 
   - **If cluster annotation task (regulator genes, functional state, cluster classification):**
     * USE Cell_Cluster_Functional_Hypothesis_Tool directly - it handles search + synthesis internally
     * DO NOT manually call Pubmed_Searcher_Tool first
   - **For other knowledge tasks:**
     * FIRST: Use search tools (Pubmed_Searcher_Tool, Google_Searcher_Tool, ArXiv_Paper_Searcher_Tool) to retrieve REAL knowledge
     * THEN: Use Generalist_Solution_Generator_Tool to synthesize search results with proper citations
     * DO NOT use Generalist_Solution_Generator_Tool alone - it should synthesize actual search results
3. For bioimage domain: Check what has been done (PREVIOUS STEPS) and what is still needed
4. For bioimage domain: Follow the bioimage analysis chain appropriate for the query type
5. Use intelligent reasoning: Evaluate whether each step adds value or if a more direct approach would be better
6. Select ONE tool that is the logical next step for the identified domain
7. Ensure dependencies are satisfied (e.g., need segmentation before cropping, need cropping before analysis)
8. For Cell_State_Analyzer_Tool: Do NOT pass original images or masks - tool automatically loads crop images from Single_Cell_Cropper_Tool metadata (query_cache_dir parameter)
9. Formulate a clear sub-goal explaining what this tool will accomplish, referencing the correct input format

Output format:
<justification>: Why this tool is the best next step, referencing the analysis pipeline
<context>: All necessary information from previous steps (file paths, data, variables)
<sub_goal>: Specific, achievable objective for this tool
<tool_name>: Exact tool name from available tools list"""
        
        # Generate next step
        try:
            next_step = self.llm_engine.generate(prompt, response_format=NextStep)
        except Exception as e:
            logger.warning(f"Structured format failed, using text: {e}")
            # Fallback to text generation
            response = self.llm_engine.generate(prompt, max_tokens=500)
            # Parse manually if needed
            next_step = response
        
        # Extract and store usage
        if isinstance(next_step, dict) and 'content' in next_step:
            result = next_step['content']
            self.last_usage = next_step.get('usage', {})
        else:
            result = next_step
            self.last_usage = {}

        # Structured output validates the field shape, not membership in the
        # registered toolbox. Normalize product names such as "Cellpose-SAM" to
        # the modality-filtered SHAPE tool and mark unknown names as invalid.
        from shape.utils.agent_io import normalize_tool_name
        if hasattr(result, "tool_name"):
            result.tool_name = normalize_tool_name(result.tool_name, available_tools)
        elif isinstance(result, dict) and result.get("tool_name"):
            result["tool_name"] = normalize_tool_name(result["tool_name"], available_tools)
            # Some structured backends (e.g. Claude thinking models via forced tool_use)
            # hand back the tool input as a raw dict instead of the parsed NextStep. Convert
            # it so the caller always receives a NextStep object (it does `next_step.tool_name`).
            try:
                result = NextStep(**{k: result.get(k, "") for k in NextStep.model_fields})
            except Exception:
                pass
        elif isinstance(result, str):
            context, sub_goal, tool_name = ResponseParser.parse_next_step(
                result, available_tools
            )
            result = NextStep(
                justification="Parsed from text planner response",
                context=context,
                sub_goal=sub_goal,
                tool_name=tool_name,
            )
        
        # J.4=B (LLM as the sole forward-planner): the post-LLM `required_next` override
        # was removed. The planner no longer rewrites the LLM's tool choice to force a
        # tool's declared next step -- the LLM owns forward selection (guided by the
        # pipeline prompt + card-derived priority hints), and the Verifier's plan gate
        # (active in _run_steps) blocks a structurally unsound plan (e.g. the analyzer
        # planned before the cropper) instead of silently overriding it.

        # VLRR: recommend initial parameters for managed pipeline tools. Carried on
        # the planner instance (NOT on NextStep) so the LLM response schema stays
        # free of free-form dicts that OpenAI strict structured outputs reject.
        # The main loop reads planner.recommended_params after this call.
        self.recommended_params = None
        try:
            from shape.agent.verification.stage_map import is_managed_tool, default_params_for_tool
            tname = getattr(result, 'tool_name', None) if hasattr(result, 'tool_name') \
                else (result.get('tool_name') if isinstance(result, dict) else None)
            if tname and is_managed_tool(tname):
                # Start from the tool defaults. The segmenter diameter is NOT guessed
                # pre-segmentation (gradient guesses underestimate large cells ~4x);
                # it is measured from the first segmentation's masks downstream.
                self.recommended_params = default_params_for_tool(tname)
        except Exception as e:
            logger.debug(f"VLRR initial-params recommendation skipped: {e}")

        return result

    def _format_tools_list(self, available_tools: List[str], used_tools: List[str]) -> str:
        """Format tools list with usage indicators."""
        used_set = set(used_tools)
        lines = []
        for tool in available_tools:
            status = "Used" if tool in used_set else "Available"
            lines.append(f"  - {tool} ({status})")
        return "\n".join(lines) if lines else "No tools available"
    
    def _metadata_for(self, tools: List[str]) -> Dict[str, Any]:
        """Return metadata for `tools`, loading it lazily for only these tools.

        Metadata is read from the tool class source by the initializer without
        instantiating the tool, and only for the (already domain/modality-filtered)
        tools the planner is about to present -- never the whole toolbox.
        """
        init = getattr(self, "initializer", None)
        if init is not None:
            for t in tools:
                if self.toolbox_metadata.get(t) is None:
                    try:
                        md = init.get_tool_metadata(t)
                        if md:
                            self.toolbox_metadata[t] = md
                    except Exception as e:
                        logger.debug(f"Lazy metadata load failed for {t}: {e}")
        return {t: self.toolbox_metadata[t] for t in tools if self.toolbox_metadata.get(t)}

    def _format_tool_metadata(self, toolbox_metadata: Dict[str, Any]) -> str:
        """Format tool metadata concisely."""
        if not toolbox_metadata:
            return "No tool metadata available"
        
        lines = []
        for tool_name, metadata in list(toolbox_metadata.items())[:10]:
            desc = metadata.get(
                'tool_description',
                metadata.get('description', 'No description')
            )[:200]
            # Emphasize critical role distinctions and input requirements
            if 'Cell_State_Analyzer' in tool_name:
                lines.append(f"{tool_name}: {desc} [REQUIRES: cell crops from Single_Cell_Cropper_Tool]")
            elif tool_name == "Analysis_Visualizer_Tool":
                lines.append(f"{tool_name}: {desc} [VISUALIZATION ONLY: requires pre-computed analysis results]")
            else:
                lines.append(f"{tool_name}: {desc}")
        
        if len(toolbox_metadata) > 10:
            lines.append(f"... and {len(toolbox_metadata) - 10} more tools")
        
        return "\n".join(lines)

    def _reinvocation_evidence(self, memory) -> str:
        """Mechanistic completion evidence -- the PLANNER-level analog of the tool VRR
        loop's own convergence stops (`step_verifier.py`: the `patience` plateau counter
        `stall>=patience`, the param-repeat `oscillation_stop`, the reconcile
        `hypothesis_invalidated`). Those stop the WITHIN-stage retune loop once re-tuning
        stops improving; the planner needs the ACROSS-invocation analog -- how many times
        it has RE-INVOKED a stage and how many of those reproduced a tuning ceiling. A
        stage re-invoked into the same ceiling concede is mechanistically AT ITS LIMIT
        (the tool already exhausted its levers each time -- see the `ledger`), so
        re-invoking is very unlikely to improve it. Derived from what actually happened
        (a count of real repeated concedes), NOT a fixed 'max N tries' cap -- and
        surfaced as EVIDENCE the completion LLM weighs, never a hard stop.
        """
        from collections import OrderedDict
        faults_by_stage: "OrderedDict[str, list]" = OrderedDict()
        for a in memory.get_actions(llm_safe=True):
            v = a.get("verification")
            if not isinstance(v, dict):
                continue
            records = v.get("per_image") if isinstance(v.get("per_image"), list) else [v]
            for rec in records:
                if not isinstance(rec, dict):
                    continue
                for s in (rec.get("stages") or []):
                    st = s.get("stage")
                    if st:
                        faults_by_stage.setdefault(st, []).append(s.get("fault"))
        lines = []
        for st, faults in faults_by_stage.items():
            n = len(faults)
            ceilings = sum(1 for f in faults if f == "ceiling")
            if n >= 2 and ceilings >= 2:
                lines.append(
                    f"- '{st}' has been re-invoked {n}x and reproduced a tuning ceiling "
                    f"{ceilings}x -- re-invoking has mechanistically hit the same limit each "
                    f"time (its levers were already exhausted, per the `ledger`); a further "
                    f"re-invocation is very unlikely to change the result.")
        if not lines:
            return ""
        return ("\nRE-INVOCATION PROGRESS (mechanistic evidence to weigh, NOT a cap):\n"
                + "\n".join(lines))

    def verificate_memory(self, question: str, image: str, query_analysis: str,
                         memory: Memory, bytes_mode: bool = False,
                         conversation_context: str = "",
                         **kwargs) -> MemoryVerification:
        """Verify if memory contains sufficient information to answer query.

        Text-only (task 045 vision-authority consolidation): no image bytes are
        attached, routed to the TEXT engine (`self.llm_engine`, not `llm_engine_mm`).
        Its job is "given the recorded structured verdicts (the `QC:` /
        `vision_verdict` digest already in `_format_memory`'s prompt), is the query
        answered?" -- it must NOT re-derive mask quality by looking at the overlay
        itself; that is the VLRR result gate's job, already forwarded here as text.
        """
        # Ensure query_analysis is a string
        if not isinstance(query_analysis, str):
            query_analysis = str(query_analysis) if query_analysis else ""
        
        # Check for termination signals from tools BEFORE calling LLM
        # Intelligent approach: Tools declare their dependencies via required_next_tools
        # Only allow termination after all required_next_tools are completed
        actions = memory.get_actions(llm_safe=False)

        def action_succeeded(action):
            result = action.get("result")
            if result is None:
                return False
            if isinstance(result, dict):
                return not result.get("error") and result.get("status") != "failed"
            if isinstance(result, str):
                text = result.strip().lower()
                return bool(text) and not text.startswith(("error", "failed"))
            return True

        used_tools = [
            action.get("tool_name", "")
            for action in actions
            if action.get("tool_name") and action_succeeded(action)
        ]

        # Empty memory can never satisfy an image-analysis query. Avoid asking
        # the LLM to judge an impossible stop condition before any tool runs.
        if not actions:
            return MemoryVerification(
                analysis="No tool has executed yet; analysis must continue.",
                stop_signal=False,
            )

        # 045 P3 Fix 1: tool self-signals are EVIDENCE the planner weighs, NOT
        # deterministic overrides that bypass it. Previously an unmet required_next_tools
        # SUPPRESSED the stop-check (forced continue) and a can_terminate_after_chain
        # flag RETURNED stop_signal=True (forced stop) -- two competing STOP authorities
        # that pre-empted the LLM. Now both are collected and surfaced in the completion
        # prompt below; the completion LLM (the single stop authority) decides, weighing
        # them like any other evidence (fault=, ledger, the required terminal stage).
        tool_signals = []
        for action in actions:
            result = action.get('result', {})
            if not isinstance(result, dict):
                continue
            tname = action.get('tool_name', 'a tool')
            unmet = [t for t in (result.get('required_next_tools') or [])
                     if t not in used_tools]
            if unmet:
                tool_signals.append(
                    f"- '{tname}' declares a required downstream step not yet run: "
                    f"{', '.join(unmet)} -- its pipeline is not complete until that runs "
                    f"(continue toward it, unless it is out of scope for this query).")
            if result.get('can_terminate_after_chain') or result.get('termination_recommended'):
                reason = (result.get('termination_reason') or result.get('summary') or "").strip()
                tool_signals.append(
                    f"- '{tname}' signals a task LIMITATION"
                    + (f": {reason}" if reason else "")
                    + " -- the data/config may not fully answer the query; consider "
                    "stopping and reporting the limitation plainly.")
        tool_signal_block = (
            "\nTOOL SELF-SIGNALS (evidence to weigh, NOT commands):\n" + "\n".join(tool_signals)
            if tool_signals else "")

        # 045 P3: mechanistic re-invocation-progress evidence (planner-level analog of the
        # tool loop's plateau/oscillation stop) -- surfaces when a stage keeps being
        # re-invoked into the same ceiling, so the LLM can weigh "re-invoking is futile".
        reinvocation_block = self._reinvocation_evidence(memory)

        image_info = self.get_image_info(image) if not bytes_mode else {}
        
        # Check detected domain for knowledge tasks
        detected_domain = getattr(self, 'detected_domain', 'general')
        is_knowledge_domain = (detected_domain == 'knowledge')

        # 045 P3 Fix 3: the query's REQUIRED TERMINAL stage, DERIVED from the TaskSpec --
        # replaces the hard-coded per-tool completion rules (no Analysis_Visualizer /
        # Cell_State_Analyzer names baked into the general prompt). Concrete stage + the
        # tool(s) that produce it, so the LLM can check "has it produced output?" in the
        # Previous Steps. Falls back to a general instruction when unclassified.
        ts = getattr(self, "task_spec", None)
        if ts is not None and getattr(ts, "required_terminal_stage", None):
            from shape.agent.verification.stage_map import stage_by_name
            _spec = stage_by_name(ts.required_terminal_stage)
            _producers = (" / ".join(_spec.tool_names) if _spec and _spec.tool_names
                          else ts.required_terminal_stage)
            terminal_block = (
                f"THIS QUERY'S REQUIRED TERMINAL STAGE: '{ts.required_terminal_stage}' "
                f"(produced by: {_producers}). The query is ANSWERED once this stage has "
                f"produced usable output -- STOP then. A stage BEFORE it being satisfied/"
                f"PASS is a reason to CONTINUE toward it, never to stop by itself. Do NOT "
                f"run stages beyond it -- they are out of scope for this query.")
        else:
            terminal_block = (
                "Identify the query's REQUIRED TERMINAL stage -- the stage whose OWN output "
                "answers the ask (segmentation for a segment/count ask; the analysis "
                "pipeline's last stage for a cell-state ask). STOP once it has produced "
                "usable output; a non-terminal stage's completion is a reason to CONTINUE "
                "toward it, not to stop.")

        prompt = f"""Verify if the following steps contain sufficient information to answer the query.

Conversation so far:
{conversation_context}

Query: {question}
Query Analysis: {query_analysis}
Image: {image_info}
Detected Domain: {detected_domain}

Previous Steps:
{self._format_memory(memory)}

{"**KNOWLEDGE DOMAIN TASK:**" if is_knowledge_domain else ""}
{"For knowledge-based tasks (literature mining, gene annotation, pathway enrichment, functional analysis):" if is_knowledge_domain else ""}
{"- CRITICAL: Knowledge tasks require a TWO-STEP workflow:" if is_knowledge_domain else ""}
{"  1. Search tools (Pubmed_Searcher_Tool, Google_Searcher_Tool) retrieve REAL knowledge from databases" if is_knowledge_domain else ""}
{"  2. Generalist_Solution_Generator_Tool synthesizes search results into comprehensive answers with citations" if is_knowledge_domain else ""}
{"- DO NOT stop after only search tools - synthesis step is REQUIRED" if is_knowledge_domain else ""}
{"- For literature mining: Must have BOTH (1) search results from Pubmed_Searcher_Tool AND (2) synthesis from Generalist_Solution_Generator_Tool" if is_knowledge_domain else ""}
{"- For gene annotation: Must have BOTH (1) search results AND (2) synthesized annotations with references" if is_knowledge_domain else ""}
{"- STOP only when BOTH search AND synthesis steps are complete, with comprehensive answers including references" if is_knowledge_domain else ""}
{"- If only search tools have been used, CONTINUE to use Generalist_Solution_Generator_Tool for synthesis" if is_knowledge_domain else ""}

CRITICAL: Verify ONLY what the query explicitly asks -- do NOT require information
beyond it, and do NOT treat running a tool beyond the answering stage as needed.
{terminal_block}
{tool_signal_block}
{reinvocation_block}

CRITICAL: Each step may carry a `QC:` line with the VLRR verdict, including a
`fault=` tag (ceiling | upstream) whenever a stage conceded. Division of
labor: a JUDGE has already assessed this stage's output quality -- the VRR verify
step's vision-LLM judge (backed by its deterministic quant) inspected the result
and issued this verdict, finished before you ever see this digest. YOUR job is a
DIFFERENT judgment: given the judge's verdict as EVIDENCE, is this stage as done
as it is going to get, and would re-invoking it plausibly change the result?
Weigh the evidence below -- don't just mechanically obey it, and don't re-judge
the mask quality yourself (that would duplicate the judge's job):
- `satisfied` -- the judge's quality check passed; weigh this stage as done. But a
  satisfied NON-TERMINAL stage (e.g. image_loading=PASS) only means "proceed to the
  next required stage" -- it does NOT itself answer a segment/count/cell-state
  query. Only STOP once the query's REQUIRED TERMINAL stage (the stage whose own
  output actually answers the ask -- e.g. segmentation for a "segment/count cells"
  ask, the analysis pipeline's terminal stage for a "cell states" ask) has itself
  produced output.
- `conceded` with NO `upstream_target` (`fault=ceiling`) -- the tool/judge has
  already done all it can here, whether it exhausted every parameter it can
  retune (a tuning ceiling) or judged the tool/data itself unfit for the job --
  either way, re-invoking the SAME stage is evidence-backed to reproduce the SAME
  concession, not a better one (the same case the ESCALATION HANDLING guidance for
  tool selection already treats as a completed step, never re-run). Accept the
  best-effort output (when its QC says "NO OUTPUT produced", there is none: say
  so) and STOP, reporting the reason plainly from the stage's
  `notes` (e.g. "best-effort result on a difficult image" for a tuning ceiling, or
  the specific unfit-tool/no-usable-data reason the notes give) rather than
  looping toward a better result the tool has already shown it cannot produce.
  The stage's `ledger=` in the QC digest is concrete EVIDENCE for this: levers marked
  `inert` (no measurable effect) or `no-headroom` (exhausted) mean re-running the SAME
  tool reproduces the concession -- weigh that strongly toward STOP. (Only a ledger
  showing an `effective` lever the loop did not yet fully exploit is a reason one
  further in-scope try might help.) This is evidence to weigh, not a rule to obey --
  you still decide, against what the query actually needs.
- `conceded fault=upstream` (carries `upstream_target=<stage>`) -- the tool is
  blaming its INPUT, not itself. Treat this SKEPTICALLY by default: it is a
  bounded, VERIFICATION-oriented look-back, NOT a from-scratch re-run --
  re-examine the NAMED upstream stage in light of THIS downstream stage's
  specific complaint, using the upstream's prior output, to check whether a
  targeted upstream adjustment would plausibly help -- not "redo the upstream
  from zero." Both stages already optimized to their own best, so a blind re-run
  rarely improves anything: pursue it only when there is a specific,
  evidence-backed reason the upstream stage can do better for THIS complaint.
  Strictly bounded to avoid a vicious cycle -- at MOST once per upstream stage:
  check PREVIOUS STEPS for whether the named upstream stage already ran again
  since this concession. If it has, and the verdict is still the same conceded
  outcome, do NOT loop again -- accept the best-effort output and STOP. Never
  chain upstream re-runs or bounce between stages.
The query's REQUIRED TERMINAL stage must itself have produced output before you
STOP -- a required stage that has not been attempted at all (no verdict recorded
yet), or only a non-terminal stage's completion, is NOT enough; that is the one
case where continuing toward the terminal stage is actually needed, regardless
of the above.

CRITICAL CHECKLIST FOR STOPPING:
- Has the MAIN query been answered EXACTLY as asked, by its REQUIRED TERMINAL
  stage's own output? If yes, STOP immediately. A non-terminal stage being
  satisfied/PASS is a reason to CONTINUE toward that terminal stage, never a
  reason to stop by itself.
- Are there results (a count / mask / analysis / visualization) from the REQUIRED
  TERMINAL stage that DIRECTLY answer the query? If yes, STOP.
- Concession is EVIDENCE, not a mandate: a `fault=ceiling` stage is done -- STOP with
  its best-effort output and an honest note from the stage's `notes`; only
  `fault=upstream` is worth ONE bounded look-back (see above). Concession is NEVER a
  reason to run tools beyond the query's scope.
- DO NOT continue just because there are unused tools available.
- DO NOT re-run tools that have already completed their workflow.

Determine if the query can be answered with the information available, or if more steps are needed."""
        
        # Try structured format first
        try:
            # Text-only: no input image, no output figures -- the completion
            # judgment reads the structured QC/vision_verdict digest already in
            # `prompt` (via _format_memory), not the pixels themselves.
            response = self.llm_engine.generate(prompt, response_format=MemoryVerification)
            if isinstance(response, dict) and 'content' in response:
                result = response['content']
                self.last_usage = response.get('usage', {})
            else:
                result = response
                self.last_usage = {}
            
            # Handle string response
            if isinstance(result, str):
                analysis, stop_signal = ResponseParser.parse_memory_verification(result)
                result = MemoryVerification(analysis=analysis, stop_signal=stop_signal)
            
            return result
        except Exception as e:
            logger.warning(f"Structured format failed, using text: {e}")
            # Fallback
            response = self.llm_engine.generate(prompt, max_tokens=500)
            if isinstance(response, dict) and 'content' in response:
                content = response['content']
            else:
                content = str(response)
            
            # Parse manually
            analysis, stop_signal = ResponseParser.parse_memory_verification(content)
            return MemoryVerification(analysis=analysis, stop_signal=stop_signal)
    
    def generate_final_output(self, question: str, image: str, memory: Memory,
                             bytes_mode: bool = False, conversation_context: str = "",
                             text_only: bool = False,
                             **kwargs) -> str:
        """Generate final answer from memory.

        `text_only` (default False, base-identical when unset): the caller
        (solver.py, `solve()`) precomputes this from the `_analysis_stage_ran`
        predicate -- True whenever no analysis stage has run yet (a segmentation-only
        session). When True: no image bytes are attached (neither the raw input nor
        the output figures), the call routes to the TEXT engine (`self.llm_engine`,
        not `llm_engine_mm`), and the prompt's visual-inspection instruction (#11) is
        dropped -- the answer is written from the structured memory (count/mask/`QC:`
        digest) only, and must not claim to have inspected a figure it was never
        shown (task 045 vision-authority consolidation, referee Cond. C-2).
        """
        image_info = self.get_image_info(image) if not bytes_mode else {}

        visual_evidence_item = "" if text_only else """
11. VISUAL EVIDENCE: the attached images are the raw input followed by the
    pipeline's OUTPUT figures (segmentation overlay, UMAP, cluster composition).
    Inspect them, not just the text: confirm cells are actually segmented, clusters
    are actually separated, and group structure is real before claiming so. If a
    figure contradicts a metric or the written claim, say so and lower confidence."""

        prompt = f"""Based on the following analysis steps, provide a comprehensive answer to the query.

Conversation so far:
{conversation_context}

Query: {question}
Image: {image_info}

Analysis Steps:
{self._format_memory(memory, max_actions=20)}

Instructions:
1. Review the query and all actions taken
2. Consider results from each tool execution
3. Incorporate relevant information from memory
4. Generate a well-organized, step-by-step final output
5. CRITICAL: The Summary section always provide a summary even if results are incomplete
6. Report only operations and measurements explicitly present in Analysis Steps
7. Never claim segmentation, cropping, embeddings, clustering, statistics, or
   biological findings unless a successful tool result contains that evidence
8. If execution failed or required steps are missing, state that the analysis
   is incomplete and list the exact failed or missing steps
9. CREDIBILITY REVIEW (do this BEFORE writing any conclusion): read the `QC:` line
   under each step. If a stage shows decision WARNING/FAIL, `action=conceded` (an
   upstream fault, or a tuning ceiling), an overall status
   other than satisfied, or weak QC metrics (e.g. negative group/batch silhouette,
   replicate_consistency near 0, high flow_error, low valid_crop_rate), you MUST
   surface it and temper the matching finding. Do NOT present a conceded,
   low-confidence, or QC-flagged result as a confident finding.
10. Judge whether the results actually answer the biological question, and label
    each finding's confidence (high / medium / low) based on its QC evidence.{visual_evidence_item}

Output Structure:
1. Summary: Brief overview of query and main findings
   - If results are complete: summarize the key findings
   - If results are incomplete: summarize what was found and what is missing
   - Always include morphological features if available (area, perimeter, circularity, etc.)
2. Detailed Analysis: Step-by-step process breakdown
3. Key Findings: Most important discoveries
   - Include morphological measurements if available
   - Include statistical comparisons if performed
4. Answer to Query: Direct, clear answer to original question
5. Additional Insights: Relevant information beyond direct answer
6. Reliability & Caveats: per-stage QC status (from the `QC:` lines), any WARNING /
   conceded stage / weak metric, and the confidence level of each finding. If a
   stage conceded or failed QC, say so plainly and do not over-claim.
7. Conclusion: Summary and potential next steps

CRITICAL: The Summary section MUST NOT be empty. Even if results are incomplete, provide a summary of:
- What analysis was performed
- What morphological features were extracted (if any)
- What statistical comparisons were made (if any)
- What limitations exist (if any)

Provide a clear answer to the query based on the analysis results."""

        if text_only:
            # [045 scope C] no image bytes -- the answer is written from the
            # structured memory (already embedded in `prompt` via `_format_memory`)
            # only, on the TEXT engine.
            response = self.llm_engine.generate(prompt)
        else:
            input_data = [prompt]
            if image_info and 'image_path' in image_info:
                try:
                    with open(image_info["image_path"], 'rb') as f:
                        input_data.append(f.read())
                except Exception:
                    pass
            # Attach the pipeline's output figures so the model can visually check the
            # results it is about to report on (overlay / UMAP / cluster plots).
            _append_figure_bytes(input_data, memory)

            response = self.llm_engine_mm.generate(input_data)
        if isinstance(response, dict) and 'content' in response:
            return response['content']
        elif isinstance(response, str):
            return response
        else:
            return str(response)
    
    def generate_base_response(self, question: str, image: str, max_tokens: str = "4000",
                              bytes_mode: bool = False) -> str:
        """Generate base response (for compatibility)."""
        prompt = f"Answer this question about the image: {question}"
        response = self.llm_engine.generate(prompt, max_tokens=int(max_tokens))
        if isinstance(response, dict) and 'content' in response:
            self.base_response = response['content']
        elif isinstance(response, str):
            self.base_response = response
        else:
            self.base_response = str(response)
        return self.base_response
    
    def extract_context_subgoal_and_tool(self, response) -> Tuple[str, str, str]:
        """Extract context, sub_goal, and tool_name from LLM response."""
        try:
            context, sub_goal, tool_name = ResponseParser.parse_next_step(response, self.available_tools)
            logger.debug(f"Extracted: context='{context[:50]}...', sub_goal='{sub_goal[:50]}...', tool_name='{tool_name}'")
            return context, sub_goal, tool_name
        except Exception as e:
            logger.error(f"Error in extract_context_subgoal_and_tool: {e}")
            return "", "", "Error extracting tool name"
    
    def extract_conclusion(self, response: MemoryVerification) -> Tuple[str, str]:
        """Extract context_verification and conclusion from verification response."""
        try:
            if hasattr(ResponseParser, 'parse_memory_verification'):
                analysis, stop_signal = ResponseParser.parse_memory_verification(response)
                conclusion = 'STOP' if stop_signal else 'CONTINUE'
                return analysis, conclusion
        except Exception:
            pass
        
        # Fallback: extract from response object
        if isinstance(response, dict):
            context_verification = response.get('context_verification', response.get('analysis', ''))
            conclusion = response.get('conclusion', response.get('stop_signal', 'CONTINUE'))
        elif hasattr(response, 'context_verification') and hasattr(response, 'conclusion'):
            context_verification = response.context_verification
            conclusion = response.conclusion
        elif hasattr(response, 'analysis') and hasattr(response, 'stop_signal'):
            context_verification = response.analysis
            conclusion = 'STOP' if response.stop_signal else 'CONTINUE'
        else:
            context_verification = str(response) if response else ''
            conclusion = 'CONTINUE'
        return context_verification, conclusion
    
    def generate_direct_output(self, question: str, image: str, memory: Memory,
                              bytes_mode: bool = False, conversation_context: str = "",
                              **kwargs) -> str:
        """Generate direct output (alias for generate_final_output)."""
        return self.generate_final_output(question, image, memory, bytes_mode, conversation_context, **kwargs)

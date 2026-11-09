from typing import Dict, Any, List, Union, Optional
import os
import uuid
from shape.utils.agent_io import sanitize_tool_output_for_llm, get_llm_safe_result, sanitize_paths_in_dict


def compact_verification(v: Any) -> Optional[Dict[str, Any]]:
    """LLM-safe QC digest of a step's VLRR verification record.

    The full record carries per-trial history and file paths; the reasoning layer
    only needs the verdict signal: per-stage decision/action/status/fault/notes plus
    the delivered trial's numeric metrics (wave mode: the selected trial, the input when
    none is; otherwise the last trial), and the overall status/upstream target. Returns
    None when there is nothing to surface.

    Also handles the batch/multi-image shape `{"tool":.., "per_image": [vrecord, ...]}`
    (task 047) -- each `vrecord` is a flat `{"status":.., "stages":[...]}` record, same
    as the single-image shape. A single-record batch (n=1, the common case) compacts to
    the SAME output as the equivalent flat record; N>1 preserves a `per_image` list of
    the per-record compacted digests (`_format_qc` renders it per image).
    """
    if not isinstance(v, dict):
        return None

    def _delivered_metrics(stage: Dict[str, Any]) -> Dict[str, float]:
        # wave mode records the delivered trial (`details["selected"]`, None = no call
        # delivered: the input, trial 0); the agent's last call is not necessarily it
        trials = stage.get("trials") or []
        details = stage.get("details") or {}
        index = len(trials) - 1
        if "selected" in details:
            index = details["selected"] if isinstance(details["selected"], int) else 0
        if 0 <= index < len(trials) and isinstance(trials[index], dict):
            m = trials[index].get("metrics") or {}
            return {k: round(float(x), 4) for k, x in m.items()
                    if isinstance(x, (int, float)) and not isinstance(x, bool)}
        return {}

    def _vision_verdict(stage: Dict[str, Any]) -> Optional[str]:
        """The vision judge is consulted on the FIRST trial, but a later
        deterministic trial often becomes the final verdict, burying the visual
        judgement in trial history. Pull it forward so the reasoning layer sees it."""
        for tr in (stage.get("trials") or []):
            note = tr.get("notes") if isinstance(tr, dict) else None
            if isinstance(note, str) and note.startswith("[vision]"):
                return note
        return None

    def _compact_one(entry: Dict[str, Any]) -> Optional[Dict[str, Any]]:
        stages = []
        for s in (entry.get("stages") or []):
            if not isinstance(s, dict):
                continue
            stage_entry = {
                "stage": s.get("stage"),
                "decision": s.get("decision"),
                "action": s.get("action"),
                "status": s.get("status"),
                "fault": s.get("fault"),
                "notes": s.get("notes"),
                "metrics": _delivered_metrics(s),
            }
            # 045 P3 (Fix-4 close-out): keep the lever ledger through compaction so it
            # reaches the reasoning digest (`_format_qc`). Without this the ledger lived
            # only in the durable trace; the completion/next-step prompts never saw it.
            # It is EVIDENCE (which retunes were inert/exhausted) the planner weighs.
            if s.get("no_output"):           # the stage delivered nothing: say so, and why
                stage_entry["no_output"] = s.get("reason") or "no output"
            if s.get("kept_input"):          # the stage deliberately delivered its input unchanged: why
                stage_entry["kept_input"] = s.get("kept_input")
            if s.get("ledger"):
                stage_entry["ledger"] = s.get("ledger")
            vision = _vision_verdict(s)
            if vision:
                stage_entry["vision_verdict"] = vision
            stages.append(stage_entry)
        if not stages and not entry.get("status"):
            return None
        out: Dict[str, Any] = {"status": entry.get("status"), "stages": stages}
        if entry.get("upstream_target"):
            out["upstream_target"] = entry["upstream_target"]
        return out

    per_image = v.get("per_image")
    if isinstance(per_image, list) and per_image:
        compacted = [_compact_one(e) for e in per_image if isinstance(e, dict)]
        compacted = [c for c in compacted if c is not None]
        if not compacted:
            return None
        return compacted[0] if len(compacted) == 1 else {"per_image": compacted}

    return _compact_one(v)

class Memory:
    # TODO Need to fix this to support multiple data sources (e.g. images, pdf, txt, etc.)
    
    def __init__(self):
        self.query: Optional[str] = None
        self.files: List[Dict[str, str]] = []
        # Use list instead of dict to avoid key collisions when step_count resets
        self.actions: List[Dict[str, Any]] = []
        self._init_file_types()

    def set_query(self, query: str) -> None:
        if not isinstance(query, str):
            raise TypeError("Query must be a string")
        self.query = query

    def _init_file_types(self):
        self.file_types = {
            'image': ['.jpg', '.jpeg', '.png', '.gif', '.bmp'],
            'text': ['.txt', '.md'],
            'document': ['.pdf', '.doc', '.docx'],
            'code': ['.py', '.js', '.java', '.cpp', '.h'],
            'data': ['.json', '.csv', '.xml'],
            'spreadsheet': ['.xlsx', '.xls'],
            'presentation': ['.ppt', '.pptx'],
        }
        self.file_type_descriptions = {
            'image': "An image file ({ext} format) provided as context for the query",
            'text': "A text file ({ext} format) containing additional information related to the query",
            'document': "A document ({ext} format) with content relevant to the query",
            'code': "A source code file ({ext} format) potentially related to the query",
            'data': "A data file ({ext} format) containing structured data pertinent to the query",
            'spreadsheet': "A spreadsheet file ({ext} format) with tabular data relevant to the query",
            'presentation': "A presentation file ({ext} format) with slides related to the query",
        }

    def _get_default_description(self, file_name: str) -> str:
        _, ext = os.path.splitext(file_name)
        ext = ext.lower()

        for file_type, extensions in self.file_types.items():
            if ext in extensions:
                return self.file_type_descriptions[file_type].format(ext=ext[1:])

        return f"A file with {ext[1:]} extension, provided as context for the query"
    
    def add_file(self, file_name: Union[str, List[str]], description: Union[str, List[str], None] = None) -> None:
        if isinstance(file_name, str):
            file_name = [file_name]
        
        if description is None:
            description = [self._get_default_description(fname) for fname in file_name]
        elif isinstance(description, str):
            description = [description]
        
        if len(file_name) != len(description):
            raise ValueError("The number of files and descriptions must match.")
        
        for fname, desc in zip(file_name, description):
            self.files.append({
                'file_name': fname,
                'description': desc
            })

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any,
                   params: Optional[Dict[str, Any]] = None, verification: Optional[Dict[str, Any]] = None,
                   planner_reasoning: Optional[str] = None) -> None:
        # Sanitize result to separate summary (LLM-safe) from artifacts (file paths)
        sanitized = sanitize_tool_output_for_llm(result)
        action = {
            'step_count': step_count,  # Store step_count in action for reference
            'step_name': f"Action Step {step_count}",  # Keep for backward compatibility if needed
            'step_id': f"step_{step_count}_{uuid.uuid4().hex[:8]}",  # Unique identifier
            'tool_name': tool_name,
            'sub_goal': sub_goal,
            'command': command,
            'result': result,  # Keep full result for executor/cache use
            'result_summary': sanitized['summary'],  # LLM-safe summary only
            'artifacts': sanitized['artifacts'],  # File paths for executor/cache
            'params': params,  # VLRR: final (possibly retuned) params for this step
            'verification': verification,  # VLRR: per-stage QC verdict summary + trial history
            'planner_reasoning': planner_reasoning  # Full planner context+justification (durable only; see get_actions)
        }
        # Append to list to avoid key collisions
        self.actions.append(action)

    def get_query(self) -> Optional[str]:
        return self.query

    def get_files(self) -> List[Dict[str, str]]:
        return self.files
    
    def get_actions(self, llm_safe: bool = False) -> List[Dict[str, Any]]:
        """
        Get list of actions.
        
        Args:
            llm_safe: If True, return only LLM-safe summaries (no file paths).
                     If False, return full results (for executor/cache use).
        
        Returns:
            List of action dictionaries
        """
        # Actions are now stored in a list, no need to convert from dict
        actions = self.actions
        if llm_safe:
            # Return LLM-safe version with only summaries, no file paths
            safe_actions = []
            for action in actions:
                safe_action = {
                    'tool_name': action.get('tool_name'),
                    'sub_goal': action.get('sub_goal'),
                    'command': action.get('command'),
                    'result': action.get('result_summary', action.get('result'))  # Use summary if available
                }
                # Recursively sanitize result to remove any remaining file paths
                if safe_action['result']:
                    safe_action['result'] = sanitize_paths_in_dict(safe_action['result'])
                # Surface the VLRR QC verdict so the reasoning layer can see WARNING/
                # conceded and weak metrics instead of silently over-claiming.
                qc = compact_verification(action.get('verification'))
                if qc:
                    safe_action['verification'] = qc
                safe_actions.append(safe_action)
            return safe_actions
        return actions
    


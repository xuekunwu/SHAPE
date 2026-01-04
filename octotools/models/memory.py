from typing import Dict, Any, List, Union, Optional
import os
import uuid
from octotools.models.utils import sanitize_tool_output_for_llm, get_llm_safe_result, sanitize_paths_in_dict

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

    def add_action(self, step_count: int, tool_name: str, sub_goal: str, command: str, result: Any) -> None:
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
            'artifacts': sanitized['artifacts']  # File paths for executor/cache
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
                safe_actions.append(safe_action)
            return safe_actions
        return actions
    
import os
import sys
import ast
import importlib
import inspect
import textwrap
import traceback
from pathlib import Path
from typing import Dict, Any, List, Tuple
import re


class Initializer:
    def __init__(self, enabled_tools: List[str] = None, model_string: str = None,
                 api_key: str = None, verbose: bool = False):
        self.toolbox_metadata = {}
        self.available_tools = []
        self.enabled_tools = enabled_tools or []
        self.model_string = model_string # llm model string
        self.api_key = api_key
        self.verbose = verbose
        self._tool_classes_cache = {}  # Cache tool classes without instantiation
        self._tool_metadata_cache = {}  # Cache metadata for loaded tools
        self._tool_load_errors = {}  # Do not retry imports that already failed

        self._set_up_tools()

    def get_project_root(self):
        current_dir = Path(__file__).resolve().parent
        for directory in (current_dir, *current_dir.parents):
            shape_dir = directory / 'shape'
            if shape_dir.exists():
                return str(shape_dir)
        raise Exception("Could not find project root")
        
    def class_name_to_dir(self, class_name: str) -> str:
        # Class name (e.g. Fibroblast_Activation_Scorer_Tool) -> directory name (fibroblast_activation_scorer)
        if class_name.endswith('_Tool'):
            class_name = class_name[:-5]
        parts = class_name.split('_')
        dir_name = '_'.join([p.lower() for p in parts])
        return dir_name

    def dir_to_class_name(self, dir_name: str) -> str:
        # Directory name -> class name (e.g. fibroblast_activation_scorer -> Fibroblast_Activation_Scorer_Tool)
        # Special cases: handle tools with non-standard capitalization
        special_cases = {
            'url_text_extractor': 'URL_Text_Extractor_Tool',  # URL is all uppercase
            'arxiv_paper_searcher': 'ArXiv_Paper_Searcher_Tool',  # ArXiv is mixed case
        }
        
        if dir_name in special_cases:
            return special_cases[dir_name]
        
        # Standard conversion: snake_case -> Pascal_Case_Tool
        parts = dir_name.split('_')
        class_name = '_'.join([p.capitalize() for p in parts]) + '_Tool'
        return class_name
    
    def _find_tool_class(self, module, expected_class_name: str):
        """
        Try to find the tool class in the module, handling case variations.
        For example, if expected_class_name is 'URL_Text_Extractor_Tool', 
        it will try 'Url_Text_Extractor_Tool' as well (for backwards compatibility).
        """
        # First try the exact name
        if hasattr(module, expected_class_name):
            return getattr(module, expected_class_name)
        
        # Try variations: convert first part to uppercase if it's a common abbreviation
        parts = expected_class_name.split('_')
        if len(parts) > 0:
            # Common abbreviations that should be uppercase
            abbreviations = ['url', 'api', 'id', 'pdf', 'html', 'xml', 'json', 'csv']
            first_part_lower = parts[0].lower()
            if first_part_lower in abbreviations:
                # Try with first part uppercase
                variant_parts = [parts[0].upper()] + parts[1:]
                variant_name = '_'.join(variant_parts)
                if hasattr(module, variant_name):
                    print(f"Found class {variant_name} instead of {expected_class_name}")
                    return getattr(module, variant_name)
        
        # Try to find any class that ends with '_Tool' and has similar name
        module_attrs = dir(module)
        for attr in module_attrs:
            if attr.endswith('_Tool') and attr.lower() == expected_class_name.lower():
                print(f"Found class {attr} instead of {expected_class_name} (case-insensitive match)")
                return getattr(module, attr)
        
        # If still not found, raise AttributeError with helpful message
        available_classes = [attr for attr in module_attrs if attr.endswith('_Tool')]
        raise AttributeError(
            f"module '{module.__name__}' has no attribute '{expected_class_name}'. "
            f"Available classes ending with '_Tool': {available_classes}"
        )

    def _load_tool_class_only(self, tool_class_name: str):
        """Load tool class without instantiation. Returns tool class or None."""
        if tool_class_name in self._tool_classes_cache:
            return self._tool_classes_cache[tool_class_name]
        if tool_class_name in self._tool_load_errors:
            return None
        
        try:
            tool_dir = self.class_name_to_dir(tool_class_name)
            module_name = f"shape.tools.{tool_dir}.tool"
            module = importlib.import_module(module_name)
            tool_class = self._find_tool_class(module, tool_class_name)
            self._tool_classes_cache[tool_class_name] = tool_class
            return tool_class
        except Exception as e:
            error = str(e)
            self._tool_load_errors[tool_class_name] = error
            print(f"Error loading tool class {tool_class_name}: {error}")
            return None
    
    def get_tool_metadata(self, tool_class_name: str) -> Dict[str, Any]:
        """Read tool metadata without constructing the tool.

        Tool constructors may allocate GPU models, create LLM clients, or access
        the network. Metadata discovery must remain side-effect free.
        """
        # Check cache first
        if tool_class_name in self._tool_metadata_cache:
            return self._tool_metadata_cache[tool_class_name]
        
        # Load tool class if not cached
        tool_class = self._load_tool_class_only(tool_class_name)
        if tool_class is None:
            return {}
        
        try:
            source = textwrap.dedent(inspect.getsource(tool_class.__init__))
            tree = ast.parse(source)
            metadata_kwargs = None
            for node in ast.walk(tree):
                if not isinstance(node, ast.Call):
                    continue
                keyword_names = {kw.arg for kw in node.keywords if kw.arg}
                if "tool_name" not in keyword_names:
                    continue
                metadata_kwargs = {}
                metadata_fields = {
                    "tool_name", "tool_description", "tool_version",
                    "input_types", "output_type", "demo_commands",
                    "user_metadata",
                }
                for keyword in node.keywords:
                    if keyword.arg not in metadata_fields:
                        continue
                    try:
                        metadata_kwargs[keyword.arg] = ast.literal_eval(keyword.value)
                    except (ValueError, TypeError):
                        # A dynamic optional field should not force construction.
                        continue
                break

            if not metadata_kwargs:
                raise ValueError("BaseTool metadata declaration not found")

            metadata = {
                'tool_name': metadata_kwargs.get('tool_name', tool_class_name),
                'tool_description': metadata_kwargs.get('tool_description', 'No description'),
                'tool_version': metadata_kwargs.get('tool_version', 'Unknown'),
                'input_types': metadata_kwargs.get('input_types', {}),
                'output_type': metadata_kwargs.get('output_type', 'Unknown'),
                'demo_commands': metadata_kwargs.get('demo_commands', []),
                'user_metadata': metadata_kwargs.get('user_metadata', {}),
                'require_llm_engine': getattr(tool_class, 'require_llm_engine', False),
            }
            self._tool_metadata_cache[tool_class_name] = metadata
            return metadata
        except Exception as e:
            print(f"Error reading metadata for {tool_class_name}: {str(e)}")
            return {}
    
    def load_tools_and_get_metadata(self) -> Dict[str, Any]:
        """Load tool classes (without instantiation) and get metadata on demand."""
        self.toolbox_metadata = {}
        shape_dir = self.get_project_root()
        tools_dir = os.path.join(shape_dir, 'tools')
        sys.path.insert(0, shape_dir)
        sys.path.insert(0, os.path.dirname(shape_dir))
        
        if not os.path.exists(tools_dir):
            print(f"Error: Tools directory does not exist: {tools_dir}")
            return self.toolbox_metadata
        
        # Only load tool classes, do not instantiate
        for tool_class_name in self.enabled_tools:
            tool_dir = self.class_name_to_dir(tool_class_name)
            module_name = f"shape.tools.{tool_dir}.tool"
            if self.verbose:
                print(f"Attempting to import: {module_name}")
            try:
                tool_class = self._load_tool_class_only(tool_class_name)
                if tool_class:
                    # For Planner, we need metadata. Load it on demand
                    # Store a placeholder that will be loaded when needed
                    self.toolbox_metadata[tool_class_name] = None  # Will be loaded on demand
                    if self.verbose:
                        print(f"[OK] Tool class {tool_class_name} loaded")
            except Exception as e:
                print(f"Error loading tool {tool_class_name}: {str(e)}")
        return self.toolbox_metadata

    def run_demo_commands(self) -> List[str]:
        """Verify tools are available (without instantiation)."""
        self.available_tools = []
        failed_tools = []
        for tool_class_name in self.enabled_tools:
            try:
                tool_class = self._load_tool_class_only(tool_class_name)
                if tool_class:
                    self.available_tools.append(tool_class_name)
                else:
                    failed_tools.append((tool_class_name, "Failed to load tool class (returned None)"))
            except Exception as e:
                error_msg = f"Error checking availability of {tool_class_name}: {str(e)}"
                failed_tools.append((tool_class_name, error_msg))
                print(error_msg)
                print(traceback.format_exc())
        
        if failed_tools:
            print(f"\nWARNING: {len(failed_tools)} tool(s) failed to load:")
            for tool_name, error in failed_tools:
                detail = self._tool_load_errors.get(tool_name, error)
                print(f"  - {tool_name}: {detail}")
        
        return self.available_tools
    
    def _set_up_tools(self) -> None:
        # If enabled_tools is empty, auto-discover all tools
        if not self.enabled_tools:
            tools_dir = os.path.join(self.get_project_root(), 'tools')
            all_tools = []
            for root, dirs, files in os.walk(tools_dir):
                if 'tool.py' in files:
                    dir_name = os.path.basename(root)
                    class_name = self.dir_to_class_name(dir_name)
                    all_tools.append(class_name)
            self.enabled_tools = all_tools

        # Load tool classes (without instantiation)
        self.load_tools_and_get_metadata()
        self.available_tools = list(self.toolbox_metadata)
        
        # Initialize metadata placeholders for available tools
        # Metadata will be loaded on demand when needed by Planner
        for tool in self.available_tools:
            if tool not in self.toolbox_metadata:
                self.toolbox_metadata[tool] = None
        
        failed_count = len(self.enabled_tools) - len(self.available_tools)
        print(
            f"[OK] Tools loaded: {len(self.available_tools)} available"
            + (f", {failed_count} unavailable" if failed_count else "")
        )
    
    def get_toolbox_metadata(self) -> Dict[str, Any]:
        """Get toolbox metadata with lazy loading. Loads metadata on demand."""
        # Load metadata for tools that haven't been loaded yet
        for tool_name in self.available_tools:
            if tool_name not in self._tool_metadata_cache:
                # Read metadata without constructing the tool.
                if self.verbose:
                    print(f"Loading metadata for {tool_name}...")
                metadata = self.get_tool_metadata(tool_name)
                if metadata:
                    self.toolbox_metadata[tool_name] = metadata
        
        # Return only tools that have metadata loaded
        return {k: v for k, v in self.toolbox_metadata.items() if v is not None}


if __name__ == "__main__":
    enabled_tools = ["Generalist_Solution_Generator_Tool"]
    initializer = Initializer(enabled_tools=enabled_tools)

    print("\nAvailable tools:")
    print(initializer.available_tools)

    print("\nToolbox metadata for available tools (lazy loaded):")
    metadata = initializer.get_toolbox_metadata()
    print(metadata)
    

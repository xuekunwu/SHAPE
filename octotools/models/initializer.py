import os
import sys
import importlib
import inspect
import traceback
from typing import Dict, Any, List, Tuple
import re


class Initializer:
    def __init__(self, enabled_tools: List[str] = [], model_string: str = None, api_key: str = None):
        self.toolbox_metadata = {}
        self.available_tools = []
        self.enabled_tools = enabled_tools
        self.model_string = model_string # llm model string
        self.api_key = api_key

        print("\nInitializing OctoTools...")
        print(f"Enabled tools: {self.enabled_tools}")
        print(f"LLM model string: {self.model_string}")

        self._set_up_tools()

    def get_project_root(self):
        current_dir = os.path.dirname(os.path.abspath(__file__))
        while current_dir != '/':
            if os.path.exists(os.path.join(current_dir, 'octotools')):
                return os.path.join(current_dir, 'octotools')
            current_dir = os.path.dirname(current_dir)
        raise Exception("Could not find project root")
        
    def class_name_to_dir(self, class_name: str) -> str:
        # 类名（如Fibroblast_Activation_Scorer_Tool）转为目录名（fibroblast_activation_scorer）
        if class_name.endswith('_Tool'):
            class_name = class_name[:-5]
        parts = class_name.split('_')
        dir_name = '_'.join([p.lower() for p in parts])
        return dir_name

    def dir_to_class_name(self, dir_name: str) -> str:
        # 目录名转为类名（如fibroblast_activation_scorer -> Fibroblast_Activation_Scorer_Tool）
        parts = dir_name.split('_')
        class_name = '_'.join([p.capitalize() for p in parts]) + '_Tool'
        return class_name
    
    def _find_tool_class(self, module, expected_class_name: str):
        """
        Try to find the tool class in the module, handling case variations.
        For example, if expected_class_name is 'Url_Text_Extractor_Tool', 
        it will try 'URL_Text_Extractor_Tool' as well.
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

    def load_tools_and_get_metadata(self) -> Dict[str, Any]:
        print("Loading tools and getting metadata...")
        self.toolbox_metadata = {}
        octotools_dir = self.get_project_root()
        tools_dir = os.path.join(octotools_dir, 'tools')
        print(f"OctoTools directory: {octotools_dir}")
        print(f"Tools directory: {tools_dir}")
        sys.path.insert(0, octotools_dir)
        sys.path.insert(0, os.path.dirname(octotools_dir))
        print(f"Updated Python path: {sys.path}")
        if not os.path.exists(tools_dir):
            print(f"Error: Tools directory does not exist: {tools_dir}")
            return self.toolbox_metadata
        for tool_class_name in self.enabled_tools:
            tool_dir = self.class_name_to_dir(tool_class_name)
            module_name = f"octotools.tools.{tool_dir}.tool"
            print(f"Attempting to import: {module_name}")
            try:
                module = importlib.import_module(module_name)
                tool_class = self._find_tool_class(module, tool_class_name)
                # 实例化工具
                inputs = {}
                if hasattr(tool_class, 'require_llm_engine') and tool_class.require_llm_engine:
                    inputs['model_string'] = self.model_string
                if hasattr(tool_class, 'require_api_key') and tool_class.require_api_key:
                    inputs['api_key'] = self.api_key
                tool_instance = tool_class(**inputs)
                self.toolbox_metadata[tool_class_name] = {
                    'tool_name': getattr(tool_instance, 'tool_name', 'Unknown'),
                    'tool_description': getattr(tool_instance, 'tool_description', 'No description'),
                    'tool_version': getattr(tool_instance, 'tool_version', 'Unknown'),
                    'input_types': getattr(tool_instance, 'input_types', {}),
                    'output_type': getattr(tool_instance, 'output_type', 'Unknown'),
                    'demo_commands': getattr(tool_instance, 'demo_commands', []),
                    'user_metadata': getattr(tool_instance, 'user_metadata', {}),
                    'require_llm_engine': getattr(tool_class, 'require_llm_engine', False),
                }
                print(f"\nMetadata for {tool_class_name}: {self.toolbox_metadata[tool_class_name]}")
            except Exception as e:
                print(f"Error loading tool {tool_class_name}: {str(e)}")
        print(f"\nTotal number of tools loaded: {len(self.toolbox_metadata)}")
        return self.toolbox_metadata

    def run_demo_commands(self) -> List[str]:
        print("\nRunning demo commands for each tool...")
        self.available_tools = []
        for tool_class_name in self.enabled_tools:
            print(f"\nChecking availability of {tool_class_name}...")
            try:
                tool_dir = self.class_name_to_dir(tool_class_name)
                module_name = f"octotools.tools.{tool_dir}.tool"
                print(f"Attempting to import: {module_name}")
                module = importlib.import_module(module_name)
                tool_class = self._find_tool_class(module, tool_class_name)
                # 实例化工具
                inputs = {}
                if hasattr(tool_class, 'require_llm_engine') and tool_class.require_llm_engine:
                    inputs['model_string'] = self.model_string
                if hasattr(tool_class, 'require_api_key') and tool_class.require_api_key:
                    inputs['api_key'] = self.api_key
                tool_instance = tool_class(**inputs)
                self.available_tools.append(tool_class_name)
            except Exception as e:
                print(f"Error checking availability of {tool_class_name}: {str(e)}")
                print(traceback.format_exc())
        self.toolbox_metadata = {tool: self.toolbox_metadata[tool] for tool in self.available_tools if tool in self.toolbox_metadata}
        print(f"\nUpdated total number of available tools: {len(self.toolbox_metadata)}")
        print(f"\nAvailable tools: {self.available_tools}")
        return self.available_tools
    
    def _set_up_tools(self) -> None:
        print("Setting up tools...")

        # 如果enabled_tools为空，则自动发现所有工具
        if not self.enabled_tools:
            tools_dir = os.path.join(self.get_project_root(), 'tools')
            all_tools = []
            for root, dirs, files in os.walk(tools_dir):
                if 'tool.py' in files:
                    dir_name = os.path.basename(root)
                    class_name = self.dir_to_class_name(dir_name)
                    all_tools.append(class_name)
            self.enabled_tools = all_tools
            print(f"Auto-discovered tools: {self.enabled_tools}")

        self.available_tools = []
        for tool in self.enabled_tools:
            self.available_tools.append(tool)

        # Load tools and get metadata
        self.load_tools_and_get_metadata()
        
        # Run demo commands to determine available tools
        self.run_demo_commands()
        
        # Filter toolbox_metadata to include only available tools
        self.toolbox_metadata = {tool: self.toolbox_metadata[tool] for tool in self.available_tools if tool in self.toolbox_metadata}
        
        print(f"\nTotal number of available tools: {len(self.available_tools)}")
        print(f"Available tools: {self.available_tools}")
        print(f"Enabled tools: {self.enabled_tools}")


if __name__ == "__main__":
    enabled_tools = ["Generalist_Solution_Generator_Tool"]
    initializer = Initializer(enabled_tools=enabled_tools)

    print("\nAvailable tools:")
    print(initializer.available_tools)

    print("\nToolbox metadata for available tools:")
    print(initializer.toolbox_metadata)
    
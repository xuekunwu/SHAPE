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
        
    def load_tools_and_get_metadata(self) -> Dict[str, Any]:
        # Implementation of load_tools_and_get_metadata function
        print("Loading tools and getting metadata...")
        self.toolbox_metadata = {}
        octotools_dir = self.get_project_root()
        tools_dir = os.path.join(octotools_dir, 'tools')
        
        print(f"OctoTools directory: {octotools_dir}")
        print(f"Tools directory: {tools_dir}")
        
        # Add the OctoTools directory and its parent to the Python path
        sys.path.insert(0, octotools_dir)
        sys.path.insert(0, os.path.dirname(octotools_dir))
        print(f"Updated Python path: {sys.path}")
        
        if not os.path.exists(tools_dir):
            print(f"Error: Tools directory does not exist: {tools_dir}")
            return self.toolbox_metadata

        for root, dirs, files in os.walk(tools_dir):
            # print(f"\nScanning directory: {root}")
            if 'tool.py' in files and os.path.basename(root) in self.available_tools: # NOTE
                file = 'tool.py'
                module_path = os.path.join(root, file)
                module_name = os.path.splitext(file)[0]
                relative_path = os.path.relpath(module_path, octotools_dir)
                import_path = '.'.join(os.path.split(relative_path)).replace(os.sep, '.')[:-3]

                print(f"\nAttempting to import: {import_path}")
                try:
                    module = importlib.import_module(import_path)
                    for name, obj in inspect.getmembers(module):
                        if inspect.isclass(obj) and name.endswith('Tool') and name != 'BaseTool':
                            print(f"Found tool class: {name}")
                            # print(f"Class attributes: {dir(obj)}")
                            # print(f"Class __dict__: {obj.__dict__}")
                            try:
                                # Check if the tool requires an LLM engine
                                inputs = {}
                                if hasattr(obj, 'require_llm_engine') and obj.require_llm_engine:
                                    inputs['model_string'] = self.model_string
                                
                                if hasattr(obj, 'require_api_key') and obj.require_api_key:
                                    inputs['api_key'] = self.api_key
                                    
                                tool_instance = obj(**inputs)

                                # print(f"\nInstance attributes: {dir(tool_instance)}")
                                # print(f"\nInstance __dict__: {tool_instance.__dict__}")
                                
                                self.toolbox_metadata[name] = {
                                    'tool_name': getattr(tool_instance, 'tool_name', 'Unknown'),
                                    'tool_description': getattr(tool_instance, 'tool_description', 'No description'),
                                    'tool_version': getattr(tool_instance, 'tool_version', 'Unknown'),
                                    'input_types': getattr(tool_instance, 'input_types', {}),
                                    'output_type': getattr(tool_instance, 'output_type', 'Unknown'),
                                    'demo_commands': getattr(tool_instance, 'demo_commands', []),
                                    'user_metadata': getattr(tool_instance, 'user_metadata', {}), # NOTE: This is a placeholder for user-defined metadata
                                    'require_llm_engine': getattr(obj, 'require_llm_engine', False),
                                }
                                print(f"\nMetadata for {name}: {self.toolbox_metadata[name]}")
                            except Exception as e:
                                print(f"Error instantiating {name}: {str(e)}")
                except Exception as e:
                    print(f"Error loading module {module_name}: {str(e)}")
                    
        print(f"\nTotal number of tools loaded: {len(self.toolbox_metadata)}")

        return self.toolbox_metadata

    def run_demo_commands(self) -> List[str]:
        print("\nRunning demo commands for each tool...")
        self.available_tools = []

        for tool_name, tool_data in self.toolbox_metadata.items():
            print(f"\nChecking availability of {tool_name}...")

            try:
                # Import the tool module - simplified import logic
                # tool_name is class name (e.g. Fibroblast_Activation_Scorer_Tool)
                # need to convert to directory name format
                
                # Remove _Tool suffix and convert to lowercase as directory name
                if tool_name.lower().endswith('_tool'):
                    tool_dir = tool_name[:-5].lower()  # Remove _Tool or _tool
                else:
                    tool_dir = tool_name.lower()
                
                module_name = f"octotools.tools.{tool_dir}.tool"
                print(f"Attempting to import: {module_name}")
                module = importlib.import_module(module_name)

                # Get the tool class
                tool_class = getattr(module, tool_name)

                # Instantiate the tool
                inputs = {}
                if hasattr(tool_class, 'require_llm_engine') and tool_class.require_llm_engine:
                    inputs['model_string'] = self.model_string
                
                if hasattr(tool_class, 'require_api_key') and tool_class.require_api_key:
                    inputs['api_key'] = self.api_key
                tool_instance = tool_class(**inputs)

                # Add to available tools list
                self.available_tools.append(tool_name)

            except Exception as e:
                print(f"Error checking availability of {tool_name}: {str(e)}")
                print(traceback.format_exc())

        # update the toolmetadata with the available tools
        self.toolbox_metadata = {tool: self.toolbox_metadata[tool] for tool in self.available_tools}
        print(f"\nUpdated total number of available tools: {len(self.toolbox_metadata)}")
        print(f"\nAvailable tools: {self.available_tools}")

        return self.available_tools
    
    def _set_up_tools(self) -> None:
        print("Setting up tools...")

        # Keep enabled tools - simplified logic, use enabled_tools directly
        self.available_tools = []
        for tool in self.enabled_tools:
            # Remove _Tool suffix and convert to lowercase as directory name
            if tool.lower().endswith('_tool'):
                tool_dir = tool[:-5].lower()  # Remove _Tool or _tool
            else:
                tool_dir = tool.lower()
            self.available_tools.append(tool_dir)
        
        # Load tools and get metadata
        self.load_tools_and_get_metadata()
        
        # Run demo commands to determine available tools
        self.run_demo_commands()
        
        # Filter toolbox_metadata to include only available tools
        self.toolbox_metadata = {tool: self.toolbox_metadata[tool] for tool in self.available_tools}
        
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
    
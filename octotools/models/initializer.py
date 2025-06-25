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
                # Import the tool module - fix the import path
                # tool_name is the class name (e.g., FibroblastActivationScorerTool)
                # We need to convert it to directory name format
                
                # 创建工具名称到目录名称的映射（类名格式：CamelCase）
                tool_to_dir_mapping = {
                    'FibroblastActivationScorerTool': 'fibroblast_activation_scorer',
                    'FibroblastStateAnalyzerTool': 'fibroblast_state_analyzer',
                    'ImagePreprocessorTool': 'image_preprocessor',
                    'NucleiSegmenterTool': 'nuclei_segmenter',
                    'SingleCellCropperTool': 'single_cell_cropper',
                    'GeneralistSolutionGeneratorTool': 'generalist_solution_generator',
                    'PythonCodeGeneratorTool': 'python_code_generator',
                    'ArxivPaperSearcherTool': 'arxiv_paper_searcher',
                    'PubmedSearchTool': 'pubmed_search',
                    'NatureNewsFetcherTool': 'nature_news_fetcher',
                    'GoogleSearchTool': 'google_search',
                    'WikipediaKnowledgeSearcherTool': 'wikipedia_knowledge_searcher',
                    'UrlTextExtractorTool': 'url_text_extractor',
                    'ObjectDetectorTool': 'object_detector',
                    'ImageCaptionerTool': 'image_captioner',
                    'RelevantPatchZoomerTool': 'relevant_patch_zoomer',
                    'TextDetectorTool': 'text_detector',
                    'AdvancedObjectDetectorTool': 'advanced_object_detector'
                }
                
                # 使用映射或默认转换
                if tool_name in tool_to_dir_mapping:
                    tool_dir = tool_to_dir_mapping[tool_name]
                else:
                    # 默认转换：去掉 Tool 后缀，转换为小写
                    if tool_name.lower().endswith('tool'):
                        base_name = tool_name[:-4] if tool_name.endswith('Tool') else tool_name[:-5]
                        tool_dir = base_name.lower()
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

                # FIXME This is a temporary workaround to avoid running demo commands
                self.available_tools.append(tool_name)

                # # TODO Run the first demo command if available
                # demo_commands = tool_data.get('demo_commands', [])
                # if demo_commands:
                #     print(f"Running demo command: {demo_commands[0]['command']}")
                #     # Extract the arguments from the demo command
                #     command = demo_commands[0]['command']
                #     args_start = command.index('(') + 1
                #     args_end = command.rindex(')')
                #     args_str = command[args_start:args_end]

                #     # Create a dictionary of arguments
                #     args_dict = eval(f"dict({args_str})")

                #     # Execute the demo command
                #     result = tool_instance.execute(**args_dict)
                #     print(f"Demo command executed successfully. Result: {result}")

                #     self.available_tools.append(tool_name)
                # else:
                #     print(f"No demo commands available for {tool_name}")
                #     # If no demo commands, we'll assume the tool is available
                #     self.available_tools.append(tool_name)

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

        # 创建工具名称到目录名称的映射（工具名格式：带下划线）
        tool_to_dir_mapping = {
            'Fibroblast_Activation_Scorer_Tool': 'fibroblast_activation_scorer',
            'Fibroblast_State_Analyzer_Tool': 'fibroblast_state_analyzer',
            'Image_Preprocessor_Tool': 'image_preprocessor',
            'Nuclei_Segmenter_Tool': 'nuclei_segmenter',
            'Single_Cell_Cropper_Tool': 'single_cell_cropper',
            'Generalist_Solution_Generator_Tool': 'generalist_solution_generator',
            'Python_Code_Generator_Tool': 'python_code_generator',
            'Arxiv_Paper_Searcher_Tool': 'arxiv_paper_searcher',
            'Pubmed_Search_Tool': 'pubmed_search',
            'Nature_News_Fetcher_Tool': 'nature_news_fetcher',
            'Google_Search_Tool': 'google_search',
            'Wikipedia_Knowledge_Searcher_Tool': 'wikipedia_knowledge_searcher',
            'Url_Text_Extractor_Tool': 'url_text_extractor',
            'Object_Detector_Tool': 'object_detector',
            'Image_Captioner_Tool': 'image_captioner',
            'Relevant_Patch_Zoomer_Tool': 'relevant_patch_zoomer',
            'Text_Detector_Tool': 'text_detector',
            'Advanced_Object_Detector_Tool': 'advanced_object_detector'
        }
        
        # 创建目录名称到类名的映射
        dir_to_class_mapping = {
            'fibroblast_activation_scorer': 'FibroblastActivationScorerTool',
            'fibroblast_state_analyzer': 'FibroblastStateAnalyzerTool',
            'image_preprocessor': 'ImagePreprocessorTool',
            'nuclei_segmenter': 'NucleiSegmenterTool',
            'single_cell_cropper': 'SingleCellCropperTool',
            'generalist_solution_generator': 'GeneralistSolutionGeneratorTool',
            'python_code_generator': 'PythonCodeGeneratorTool',
            'arxiv_paper_searcher': 'ArxivPaperSearcherTool',
            'pubmed_search': 'PubmedSearchTool',
            'nature_news_fetcher': 'NatureNewsFetcherTool',
            'google_search': 'GoogleSearchTool',
            'wikipedia_knowledge_searcher': 'WikipediaKnowledgeSearcherTool',
            'url_text_extractor': 'UrlTextExtractorTool',
            'object_detector': 'ObjectDetectorTool',
            'image_captioner': 'ImageCaptionerTool',
            'relevant_patch_zoomer': 'RelevantPatchZoomerTool',
            'text_detector': 'TextDetectorTool',
            'advanced_object_detector': 'AdvancedObjectDetectorTool'
        }
        
        # Keep enabled tools - 修复工具名称处理逻辑
        self.available_tools = []
        for tool in self.enabled_tools:
            if tool in tool_to_dir_mapping:
                tool_dir = tool_to_dir_mapping[tool]
                # 将目录名转换为类名并存储
                if tool_dir in dir_to_class_mapping:
                    class_name = dir_to_class_mapping[tool_dir]
                    self.available_tools.append(class_name)
            else:
                # 默认转换：去掉 _Tool 后缀，转换为小写
                if tool.lower().endswith('_tool'):
                    tool_dir = tool[:-5].lower()  # 去掉 _Tool 或 _tool
                else:
                    tool_dir = tool.lower()
                # 尝试转换为类名
                if tool_dir in dir_to_class_mapping:
                    class_name = dir_to_class_mapping[tool_dir]
                    self.available_tools.append(class_name)
        
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
    
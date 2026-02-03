import os
from shape.tools.base import BaseTool
from shape.engine.openai import ChatOpenAI

class Image_Captioner_Tool(BaseTool):
    require_llm_engine = True
    require_api_key = True

    def __init__(self, model_string="gpt-4o-mini", api_key=None):
        super().__init__(
            tool_name="Image_Captioner_Tool",
            tool_description="A tool that generates captions for images using OpenAI's multimodal model. Supports general image description, cluster exemplar montage analysis, and cell morphological feature analysis.",
            tool_version="1.1.0",
            input_types={
                "image": "str - The path to the image file.",
                "prompt": "str - Optional custom prompt to guide the image captioning (default: 'Describe this image in detail.'). For cluster exemplars, a specialized prompt is automatically used unless overridden.",
                "analysis_type": "str - Type of analysis to perform. Options: 'general' (default), 'cluster_exemplars' (for cluster exemplar montages), 'cell_morphology' (for cell morphological analysis).",
                "query_cache_dir": "str - Optional directory for caching results (for consistency with other tools, not used).",
            },
            output_type="str - The generated caption for the image, with specialized analysis for cluster exemplars or cell morphology if requested.",
            demo_commands=[
                {
                    "command": 'execution = tool.execute(image="path/to/image.png")',
                    "description": "Generate a caption for an image using the default prompt and model."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/cluster_exemplars.png", analysis_type="cluster_exemplars")',
                    "description": "Analyze cluster exemplar montage, describing morphological features for each cluster."
                },
                {
                    "command": 'execution = tool.execute(image="path/to/cell_image.png", analysis_type="cell_morphology")',
                    "description": "Analyze cell morphological features in a microscopy image."
                }
            ],
            user_metadata = {
                "limitation": "The Image_Captioner_Tool provides general image descriptions but has limitations: 1) May make mistakes in complex scenes, counting, attribute detection, and understanding object relationships. 2) Might not generate comprehensive captions, especially for images with multiple objects or abstract concepts. 3) Performance varies with image complexity. 4) Struggles with culturally specific or domain-specific content. 5) May overlook details or misinterpret object relationships. For precise descriptions, consider: using it with other tools for context/verification, as an initial step before refinement, or in multi-step processes for ambiguity resolution. Verify critical information with specialized tools or human expertise when necessary."
            },
        )
        print(f"\nInitializing Image Captioner Tool with model: {model_string}")
        self.llm_engine = ChatOpenAI(model_string=model_string, is_multimodal=True, api_key=api_key) if model_string else None

    def execute(self, image, prompt="Describe this image in detail.", analysis_type="general", query_cache_dir=None):
        """
        Execute image captioning.
        
        Args:
            image: Path to the image file
            prompt: Custom prompt for captioning (default: "Describe this image in detail.")
            analysis_type: Type of analysis to perform. Options:
                - "general": General image description (default)
                - "cluster_exemplars": Specialized analysis for cluster exemplar montages
                - "cell_morphology": Focused analysis on cell morphological features
            query_cache_dir: Optional directory for caching results (for consistency with other tools, not used)
        
        Returns:
            str: Generated caption
        """
        try:
            if not self.llm_engine:
                return "Error: LLM engine not initialized. Please provide a valid model_string."
            
            # Use specialized prompts based on analysis_type
            if analysis_type == "cluster_exemplars":
                final_prompt = self._get_cluster_exemplar_prompt(prompt)
            elif analysis_type == "cell_morphology":
                final_prompt = self._get_cell_morphology_prompt(prompt)
            else:
                final_prompt = prompt
                
            input_data = [final_prompt]
            
            if image and os.path.isfile(image):
                try:
                    with open(image, 'rb') as file:
                        image_bytes = file.read()
                    input_data.append(image_bytes)
                except Exception as e:
                    return f"Error reading image file: {str(e)}"
            else:
                return "Error: Invalid image file path."

            caption = self.llm_engine(input_data)
            return caption
        except Exception as e:
            return f"Error generating caption: {str(e)}"
    
    def _get_cluster_exemplar_prompt(self, custom_prompt=None):
        """Generate specialized prompt for cluster exemplar montage analysis."""
        base_prompt = """You are analyzing a cluster exemplar montage image showing multiple cell clusters. Each row represents a different cluster, and each row contains multiple example cell images from that cluster.

Please provide a detailed analysis of the morphological features for each cluster:

1. For each cluster (row), describe:
   - Overall cell morphology (size, shape, elongation)
   - Cell boundaries and edges
   - Internal structures visible (if any)
   - Texture and intensity patterns
   - Any distinctive features that differentiate this cluster from others

2. Compare clusters:
   - Identify key morphological differences between clusters
   - Note any similarities or shared features
   - Describe the progression or variation across clusters (if applicable)

3. Provide a summary:
   - Overall assessment of the cell state diversity
   - Key morphological characteristics that define each cluster

Be specific and focus on observable morphological features that would be relevant for cell state classification."""
        
        if custom_prompt and custom_prompt != "Describe this image in detail.":
            return f"{base_prompt}\n\nAdditional context: {custom_prompt}"
        return base_prompt
    
    def _get_cell_morphology_prompt(self, custom_prompt=None):
        """Generate specialized prompt for cell morphological analysis."""
        base_prompt = """You are analyzing cell morphological features in a microscopy image. Please provide a detailed description focusing on:

1. Cell shape and size:
   - Overall cell shape (round, elongated, irregular, etc.)
   - Cell size relative to the field of view
   - Aspect ratio and elongation

2. Cell boundaries:
   - Edge definition and clarity
   - Membrane characteristics
   - Cell-cell contact patterns

3. Internal structures:
   - Visible organelles or internal features
   - Intensity patterns and distribution
   - Texture characteristics

4. Spatial organization:
   - Cell arrangement and density
   - Clustering or dispersion patterns
   - Any spatial relationships between cells

Be specific and use terminology appropriate for cell biology and microscopy analysis."""
        
        if custom_prompt and custom_prompt != "Describe this image in detail.":
            return f"{base_prompt}\n\nAdditional context: {custom_prompt}"
        return base_prompt

    def get_metadata(self):
        metadata = super().get_metadata()
        metadata['require_llm_engine'] = self.require_llm_engine # NOTE: can be removed if not needed
        return metadata

if __name__ == "__main__":
    # Test command:
    """
    Run the following commands in the terminal to test the script:
    
    cd shape/tools/image_captioner
    python tool.py
    """

    import json

    # Get the directory of the current script
    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Example usage of the Image_Captioner_Tool
    # tool = Image_Captioner_Tool()
    tool = Image_Captioner_Tool(model_string="gpt-4o")

    # Get tool metadata
    metadata = tool.get_metadata()
    print(metadata)

    # Construct the full path to the image using the script's directory
    relative_image_path = "examples/baseball.png"
    image_path = os.path.join(script_dir, relative_image_path)

    # Execute the tool with default prompt
    try:
        execution = tool.execute(image=image_path)
        print("Generated Caption:")
        print(json.dumps(execution, indent=4)) 
    except Exception as e: 
        print(f"Execution failed: {e}")

    print("Done!")




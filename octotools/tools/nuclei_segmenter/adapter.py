from pydantic import BaseModel, Field
from octotools.tools.nuclei_segmenter.tool import Nuclei_Segmenter_Tool
from octotools.models.tool_adapter import ToolAdapter


class InputModel(BaseModel):
    image: str = Field(..., description="Path to input image")
    diameter: float | None = None
    flow_threshold: float | None = None
    query_cache_dir: str | None = None
    image_id: str | None = None


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Nuclei_Segmenter_Tool()
        return tool.execute(
            image=args.image,
            diameter=args.diameter if args.diameter is not None else 30,
            flow_threshold=args.flow_threshold if args.flow_threshold is not None else 0.4,
            query_cache_dir=args.query_cache_dir,
            image_id=args.image_id,
        )

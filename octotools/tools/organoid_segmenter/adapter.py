from pydantic import BaseModel, Field
from octotools.tools.organoid_segmenter.tool import Organoid_Segmenter_Tool
from octotools.models.tool_adapter import ToolAdapter


class InputModel(BaseModel):
    image: str = Field(..., description="Path to input image")
    diameter: float | None = None
    flow_threshold: float | None = None
    model_path: str | None = None


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Organoid_Segmenter_Tool()
        return tool.execute(
            image=args.image,
            diameter=args.diameter if args.diameter is not None else 100,
            flow_threshold=args.flow_threshold if args.flow_threshold is not None else 0.4,
            model_path=args.model_path,
        )

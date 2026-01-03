from pydantic import BaseModel, Field
from octotools.tools.cell_segmenter.tool import Cell_Segmenter_Tool
from octotools.models.tool_adapter import ToolAdapter


class InputModel(BaseModel):
    image: str = Field(..., description="Path to input image")
    diameter: float | None = None
    flow_threshold: float | None = None
    model_type: str | None = None


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Cell_Segmenter_Tool()
        return tool.execute(
            image=args.image,
            diameter=args.diameter if args.diameter is not None else 30,
            flow_threshold=args.flow_threshold if args.flow_threshold is not None else 0.4,
            model_type=args.model_type if args.model_type is not None else "cpsam",
        )

from pydantic import BaseModel, Field
from octotools.tools.single_cell_cropper.tool import Single_Cell_Cropper_Tool
from octotools.models.tool_adapter import ToolAdapter


class InputModel(BaseModel):
    original_image: str = Field(..., description="Path to original image")
    nuclei_mask: str = Field(..., description="Path to segmentation mask (accepts nuclei_mask, cell_mask, or organoid_mask from segmentation tools)")
    min_area: int | None = None
    margin: int | None = None
    output_format: str | None = None
    query_cache_dir: str | None = None


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Single_Cell_Cropper_Tool()
        return tool.execute(
            original_image=args.original_image,
            nuclei_mask=args.nuclei_mask,
            min_area=args.min_area or 50,
            margin=args.margin or 25,
            output_format=args.output_format or "png",
            query_cache_dir=args.query_cache_dir,
        )

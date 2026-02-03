from pydantic import BaseModel, Field
from shape.tools.image_preprocessor.tool import Image_Preprocessor_Tool
from shape.models.tool_adapter import ToolAdapter


class InputModel(BaseModel):
    image: str = Field(..., description="Path to input image")
    target_brightness: int | None = None
    gaussian_kernel_size: int | None = None
    output_format: str | None = None
    save_intermediate: bool | None = None


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Image_Preprocessor_Tool()
        return tool.execute(
            image=args.image,
            target_brightness=args.target_brightness or 120,
            gaussian_kernel_size=args.gaussian_kernel_size or 151,
            output_format=args.output_format or "png",
            save_intermediate=args.save_intermediate or False,
        )




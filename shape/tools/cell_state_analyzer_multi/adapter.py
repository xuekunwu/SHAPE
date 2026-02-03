from pydantic import BaseModel, Field
from shape.tools.cell_state_analyzer_multi.tool import Cell_State_Analyzer_Multi_Tool
from shape.models.tool_adapter import ToolAdapter


class InputModel(BaseModel):
    cell_crops: list[str] | None = None
    cell_metadata: list[dict] | None = None
    max_epochs: int | None = None
    early_stop_loss: float | None = None
    batch_size: int | None = None
    learning_rate: float | None = None
    cluster_resolution: float | None = None
    query_cache_dir: str | None = None
    in_channels: int | None = None  # Number of input channels (default: 2 for BF+GFP)
    selected_channels: list[int] | None = Field(
        None, description="List of channel indices to use for multi-channel analysis (e.g., [0, 1] for BF and GFP)."
    )
    freeze_patch_embed: bool | None = Field(
        False, description="Whether to freeze the patch embedding layer of the DINOv3 backbone."
    )
    freeze_blocks: int | None = Field(
        0, description="Number of transformer blocks to freeze in the DINOv3 backbone."
    )


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Cell_State_Analyzer_Multi_Tool()
        return tool.execute(
            cell_crops=args.cell_crops,
            cell_metadata=args.cell_metadata,
            max_epochs=args.max_epochs if args.max_epochs is not None else 100,
            early_stop_loss=args.early_stop_loss if args.early_stop_loss is not None else 0.05,
            batch_size=args.batch_size if args.batch_size is not None else 16,
            learning_rate=args.learning_rate if args.learning_rate is not None else 3e-5,
            cluster_resolution=args.cluster_resolution if args.cluster_resolution is not None else 0.5,
            query_cache_dir=args.query_cache_dir,
            in_channels=args.in_channels,
            selected_channels=args.selected_channels,
            freeze_patch_embed=args.freeze_patch_embed if args.freeze_patch_embed is not None else False,
            freeze_blocks=args.freeze_blocks if args.freeze_blocks is not None else 0,
        )




from pydantic import BaseModel, Field
from octotools.tools.cell_state_analyzer_multi.tool import Cell_State_Analyzer_Multi_Tool
from octotools.models.tool_adapter import ToolAdapter


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
        )

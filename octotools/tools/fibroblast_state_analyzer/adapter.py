from pydantic import BaseModel, Field
from octotools.tools.fibroblast_state_analyzer.tool import Fibroblast_State_Analyzer_Tool
from octotools.models.tool_adapter import ToolAdapter


class InputModel(BaseModel):
    cell_crops: list[str] = Field(default_factory=list)
    cell_metadata: list[dict] = Field(default_factory=list)
    batch_size: int | None = None
    query_cache_dir: str | None = None
    visualization_type: str | None = None


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Fibroblast_State_Analyzer_Tool()
        return tool.execute(
            cell_crops=args.cell_crops,
            cell_metadata=args.cell_metadata,
            batch_size=args.batch_size or 16,
            query_cache_dir=args.query_cache_dir or "solver_cache",
            visualization_type=args.visualization_type or "all",
        )

from pydantic import BaseModel, Field
from octotools.tools.fibroblast_activation_scorer.tool import Fibroblast_Activation_Scorer_Tool
from octotools.models.tool_adapter import ToolAdapter


class InputModel(BaseModel):
    cell_data: str | list[str] = Field(..., description="Path to h5ad or list of crops")
    reference_source: str | None = None
    reference_repo_id: str | None = None
    reference_filename: str | None = None
    local_reference_path: str | None = None
    output_dir: str | None = None
    visualization_type: str | None = None
    confidence_threshold: float | None = None
    batch_size: int | None = None


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Fibroblast_Activation_Scorer_Tool()
        return tool.execute(
            cell_data=args.cell_data,
            reference_source=args.reference_source or "huggingface",
            reference_repo_id=args.reference_repo_id or "5xuekun/adata_reference",
            reference_filename=args.reference_filename or "adata_reference.h5ad",
            local_reference_path=args.local_reference_path,
            output_dir=args.output_dir or "output_visualizations",
            visualization_type=args.visualization_type or "all",
            confidence_threshold=args.confidence_threshold or 0.5,
            batch_size=args.batch_size or 100,
        )

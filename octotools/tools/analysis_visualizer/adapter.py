from pydantic import BaseModel, Field
from octotools.tools.analysis_visualizer.tool import Analysis_Visualizer_Tool
from octotools.models.tool_adapter import ToolAdapter
from typing import Union, Dict, Tuple, Optional


class InputModel(BaseModel):
    analysis_data: Union[Dict, str] = Field(description="Analysis results data (dict) or path to JSON file containing analysis results")
    chart_type: str = Field(default="auto", description="Type of chart to create ('auto', 'bar', 'pie', 'box', 'violin', 'scatter', 'line')")
    comparison_metric: str = Field(default="cell_count", description="Metric to compare across groups (e.g., 'cell_count', 'cell_type_distribution', 'confidence_mean')")
    group_column: str = Field(default="group", description="Column name in data that contains group labels")
    output_dir: str = Field(default="output_visualizations", description="Directory to save visualization outputs")
    figure_size: Optional[Tuple[float, float]] = Field(default=None, description="Figure size (width, height) in inches")
    dpi: int = Field(default=300, description="Resolution for saved figures")


class Adapter(ToolAdapter):
    input_model = InputModel

    def run(self, args: InputModel, session_state=None):
        tool = Analysis_Visualizer_Tool()
        return tool.execute(
            analysis_data=args.analysis_data,
            chart_type=args.chart_type,
            comparison_metric=args.comparison_metric,
            group_column=args.group_column,
            output_dir=args.output_dir,
            figure_size=args.figure_size if args.figure_size else (10, 6),
            dpi=args.dpi,
        )


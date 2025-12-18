from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Literal, Any
import uuid


class TaskType(str, Enum):
    CODING = "coding"
    ANALYSIS = "analysis"
    PIPELINE = "pipeline"
    DEBUG = "debug"


@dataclass
class PlanStep:
    id: str
    description: str
    status: Literal["pending", "in_progress", "completed"] = "pending"


@dataclass
class ActiveTask:
    task_id: str
    goal: str
    task_type: TaskType
    plan_steps: List[PlanStep] = field(default_factory=list)
    completed_steps: List[str] = field(default_factory=list)
    artifacts: Dict[str, Any] = field(default_factory=dict)

    @classmethod
    def new(cls, goal: str, task_type: TaskType = TaskType.ANALYSIS) -> "ActiveTask":
        return cls(task_id=str(uuid.uuid4()), goal=goal, task_type=task_type)


@dataclass
class PlanDelta:
    intent: Literal["NEW_TASK", "CONTINUE_TASK", "MODIFY_TASK"]
    added_steps: List[PlanStep] = field(default_factory=list)
    completed_step_ids: List[str] = field(default_factory=list)
    updated_goal: Optional[str] = None
    updated_task_type: Optional[TaskType] = None


@dataclass
class AnalysisInput:
    name: str
    path: str
    input_type: str = "image"  # image, folder, etc.
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnalysisSession:
    inputs: Dict[str, AnalysisInput] = field(default_factory=dict)
    results: Dict[str, Dict[str, Any]] = field(default_factory=dict)
    active_input: Optional[str] = None
    compare_requested: bool = False


@dataclass
class InputDelta:
    new_inputs: Dict[str, AnalysisInput] = field(default_factory=dict)
    set_active: Optional[str] = None
    compare_requested: bool = False


@dataclass
class ConversationState:
    conversation: List[Any] = field(default_factory=list)
    active_task: Optional[ActiveTask] = None
    analysis_session: Optional[AnalysisSession] = None


@dataclass
class BatchImage:
    group: str
    image_id: str
    image_path: str


@dataclass
class CellCrop:
    crop_id: str
    group: str
    image_id: str
    cell_id: str
    image: Any

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
class ConversationState:
    conversation: List[Any] = field(default_factory=list)
    active_task: Optional[ActiveTask] = None


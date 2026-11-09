from typing import Optional, Dict, Any
from pydantic import BaseModel

# Planner: QueryAnalysis
class QueryAnalysis(BaseModel):
    consice_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str
    # 045 Phase 3 (Fix 0): the query type, used to DERIVE the task boundary
    # (in-/out-of-scope stages) deterministically from the stage_map DAG. Default ""
    # -> unclassifiable -> the scope guard fails OPEN. One of: Simple Counting /
    # Basic Morphology / Cell State Analysis / Comparison (free text; normalized).
    query_type: str = ""

    def __str__(self):
        return f"""
Concise Summary: {self.consice_summary}

Query Type: {self.query_type}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""

# Planner: NextStep
# NOTE: keep this schema free of free-form dict fields. OpenAI structured outputs
# (strict JSON schema) reject `Dict[str, Any]` ("additionalProperties must be
# false"), so the VLRR initial-parameter recommendation is carried separately on
# the Planner instance (planner.recommended_params), not as a field here.
class NextStep(BaseModel):
    justification: str
    context: str
    sub_goal: str
    tool_name: str

# Executor: MemoryVerification
class MemoryVerification(BaseModel):
    analysis: str
    stop_signal: bool

# Executor: ToolCommand
class ToolCommand(BaseModel):
    analysis: str
    explanation: str
    command: str



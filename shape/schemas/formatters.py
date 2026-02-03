"""
Structured data formats for agent communication.

These schemas define the interfaces between planner, executor, and tools,
ensuring type safety and consistency across the framework.
"""

from pydantic import BaseModel


class QueryAnalysis(BaseModel):
    """Structured analysis of a user query."""
    consice_summary: str
    required_skills: str
    relevant_tools: str
    additional_considerations: str

    def __str__(self):
        return f"""
Concise Summary: {self.consice_summary}

Required Skills:
{self.required_skills}

Relevant Tools:
{self.relevant_tools}

Additional Considerations:
{self.additional_considerations}
"""


class NextStep(BaseModel):
    """Next step in the agent's plan."""
    justification: str
    context: str
    sub_goal: str
    tool_name: str


class MemoryVerification(BaseModel):
    """Verification of whether memory contains sufficient information."""
    analysis: str
    stop_signal: bool


class ToolCommand(BaseModel):
    """Command to execute a tool."""
    analysis: str
    explanation: str
    command: str


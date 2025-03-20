from pydantic import BaseModel, Field
from typing import Optional


class InputSchema(BaseModel):
    topic: str = Field(..., title="Topic to discuss")


class ResearchInput(BaseModel):
    research_topic: str = Field(..., title="Research topic to investigate")
    num_agents: int = Field(default=3, title="Number of research agents")
    use_roles: bool = Field(default=False, title="Whether to use specialized roles")


class ResearchOutput(BaseModel):
    structured_report: str = Field(..., title="Path to structured research report")
    summary: str = Field(..., title="Path to research summary")
    findings: str = Field(..., title="Path to key findings")

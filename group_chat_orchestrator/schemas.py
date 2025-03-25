from pydantic import BaseModel, Field
from typing import Optional, Dict, Any, List


class InputSchema(BaseModel):
    topic: str = Field(..., title="Topic to discuss")
    num_rounds: int = Field(
        default=3,
        title="Number of rounds for the conversation",
        description="Number of rounds each agent will participate in"
    )
    system_prompt: Optional[str] = Field(
        default=None,
        title="Custom system prompt for the agent",
        description="Optional custom system prompt to override the default one"
    )
    temperature: float = Field(
        default=0.7,
        title="Temperature for LLM responses",
        description="Controls randomness in the model's output"
    )
    max_tokens: int = Field(
        default=1000,
        title="Maximum tokens for LLM responses",
        description="Maximum length of the model's response"
    )
    questions_to_answer: Optional[List[str]] = Field(
        default=None,
        title="Specific questions to answer",
        description="List of questions the agent should focus on answering"
    )


class ResearchInput(BaseModel):
    research_topic: str = Field(..., title="Research topic to investigate")
    num_agents: int = Field(default=3, title="Number of research agents")
    use_roles: bool = Field(default=False, title="Whether to use specialized roles")
    system_prompts: Optional[Dict[str, str]] = Field(
        default=None,
        title="Custom system prompts for each agent",
        description="Dictionary mapping agent numbers to their custom system prompts"
    )
    questions_per_agent: int = Field(
        default=2,
        title="Number of questions each agent should answer",
        description="Number of questions to assign to each subsequent agent"
    )
    temperature: float = Field(
        default=0.7,
        title="Temperature for LLM responses",
        description="Controls randomness in the model's output"
    )
    max_tokens: int = Field(
        default=1000,
        title="Maximum tokens for LLM responses",
        description="Maximum length of the model's response"
    )
    context_window: int = Field(
        default=5,
        title="Number of previous messages to include in context",
        description="Number of previous agent responses to include in context"
    )


class ResearchOutput(BaseModel):
    structured_report: str = Field(..., title="Path to structured research report")
    summary: str = Field(..., title="Path to research summary")
    findings: str = Field(..., title="Path to key findings")
    initial_summary: str = Field(..., title="Path to initial summary from first agent")

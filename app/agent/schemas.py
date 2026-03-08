from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field, ConfigDict


class AgentRequest(BaseModel):
    model_config = ConfigDict(extra="ignore")

    query: str
    llm_engine: Literal["gemini", "ollama", "rule_based"] = "ollama"
    max_steps: int = 4
    return_trace: bool = True


class ToolCall(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)


class PlannerOutput(BaseModel):
    model_config = ConfigDict(extra="ignore")

    thought: str = ""
    action: Optional[ToolCall] = None
    should_stop: bool = False
    final_answer: Optional[str] = None
    intent: Optional[Literal["conversation", "seller_analysis", "product_search", "hybrid"]] = None


class Observation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool: str
    ok: bool = True
    data: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    error: Optional[str] = None


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step: int
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = Field(default_factory=dict)
    observation_summary: str = ""
    status: Literal["ok", "error", "final"] = "ok"


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_query: str
    final_answer: str
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)
    final_data: Dict[str, Any] = Field(default_factory=dict)
    steps_used: int = 0
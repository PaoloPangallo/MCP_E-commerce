from __future__ import annotations

from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, ConfigDict, Field

IntentType = Literal["conversation", "seller_analysis", "product_search", "hybrid"]
ObservationStatus = Literal["ok", "no_data", "error"]
ObservationQuality = Literal["empty", "partial", "good"]


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
    intent: Optional[IntentType] = None


class Observation(BaseModel):
    model_config = ConfigDict(extra="ignore")

    tool: str
    ok: bool = True
    status: ObservationStatus = "ok"
    data: Dict[str, Any] = Field(default_factory=dict)
    summary: str = ""
    error: Optional[str] = None
    retryable: bool = False

    state_key: Optional[str] = None
    state_update: Dict[str, Any] = Field(default_factory=dict)
    terminal: bool = False
    quality: ObservationQuality = "good"


class AgentStep(BaseModel):
    model_config = ConfigDict(extra="ignore")

    step: int
    thought: str = ""
    action: str = ""
    action_input: Dict[str, Any] = Field(default_factory=dict)
    observation_summary: str = ""
    status: Literal["ok", "no_data", "error", "final"] = "ok"


class AgentResponse(BaseModel):
    model_config = ConfigDict(extra="ignore")

    user_query: str
    final_answer: str
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)
    final_data: Dict[str, Any] = Field(default_factory=dict)
    steps_used: int = 0


class StartEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["start"] = "start"
    query: str
    llm_engine: str
    max_steps: int
    planned_tasks: List[Dict[str, Any]] = Field(default_factory=list)


class ThinkingEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["thinking"] = "thinking"
    step: int
    thought: Optional[str] = None
    action: Optional[str] = None
    message: Optional[str] = None


class ToolStartEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["tool_start"] = "tool_start"
    step: int
    tool: str
    input: Dict[str, Any] = Field(default_factory=dict)


class ToolResultEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["tool_result"] = "tool_result"
    step: int
    tool: str
    ok: bool
    status: ObservationStatus = "ok"
    quality: ObservationQuality = "good"
    summary: str = ""


class FinalEvent(BaseModel):
    model_config = ConfigDict(extra="ignore")

    type: Literal["final"] = "final"
    final_answer: str
    agent_trace: List[Dict[str, Any]] = Field(default_factory=list)
    final_data: Dict[str, Any] = Field(default_factory=dict)
    steps_used: int = 0

from __future__ import annotations

from app.agent.schemas import ToolCall, Observation, ObservationStatus, ObservationQuality
from app.agent.tool_registry import ToolContext, TOOLS


class ToolExecutor:
    def __init__(self, context: ToolContext):
        self.context = context

    def execute(self, tool_call: ToolCall) -> Observation:
        spec = TOOLS.get(tool_call.tool)

        if spec is None:
            return Observation(
                tool=tool_call.tool,
                ok=False,
                status="error",
                error=f"Unknown tool '{tool_call.tool}'",
                summary=f"Tool '{tool_call.tool}' non disponibile.",
                retryable=False,
            )

        attempts = max(1, int(spec.max_retries) + 1)
        last_error: Exception | None = None

        for attempt in range(1, attempts + 1):
            try:
                result = spec.executor(tool_call.input, self.context)

                if not isinstance(result, dict):
                    result = {"result": result}

                if spec.result_normalizer:
                    result = spec.result_normalizer(result, tool_call.input)

                status: ObservationStatus = "ok"
                if spec.status_resolver:
                    status = spec.status_resolver(result)

                quality: ObservationQuality = "good"
                if spec.quality_resolver:
                    quality = spec.quality_resolver(result)

                terminal = False
                if spec.terminal_resolver:
                    terminal = spec.terminal_resolver(result)

                summary = spec.summarizer(result) if spec.summarizer else "Tool eseguito."

                result["_tool_attempts"] = attempt

                return Observation(
                    tool=tool_call.tool,
                    ok=status != "error",
                    status=status,
                    data=result,
                    summary=summary,
                    error=result.get("error"),
                    retryable=False,
                    state_key=spec.state_key or None,
                    state_update=result,
                    terminal=terminal,
                    quality=quality,
                )

            except Exception as exc:
                last_error = exc
                if attempt >= attempts:
                    break

        return Observation(
            tool=tool_call.tool,
            ok=False,
            status="error",
            error=str(last_error) if last_error else "Unknown tool execution error",
            summary=f"{tool_call.tool} failed: {last_error}" if last_error else f"{tool_call.tool} failed",
            retryable=False,
            state_key=spec.state_key or None,
            terminal=False,
            quality="empty",
        )

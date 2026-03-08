export interface AgentEvent {
  type:
    | "start"
    | "thinking"
    | "tool_start"
    | "tool_result"
    | "final"
    | "error"
    | "heartbeat"
    | "done"

  step?: number

  message?: string
  thought?: string
  action?: string

  tool?: string
  input?: Record<string, any>

  ok?: boolean
  summary?: string

  final_answer?: string
  final_data?: any
  agent_trace?: any[]
  steps_used?: number
}

export function streamAgent(
  query: string,
  onEvent: (event: AgentEvent) => void
) {

  const url =
    `http://localhost:8040/agent/stream?query=${encodeURIComponent(query)}&llm_engine=ollama`

  const source = new EventSource(url)

  source.onmessage = (event) => {

    if (!event.data) return

    try {

      const data: AgentEvent = JSON.parse(event.data)

      onEvent(data)

      if (
        data.type === "final" ||
        data.type === "done" ||
        data.type === "error"
      ) {
        source.close()
      }

    } catch (err) {

      console.error("SSE parse error", err)

    }

  }

  source.onerror = (err) => {

    console.error("SSE error", err)

    source.close()

  }

  return source
}
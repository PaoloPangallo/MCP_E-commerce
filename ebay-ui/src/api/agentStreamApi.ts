export interface AgentEvent {
  type: "start" | "thinking" | "step" | "final"

  step?: number
  thought?: string
  action?: string
  action_input?: any

  observation?: any

  final_answer?: string
  final_data?: any
  agent_trace?: any[]
}

export function streamAgent(
  query: string,
  onEvent: (event: AgentEvent) => void
) {

  const url =
    `http://localhost:8040/agent/stream?query=${encodeURIComponent(query)}`

  const source = new EventSource(url)

  source.onmessage = (event) => {

    try {

      const data: AgentEvent = JSON.parse(event.data)

      onEvent(data)

      if (data.type === "final") {
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
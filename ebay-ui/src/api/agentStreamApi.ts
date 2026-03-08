import { API_BASE } from "./apiClient"

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
  onEvent: (event: AgentEvent) => void,
  llmEngine = "ollama"
) {
  const url =
    `${API_BASE}/agent/stream?query=${encodeURIComponent(query)}&llm_engine=${encodeURIComponent(llmEngine)}`

  const source = new EventSource(url)

  let closed = false

  const closeOnce = () => {
    if (closed) return
    closed = true
    source.close()
  }

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
        closeOnce()
      }
    } catch (err) {
      console.error("SSE parse error", err)

      onEvent({
        type: "error",
        message: "Errore nel parsing dello stream SSE."
      })

      closeOnce()
    }
  }

  source.onerror = (err) => {
    console.error("SSE error", err)

    onEvent({
      type: "error",
      message: "Connessione SSE interrotta o non disponibile."
    })

    closeOnce()
  }

  return source
}
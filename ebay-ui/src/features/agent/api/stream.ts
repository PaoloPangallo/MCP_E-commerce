import type { AgentEvent } from "../types"
import { API_BASE } from "../../../api/apiClient.ts"
import { getToken } from "../../../auth/authStore.ts"

export function streamAgent(
  query: string,
  onEvent: (event: AgentEvent) => void,
  llmEngine = "ollama"
) {
  const token = getToken()
  const url = `${API_BASE}/agent/stream?query=${encodeURIComponent(query)}&llm_engine=${encodeURIComponent(llmEngine)}${token ? `&token=${encodeURIComponent(token)}` : ""}`
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
    if (closed) return
    console.error("SSE error", err)
    onEvent({
      type: "error",
      message: "Connessione SSE interrotta o non disponibile."
    })
    closeOnce()
  }

  return source
}
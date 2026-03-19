import type { AgentEvent } from "../types"
import { API_BASE } from "../../../api/apiClient.ts"
import { getToken } from "../../../auth/authStore.ts"
import {fetchEventSource} from "@microsoft/fetch-event-source";


/**
 * Streams agent events using @microsoft/fetch-event-source.
 *
 * Switch to POST request to avoid URI length limits on complex queries
 * and safely pass JWT via headers. Handles broken TCP chunks natively.
 */
export function streamAgent(
  query: string,
  onEvent: (event: AgentEvent) => void,
  llmEngine = "ollama_cloud"
) {
  const token = getToken()
  const url = `${API_BASE}/agent/stream`

  const controller = new AbortController()

  const headers: Record<string, string> = {
    "Content-Type": "application/json",
    "Accept": "text/event-stream",
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`
  }

  fetchEventSource(url, {
    method: "POST",
    headers,
    body: JSON.stringify({
      query,
      llm_engine: llmEngine
    }),
    signal: controller.signal,
    openWhenHidden: true, // Keep connection alive even when tab is in background
    
    async onopen(response) {
      if (response.ok && response.headers.get("content-type")?.includes("text/event-stream")) {
        console.log("SSE connection opened successfully")
        return // everything's good
      }
      
      let message = `Errore di connessione (${response.status})`
      try {
          const errorData = await response.json()
          message = errorData.detail || message
      } catch (e) {
          // fallback
      }
      onEvent({ type: "error", message })
      controller.abort()
    },

    onmessage(msg) {
      if (!msg.data) return

      try {
        const data: AgentEvent = JSON.parse(msg.data)
        onEvent(data)

        if (
          data.type === "final" ||
          data.type === "done" ||
          data.type === "error"
        ) {
          controller.abort()
        }
      } catch (err) {
        console.error("SSE parse error", err, msg.data)
      }
    },
    
    onerror(err) {
      console.error("SSE fetch error", err)
      onEvent({ type: "error", message: "Connessione SSE interrotta o non disponibile." })
      controller.abort()
      throw err // Prevent retry mechanism for now to mimic old behavior
    }
  })

  return {
    close: () => controller.abort(),
  }
}

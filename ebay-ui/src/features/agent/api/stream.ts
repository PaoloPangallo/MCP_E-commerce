import type { AgentEvent } from "../types"
import { API_BASE } from "../../../api/apiClient.ts"
import { getToken } from "../../../auth/authStore.ts"

/**
 * Streams agent events using fetch() + ReadableStream instead of EventSource.
 *
 * Why not EventSource?
 * - EventSource (SSE) does NOT support custom headers in browsers.
 * - Passing the JWT as ?token= in the URL is a security risk:
 *   tokens appear in server logs, proxy logs, and ngrok dashboards.
 * - fetch() allows us to send Authorization: Bearer header securely.
 */
export function streamAgent(
  query: string,
  onEvent: (event: AgentEvent) => void,
  llmEngine = "ollama"
) {
  const token = getToken()
  const url = `${API_BASE}/agent/stream?query=${encodeURIComponent(query)}&llm_engine=${encodeURIComponent(llmEngine)}`

  const controller = new AbortController()

  const headers: Record<string, string> = {
    "Accept": "text/event-stream",
    "Cache-Control": "no-cache",
  }

  if (token) {
    headers["Authorization"] = `Bearer ${token}`
  }

  fetch(url, {
    method: "GET",
    headers,
    signal: controller.signal,
  })
    .then(async (response) => {
      if (!response.ok || !response.body) {
        onEvent({ type: "error", message: `HTTP error ${response.status}` })
        return
      }

      const reader = response.body.getReader()
      const decoder = new TextDecoder()
      let buffer = ""

      while (true) {
        const { done, value } = await reader.read()
        if (done) break

        buffer += decoder.decode(value, { stream: true })

        // SSE lines are separated by \n\n
        const parts = buffer.split("\n\n")
        buffer = parts.pop() ?? ""

        for (const part of parts) {
          const line = part.trim()
          if (!line.startsWith("data:")) continue

          const jsonStr = line.slice(5).trim()
          if (!jsonStr) continue

          try {
            const data: AgentEvent = JSON.parse(jsonStr)
            onEvent(data)

            if (
              data.type === "final" ||
              data.type === "done" ||
              data.type === "error"
            ) {
              controller.abort()
              return
            }
          } catch (err) {
            console.error("SSE parse error", err)
          }
        }
      }
    })
    .catch((err) => {
      if (err.name === "AbortError") return
      console.error("SSE fetch error", err)
      onEvent({ type: "error", message: "Connessione SSE interrotta o non disponibile." })
    })

  // Return an object compatible with the previous EventSource API
  return {
    close: () => controller.abort(),
  }
}
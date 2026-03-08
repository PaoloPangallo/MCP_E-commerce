import type { AgentResponse } from "../types"
import {apiFetch} from "../../../api/apiClient.ts";

export async function searchProducts(
  query: string,
  llmEngine = "ollama"
): Promise<AgentResponse> {
  return apiFetch<AgentResponse>("/agent", {
    method: "POST",
    timeout: 120000,
    body: JSON.stringify({
      query,
      llm_engine: llmEngine,
      max_steps: 6,
      return_trace: true
    })
  })
}

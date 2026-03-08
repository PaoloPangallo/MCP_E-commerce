import { apiFetch } from "./apiClient"
import type {
  AgentStep,
  IRMetrics,
  RagContext,
  SearchItem,
  SellerSummaryBlock
} from "../types/searchTypes.ts"

export interface ParsedQuery {
  semantic_query?: string
  product?: string | null
  brands?: string[]
  constraints?: any[]
  preferences?: any[]
  compatibilities?: Record<string, string>
}

export interface SearchPipelineResponse {
  parsed_query?: ParsedQuery
  ebay_query_used?: string
  results: SearchItem[]
  results_count?: number
  saved_new_count?: number
  analysis?: string
  rag_context?: RagContext
  metrics?: IRMetrics
  _timings?: Record<string, number>
}

export interface SellerPipelineResponse extends SellerSummaryBlock {
  page?: number
  limit?: number
  feedbacks?: any[]
}

export interface AgentResponse {
  user_query: string
  final_answer: string
  agent_trace?: AgentStep[]
  final_data?: {
    search?: SearchPipelineResponse
    seller?: SellerPipelineResponse
    top_result?: SearchItem | null
    last_seller_name?: string | null
    errors?: string[]
  }
  steps_used?: number
}

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
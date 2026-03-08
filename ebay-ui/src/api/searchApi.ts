import { apiFetch } from "./apiClient"
import type { IRMetrics, RagContext } from "../component/searchTypes"

export interface ParsedQuery {
  semantic_query?: string
  product?: string | null
  brands?: string[]
  constraints?: any[]
  preferences?: any[]
  compatibilities?: Record<string, string>
}

export interface SearchItem {
  ebay_id: string
  title: string
  price: number
  currency: string
  condition: string
  seller_name: string
  seller_rating: number
  url: string
  image_url: string
  trust_score?: number
  ranking_score?: number
  explanations?: string[]
  rag_feedback?: any[]
  _already_in_db?: boolean
}

export interface SearchResponse {
  parsed_query?: ParsedQuery
  ebay_query_used?: string
  results: SearchItem[]
  results_count?: number
  saved_new_count?: number
  analysis?: string
  rag_context?: RagContext
  metrics?: IRMetrics
  _timings?: Record<string, number>
  thinking_trace?: string[]
}

export interface ChatMessage {
  role: string
  content: string
}

export async function searchProducts(
  query: string,
  history: ChatMessage[] = [],
  context: Record<string, any> = {}
): Promise<SearchResponse> {
  return apiFetch<SearchResponse>("/search", {
    method: "POST",
    timeout: 180000,
    body: JSON.stringify({
      query,
      llm_engine: "ollama",
      history,
      context
    })
  })
}
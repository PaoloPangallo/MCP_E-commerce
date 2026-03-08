import type { AgentStep } from "../agent/types"
import type { AppMode } from "../../types"
import type { Feedback, SellerSummaryBlock } from "../seller/types"

export interface SearchItem {
  ebay_id?: string
  title?: string
  price?: number
  currency?: string
  condition?: string
  image_url?: string
  url?: string
  seller_name?: string
  seller_rating?: number
  trust_score?: number
  ranking_score?: number
  explanations?: string[]
  rag_feedback?: Feedback[]
  _already_in_db?: boolean
}

export interface IRMetrics {
  "precision@5"?: number
  "precision@10"?: number
  "recall@10"?: number
  "ndcg@10"?: number
}

export type RagContext = string | string[] | null | undefined

export interface SearchBlock {
  query: string
  results: SearchItem[]
  analysis: string | null
  metrics?: IRMetrics
  rag_context?: RagContext
  timings?: Record<string, number>
  agent_trace?: AgentStep[]
  seller_summary?: SellerSummaryBlock | null
  final_answer?: string | null
  mode?: AppMode
  errors?: string[]
}

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

export interface AgentResponse {
  user_query: string
  final_answer: string
  agent_trace?: AgentStep[]
  final_data?: {
    search?: SearchPipelineResponse
    seller?: import("../seller/types").SellerFeedbackResponse
    top_result?: SearchItem | null
    last_seller_name?: string | null
    errors?: string[]
  }
  steps_used?: number
}

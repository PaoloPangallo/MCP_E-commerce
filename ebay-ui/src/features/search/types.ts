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
  _scores?: {
    overall?: number
    price?: number
    trust?: number
    relevance?: number
    condition?: number
  }
}

export interface ComparisonCandidate extends SearchItem {
  query: string
  scores?: {
    price: number
    trust: number
    relevance: number
    condition: number
    overall: number
  }
}

export interface ItemDetailsData {
  item_id?: string
  title?: string
  short_description?: string
  description?: string
  category_path?: string
  condition?: string
  item_specifics?: any[]
  return_terms?: Record<string, any>
  shipping_options?: any[]
  seller?: Record<string, any>
  price?: Record<string, any>
  brand?: string
  color?: string
  mpn?: string
  [key: string]: any
}

export interface ShippingCostsData {
  item_id?: string
  shipping_options?: any[]
  item_location?: Record<string, any>
  estimated_delivery?: any[]
  [key: string]: any
}

export interface ComparisonData {
  status: string
  queries_compared: number
  candidates_found: number
  winner: ComparisonCandidate
  winner_reason: string
  comparison_matrix: ComparisonCandidate[]
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
  comparison?: ComparisonData | null
  item_details?: ItemDetailsData | null
  shipping_costs?: ShippingCostsData | null
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
    item_details?: ItemDetailsData
    shipping_costs?: ShippingCostsData
    top_result?: SearchItem | null
    last_seller_name?: string | null
    errors?: string[]
  }
  steps_used?: number
}

import type { AgentStep } from "../agent/types"
export type AppMode = "search" | "seller" | "hybrid" | "conversation" | "comparison"
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
  ner_attributes?: {
    brand?: string | null
    model?: string | null
    specs?: Record<string, any> | null
    category?: string | null
  }
  _already_in_db?: boolean
  _scores?: {
    overall?: number
    price?: number
    trust?: number
    relevance?: number
    condition?: number
    value?: number
  }
  shipping?: {
    cost: number
    currency: string
    free: boolean
    service: string
    min_delivery?: string
    max_delivery?: string
    options_count?: number
  } | null
}

export interface ComparisonCandidate extends SearchItem {
  query: string
  scores?: {
    price: number
    trust: number
    relevance: number
    condition: number
    overall: number
    value?: number
  }
  shipping?: {
    cost: number
    currency: string
    free: boolean
    service: string
    min_delivery?: string
    max_delivery?: string
    options_count?: number
  } | null
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
  image?: Record<string, any>
  additional_images?: any[]
  item_url?: string
  similar_items?: SearchItem[]
  [key: string]: any
}

export interface DealItem {
  title: string
  link: string
  price: {
    raw: string
    extracted: number
  }
  old_price?: {
    raw: string
    extracted: number
    discount: string
  }
  thumbnail?: string
  product_id?: string
  extensions?: string[]
}

export interface DealsData {
  status: string
  title?: string
  subtitle?: string
  deals: DealItem[]
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
  winner: ComparisonCandidate & { value_score?: number }
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
  deals?: DealsData | null
  metadata?: any | null
  final_answer?: string | null
  vision_analysis?: {
    description: string
    tags: string[]
    brand?: string | null
    condition_clues?: string | null
    confidence: number
  } | null
  market_trends?: {
    status: string
    query: string
    shopping_data: {
      status: string
      min_price?: number
      max_price?: number
      average_price?: number
      samples?: number
      top_result?: any
    }
    trends_data: {
      status: string
      current_interest?: number
      trend_direction?: string
      interest_graph?: { date: string, value: number }[]
    }
    verdetto?: string
  } | null
  aspect_distributions?: any[]
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
  aspect_distributions?: any[]
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
    deals?: DealsData
    metadata?: any
    top_result?: SearchItem | null
    last_seller_name?: string | null
    errors?: string[]
  }
  steps_used?: number
}

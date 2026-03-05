import { apiFetch } from "./apiClient"


// ============================================================
// FEEDBACK TYPES
// ============================================================

export type Feedback = {
  user: string
  rating: number
  comment: string
  time: string
}

export interface SellerFeedbackResponse {
  seller: string
  feedbacks: Feedback[]
}


// ============================================================
// SEARCH TYPES
// ============================================================

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

export interface IRMetrics {
  "precision@5": number
  "precision@10": number
  "recall@10": number
  "ndcg@10": number
}

export interface SearchResponse {

  parsed_query?: ParsedQuery

  ebay_query_used?: string

  results: SearchItem[]

  results_count?: number

  saved_new_count?: number

  analysis?: string

  rag_context?: string

  metrics?: IRMetrics

  _timings?: Record<string, number>
}


// ============================================================
// API CALLS
// ============================================================

export async function fetchSellerFeedback(
  seller: string,
  page = 1,
  limit = 10
): Promise<SellerFeedbackResponse> {

  const safeSeller = encodeURIComponent(seller)

  return apiFetch(
    `/seller/${safeSeller}/feedback?page=${page}&limit=${limit}`
  )
}


export async function searchProducts(
  query: string
): Promise<SearchResponse> {

  return apiFetch("/search", {

    method: "POST",

    body: JSON.stringify({
      query,
      llm_engine: "ollama",
    })

  })
}
export interface Message {
  role: "user" | "assistant"
  content: string
  timestamp?: string
}

export interface Feedback {
  user: string
  rating: "Positive" | "Neutral" | "Negative"
  comment: string
  time: string
}

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
  thinking_trace?: string[]
}

export type ChatEntry =
  | { type: "message"; msg: Message }
  | { type: "search"; search: SearchBlock }
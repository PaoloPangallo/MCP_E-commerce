export interface Feedback {
  user?: string
  comment: string
  rating?: number
  date?: string
  time?: string
  sentiment?: number
}

export interface SellerSummaryBlock {
  seller_name?: string
  trust_score?: number
  sentiment_score?: number
  count?: number
  feedbacks?: Feedback[]
  page?: number
  limit?: number
}
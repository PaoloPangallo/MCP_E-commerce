import { apiFetch } from "./apiClient"

// ==============================
// TYPES
// ==============================

export type Feedback = {
  user: string
  rating: "Positive" | "Neutral" | "Negative"
  comment: string
  time: string
}

export interface SellerFeedbackResponse {

  seller_name: string

  page: number
  limit: number
  count: number

  feedbacks: Feedback[]

  trust_score: number
  sentiment_score: number
}

// ==============================
// API
// ==============================

export async function fetchSellerFeedback(
  seller: string,
  page = 1,
  limit = 5
): Promise<SellerFeedbackResponse> {

  const safeSeller = encodeURIComponent(seller)

  return apiFetch<SellerFeedbackResponse>(
    `/seller/${safeSeller}/feedback?page=${page}&limit=${limit}`
  )
}
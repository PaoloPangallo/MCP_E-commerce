export type Feedback = {
  user: string
  rating: string
  comment: string
  time: string
}

export interface SellerFeedbackResponse {
  seller: string
  feedbacks: Feedback[]
}

export async function fetchSellerFeedback(
  seller: string
): Promise<Feedback[]> {

  const safeSeller = encodeURIComponent(seller)

  const response = await fetch(
    `http://127.0.0.1:8000/seller/${safeSeller}/feedback`
  )

  if (!response.ok) {
    throw new Error("Errore API feedback")
  }

  const data: SellerFeedbackResponse = await response.json()

  return data?.feedbacks ?? []
}
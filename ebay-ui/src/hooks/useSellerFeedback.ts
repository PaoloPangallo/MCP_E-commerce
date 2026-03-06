import { useState } from "react"

export interface SellerFeedback {
  comment: string
  sentiment_score?: number
  date?: string
}

export function useSellerFeedback(seller?: string) {

  const [feedbacks, setFeedbacks] = useState<SellerFeedback[]>([])
  const [loading, setLoading] = useState(false)

  const loadFeedback = async () => {

    if (!seller) return

    if (feedbacks.length > 0) return

    try {

      setLoading(true)

      const res = await fetch(
        `http://127.0.0.1:8030/seller/${encodeURIComponent(seller)}/feedback`
      )

      const data = await res.json()

      setFeedbacks(data.feedbacks || [])

    } catch (err) {

      console.error("Feedback load error", err)

    } finally {

      setLoading(false)

    }

  }

  return {
    feedbacks,
    loading,
    loadFeedback
  }
}
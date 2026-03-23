import { useRef, useState } from "react"
import { fetchSellerFeedback } from "../api/sellerApi"
import type { Feedback } from "../types"

export function useSellerFeedback() {
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const cache = useRef<Record<string, Feedback[]>>({})

  const getFeedback = async (seller: string): Promise<Feedback[]> => {
    if (!seller) return []
    if (cache.current[seller]) return cache.current[seller]

    setLoading(true)
    setError(null)

    try {
      const res = await fetchSellerFeedback(seller)
      const data: Feedback[] = res.feedbacks ?? []
      cache.current[seller] = data
      return data
    } catch (err) {
      console.error("Feedback fetch error:", err)
      setError("Errore nel caricamento feedback")
      return []
    } finally {
      setLoading(false)
    }
  }

  return {
    getFeedback,
    loading,
    error
  }
}

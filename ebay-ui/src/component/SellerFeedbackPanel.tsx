import { useEffect, useState } from "react"
import {
  Box,
  Typography,
  Button,
  CircularProgress,
  Divider,
  Chip
} from "@mui/material"

import FeedbackCard from "./FeedbackCard"
import SellerTrustGauge from "./SellerTrustGauge"
import {
  fetchSellerFeedback,
  type Feedback,
  type SellerFeedbackResponse
} from "../api/sellerApi"

interface Props {
  seller?: string
}

const PAGE_SIZE = 5

export default function SellerFeedbackPanel({ seller }: Props) {
  const [feedbacks, setFeedbacks] = useState<Feedback[]>([])
  const [loading, setLoading] = useState(false)
  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)
  const [trustScore, setTrustScore] = useState<number | null>(null)
  const [sentimentScore, setSentimentScore] = useState<number | null>(null)

  const loadFeedback = async (pageToLoad: number) => {
    if (!seller) {
      return
    }

    try {
      setLoading(true)

      const data: SellerFeedbackResponse = await fetchSellerFeedback(
        seller,
        pageToLoad,
        PAGE_SIZE
      )

      const newFeedbacks: Feedback[] = data.feedbacks ?? []

      setTrustScore(
        typeof data.trust_score === "number" ? data.trust_score : null
      )
      setSentimentScore(
        typeof data.sentiment_score === "number" ? data.sentiment_score : null
      )

      if (pageToLoad === 1) {
        setFeedbacks(newFeedbacks)
      } else {
        setFeedbacks((prev) => [...prev, ...newFeedbacks])
      }

      setHasMore(newFeedbacks.length === PAGE_SIZE)
    } catch (err) {
      console.error("Feedback load error:", err)

      if (pageToLoad === 1) {
        setFeedbacks([])
        setTrustScore(null)
        setSentimentScore(null)
      }

      setHasMore(false)
    } finally {
      setLoading(false)
    }
  }

  useEffect(() => {
    if (!seller) {
      setFeedbacks([])
      setTrustScore(null)
      setSentimentScore(null)
      setPage(1)
      setHasMore(false)
      return
    }

    setFeedbacks([])
    setTrustScore(null)
    setSentimentScore(null)
    setPage(1)
    setHasMore(true)
    loadFeedback(1)
  }, [seller])

  const loadMore = () => {
    const nextPage = page + 1
    setPage(nextPage)
    loadFeedback(nextPage)
  }

  return (
    <Box mt={2}>
      <Typography variant="h6" fontWeight={600} mb={2}>
        Seller analysis
      </Typography>

      {trustScore !== null && <SellerTrustGauge score={trustScore} />}

      {sentimentScore !== null && (
        <Box mt={1} mb={2}>
          <Chip
            label={`Sentiment score: ${sentimentScore.toFixed(2)}`}
            size="small"
            sx={{
              bgcolor: "#f5f5f5",
              border: "1px solid #e5e5e5"
            }}
          />
        </Box>
      )}

      {loading && feedbacks.length === 0 && (
        <Box display="flex" justifyContent="center" py={3}>
          <CircularProgress size={28} />
        </Box>
      )}

      {!loading && feedbacks.length === 0 && (
        <Typography variant="body2" color="text.secondary">
          Nessun feedback disponibile
        </Typography>
      )}

      {feedbacks.map((fb, index) => (
        <Box key={`${fb.user}-${fb.time}-${index}`} mb={2}>
          <FeedbackCard feedback={fb} />
          {index !== feedbacks.length - 1 && <Divider sx={{ mt: 2 }} />}
        </Box>
      ))}

      {hasMore && !loading && feedbacks.length > 0 && (
        <Box mt={2} textAlign="center">
          <Button variant="outlined" onClick={loadMore} sx={{ borderRadius: 999 }}>
            Carica altri feedback
          </Button>
        </Box>
      )}

      {loading && feedbacks.length > 0 && (
        <Box mt={2} textAlign="center">
          <CircularProgress size={22} />
        </Box>
      )}
    </Box>
  )
}
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
import type {Feedback} from "./searchTypes.ts";
import {fetchSellerFeedback, type SellerFeedbackResponse} from "../api/sellerApi.ts";


interface Props {
  seller?: string
}

export default function SellerFeedbackPanel({ seller }: Props) {

  const [feedbacks, setFeedbacks] = useState<Feedback[]>([])
  const [loading, setLoading] = useState(false)

  const [page, setPage] = useState(1)
  const [hasMore, setHasMore] = useState(true)

  const [trustScore, setTrustScore] = useState<number | null>(null)
  const [sentimentScore, setSentimentScore] = useState<number | null>(null)

  const loadFeedback = async (pageToLoad: number) => {

    if (!seller) return

    try {

      setLoading(true)

      const data: SellerFeedbackResponse =
        await fetchSellerFeedback(seller, pageToLoad, 5)

      const newFeedbacks = data.feedbacks ?? []

      setTrustScore(data.trust_score)
      setSentimentScore(data.sentiment_score)

      if (pageToLoad === 1) {
        setFeedbacks(newFeedbacks)
      } else {
        setFeedbacks(prev => [...prev, ...newFeedbacks])
      }

      if (newFeedbacks.length < 5) {
        setHasMore(false)
      }

    } catch (err) {

      console.error("Feedback load error:", err)

    } finally {

      setLoading(false)

    }

  }

  useEffect(() => {

    if (!seller) return

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

    <Box mt={3}>

      <Typography variant="h6" fontWeight={600} mb={2}>
        Seller analysis
      </Typography>

      {trustScore !== null && (
        <SellerTrustGauge score={trustScore} />
      )}

      {sentimentScore !== null && (

        <Box mt={1} mb={2}>

          <Chip
            label={`Sentiment score: ${sentimentScore.toFixed(2)}`}
            size="small"
          />

        </Box>

      )}

      {loading && feedbacks.length === 0 && (

        <Box display="flex" justifyContent="center" py={3}>
          <CircularProgress size={28} />
        </Box>

      )}

      {feedbacks.map((fb, i) => (

        <Box key={`${fb.user}-${i}`} mb={2}>

          <FeedbackCard feedback={fb} />

          {i !== feedbacks.length - 1 && (
            <Divider sx={{ mt: 2 }} />
          )}

        </Box>

      ))}

      {hasMore && !loading && (

        <Box mt={2} textAlign="center">

          <Button
            variant="outlined"
            onClick={loadMore}
          >
            Carica altri feedback
          </Button>

        </Box>

      )}

    </Box>

  )
}
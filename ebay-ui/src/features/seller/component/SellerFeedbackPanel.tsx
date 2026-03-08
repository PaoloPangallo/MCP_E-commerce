import { useEffect, useMemo, useState } from "react"
import {
  Box,
  Button,
  CircularProgress,
  Divider,
  Typography,
  Chip
} from "@mui/material"

import SellerTrustGauge from "./SellerTrustGauge.tsx"
import SellerFeedbackList from "../SellerFeedbackList.tsx"
import type { Feedback } from "../../../types"
import { API_BASE } from "../../../api/apiClient.ts"

interface ApiResponse {
  seller_name?: string
  feedbacks?: Feedback[]
  feedback?: Feedback[]
  trust_score?: number
  sentiment_score?: number
  error?: string
}

interface Props {
  seller?: string
}

async function fetchSellerFeedback(seller: string): Promise<ApiResponse> {
  const encodedSeller = encodeURIComponent(seller)

  const candidateUrls = [
    `${API_BASE}/seller/${encodedSeller}/feedback`,
    `${API_BASE}/seller-feedback?seller=${encodedSeller}`
  ]

  let lastError: Error | null = null

  for (const url of candidateUrls) {
    try {
      const res = await fetch(url)

      if (!res.ok) {
        throw new Error(`HTTP ${res.status} on ${url}`)
      }

      const data = (await res.json()) as ApiResponse
      return data
    } catch (err) {
      lastError = err instanceof Error ? err : new Error("Unknown fetch error")
      console.warn("Seller feedback fetch failed for:", url, lastError)
    }
  }

  throw lastError ?? new Error("Unable to load seller feedback")
}

export default function SellerFeedbackPanel({ seller }: Props) {
  const [open, setOpen] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const [feedbacks, setFeedbacks] = useState<Feedback[]>([])
  const [trustScore, setTrustScore] = useState<number | null>(null)
  const [sentimentScore, setSentimentScore] = useState<number | null>(null)

  useEffect(() => {
    setOpen(false)
    setFeedbacks([])
    setTrustScore(null)
    setSentimentScore(null)
    setError(null)
  }, [seller])

  const positive = useMemo(
    () => feedbacks.filter((item) => (item.rating ?? 0) >= 4),
    [feedbacks]
  )

  const negative = useMemo(
    () => feedbacks.filter((item) => (item.rating ?? 0) <= 2),
    [feedbacks]
  )

  const neutral = Math.max(
    feedbacks.length - positive.length - negative.length,
    0
  )

  const loadFeedback = async () => {
    if (!seller || loading) return

    if (feedbacks.length > 0) {
      setOpen((prev) => !prev)
      return
    }

    try {
      setLoading(true)
      setError(null)

      const data = await fetchSellerFeedback(seller)

      if (data.error) {
        throw new Error(data.error)
      }

      const items = Array.isArray(data.feedbacks)
        ? data.feedbacks
        : Array.isArray(data.feedback)
          ? data.feedback
          : []

      setFeedbacks(items)

      setTrustScore(
        typeof data.trust_score === "number" ? data.trust_score : null
      )

      setSentimentScore(
        typeof data.sentiment_score === "number" ? data.sentiment_score : null
      )

      setOpen(true)
    } catch (err) {
      console.error("SellerFeedbackPanel error:", err)
      setError("Errore nel caricamento dell'analisi venditore")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box mt={2}>
      <Button
        size="small"
        variant="outlined"
        disabled={loading || !seller}
        onClick={loadFeedback}
        sx={{
          textTransform: "none",
          borderRadius: "16px",
          borderColor: "#e5e5e5",
          color: "#0d0d0d"
        }}
      >
        {loading ? (
          <CircularProgress size={16} />
        ) : open ? (
          "Nascondi analisi venditore"
        ) : (
          "Analizza venditore"
        )}
      </Button>

      {error && !open ? (
        <Typography sx={{ mt: 1, color: "#c62828", fontSize: 13 }}>
          {error}
        </Typography>
      ) : null}

      {open ? (
        <Box
          mt={2}
          sx={{
            p: 2.5,
            backgroundColor: "#f8f8f8",
            borderRadius: "12px",
            border: "1px solid #ececec"
          }}
        >
          <Box
            display="flex"
            alignItems="flex-start"
            justifyContent="space-between"
            gap={2}
            flexWrap="wrap"
          >
            <Box>
              <Typography
                sx={{
                  fontSize: 13,
                  fontWeight: 700,
                  mb: 0.75
                }}
              >
                Seller analysis
              </Typography>

              <Box display="flex" gap={1} flexWrap="wrap">
                <Chip
                  label={`${positive.length} positive`}
                  size="small"
                  sx={{ fontSize: 11 }}
                />
                <Chip
                  label={`${negative.length} negative`}
                  size="small"
                  sx={{ fontSize: 11 }}
                />
                <Chip
                  label={`${neutral} neutral`}
                  size="small"
                  sx={{ fontSize: 11 }}
                />
              </Box>
            </Box>

            <Box sx={{ minWidth: 180, flex: 1, maxWidth: 320 }}>
              <SellerTrustGauge score={trustScore ?? 0} />
            </Box>
          </Box>

          {sentimentScore !== null ? (
            <Typography sx={{ fontSize: 13, color: "#444", mt: 1.5 }}>
              Sentiment score: {sentimentScore.toFixed(2)}
            </Typography>
          ) : null}

          <Divider sx={{ my: 2 }} />

          {positive.length > 0 ? (
            <Box mb={2}>
              <SellerFeedbackList
                feedbacks={positive.slice(0, 3)}
                initialLimit={3}
                title="Top positive feedback"
              />
            </Box>
          ) : null}

          {negative.length > 0 ? (
            <Box mb={2}>
              <SellerFeedbackList
                feedbacks={negative.slice(0, 3)}
                initialLimit={3}
                title="Top negative feedback"
              />
            </Box>
          ) : null}

          <SellerFeedbackList
            feedbacks={feedbacks}
            title="Tutti i feedback"
          />
        </Box>
      ) : null}
    </Box>
  )
}
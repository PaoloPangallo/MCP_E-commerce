import { useEffect, useMemo, useState } from "react"
import { Box, CircularProgress, Collapse, Typography } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import WarningAmberIcon from "@mui/icons-material/WarningAmber"

import SellerTrustGauge from "./SellerTrustGauge.tsx"
import SellerFeedbackList from "../SellerFeedbackList.tsx"
import type { Feedback } from "../types"
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
  const enc = encodeURIComponent(seller)
  const urls = [
    `${API_BASE}/seller/${enc}/feedback`,
    `${API_BASE}/seller-feedback?seller=${enc}`
  ]
  let lastError: Error | null = null
  for (const url of urls) {
    try {
      const res = await fetch(url)
      if (!res.ok) throw new Error(`HTTP ${res.status}`)
      return (await res.json()) as ApiResponse
    } catch (err) {
      lastError = err instanceof Error ? err : new Error("Unknown error")
    }
  }
  throw lastError ?? new Error("Unable to load seller feedback")
}

export default function SellerFeedbackPanel({ seller }: Props) {
  const [open, setOpen]               = useState(false)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState<string | null>(null)
  const [feedbacks, setFeedbacks]     = useState<Feedback[]>([])
  const [trustScore, setTrustScore]   = useState<number | null>(null)
  const [sentiment, setSentiment]     = useState<number | null>(null)

  useEffect(() => {
    setOpen(false)
    setFeedbacks([])
    setTrustScore(null)
    setSentiment(null)
    setError(null)
  }, [seller])

  const positive = useMemo(() => feedbacks.filter((f) => (f.rating ?? 0) >= 4), [feedbacks])
  const negative = useMemo(() => feedbacks.filter((f) => (f.rating ?? 0) <= 2), [feedbacks])
  const neutral  = Math.max(feedbacks.length - positive.length - negative.length, 0)

  const flaggedFeedbacks = useMemo(() => feedbacks.filter((f) => {
    const type = (f.rating ?? 0) >= 4 ? "positive" : (f.rating ?? 0) <= 2 ? "negative" : "neutral";
    const nlp = f.nlp_sentiment;
    const isFalsePos = type === "positive" && nlp !== undefined && nlp < 0.40;
    const isFalseNeg = type === "negative" && nlp !== undefined && nlp > 0.60;
    return isFalsePos || isFalseNeg;
  }), [feedbacks])

  const handleToggle = async () => {
    if (!seller || loading) return

    // If already loaded, just toggle visibility
    if (feedbacks.length > 0) {
      setOpen((v) => !v)
      return
    }

    try {
      setLoading(true)
      setError(null)
      const data = await fetchSellerFeedback(seller)
      if (data.error) throw new Error(data.error)

      const items = Array.isArray(data.feedbacks)
        ? data.feedbacks
        : Array.isArray(data.feedback)
          ? data.feedback
          : []

      setFeedbacks(items)
      setTrustScore(typeof data.trust_score === "number" ? data.trust_score : null)
      setSentiment(typeof data.sentiment_score === "number" ? data.sentiment_score : null)
      setOpen(true)
    } catch (err) {
      setError("Errore nel caricamento dell'analisi venditore")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Box>
      {/* Toggle button */}
      <Box
        component="button"
        onClick={handleToggle}
        disabled={!seller || loading}
        sx={{
          background: "none",
          border: "none",
          p: 0,
          cursor: seller && !loading ? "pointer" : "default",
          display: "inline-flex",
          alignItems: "center",
          gap: 0.5,
          fontFamily: "inherit"
        }}
      >
        {loading ? (
          <CircularProgress size={12} sx={{ color: "#9ca3af" }} />
        ) : (
          <>
            <Typography
              sx={{
                fontSize: 12,
                color: "#6b7280",
                textDecoration: "underline",
                textDecorationColor: "#e5e7eb",
                textUnderlineOffset: "3px",
                "&:hover": { color: "#374151" }
              }}
            >
              {open ? "Nascondi analisi" : "Analisi venditore"}
            </Typography>
            <KeyboardArrowDownIcon
              sx={{
                fontSize: 14,
                color: "#9ca3af",
                transform: open ? "rotate(180deg)" : "none",
                transition: "transform 0.2s"
              }}
            />
          </>
        )}
      </Box>

      {/* Error */}
      {error && (
        <Typography sx={{ mt: 0.75, fontSize: 12, color: "#dc2626" }}>{error}</Typography>
      )}

      {/* Panel */}
      <Collapse in={open} timeout={200}>
        <Box
          sx={{
            mt: 1.25,
            p: 1.75,
            border: "1px solid #f0f0f0",
            borderRadius: 3,
            bgcolor: "#fafafa",
            display: "flex",
            flexDirection: "column",
            gap: 1.5
          }}
        >
          {/* Trust gauge */}
          {trustScore !== null && (
            <Box sx={{ maxWidth: 260 }}>
              <SellerTrustGauge score={trustScore} />
            </Box>
          )}

          {/* Sentiment + counts */}
          <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap" }}>
            {sentiment !== null && (
              <Box>
                <Typography sx={{ fontSize: 10, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                  Sentiment
                </Typography>
                <Typography sx={{ fontSize: 13, fontWeight: 500, color: "#374151" }}>
                  {Math.round(sentiment * 100)}%
                </Typography>
              </Box>
            )}
            <Box>
              <Typography sx={{ fontSize: 10, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>
                Distribuzione
              </Typography>
              <Box sx={{ display: "flex", gap: 0.75, mt: 0.25, flexWrap: "wrap" }}>
                {[
                  { label: `${positive.length} pos`, color: "#059669" },
                  { label: `${neutral} neu`,          color: "#d97706" },
                  { label: `${negative.length} neg`,  color: "#dc2626" }
                ].map((item) => (
                  <Box
                    key={item.label}
                    sx={{
                      px: 0.75,
                      py: 0.15,
                      borderRadius: "6px",
                      bgcolor: "#f3f4f6",
                      border: "1px solid #e5e7eb"
                    }}
                  >
                    <Typography sx={{ fontSize: 11, color: item.color }}>{item.label}</Typography>
                  </Box>
                ))}
              </Box>
            </Box>
            {flaggedFeedbacks.length > 0 && (
              <Box sx={{ width: '100%', display: 'flex', alignItems: 'center', gap: 0.75, mt: 1, p: 1, bgcolor: '#fef2f2', border: '1px solid #fca5a5', borderRadius: '6px' }}>
                <WarningAmberIcon sx={{ fontSize: 14, color: '#dc2626' }} />
                <Typography sx={{ fontSize: 10, fontWeight: 600, color: '#991b1b', lineHeight: 1.3 }}>
                  L'Intelligenza Artificiale ha rilevato <strong>{flaggedFeedbacks.length}</strong> recensioni il cui testo contraddice il rating rilasciato.
                </Typography>
              </Box>
            )}
          </Box>

          {/* Feedback lists */}
          <Box sx={{ borderTop: "1px solid #f0f0f0", pt: 1 }}>
            {positive.length > 0 && (
              <Box sx={{ mb: 1.5 }}>
                <Typography sx={{ fontSize: 11, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.75 }}>
                  Top positivi
                </Typography>
                <SellerFeedbackList feedbacks={positive.slice(0, 3)} initialLimit={3} title="" />
              </Box>
            )}
            {negative.length > 0 && (
              <Box sx={{ mb: 1.5 }}>
                <Typography sx={{ fontSize: 11, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.75 }}>
                  Da attenzionare
                </Typography>
                <SellerFeedbackList feedbacks={negative.slice(0, 3)} initialLimit={3} title="" />
              </Box>
            )}
            <Typography sx={{ fontSize: 11, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.75 }}>
              Tutti i feedback
            </Typography>
            <SellerFeedbackList feedbacks={feedbacks} title="" />
          </Box>
        </Box>
      </Collapse>
    </Box>
  )
}
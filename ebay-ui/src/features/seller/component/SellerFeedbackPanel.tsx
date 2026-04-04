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
          <CircularProgress size={12} sx={{ color: "var(--text-secondary)" }} />
        ) : (
          <>
            <Typography
              sx={{
                fontSize: 12,
                color: "var(--text-secondary)",
                textDecoration: "underline",
                textDecorationColor: "var(--border-color)",
                textUnderlineOffset: "3px",
                "&:hover": { color: "var(--text-primary)" }
              }}
            >
              {open ? "Nascondi analisi" : "Analisi venditore"}
            </Typography>
            <KeyboardArrowDownIcon
              sx={{
                fontSize: 14,
                color: "var(--text-secondary)",
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
            border: "1px solid var(--border-color)",
            borderRadius: 3,
            bgcolor: "var(--bg-primary)",
            display: "flex",
            flexDirection: "column",
            gap: 1.5,
            boxShadow: "0 4px 12px rgba(0,0,0,0.03)"
          }}
        >
          {/* Trust gauge */}
          {trustScore !== null && (
            <Box sx={{ maxWidth: 260 }}>
              <SellerTrustGauge score={trustScore} />
            </Box>
          )}

          {/* Sentiment + counts */}
          <Box sx={{ display: "flex", gap: 1.5, flexWrap: "wrap", alignItems: "flex-end" }}>
            {sentiment !== null && (
              <Box>
                <Typography sx={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.5 }}>
                  Sentiment
                </Typography>
                <Box 
                  sx={{ 
                    px: 1, py: 0.25, borderRadius: '6px', 
                    bgcolor: sentiment >= 0.7 ? '#ecfdf5' : sentiment >= 0.4 ? '#fffbeb' : '#fef2f2',
                    border: `1px solid ${sentiment >= 0.7 ? '#10b981' : sentiment >= 0.4 ? '#f59e0b' : '#ef4444'}40`
                  }}
                >
                  <Typography sx={{ fontSize: 13, fontWeight: 800, color: sentiment >= 0.7 ? '#059669' : sentiment >= 0.4 ? '#d97706' : '#dc2626' }}>
                    {Math.round(sentiment * 100)}%
                  </Typography>
                </Box>
              </Box>
            )}
            
            <Box>
              <Typography sx={{ fontSize: 10, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.5 }}>
                Distribuzione
              </Typography>
              <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap" }}>
                {[
                  { label: positive.length, desc: 'pos', color: "#10b981", bg: '#ecfdf5' },
                  { label: neutral,         desc: 'neu', color: "#f59e0b", bg: '#fffbeb' },
                  { label: negative.length, desc: 'neg', color: "#ef4444", bg: '#fef2f2' }
                ].map((item) => (
                  <Box
                    key={item.desc}
                    sx={{
                      px: 0.75,
                      py: 0.25,
                      borderRadius: "6px",
                      bgcolor: "var(--bg-secondary)",
                      border: `1px solid var(--border-color)`,
                      display: "flex",
                      alignItems: "center",
                      gap: 0.5
                    }}
                  >
                    <Typography sx={{ fontSize: 12, fontWeight: 800, color: item.color }}>{item.label}</Typography>
                    <Typography sx={{ fontSize: 10, fontWeight: 600, color: "var(--text-secondary)", textTransform: "uppercase" }}>{item.desc}</Typography>
                  </Box>
                ))}
              </Box>
            </Box>

            {flaggedFeedbacks.length > 0 && (
              <Box sx={{ width: '100%', display: 'flex', alignItems: 'flex-start', gap: 1, mt: 0.5, p: 1.25, bgcolor: 'var(--bg-secondary)', border: '1px solid var(--border-color)', borderRadius: '8px', borderLeft: '3px solid #dc2626' }}>
                <WarningAmberIcon sx={{ fontSize: 16, color: '#dc2626', mt: "2px" }} />
                <Typography sx={{ fontSize: 11, fontWeight: 500, color: 'var(--text-primary)', lineHeight: 1.4 }}>
                  L'AI ha rilevato <strong>{flaggedFeedbacks.length}</strong> recensioni potenzialmente fuorvianti, con testo in disaccordo col rating.
                </Typography>
              </Box>
            )}
          </Box>

          {/* Feedback lists */}
          <Box sx={{ borderTop: "1px solid var(--border-color)", pt: 1.5 }}>
            {positive.length > 0 && (
              <Box sx={{ mb: 1.5 }}>
                <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.75 }}>
                  Top positivi
                </Typography>
                <SellerFeedbackList feedbacks={positive.slice(0, 3)} initialLimit={3} title="" />
              </Box>
            )}
            {negative.length > 0 && (
              <Box sx={{ mb: 1.5 }}>
                <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.75 }}>
                  Da attenzionare
                </Typography>
                <SellerFeedbackList feedbacks={negative.slice(0, 3)} initialLimit={3} title="" />
              </Box>
            )}
            <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.75 }}>
              Tutti i feedback
            </Typography>
            <SellerFeedbackList feedbacks={feedbacks} title="" />
          </Box>
        </Box>
      </Collapse>
    </Box>
  )
}
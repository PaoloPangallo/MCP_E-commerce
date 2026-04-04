import { useEffect, useMemo, useState } from "react"
import { Box, Button, CircularProgress, Collapse, Typography, Avatar, Divider, Chip, Paper } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import StorefrontIcon from '@mui/icons-material/Storefront';
import EventIcon from '@mui/icons-material/Event';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import StarsIcon from '@mui/icons-material/Stars';
import VerifiedUserIcon from '@mui/icons-material/VerifiedUser';
import InfoOutlinedIcon from '@mui/icons-material/InfoOutlined';
import WarningAmberIcon from '@mui/icons-material/WarningAmber';

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
  // Enriched
  registration_date?: string
  location?: string
  feedback_score?: number
  store_name?: string
  logo_url?: string
  store_description?: string
  ai_summary?: string
}

interface Props {
  seller?: string
  sellerName?: string
  trustScore?: number
  sentimentScore?: number
  count?: number
  feedbacks?: Feedback[]
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

export default function SellerSummaryCard(props: Props) {
  const { seller: sellerProp, sellerName, trustScore: trustProp, sentimentScore: sentimentProp, feedbacks: feedbacksProp } = props
  const seller = sellerProp || sellerName

  const [open, setOpen]               = useState(!!feedbacksProp?.length)
  const [loading, setLoading]         = useState(false)
  const [error, setError]             = useState<string | null>(null)
  
  // Data state
  const [feedbacks, setFeedbacks]     = useState<Feedback[]>(feedbacksProp || [])
  const [trustScore, setTrustScore]   = useState<number | null>(trustProp ?? null)
  const [sentiment, setSentiment]     = useState<number | null>(sentimentProp ?? null)
  
  // Metadata state
  const [regDate, setRegDate]         = useState<string | null>(null)
  const [location, setLocation]       = useState<string | null>(null)
  const [fbScore, setFbScore]         = useState<number | null>(null)
  const [storeName, setStoreName]     = useState<string | null>(null)
  const [logoUrl, setLogoUrl]         = useState<string | null>(null)
  const [aiSummary, setAiSummary]     = useState<string | null>(null)

  useEffect(() => {
    if (feedbacksProp) {
        setFeedbacks(feedbacksProp)
        setOpen(feedbacksProp.length > 0)
    }
    if (trustProp !== undefined) setTrustScore(trustProp)
    if (sentimentProp !== undefined) setSentiment(sentimentProp)
  }, [feedbacksProp, trustProp, sentimentProp])

  useEffect(() => {
    if (!feedbacksProp) {
        setOpen(false)
        setFeedbacks([])
        setTrustScore(trustProp??null)
        setSentiment(sentimentProp??null)
        setError(null)
        // Reset metadata
        setRegDate(null); setLocation(null); setFbScore(null);
        setStoreName(null); setLogoUrl(null); setAiSummary(null);
    }
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
      
      setRegDate(data.registration_date ?? null)
      setLocation(data.location ?? null)
      setFbScore(data.feedback_score ?? null)
      setStoreName(data.store_name ?? null)
      setLogoUrl(data.logo_url ?? null)
      setAiSummary(data.ai_summary ?? null)

      setOpen(true)
    } catch (err) {
      setError("Errore nel caricamento dell'analisi venditore")
    } finally {
      setLoading(false)
    }
  }

  const formattedDate = useMemo(() => {
    if (!regDate) return null
    try {
      const d = new Date(regDate)
      return d.toLocaleDateString('it-IT', { year: 'numeric', month: 'long' })
    } catch { return regDate }
  }, [regDate])

  const trustColor = (score: number) => {
    if (score >= 0.90) return { main: '#10b981', bg: '#ecfdf5', text: '#065f46', label: 'Ecellente' }
    if (score >= 0.70) return { main: '#f59e0b', bg: '#fffbeb', text: '#92400e', label: 'Discreto' }
    return { main: '#ef4444', bg: '#fef2f2', text: '#991b1b', label: 'Rischioso' }
  }

  return (
    <Box>
      <Button
        size="small"
        variant="outlined"
        onClick={handleToggle}
        disabled={!seller || loading}
        startIcon={loading ? <CircularProgress size={12} sx={{ color: "var(--brand-primary)" }} /> : <VerifiedUserIcon sx={{ color: "var(--brand-primary)" }} />}
        endIcon={!loading ? (
          <KeyboardArrowDownIcon
            sx={{
              fontSize: 16,
              transform: open ? "rotate(180deg)" : "none",
              transition: "transform 0.2s"
            }}
          />
        ) : null}
        sx={{
          textTransform: "none",
          fontSize: 13,
          fontWeight: 700,
          color: "var(--text-primary)",
          borderColor: "var(--border-color)",
          borderRadius: "12px",
          px: 2,
          py: 0.75,
          "&:hover": { borderColor: "var(--brand-primary)", bgcolor: "var(--bg-secondary)" }
        }}
      >
        {loading ? "Analisi venditore in corso..." : open ? "Chiudi analisi venditore" : seller ? `Verifica affidabilità di ${seller}` : "Analisi venditore"}
      </Button>

      {error && (
        <Typography sx={{ mt: 1, fontSize: 13, fontWeight: 500, color: "var(--danger)" }}>{error}</Typography>
      )}

      <Collapse in={open} timeout={400}>
        <Paper
          elevation={0}
          sx={{
            mt: 2,
            border: "1px solid var(--border-color)",
            borderRadius: "16px",
            overflow: "hidden",
            bgcolor: "var(--bg-primary)",
            boxShadow: "0 10px 30px -10px rgba(0,0,0,0.05)"
          }}
        >
          {/* Top Hero Section */}
          <Box sx={{ 
            p: 3, 
            background: "linear-gradient(145deg, var(--bg-secondary) 0%, var(--bg-primary) 100%)",
            display: "flex", 
            alignItems: "center", 
            gap: 3,
            position: "relative"
          }}>
            <Avatar 
                src={logoUrl || ''} 
                sx={{ 
                    width: 72, 
                    height: 72, 
                    bgcolor: 'var(--brand-soft)', 
                    color: 'var(--brand-primary)',
                    boxShadow: '0 8px 16px rgba(0,0,0,0.06)',
                    border: '3px solid var(--bg-primary)'
                }}
            >
                {seller ? seller.charAt(0).toUpperCase() : <StorefrontIcon sx={{ fontSize: 32 }} />}
            </Avatar>
            <Box sx={{ flex: 1, zIndex: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5, mb: 1 }}>
                    <Typography sx={{ fontSize: 22, fontWeight: 900, color: "var(--text-primary)", letterSpacing: "-0.01em" }}>
                        {storeName || seller}
                    </Typography>
                    {fbScore !== null && (
                        <Chip 
                            icon={<StarsIcon sx={{ fontSize: '14px !important' }} />}
                            label={fbScore.toLocaleString()}
                            size="small"
                            sx={{ 
                                height: 22, 
                                fontSize: 12, 
                                fontWeight: 800, 
                                bgcolor: '#fef9c3', 
                                color: '#ca8a04',
                                '& .MuiChip-label': { px: 1 },
                                boxShadow: "0 2px 4px rgba(202, 138, 4, 0.1)"
                            }}
                        />
                    )}
                </Box>
                <Box sx={{ display: 'flex', gap: 2.5, flexWrap: 'wrap' }}>
                    {formattedDate && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'var(--text-secondary)' }}>
                            <EventIcon sx={{ fontSize: 15 }} />
                            <Typography sx={{ fontSize: 13, fontWeight: 500 }}>Iscritto dal {formattedDate}</Typography>
                        </Box>
                    )}
                    {location && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: 'var(--text-secondary)' }}>
                            <LocationOnIcon sx={{ fontSize: 15 }} />
                            <Typography sx={{ fontSize: 13, fontWeight: 500 }}>{location}</Typography>
                        </Box>
                    )}
                </Box>
            </Box>
          </Box>

          <Divider sx={{ borderColor: 'var(--border-color)' }} />

          {/* AI Explanatory Summary */}
          {aiSummary && (
            <Box sx={{ p: 3, pt: 3, pb: 2 }}>
              <Box sx={{ p: 2.5, bgcolor: "#f8fafc", borderRadius: "12px", border: "1px solid #e2e8f0", borderLeft: "4px solid var(--brand-primary)" }}>
                <Typography sx={{ fontSize: 13, fontWeight: 700, color: "var(--brand-primary)", textTransform: "uppercase", letterSpacing: "0.05em", mb: 0.5, display: 'flex', alignItems: 'center', gap: 0.75 }}>
                  <StarsIcon sx={{ fontSize: 16 }} />
                  Insight Recensioni
                </Typography>
                <Typography sx={{ fontSize: 14, color: "#334155", lineHeight: 1.6, fontWeight: 500 }}>
                  {aiSummary}
                </Typography>
              </Box>
            </Box>
          )}

          {/* Key Metrics Dashboard */}
          <Box sx={{ p: 3, pt: aiSummary ? 0 : 3, display: 'grid', gridTemplateColumns: { xs: '1fr', md: '1fr 1.5fr' }, gap: 3 }}>
            
            {/* Trust Score Spotlight */}
            {trustScore !== null && (
                <Box 
                  sx={{ 
                    bgcolor: trustColor(trustScore).bg, 
                    p: 3, 
                    borderRadius: '20px', 
                    display: 'flex', 
                    flexDirection: 'column', 
                    alignItems: 'center',
                    justifyContent: 'center',
                    textAlign: 'center',
                    border: `1px solid ${trustColor(trustScore).main}33`
                  }}
                >
                    <Typography sx={{ fontSize: 12, fontWeight: 800, color: trustColor(trustScore).main, textTransform: "uppercase", letterSpacing: "0.1em", mb: 1 }}>
                        Affidabilità AI
                    </Typography>
                    <Typography sx={{ fontSize: 48, fontWeight: 900, color: trustColor(trustScore).text, lineHeight: 1, mb: 1, letterSpacing: "-0.02em" }}>
                        {Math.round(trustScore * 100)}%
                    </Typography>
                    <Chip 
                      label={trustColor(trustScore).label} 
                      sx={{ 
                        bgcolor: trustColor(trustScore).main, 
                        color: "#fff", 
                        fontWeight: 800, 
                        fontSize: 12,
                        height: 24
                      }} 
                    />
                </Box>
            )}

            {/* AI Sentiment & Distribution Grid */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2.5 }}>
                {sentiment !== null && (
                    <Box sx={{ display: 'flex', alignItems: 'center', justifyContent: 'space-between', p: 2, bgcolor: "var(--bg-secondary)", borderRadius: "16px", border: "1px solid var(--border-color)" }}>
                        <Box display="flex" alignItems="center" gap={1}>
                          <InfoOutlinedIcon sx={{ color: "var(--text-secondary)", fontSize: 18 }} />
                          <Typography sx={{ fontSize: 13, fontWeight: 700, color: "var(--text-secondary)" }}>
                              Sentiment Analisi Feedback
                          </Typography>
                        </Box>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                            <Typography sx={{ fontSize: 20, fontWeight: 900, color: "var(--text-primary)", lineHeight: 1 }}>
                                {Math.round(sentiment * 100)}%
                            </Typography>
                            <Box 
                                sx={{ 
                                    px: 1.5, 
                                    py: 0.5, 
                                    borderRadius: '8px', 
                                    bgcolor: sentiment >= 0.7 ? '#d1fae5' : sentiment >= 0.4 ? '#fef3c7' : '#fee2e2',
                                    color: sentiment >= 0.7 ? '#065f46' : sentiment >= 0.4 ? '#92400e' : '#991b1b',
                                    fontSize: 11,
                                    fontWeight: 800,
                                }}
                            >
                                {sentiment >= 0.7 ? 'POSITIVO' : sentiment >= 0.4 ? 'NEUTRO' : 'NEGATIVO'}
                            </Box>
                        </Box>
                    </Box>
                )}

                <Box sx={{ p: 2, bgcolor: "var(--bg-secondary)", borderRadius: "16px", border: "1px solid var(--border-color)" }}>
                    <Typography sx={{ fontSize: 11, fontWeight: 800, color: "var(--text-primary)", textTransform: "uppercase", letterSpacing: "0.08em", mb: 1.5 }}>
                        Tono Recensioni ({feedbacks.length})
                    </Typography>
                    <Box sx={{ display: "flex", gap: 1.5, mb: flaggedFeedbacks.length > 0 ? 1.5 : 0 }}>
                        {[
                            { label: positive.length, desc: 'Positivi', color: "#10b981", bg: '#ecfdf5' },
                            { label: neutral,         desc: 'Neutri',   color: "#f59e0b", bg: '#fffbeb' },
                            { label: negative.length, desc: 'Negativi', color: "#ef4444", bg: '#fef2f2' }
                        ].map((item) => (
                            <Box
                                key={item.desc}
                                sx={{
                                    flex: 1,
                                    px: 1,
                                    py: 1.5,
                                    borderRadius: "12px",
                                    bgcolor: "var(--bg-primary)",
                                    textAlign: 'center',
                                    border: `1px solid ${item.bg}`,
                                    boxShadow: `0 2px 4px ${item.bg}`
                                }}
                            >
                                <Typography sx={{ fontSize: 16, fontWeight: 900, color: item.color, lineHeight: 1, mb: 0.5 }}>{item.label}</Typography>
                                <Typography sx={{ fontSize: 10, fontWeight: 700, color: "var(--text-secondary)", textTransform: 'uppercase' }}>{item.desc}</Typography>
                            </Box>
                        ))}
                    </Box>
                    {flaggedFeedbacks.length > 0 && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, p: 1.5, bgcolor: '#fef2f2', border: '1px solid #fca5a5', borderRadius: '10px' }}>
                            <WarningAmberIcon sx={{ fontSize: 18, color: '#dc2626' }} />
                            <Typography sx={{ fontSize: 11, fontWeight: 600, color: '#991b1b', lineHeight: 1.4 }}>
                                L'Intelligenza Artificiale ha rilevato <strong>{flaggedFeedbacks.length}</strong> recensioni potenzialmente fuorvianti, il cui testo analizzato è in forte disaccordo con il rating a stelle rilasciato.
                            </Typography>
                        </Box>
                    )}
                </Box>
            </Box>
          </Box>

          <Divider sx={{ borderColor: 'var(--border-color)' }} />

          {/* Feedback details */}
          <Box sx={{ p: 3, pt: 2, bgcolor: "var(--bg-secondary)" }}>
            <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1fr 1fr" }, gap: 3, mb: 3 }}>
              {positive.length > 0 && (
                <Box>
                  <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#10b981", textTransform: "uppercase", letterSpacing: "0.05em", mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#10b981', boxShadow: "0 0 8px #10b981" }} />
                      Cosa apprezzano
                  </Typography>
                  <SellerFeedbackList feedbacks={positive.slice(0, 3)} initialLimit={3} title="" />
                </Box>
              )}
              {negative.length > 0 && (
                <Box>
                  <Typography sx={{ fontSize: 12, fontWeight: 800, color: "#ef4444", textTransform: "uppercase", letterSpacing: "0.05em", mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
                      <Box sx={{ width: 8, height: 8, borderRadius: '50%', bgcolor: '#ef4444', boxShadow: "0 0 8px #ef4444" }} />
                      Possibili problemi
                  </Typography>
                  <SellerFeedbackList feedbacks={negative.slice(0, 3)} initialLimit={3} title="" />
                </Box>
              )}
            </Box>
            
            <Box sx={{ mt: 3, p: 2, bgcolor: "var(--bg-primary)", borderRadius: "12px", border: "1px solid var(--border-color)" }}>
                <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
                    <Typography sx={{ fontSize: 12, fontWeight: 800, color: "var(--text-primary)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                        Cronologia Completa
                    </Typography>
                    <Typography sx={{ fontSize: 12, fontWeight: 600, color: 'var(--text-secondary)' }}>
                      Analisi automatica su {feedbacks.length} recensioni
                    </Typography>
                </Box>
                <SellerFeedbackList feedbacks={feedbacks} title="" />
            </Box>
          </Box>
        </Paper>
      </Collapse>
    </Box>
  )
}
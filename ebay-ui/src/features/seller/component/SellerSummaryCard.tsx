import { useEffect, useMemo, useState } from "react"
import {Box, Button, CircularProgress, Collapse, Typography, Avatar, Divider, Chip} from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import StorefrontIcon from '@mui/icons-material/Storefront';
import EventIcon from '@mui/icons-material/Event';
import LocationOnIcon from '@mui/icons-material/LocationOn';
import StarsIcon from '@mui/icons-material/Stars';

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
  // Enriched
  registration_date?: string
  location?: string
  feedback_score?: number
  store_name?: string
  logo_url?: string
  store_description?: string
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
  const [, setStoreDesc]     = useState<string | null>(null)

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
        setStoreName(null); setLogoUrl(null); setStoreDesc(null);
    }
  }, [seller])

  const positive = useMemo(() => feedbacks.filter((f) => (f.rating ?? 0) >= 4), [feedbacks])
  const negative = useMemo(() => feedbacks.filter((f) => (f.rating ?? 0) <= 2), [feedbacks])
  const neutral  = Math.max(feedbacks.length - positive.length - negative.length, 0)

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
      
      // Update metadata
      setRegDate(data.registration_date ?? null)
      setLocation(data.location ?? null)
      setFbScore(data.feedback_score ?? null)
      setStoreName(data.store_name ?? null)
      setLogoUrl(data.logo_url ?? null)
      setStoreDesc(data.store_description ?? null)

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

  return (
    <Box>
      <Button
        size="small"
        variant="text"
        onClick={handleToggle}
        disabled={!seller || loading}
        startIcon={loading ? <CircularProgress size={12} sx={{ color: "#9ca3af" }} /> : null}
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
          fontWeight: 600,
          color: "#6b7280",
          p: 0,
          minWidth: 0,
          "&:hover": { color: "#111827", bgcolor: "transparent" }
        }}
      >
        {loading ? "Analisi in corso..." : open ? "Nascondi analisi venditore" : "Vedi analisi venditore"}
      </Button>

      {/* Error */}
      {error && (
        <Typography sx={{ mt: 0.75, fontSize: 12, color: "#dc2626" }}>{error}</Typography>
      )}

      {/* Panel */}
      <Collapse in={open} timeout={300}>
        <Box
          sx={{
            mt: 1.5,
            p: 3,
            border: "1px solid #f0f0f0",
            borderRadius: "16px",
            bgcolor: "#fff",
            display: "flex",
            flexDirection: "column",
            gap: 2.5,
            boxShadow: "0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.05)"
          }}
        >
          {/* Enhanced Header */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2.5 }}>
            <Avatar 
                src={logoUrl || ''} 
                sx={{ 
                    width: 64, 
                    height: 64, 
                    bgcolor: '#f3f4f6', 
                    color: '#9ca3af',
                    boxShadow: '0 4px 10px rgba(0,0,0,0.04)',
                    border: '2px solid #fff'
                }}
            >
                {seller ? seller.charAt(0).toUpperCase() : <StorefrontIcon />}
            </Avatar>
            <Box sx={{ flex: 1 }}>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 0.5 }}>
                    <Typography sx={{ fontSize: 18, fontWeight: 800, color: "#111827" }}>
                        {storeName || seller}
                    </Typography>
                    {fbScore !== null && (
                        <Chip 
                            icon={<StarsIcon sx={{ fontSize: '14px !important' }} />}
                            label={fbScore.toLocaleString()}
                            size="small"
                            sx={{ 
                                height: 20, 
                                fontSize: 11, 
                                fontWeight: 700, 
                                bgcolor: '#fef9c3', 
                                color: '#854d0e',
                                '& .MuiChip-label': { px: 1 }
                            }}
                        />
                    )}
                </Box>
                <Box sx={{ display: 'flex', gap: 2, flexWrap: 'wrap' }}>
                    {formattedDate && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: '#6b7280' }}>
                            <EventIcon sx={{ fontSize: 14 }} />
                            <Typography sx={{ fontSize: 12 }}>Iscritto da {formattedDate}</Typography>
                        </Box>
                    )}
                    {location && (
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, color: '#6b7280' }}>
                            <LocationOnIcon sx={{ fontSize: 14 }} />
                            <Typography sx={{ fontSize: 12 }}>{location}</Typography>
                        </Box>
                    )}
                </Box>
            </Box>
          </Box>

          <Divider sx={{ borderColor: '#f3f4f6' }} />

          {/* Core Metrics Grid */}
          <Box sx={{ display: 'grid', gridTemplateColumns: { xs: '1fr', sm: '1fr 1fr' }, gap: 3 }}>
            {/* Left Col: Trust Gauge */}
            {trustScore !== null && (
                <Box sx={{ bgcolor: '#f9fafb', p: 2, borderRadius: '12px' }}>
                    <Typography sx={{ fontSize: 10, fontWeight: 800, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.08em", mb: 2 }}>
                        Affidabilità Complessiva
                    </Typography>
                    <SellerTrustGauge score={trustScore} />
                </Box>
            )}

            {/* Right Col: AI Sentiment & Distribution */}
            <Box sx={{ display: 'flex', flexDirection: 'column', gap: 3 }}>
                {sentiment !== null && (
                    <Box>
                        <Typography sx={{ fontSize: 10, fontWeight: 800, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.08em", mb: 1.5 }}>
                            Analisi Sentiment AI
                        </Typography>
                        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
                            <Typography sx={{ fontSize: 28, fontWeight: 900, color: "#111827", lineHeight: 1 }}>
                                {Math.round(sentiment * 100)}%
                            </Typography>
                            <Box 
                                sx={{ 
                                    px: 1.5, 
                                    py: 0.5, 
                                    borderRadius: '20px', 
                                    bgcolor: sentiment >= 0.7 ? '#d1fae5' : sentiment >= 0.4 ? '#fef3c7' : '#fee2e2',
                                    color: sentiment >= 0.7 ? '#065f46' : sentiment >= 0.4 ? '#92400e' : '#991b1b',
                                    fontSize: 10,
                                    fontWeight: 800,
                                    letterSpacing: '0.02em'
                                }}
                            >
                                {sentiment >= 0.7 ? 'POSITIVO' : sentiment >= 0.4 ? 'NEUTRO' : 'NEGATIVO'}
                            </Box>
                        </Box>
                    </Box>
                )}

                <Box>
                    <Typography sx={{ fontSize: 10, fontWeight: 800, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.08em", mb: 1.5 }}>
                        Distribuzione Feedback
                    </Typography>
                    <Box sx={{ display: "flex", gap: 1 }}>
                        {[
                            { label: positive.length, desc: 'Positivi', color: "#059669", bg: '#ecfdf5' },
                            { label: neutral,         desc: 'Neutri',   color: "#d97706", bg: '#fffbeb' },
                            { label: negative.length, desc: 'Negativi', color: "#dc2626", bg: '#fef2f2' }
                        ].map((item) => (
                            <Box
                                key={item.desc}
                                sx={{
                                    flex: 1,
                                    px: 1,
                                    py: 1,
                                    borderRadius: "8px",
                                    bgcolor: item.bg,
                                    textAlign: 'center'
                                }}
                            >
                                <Typography sx={{ fontSize: 14, fontWeight: 800, color: item.color }}>{item.label}</Typography>
                                <Typography sx={{ fontSize: 9, fontWeight: 700, color: item.color, opacity: 0.7, textTransform: 'uppercase' }}>{item.desc}</Typography>
                            </Box>
                        ))}
                    </Box>
                </Box>
            </Box>
          </Box>

          {/* Feedback details */}
          <Box sx={{ borderTop: "1px solid #f3f4f6", pt: 2.5 }}>
            {positive.length > 0 && (
              <Box sx={{ mb: 2.5 }}>
                <Typography sx={{ fontSize: 11, fontWeight: 800, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em", mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#10b981' }} />
                    Highlights Positivi
                </Typography>
                <SellerFeedbackList feedbacks={positive.slice(0, 3)} initialLimit={3} title="" />
              </Box>
            )}
            {negative.length > 0 && (
              <Box sx={{ mb: 2.5 }}>
                <Typography sx={{ fontSize: 11, fontWeight: 800, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em", mb: 1.5, display: 'flex', alignItems: 'center', gap: 1 }}>
                    <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: '#ef4444' }} />
                    Segnalazioni Critiche
                </Typography>
                <SellerFeedbackList feedbacks={negative.slice(0, 3)} initialLimit={3} title="" />
              </Box>
            )}
            <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center', mb: 1.5 }}>
                <Typography sx={{ fontSize: 11, fontWeight: 800, color: "#6b7280", textTransform: "uppercase", letterSpacing: "0.08em" }}>
                    Tutti i Feedback
                </Typography>
                <Typography sx={{ fontSize: 11, color: '#9ca3af' }}>Ultimi {feedbacks.length} feedback analizzati</Typography>
            </Box>
            <SellerFeedbackList feedbacks={feedbacks} title="" />
          </Box>
        </Box>
      </Collapse>
    </Box>
  )
}
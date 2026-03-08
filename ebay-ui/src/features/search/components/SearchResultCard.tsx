import { useMemo, useState } from "react"
import { Box, Button, Chip, Collapse, Link, Paper, Typography } from "@mui/material"

import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ExpandLessIcon from "@mui/icons-material/ExpandLess"

import SellerTrustGauge from "../../../component/SellerTrustGauge"
import SellerFeedbackPanel from "../../../component/SellerFeedbackPanel"
import ExplanationChips from "./ExplanationChips"
import type { SearchItem } from "../types"
import SellerInfo from "../../../component/SellerInfo.tsx";

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") return "Prezzo non disponibile"
  return `${price} ${currency ?? ""}`.trim()
}

function getFallbackLabel(title?: string) {
  if (!title?.trim()) return "No image"
  const trimmed = title.trim()
  return trimmed.length > 18 ? `${trimmed.slice(0, 18)}…` : trimmed
}

export default function SearchResultCard({ item }: { item: SearchItem }) {
  const [imageError, setImageError] = useState(false)
  const [showSellerPanel, setShowSellerPanel] = useState(false)

  const trustPercent = typeof item.trust_score === "number" ? Math.round(item.trust_score * 100) : null
  const rankingPercent = typeof item.ranking_score === "number" ? Math.round(item.ranking_score * 100) : null

  const imageProps = useMemo(() => {
    if (!item.url) return {}
    return { href: item.url, target: "_blank", rel: "noreferrer" }
  }, [item.url])

  const ragFeedbackPreview = Array.isArray(item.rag_feedback)
    ? item.rag_feedback.map((fb) => fb?.comment || "").filter(Boolean).slice(0, 2)
    : []

  return (
    <Paper elevation={0} sx={{ p: { xs: 2, md: 2.5 }, borderRadius: 4, border: "1px solid #e5e7eb", backgroundColor: "#ffffff", transition: "border-color 0.18s ease, box-shadow 0.18s ease", "&:hover": { borderColor: "#d5dae1", boxShadow: "0 10px 26px rgba(15, 23, 42, 0.05)" } }}>
      <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", sm: "132px 1fr" }, gap: 2.25 }}>
        <Box component={item.url ? "a" : "div"} aria-label={item.title ? `Apri ${item.title}` : "Apri prodotto"} {...imageProps} sx={{ width: "100%", maxWidth: { xs: "100%", sm: 132 }, aspectRatio: "1 / 1", borderRadius: 3, overflow: "hidden", backgroundColor: "#f8fafc", border: "1px solid #e5e7eb", display: "flex", alignItems: "center", justifyContent: "center", textDecoration: "none" }}>
          {!imageError && item.image_url ? (
            <Box component="img" src={item.image_url} alt={item.title || "Prodotto eBay"} loading="lazy" onError={() => setImageError(true)} sx={{ width: "100%", height: "100%", objectFit: "cover", display: "block" }} />
          ) : (
            <Typography sx={{ px: 1.25, textAlign: "center", fontSize: 12.5, color: "#6b7280", lineHeight: 1.5 }}>{getFallbackLabel(item.title)}</Typography>
          )}
        </Box>

        <Box minWidth={0}>
          <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 2, flexWrap: "wrap" }}>
            <Box minWidth={0} flex={1}>
              {item.url ? (
                <Link href={item.url} target="_blank" rel="noreferrer" underline="none" color="inherit" sx={{ display: "inline-flex", alignItems: "center", gap: 0.75 }}>
                  <Typography sx={{ fontWeight: 700, fontSize: 17, color: "#111827", lineHeight: 1.45 }}>{item.title || "Titolo non disponibile"}</Typography>
                  <OpenInNewIcon sx={{ fontSize: 15, color: "#6b7280" }} />
                </Link>
              ) : (
                <Typography sx={{ fontWeight: 700, fontSize: 17, color: "#111827", lineHeight: 1.45 }}>{item.title || "Titolo non disponibile"}</Typography>
              )}

              <Typography sx={{ fontSize: 27, fontWeight: 700, color: "#111827", mt: 1 }}>{formatPrice(item.price, item.currency)}</Typography>
              {item.condition ? <Typography sx={{ fontSize: 13, color: "#6b7280", mt: 0.5 }}>Condizione: {item.condition}</Typography> : null}
            </Box>

            <Box display="flex" gap={1} flexWrap="wrap" justifyContent="flex-end">
              {trustPercent !== null ? <Chip icon={<VerifiedUserIcon sx={{ fontSize: 16 }} />} label={`Trust ${trustPercent}%`} size="small" sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb", color: "#374151" }} /> : null}
              {rankingPercent !== null ? <Chip icon={<AutoAwesomeIcon sx={{ fontSize: 16 }} />} label={`AI match ${rankingPercent}%`} size="small" sx={{ bgcolor: "#f5f7ff", border: "1px solid #dbe4ff", color: "#334155" }} /> : null}
              {item._already_in_db ? <Chip label="già in DB" size="small" sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb", color: "#6b7280" }} /> : null}
            </Box>
          </Box>

          <Box mt={1.6}><SellerInfo seller_name={item.seller_name} seller_rating={item.seller_rating} /></Box>
          {typeof item.trust_score === "number" ? <Box mt={1.35}><SellerTrustGauge score={item.trust_score} /></Box> : null}
          {item.explanations?.length ? <Box mt={1.5}><ExplanationChips explanations={item.explanations} /></Box> : null}

          {ragFeedbackPreview.length > 0 ? (
            <Box sx={{ mt: 1.75, border: "1px solid #eef1f4", bgcolor: "#fafbfc", borderRadius: 3, px: 1.5, py: 1.3 }}>
              <Typography sx={{ fontSize: 12.5, fontWeight: 700, color: "#111827", mb: 0.85 }}>Segnali retrieval sul venditore</Typography>
              {ragFeedbackPreview.map((text, index) => <Typography key={`${item.ebay_id}-rag-${index}`} sx={{ fontSize: 13, color: "#4b5563", lineHeight: 1.65, mb: index === ragFeedbackPreview.length - 1 ? 0 : 0.7 }}>“{text}”</Typography>)}
            </Box>
          ) : null}

          {item.seller_name ? (
            <Box mt={1.75}>
              <Button variant="text" onClick={() => setShowSellerPanel((prev) => !prev)} endIcon={showSellerPanel ? <ExpandLessIcon /> : <ExpandMoreIcon />} sx={{ px: 0, textTransform: "none", fontWeight: 600, color: "#111827", "&:hover": { bgcolor: "transparent" } }}>
                {showSellerPanel ? "Nascondi seller deep dive" : "Mostra seller deep dive"}
              </Button>
              <Collapse in={showSellerPanel} timeout="auto" unmountOnExit>
                <Box mt={1}><SellerFeedbackPanel seller={item.seller_name} /></Box>
              </Collapse>
            </Box>
          ) : null}
        </Box>
      </Box>
    </Paper>
  )
}

import { useState } from "react"
import { Box, Collapse, Link, Typography } from "@mui/material"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"

import SellerTrustGauge from "../../seller/component/SellerTrustGauge.tsx"
import SellerFeedbackPanel from "../../seller/component/SellerFeedbackPanel.tsx"
import SellerInfo from "../../seller/SellerInfo.tsx"
import ExplanationChips from "./ExplanationChips.tsx"
import type { SearchItem } from "../types"

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") return "—"
  return `${price} ${currency ?? ""}`.trim()
}

function TrustBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const good = pct >= 90
  return (
    <Box
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.4,
        px: 0.875,
        py: 0.25,
        borderRadius: "6px",
        bgcolor: good ? "#f0fdf4" : "#fafafa",
        border: "1px solid",
        borderColor: good ? "#bbf7d0" : "#e5e7eb"
      }}
    >
      <VerifiedUserIcon sx={{ fontSize: 11, color: good ? "#16a34a" : "#9ca3af" }} />
      <Typography sx={{ fontSize: 11, fontWeight: 500, color: good ? "#15803d" : "#6b7280" }}>
        {pct}%
      </Typography>
    </Box>
  )
}

function AiMatchBadge({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  return (
    <Box
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: 0.4,
        px: 0.875,
        py: 0.25,
        borderRadius: "6px",
        bgcolor: "#f5f3ff",
        border: "1px solid #e0d9ff"
      }}
    >
      <Typography sx={{ fontSize: 11, fontWeight: 500, color: "#6d28d9" }}>
        AI {pct}%
      </Typography>
    </Box>
  )
}

export default function SearchResultCard({ item }: { item: SearchItem }) {
  const [imageError, setImageError] = useState(false)
  const [sellerOpen, setSellerOpen] = useState(false)

  const trustPercent = typeof item.trust_score === "number" ? item.trust_score : null
  const rankingPercent = typeof item.ranking_score === "number" ? item.ranking_score : null

  const ragFeedbackPreview = Array.isArray(item.rag_feedback)
    ? item.rag_feedback.map((fb) => fb?.comment || "").filter(Boolean).slice(0, 2)
    : []

  return (
    <Box
      sx={{
        display: "grid",
        gridTemplateColumns: "80px 1fr",
        gap: 1.75,
        py: 2,
        borderBottom: "1px solid #f5f5f5",
        "&:last-child": { borderBottom: "none" },
        "&:first-of-type": { pt: 0 }
      }}
    >
      {/* Thumbnail */}
      <Box
        component={item.url ? "a" : "div"}
        href={item.url}
        target="_blank"
        rel="noreferrer"
        sx={{
          width: 80,
          height: 80,
          borderRadius: 2,
          overflow: "hidden",
          bgcolor: "#f9fafb",
          border: "1px solid #f0f0f0",
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          flexShrink: 0,
          textDecoration: "none"
        }}
      >
        {!imageError && item.image_url ? (
          <Box
            component="img"
            src={item.image_url}
            alt={item.title || ""}
            loading="lazy"
            onError={() => setImageError(true)}
            sx={{ width: "100%", height: "100%", objectFit: "cover" }}
          />
        ) : (
          <Typography sx={{ fontSize: 10, color: "#d1d5db", textAlign: "center", px: 0.5 }}>
            no img
          </Typography>
        )}
      </Box>

      {/* Content */}
      <Box sx={{ minWidth: 0 }}>
        {/* Title row */}
        <Box sx={{ display: "flex", alignItems: "flex-start", gap: 0.5, mb: 0.25 }}>
          {item.url ? (
            <Link
              href={item.url}
              target="_blank"
              rel="noreferrer"
              underline="none"
              sx={{
                fontSize: 13,
                fontWeight: 500,
                color: "#111827",
                lineHeight: 1.4,
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                "&:hover": { color: "#374151" }
              }}
            >
              {item.title || "Titolo non disponibile"}
            </Link>
          ) : (
            <Typography
              sx={{
                fontSize: 13,
                fontWeight: 500,
                color: "#111827",
                lineHeight: 1.4,
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden"
              }}
            >
              {item.title || "Titolo non disponibile"}
            </Typography>
          )}
          {item.url && (
            <OpenInNewIcon sx={{ fontSize: 12, color: "#d1d5db", flexShrink: 0, mt: 0.2 }} />
          )}
        </Box>

        {/* Price + badges */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mb: 0.75 }}>
          <Typography sx={{ fontSize: 16, fontWeight: 600, color: "#111827" }}>
            {formatPrice(item.price, item.currency)}
          </Typography>
          {item.condition && (
            <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>
              · {item.condition}
            </Typography>
          )}
          {trustPercent !== null && <TrustBadge score={trustPercent} />}
          {rankingPercent !== null && <AiMatchBadge score={rankingPercent} />}
        </Box>

        {/* Seller */}
        <Box sx={{ mb: 0.5 }}>
          <SellerInfo seller_name={item.seller_name} seller_rating={item.seller_rating} />
        </Box>

        {/* Trust gauge */}
        {typeof item.trust_score === "number" && (
          <Box sx={{ mb: 0.75 }}>
            <SellerTrustGauge score={item.trust_score} />
          </Box>
        )}

        {/* Why this result */}
        {item.explanations?.length ? (
          <Box sx={{ mb: 0.75 }}>
            <ExplanationChips explanations={item.explanations} />
          </Box>
        ) : null}

        {/* Market signals */}
        {ragFeedbackPreview.length > 0 && (
          <Box sx={{ mb: 0.75 }}>
            {ragFeedbackPreview.map((text, i) => (
              <Typography
                key={i}
                sx={{ fontSize: 11, color: "#9ca3af", fontStyle: "italic", lineHeight: 1.5 }}
              >
                "{text}"
              </Typography>
            ))}
          </Box>
        )}

        {/* Actions */}
        <Box sx={{ display: "flex", gap: 1.5, alignItems: "center", flexWrap: "wrap" }}>
          <Box
            component="button"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent("send-chat", {
                  detail: `Analizza nel dettaglio il prodotto con ID ${item.ebay_id}`
                })
              )
            }
            sx={{
              background: "none",
              border: "none",
              p: 0,
              fontSize: 12,
              color: "#6b7280",
              cursor: "pointer",
              textDecoration: "underline",
              textDecorationColor: "#e5e7eb",
              textUnderlineOffset: "3px",
              fontFamily: "inherit",
              "&:hover": { color: "#374151" }
            }}
          >
            Dettagli prodotto
          </Box>

          {item.seller_name && (
            <Box
              component="button"
              onClick={() => setSellerOpen((v) => !v)}
              sx={{
                background: "none",
                border: "none",
                p: 0,
                fontSize: 12,
                color: "#6b7280",
                cursor: "pointer",
                display: "inline-flex",
                alignItems: "center",
                gap: 0.25,
                fontFamily: "inherit",
                textDecoration: "underline",
                textDecorationColor: "#e5e7eb",
                textUnderlineOffset: "3px",
                "&:hover": { color: "#374151" }
              }}
            >
              Seller deep dive
              <KeyboardArrowDownIcon
                sx={{
                  fontSize: 13,
                  transform: sellerOpen ? "rotate(180deg)" : "none",
                  transition: "transform 0.2s"
                }}
              />
            </Box>
          )}
        </Box>

        {/* Seller panel */}
        {item.seller_name && (
          <Collapse in={sellerOpen} timeout={200} unmountOnExit>
            <Box sx={{ mt: 1 }}>
              <SellerFeedbackPanel seller={item.seller_name} />
            </Box>
          </Collapse>
        )}
      </Box>
    </Box>
  )
}
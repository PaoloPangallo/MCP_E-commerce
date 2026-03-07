import { useMemo, useState } from "react"
import {
  Box,
  Chip,
  Link,
  Paper,
  Typography,
  Button,
  Collapse
} from "@mui/material"

import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ExpandLessIcon from "@mui/icons-material/ExpandLess"

import SellerInfo from "./SellerInfo"
import SellerTrustGauge from "./SellerTrustGauge"
import ExplanationChips from "./ExplanationChips"
import SellerFeedbackPanel from "./SellerFeedbackPanel"
import type { SearchItem } from "../component/searchTypes"

const FALLBACK_IMAGE = "https://via.placeholder.com/120?text=No+Image"

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") {
    return "Prezzo non disponibile"
  }

  return `${price} ${currency ?? ""}`.trim()
}

export default function SearchResultCard({ item }: { item: SearchItem }) {
  const [imageError, setImageError] = useState(false)
  const [showSellerPanel, setShowSellerPanel] = useState(false)

  const trustPercent =
    typeof item.trust_score === "number"
      ? Math.round(item.trust_score * 100)
      : null

  const rankingPercent =
    typeof item.ranking_score === "number"
      ? Math.round(item.ranking_score * 100)
      : null

  const imageProps = useMemo(() => {
    if (!item.url) {
      return {}
    }

    return {
      href: item.url,
      target: "_blank",
      rel: "noreferrer"
    }
  }, [item.url])

  const ragFeedbackPreview = Array.isArray(item.rag_feedback)
    ? item.rag_feedback
        .map((fb) => fb?.comment || "")
        .filter(Boolean)
        .slice(0, 2)
    : []

  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        mb: 2,
        borderRadius: "16px",
        border: "1px solid #e5e5e5",
        display: "flex",
        gap: 3,
        transition: "all 0.2s",
        backgroundColor: "#fff",
        flexDirection: { xs: "column", sm: "row" },
        "&:hover": {
          borderColor: "#d8d8d8",
          boxShadow: "0 4px 12px rgba(0,0,0,0.04)"
        }
      }}
    >
      <Box
        component={item.url ? "a" : "div"}
        aria-label={item.title ? `Apri ${item.title}` : "Apri prodotto"}
        {...imageProps}
        sx={{
          width: 120,
          height: 120,
          borderRadius: "10px",
          overflow: "hidden",
          flexShrink: 0,
          backgroundColor: "#f4f4f4",
          border: "1px solid #e5e5e5",
          display: "block"
        }}
      >
        <Box
          component="img"
          src={imageError ? FALLBACK_IMAGE : item.image_url || FALLBACK_IMAGE}
          alt={item.title || "Prodotto eBay"}
          loading="lazy"
          onError={() => setImageError(true)}
          sx={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
            display: "block"
          }}
        />
      </Box>

      <Box flex={1} minWidth={0}>
        {item.url ? (
          <Link
            href={item.url}
            target="_blank"
            rel="noreferrer"
            underline="hover"
            color="inherit"
            sx={{ display: "inline-block", mb: 1 }}
          >
            <Typography sx={{ fontWeight: 700, fontSize: 18, color: "#202123" }}>
              {item.title || "Titolo non disponibile"}
            </Typography>
          </Link>
        ) : (
          <Typography sx={{ fontWeight: 700, fontSize: 18, color: "#202123", mb: 1 }}>
            {item.title || "Titolo non disponibile"}
          </Typography>
        )}

        <Typography sx={{ fontSize: 24, fontWeight: 700, mb: 0.6 }}>
          {formatPrice(item.price, item.currency)}
        </Typography>

        {item.condition && (
          <Typography sx={{ fontSize: 13, color: "#6e6e80", mb: 1.5 }}>
            Condizione: {item.condition}
          </Typography>
        )}

        <SellerInfo
          seller_name={item.seller_name}
          seller_rating={item.seller_rating}
        />

        {typeof item.trust_score === "number" && (
          <SellerTrustGauge score={item.trust_score} />
        )}

        <Box display="flex" gap={1} mb={1.5} flexWrap="wrap">
          {trustPercent !== null && (
            <Chip
              icon={<VerifiedUserIcon sx={{ fontSize: 16 }} />}
              label={`Trust Score ${trustPercent}%`}
              size="small"
              sx={{
                backgroundColor: "#f4f4f4",
                border: "1px solid #e5e5e5",
                fontSize: 12
              }}
            />
          )}

          {rankingPercent !== null && (
            <Chip
              icon={<AutoAwesomeIcon sx={{ fontSize: 16 }} />}
              label={`AI Match ${rankingPercent}%`}
              size="small"
              sx={{
                backgroundColor: "#eef3ff",
                color: "#3b5ccc",
                fontSize: 12
              }}
            />
          )}

          {item._already_in_db && (
            <Chip
              label="Già presente nel DB"
              size="small"
              sx={{
                bgcolor: "#f6f6f6",
                color: "#666",
                fontSize: 11
              }}
            />
          )}
        </Box>

        <ExplanationChips explanations={item.explanations} />

        {ragFeedbackPreview.length > 0 && (
          <Box
            sx={{
              mt: 1.5,
              p: 1.5,
              borderRadius: 2,
              bgcolor: "#fafafa",
              border: "1px solid #ececf1"
            }}
          >
            <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#6e6e80", mb: 0.75 }}>
              Segnali RAG sul venditore
            </Typography>

            {ragFeedbackPreview.map((text, index) => (
              <Typography
                key={`${item.ebay_id}-rag-${index}`}
                sx={{ fontSize: 12.5, color: "#4b4b5a", lineHeight: 1.55, mb: index === ragFeedbackPreview.length - 1 ? 0 : 0.5 }}
              >
                “{text}”
              </Typography>
            ))}
          </Box>
        )}

        {item.seller_name && (
          <Box mt={2}>
            <Button
              variant="text"
              onClick={() => setShowSellerPanel((prev) => !prev)}
              endIcon={showSellerPanel ? <ExpandLessIcon /> : <ExpandMoreIcon />}
              sx={{
                px: 0,
                textTransform: "none",
                fontWeight: 600
              }}
            >
              {showSellerPanel ? "Nascondi seller deep dive" : "Mostra seller deep dive"}
            </Button>

            <Collapse in={showSellerPanel} timeout="auto" unmountOnExit>
              <SellerFeedbackPanel seller={item.seller_name} />
            </Collapse>
          </Box>
        )}
      </Box>
    </Paper>
  )
}
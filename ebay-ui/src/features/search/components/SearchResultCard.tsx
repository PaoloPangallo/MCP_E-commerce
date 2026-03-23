import { useState } from "react"
import { Box, Button, Collapse, Link, Typography, Chip } from "@mui/material"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import SettingsSuggestIcon from '@mui/icons-material/SettingsSuggest';
import LocalShippingIcon from "@mui/icons-material/LocalShipping"

import SellerTrustGauge from "../../seller/component/SellerTrustGauge.tsx"
import SellerFeedbackPanel from "../../seller/component/SellerFeedbackPanel.tsx"
import SellerInfo from "../../seller/SellerInfo.tsx"
import ExplanationChips from "./ExplanationChips.tsx"
import type { SearchItem } from "../types"

// ─── Helpers ──────────────────────────────────────────────────────────────────

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") return "—"
  return `${price} ${currency ?? ""}`.trim()
}

// Small inline trust pill — no card, no color fill
function TrustPill({ score }: { score: number }) {
  const pct = Math.round(score * 100)
  const good = pct >= 90
  return (
    <Box
      component="span"
      sx={{
        display: "inline-flex",
        alignItems: "center",
        gap: "4px",
        px: 1,
        py: "2px",
        borderRadius: "6px",
        border: "1px solid",
        borderColor: good ? "#bbf7d0" : "var(--border-color)",
        bgcolor: good ? "#f0fdf4" : "var(--bg-secondary)",
        boxShadow: good ? "0 1px 2px rgba(22, 163, 74, 0.05)" : "none",
      }}
    >
      <VerifiedUserIcon sx={{ fontSize: 12, color: good ? "#16a34a" : "#9ca3af" }} />
      <Typography
        component="span"
        sx={{
          fontSize: 11,
          fontWeight: 600,
          color: good ? "#15803d" : "var(--text-secondary)",
          letterSpacing: "0.01em"
        }}
      >
        {pct}% Trust
      </Typography>
    </Box>
  )
}

// AI match pill — same minimal style
function AiPill({ score }: { score: number }) {
  return (
    <Box
      component="span"
      sx={{
        display: "inline-flex",
        alignItems: "center",
        px: 1,
        py: "2px",
        borderRadius: "6px",
        border: "1px solid var(--border-color)",
        bgcolor: "rgba(124, 58, 237, 0.15)",
        boxShadow: "0 1px 2px rgba(124, 58, 237, 0.1)",
      }}
    >
      <Typography
        component="span"
        sx={{
          fontSize: 11,
          fontWeight: 600,
          color: "#7c3aed",
          letterSpacing: "0.01em"
        }}
      >
        {Math.round(score * 100)}% Match
      </Typography>
    </Box>
  )
}

// ─── Main card ────────────────────────────────────────────────────────────────

export default function SearchResultCard({ item }: { item: SearchItem }) {
  const [imageError,  setImageError]  = useState(false)
  const [sellerOpen,  setSellerOpen]  = useState(false)

  const trustPct   = typeof item.trust_score   === "number" ? item.trust_score   : null
  const rankingPct = typeof item.ranking_score === "number" ? item.ranking_score : null

  const ragPreviews = Array.isArray(item.rag_feedback)
    ? item.rag_feedback.map((fb) => fb?.comment || "").filter(Boolean).slice(0, 2)
    : []

  // Extract specs from NER
  const specs = item.ner_attributes?.specs || {};
  const hasNer = !!(item.ner_attributes?.brand || item.ner_attributes?.model || Object.keys(specs).length > 0);

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "flex-start",
        gap: 1.5,
        py: 1.75,
        borderBottom: "1px solid var(--border-color)",
        "&:last-child":    { borderBottom: "none" },
        "&:first-of-type": { pt: 0.5 },
      }}
    >
      {/* ── Thumbnail — small, quiet ──────────────────────────────────────── */}
      <Box
        component={item.url ? "a" : "div"}
        href={item.url}
        target="_blank"
        rel="noreferrer"
        sx={{
          width: 88,
          height: 88,
          borderRadius: "12px",
          overflow: "hidden",
          bgcolor: "var(--bg-primary)",
          border: "1px solid var(--border-color)",
          flexShrink: 0,
          display: "flex",
          alignItems: "center",
          justifyContent: "center",
          mt: 0.5,
          textDecoration: "none",
          transition: "all 0.2s ease-in-out",
          boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
          "&:hover": { 
            transform: "scale(1.02)",
            boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
            borderColor: "var(--accent-primary)"
          },
        }}
      >
        {!imageError && item.image_url ? (
          <Box
            component="img"
            src={item.image_url}
            alt={item.title || ""}
            loading="lazy"
            onError={() => setImageError(true)}
            sx={{ width: "100%", height: "100%", objectFit: "contain", p: 0.5 }}
          />
        ) : (
          <Box sx={{ width: 32, height: 32, bgcolor: "var(--bg-secondary)", borderRadius: "8px", display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
             <Typography variant="caption" color="text.disabled">No img</Typography>
          </Box>
        )}
      </Box>

      {/* ── Content ───────────────────────────────────────────────────────── */}
      <Box sx={{ flex: 1, minWidth: 0 }}>

        {/* Title + external icon */}
        <Box sx={{ display: "flex", alignItems: "flex-start", gap: 0.5, mb: 0.25 }}>
          {item.url ? (
            <Link
              href={item.url}
              target="_blank"
              rel="noreferrer"
              underline="none"
              sx={{
                fontSize: 15,
                fontWeight: 600,
                color: "var(--text-primary)",
                lineHeight: 1.3,
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
                "&:hover": { color: "#2563eb" },
              }}
            >
              {item.title || "Titolo non disponibile"}
            </Link>
          ) : (
            <Typography
              sx={{
                fontSize: 15, fontWeight: 600, color: "var(--text-primary)",
                lineHeight: 1.3,
                display: "-webkit-box",
                WebkitLineClamp: 2,
                WebkitBoxOrient: "vertical",
                overflow: "hidden",
              }}
            >
              {item.title || "Titolo non disponibile"}
            </Typography>
          )}
          {item.url && (
            <OpenInNewIcon sx={{ fontSize: 13, color: "#9ca3af", flexShrink: 0, mt: 0.4 }} />
          )}
        </Box>

        {/* ── NER Technical Attributes Row ────────────────────────────────── */}
        {hasNer && (
            <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 0.75 }}>
                {(item.ner_attributes?.brand || item.ner_attributes?.model) && (
                    <Box 
                        sx={{ 
                            px: 1, py: 0.25, borderRadius: '4px', bgcolor: 'var(--bg-secondary)', border: '1px solid var(--border-color)',
                            display: 'flex', alignItems: 'center', gap: 0.5
                        }}
                    >
                        <SettingsSuggestIcon sx={{ fontSize: 12, color: 'var(--text-secondary)' }} />
                        <Typography sx={{ fontSize: 11, fontWeight: 700, color: 'var(--text-secondary)', textTransform: 'uppercase' }}>
                            {item.ner_attributes?.brand} {item.ner_attributes?.model}
                        </Typography>
                    </Box>
                )}
                {Object.entries(specs).map(([key, value]) => (
                    value && (
                        <Chip 
                            key={key}
                            label={`${key}: ${value}`}
                            size="small" 
                            variant="outlined"
                            sx={{ 
                                height: 20, fontSize: 10, fontWeight: 600, color: 'var(--text-secondary)', borderColor: 'var(--border-color)', bgcolor: 'var(--bg-secondary)',
                                '& .MuiChip-label': { px: 1 }
                            }} 
                        />
                    )
                ))}
            </Box>
        )}

        {/* ── Meta row: price · condition · pills ───────────────────────── */}
        <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 1, mb: 0.75 }}>
          <Typography sx={{ fontSize: 18, fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
            {formatPrice(item.price, item.currency)}
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 0.5, flexWrap: 'wrap', alignItems: 'center' }}>
            {item.condition && (
              <Box
                sx={{
                  px: 1,
                  py: 0.25,
                  borderRadius: "4px",
                  bgcolor: (item.condition || "").toLowerCase().includes("nuovo") ? "rgba(37, 99, 235, 0.08)" : "var(--bg-secondary)",
                  border: "1px solid",
                  borderColor: (item.condition || "").toLowerCase().includes("nuovo") ? "rgba(37, 99, 235, 0.2)" : "var(--border-color)",
                }}
              >
                <Typography sx={{ fontSize: 10, fontWeight: 700, color: (item.condition || "").toLowerCase().includes("nuovo") ? "#2563eb" : "var(--text-secondary)", textTransform: 'uppercase' }}>
                  {item.condition}
                </Typography>
              </Box>
            )}
            {item.shipping && (
              <Box
                sx={{
                  px: 1,
                  py: 0.25,
                  borderRadius: "4px",
                  bgcolor: item.shipping.free ? "rgba(22, 163, 74, 0.08)" : "var(--bg-secondary)",
                  border: "1px solid",
                  borderColor: item.shipping.free ? "rgba(22, 163, 74, 0.2)" : "var(--border-color)",
                  display: 'flex',
                  alignItems: 'center',
                  gap: 0.5
                }}
              >
                <LocalShippingIcon sx={{ fontSize: 12, color: item.shipping.free ? "#16a34a" : "var(--text-secondary)" }} />
                <Typography sx={{ fontSize: 10, fontWeight: 800, color: item.shipping.free ? "#16a34a" : "var(--text-secondary)", textTransform: 'uppercase' }}>
                  {item.shipping.free ? "FREE SHIP" : `+ ${item.shipping.cost} ${item.shipping.currency || '€'}`}
                </Typography>
              </Box>
            )}

            {trustPct !== null && <TrustPill score={trustPct} />}
            {rankingPct !== null && <AiPill score={rankingPct} />}
          </Box>
        </Box>

        {/* ── Seller ────────────────────────────────────────────────────── */}
        <Box sx={{ mb: 0.5 }}>
          <SellerInfo seller_name={item.seller_name} seller_rating={item.seller_rating} />
        </Box>

        {/* ── Trust gauge ───────────────────────────────────────────────── */}
        {typeof item.trust_score === "number" && (
          <Box sx={{ mb: 0.6 }}>
            <SellerTrustGauge score={item.trust_score} />
          </Box>
        )}

        {/* ── Why chips ─────────────────────────────────────────────────── */}
        {item.explanations?.length ? (
          <Box sx={{ mb: 0.6 }}>
            <ExplanationChips explanations={item.explanations} />
          </Box>
        ) : null}

        {/* ── RAG feedback quotes ───────────────────────────────────────── */}
        {ragPreviews.length > 0 && (
          <Box sx={{ mb: 0.75 }}>
            {ragPreviews.map((text, i) => (
              <Typography key={i} sx={{ fontSize: 11.5, color: "var(--text-secondary)", fontStyle: "italic", lineHeight: 1.55 }}>
                "{text}"
              </Typography>
            ))}
          </Box>
        )}

        {/* ── Actions — inline text links, no buttons ───────────────────── */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, flexWrap: "wrap", mt: 1 }}>
          <Button
            size="small"
            variant="contained"
            disableElevation
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent("send-chat", {
                  detail: `Dettagli per ${item.title} (ID: ${item.ebay_id})`,
                })
              )
            }
            sx={{
              textTransform: "none",
              borderRadius: "8px",
              fontSize: 12,
              fontWeight: 600,
              bgcolor: "var(--accent-primary)",
              color: "#fff",
              px: 2,
              "&:hover": { bgcolor: "var(--accent-primary)", opacity: 0.9, boxShadow: "0 4px 12px var(--accent-primary)" }
            }}
          >
            Dettagli
          </Button>

          <Button
            size="small"
            variant="outlined"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent("send-chat", {
                  detail: `Calcola i costi di spedizione per l'oggetto ${item.ebay_id} in Italia 🇮🇹`,
                })
              )
            }
            sx={{
              textTransform: "none",
              borderRadius: "8px",
              fontSize: 12,
              fontWeight: 600,
              color: "var(--text-primary)",
              px: 2,
              "&:hover": { borderColor: "var(--accent-primary)", bgcolor: "var(--bg-secondary)" }
            }}
          >
            Spedizione
          </Button>

          <Button
            size="small"
            variant="outlined"
            onClick={() =>
              window.dispatchEvent(
                new CustomEvent("send-chat", {
                  detail: `Trend di mercato, statistiche e andamento prezzi medi online per: ${item.title}`,
                })
              )
            }
            sx={{
              textTransform: "none",
              borderRadius: "8px",
              fontSize: 12,
              fontWeight: 600,
              color: "var(--text-primary)",
              px: 2,
              "&:hover": { borderColor: "var(--accent-primary)", bgcolor: "var(--bg-secondary)" }
            }}
          >
            Analisi mercato
          </Button>

          {item.seller_name && (
            <Button
              size="small"
              variant="text"
              onClick={() => setSellerOpen((v) => !v)}
              endIcon={
                <KeyboardArrowDownIcon
                  sx={{
                    fontSize: 14,
                    transform: sellerOpen ? "rotate(180deg)" : "none",
                    transition: "transform 0.18s",
                  }}
                />
              }
              sx={{
                textTransform: "none",
                fontSize: 12,
                fontWeight: 600,
                color: "var(--text-secondary)",
                "&:hover": { color: "var(--text-primary)", bgcolor: "transparent" }
              }}
            >
              Seller
            </Button>
          )}
        </Box>

        {/* ── Seller deep dive panel ────────────────────────────────────── */}
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
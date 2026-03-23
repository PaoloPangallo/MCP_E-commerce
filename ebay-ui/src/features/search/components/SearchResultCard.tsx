import { useState } from "react"
import { Box, Button, Collapse, Link, Typography, Chip } from "@mui/material"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
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
  const formatted = price.toLocaleString('it-IT', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${formatted} ${currency ?? ""}`.trim()
}

function ScoreBar({ label, score, color }: { label: string, score: number, color: string }) {
  const pct = Math.round(score * 100)
  return (
    <Box sx={{ width: "100%" }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", mb: 0.5 }}>
        <Typography sx={{ fontSize: 9, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: '0.05em' }}>{label}</Typography>
        <Typography sx={{ fontSize: 9, fontWeight: 800, color }}>{pct}%</Typography>
      </Box>
      <Box sx={{ height: 4, bgcolor: "var(--bg-secondary)", borderRadius: 2, overflow: "hidden", border: '1px solid var(--border-color)' }}>
        <Box sx={{ width: `${pct}%`, height: "100%", bgcolor: color, transition: 'width 0.8s ease' }} />
      </Box>
    </Box>
  )
}

// ─── Main card ────────────────────────────────────────────────────────────────

export default function SearchResultCard({ 
  item, 
  variant = 'list',
  index = 0
}: { 
  item: SearchItem, 
  variant?: 'list' | 'compact',
  index?: number
}) {
  const [imageError, setImageError] = useState(false)
  const [sellerOpen, setSellerOpen] = useState(false)

  const rankingPct = typeof item.ranking_score === "number" ? item.ranking_score : null
  const valuePct = (item as any).value_score ?? 0

  const isCompact = variant === 'compact'

  if (isCompact) {
    return (
      <Box
        sx={{
          width: 260,
          flexShrink: 0,
          borderRadius: "16px",
          bgcolor: "var(--bg-primary)",
          border: "1px solid var(--border-color)",
          p: 2,
          display: "flex",
          flexDirection: "column",
          gap: 1.5,
          transition: "all 0.2s ease",
          "&:hover": {
            transform: "translateY(-4px)",
            boxShadow: "0 12px 24px rgba(0,0,0,0.1)",
            borderColor: "var(--brand-primary)"
          },
          cursor: "pointer",
          scrollSnapAlign: "start"
        }}
        onClick={() => item.url && window.open(item.url, '_blank')}
      >
        <Box sx={{ position: "relative", width: "100%", aspectRatio: "4/3", borderRadius: "10px", overflow: "hidden", bgcolor: "var(--bg-secondary)", display: 'flex', alignItems: 'center', justifyContent: 'center' }}>
          {!imageError && item.image_url ? (
            <Box
              component="img"
              src={item.image_url}
              alt={item.title || ""}
              onError={() => setImageError(true)}
              sx={{ width: "100%", height: "100%", objectFit: "contain", p: 0.5 }}
            />
          ) : (
             <Typography variant="caption" color="text.disabled">No Image</Typography>
          )}
          {index < 2 && rankingPct && rankingPct > 0.6 && (
            <Chip
              label="PICK"
              size="small"
              sx={{
                position: "absolute",
                top: 8,
                left: 8,
                bgcolor: "var(--brand-primary)",
                color: "#ffffff",
                fontWeight: 800,
                fontSize: 9,
                height: 18
              }}
            />
          )}
        </Box>

        <Typography
          sx={{
            fontSize: 13.5,
            fontWeight: 700,
            color: "var(--text-primary)",
            lineHeight: 1.3,
            display: "-webkit-box",
            WebkitLineClamp: 2,
            WebkitBoxOrient: "vertical",
            overflow: "hidden",
            minHeight: "2.6em"
          }}
        >
          {item.title}
        </Typography>

        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <Typography sx={{ fontSize: 17, fontWeight: 800, color: "var(--text-primary)" }}>
            {formatPrice(item.price, item.currency)}
          </Typography>
          {item.condition && (
             <Typography sx={{ fontSize: 10, fontWeight: 700, color: "var(--text-secondary)", textTransform: 'uppercase', opacity: 0.8 }}>
               · {item.condition}
             </Typography>
          )}
        </Box>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.25, my: 0.5 }}>
          <ScoreBar label="Match" score={rankingPct || 0} color="var(--brand-primary)" />
          <ScoreBar label="Valore" score={valuePct} color="var(--success)" />
        </Box>

        {item.shipping && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mt: 'auto', p: 1, borderRadius: "24px", bgcolor: "var(--bg-secondary)", border: '1px solid var(--border-color)' }}>
            <LocalShippingIcon sx={{ fontSize: 14, color: "var(--text-secondary)" }} />
            <Typography sx={{ fontSize: 10, fontWeight: 700, color: item.shipping.free ? "var(--success)" : "var(--text-primary)" }}>
              {item.shipping.free ? "CONSEGNA GRATIS" : `SPED. ${formatPrice(item.shipping.cost, item.shipping.currency || '€')}`}
            </Typography>
          </Box>
        )}

        <Button
          fullWidth
          variant="outlined"
          sx={{
            mt: 0.5,
            textTransform: "none",
            borderRadius: "24px",
            fontSize: 12,
            fontWeight: 600,
            borderColor: "var(--border-color)",
            color: "var(--text-secondary)",
            py: 0.75,
            "&:hover": { borderColor: "var(--brand-primary)", color: "var(--brand-primary)", bgcolor: 'transparent' }
          }}
        >
          Dettagli eBay <OpenInNewIcon sx={{ fontSize: 12, ml: 0.5 }} />
        </Button>
      </Box>
    )
  }

  // Original List View
  const ragPreviews = Array.isArray(item.rag_feedback)
    ? item.rag_feedback.map((fb) => fb?.comment || "").filter(Boolean).slice(0, 2)
    : []

  const specs = item.ner_attributes?.specs || {};
  const hasNer = !!(item.ner_attributes?.brand || item.ner_attributes?.model || Object.keys(specs).length > 0);

  return (
    <Box
      sx={{
        display: "flex",
        alignItems: "flex-start",
        gap: 2,
        py: 2.5,
        borderBottom: "1px solid var(--border-color)",
        "&:last-child": { borderBottom: "none" },
        "&:first-of-type": { pt: 0.5 },
      }}
    >
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
          textDecoration: "none",
          transition: "all 0.2s ease-in-out",
          boxShadow: "0 2px 8px rgba(0,0,0,0.04)",
          "&:hover": { 
            transform: "scale(1.02)",
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)",
            borderColor: "var(--brand-primary)"
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
        {index < 2 && rankingPct && rankingPct > 0.6 && (
           <Chip
             label="PICK"
             size="small"
             sx={{
               position: "absolute",
               top: 4,
               left: 4,
               bgcolor: "var(--brand-primary)",
               color: "#fff",
               fontWeight: 800,
               fontSize: 8,
               height: 16
             }}
           />
        )}
      </Box>

      <Box sx={{ flex: 1, minWidth: 0 }}>
        <Box sx={{ display: "flex", alignItems: "flex-start", gap: 0.5, mb: 0.5 }}>
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
              "&:hover": { color: "var(--brand-primary)" },
            }}
          >
            {item.title || "Titolo non disponibile"}
          </Link>
          {item.url && <OpenInNewIcon sx={{ fontSize: 13, color: "#9ca3af", flexShrink: 0, mt: 0.4 }} />}
        </Box>

        {hasNer && (
          <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5, mb: 1 }}>
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
          </Box>
        )}

        <Box sx={{ display: "flex", alignItems: "center", flexWrap: "wrap", gap: 1.5, mb: 1.5 }}>
          <Typography sx={{ fontSize: 18, fontWeight: 700, color: "var(--text-primary)", letterSpacing: "-0.02em" }}>
            {formatPrice(item.price, item.currency)}
          </Typography>
          
          <Box sx={{ display: 'flex', gap: 0.75, flexWrap: 'wrap', alignItems: 'center' }}>
            {item.condition && (
              <Box
                sx={{
                  px: 1, py: 0.25, borderRadius: "4px",
                  bgcolor: (item.condition || "").toLowerCase().includes("nuovo") ? "var(--brand-soft)" : "var(--bg-secondary)",
                  border: "1px solid",
                  borderColor: (item.condition || "").toLowerCase().includes("nuovo") ? "var(--brand-primary)" : "var(--border-color)",
                  opacity: (item.condition || "").toLowerCase().includes("nuovo") ? 1 : 0.8
                }}
              >
                <Typography sx={{ fontSize: 10, fontWeight: 700, color: (item.condition || "").toLowerCase().includes("nuovo") ? "var(--brand-primary)" : "var(--text-secondary)", textTransform: 'uppercase' }}>
                  {item.condition}
                </Typography>
              </Box>
            )}
          </Box>
        </Box>

        <Box sx={{ mb: 0.5 }}>
          <SellerInfo seller_name={item.seller_name} seller_rating={item.seller_rating} />
        </Box>

        {typeof item.trust_score === "number" && (
          <Box sx={{ mb: 0.6 }}>
            <SellerTrustGauge score={item.trust_score} />
          </Box>
        )}

        {item.explanations?.length ? (
          <Box sx={{ mb: 0.6 }}>
            <ExplanationChips explanations={item.explanations} />
          </Box>
        ) : null}

        {ragPreviews.length > 0 && (
          <Box sx={{ mb: 1, p: 0.5 }}>
            {ragPreviews.map((text, i) => (
              <Typography key={i} sx={{ fontSize: 11.5, color: "var(--text-secondary)", fontStyle: "italic", lineHeight: 1.55 }}>
                "{text}"
              </Typography>
            ))}
          </Box>
        )}

        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 1 }}>
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
              bgcolor: "var(--brand-primary)",
              color: "#fff",
              px: 2,
              "&:hover": { bgcolor: "var(--brand-primary)", opacity: 0.9 }
            }}
          >
            Dettagli
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
                    transition: "transform 0.2s",
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
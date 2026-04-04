import { Box, Typography, Tooltip } from "@mui/material"
import WarningAmberIcon from "@mui/icons-material/WarningAmber"
import type { Feedback } from "../types"

function normalizeRating(rating?: number): "positive" | "neutral" | "negative" {
  if (rating === undefined) return "neutral"
  if (rating >= 4) return "positive"
  if (rating <= 2) return "negative"
  return "neutral"
}

function formatDate(value?: string | number) {
  if (!value) return null
  const d = typeof value === "number" ? new Date(value) : new Date(String(value))
  if (Number.isNaN(d.getTime())) return String(value)
  return d.toLocaleDateString("it-IT", { 
    year: "numeric", 
    month: "short", 
    day: "numeric",
    hour: "2-digit",
    minute: "2-digit"
  })
}

const ratingMeta = {
  positive: { dot: "#10b981", text: "#059669", bg: "#ecfdf5", label: "Positivo" },
  neutral:  { dot: "#f59e0b", text: "#d97706", bg: "#fffbeb", label: "Neutro"   },
  negative: { dot: "#ef4444", text: "#dc2626", bg: "#fef2f2", label: "Negativo" }
}

export default function FeedbackCard({ feedback }: { feedback: Feedback }) {
  const type = normalizeRating(feedback.rating)
  const meta = ratingMeta[type]
  const date = formatDate(feedback.time ?? feedback.date)
  const initial = (feedback.user || "U").charAt(0).toUpperCase()

  const nlp = feedback.nlp_sentiment
  const isFalsePositive = type === "positive" && nlp !== undefined && nlp < 0.40
  const isFalseNegative = type === "negative" && nlp !== undefined && nlp > 0.60
  const hasMismatch = isFalsePositive || isFalseNegative
  
  const mismatchTooltip = isFalsePositive 
    ? `Il testo è risultato tossico/negativo all'analisi AI (${Math.round(nlp! * 100)}%) nonostante la stella Positiva eBay`
    : isFalseNegative 
      ? `Il testo è risultato positivo/entusiasta all'analisi AI (${Math.round(nlp! * 100)}%) nonostante la stella Negativa eBay`
      : ""

  return (
    <Box
      sx={{
        py: 2,
        borderBottom: "1px solid var(--border-color)",
        "&:last-child": { borderBottom: "none" }
      }}
    >
      {/* Header row */}
      <Box sx={{ display: "flex", alignItems: "flex-start", gap: 1.5, mb: 1 }}>
        <Box
          sx={{
            width: 28,
            height: 28,
            borderRadius: "50%",
            bgcolor: "var(--bg-secondary)",
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 11,
            fontWeight: 800,
            color: "var(--text-primary)",
            border: '1px solid var(--border-color)',
            flexShrink: 0
          }}
        >
          {initial}
        </Box>
        
        <Box sx={{ flex: 1, minWidth: 0 }}>
           <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, flexWrap: "wrap", mb: 0.5 }}>
            <Typography sx={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>
              {feedback.user || "Utente"}
            </Typography>
            <Box
              sx={{
                px: 1,
                py: "1px",
                borderRadius: "6px",
                bgcolor: meta.bg,
                border: `1px solid ${meta.dot}40`,
                display: 'flex',
                alignItems: 'center',
                gap: 0.5
              }}
            >
              <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: meta.dot }} />
              <Typography sx={{ fontSize: 10, fontWeight: 800, color: meta.text, textTransform: 'uppercase' }}>
                {meta.label}
              </Typography>
            </Box>

            {hasMismatch && (
              <Tooltip title={mismatchTooltip} placement="top" arrow>
                <Box sx={{ cursor: 'help', display: 'flex', alignItems: 'center', gap: 0.5, bgcolor: '#fef2f2', border: '1px solid #fca5a5', px: 1, py: '2px', borderRadius: '4px' }}>
                  <WarningAmberIcon sx={{ fontSize: 12, color: '#dc2626' }} />
                  <Typography sx={{ fontSize: 9, fontWeight: 800, color: '#dc2626', textTransform: 'uppercase', letterSpacing: '0.05em' }}>
                    Alert AI
                  </Typography>
                </Box>
              </Tooltip>
            )}

            {date && (
              <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", ml: "auto", fontWeight: 500 }}>
                {date.replace(",", " ·")}
              </Typography>
            )}
           </Box>

           {feedback.item_title && (
             <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5, mt: 0.5, mb: 1 }}>
               <Typography 
                 sx={{ 
                   fontSize: 11, 
                   color: "var(--text-secondary)", 
                   whiteSpace: "nowrap", 
                   overflow: "hidden", 
                   textOverflow: "ellipsis",
                   bgcolor: "var(--bg-secondary)",
                   px: 1,
                   py: 0.25,
                   borderRadius: "4px",
                   border: "1px solid var(--border-color)",
                   display: "inline-block"
                 }}
               >
                 Acquisto: {feedback.item_title}
               </Typography>
             </Box>
           )}

          {/* Comment */}
          <Typography
            sx={{
              fontSize: 13,
              color: hasMismatch ? "#dc2626" : "var(--text-primary)",
              lineHeight: 1.5,
              fontWeight: hasMismatch ? 500 : 400,
              mt: feedback.item_title ? 0.5 : 1,
              wordBreak: "break-word",
              overflowWrap: "anywhere",
              whiteSpace: "pre-line"
            }}
          >
            {feedback.comment || "Nessun commento fornito."}
          </Typography>
        </Box>
      </Box>
    </Box>
  )

}
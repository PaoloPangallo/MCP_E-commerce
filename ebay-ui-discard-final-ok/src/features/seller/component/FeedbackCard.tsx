import { Box, Typography } from "@mui/material"
import type { Feedback } from "../../../types"

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
  return d.toLocaleDateString("it-IT", { year: "numeric", month: "short", day: "numeric" })
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

  return (
    <Box
      sx={{
        py: 2,
        borderBottom: "1px solid #f0f0f0",
        "&:last-child": { borderBottom: "none" }
      }}
    >
      {/* Header row */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, mb: 1 }}>
        <Box
          sx={{
            width: 24,
            height: 24,
            borderRadius: "50%",
            bgcolor: "#f3f4f6",
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 10,
            fontWeight: 700,
            color: "#6b7280",
            border: '1px solid #e5e7eb',
            flexShrink: 0
          }}
        >
          {initial}
        </Box>
        
        <Box sx={{ flex: 1 }}>
           <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
            <Typography sx={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>
              {feedback.user || "Utente"}
            </Typography>
            <Box
              sx={{
                px: 1,
                py: "1px",
                borderRadius: "4px",
                bgcolor: meta.bg,
                border: "1px solid transparent",
                display: 'flex',
                alignItems: 'center',
                gap: 0.5
              }}
            >
              <Box sx={{ width: 6, height: 6, borderRadius: '50%', bgcolor: meta.dot }} />
              <Typography sx={{ fontSize: 10, fontWeight: 700, color: meta.text, textTransform: 'uppercase' }}>
                {meta.label}
              </Typography>
            </Box>
           </Box>
        </Box>

        {date && (
          <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>
            {date}
          </Typography>
        )}
      </Box>

      {/* Comment */}
      <Typography
        sx={{
          fontSize: 13,
          color: "#4b5563",
          lineHeight: 1.5,
          pl: "36px" // align with text after avatar
        }}
      >
        {feedback.comment || "Nessun commento disponibile"}
      </Typography>
    </Box>
  )
}
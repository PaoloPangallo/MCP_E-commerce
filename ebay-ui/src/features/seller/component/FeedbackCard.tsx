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
  positive: { dot: "#6ee7b7", text: "#059669", label: "positivo" },
  neutral:  { dot: "#fcd34d", text: "#d97706", label: "neutro"   },
  negative: { dot: "#fca5a5", text: "#dc2626", label: "negativo" }
}

export default function FeedbackCard({ feedback }: { feedback: Feedback }) {
  const type = normalizeRating(feedback.rating)
  const meta = ratingMeta[type]
  const date = formatDate(feedback.time ?? feedback.date)

  return (
    <Box
      sx={{
        py: 1.5,
        borderBottom: "1px solid #f5f5f5",
        "&:last-child": { borderBottom: "none" }
      }}
    >
      {/* Header row */}
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5 }}>
        {/* Colored dot instead of Avatar */}
        <Box
          sx={{
            width: 7,
            height: 7,
            borderRadius: "50%",
            bgcolor: meta.dot,
            flexShrink: 0
          }}
        />
        <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>
          {feedback.user || "Utente"}
        </Typography>
        <Box
          sx={{
            px: 0.75,
            py: 0.1,
            borderRadius: "6px",
            bgcolor: "#f9fafb",
            border: "1px solid #f0f0f0"
          }}
        >
          <Typography sx={{ fontSize: 10, color: meta.text }}>{meta.label}</Typography>
        </Box>
        {date && (
          <Typography sx={{ fontSize: 11, color: "#d1d5db", ml: "auto" }}>
            {date}
          </Typography>
        )}
      </Box>

      {/* Comment */}
      <Typography
        sx={{
          fontSize: 12,
          color: "#6b7280",
          lineHeight: 1.6,
          pl: "15px" // align with text after dot
        }}
      >
        {feedback.comment || "Nessun commento disponibile"}
      </Typography>
    </Box>
  )
}
import { useEffect, useMemo, useState } from "react"
import { Box, CircularProgress, Typography } from "@mui/material"

import FeedbackCard from "./component/FeedbackCard.tsx"
import type { Feedback } from "../../types"

interface Props {
  feedbacks?: Feedback[]
  loading?: boolean
  error?: string | null
  initialLimit?: number
  title?: string
}

export default function SellerFeedbackList({
  feedbacks = [],
  loading = false,
  error = null,
  initialLimit = 6,
  title = "Feedback venditore"
}: Props) {
  const safe = Array.isArray(feedbacks) ? feedbacks : []
  const [visibleCount, setVisibleCount] = useState(initialLimit)

  useEffect(() => {
    setVisibleCount(initialLimit)
  }, [safe, initialLimit])

  const sorted = useMemo(
    () =>
      [...safe].sort((a, b) => {
        const da = new Date(a.date ?? a.time ?? 0).getTime()
        const db = new Date(b.date ?? b.time ?? 0).getTime()
        return db - da
      }),
    [safe]
  )

  const visible = sorted.slice(0, visibleCount)

  if (loading) {
    return (
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, py: 1 }}>
        <CircularProgress size={13} sx={{ color: "#9ca3af" }} />
        <Typography sx={{ fontSize: 12, color: "#9ca3af" }}>Caricamento…</Typography>
      </Box>
    )
  }

  if (error) {
    return (
      <Typography sx={{ fontSize: 12, color: "#dc2626", py: 0.5 }}>{error}</Typography>
    )
  }

  if (safe.length === 0) {
    return (
      <Typography sx={{ fontSize: 12, color: "#9ca3af", py: 0.5 }}>
        Nessun feedback disponibile
      </Typography>
    )
  }

  return (
    <Box>
      {/* Title + count */}
      {title && (
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 0.5 }}>
          <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#6b7280" }}>
            {title}
          </Typography>
          <Typography sx={{ fontSize: 11, color: "#d1d5db" }}>
            {safe.length}
          </Typography>
        </Box>
      )}

      {/* Cards */}
      {visible.map((feedback, index) => (
        <FeedbackCard
          key={`${feedback.user ?? "anon"}-${feedback.date ?? feedback.time ?? "nd"}-${index}`}
          feedback={feedback}
        />
      ))}

      {/* Load more */}
      {visibleCount < sorted.length && (
        <Box
          component="button"
          onClick={() =>
            setVisibleCount((prev) => Math.min(prev + initialLimit, sorted.length))
          }
          sx={{
            background: "none",
            border: "none",
            p: 0,
            mt: 0.75,
            fontSize: 12,
            color: "#9ca3af",
            cursor: "pointer",
            fontFamily: "inherit",
            textDecoration: "underline",
            textDecorationColor: "#e5e7eb",
            textUnderlineOffset: "3px",
            "&:hover": { color: "#6b7280" }
          }}
        >
          Mostra altri ({sorted.length - visibleCount} rimasti)
        </Box>
      )}
    </Box>
  )
}
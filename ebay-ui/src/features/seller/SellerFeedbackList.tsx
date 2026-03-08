import { useEffect, useMemo, useState } from "react"
import { Box, Typography, CircularProgress, Chip, Button } from "@mui/material"

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

  // --------------------------------------------------
  // SAFE ARRAY
  // --------------------------------------------------

  const safeFeedbacks = Array.isArray(feedbacks) ? feedbacks : []

  const [visibleCount, setVisibleCount] = useState(initialLimit)

  useEffect(() => {
    setVisibleCount(initialLimit)
  }, [safeFeedbacks, initialLimit])

  // --------------------------------------------------
  // SORT BY DATE (most recent first)
  // --------------------------------------------------

  const sortedFeedbacks = useMemo(() => {
    return [...safeFeedbacks].sort((a, b) => {
      const da = new Date(a.date ?? a.time ?? 0).getTime()
      const db = new Date(b.date ?? b.time ?? 0).getTime()
      return db - da
    })
  }, [safeFeedbacks])

  const visibleFeedbacks = sortedFeedbacks.slice(0, visibleCount)

  // --------------------------------------------------
  // LOADING
  // --------------------------------------------------

  if (loading) {
    return (
      <Box display="flex" alignItems="center" gap={1} mt={1}>
        <CircularProgress size={16} />
        <Typography sx={{ color: "#666", fontSize: 13 }}>
          Caricamento feedback venditore...
        </Typography>
      </Box>
    )
  }

  // --------------------------------------------------
  // ERROR
  // --------------------------------------------------

  if (error) {
    return (
      <Typography sx={{ mt: 1, color: "#c62828", fontSize: 13 }}>
        {error}
      </Typography>
    )
  }

  // --------------------------------------------------
  // EMPTY
  // --------------------------------------------------

  if (safeFeedbacks.length === 0) {
    return (
      <Typography sx={{ mt: 1, color: "#666", fontSize: 13 }}>
        Nessun feedback disponibile
      </Typography>
    )
  }

  // --------------------------------------------------
  // RENDER
  // --------------------------------------------------

  return (
    <Box mt={1} sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>

      <Box
        display="flex"
        alignItems="center"
        justifyContent="space-between"
        gap={2}
      >
        <Typography
          sx={{
            fontSize: 13,
            fontWeight: 600,
            color: "#444"
          }}
        >
          {title}
        </Typography>

        <Chip
          label={`${safeFeedbacks.length} recensioni`}
          size="small"
          sx={{
            fontSize: 11,
            bgcolor: "#f4f4f4"
          }}
        />
      </Box>

      {visibleFeedbacks.map((feedback, index) => {

  const key =
    `${feedback.user ?? "anon"}-${feedback.date ?? feedback.time ?? "nodate"}-${index}`

  return (
    <FeedbackCard
      key={key}
      feedback={feedback}
    />
  )
})}

      {/* LOAD MORE */}

      {visibleCount < sortedFeedbacks.length && (
        <Box mt={1} textAlign="center">

          <Button
            size="small"
            variant="outlined"
            onClick={() =>
              setVisibleCount(prev =>
                Math.min(prev + initialLimit, sortedFeedbacks.length)
              )
            }
            sx={{
              textTransform: "none",
              borderRadius: "16px"
            }}
          >
            Mostra altri feedback
          </Button>

        </Box>
      )}

      {/* COUNTER */}

      <Typography
        sx={{
          fontSize: 12,
          color: "#888",
          textAlign: "center",
          mt: 0.5
        }}
      >
        Mostrati {Math.min(visibleCount, sortedFeedbacks.length)} di {sortedFeedbacks.length} feedback
      </Typography>

    </Box>
  )
}
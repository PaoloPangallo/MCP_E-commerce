import { useState } from "react"
import {
  Box,
  Typography,
  CircularProgress,
  Chip,
  Button
} from "@mui/material"

import FeedbackCard from "./FeedbackCard"
import type { Feedback } from "./FeedbackCard"


// --------------------------------------------------
// PROPS TYPE
// --------------------------------------------------

interface Props {
  feedbacks?: Feedback[]
  loading?: boolean
  initialLimit?: number
}


// --------------------------------------------------
// COMPONENT
// --------------------------------------------------

export default function SellerFeedbackList({
  feedbacks = [],
  loading = false,
  initialLimit = 10
}: Props) {

  const [visibleCount, setVisibleCount] = useState<number>(initialLimit)

  // --------------------------------------------------
  // LOADING STATE
  // --------------------------------------------------

  if (loading) {
    return (
      <Box display="flex" alignItems="center" gap={1} mt={1}>
        <CircularProgress size={16} />

        <Typography
          sx={{
            color: "#666",
            fontSize: 13
          }}
        >
          Caricamento feedback venditore...
        </Typography>
      </Box>
    )
  }

  // --------------------------------------------------
  // EMPTY STATE
  // --------------------------------------------------

  if (!feedbacks || feedbacks.length === 0) {
    return (
      <Typography
        sx={{
          mt: 1,
          color: "#666",
          fontSize: 13
        }}
      >
        Nessun feedback disponibile
      </Typography>
    )
  }

  // --------------------------------------------------
  // LIMIT FEEDBACKS
  // --------------------------------------------------

  const visibleFeedbacks: Feedback[] = feedbacks.slice(0, visibleCount)

  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (

    <Box
      mt={1}
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 1.5
      }}
    >

      {/* HEADER */}
      <Box display="flex" alignItems="center" justifyContent="space-between">

        <Typography
          sx={{
            fontSize: 13,
            fontWeight: 600,
            color: "#444"
          }}
        >
          Feedback venditore
        </Typography>

        <Chip
          label={`${feedbacks.length} recensioni`}
          size="small"
          sx={{
            fontSize: 11,
            bgcolor: "#f4f4f4"
          }}
        />

      </Box>


      {/* LIST */}
      {visibleFeedbacks.map((f: Feedback, i: number) => {

        const key =
          `${f.user ?? "user"}-${f.time ?? "time"}-${i}`

        return (
          <FeedbackCard
            key={key}
            feedback={f}
          />
        )

      })}


      {/* LOAD MORE BUTTON */}
      {visibleCount < feedbacks.length && (

        <Box mt={2} textAlign="center">

          <Button
            size="small"
            onClick={() =>
              setVisibleCount(prev => prev + initialLimit)
            }
          >
            Mostra altri feedback
          </Button>

        </Box>

      )}


      {/* FOOTER INFO */}
      <Typography
        sx={{
          fontSize: 12,
          color: "#888",
          textAlign: "center",
          mt: 1
        }}
      >
        Mostrati {Math.min(visibleCount, feedbacks.length)} di {feedbacks.length} feedback
      </Typography>

    </Box>

  )

}
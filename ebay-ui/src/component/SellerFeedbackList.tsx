import { Box, Typography, CircularProgress, Chip } from "@mui/material";
import FeedbackCard from "./FeedbackCard";


// --------------------------------------------------
// TYPES
// --------------------------------------------------

export interface Feedback {
  user?: string
  rating?: number | string
  comment?: string
  time?: string
}


// --------------------------------------------------
// COMPONENT
// --------------------------------------------------

export default function SellerFeedbackList({
  feedbacks = [],
  loading = false,
  limit = 10,
}: {
  feedbacks?: Feedback[]
  loading?: boolean
  limit?: number
}) {

  // --------------------------------------------------
  // LOADING STATE
  // --------------------------------------------------

  if (loading) {
    return (
      <Box
        display="flex"
        alignItems="center"
        gap={1}
        mt={1}
      >
        <CircularProgress size={16} />

        <Typography
          sx={{
            color: "#666",
            fontSize: 13,
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
          fontSize: 13,
        }}
      >
        Nessun feedback disponibile
      </Typography>
    )

  }


  // --------------------------------------------------
  // LIMIT FEEDBACKS
  // --------------------------------------------------

  const visibleFeedbacks = feedbacks.slice(0, limit)


  // --------------------------------------------------
  // UI
  // --------------------------------------------------

  return (

    <Box
      mt={1}
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 1.5,
      }}
    >

      {/* HEADER */}
      <Box display="flex" alignItems="center" justifyContent="space-between">

        <Typography
          sx={{
            fontSize: 13,
            fontWeight: 600,
            color: "#444",
          }}
        >
          Feedback venditore
        </Typography>

        <Chip
          label={`${feedbacks.length} recensioni`}
          size="small"
          sx={{
            fontSize: 11,
            bgcolor: "#f4f4f4",
          }}
        />

      </Box>


      {/* LIST */}
      {visibleFeedbacks.map((f, i) => {

        const key =
          `${f.user ?? "user"}-${f.time ?? "time"}-${i}`

        return (

          <FeedbackCard
            key={key}
            feedback={f}
          />

        )

      })}


      {/* MORE */}
      {feedbacks.length > limit && (

        <Typography
          sx={{
            fontSize: 12,
            color: "#888",
            textAlign: "center",
            mt: 1
          }}
        >
          Mostrati {limit} di {feedbacks.length} feedback
        </Typography>

      )}

    </Box>

  )

}
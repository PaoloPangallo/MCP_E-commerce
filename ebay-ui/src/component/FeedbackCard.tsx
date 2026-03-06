import { Paper, Typography, Box, Chip, Avatar } from "@mui/material"
import type { Feedback } from "../api/sellerApi"

interface Props {
  feedback: Feedback
}

export default function FeedbackCard({ feedback }: Props) {
  const ratingColor =
    feedback.rating === "Positive"
      ? "success"
      : feedback.rating === "Negative"
      ? "error"
      : "default"

  const formattedDate = feedback.time
    ? new Date(feedback.time).toLocaleDateString("it-IT", {
        day: "2-digit",
        month: "short",
        year: "numeric"
      })
    : "—"

  return (
    <Paper
      elevation={0}
      sx={{
        p: 2,
        border: "1px solid #e5e5e5",
        borderRadius: 2,
        transition: "all 0.2s",
        "&:hover": {
          borderColor: "#d1d1d1",
          boxShadow: "0 2px 8px rgba(0,0,0,0.06)"
        }
      }}
    >
      <Box display="flex" alignItems="flex-start" gap={2}>
        <Avatar
          sx={{
            width: 36,
            height: 36,
            bgcolor: "#10a37f",
            fontSize: 14,
            fontWeight: 600
          }}
        >
          {feedback.user?.charAt(0).toUpperCase() || "U"}
        </Avatar>

        <Box flex={1}>
          <Box
            display="flex"
            alignItems="center"
            justifyContent="space-between"
            mb={0.5}
          >
            <Typography fontWeight={600}>
              {feedback.user}
            </Typography>

            <Chip
              label={feedback.rating}
              color={ratingColor as "success" | "error" | "default"}
              size="small"
            />
          </Box>

          <Typography
            variant="body2"
            sx={{
              mb: 1,
              lineHeight: 1.5
            }}
          >
            {feedback.comment}
          </Typography>

          <Typography variant="caption" color="text.secondary">
            {formattedDate}
          </Typography>
        </Box>
      </Box>
    </Paper>
  )
}
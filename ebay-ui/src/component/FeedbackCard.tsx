import { Paper, Typography, Box, Rating, Avatar, Chip } from "@mui/material";

type Feedback = {
  user?: string;
  rating?: "positive" | "neutral" | "negative" | string;
  comment?: string;
  time?: string;
};

// -----------------------------
// Rating utilities
// -----------------------------

function normalizeRating(rating?: string) {
  if (!rating) return "neutral";

  const r = rating.toLowerCase();

  if (r.includes("positive")) return "positive";
  if (r.includes("negative")) return "negative";

  return "neutral";
}

function ratingToStars(type: string) {
  switch (type) {
    case "positive":
      return 5;
    case "neutral":
      return 3;
    case "negative":
      return 1;
    default:
      return 3;
  }
}

function ratingColor(type: string) {
  switch (type) {
    case "positive":
      return "#10a37f";
    case "neutral":
      return "#f59e0b";
    case "negative":
      return "#ef4444";
    default:
      return "#9ca3af";
  }
}

// -----------------------------
// Component
// -----------------------------

export default function FeedbackCard({
  feedback,
}: {
  feedback: Feedback;
}) {
  const type = normalizeRating(feedback.rating);
  const stars = ratingToStars(type);
  const color = ratingColor(type);

  const initial = feedback.user?.charAt(0).toUpperCase() || "U";

  const formattedDate = feedback.time
    ? new Date(feedback.time).toLocaleDateString("it-IT", {
        year: "numeric",
        month: "long",
        day: "numeric",
      })
    : "Data non disponibile";

  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        borderRadius: 3,
        border: "1px solid #e5e5e5",
        bgcolor: "#fff",
        transition: "all 0.2s",
        "&:hover": {
          borderColor: "#d1d1d1",
          boxShadow: "0 2px 8px rgba(0,0,0,0.06)",
        },
      }}
    >
      {/* HEADER */}
      <Box display="flex" alignItems="flex-start" gap={2} mb={2}>
        <Avatar
          sx={{
            width: 40,
            height: 40,
            bgcolor: color,
            fontSize: 16,
            fontWeight: 600,
          }}
        >
          {initial}
        </Avatar>

        <Box flex={1}>
          {/* User */}
          <Typography
            sx={{
              fontWeight: 600,
              fontSize: 14,
              color: "#202123",
            }}
          >
            {feedback.user || "Utente"}
          </Typography>

          {/* Rating row */}
          <Box display="flex" alignItems="center" gap={1} mt={0.3}>
            <Rating value={stars} readOnly size="small" sx={{ color }} />

            <Chip
              label={type}
              size="small"
              sx={{
                bgcolor: "#f4f4f4",
                fontSize: 11,
                textTransform: "capitalize",
              }}
            />
          </Box>

          {/* Date */}
          <Typography
            variant="caption"
            sx={{
              color: "#6e6e80",
              fontSize: 12,
              display: "block",
              mt: 0.5,
            }}
          >
            {formattedDate}
          </Typography>
        </Box>
      </Box>

      {/* COMMENT */}
      <Typography
        sx={{
          color: "#202123",
          fontSize: 15,
          lineHeight: 1.6,
        }}
      >
        {feedback.comment || "Nessun commento"}
      </Typography>
    </Paper>
  );
}
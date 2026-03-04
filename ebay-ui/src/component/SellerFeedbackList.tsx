import { Box, Typography, CircularProgress } from "@mui/material";
import FeedbackCard from "./FeedbackCard";

type Feedback = {
  user?: string;
  rating?: string;
  comment?: string;
  time?: string;
};

export default function SellerFeedbackList({
  feedbacks = [],
  loading = false,
}: {
  feedbacks?: Feedback[];
  loading?: boolean;
}) {

  if (loading) {
    return (
      <Box display="flex" alignItems="center" gap={1} mt={1}>
        <CircularProgress size={16} />
        <Typography sx={{ color: "#666", fontSize: 13 }}>
          Caricamento feedback...
        </Typography>
      </Box>
    );
  }

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
    );
  }

  return (
    <Box
      mt={1}
      sx={{
        display: "flex",
        flexDirection: "column",
        gap: 1.5,
      }}
    >
      {feedbacks.map((f, i) => (
        <FeedbackCard
          key={`${f.user}-${f.time}-${i}`}
          feedback={f}
        />
      ))}
    </Box>
  );
}
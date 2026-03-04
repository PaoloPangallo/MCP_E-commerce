import { Box, Typography } from "@mui/material";
import FeedbackCard from "./FeedbackCard";

export default function SellerFeedbackList({
  feedbacks,
  loading = false,
}: {
  feedbacks: any[];
  loading?: boolean;
}) {
  if (loading) {
    return (
      <Typography
        sx={{
          mt: 1,
          color: "#666",
          fontSize: 13,
        }}
      >
        Caricamento feedback...
      </Typography>
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
    <Box mt={1}>
      {feedbacks.map((f, i) => (
        <FeedbackCard key={i} feedback={f} />
      ))}
    </Box>
  );
}
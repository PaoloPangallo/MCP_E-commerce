import { Box, Typography } from "@mui/material";
import FeedbackCard from "./FeedbackCard";

export default function SellerFeedbackList({
  feedbacks,
}: {
  feedbacks: any[];
}) {
  if (!feedbacks || feedbacks.length === 0) {
    return (
      <Typography
        sx={{
          mt: 1,
          color: "#6e6e80",
          fontSize: 13,
        }}
      >
        Nessun feedback disponibile
      </Typography>
    );
  }

  return (
    <Box mt={2}>
      {feedbacks.map((f, i) => (
        <FeedbackCard key={i} feedback={f} />
      ))}
    </Box>
  );
}
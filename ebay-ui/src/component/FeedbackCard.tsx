import { Paper, Typography, Box, Rating, Avatar } from "@mui/material";

export default function FeedbackCard({ feedback }: { feedback: any }) {
  return (
    <Paper
      elevation={0}
      sx={{
        p: 3,
        mb: 2,
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
      <Box display="flex" alignItems="flex-start" gap={2} mb={2}>
        <Avatar
          sx={{
            width: 40,
            height: 40,
            bgcolor: "#10a37f",
            fontSize: 16,
            fontWeight: 600,
          }}
        >
          U
        </Avatar>

        <Box flex={1}>
          <Box display="flex" alignItems="center" gap={1} mb={0.5}>
            <Rating
              value={feedback.rating}
              readOnly
              size="small"
              sx={{
                color: "#10a37f",
              }}
            />
            <Typography
              sx={{
                fontWeight: 700,
                color: "#202123",
                fontSize: 14,
              }}
            >
              {feedback.rating}/5
            </Typography>
          </Box>

          <Typography
            variant="caption"
            sx={{
              color: "#6e6e80",
              fontSize: 12,
            }}
          >
            {new Date(feedback.time).toLocaleDateString("it-IT", {
              year: "numeric",
              month: "long",
              day: "numeric",
            })}
          </Typography>
        </Box>
      </Box>

      <Typography
        sx={{
          color: "#202123",
          fontSize: 15,
          lineHeight: 1.6,
        }}
      >
        {feedback.comment}
      </Typography>
    </Paper>
  );
}

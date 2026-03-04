import { Box, Typography } from "@mui/material";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";

export default function AIAnalysisCard({ text }: { text: string }) {

  if (!text) return null;

  return (
    <Box
      sx={{
        p: 2.5,
        mb: 3,
        borderRadius: "16px",
        bgcolor: "#f4f4f4",
        color: "#0d0d0d",
      }}
    >
      <Box display="flex" alignItems="center" gap={1} mb={1.5}>
        <AutoAwesomeIcon sx={{ fontSize: 18, color: "#a3a3a3" }} />
        <Typography fontWeight={600} fontSize={14} color="#666">
          Analisi in corso...
        </Typography>
      </Box>

      <Typography
        sx={{
          fontSize: 15,
          lineHeight: 1.6,
          color: "#0d0d0d",
        }}
      >
        {text}
      </Typography>
    </Box>
  );
}
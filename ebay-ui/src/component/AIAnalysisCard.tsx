import { Box, Typography, Chip } from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

interface IRMetrics {
  "precision@5"?: number
  "precision@10"?: number
  "recall@10"?: number
  "ndcg@10"?: number
}

interface Props {
  text?: string
  metrics?: IRMetrics
}

export default function AIAnalysisCard({
  text,
  metrics
}: Props) {

  if (!text) return null

  return (

    <Box
      sx={{
        p: 2.5,
        mb: 3,
        borderRadius: "16px",
        bgcolor: "#f6f6f7",
        border: "1px solid #e5e5e5"
      }}
    >

      <Box display="flex" alignItems="center" gap={1} mb={1.5}>

        <AutoAwesomeIcon
          sx={{
            fontSize: 18,
            color: "#10a37f"
          }}
        />

        <Typography fontWeight={600} fontSize={14}>
          AI Analysis
        </Typography>

        {metrics && (
          <Chip
            label="AI Ranked"
            size="small"
            sx={{
              fontSize: 11,
              bgcolor: "#e8f5f0",
              color: "#0a7a5a"
            }}
          />
        )}

      </Box>

      <Typography
        sx={{
          fontSize: 15,
          lineHeight: 1.6,
          whiteSpace: "pre-line"
        }}
      >
        {text}
      </Typography>

    </Box>

  )
}
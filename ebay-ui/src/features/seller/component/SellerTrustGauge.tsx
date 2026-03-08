import { Box, Typography } from "@mui/material"

interface Props {
  score: number
}

export default function SellerTrustGauge({ score }: Props) {

  const percent = Math.round(score * 100)

  return (
    <Box mt={1}>
      <Typography
        sx={{
          fontSize: 13,
          color: "#6b7280",
          mb: 0.5
        }}
      >
        Seller trust
      </Typography>

      <Box
        sx={{
          height: 6,
          borderRadius: 999,
          background: "#e5e7eb",
          overflow: "hidden"
        }}
      >
        <Box
          sx={{
            width: `${percent}%`,
            height: "100%",
            background: percent > 70 ? "#22c55e" : "#f59e0b"
          }}
        />
      </Box>

      <Typography
        sx={{
          fontSize: 12,
          color: "#6b7280",
          mt: 0.5
        }}
      >
        {percent}% reliability
      </Typography>
    </Box>
  )
}
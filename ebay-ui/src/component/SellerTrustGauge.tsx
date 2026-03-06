import { Box, LinearProgress, Typography } from "@mui/material"

export default function SellerTrustGauge({
  score
}: {
  score: number
}) {

  const percentage = Math.round(score * 100)

  const color =
    percentage > 85
      ? "success"
      : percentage > 70
      ? "warning"
      : "error"

  return (

    <Box mb={2}>

      <Typography variant="body2" mb={1}>
        Seller Trust {percentage}%
      </Typography>

      <LinearProgress
        variant="determinate"
        value={percentage}
        color={color as any}
        sx={{
          height: 8,
          borderRadius: 4
        }}
      />

    </Box>

  )
}
import { Box, Typography } from "@mui/material"

interface Props {
  score: number
}

export default function SellerTrustGauge({ score }: Props) {
  const percent = Math.round(score * 100)
  const good = percent >= 90
  const mid  = percent >= 70

  const barColor = good ? "#6ee7b7" : mid ? "#fcd34d" : "#fca5a5"
  const textColor = good ? "#059669" : mid ? "#d97706" : "#dc2626"

  return (
    <Box>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.4 }}>
        <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>seller trust</Typography>
        <Typography sx={{ fontSize: 11, fontWeight: 500, color: textColor }}>{percent}%</Typography>
      </Box>
      <Box sx={{ height: 4, borderRadius: 4, bgcolor: "#f3f4f6", overflow: "hidden" }}>
        <Box sx={{ width: `${percent}%`, height: "100%", bgcolor: barColor, borderRadius: 4 }} />
      </Box>
    </Box>
  )
}
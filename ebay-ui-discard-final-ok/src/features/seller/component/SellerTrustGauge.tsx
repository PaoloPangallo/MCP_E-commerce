import { Box, Typography } from "@mui/material"

interface Props {
  score: number
}

export default function SellerTrustGauge({ score }: Props) {
  const percent = Math.round(score * 100)
  const good = percent >= 90
  const mid  = percent >= 70

  const barColor = good ? "#10b981" : mid ? "#f59e0b" : "#ef4444"
  const textColor = good ? "#059669" : mid ? "#d97706" : "#dc2626"
  const label = good ? "Eccellente" : mid ? "Buona" : "Da verificare"

  return (
    <Box sx={{ width: '100%' }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-end", mb: 0.75 }}>
        <Box>
          <Typography sx={{ fontSize: 10, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.05em", mb: 0.25 }}>
            Affidabilità
          </Typography>
          <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#374151" }}>
            {label}
          </Typography>
        </Box>
        <Typography sx={{ fontSize: 14, fontWeight: 800, color: textColor, lineHeight: 1 }}>
          {percent}%
        </Typography>
      </Box>
      <Box sx={{ height: 6, borderRadius: 3, bgcolor: "#f3f4f6", overflow: "hidden", position: 'relative' }}>
        <Box 
          sx={{ 
            width: `${percent}%`, 
            height: "100%", 
            bgcolor: barColor, 
            borderRadius: 3,
            transition: 'width 1s ease-out',
            boxShadow: `0 0 8px ${barColor}44`
          }} 
        />
      </Box>
    </Box>
  )
}
import { Box, Typography } from "@mui/material"

interface Props {
  trust?: number
}

export default function SellerTrustGauge({ trust }: Props) {
  if (trust === undefined || trust === null) {
    return null
  }

  const percent = Math.max(0, Math.min(100, Math.round(trust * 100)))

  let color = "#999"
  let label = "Insufficiente"

  if (percent >= 85) {
    color = "#2e7d32"
    label = "Alta affidabilità"
  } else if (percent >= 70) {
    color = "#f9a825"
    label = "Affidabilità discreta"
  } else {
    color = "#c62828"
    label = "Rischio da verificare"
  }

  return (
    <Box sx={{ mt: 1 }}>
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={0.5}>
        <Typography sx={{ fontSize: 12, color: "#666" }}>Seller Trust</Typography>
        <Typography sx={{ fontSize: 11, color: "#777" }}>{label}</Typography>
      </Box>

      <Box
        sx={{
          width: "100%",
          height: 8,
          borderRadius: 999,
          background: "#eee",
          overflow: "hidden"
        }}
      >
        <Box
          sx={{
            width: `${percent}%`,
            height: "100%",
            background: color,
            transition: "width 0.3s"
          }}
        />
      </Box>

      <Typography sx={{ fontSize: 11, color: "#777", mt: 0.4 }}>{percent}%</Typography>
    </Box>
  )
}

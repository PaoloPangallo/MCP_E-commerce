import { Box, Typography } from "@mui/material"

interface Props {
  trust?: number
}

export default function SellerTrustGauge({ trust }: Props) {

  if (trust === undefined || trust === null) return null

  const percent = Math.round(trust * 100)

  let color = "#999"

  if (percent >= 85) color = "#2e7d32"
  else if (percent >= 70) color = "#f9a825"
  else color = "#c62828"

  return (

    <Box sx={{ mt: 1 }}>

      <Typography
        sx={{
          fontSize: 12,
          color: "#666",
          mb: 0.5
        }}
      >
        Seller Trust
      </Typography>

      <Box
        sx={{
          width: "100%",
          height: 6,
          borderRadius: 4,
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

      <Typography
        sx={{
          fontSize: 11,
          color: "#777",
          mt: 0.3
        }}
      >
        {percent}%
      </Typography>

    </Box>

  )

}
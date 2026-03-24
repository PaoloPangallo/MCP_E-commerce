import { Box, Typography } from "@mui/material"

interface Props {
  explanations?: string[]
}

export default function ExplanationChips({ explanations = [] }: Props) {
  const clean = explanations.map((s) => s?.trim()).filter(Boolean) as string[]
  if (clean.length === 0) return null

  return (
    <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
      {clean.map((item) => (
        <Box
          key={item}
          sx={{
            display: "inline-flex",
            alignItems: "center",
            px: 0.875,
            py: 0.2,
            borderRadius: "6px",
            bgcolor: "#f5f3ff",
            border: "1px solid #e0d9ff"
          }}
        >
          <Typography sx={{ fontSize: 11, color: "#6d28d9", lineHeight: 1 }}>
            {item}
          </Typography>
        </Box>
      ))}
    </Box>
  )
}
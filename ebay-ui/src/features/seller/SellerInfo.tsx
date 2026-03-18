import { Box, Typography } from "@mui/material"

interface Props {
  seller_name?: string
  seller_rating?: number
}

export default function SellerInfo({ seller_name, seller_rating }: Props) {
  if (!seller_name) return null

  // eBay seller_rating is 0–100 (percentage of positive feedback)
  const hasRating = typeof seller_rating === "number"
  const pct = hasRating ? Math.round(seller_rating) : null
  const good = pct !== null && pct >= 99
  const mid  = pct !== null && pct >= 95

  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap" }}>
      <Typography sx={{ fontSize: 12, color: "#9ca3af" }}>Venditore:</Typography>

      <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>
        {seller_name}
      </Typography>

      {pct !== null && (
        <Box
          sx={{
            display: "inline-flex",
            alignItems: "center",
            px: 0.75,
            py: 0.15,
            borderRadius: "6px",
            bgcolor: good ? "#f0fdf4" : mid ? "#fefce8" : "#f9fafb",
            border: "1px solid",
            borderColor: good ? "#bbf7d0" : mid ? "#fef08a" : "#e5e7eb"
          }}
        >
          <Typography
            sx={{
              fontSize: 11,
              fontWeight: 500,
              color: good ? "#15803d" : mid ? "#854d0e" : "#6b7280"
            }}
          >
            {pct}%
          </Typography>
        </Box>
      )}
    </Box>
  )
}
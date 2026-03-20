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
  const good = pct !== null && pct >= 98
  const mid  = pct !== null && pct >= 95

  return (
    <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, flexWrap: "wrap" }}>
      <Typography sx={{ fontSize: 11, fontWeight: 500, color: "#9ca3af", textTransform: 'uppercase', letterSpacing: '0.02em' }}>
        Seller:
      </Typography>

      <Typography sx={{ fontSize: 13, fontWeight: 600, color: "#111827" }}>
        {seller_name}
      </Typography>

      {pct !== null && (
        <Box
          sx={{
            display: "inline-flex",
            alignItems: "center",
            px: 1,
            py: "1px",
            borderRadius: "5px",
            bgcolor: good ? "#f0fdf4" : mid ? "#fffbeb" : "#f9fafb",
            border: "1px solid",
            borderColor: good ? "#bbf7d0" : mid ? "#fef3c7" : "#e5e7eb"
          }}
        >
          <Typography
            sx={{
              fontSize: 11,
              fontWeight: 700,
              color: good ? "#16a34a" : mid ? "#d97706" : "#6b7280"
            }}
          >
            {pct}% Positivo
          </Typography>
        </Box>
      )}
    </Box>
  )
}
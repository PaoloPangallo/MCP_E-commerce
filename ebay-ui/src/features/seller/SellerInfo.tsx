import { Box, Rating, Typography } from "@mui/material"

interface Props {
  seller_name?: string
  seller_rating?: number
}

function normalizeRating(sellerRating?: number) {
  if (typeof sellerRating !== "number") {
    return null
  }

  const value = Math.max(0, Math.min(5, sellerRating / 20))
  return value
}

export default function SellerInfo({ seller_name, seller_rating }: Props) {
  const ratingStars = normalizeRating(seller_rating)

  return (
    <Box display="flex" alignItems="center" gap={1} mb={1} flexWrap="wrap">
      <Typography sx={{ color: "#666", fontSize: 14 }}>Venditore:</Typography>

      <Typography sx={{ fontWeight: 600, fontSize: 14, color: "#202123" }}>
        {seller_name || "N/A"}
      </Typography>

      {ratingStars !== null && (
        <>
          <Rating value={ratingStars} precision={0.1} size="small" readOnly />
          <Typography variant="caption" sx={{ color: "#666" }}>
            ({seller_rating?.toFixed?.(1) ?? seller_rating}%)
          </Typography>
        </>
      )}
    </Box>
  )
}

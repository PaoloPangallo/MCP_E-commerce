import { Paper, Typography, Box, Chip, Rating } from "@mui/material";
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser";

export default function SearchResultCard({ item }: { item: any }) {
  const trustPercent =
    item.trust_score !== undefined
      ? Math.round(item.trust_score * 100)
      : null;

  const trustColor =
    trustPercent === null
      ? "#ececf1"
      : trustPercent >= 80
      ? "#d4f4dd"
      : trustPercent >= 60
      ? "#fff3cd"
      : "#f8d7da";

  const trustTextColor =
    trustPercent === null
      ? "#6e6e80"
      : trustPercent >= 80
      ? "#0d8c6b"
      : trustPercent >= 60
      ? "#856404"
      : "#842029";

  return (
    <Paper
      elevation={0}
      onClick={() => item.url && window.open(item.url, "_blank")}
      sx={{
        p: 3,
        mb: 2,
        borderRadius: 3,
        border: "1px solid #e5e5e5",
        display: "flex",
        gap: 3,
        transition: "all 0.2s",
        cursor: "pointer",
        bgcolor: "#fff",
        "&:hover": {
          borderColor: "#10a37f",
          boxShadow: "0 4px 12px rgba(0,0,0,0.08)",
          transform: "translateY(-2px)",
        },
      }}
    >
      {/* Image */}
      <Box
        sx={{
          width: 120,
          height: 120,
          borderRadius: 2,
          overflow: "hidden",
          flexShrink: 0,
          bgcolor: "#f7f7f8",
        }}
      >
        <img
          src={item.image_url || "https://via.placeholder.com/120"}
          alt={item.title}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      </Box>

      {/* Content */}
      <Box flex={1}>
        {/* Already in DB */}
        {item._already_in_db && (
          <Chip
            label="Già visto"
            size="small"
            sx={{
              mb: 1,
              bgcolor: "#f1f1f1",
              fontSize: 11,
            }}
          />
        )}

        {/* Title */}
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            fontSize: 18,
            color: "#202123",
            mb: 1,
            lineHeight: 1.4,
          }}
        >
          {item.title}
        </Typography>

        {/* Price */}
        <Typography
          sx={{
            fontSize: 24,
            fontWeight: 700,
            color: "#10a37f",
            mb: 2,
          }}
        >
          {item.price} {item.currency}
        </Typography>

        {/* Seller Info */}
        <Box display="flex" alignItems="center" gap={1} mb={1}>
          <Typography
            variant="body2"
            sx={{
              color: "#6e6e80",
              fontSize: 14,
            }}
          >
            Venditore:
          </Typography>

          <Typography
            variant="body2"
            sx={{
              fontWeight: 600,
              color: "#202123",
              fontSize: 14,
            }}
          >
            {item.seller_name || "N/A"}
          </Typography>

          <Rating
            value={(item.seller_rating ?? 0) / 20}
            precision={0.1}
            size="small"
            readOnly
            sx={{ ml: 0.5 }}
          />

          <Typography
            variant="caption"
            sx={{
              color: "#6e6e80",
              fontSize: 13,
            }}
          >
            ({item.seller_rating ?? "N/A"}%)
          </Typography>
        </Box>

        {/* Trust Score */}
        {trustPercent !== null && (
          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              icon={<VerifiedUserIcon sx={{ fontSize: 16 }} />}
              label={`Trust Score: ${trustPercent}%`}
              size="small"
              sx={{
                bgcolor: trustColor,
                color: trustTextColor,
                fontWeight: 600,
                fontSize: 12,
                border: "none",
              }}
            />
          </Box>
        )}
      </Box>
    </Paper>
  );
}
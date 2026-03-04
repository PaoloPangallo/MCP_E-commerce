import { Paper, Typography, Box, Chip, Rating } from "@mui/material";
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser";

interface SearchItem {
  title?: string;
  price?: number;
  currency?: string;
  url?: string;
  image_url?: string;
  seller_name?: string;
  seller_rating?: number;
  trust_score?: number;
  _already_in_db?: boolean;
}

function getTrustInfo(score?: number) {
  if (score === undefined || score === null) {
    return {
      percent: null,
      bg: "#ececf1",
      text: "#6e6e80",
    };
  }

  const percent = Math.round(score * 100);

  if (percent >= 80) {
    return { percent, bg: "#d4f4dd", text: "#0d8c6b" };
  }

  if (percent >= 60) {
    return { percent, bg: "#fff3cd", text: "#856404" };
  }

  return { percent, bg: "#f8d7da", text: "#842029" };
}

export default function SearchResultCard({ item }: { item: SearchItem }) {
  const trust = getTrustInfo(item.trust_score);

  const openItem = () => {
    if (item.url) {
      window.open(item.url, "_blank");
    }
  };

  return (
    <Paper
      elevation={0}
      onClick={openItem}
      sx={{
        p: 3,
        mb: 2,
        borderRadius: 3,
        border: "1px solid #e5e5e5",
        display: "flex",
        gap: 3,
        transition: "all 0.2s",
        cursor: item.url ? "pointer" : "default",
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
          alt={item.title || "product"}
          style={{
            width: "100%",
            height: "100%",
            objectFit: "cover",
          }}
        />
      </Box>

      {/* Content */}
      <Box flex={1}>

        {/* Already seen */}
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
          sx={{
            fontWeight: 600,
            fontSize: 18,
            color: "#202123",
            mb: 1,
            lineHeight: 1.4,
          }}
        >
          {item.title || "Prodotto"}
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
          {item.price ?? "—"} {item.currency ?? ""}
        </Typography>

        {/* Seller */}
        <Box display="flex" alignItems="center" gap={1} mb={1} flexWrap="wrap">
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
        {trust.percent !== null && (
          <Box display="flex" alignItems="center" gap={1}>
            <Chip
              icon={<VerifiedUserIcon sx={{ fontSize: 16 }} />}
              label={`Trust Score: ${trust.percent}%`}
              size="small"
              sx={{
                bgcolor: trust.bg,
                color: trust.text,
                fontWeight: 600,
                fontSize: 12,
              }}
            />
          </Box>
        )}
      </Box>
    </Paper>
  );
}
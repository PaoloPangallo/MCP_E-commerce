import { Paper, Typography, Box, Chip, Rating, Button } from "@mui/material";
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser";
import { useState } from "react";
import SellerFeedbackList from "./SellerFeedbackList";

export default function SearchResultCard({ item }: { item: any }) {

  const [feedbacks, setFeedbacks] = useState<any[]>([]);
  const [open, setOpen] = useState(false);

  const trustPercent =
    item?.trust_score !== undefined
      ? Math.round(item.trust_score * 100)
      : null;

  const loadFeedback = async () => {

    try {

      const res = await fetch(
        `http://localhost:8000/seller/${item.seller_name}/feedback`
      );

      const data = await res.json();

      setFeedbacks(data.feedbacks || []);
      setOpen(!open);

    } catch (err) {
      console.error("Errore feedback venditore");
    }
  };

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
        },
      }}
    >
      {/* IMAGE */}
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

      {/* CONTENT */}
      <Box flex={1}>

        {/* TITLE */}
        <Typography
          variant="h6"
          sx={{
            fontWeight: 600,
            fontSize: 18,
            color: "#202123",
            mb: 1,
          }}
        >
          {item.title}
        </Typography>

        {/* PRICE */}
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

        {/* SELLER */}
        <Box display="flex" alignItems="center" gap={1} mb={1}>

          <Typography
            sx={{
              color: "#6e6e80",
              fontSize: 14,
            }}
          >
            Venditore:
          </Typography>

          <Typography
            sx={{
              fontWeight: 600,
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
          />

          <Typography
            variant="caption"
            sx={{
              color: "#6e6e80",
            }}
          >
            ({item.seller_rating ?? "N/A"}%)
          </Typography>

        </Box>

        {/* TRUST SCORE */}
        {trustPercent !== null && (
          <Chip
            icon={<VerifiedUserIcon />}
            label={`Trust Score: ${trustPercent}%`}
            size="small"
            sx={{
              bgcolor: trustPercent > 80 ? "#d4f4dd" : "#fff3cd",
              color: trustPercent > 80 ? "#0d8c6b" : "#856404",
              fontWeight: 600,
              fontSize: 12,
            }}
          />
        )}

        {/* FEEDBACK BUTTON */}
        <Box mt={1}>
          <Button
            size="small"
            onClick={(e) => {
              e.stopPropagation();
              loadFeedback();
            }}
          >
            {open ? "Nascondi feedback" : "Mostra feedback venditore"}
          </Button>
        </Box>

        {/* FEEDBACK LIST */}
        {open && <SellerFeedbackList feedbacks={feedbacks} />}

      </Box>
    </Paper>
  );
}
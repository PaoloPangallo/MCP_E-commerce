import { Paper, Typography, Box, Chip, Rating, Button } from "@mui/material";
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser";
import { useState } from "react";
import SellerFeedbackList from "./SellerFeedbackList";

export default function SearchResultCard({ item }: { item: any }) {

  const [feedbacks, setFeedbacks] = useState<any[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const trustPercent =
    item?.trust_score !== undefined
      ? Math.round(item.trust_score * 100)
      : null;

  const loadFeedback = async () => {

    // se già caricati, solo toggle
    if (feedbacks.length > 0) {
      setOpen(!open);
      return;
    }

    try {

      setLoading(true);

      const res = await fetch(
        `http://localhost:8000/seller/${item.seller_name}/feedback`
      );

      const data = await res.json();

      setFeedbacks(data.feedbacks || []);
      setOpen(true);

    } catch (err) {
      console.error("Errore feedback venditore");
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper
      elevation={0}
      onClick={() => item.url && window.open(item.url, "_blank")}
      sx={{
        p: 3,
        mb: 2,
        borderRadius: "16px",
        border: "1px solid #e5e5e5",
        display: "flex",
        gap: 3,
        transition: "all 0.2s",
        cursor: "pointer",
        bgcolor: "#fff",
        "&:hover": {
          borderColor: "#a3a3a3",
          boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
        },
      }}
    >

      {/* IMAGE */}
      <Box
        sx={{
          width: 120,
          height: 120,
          borderRadius: "8px",
          overflow: "hidden",
          flexShrink: 0,
          bgcolor: "#f4f4f4",
          border: "1px solid #e5e5e5",
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
            color: "#0d0d0d",
            mb: 1,
            lineHeight: 1.4,
          }}
        >
          {item.title}
        </Typography>

        {/* PRICE */}
        <Typography
          sx={{
            fontSize: 24,
            fontWeight: 700,
            color: "#0d0d0d",
            mb: 2,
          }}
        >
          {item.price} {item.currency}
        </Typography>

        {/* SELLER */}
        <Box display="flex" alignItems="center" gap={1} mb={1}>

          <Typography
            sx={{
              color: "#666",
              fontSize: 14,
            }}
          >
            Venditore:
          </Typography>

          <Typography
            sx={{
              fontWeight: 500,
              fontSize: 14,
              color: "#0d0d0d",
            }}
          >
            {item.seller_name || "N/A"}
          </Typography>

          <Rating
            value={(item.seller_rating ?? 0) / 20}
            precision={0.1}
            size="small"
            readOnly
            sx={{ color: "#0d0d0d" }}
          />

          <Typography
            variant="caption"
            sx={{
              color: "#666",
            }}
          >
            ({item.seller_rating ?? "N/A"}%)
          </Typography>

        </Box>

        {/* TRUST + AI SCORE */}
        <Box display="flex" alignItems="center" gap={2} mb={1}>

          {trustPercent !== null && (
            <Chip
              icon={<VerifiedUserIcon sx={{ fontSize: 16 }} />}
              label={`Trust Score: ${trustPercent}%`}
              size="small"
              sx={{
                bgcolor: trustPercent > 80 ? "#f4f4f4" : "#fff3cd",
                color: trustPercent > 80 ? "#0d0d0d" : "#856404",
                fontWeight: 500,
                fontSize: 12,
                border: trustPercent > 80 ? "1px solid #e5e5e5" : "none",
              }}
              onClick={(e) => e.stopPropagation()}
            />
          )}

          {item._rerank_score && (
            <Typography
              variant="caption"
              sx={{
                color: "#666",
                fontWeight: 500
              }}
              onClick={(e) => e.stopPropagation()}
            >
              AI Match: {(item._rerank_score * 100).toFixed(0)}%
            </Typography>
          )}

        </Box>

        {/* FEEDBACK BUTTON */}
        <Box mt={2}>
          <Button
            size="small"
            variant="outlined"
            onClick={(e) => {
              e.stopPropagation();
              loadFeedback();
            }}
            sx={{
              textTransform: "none",
              borderRadius: "16px",
              borderColor: "#e5e5e5",
              color: "#0d0d0d",
              "&:hover": {
                borderColor: "#a3a3a3",
                bgcolor: "#f4f4f4",
              },
            }}
          >
            {open ? "Nascondi feedback" : "Mostra feedback venditore"}
          </Button>
        </Box>

        {/* FEEDBACK LIST */}
        {open && (
          <Box mt={2} sx={{ p: 2, bgcolor: "#f4f4f4", borderRadius: "8px" }}>
            <SellerFeedbackList feedbacks={feedbacks} loading={loading} />
          </Box>
        )}

      </Box>

    </Paper>
  );
}
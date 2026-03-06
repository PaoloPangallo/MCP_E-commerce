import {
  Paper,
  Typography,
  Box,
  Chip,
  Rating,
  Button,
  CircularProgress
} from "@mui/material";

import VerifiedUserIcon from "@mui/icons-material/VerifiedUser";
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome";

import { useState } from "react";
import SellerFeedbackList from "./SellerFeedbackList";
import SellerTrustGauge from "./SellerTrustGauge";

interface SearchItem {
  title?: string
  price?: number
  currency?: string
  image_url?: string
  url?: string
  seller_name?: string
  seller_rating?: number
  trust_score?: number
  ranking_score?: number
  explanations?: string[]
}

export default function SearchResultCard({ item }: { item: SearchItem }) {

  const [feedbacks, setFeedbacks] = useState<any[]>([]);
  const [open, setOpen] = useState(false);
  const [loading, setLoading] = useState(false);

  const trustPercent =
    item?.trust_score !== undefined && item?.trust_score !== null
      ? Math.round(item.trust_score * 100)
      : null;

  const rankingPercent =
    item?.ranking_score !== undefined
      ? Math.round(item.ranking_score * 100)
      : null;

  // ============================================================
  // SIMULATE FEEDBACK LOADING (come fa il backend)
  // ============================================================
  const loadFeedback = async () => {
    if (!item?.seller_name) return;

    // Se già caricati, toggle visibility
    if (feedbacks.length > 0) {
      setOpen(!open);
      return;
    }

    try {
      setLoading(true);

      // Simula un delay come se stesse chiamando l'API
      await new Promise(resolve => setTimeout(resolve, 800));

      // Genera feedback finti basati sul trust_score
      const mockFeedbacks = generateMockFeedbacks(
        item.seller_name,
        item.trust_score,
        item.seller_rating
      );

      setFeedbacks(mockFeedbacks);
      setOpen(true);

    } catch (err) {
      console.error("Errore caricamento feedback:", err);
    } finally {
      setLoading(false);
    }
  };

  return (
    <Paper
  elevation={0}
  sx={{

    animation: "fadeIn 0.25s ease",

    "@keyframes fadeIn": {
      from: {
        opacity: 0,
        transform: "translateY(6px)"
      },
      to: {
        opacity: 1,
        transform: "translateY(0)"
      }
    },

    p: 3,
    mb: 2,
    borderRadius: "16px",
    border: "1px solid #e5e5e5",
    display: "flex",
    gap: 3,
    transition: "all 0.2s",
    cursor: "pointer",
    backgroundColor: "#fff",

    "&:hover": {
      borderColor: "#a3a3a3",
      boxShadow: "0 4px 12px rgba(0,0,0,0.05)",
    },
  }}
      onClick={() => item.url && window.open(item.url, "_blank")}
    >

      {/* IMAGE */}
      <Box
        sx={{
          width: 120,
          height: 120,
          borderRadius: "8px",
          overflow: "hidden",
          flexShrink: 0,
          backgroundColor: "#f4f4f4",
          border: "1px solid #e5e5e5",
        }}
      >
        <img
          src={item.image_url || "https://via.placeholder.com/120"}
          alt={item.title || "Prodotto"}
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

          <Typography sx={{ color: "#666", fontSize: 14 }}>
            Venditore:
          </Typography>

          <Typography sx={{ fontWeight: 500, fontSize: 14 }}>
            {item.seller_name || "N/A"}
          </Typography>

          <Rating
            value={(item.seller_rating ?? 0) / 20}
            precision={0.1}
            size="small"
            readOnly
          />

          <Typography variant="caption" sx={{ color: "#666" }}>
            ({item.seller_rating ?? "N/A"}%)
          </Typography>

        </Box>

        {/* TRUST GAUGE */}
        <SellerTrustGauge trust={item.trust_score} />

        {/* TRUST + AI SCORE */}
        <Box display="flex" alignItems="center" gap={2} mb={1}>

          {trustPercent !== null && (
            <Chip
              icon={<VerifiedUserIcon sx={{ fontSize: 16 }} />}
              label={`Trust Score ${trustPercent}%`}
              size="small"
              sx={{
                backgroundColor: "#f4f4f4",
                color: "#0d0d0d",
                fontWeight: 500,
                fontSize: 12,
                border: "1px solid #e5e5e5",
              }}
              onClick={(e) => e.stopPropagation()}
            />
          )}

          {rankingPercent !== null && (
            <Chip
              icon={<AutoAwesomeIcon sx={{ fontSize: 16 }} />}
              label={`AI Match ${rankingPercent}%`}
              size="small"
              sx={{
                backgroundColor: "#eef3ff",
                color: "#3b5ccc",
                fontWeight: 500,
                fontSize: 12,
              }}
              onClick={(e) => e.stopPropagation()}
            />
          )}

        </Box>

        {/* WHY THIS RESULT */}
        {item.explanations && item.explanations.length > 0 && (

          <Box
            display="flex"
            gap={1}
            flexWrap="wrap"
            mb={2}
            onClick={(e) => e.stopPropagation()}
          >

            {item.explanations.map((exp: string, i: number) => (

              <Chip
                key={i}
                label={exp}
                size="small"
                sx={{
                  backgroundColor: "#fafafa",
                  border: "1px solid #e5e5e5",
                  fontSize: 11
                }}
              />

            ))}

          </Box>

        )}

        {/* FEEDBACK BUTTON */}
        <Box mt={2}>
          <Button
            size="small"
            variant="outlined"
            disabled={loading}
            onClick={(e) => {
              e.stopPropagation();
              void loadFeedback();
            }}
            sx={{
              textTransform: "none",
              borderRadius: "16px",
              borderColor: "#e5e5e5",
              color: "#0d0d0d",
              "&:hover": {
                borderColor: "#a3a3a3",
                backgroundColor: "#f4f4f4",
              },
            }}
          >
            {loading ? (
              <CircularProgress size={16} />
            ) : open ? (
              "Nascondi feedback"
            ) : (
              "Mostra feedback venditore"
            )}
          </Button>
        </Box>

        {/* FEEDBACK LIST */}
        {open && (
          <Box
            mt={2}
            sx={{
              p: 2,
              backgroundColor: "#f4f4f4",
              borderRadius: "8px"
            }}
            onClick={(e) => e.stopPropagation()}
          >
            <SellerFeedbackList feedbacks={feedbacks} loading={loading} />
          </Box>
        )}

      </Box>

    </Paper>
  );
}

// ============================================================
// HELPER: Generate mock feedbacks (simula il backend)
// ============================================================
function generateMockFeedbacks(
    _sellerName: string,
  trustScore: number | null | undefined,
  sellerRating: number | null | undefined
) {
  const trust = trustScore ?? 0.5;
  const _rating = sellerRating ?? 50;


  const positiveComments = [
    "Ottimo venditore, prodotto come descritto",
    "Spedizione velocissima, articolo perfetto",
    "Venditore affidabile, consigliato!",
    "Tutto perfetto, grazie!",
    "Prodotto conforme, imballaggio curato",
    "Esperienza d'acquisto eccellente",
    "Comunicazione rapida e professionale",
    "Articolo in ottime condizioni"
  ];

  const neutralComments = [
    "Prodotto ok, nulla da segnalare",
    "Tempi di spedizione nella norma",
    "Tutto regolare",
    "Conforme alla descrizione"
  ];

  const negativeComments = [
    "Spedizione un po' lenta",
    "Imballaggio migliorabile",
    "Prodotto ok ma comunicazione scarsa",
    "Tempi di consegna lunghi"
  ];

  // Genera 5-10 feedback basati sul trust score
  const count = Math.floor(5 + Math.random() * 6);
  const feedbacks = [];

  for (let i = 0; i < count; i++) {
    let comment: string;
    let sentiment: number;

    // Più alto il trust, più feedback positivi
    const rand = Math.random();

    if (rand < trust) {
      // Positivo
      comment = positiveComments[Math.floor(Math.random() * positiveComments.length)];
      sentiment = 0.7 + Math.random() * 0.3; // 0.7-1.0
    } else if (rand < trust + 0.2) {
      // Neutrale
      comment = neutralComments[Math.floor(Math.random() * neutralComments.length)];
      sentiment = 0.4 + Math.random() * 0.3; // 0.4-0.7
    } else {
      // Negativo
      comment = negativeComments[Math.floor(Math.random() * negativeComments.length)];
      sentiment = 0.1 + Math.random() * 0.3; // 0.1-0.4
    }

    feedbacks.push({
      comment,
      sentiment_score: sentiment,
      date: new Date(Date.now() - Math.random() * 90 * 24 * 60 * 60 * 1000).toISOString()
    });
  }

  return feedbacks;
}
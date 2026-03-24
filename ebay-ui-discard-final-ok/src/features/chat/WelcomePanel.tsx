import { Box, Typography, Grid, Paper } from "@mui/material"
import TrendingUpIcon from "@mui/icons-material/TrendingUp"
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import CompareArrowsIcon from "@mui/icons-material/CompareArrows"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

const actionCards = [
  {
    title: "Analisi di Mercato",
    description: "Scopri i prezzi medi e i trend di interesse su Google per qualsiasi prodotto.",
    icon: <TrendingUpIcon sx={{ color: "#10b981" }} />,
    prompt: "Quali sono i trend di mercato per le AirPods Pro 2?",
    color: "#10b981"
  },
  {
    title: "Shopping Intelligente",
    description: "Trova le migliori offerte su eBay filtrando per prezzo, condizione e altro.",
    icon: <ShoppingCartIcon sx={{ color: "#3b82f6" }} />,
    prompt: "Cerca un iPhone 15 Pro Max nuovo sotto i 1000€",
    color: "#3b82f6"
  },
  {
    title: "Affidabilità Venditore",
    description: "Analizziamo i feedback e la reputazione dei seller per acquisti sicuri.",
    icon: <VerifiedUserIcon sx={{ color: "#f59e0b" }} />,
    prompt: "Controlla l'affidabilità del venditore pegaso_italia",
    color: "#f59e0b"
  },
  {
    title: "Confronto Prodotti",
    description: "Compara istantaneamente più modelli per trovare quello più conveniente.",
    icon: <CompareArrowsIcon sx={{ color: "#8b5cf6" }} />,
    prompt: "Confronta iPhone 15 Pro, Samsung S24 e Google Pixel 8",
    color: "#8b5cf6"
  }
]

export default function WelcomePanel() {
  const dispatch = (prompt: string) => {
    window.dispatchEvent(new CustomEvent("send-chat", { detail: prompt }))
  }

  return (
    <Box
      sx={{
        py: { xs: 4, md: 8 },
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center"
      }}
    >
      {/* Hero Section */}
      <Box sx={{ mb: 6, maxWidth: 640 }}>
        <Box
          sx={{
            display: "inline-flex",
            p: 1.5,
            borderRadius: "16px",
            background: "linear-gradient(135deg, #111827 0%, #374151 100%)",
            color: "#fff",
            mb: 3,
            boxShadow: "0 4px 12px rgba(0,0,0,0.1)"
          }}
        >
          <AutoAwesomeIcon sx={{ fontSize: 28 }} />
        </Box>
        
        <Typography
          variant="h3"
          sx={{
            fontSize: { xs: 32, md: 42 },
            fontWeight: 850,
            background: "linear-gradient(90deg, #111827 0%, #4b5563 100%)",
            WebkitBackgroundClip: "text",
            WebkitTextFillColor: "transparent",
            letterSpacing: "-0.03em",
            lineHeight: 1.1,
            mb: 2
          }}
        >
          Porta la tua ricerca su eBay al livello successivo
        </Typography>

        <Typography
          variant="body1"
          sx={{
            color: "#6b7280",
            fontSize: { xs: 16, md: 18 },
            lineHeight: 1.6,
            px: { xs: 2, md: 0 }
          }}
        >
          ebayGPT è il tuo assistente personale che analizza dati in tempo reale, 
          confronta prezzi e verifica venditori per te.
        </Typography>
      </Box>

      {/* Action Cards Grid */}
      <Grid container spacing={2.5} sx={{ maxWidth: 800 }}>
        {actionCards.map((card, i) => (
          <Grid size={{ xs: 12, sm: 6 }} key={i}>
            <Paper
              component="button"
              onClick={() => dispatch(card.prompt)}
              elevation={0}
              sx={{
                width: "100%",
                p: 3,
                height: "100%",
                display: "flex",
                flexDirection: "column",
                alignItems: "flex-start",
                textAlign: "left",
                border: "1px solid #e5e7eb",
                borderRadius: 4,
                bgcolor: "#ffffff",
                cursor: "pointer",
                transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                fontFamily: "inherit",
                "&:hover": {
                  transform: "translateY(-4px)",
                  boxShadow: "0 12px 24px -8px rgba(0,0,0,0.08)",
                  borderColor: card.color,
                  "& .icon-box": {
                    bgcolor: `${card.color}15`,
                    transform: "scale(1.1)"
                  }
                },
                "&:active": {
                  transform: "translateY(-1px)"
                }
              }}
            >
              <Box
                className="icon-box"
                sx={{
                  p: 1.25,
                  borderRadius: "12px",
                  bgcolor: "#f9fafb",
                  mb: 2,
                  display: "flex",
                  transition: "all 0.2s ease"
                }}
              >
                {card.icon}
              </Box>
              <Typography variant="subtitle1" sx={{ fontWeight: 700, color: "#111827", mb: 0.5 }}>
                {card.title}
              </Typography>
              <Typography variant="body2" sx={{ color: "#6b7280", mb: 0 }}>
                {card.description}
              </Typography>
            </Paper>
          </Grid>
        ))}
      </Grid>
      
      <Typography sx={{ mt: 6, fontSize: 13, color: "#9ca3af", fontWeight: 500 }}>
        Scrivi un messaggio qui sotto per iniziare una conversazione libera
      </Typography>
    </Box>
  )
}
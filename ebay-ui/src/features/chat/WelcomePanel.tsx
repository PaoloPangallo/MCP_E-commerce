import { Box, Typography, Grid, Paper } from "@mui/material"
import TrendingUpIcon from "@mui/icons-material/TrendingUp"
import ShoppingCartIcon from "@mui/icons-material/ShoppingCart"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import CompareArrowsIcon from "@mui/icons-material/CompareArrows"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

const actionCards = [
  {
    title: "Analisi di Mercato",
    description: "Scopri i prezzi medi e i trend di interesse.",
    icon: <TrendingUpIcon sx={{ color: "#10b981" }} />,
    prompt: "Quali sono i trend di mercato per le AirPods Pro 2?",
    color: "#10b981"
  },
  {
    title: "Shopping Intelligente",
    description: "Trova le migliori offerte su eBay.",
    icon: <ShoppingCartIcon sx={{ color: "#3b82f6" }} />,
    prompt: "Cerca un iPhone 15 Pro Max nuovo sotto i 1000€",
    color: "#3b82f6"
  },
  {
    title: "Affidabilità Venditore",
    description: "Analizziamo i feedback dei seller.",
    icon: <VerifiedUserIcon sx={{ color: "#f59e0b" }} />,
    prompt: "Controlla l'affidabilità del venditore pegaso_italia",
    color: "#f59e0b"
  },
  {
    title: "Confronto Prodotti",
    description: "Compara istantaneamente più modelli.",
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
        py: { xs: 3, md: 5 },
        display: "flex",
        flexDirection: "column",
        alignItems: "center",
        textAlign: "center"
      }}
    >
      {/* Hero Section */}
      <Box sx={{ mb: 4, maxWidth: 850 }}>
        <Box
          sx={{
            display: "inline-flex",
            p: 1.25,
            borderRadius: "12px",
            background: "linear-gradient(135deg, #111827 0%, #374151 100%)",
            color: "#fff",
            mb: 2.5,
            boxShadow: "0 4px 12px rgba(0,0,0,0.2)"
          }}
        >
          <AutoAwesomeIcon sx={{ fontSize: 22 }} />
        </Box>
        
        <Typography
          variant="h4"
          sx={{
            fontSize: { xs: 24, md: 32 },
            fontWeight: 800,
            color: "var(--text-primary)",
            letterSpacing: "-0.015em",
            lineHeight: 1.2,
            mb: 1,
            whiteSpace: { md: "nowrap" }
          }}
        >
          Porta la tua ricerca su eBay al livello successivo
        </Typography>

        <Typography
          variant="body2"
          sx={{
            color: "var(--text-secondary)",
            fontSize: { xs: 13, md: 15 },
            lineHeight: 1.5,
            px: { xs: 2, md: 2 },
            opacity: 0.9,
            whiteSpace: { md: "nowrap" }
          }}
        >
          ebayGPT analizza i dati in tempo reale, confronta i prezzi e verifica i venditori per te.
        </Typography>
      </Box>

      {/* Action Cards Grid */}
      <Grid container spacing={2} sx={{ maxWidth: 740 }}>
        {actionCards.map((card, i) => (
          <Grid size={{ xs: 12, sm: 6 }} key={i}>
            <Paper
              component="button"
              onClick={() => dispatch(card.prompt)}
              elevation={0}
              sx={{
                width: "100%",
                p: 2,
                height: "100%",
                display: "flex",
                flexDirection: "row",
                alignItems: "center",
                textAlign: "left",
                gap: 2,
                border: "1px solid var(--border-color)",
                borderRadius: "16px",
                bgcolor: "var(--bg-primary)",
                color: "var(--text-primary)",
                cursor: "pointer",
                transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                fontFamily: "inherit",
                "&:hover": {
                  transform: "translateY(-3px)",
                  boxShadow: "0 10px 20px -10px rgba(0,0,0,0.2)",
                  borderColor: card.color,
                  "& .icon-box": {
                    bgcolor: `${card.color}15`,
                    transform: "scale(1.05)"
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
                  p: 1.1,
                  borderRadius: "10px",
                  bgcolor: "var(--bg-secondary)",
                  display: "flex",
                  transition: "all 0.2s ease"
                }}
              >
                {card.icon}
              </Box>
              <Box>
                <Typography sx={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)", mb: 0.25 }}>
                  {card.title}
                </Typography>
                <Typography variant="caption" sx={{ color: "var(--text-secondary)", display: "block", lineHeight: 1.3 }}>
                  {card.description}
                </Typography>
              </Box>
            </Paper>
          </Grid>
        ))}
      </Grid>
      
      <Typography sx={{ mt: 5, fontSize: 12, color: "var(--text-secondary)", fontWeight: 500, letterSpacing: '0.02em', opacity: 0.7 }}>
        Inizia scrivendo un messaggio qui sotto
      </Typography>
    </Box>
  )
}
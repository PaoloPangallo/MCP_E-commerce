import { 
  Box, 
  Typography, 
  Paper, 
  Chip, 
  Button, 
  Avatar, 
  Collapse,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  IconButton
} from "@mui/material"
import SellIcon from "@mui/icons-material/Sell"
import TrendingDownIcon from "@mui/icons-material/TrendingDown"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import FlashOnIcon from "@mui/icons-material/FlashOn"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import KeyboardArrowUpIcon from "@mui/icons-material/KeyboardArrowUp"
import { useState } from "react"
import type { DealsData } from "../types"

interface DealsDisplayProps {
  data: DealsData
}

export const EBAY_CATEGORIES = [
  { id: "9355", name: "Smartphone", icon: "📱", desc: "Ultimi modelli e accessori" },
  { id: "111422", name: "Informatica", icon: "💻", desc: "Laptop, Tablet e Componenti" },
  { id: "1249", name: "Console & Giochi", icon: "🎮", desc: "PS5, Xbox, Switch e titoli" },
  { id: "15032", name: "Orologi", icon: "⌚", desc: "Brand di lusso e sportivi" },
  { id: "11450", name: "Moda", icon: "👕", desc: "Abbigliamento e scarpe" },
  { id: "11700", name: "Fotografia", icon: "📷", desc: "Mirrorless e obiettivi" },
  { id: "20081", name: "Beauty", icon: "✨", desc: "Cura persona e profumi" },
  { id: "159912", name: "Fai da te", icon: "🛠️", desc: "Utensili e domotica" },
]

export default function DealsDisplay({ data }: DealsDisplayProps) {
  const { title, subtitle, deals } = data
  const [showTable, setShowTable] = useState(false)

  const handleCategoryClick = (category: typeof EBAY_CATEGORIES[0]) => {
    const event = new CustomEvent("send-chat", {
      detail: `Cerca offerte eBay per la categoria ${category.name} (ID: ${category.id}) 🏷️`
    })
    window.dispatchEvent(event)
    setShowTable(false)
  }

  const hasDeals = deals && deals.length > 0

  return (
    <Box sx={{ my: 4 }}>
      {/* Header Section with Gradient Background */}
      <Box 
        sx={{ 
          mb: 4, 
          p: 3, 
          background: "var(--bg-secondary)",
          color: "var(--text-primary)",
          border: "1px solid var(--border-color)",
          borderRadius: "16px",
          position: "relative",
          overflow: "hidden"
        }}
      >
        <Box sx={{ position: "relative", zIndex: 1 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 2, mb: 1 }}>
            <Avatar sx={{ bgcolor: "var(--bg-primary)", width: 40, height: 40, border: "1px solid var(--border-color)" }}>
              <FlashOnIcon sx={{ color: "var(--text-secondary)" }} />
            </Avatar>
            <Typography variant="h5" sx={{ fontWeight: 900, letterSpacing: "-0.02em" }}>
              {title || "Offerte eBay del Giorno"}
            </Typography>
          </Box>
          <Typography variant="body1" sx={{ color: "var(--text-secondary)", fontWeight: 500, ml: 7.2 }}>
            {subtitle || "Esplora i migliori sconti selezionati per te."}
          </Typography>
        </Box>
        
        {/* Background Decor */}
        <Box sx={{ 
          position: "absolute", 
          right: -20, 
          top: -20, 
          opacity: 0.1, 
          transform: "rotate(15deg)" 
        }}>
          <SellIcon sx={{ fontSize: 180 }} />
        </Box>
      </Box>

      {/* Interactive Category Selector (Top Bar) */}
      <Box sx={{ mb: 4 }}>
        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2 }}>
          <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "var(--text-secondary)", display: "flex", alignItems: "center", gap: 1 }}>
            <TrendingDownIcon sx={{ fontSize: 18, color: "var(--danger)" }} />
            {hasDeals ? "Filtra per categoria:" : "Scegli una categoria per iniziare:"}
          </Typography>
          <Button 
            size="small" 
            startIcon={showTable ? <KeyboardArrowUpIcon /> : <KeyboardArrowDownIcon />}
            onClick={() => setShowTable(!showTable)}
            sx={{ fontWeight: 700, color: "var(--text-secondary)", textTransform: "none" }}
          >
            {showTable ? "Mostra chip" : "Tutte le categorie"}
          </Button>
        </Box>

        <Collapse in={!showTable}>
          <Box 
            sx={{ 
              display: "flex", 
              gap: 1.25, 
              overflowX: "auto", 
              pb: 2,
              "&::-webkit-scrollbar": { height: 4 },
              "&::-webkit-scrollbar-thumb": { bgcolor: "var(--border-color)", borderRadius: 4 }
            }}
          >
            {EBAY_CATEGORIES.map((cat) => (
              <Chip
                key={cat.id}
                label={`${cat.icon} ${cat.name}`}
                onClick={() => handleCategoryClick(cat)}
                sx={{
                  px: 1,
                  py: 2.2,
                  flexShrink: 0,
                  borderRadius: 3,
                  border: "1px solid var(--border-color)",
                  bgcolor: "var(--bg-primary)",
                  fontWeight: 700,
                  color: "var(--text-primary)",
                  fontSize: 13,
                  transition: "all 0.2s",
                  "&:hover": {
                    bgcolor: "var(--bg-secondary)",
                    borderColor: "var(--accent-primary)",
                    color: "var(--accent-primary)",
                    transform: "translateY(-2px)"
                  }
                }}
              />
            ))}
          </Box>
        </Collapse>

        <Collapse in={showTable}>
          <TableContainer component={Paper} elevation={0} sx={{ border: "1px solid var(--border-color)", borderRadius: "16px", mb: 0, bgcolor: "var(--bg-primary)" }}>
            <Table size="small">
              <TableHead>
                <TableRow sx={{ bgcolor: "var(--bg-secondary)" }}>
                  <TableCell sx={{ fontWeight: 800, fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase" }}>Categoria</TableCell>
                  <TableCell sx={{ fontWeight: 800, fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase" }}>Descrizione</TableCell>
                  <TableCell align="right" sx={{ fontWeight: 800, fontSize: 11, color: "var(--text-secondary)", textTransform: "uppercase" }}>Azione</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {EBAY_CATEGORIES.map((cat) => (
                  <TableRow 
                    key={cat.id} 
                    hover 
                    onClick={() => handleCategoryClick(cat)}
                    sx={{ cursor: "pointer", "&:last-child td, &:last-child th": { border: 0 } }}
                  >
                    <TableCell sx={{ py: 1.5 }}>
                      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
                        <Typography sx={{ fontSize: 20 }}>{cat.icon}</Typography>
                        <Typography sx={{ fontWeight: 700, color: "var(--text-primary)", fontSize: 14 }}>{cat.name}</Typography>
                      </Box>
                    </TableCell>
                    <TableCell sx={{ color: "var(--text-secondary)", fontSize: 13 }}>{cat.desc}</TableCell>
                    <TableCell align="right">
                      <IconButton size="small" sx={{ bgcolor: "var(--bg-secondary)", color: "var(--text-primary)" }}>
                        <FlashOnIcon sx={{ fontSize: 16 }} />
                      </IconButton>
                    </TableCell>
                  </TableRow>
                ))}
              </TableBody>
            </Table>
          </TableContainer>
        </Collapse>
      </Box>

      {/* Grid of Deals Error or Placeholder */}
      {!hasDeals && (
        <Paper
          elevation={0}
          sx={{
            p: 6,
            textAlign: "center",
            border: "2px dashed var(--border-color)",
            bgcolor: "var(--bg-secondary)"
          }}
        >
          <Avatar sx={{ bgcolor: "#f1f5f9", width: 64, height: 64, mx: "auto", mb: 2 }}>
            <FlashOnIcon sx={{ color: "#94a3b8", fontSize: 32 }} />
          </Avatar>
          <Typography variant="h6" sx={{ fontWeight: 800, color: "#475569", mb: 1 }}>
            Nessuna offerta caricata
          </Typography>
          <Typography variant="body2" sx={{ color: "#64748b", maxWidth: 300, mx: "auto" }}>
            Scegli una delle categorie qui sopra per visualizzare le offerte a tempo limitato di oggi.
          </Typography>
        </Paper>
      )}

      {/* Grid of Deals using Box Grid */}
      {hasDeals && (
        <Box 
          sx={{ 
            display: "grid",
            gridTemplateColumns: {
              xs: "1fr",
              sm: "1fr 1fr",
              md: "1fr 1fr 1fr"
            },
            gap: 3
          }}
        >
          {deals.map((deal, idx) => (
            <Paper
              key={idx}
              elevation={0}
              sx={{
                height: "100%",
                display: "flex",
                flexDirection: "column",
                border: "1px solid var(--border-color)",
                bgcolor: "var(--bg-primary)",
                position: "relative",
                transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                "&:hover": {
                  transform: "translateY(-4px)",
                  borderColor: "var(--text-secondary)",
                  "& .deal-img": { transform: "scale(1.05)" },
                  "& .view-btn": { bgcolor: "var(--text-secondary)", color: "var(--bg-primary)" }
                },
              }}
            >
              {/* Hot Badge */}
              <Box
                sx={{
                  position: "absolute",
                  top: 16,
                  left: 16,
                  zIndex: 2,
                  bgcolor: "var(--text-primary)",
                  color: "var(--bg-primary)",
                  px: 1.5,
                  py: 0.5,
                  borderRadius: 2,
                  display: "flex",
                  alignItems: "center",
                  gap: 0.5
                }}
              >
                <TrendingDownIcon sx={{ fontSize: 14 }} />
                <Typography sx={{ fontSize: 10, fontWeight: 900, textTransform: "uppercase" }}>
                  Sconto Top
                </Typography>
              </Box>

              {/* Image Container */}
              <Box
                sx={{
                  p: 2,
                  bgcolor: "var(--bg-secondary)",
                  borderTopLeftRadius: 16,
                  borderTopRightRadius: 16,
                  overflow: "hidden",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  height: 200
                }}
              >
                {deal.thumbnail ? (
                  <Box
                    className="deal-img"
                    component="img"
                    src={deal.thumbnail}
                    alt={deal.title}
                    sx={{ 
                      maxWidth: "100%", 
                      maxHeight: "100%", 
                      objectFit: "contain",
                      transition: "transform 0.5s ease" 
                    }}
                  />
                ) : (
                  <Box sx={{ color: "var(--text-secondary)", textAlign: "center" }}>
                    <SellIcon sx={{ fontSize: 40, mb: 1 }} />
                    <Typography variant="caption">Nessuna immagine</Typography>
                  </Box>
                )}
              </Box>

              {/* Content */}
              <Box sx={{ p: 2.5, flexGrow: 1, display: "flex", flexDirection: "column" }}>
                <Typography
                  sx={{
                    fontSize: 15,
                    fontWeight: 700,
                    color: "var(--text-primary)",
                    mb: 1.5,
                    display: "-webkit-box",
                    WebkitLineClamp: 2,
                    WebkitBoxOrient: "vertical",
                    overflow: "hidden",
                    minHeight: "2.6em",
                    lineHeight: 1.3
                  }}
                >
                  {deal.title}
                </Typography>

                <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5, mb: 2 }}>
                  {deal.old_price?.discount && (
                    <Chip
                      label={deal.old_price.discount}
                      size="small"
                      sx={{ 
                        bgcolor: "#fff1f2", 
                        color: "#e11d48", 
                        fontWeight: 900, 
                        fontSize: 10,
                        border: "1px solid #fecdd3"
                      }}
                    />
                  )}
                  {deal.extensions?.slice(0, 1).map((ext, i) => (
                    <Chip
                      key={i}
                      label={ext}
                      size="small"
                      variant="outlined"
                      sx={{ fontSize: 10, color: "var(--text-secondary)", borderColor: "var(--border-color)" }}
                    />
                  ))}
                </Box>

                <Box sx={{ mt: "auto", mb: 3 }}>
                  <Box sx={{ display: "flex", alignItems: "baseline", gap: 1 }}>
                    <Typography sx={{ fontSize: 24, fontWeight: 900, color: "var(--text-primary)" }}>
                      {deal.price.raw}
                    </Typography>
                    {deal.old_price && (
                      <Typography
                        sx={{
                          fontSize: 14,
                          color: "var(--text-secondary)",
                          textDecoration: "line-through",
                          fontWeight: 500,
                        }}
                      >
                        {deal.old_price.raw}
                      </Typography>
                    )}
                  </Box>
                </Box>

                <Button
                  className="view-btn"
                  component="a"
                  href={deal.link}
                  target="_blank"
                  fullWidth
                  variant="outlined"
                  endIcon={<OpenInNewIcon sx={{ fontSize: 16 }} />}
                  sx={{
                    textTransform: "none",
                    fontWeight: 800,
                    borderRadius: 3,
                    py: 1.2,
                    borderColor: "var(--border-color)",
                    color: "var(--text-secondary)",
                    transition: "all 0.2s ease"
                  }}
                >
                  Acquista Ora
                </Button>
              </Box>
            </Paper>
          ))}
        </Box>
      )}
    </Box>
  )
}

import {
  Box,
  Link,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Chip,
  Button,
} from "@mui/material"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import LocalShippingIcon from "@mui/icons-material/LocalShipping"
import WorkspacePremiumIcon from "@mui/icons-material/WorkspacePremium"
import TrendingUpIcon from "@mui/icons-material/TrendingUp"
import SpeedIcon from "@mui/icons-material/Speed"

import type { ComparisonData } from "../types"

interface ComparisonDisplayProps {
  data: ComparisonData
}

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") return "—"
  return `${price} ${currency ?? ""}`.trim()
}

function formatDate(dateString?: string) {
  if (!dateString) return null
  try {
    const d = new Date(dateString)
    return d.toLocaleDateString("it-IT", { day: "2-digit", month: "short" })
  } catch (e) {
    return dateString
  }
}

interface ScoreItemProps {
  label: string
  score: number
  color: string
  icon?: React.ReactNode
}

function ScoreItem({ label, score, color, icon }: ScoreItemProps) {
  const pct = Math.round(score * 100)
  return (
    <Box sx={{ flex: 1 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          {icon}
          <Typography sx={{ fontSize: 10, color: "#6b7280", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.02em" }}>
            {label}
          </Typography>
        </Box>
        <Typography sx={{ fontSize: 10, fontWeight: 700, color: color }}>{pct}%</Typography>
      </Box>
      <Box
        sx={{
          width: "100%",
          height: 6,
          bgcolor: "#f3f4f6",
          borderRadius: 4,
          overflow: "hidden",
          position: "relative"
        }}
      >
        <Box
          sx={{
            width: `${pct}%`,
            height: "100%",
            bgcolor: color,
            borderRadius: 4,
            transition: "width 1s ease-out"
          }}
        />
      </Box>
    </Box>
  )
}

export default function ComparisonDisplay({ data }: ComparisonDisplayProps) {
  const { winner, comparison_matrix, winner_reason } = data

  const minPrice = Math.min(...comparison_matrix.map((c) => c.price ?? Infinity))
  const maxOverall = Math.max(...comparison_matrix.map((c) => (c.scores as any)?.overall ?? 0))
  const maxValue = Math.max(...comparison_matrix.map((c) => (c.scores as any)?.value ?? 0))

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 4, py: 1 }}>
      {/* 🔹 WINNER HERO SECTION */}
      <Box
        sx={{
          position: "relative",
          borderRadius: 4,
          background: "linear-gradient(135deg, #ffffff 0%, #f9fafb 100%)",
          border: "1px solid #e2e8f0",
          boxShadow: "0 10px 25px -5px rgba(0,0,0,0.05), 0 8px 10px -6px rgba(0,0,0,0.05)",
          overflow: "hidden",
        }}
      >
        {/* Banner "AI Picking" */}
        <Box
          sx={{
            bgcolor: "#7c3aed",
            color: "#fff",
            px: 2.5,
            py: 1,
            display: "flex",
            alignItems: "center",
            gap: 1,
          }}
        >
          <WorkspacePremiumIcon sx={{ fontSize: 18 }} />
          <Typography sx={{ fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
            Miglior Scelta dell'Agente
          </Typography>
        </Box>

        <Box sx={{ p: { xs: 2.5, sm: 4 }, display: "grid", gridTemplateColumns: { xs: "1fr", md: "180px 1fr" }, gap: 4 }}>
          {/* Image */}
          <Box
            sx={{
              width: "100%",
              height: 180,
              bgcolor: "#fff",
              borderRadius: 3,
              border: "1px solid #f1f5f9",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              p: 1.5,
              position: "relative"
            }}
          >
            {winner.image_url ? (
              <Box
                component="img"
                src={winner.image_url}
                alt={winner.title}
                sx={{ width: "100%", height: "100%", objectFit: "contain" }}
              />
            ) : (
              <Typography sx={{ color: "#cbd5e1" }}>No Image</Typography>
            )}
            {winner.shipping?.free && (
              <Chip 
                label="Free Delivery" 
                size="small" 
                color="success" 
                icon={<LocalShippingIcon sx={{ fontSize: "14px !important" }} />}
                sx={{ position: "absolute", bottom: -10, left: "50%", transform: "translateX(-50%)", fontWeight: 700, fontSize: 10 }} 
              />
            )}
          </Box>

          <Box sx={{ display: "flex", flexDirection: "column" }}>
            <Typography variant="h6" sx={{ color: "#1e293b", fontWeight: 700, lineHeight: 1.3, mb: 1.5 }}>
              {winner.title}
            </Typography>
            
            <Box sx={{ display: "flex", alignItems: "baseline", gap: 1.5, mb: 2 }}>
              <Typography sx={{ fontSize: 28, fontWeight: 800, color: "#111827" }}>
                {formatPrice(winner.price, winner.currency)}
              </Typography>
              {winner.price === minPrice && (
                <Chip label="Miglior Prezzo" size="small" variant="outlined" color="success" sx={{ height: 20, fontSize: 10, fontWeight: 700 }} />
              )}
            </Box>

            <Box
              sx={{
                p: 2,
                borderRadius: 2,
                bgcolor: "#f5f3ff",
                borderLeft: "4px solid #7c3aed",
                mb: 3
              }}
            >
              <Typography sx={{ fontSize: 13, color: "#5b21b6", lineHeight: 1.6, fontWeight: 500 }}>
                {winner_reason}
              </Typography>
            </Box>

            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, mb: 3 }}>
              <ScoreItem label="Match Affinità" score={winner.scores?.overall || 0} color="#7c3aed" icon={<TrendingUpIcon sx={{ fontSize: 14 }} />} />
              <ScoreItem label="Value Score" score={(winner as any).value_score || 0} color="#10b981" icon={<WorkspacePremiumIcon sx={{ fontSize: 14 }} />} />
              <ScoreItem label="Prezzo" score={winner.scores?.price || 0} color="#0ea5e9" icon={<SpeedIcon sx={{ fontSize: 14 }} />} />
            </Box>

            <Box sx={{ display: "flex", gap: 2 }}>
              {winner.url && (
                <Button
                  component="a"
                  href={winner.url}
                  target="_blank"
                  variant="contained"
                  fullWidth
                  sx={{
                    bgcolor: "#111827",
                    color: "#fff",
                    textTransform: "none",
                    fontWeight: 700,
                    borderRadius: "10px",
                    py: 1,
                    "&:hover": { bgcolor: "#374151" }
                  }}
                >
                  Acquista ora su eBay
                </Button>
              )}
            </Box>
          </Box>
        </Box>
      </Box>

      {/* 🔹 CANDIDATES GRID */}
      <Box>
        <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", mb: 2, px: 1 }}>
          Tutte le Alternative
        </Typography>
        
        <Box
          sx={{
            display: "grid",
            gridTemplateColumns: {
              xs: "1fr",
              sm: "repeat(auto-fit, minmax(240px, 1fr))"
            },
            gap: 2.5
          }}
        >
          {comparison_matrix.map((item, idx) => {
            const isWinner = item.title === winner.title
            return (
              <Box
                key={idx}
                sx={{
                  bgcolor: "#fff",
                  border: "1px solid",
                  borderColor: isWinner ? "#7c3aed" : "#f1f5f9",
                  borderRadius: 3,
                  p: 2,
                  display: "flex",
                  flexDirection: "column",
                  transition: "all 0.2s ease",
                  "&:hover": {
                    transform: "translateY(-4px)",
                    boxShadow: "0 12px 20px -8px rgba(0,0,0,0.1)",
                    borderColor: "#7c3aed"
                  }
                }}
              >
                <Box sx={{ position: "relative", mb: 1.5 }}>
                   <Box
                      sx={{
                        width: "100%",
                        aspectRatio: "4/3",
                        bgcolor: "#f9fafb",
                        borderRadius: 2,
                        display: "flex",
                        alignItems: "center",
                        justifyContent: "center",
                        p: 1
                      }}
                    >
                      {item.image_url ? (
                        <Box component="img" src={item.image_url} sx={{ width: "100%", height: "100%", objectFit: "contain" }} />
                      ) : (
                        <Typography variant="caption" color="text.disabled">No Img</Typography>
                      )}
                    </Box>
                    {isWinner && (
                      <Chip 
                        label="PICK" 
                        size="small" 
                        sx={{ position: "absolute", top: 8, left: 8, bgcolor: "#7c3aed", color: "#fff", fontSize: 9, fontWeight: 800, height: 18 }} 
                      />
                    )}
                </Box>

                <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#1e293b", mb: 0.5, lineHeight: 1.3, display: "-webkit-box", WebkitLineClamp: 2, WebkitBoxOrient: "vertical", overflow: "hidden", minHeight: "2.6em" }}>
                  {item.title}
                </Typography>

                <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
                  <Typography sx={{ fontSize: 17, fontWeight: 800, color: "#111827" }}>
                    {formatPrice(item.price, item.currency)}
                  </Typography>
                  {item.price === minPrice && (
                    <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#16a34a" }}>TOP DEAL</Typography>
                  )}
                </Box>

                <Box sx={{ display: "flex", flexDirection: "column", gap: 1, mb: 2 }}>
                   <ScoreItem label="Match" score={(item.scores as any).overall} color="#7c3aed" />
                   <ScoreItem label="Valore" score={(item.scores as any).value || 0} color="#10b981" />
                </Box>

                {item.shipping && (
                  <Box sx={{ display: "flex", alignItems: "center", gap: 0.75, mb: 2, bgcolor: "#f1f5f9", p: 1, borderRadius: 1.5 }}>
                    <LocalShippingIcon sx={{ fontSize: 14, color: "#64748b" }} />
                    <Box>
                      <Typography sx={{ fontSize: 10, fontWeight: 700, color: item.shipping.free ? "#16a34a" : "#475569" }}>
                        {item.shipping.free ? "Consegna GRATIS" : `Sped. ${item.shipping.cost} ${item.shipping.currency}`}
                      </Typography>
                      {item.shipping.max_delivery && (
                        <Typography sx={{ fontSize: 9, color: "#94a3b8" }}>
                          Entro il {formatDate(item.shipping.max_delivery)}
                        </Typography>
                      )}
                    </Box>
                  </Box>
                )}

                <Link
                  href={item.url}
                  target="_blank"
                  underline="none"
                  sx={{
                    mt: "auto",
                    display: "flex",
                    alignItems: "center",
                    justifyContent: "center",
                    gap: 0.5,
                    fontSize: 12,
                    fontWeight: 600,
                    color: "#6b7280",
                    py: 1,
                    border: "1px solid #e2e8f0",
                    borderRadius: 2,
                    "&:hover": { bgcolor: "#f9fafb", color: "#111827", borderColor: "#7c3aed" }
                  }}
                >
                  Dettagli eBay <OpenInNewIcon sx={{ fontSize: 12 }} />
                </Link>
              </Box>
            )
          })}
        </Box>
      </Box>

      {/* 🔹 COMPARISON TABLE */}
      <Box>
        <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#64748b", textTransform: "uppercase", letterSpacing: "0.1em", mb: 2, px: 1 }}>
          Tabella Comparativa
        </Typography>

        <TableContainer sx={{ borderRadius: 3, border: "1px solid #e2e8f0", bgcolor: "#fff", overflow: "hidden" }}>
          <Table size="small">
            <TableHead>
              <TableRow sx={{ bgcolor: "#f8fafc" }}>
                <TableCell sx={{ color: "#64748b", fontWeight: 700, fontSize: 11, py: 2 }}>FEATURE</TableCell>
                {comparison_matrix.map((item, i) => (
                  <TableCell key={i} align="center" sx={{ fontWeight: 800, color: item.title === winner.title ? "#7c3aed" : "#1e293b", fontSize: 11 }}>
                    {item.title?.slice(0, 15)}...
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow>
                <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "#475569" }}>PREZZO</TableCell>
                {comparison_matrix.map((item, i) => (
                  <TableCell key={i} align="center" sx={{ fontSize: 13, fontWeight: 700 }}>
                    {formatPrice(item.price, item.currency)}
                  </TableCell>
                ))}
              </TableRow>
              <TableRow sx={{ bgcolor: "#fcfcfc" }}>
                <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "#475569" }}>VALORE / PREZZO</TableCell>
                {comparison_matrix.map((item, i) => (
                  <TableCell key={i} align="center">
                    <Typography sx={{ fontWeight: 700, fontSize: 12, color: (item.scores as any).value === maxValue ? "#10b981" : "#475569" }}>
                      {Math.round(((item.scores as any).value || 0) * 100)}%
                    </Typography>
                  </TableCell>
                ))}
              </TableRow>
              <TableRow>
                <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "#475569" }}>SPEDIZIONE</TableCell>
                {comparison_matrix.map((item, i) => (
                  <TableCell key={i} align="center">
                    <Typography sx={{ fontSize: 11, fontWeight: 600, color: item.shipping?.free ? "#22c55e" : "#1e293b" }}>
                      {item.shipping?.free ? "GRATIS" : item.shipping?.cost ? `${item.shipping.cost} €` : "—"}
                    </Typography>
                  </TableCell>
                ))}
              </TableRow>
              <TableRow sx={{ bgcolor: "#fcfcfc" }}>
                <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "#475569" }}>AFFIDABILITÀ</TableCell>
                {comparison_matrix.map((item, i) => (
                  <TableCell key={i} align="center">
                    <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.5 }}>
                      <VerifiedUserIcon sx={{ fontSize: 12, color: (item.trust_score || 0) > 0.9 ? "#10b981" : "#cbd5e1" }} />
                      <Typography sx={{ fontSize: 12, fontWeight: 700 }}>{Math.round((item.trust_score || 0) * 100)}%</Typography>
                    </Box>
                  </TableCell>
                ))}
              </TableRow>
              <TableRow>
                <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "#475569", border: "none" }}>AI SCORE</TableCell>
                {comparison_matrix.map((item, i) => (
                  <TableCell key={i} align="center" sx={{ border: "none" }}>
                    <Typography sx={{ fontSize: 13, fontWeight: 800, color: (item.scores as any).overall === maxOverall ? "#7c3aed" : "#94a3b8" }}>
                       {Math.round(((item.scores as any).overall || 0) * 100)}%
                    </Typography>
                  </TableCell>
                ))}
              </TableRow>
            </TableBody>
          </Table>
        </TableContainer>
      </Box>
    </Box>
  )
}
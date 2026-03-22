import {
  Box,
  Typography,
  Paper,
  Chip,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Divider,
} from "@mui/material"
import LocalShippingIcon from "@mui/icons-material/LocalShipping"
import LocationOnIcon from "@mui/icons-material/LocationOn"
import CheckCircleIcon from "@mui/icons-material/CheckCircle"
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined"
import EventAvailableIcon from "@mui/icons-material/EventAvailable"
import type { ShippingCostsData } from "../search/types"

interface ShippingCostsCardProps {
  data: ShippingCostsData
}

function formatDate(dateString?: string) {
  if (!dateString) return "N/D"
  try {
    const d = new Date(dateString)
    return d.toLocaleDateString("it-IT", { day: "2-digit", month: "short" })
  } catch (e) {
    return dateString
  }
}

export default function ShippingCostsCard({ data }: ShippingCostsCardProps) {
  const {
    shipping_options,
    item_location,
    free_shipping_available,
    cheapest_option,
  } = data as any
  const options = Array.isArray(shipping_options) ? shipping_options : []

  return (
    <Paper
      elevation={0}
      sx={{
        width: "100%",
        borderRadius: 4,
        border: "1px solid #e1e7ef",
        bgcolor: "#ffffff",
        overflow: "hidden",
        boxShadow: "0 10px 15px -3px rgba(0, 0, 0, 0.05)",
      }}
    >
      {/* 🔹 HEADER & SUMMARY */}
      <Box sx={{ p: 3, bgcolor: "#f8fafc", borderBottom: "1px solid #e2e8f0" }}>
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "flex-start", mb: 2 }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
            <Box
              sx={{
                width: 42,
                height: 42,
                borderRadius: "12px",
                bgcolor: "#2563eb",
                color: "#fff",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
              }}
            >
              <LocalShippingIcon />
            </Box>
            <Box>
              <Typography variant="subtitle1" fontWeight={800} color="#1e293b">
                Logistica Dettagliata
              </Typography>
              <Typography variant="caption" color="#64748b" fontWeight={600} sx={{ textTransform: "uppercase", letterSpacing: "0.05em" }}>
                Prezzi e Tempi di Consegna stimati
              </Typography>
            </Box>
          </Box>
          
          {free_shipping_available && (
            <Chip
              label="FREE DELIVERY"
              color="success"
              size="small"
              icon={<CheckCircleIcon sx={{ fontSize: "14px !important" }} />}
              sx={{ fontWeight: 800, fontSize: 10, borderRadius: "6px" }}
            />
          )}
        </Box>

        {item_location && (
          <Box sx={{ display: "flex", alignItems: "center", gap: 1, mt: 2 }}>
            <LocationOnIcon sx={{ fontSize: 16, color: "#64748b" }} />
            <Typography variant="body2" sx={{ color: "#475569", fontWeight: 500 }}>
              Oggetto spedito da: <Typography component="span" fontWeight={700} color="#1e293b">
                {item_location.city ? `${item_location.city}, ` : ""}{item_location.country || "Estero"}
              </Typography>
            </Typography>
          </Box>
        )}
      </Box>

      {/* 🔹 TABLE OF OPTIONS */}
      <Box sx={{ p: 0 }}>
        {options.length === 0 ? (
          <Box sx={{ p: 3, display: "flex", alignItems: "center", gap: 2, color: "#ef4444" }}>
            <InfoOutlinedIcon />
            <Typography variant="body2" fontWeight={600}>Nessuna opzione di spedizione calcolata.</Typography>
          </Box>
        ) : (
          <TableContainer>
            <Table size="medium">
              <TableHead sx={{ bgcolor: "#f1f5f9" }}>
                <TableRow>
                  <TableCell sx={{ fontSize: 11, fontWeight: 700, color: "#64748b", textTransform: "uppercase" }}>Servizio</TableCell>
                  <TableCell sx={{ fontSize: 11, fontWeight: 700, color: "#64748b", textTransform: "uppercase" }} align="center">Consegna</TableCell>
                  <TableCell sx={{ fontSize: 11, fontWeight: 700, color: "#64748b", textTransform: "uppercase" }} align="right">Costo</TableCell>
                </TableRow>
              </TableHead>
              <TableBody>
                {options.map((opt, idx) => {
                  const costValue = Number(opt.shippingCost?.value || 0)
                  const isFree = costValue === 0
                  const minDate = formatDate(opt.minEstimatedDeliveryDate)
                  const maxDate = formatDate(opt.maxEstimatedDeliveryDate)
                  const isCheapest = cheapest_option && costValue === cheapest_option.cost

                  return (
                    <TableRow 
                      key={idx} 
                      sx={{ 
                        "&:last-child td, &:last-child th": { border: 0 },
                        bgcolor: isCheapest ? "#f0f9ff" : "inherit"
                      }}
                    >
                      <TableCell>
                        <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#1e293b" }}>
                          {opt.shippingServiceCode?.replace(/_/g, " ") || "Standard"}
                        </Typography>
                        <Typography sx={{ fontSize: 10, color: "#94a3b8", fontWeight: 500 }}>
                          {opt.shippingCostType || "Tariffa fissa"}
                        </Typography>
                      </TableCell>
                      
                      <TableCell align="center">
                        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center" }}>
                          <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
                            <EventAvailableIcon sx={{ fontSize: 14, color: "#10b981" }} />
                            <Typography sx={{ fontSize: 12, fontWeight: 600, color: "#334155" }}>
                              {minDate === maxDate ? maxDate : `${minDate} - ${maxDate}`}
                            </Typography>
                          </Box>
                          {idx === 0 && (
                            <Typography sx={{ fontSize: 9, color: "#64748b", fontWeight: 700, mt: 0.25 }}>
                              STIMA PRIORITARIA
                            </Typography>
                          )}
                        </Box>
                      </TableCell>

                      <TableCell align="right">
                        <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
                          <Typography 
                            sx={{ 
                              fontSize: 14, 
                              fontWeight: 800, 
                              color: isFree ? "#10b981" : "#1e293b" 
                            }}
                          >
                            {isFree ? "GRATIS" : `${costValue} ${opt.shippingCost?.currency || "€"}`}
                          </Typography>
                          {isCheapest && !isFree && (
                            <Chip 
                              label="BEST PRICE" 
                              size="small" 
                              sx={{ height: 16, fontSize: 8, fontWeight: 800, bgcolor: "#2563eb", color: "#fff", mt: 0.5 }} 
                            />
                          )}
                        </Box>
                      </TableCell>
                    </TableRow>
                  )
                })}
              </TableBody>
            </Table>
          </TableContainer>
        )}
      </Box>

      {/* 🔹 FOOTER INFO */}
      <Box sx={{ p: 2, bgcolor: "#f8fafc", borderTop: "1px solid #e2e8f0" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
          <InfoOutlinedIcon sx={{ fontSize: 14, color: "#94a3b8" }} />
          <Typography sx={{ fontSize: 10, color: "#94a3b8", fontWeight: 500 }}>
            I tempi di consegna sono stime fornite da eBay e dipendono dalla data di ricezione del pagamento.
          </Typography>
        </Box>
      </Box>
    </Paper>
  )
}

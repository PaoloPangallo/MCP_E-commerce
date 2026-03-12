import { Box, Typography, Paper, Chip, Divider } from "@mui/material"
import LocalShippingOutlinedIcon from "@mui/icons-material/LocalShippingOutlined"
import LocationOnOutlinedIcon from "@mui/icons-material/LocationOnOutlined"
import CalendarMonthOutlinedIcon from "@mui/icons-material/CalendarMonthOutlined"
import type { ShippingCostsData } from "../search/types"

interface ShippingCostsCardProps {
    data: ShippingCostsData
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

export default function ShippingCostsCard({ data }: ShippingCostsCardProps) {
    const { shipping_options, item_location } = data
    const options = Array.isArray(shipping_options) ? shipping_options : []

    return (
        <Paper
            elevation={0}
            sx={{
                width: "100%",
                borderRadius: 4,
                border: "1px solid #e5e7eb",
                bgcolor: "#ffffff",
                overflow: "hidden",
                mt: 1,
                mb: 2
            }}
        >
            <Box p={2.5}>
                <Box display="flex" alignItems="center" gap={1.5} mb={2}>
                    <Box
                        sx={{
                            display: "flex",
                            alignItems: "center",
                            justifyContent: "center",
                            width: 36,
                            height: 36,
                            borderRadius: "50%",
                            bgcolor: "#eef2f7",
                            color: "#3b82f6"
                        }}
                    >
                        <LocalShippingOutlinedIcon fontSize="small" />
                    </Box>
                    <Typography variant="subtitle1" fontWeight={600} color="text.primary">
                        Opzioni di Spedizione
                    </Typography>
                </Box>

                {item_location && (
                    <Box display="flex" alignItems="center" gap={1} mb={2.5}>
                        <LocationOnOutlinedIcon fontSize="small" color="action" />
                        <Typography variant="body2" color="text.secondary">
                            Oggetto si trova a:{" "}
                            <Typography component="span" fontWeight={500} color="text.primary">
                                {item_location.city ? `${item_location.city}, ` : ""}
                                {item_location.country || "Sconosciuto"}
                            </Typography>
                        </Typography>
                    </Box>
                )}

                {options.length === 0 ? (
                    <Typography variant="body2" color="text.secondary">
                        Nessuna opzione di spedizione disponibile o errore nel calcolo.
                    </Typography>
                ) : (
                    <Box display="flex" flexDirection="column" gap={1.5}>
                        {options.map((opt, idx) => {
                            const costValue = Number(opt.shippingCost?.value || 0)
                            const isFree = costValue === 0
                            const costStr = isFree
                                ? "Gratuita"
                                : `${opt.shippingCost?.value} ${opt.shippingCost?.currency || "EUR"}`

                            const minDate = formatDate(opt.minEstimatedDeliveryDate)
                            const maxDate = formatDate(opt.maxEstimatedDeliveryDate)
                            let deliveryStr = ""
                            if (minDate && maxDate && minDate !== maxDate) {
                                deliveryStr = `${minDate} - ${maxDate}`
                            } else if (maxDate) {
                                deliveryStr = `Entro il ${maxDate}`
                            } else {
                                deliveryStr = "Tempistiche non disponibili"
                            }

                            return (
                                <Box key={idx}>
                                    {idx > 0 && <Divider sx={{ my: 1.5 }} />}
                                    <Box display="flex" justifyContent="space-between" alignItems="flex-start">
                                        <Box>
                                            <Typography variant="body2" fontWeight={600} color="text.primary" mb={0.5}>
                                                {opt.shippingServiceCode || "Spedizione Standard"}
                                            </Typography>
                                            <Box display="flex" alignItems="center" gap={0.5}>
                                                <CalendarMonthOutlinedIcon sx={{ fontSize: 14, color: "#6b7280" }} />
                                                <Typography variant="caption" color="text.secondary">
                                                    {deliveryStr}
                                                </Typography>
                                            </Box>
                                        </Box>

                                        <Chip
                                            size="small"
                                            label={costStr}
                                            sx={{
                                                fontWeight: 600,
                                                bgcolor: isFree ? "#dcfce7" : "#f3f4f6",
                                                color: isFree ? "#166534" : "#111827",
                                                borderRadius: "8px"
                                            }}
                                        />
                                    </Box>
                                </Box>
                            )
                        })}
                    </Box>
                )}
            </Box>
        </Paper>
    )
}

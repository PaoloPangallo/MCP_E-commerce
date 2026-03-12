import {
    Box,
    Typography,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Divider,
    Chip,
    Paper
} from "@mui/material"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined"
import VerifiedUserOutlinedIcon from "@mui/icons-material/VerifiedUserOutlined"
import type { ItemDetailsData } from "../search/types"

interface ItemDetailsCardProps {
    data: ItemDetailsData
}

export default function ItemDetailsCard({ data }: ItemDetailsCardProps) {
    const {
        title,
        condition,
        price,
        short_description,
        description,
        item_specifics,
        return_terms,
        brand,
        color
    } = data

    const priceValue = price?.value ? `${price.value} ${price.currency || "EUR"}` : "Non disponibile"

    // eBay localizedAspects format: [{ name: "Brand", value: "Nike" }, ...]
    const specifics = Array.isArray(item_specifics) ? item_specifics : []

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
                <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={1.5}>
                    <Typography variant="subtitle1" fontWeight={600} sx={{ color: "#111827", lineHeight: 1.3 }}>
                        {title || "Dettagli Oggetto"}
                    </Typography>
                </Box>

                <Box display="flex" flexWrap="wrap" gap={1} mb={2}>
                    <Chip label={priceValue} size="small" sx={{ fontWeight: 600, bgcolor: "#f3f4f6", color: "#111827" }} />
                    {condition && <Chip label={condition} size="small" variant="outlined" />}
                    {brand && <Chip label={`Brand: ${brand}`} size="small" variant="outlined" />}
                    {color && <Chip label={`Colore: ${color}`} size="small" variant="outlined" />}
                </Box>

                {short_description && (
                    <Typography variant="body2" color="text.secondary" mb={2}>
                        {short_description}
                    </Typography>
                )}

                <Divider sx={{ my: 2 }} />

                {specifics.length > 0 && (
                    <Accordion
                        elevation={0}
                        disableGutters
                        sx={{
                            "&:before": { display: "none" },
                            bgcolor: "transparent",
                            border: "1px solid #e5e7eb",
                            borderRadius: "12px !important",
                            mb: 1
                        }}
                    >
                        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 48 }}>
                            <Box display="flex" alignItems="center" gap={1}>
                                <InfoOutlinedIcon fontSize="small" color="action" />
                                <Typography variant="body2" fontWeight={500}>
                                    Specifiche Tecniche
                                </Typography>
                            </Box>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0, pb: 2 }}>
                            <Box display="grid" gridTemplateColumns="1fr 1fr" gap={1.5}>
                                {specifics.map((spec, idx) => (
                                    <Box key={idx}>
                                        <Typography variant="caption" color="text.secondary" display="block">
                                            {spec.name}
                                        </Typography>
                                        <Typography variant="body2" fontWeight={500} color="text.primary">
                                            {spec.value}
                                        </Typography>
                                    </Box>
                                ))}
                            </Box>
                        </AccordionDetails>
                    </Accordion>
                )}

                {description && (
                    <Accordion
                        elevation={0}
                        disableGutters
                        sx={{
                            "&:before": { display: "none" },
                            bgcolor: "transparent",
                            border: "1px solid #e5e7eb",
                            borderRadius: "12px !important",
                            mb: 1
                        }}
                    >
                        <AccordionSummary expandIcon={<ExpandMoreIcon />} sx={{ minHeight: 48 }}>
                            <Box display="flex" alignItems="center" gap={1}>
                                <InfoOutlinedIcon fontSize="small" color="action" />
                                <Typography variant="body2" fontWeight={500}>
                                    Descrizione Completa
                                </Typography>
                            </Box>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0, pb: 2 }}>
                            <Box
                                sx={{
                                    maxHeight: 250,
                                    overflowY: "auto",
                                    fontSize: 13,
                                    color: "#4b5563",
                                    "& *": { maxWidth: "100%", breakWord: "break-all" }
                                }}
                                dangerouslySetInnerHTML={{ __html: description }}
                            />
                        </AccordionDetails>
                    </Accordion>
                )}

                {return_terms && Object.keys(return_terms).length > 0 && (
                    <Box display="flex" alignItems="center" gap={1} mt={2} bgcolor="#f8fafc" p={1.5} borderRadius={2}>
                        <VerifiedUserOutlinedIcon fontSize="small" color="success" />
                        <Typography variant="caption" color="text.secondary">
                            Reso: {return_terms.returnsAccepted ? "Accettato" : "Non Accettato"}
                            {return_terms.returnPeriod?.value && ` entro ${return_terms.returnPeriod.value} ${return_terms.returnPeriod.unit}`}
                            {return_terms.returnShippingCostPayer && ` (Spese a carico del ${return_terms.returnShippingCostPayer.toLowerCase()})`}
                        </Typography>
                    </Box>
                )}
            </Box>
        </Paper>
    )
}

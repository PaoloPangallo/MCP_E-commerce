import {
    Box,
    Typography,
    Accordion,
    AccordionSummary,
    AccordionDetails,
    Divider,
    Chip,
    Paper,
    Button
} from "@mui/material"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ListIcon from "@mui/icons-material/List"
import DescriptionIcon from "@mui/icons-material/Description"
import VerifiedUserOutlinedIcon from "@mui/icons-material/VerifiedUserOutlined"
import ImageIcon from "@mui/icons-material/Image"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import type { ItemDetailsData } from "../search/types"

interface ItemDetailsCardProps {
    data: ItemDetailsData
}

export default function ItemDetailsCard({ data }: ItemDetailsCardProps) {
    const {
        title,
        condition,
        price,
        description,
        item_specifics,
        return_terms,
        brand,
        image,
        additional_images,
        item_url
    } = data

    const priceValue = price?.value ? `${price.value} ${price.currency || "EUR"}` : "N/A"
    const specifics = Array.isArray(item_specifics) ? item_specifics : []
    const mainImageUrl = image?.imageUrl || ""
    const allImages = [
        ...(mainImageUrl ? [mainImageUrl] : []),
        ...(additional_images?.map((img: any) => img.imageUrl) || [])
    ].slice(0, 5)

    return (
        <Paper
            elevation={0}
            sx={{
                width: "100%",
                borderRadius: 4,
                border: "1px solid #e2e8f0",
                bgcolor: "#ffffff",
                overflow: "hidden",
                mt: 1,
                mb: 2,
                boxShadow: "0 4px 6px -1px rgb(0 0 0 / 0.1)"
            }}
        >
            <Box p={3}>
                {/* Header with Title & Price */}
                <Box mb={3}>
                    <Typography variant="h6" fontWeight={700} sx={{ color: "#1e293b", mb: 1, lineHeight: 1.2 }}>
                        {title || "Dettagli Prodotto"}
                    </Typography>
                    <Box display="flex" alignItems="center" gap={1.5} flexWrap="wrap">
                        <Typography variant="h5" fontWeight={800} color="primary.main">
                            {priceValue}
                        </Typography>
                        {condition && (
                            <Chip
                                label={condition}
                                size="small"
                                sx={{ bgcolor: "#eff6ff", color: "#2563eb", fontWeight: 600, border: "1px solid #dbeafe" }}
                            />
                        )}
                        {brand && <Chip label={brand} size="small" variant="outlined" sx={{ fontWeight: 500 }} />}
                    </Box>
                </Box>

                {/* Hero Image & Gallery */}
                {allImages.length > 0 && (
                    <Box mb={3}>
                        <Box
                            component="img"
                            src={allImages[0]}
                            sx={{
                                width: "100%",
                                maxHeight: 300,
                                objectFit: "contain",
                                borderRadius: 3,
                                bgcolor: "#f8fafc",
                                border: "1px solid #f1f5f9"
                            }}
                        />
                        {allImages.length > 1 && (
                            <Box display="flex" gap={1} mt={1} overflow="auto" pb={0.5}>
                                {allImages.slice(1).map((img, idx) => (
                                    <Box
                                        key={idx}
                                        component="img"
                                        src={img}
                                        sx={{
                                            width: 60,
                                            height: 60,
                                            borderRadius: 2,
                                            objectFit: "cover",
                                            border: "1px solid #e2e8f0",
                                            flexShrink: 0
                                        }}
                                    />
                                ))}
                            </Box>
                        )}
                    </Box>
                )}

                {/* Action Buttons */}
                {item_url && (
                    <Button
                        variant="contained"
                        fullWidth
                        href={item_url}
                        target="_blank"
                        startIcon={<OpenInNewIcon />}
                        sx={{
                            py: 1.2,
                            borderRadius: 3,
                            fontWeight: 700,
                            textTransform: "none",
                            boxShadow: "none",
                            "&:hover": { boxShadow: "0 10px 15px -3px rgb(0 0 0 / 0.1)" }
                        }}
                    >
                        Vedi e Acquista su eBay
                    </Button>
                )}

                <Divider sx={{ my: 3 }} />

                {/* Technical Specs */}
                {specifics.length > 0 && (
                    <Accordion
                        elevation={0}
                        disableGutters
                        defaultExpanded={specifics.length < 8}
                        sx={{
                            "&:before": { display: "none" },
                            bgcolor: "#f8fafc",
                            borderRadius: "16px !important",
                            border: "1px solid #f1f5f9",
                            mb: 2
                        }}
                    >
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Box display="flex" alignItems="center" gap={1}>
                                <ListIcon sx={{ color: "#64748b" }} />
                                <Typography variant="subtitle2" fontWeight={700} color="#334155">
                                    Specifiche Tecniche
                                </Typography>
                            </Box>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0, pb: 2.5, px: 2.5 }}>
                            <Box
                                sx={{
                                    display: "grid",
                                    gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" },
                                    gap: 2
                                }}
                            >
                                {specifics.map((spec, idx) => (
                                    <Box key={idx}>
                                        <Typography variant="caption" sx={{ color: "#94a3b8", fontWeight: 600, textTransform: "uppercase", letterSpacing: 0.5 }}>
                                            {spec.name}
                                        </Typography>
                                        <Typography variant="body2" sx={{ color: "#1e293b", fontWeight: 500 }}>
                                            {spec.value}
                                        </Typography>
                                    </Box>
                                ))}
                            </Box>
                        </AccordionDetails>
                    </Accordion>
                )}

                {/* Detailed Description */}
                {description && (
                    <Accordion
                        elevation={0}
                        disableGutters
                        sx={{
                            "&:before": { display: "none" },
                            bgcolor: "#ffffff",
                            borderRadius: "16px !important",
                            border: "1px solid #e2e8f0",
                            mb: 2
                        }}
                    >
                        <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                            <Box display="flex" alignItems="center" gap={1}>
                                <DescriptionIcon sx={{ color: "#64748b" }} />
                                <Typography variant="subtitle2" fontWeight={700} color="#334155">
                                    Descrizione Completa
                                </Typography>
                            </Box>
                        </AccordionSummary>
                        <AccordionDetails sx={{ pt: 0, px: 2.5, pb: 2.5 }}>
                            <Box
                                sx={{
                                    fontSize: 14,
                                    color: "#475569",
                                    lineHeight: 1.6,
                                    "& *": {
                                        maxWidth: "100% !important",
                                        boxSizing: "border-box !important",
                                        overflowWrap: "anywhere !important",
                                        wordBreak: "break-word !important"
                                    },
                                    "& img": { height: "auto !important", borderRadius: 2 },
                                    "& table": { width: "100% !important", borderCollapse: "collapse", my: 1 },
                                    "& td, & th": { border: "1px solid #e2e8f0", p: 1 }
                                }}
                                dangerouslySetInnerHTML={{ __html: description }}
                            />
                        </AccordionDetails>
                    </Accordion>
                )}

                {/* Return Policy */}
                {return_terms && (
                    <Box
                        display="flex"
                        alignItems="flex-start"
                        gap={1.5}
                        mt={1}
                        bgcolor="#eff6ff"
                        p={2}
                        borderRadius={3}
                        border="1px solid #dbeafe"
                    >
                        <VerifiedUserOutlinedIcon sx={{ color: "#2563eb", mt: 0.2 }} />
                        <Box>
                            <Typography variant="subtitle2" fontWeight={700} color="#1e40af">
                                Politica dei Resi
                            </Typography>
                            <Typography variant="caption" sx={{ color: "#60a5fa", display: "block", mt: 0.5, lineHeight: 1.4 }}>
                                {return_terms.returnsAccepted ? "Reso accettato" : "Reso non accettato"}
                                {return_terms.returnPeriod?.value && ` entro ${return_terms.returnPeriod.value} ${return_terms.returnPeriod.unit}`}.
                                {return_terms.returnShippingCostPayer && ` Spese a carico del ${return_terms.returnShippingCostPayer === "BUYER" ? "acquirente" : "venditore"}.`}
                            </Typography>
                        </Box>
                    </Box>
                )}
            </Box>
            {/* SIMILAR ITEMS */}
            {data.similar_items && data.similar_items.length > 0 && (
                <Accordion
                    elevation={0}
                    sx={{
                        borderTop: "1px solid #f1f5f9",
                        "&:before": { display: "none" }
                    }}
                >
                    <AccordionSummary expandIcon={<ExpandMoreIcon />}>
                        <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
                            <Typography sx={{ fontWeight: 600, fontSize: 13, color: "#64748b" }}>
                                PRODOTTI SIMILI
                            </Typography>
                            <Chip
                                label={data.similar_items.length}
                                size="small"
                                sx={{ height: 18, fontSize: 10, bgcolor: "#f1f5f9" }}
                            />
                        </Box>
                    </AccordionSummary>
                    <AccordionDetails sx={{ pt: 0, px: 2, pb: 2.5 }}>
                        <Box
                            sx={{
                                display: "flex",
                                gap: 2,
                                overflowX: "auto",
                                pb: 1,
                                "&::-webkit-scrollbar": { height: 6 },
                                "&::-webkit-scrollbar-thumb": { bgcolor: "#e2e8f0", borderRadius: 3 }
                            }}
                        >
                            {data.similar_items.map((item, idx) => (
                                <Paper
                                    key={`similar-${idx}`}
                                    elevation={0}
                                    component="a"
                                    href={item.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    sx={{
                                        minWidth: 160,
                                        maxWidth: 160,
                                        p: 1.5,
                                        borderRadius: 3,
                                        border: "1px solid #f1f5f9",
                                        textDecoration: "none",
                                        transition: "transform 0.2s, box-shadow 0.2s",
                                        "&:hover": {
                                            transform: "translateY(-4px)",
                                            boxShadow: "0 10px 15px -3px rgba(0,0,0,0.05)",
                                            borderColor: "#e2e8f0"
                                        }
                                    }}
                                >
                                    <Box sx={{ width: "100%", height: 100, mb: 1.5, borderRadius: 2, overflow: "hidden", bgcolor: "#f8fafc", display: "flex", alignItems: "center", justifyContent: "center" }}>
                                        {item.image_url ? (
                                            <Box component="img" src={item.image_url} alt={item.title} sx={{ width: "100%", height: "100%", objectFit: "contain" }} />
                                        ) : (
                                            <ImageIcon sx={{ color: "#cbd5e1", fontSize: 32 }} />
                                        )}
                                    </Box>
                                    <Typography
                                        sx={{
                                            fontSize: 12,
                                            fontWeight: 600,
                                            color: "#1e293b",
                                            lineHeight: 1.3,
                                            mb: 1,
                                            display: "-webkit-box",
                                            WebkitLineClamp: 2,
                                            WebkitBoxOrientation: "vertical",
                                            overflow: "hidden",
                                            minHeight: 31
                                        }}
                                    >
                                        {item.title}
                                    </Typography>
                                    <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#10a37f" }}>
                                        {item.price} {item.currency}
                                    </Typography>
                                </Paper>
                            ))}
                        </Box>
                    </AccordionDetails>
                </Accordion>
            )}
        </Paper>
    )
}

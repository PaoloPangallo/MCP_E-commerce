import {
    Box,
    Card,
    CardMedia,
    Chip,
    Link,
    Paper,
    Table,
    TableBody,
    TableCell,
    TableContainer,
    TableHead,
    TableRow,
    Typography,
    Divider,
    IconButton,
    Tooltip,
    Button
} from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import InfoOutlinedIcon from "@mui/icons-material/InfoOutlined"
import CheckCircleIcon from "@mui/icons-material/CheckCircle"
import TrendingUpIcon from "@mui/icons-material/TrendingUp"

import type { ComparisonData } from "../types"

interface ComparisonDisplayProps {
    data: ComparisonData
}

function formatPrice(price?: number, currency?: string) {
    if (typeof price !== "number") return "N/A"
    return `${price} ${currency ?? ""}`.trim()
}

/**
 * Enhanced Score Indicator with Glassmorphism
 */
function ScoreIndicator({ score, label, color }: { score: number; label: string; color?: string }) {
    const percent = Math.round(score * 100)
    const effectiveColor = color || (score >= 0.8 ? "#10b981" : score >= 0.6 ? "#f59e0b" : "#ef4444")

    return (
        <Box sx={{ mt: 1.5 }}>
            <Box display="flex" justifyContent="space-between" alignItems="center" mb={0.6}>
                <Typography sx={{
                    fontSize: 10,
                    fontWeight: 800,
                    color: "#94a3b8",
                    textTransform: "uppercase",
                    letterSpacing: "0.05em",
                    display: "flex",
                    alignItems: "center",
                    gap: 0.5
                }}>
                    {label}
                </Typography>
                <Typography sx={{ fontSize: 11, fontWeight: 900, color: effectiveColor }}>
                    {percent}%
                </Typography>
            </Box>
            <Box sx={{
                width: "100%",
                height: 6,
                bgcolor: "rgba(241, 245, 249, 0.5)",
                borderRadius: 10,
                overflow: "hidden",
                border: "1px solid rgba(0,0,0,0.02)"
            }}>
                <Box sx={{
                    width: `${percent}%`,
                    height: "100%",
                    bgcolor: effectiveColor,
                    borderRadius: 10,
                    boxShadow: `0 0 8px ${effectiveColor}44`
                }} />
            </Box>
        </Box>
    )
}

export default function ComparisonDisplay({ data }: ComparisonDisplayProps) {
    const { winner, comparison_matrix, winner_reason } = data

    // Logic to find "best" in matrix for highlighting
    const minPrice = Math.min(...comparison_matrix.map(c => c.price || Infinity))
    const maxOverall = Math.max(...comparison_matrix.map(c => c.scores?.overall || 0))

    return (
        <Box sx={{ width: "100%", py: 2 }}>

            {/* 🌟 AI RECOMMENDATION HERO SECTION 🌟 */}
            <Box sx={{ position: "relative", mb: 6 }}>
                <Paper
                    elevation={0}
                    sx={{
                        p: { xs: 3, md: 5 },
                        borderRadius: 6,
                        position: "relative",
                        overflow: "hidden",
                        background: "rgba(255, 255, 255, 0.7)",
                        backdropFilter: "blur(20px)",
                        border: "1px solid rgba(255, 255, 255, 0.4)",
                        boxShadow: "0 20px 50px -12px rgba(15, 23, 42, 0.1)",
                        "&::before": {
                            content: '""',
                            position: "absolute",
                            top: 0,
                            left: 0,
                            right: 0,
                            bottom: 0,
                            background: "radial-gradient(circle at 0% 0%, rgba(59, 130, 246, 0.05) 0%, transparent 50%), radial-gradient(circle at 100% 100%, rgba(139, 92, 246, 0.05) 0%, transparent 50%)",
                            zIndex: 0
                        }
                    }}
                >
                    <Box sx={{ position: "relative", zIndex: 1 }}>
                        <Box display="flex" justifyContent="space-between" alignItems="flex-start" mb={3}>
                            <Box sx={{
                                display: "flex",
                                alignItems: "center",
                                gap: 1.5,
                                px: 2,
                                py: 0.75,
                                borderRadius: 10,
                                bgcolor: "#fffbeb",
                                border: "1px solid #fef3c7",
                                boxShadow: "0 4px 12px rgba(245, 158, 11, 0.1)"
                            }}>
                                <EmojiEventsIcon sx={{ color: "#f59e0b", fontSize: 20 }} />
                                <Typography sx={{
                                    fontWeight: 900,
                                    color: "#92400e",
                                    fontSize: 11,
                                    textTransform: "uppercase",
                                    letterSpacing: "0.1em"
                                }}>
                                    AI Top Pick
                                </Typography>
                            </Box>

                            <Tooltip title="Questa raccomandazione è basata su analisi tecnica, trust del venditore e convenienza economica.">
                                <IconButton size="small" sx={{ color: "#64748b" }}>
                                    <InfoOutlinedIcon fontSize="small" />
                                </IconButton>
                            </Tooltip>
                        </Box>

                        <Box sx={{ display: "grid", gridTemplateColumns: { xs: "1fr", md: "1.5fr 1fr" }, gap: 4, alignItems: "center" }}>
                            <Box>
                                <Typography variant="h4" sx={{
                                    fontWeight: 900,
                                    color: "#0f172a",
                                    lineHeight: 1.2,
                                    mb: 2.5,
                                    letterSpacing: "-0.02em"
                                }}>
                                    {winner.title}
                                </Typography>
                                <Paper elevation={0} sx={{
                                    bgcolor: "rgba(241, 245, 249, 0.6)",
                                    p: 2.5,
                                    borderRadius: 4,
                                    border: "1px solid rgba(226, 232, 240, 0.8)"
                                }}>
                                    <Typography variant="body1" sx={{
                                        color: "#334155",
                                        lineHeight: 1.8,
                                        fontWeight: 500,
                                        fontSize: 15,
                                        "&::first-letter": { fontSize: "120%", fontWeight: "bold" }
                                    }}>
                                        {winner_reason}
                                    </Typography>
                                </Paper>
                            </Box>

                            <Box sx={{
                                position: "relative",
                                display: "flex",
                                flexDirection: "column",
                                alignItems: "center",
                                justifyContent: "center",
                                p: 3,
                                borderRadius: 5,
                                bgcolor: "#ffffff",
                                border: "1px solid #f1f5f9",
                                boxShadow: "0 10px 25px -10px rgba(0,0,0,0.05)",
                                minHeight: 240
                            }}>
                                {winner.image_url ? (
                                    <Box component="img"
                                        src={winner.image_url}
                                        sx={{
                                            width: "100%",
                                            height: 180,
                                            objectFit: "contain",
                                            filter: "drop-shadow(0 10px 15px rgba(0,0,0,0.05))",
                                            mb: 2
                                        }}
                                    />
                                ) : (
                                    <Box sx={{
                                        width: "100%",
                                        height: 180,
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        bgcolor: "#f8fafc",
                                        borderRadius: 3,
                                        mb: 2
                                    }}>
                                        <Typography sx={{ color: "#94a3b8", fontWeight: 700 }}>Foto non disponibile</Typography>
                                    </Box>
                                )}
                                <Typography sx={{ fontSize: 32, fontWeight: 900, color: "#0f172a" }}>
                                    {formatPrice(winner.price, winner.currency)}
                                </Typography>
                                <Button
                                    variant="contained"
                                    fullWidth
                                    component="a"
                                    href={winner.url || "#"}
                                    target="_blank"
                                    startIcon={<OpenInNewIcon />}
                                    sx={{
                                        mt: 2,
                                        borderRadius: 3,
                                        py: 1.2,
                                        textTransform: "none",
                                        fontWeight: 800,
                                        bgcolor: "#0f172a",
                                        "&:hover": { bgcolor: "#1e293b" }
                                    }}
                                >
                                    Acquista subito
                                </Button>
                            </Box>
                        </Box>
                    </Box>
                </Paper>

                {/* Background Decor */}
                <AutoAwesomeIcon sx={{
                    position: "absolute",
                    top: -20,
                    right: -20,
                    fontSize: 80,
                    color: "rgba(59, 130, 246, 0.08)",
                    zIndex: 0
                }} />
            </Box>

            {/* 💎 PRODUCT CARDS 💎 */}
            <Box
                sx={{
                    display: "grid",
                    gridTemplateColumns: {
                        xs: "1fr",
                        sm: comparison_matrix.length > 2 ? "repeat(3, 1fr)" : "repeat(2, 1fr)"
                    },
                    gap: 3,
                    mb: 8
                }}
            >
                {comparison_matrix.map((candidate, idx) => {
                    const isWinner = candidate.title === winner.title;
                    return (
                        <Card
                            key={idx}
                            elevation={0}
                            sx={{
                                height: "100%",
                                borderRadius: 5,
                                border: isWinner ? "2px solid #fbbf24" : "1px solid #e2e8f0",
                                background: isWinner ? "linear-gradient(180deg, #ffffff 0%, #fffdf5 100%)" : "#ffffff",
                                position: "relative",
                                overflow: "visible",
                                transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
                                "&:hover": {
                                    transform: "translateY(-8px)",
                                    boxShadow: isWinner
                                        ? "0 30px 60px -12px rgba(245, 158, 11, 0.15)"
                                        : "0 30px 60px -12px rgba(15, 23, 42, 0.12)"
                                }
                            }}
                        >
                            {isWinner && (
                                <Box sx={{
                                    position: "absolute",
                                    top: -12,
                                    left: "50%",
                                    transform: "translateX(-50%)",
                                    zIndex: 5
                                }}>
                                    <Chip
                                        icon={<AutoAwesomeIcon sx={{ fontSize: "12px !important", color: "#ffffff !important" }} />}
                                        label="MIGLIORE"
                                        size="small"
                                        sx={{
                                            bgcolor: "#f59e0b",
                                            color: "#ffffff",
                                            fontWeight: 900,
                                            height: 24,
                                            fontSize: 10,
                                            letterSpacing: "0.1em",
                                            boxShadow: "0 4px 10px rgba(245, 158, 11, 0.4)",
                                            "& .MuiChip-label": { px: 1.5 }
                                        }}
                                    />
                                </Box>
                            )}

                            <Box sx={{ p: 3 }}>
                                <Box
                                    sx={{
                                        width: "100%",
                                        aspectRatio: "1/1",
                                        borderRadius: 4,
                                        bgcolor: "#f8fafc",
                                        overflow: "hidden",
                                        mb: 3,
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        border: "1px solid rgba(226, 232, 240, 0.5)"
                                    }}
                                >
                                    <CardMedia
                                        component="img"
                                        image={candidate.image_url || "/placeholder-product.png"}
                                        alt={candidate.title}
                                        sx={{ width: "85%", height: "85%", objectFit: "contain", transition: "transform 0.5s ease", "&:hover": { transform: "scale(1.1)" } }}
                                    />
                                </Box>

                                <Typography
                                    sx={{
                                        fontSize: 15,
                                        fontWeight: 800,
                                        color: "#1e293b",
                                        lineHeight: 1.5,
                                        mb: 2,
                                        display: "-webkit-box",
                                        WebkitLineClamp: 2,
                                        WebkitBoxOrient: "vertical",
                                        overflow: "hidden",
                                        height: 45
                                    }}
                                >
                                    {candidate.title}
                                </Typography>

                                <Box display="flex" alignItems="baseline" gap={1} mb={3}>
                                    <Typography sx={{ fontSize: 26, fontWeight: 900, color: "#0f172a" }}>
                                        {formatPrice(candidate.price, candidate.currency)}
                                    </Typography>
                                </Box>

                                <Box sx={{ bgcolor: "rgba(248, 250, 252, 0.8)", p: 2, borderRadius: 3, mb: 3 }}>
                                    {candidate.scores && (
                                        <>
                                            <ScoreIndicator score={candidate.scores.overall} label="AI Match" color="#3b82f6" />
                                            <ScoreIndicator score={candidate.scores.price} label="Price Score" />
                                            <ScoreIndicator score={candidate.scores.trust} label="Trust Score" />
                                        </>
                                    )}
                                </Box>

                                <Divider sx={{ mb: 2, opacity: 0.5 }} />

                                <Link
                                    href={candidate.url}
                                    target="_blank"
                                    rel="noopener noreferrer"
                                    sx={{
                                        display: "flex",
                                        alignItems: "center",
                                        justifyContent: "center",
                                        gap: 1,
                                        color: "#475569",
                                        textDecoration: "none",
                                        fontSize: 12,
                                        fontWeight: 700,
                                        py: 1,
                                        borderRadius: 2,
                                        transition: "all 0.2s",
                                        "&:hover": { color: "#0f172a", bgcolor: "#f1f5f9" }
                                    }}
                                >
                                    Vedi su eBay <OpenInNewIcon sx={{ fontSize: 14 }} />
                                </Link>
                            </Box>
                        </Card>
                    );
                })}
            </Box>

            {/* 📊 ADVANCED ANALYTICS MATRIX 📊 */}
            <Typography sx={{
                fontWeight: 900,
                fontSize: 18,
                mb: 3,
                color: "#1e293b",
                display: "flex",
                alignItems: "center",
                gap: 1.5,
                letterSpacing: "-0.02em"
            }}>
                <TrendingUpIcon sx={{ color: "#3b82f6" }} /> Analisi Comparativa Dettagliata
            </Typography>

            <TableContainer component={Paper} elevation={0} sx={{
                borderRadius: 5,
                border: "1px solid #e2e8f0",
                overflow: "hidden",
                boxShadow: "0 4px 20px -5px rgba(0,0,0,0.05)"
            }}>
                <Table>
                    <TableHead>
                        <TableRow sx={{ bgcolor: "#F8FAFC" }}>
                            <TableCell sx={{
                                fontWeight: 900,
                                fontSize: 11,
                                color: "#64748b",
                                textTransform: "uppercase",
                                letterSpacing: "0.1em",
                                py: 2.5
                            }}>
                                Caratteristica
                            </TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center" sx={{
                                    fontWeight: 900,
                                    fontSize: 12,
                                    color: c.title === winner.title ? "#0f172a" : "#64748b",
                                    bgcolor: c.title === winner.title ? "rgba(245, 158, 11, 0.03)" : "transparent",
                                    minWidth: 160
                                }}>
                                    <Box display="flex" flexDirection="column" gap={0.5}>
                                        <Typography sx={{ fontSize: 10, opacity: 0.6 }}>OPZIONE {i + 1}</Typography>
                                        <Typography noWrap sx={{ fontSize: 12, fontWeight: 900 }}>
                                            {(c.title || "Prodotto").split(' ').slice(0, 3).join(' ')}...
                                        </Typography>
                                    </Box>
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        {/* PRICE ROW with highlights */}
                        <TableRow>
                            <TableCell sx={{ fontSize: 13, fontWeight: 800, color: "#334155" }}>Prezzo Oggetto</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center">
                                    <Typography sx={{
                                        fontSize: 14,
                                        fontWeight: 900,
                                        color: c.price === minPrice ? "#10b981" : "#0f172a",
                                        display: "inline-flex",
                                        alignItems: "center",
                                        gap: 0.5
                                    }}>
                                        {formatPrice(c.price, c.currency)}
                                        {c.price === minPrice && <CheckCircleIcon sx={{ fontSize: 14 }} />}
                                    </Typography>
                                </TableCell>
                            ))}
                        </TableRow>

                        <TableRow sx={{ bgcolor: "#FDFDFD" }}>
                            <TableCell sx={{ fontSize: 13, fontWeight: 800, color: "#334155" }}>Spedizione</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center">
                                    <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#059669" }}>
                                        {(() => {
                                            const opt = c.shipping_info?.shipping_options?.[0]
                                            const costValue = opt?.shippingCost?.value || opt?.cost?.value
                                            if (!costValue || costValue === "0.00") return "GRATIS"
                                            return `${costValue} ${opt?.shippingCost?.currency || opt?.cost?.currency || ""}`
                                        })()}
                                    </Typography>
                                </TableCell>
                            ))}
                        </TableRow>

                        <TableRow sx={{ bgcolor: "#FDFDFD" }}>
                            <TableCell sx={{ fontSize: 13, fontWeight: 800, color: "#334155" }}>Condizioni d'Uso</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center">
                                    <Chip
                                        label={c.condition || "N/A"}
                                        size="small"
                                        sx={{
                                            fontSize: 10,
                                            fontWeight: 800,
                                            bgcolor: c.condition?.toLowerCase().includes("nuov") ? "#ecfdf5" : "#f1f5f9",
                                            color: c.condition?.toLowerCase().includes("nuov") ? "#065f46" : "#475569",
                                            border: "1px solid rgba(0,0,0,0.05)"
                                        }}
                                    />
                                </TableCell>
                            ))}
                        </TableRow>

                        <TableRow>
                            <TableCell sx={{ fontSize: 13, fontWeight: 800, color: "#334155" }}>Affidabilità Seller</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center">
                                    <Box display="flex" flexDirection="column" alignItems="center" gap={0.5}>
                                        <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#1e293b" }}>{c.seller_name}</Typography>
                                        <Box display="flex" gap={0.5} alignItems="center">
                                            <Typography sx={{ fontSize: 10, fontWeight: 900, color: "#94a3b8" }}>
                                                {Math.round((c.trust_score || 0) * 100)}% TRUST
                                            </Typography>
                                            {(c.trust_score || 0) >= 0.9 && <VerifiedUserIcon sx={{ fontSize: 12, color: "#3b82f6" }} />}
                                        </Box>
                                    </Box>
                                </TableCell>
                            ))}
                        </TableRow>

                        <TableRow sx={{ bgcolor: "#FDFDFD" }}>
                            <TableCell sx={{ fontSize: 13, fontWeight: 800, color: "#334155" }}>AI Relevance</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center">
                                    <Box sx={{
                                        display: "inline-flex",
                                        px: 1.5,
                                        py: 0.5,
                                        borderRadius: 10,
                                        bgcolor: c.scores?.overall === maxOverall ? "rgba(59, 130, 246, 0.1)" : "transparent",
                                        border: c.scores?.overall === maxOverall ? "1px solid rgba(59, 130, 246, 0.2)" : "none"
                                    }}>
                                        <Typography sx={{
                                            fontSize: 13,
                                            fontWeight: 900,
                                            color: c.scores?.overall === maxOverall ? "#3b82f6" : "#94a3b8"
                                        }}>
                                            {Math.round((c.scores?.overall || 0) * 100)}%
                                        </Typography>
                                    </Box>
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableBody>
                </Table>
            </TableContainer>

            <Box mt={4} sx={{
                p: 3,
                bgcolor: "#f8fafc",
                borderRadius: 4,
                border: "1px dashed #e2e8f0",
                textAlign: 'center'
            }}>
                <Typography variant="caption" sx={{ color: '#64748b', fontWeight: 600, display: "flex", alignItems: "center", justifyContent: "center", gap: 1 }}>
                    <AutoAwesomeIcon sx={{ fontSize: 14 }} />
                    Analisi generata in tempo reale incrociando dati di mercato, reputazione storica del venditore e pertinenza semantica con la tua ricerca.
                </Typography>
            </Box>
        </Box>
    )
}

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
    Typography
} from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"

import type { ComparisonData } from "../types"

interface ComparisonDisplayProps {
    data: ComparisonData
}

function formatPrice(price?: number, currency?: string) {
    if (typeof price !== "number") return "N/A"
    return `${price} ${currency ?? ""}`.trim()
}

function ScoreIndicator({ score, label }: { score: number; label: string }) {
    const percent = Math.round(score * 100)
    let color = "#ef4444" // red
    if (score >= 0.8) color = "#22c55e" // green
    else if (score >= 0.6) color = "#eab308" // yellow

    return (
        <Box sx={{ mt: 0.5 }}>
            <Box display="flex" justifyContent="space-between" mb={0.25}>
                <Typography sx={{ fontSize: 10, fontWeight: 600, color: "#6b7280", textTransform: "uppercase" }}>
                    {label}
                </Typography>
                <Typography sx={{ fontSize: 10, fontWeight: 700, color }}>
                    {percent}%
                </Typography>
            </Box>
            <Box sx={{ width: "100%", height: 4, bgcolor: "#f1f5f9", borderRadius: 2, overflow: "hidden" }}>
                <Box sx={{ width: `${percent}%`, height: "100%", bgcolor: color }} />
            </Box>
        </Box>
    )
}

export default function ComparisonDisplay({ data }: ComparisonDisplayProps) {
    const { winner, comparison_matrix, winner_reason } = data

    return (
        <Box sx={{ width: "100%" }}>
            {/* WINNER BANNER */}
            <Paper
                elevation={0}
                sx={{
                    p: 2.5,
                    mb: 3,
                    borderRadius: 5,
                    background: "linear-gradient(135deg, #0f172a 0%, #1e293b 50%, #334155 100%)",
                    boxShadow: "0 10px 25px -5px rgba(0, 0, 0, 0.1), 0 8px 10px -6px rgba(0, 0, 0, 0.1)",
                    color: "#fff",
                    position: "relative",
                    overflow: "hidden"
                }}
            >
                <Box sx={{ position: "relative", zIndex: 1 }}>
                    <Box display="flex" alignItems="center" gap={1} mb={1}>
                        <EmojiEventsIcon sx={{ color: "#fbbf24" }} />
                        <Typography variant="subtitle2" sx={{ fontWeight: 700, textTransform: "uppercase", letterSpacing: 1 }}>
                            Vincitore Consigliato
                        </Typography>
                    </Box>
                    <Typography variant="h6" sx={{ fontWeight: 700, mb: 1 }}>
                        {winner.title}
                    </Typography>
                    <Typography variant="body2" sx={{ color: "#d1d5db", lineHeight: 1.6 }}>
                        {winner_reason}
                    </Typography>
                </Box>
                <AutoAwesomeIcon
                    sx={{
                        position: "absolute",
                        right: -20,
                        bottom: -20,
                        fontSize: 120,
                        color: "rgba(255,255,255,0.05)"
                    }}
                />
            </Paper>

            {/* COMPARISON GRID */}
            <Box
                sx={{
                    display: "grid",
                    gridTemplateColumns: {
                        xs: "1fr",
                        sm: comparison_matrix.length > 2 ? "repeat(3, 1fr)" : "repeat(2, 1fr)"
                    },
                    gap: 2,
                    mb: 3
                }}
            >
                {comparison_matrix.map((candidate, idx) => (
                    <Card
                        key={idx}
                        elevation={0}
                        sx={{
                            height: "100%",
                            borderRadius: 4,
                            border: "1px solid #e5e7eb",
                            position: "relative",
                            transition: "transform 0.2s ease",
                            "&:hover": { transform: "translateY(-4px)" },
                            ...(candidate.title === winner.title && {
                                borderColor: "#fbbf24",
                                boxShadow: "0 0 0 1px #fbbf24"
                            })
                        }}
                    >
                        {candidate.title === winner.title && (
                            <Chip
                                icon={<EmojiEventsIcon sx={{ fontSize: "14px !important", color: "#92400e !important" }} />}
                                label="Best Choice"
                                size="small"
                                sx={{
                                    position: "absolute",
                                    top: 14,
                                    right: 14,
                                    zIndex: 2,
                                    bgcolor: "#fef3c7",
                                    color: "#92400e",
                                    fontWeight: 800,
                                    border: "1px solid #fcd34d",
                                    boxShadow: "0 2px 4px rgba(0,0,0,0.05)"
                                }}
                            />
                        )}

                        <Box sx={{ p: 2 }}>
                            <Box
                                sx={{
                                    width: "100%",
                                    aspectRatio: "1/1",
                                    borderRadius: 3,
                                    bgcolor: "#f8fafc",
                                    overflow: "hidden",
                                    mb: 1.5,
                                    display: "flex",
                                    alignItems: "center",
                                    justifyContent: "center",
                                    border: "1px solid #f1f5f9"
                                }}
                            >
                                {candidate.image_url ? (
                                    <CardMedia
                                        component="img"
                                        image={candidate.image_url}
                                        alt={candidate.title}
                                        sx={{ width: "100%", height: "100%", objectFit: "contain" }}
                                    />
                                ) : (
                                    <Typography sx={{ color: "#94a3b8", fontSize: 12 }}>No image</Typography>
                                )}
                            </Box>

                            <Typography
                                sx={{
                                    fontSize: 14,
                                    fontWeight: 700,
                                    lineHeight: 1.4,
                                    mb: 1,
                                    display: "-webkit-box",
                                    WebkitLineClamp: 2,
                                    WebkitBoxOrient: "vertical",
                                    overflow: "hidden",
                                    height: 40
                                }}
                            >
                                {candidate.title}
                            </Typography>

                            <Typography sx={{ fontSize: 20, fontWeight: 800, color: "#111827" }}>
                                {formatPrice(candidate.price, candidate.currency)}
                            </Typography>

                            <Box mt={2}>
                                {candidate.scores && (
                                    <>
                                        <ScoreIndicator score={candidate.scores.overall} label="Overall Match" />
                                        <ScoreIndicator score={candidate.scores.price} label="Price Score" />
                                        <ScoreIndicator score={candidate.scores.trust} label="Seller Trust" />
                                    </>
                                )}
                            </Box>

                            <Box mt={2}>
                                <Link
                                    href={candidate.url}
                                    target="_blank"
                                    sx={{
                                        display: "flex",
                                        alignItems: "center",
                                        gap: 0.5,
                                        fontSize: 12,
                                        fontWeight: 600,
                                        textDecoration: "none"
                                    }}
                                >
                                    Vedi su eBay <OpenInNewIcon sx={{ fontSize: 12 }} />
                                </Link>
                            </Box>
                        </Box>
                    </Card>
                ))}
            </Box>

            {/* MATRIX TABLE */}
            <TableContainer component={Paper} elevation={0} sx={{
                borderRadius: 5,
                border: "1px solid #e5e7eb",
                overflow: "hidden",
                boxShadow: "0 4px 6px -1px rgba(0, 0, 0, 0.05)"
            }}>
                <Table size="small">
                    <TableHead>
                        <TableRow sx={{ bgcolor: "#F9FAFB" }}>
                            <TableCell sx={{ fontWeight: 700, fontSize: 12 }}>Caratteristica</TableCell>
                            {comparison_matrix.map((_, i) => (
                                <TableCell key={i} align="center" sx={{ fontWeight: 700, fontSize: 12, minWidth: 120 }}>
                                    Prodotto {i + 1}
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableHead>
                    <TableBody>
                        <TableRow>
                            <TableCell sx={{ fontSize: 12, color: "#6b7280" }}>Prezzo</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center" sx={{ fontSize: 12, fontWeight: 600 }}>
                                    {formatPrice(c.price, c.currency)}
                                </TableCell>
                            ))}
                        </TableRow>
                        <TableRow>
                            <TableCell sx={{ fontSize: 12, color: "#6b7280" }}>Condizione</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center" sx={{ fontSize: 12 }}>
                                    {c.condition || "N/A"}
                                </TableCell>
                            ))}
                        </TableRow>
                        <TableRow>
                            <TableCell sx={{ fontSize: 12, color: "#6b7280" }}>Venditore</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center" sx={{ fontSize: 12 }}>
                                    <Box display="flex" alignItems="center" justifyContent="center" gap={0.5}>
                                        {c.seller_name}
                                        {typeof c.trust_score === "number" && c.trust_score >= 0.9 && <VerifiedUserIcon sx={{ fontSize: 12, color: "#22c55e" }} />}
                                    </Box>
                                </TableCell>
                            ))}
                        </TableRow>
                        <TableRow>
                            <TableCell sx={{ fontSize: 12, color: "#6b7280" }}>Corrispondenza Query</TableCell>
                            {comparison_matrix.map((c, i) => (
                                <TableCell key={i} align="center" sx={{ fontSize: 11, fontStyle: "italic", color: "#9ca3af" }}>
                                    "{c.query}"
                                </TableCell>
                            ))}
                        </TableRow>
                    </TableBody>
                </Table>
            </TableContainer>

            <Box mt={1.5} sx={{ textAlign: 'center' }}>
                <Typography variant="caption" sx={{ color: '#94a3b8', fontStyle: 'italic' }}>
                    * Punteggi calcolati dal motore di trust e analisi semantica di ebayGPT
                </Typography>
            </Box>
        </Box>
    )
}

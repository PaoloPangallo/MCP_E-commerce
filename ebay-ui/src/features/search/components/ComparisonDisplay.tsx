import {
  Box,
  Link,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography
} from "@mui/material"
import OpenInNewIcon from "@mui/icons-material/OpenInNew"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import CheckIcon from "@mui/icons-material/Check"

import type { ComparisonData } from "../types"

interface ComparisonDisplayProps {
  data: ComparisonData
}

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") return "—"
  return `${price} ${currency ?? ""}`.trim()
}

interface ScoreBarProps {
  score: number
  color: string
}

function ScoreBar({ score, color }: ScoreBarProps) {
  const pct = Math.round(score * 100)
  return (
    <Box>
      <Typography sx={{ fontSize: 10, color: "#9ca3af", mb: 0.4 }}>{pct}%</Typography>
      <Box
        sx={{
          width: "100%",
          height: 4,
          bgcolor: "#f3f4f6",
          borderRadius: 4,
          overflow: "hidden"
        }}
      >
        <Box
          sx={{
            width: `${pct}%`,
            height: "100%",
            bgcolor: color,
            borderRadius: 4
          }}
        />
      </Box>
    </Box>
  )
}

export default function ComparisonDisplay({ data }: ComparisonDisplayProps) {
  const { winner, comparison_matrix, winner_reason } = data

  const minPrice = Math.min(...comparison_matrix.map((c) => c.price ?? Infinity))
  const maxOverall = Math.max(...comparison_matrix.map((c) => c.scores?.overall ?? 0))

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>

      {/* Winner card */}
      <Box
        sx={{
          border: "1px solid #e5e7eb",
          borderRadius: 3,
          overflow: "hidden",
          bgcolor: "#fff"
        }}
      >
        <Box
          sx={{
            px: 2,
            py: 1.25,
            borderBottom: "1px solid #f5f5f5",
            display: "flex",
            alignItems: "center",
            gap: 0.75
          }}
        >
          <Box
            sx={{
              width: 18,
              height: 18,
              borderRadius: "50%",
              bgcolor: "#fef3c7",
              border: "1px solid #fde68a",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              fontSize: 10
            }}
          >
            ★
          </Box>
          <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>
            AI Top Pick
          </Typography>
        </Box>

        <Box
          sx={{
            p: 2,
            display: "grid",
            gridTemplateColumns: { xs: "1fr", sm: "80px 1fr" },
            gap: 1.75
          }}
        >
          <Box
            sx={{
              width: 80,
              height: 80,
              borderRadius: 2,
              overflow: "hidden",
              bgcolor: "#f9fafb",
              border: "1px solid #f0f0f0",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              flexShrink: 0
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
              <Typography sx={{ fontSize: 10, color: "#d1d5db" }}>no img</Typography>
            )}
          </Box>

          <Box>
            <Typography
              sx={{ fontSize: 13, fontWeight: 500, color: "#111827", lineHeight: 1.4, mb: 0.5 }}
            >
              {winner.title}
            </Typography>
            <Typography sx={{ fontSize: 18, fontWeight: 600, color: "#111827", mb: 0.75 }}>
              {formatPrice(winner.price, winner.currency)}
            </Typography>
            <Typography sx={{ fontSize: 12, color: "#6b7280", lineHeight: 1.6 }}>
              {winner_reason}
            </Typography>
            {winner.url && (
              <Link
                href={winner.url}
                target="_blank"
                rel="noreferrer"
                underline="none"
                sx={{
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 0.4,
                  mt: 1,
                  fontSize: 12,
                  color: "#374151",
                  border: "1px solid #e5e7eb",
                  borderRadius: "20px",
                  px: 1.25,
                  py: 0.4,
                  "&:hover": { bgcolor: "#f9fafb" }
                }}
              >
                Acquista su eBay
                <OpenInNewIcon sx={{ fontSize: 11 }} />
              </Link>
            )}
          </Box>
        </Box>
      </Box>

      {/* Candidate cards */}
      <Box
        sx={{
          display: "grid",
          gridTemplateColumns: {
            xs: "1fr",
            sm: comparison_matrix.length > 2 ? "repeat(3, 1fr)" : "repeat(2, 1fr)"
          },
          gap: 1.5
        }}
      >
        {comparison_matrix.map((candidate, idx) => {
          const isWinner = candidate.title === winner.title
          return (
            <Box
              key={idx}
              sx={{
                border: "1px solid",
                borderColor: isWinner ? "#fde68a" : "#f0f0f0",
                borderRadius: 3,
                bgcolor: isWinner ? "#fffdf5" : "#fff",
                p: 1.75,
                display: "flex",
                flexDirection: "column",
                gap: 1
              }}
            >
              <Box
                sx={{
                  width: "100%",
                  aspectRatio: "1 / 1",
                  borderRadius: 2,
                  bgcolor: "#f9fafb",
                  overflow: "hidden",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  border: "1px solid #f0f0f0"
                }}
              >
                {candidate.image_url ? (
                  <Box
                    component="img"
                    src={candidate.image_url}
                    alt={candidate.title}
                    sx={{ width: "100%", height: "100%", objectFit: "contain" }}
                  />
                ) : (
                  <Typography sx={{ fontSize: 10, color: "#d1d5db" }}>no img</Typography>
                )}
              </Box>

              <Typography
                sx={{
                  fontSize: 12,
                  fontWeight: 500,
                  color: "#111827",
                  lineHeight: 1.4,
                  display: "-webkit-box",
                  WebkitLineClamp: 2,
                  WebkitBoxOrient: "vertical",
                  overflow: "hidden"
                }}
              >
                {candidate.title}
              </Typography>

              <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
                <Typography sx={{ fontSize: 15, fontWeight: 600, color: "#111827" }}>
                  {formatPrice(candidate.price, candidate.currency)}
                </Typography>
                {candidate.price === minPrice && (
                  <Box
                    sx={{
                      display: "inline-flex",
                      alignItems: "center",
                      gap: 0.25,
                      px: 0.75,
                      py: 0.15,
                      borderRadius: "6px",
                      bgcolor: "#f0fdf4",
                      border: "1px solid #bbf7d0"
                    }}
                  >
                    <CheckIcon sx={{ fontSize: 9, color: "#16a34a" }} />
                    <Typography sx={{ fontSize: 10, color: "#15803d", fontWeight: 500 }}>
                      min
                    </Typography>
                  </Box>
                )}
              </Box>

              {candidate.scores && (
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                  <Box>
                    <Typography sx={{ fontSize: 10, color: "#9ca3af", mb: 0.3 }}>AI match</Typography>
                    <ScoreBar score={candidate.scores.overall} color="#7c3aed" />
                  </Box>
                  <Box>
                    <Typography sx={{ fontSize: 10, color: "#9ca3af", mb: 0.3 }}>Price</Typography>
                    <ScoreBar score={candidate.scores.price} color="#0ea5e9" />
                  </Box>
                  <Box>
                    <Typography sx={{ fontSize: 10, color: "#9ca3af", mb: 0.3 }}>Trust</Typography>
                    <ScoreBar score={candidate.scores.trust} color="#10b981" />
                  </Box>
                </Box>
              )}

              {candidate.url && (
                <Link
                  href={candidate.url}
                  target="_blank"
                  rel="noreferrer"
                  underline="none"
                  sx={{
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 0.4,
                    fontSize: 11,
                    color: "#6b7280",
                    mt: "auto",
                    pt: 0.5,
                    "&:hover": { color: "#374151" }
                  }}
                >
                  Vedi su eBay
                  <OpenInNewIcon sx={{ fontSize: 11 }} />
                </Link>
              )}
            </Box>
          )
        })}
      </Box>

      {/* Comparison table */}
      <Box>
        <Typography
          sx={{
            fontSize: 12,
            fontWeight: 500,
            color: "#9ca3af",
            textTransform: "uppercase",
            letterSpacing: "0.06em",
            mb: 1
          }}
        >
          Confronto dettagliato
        </Typography>

        <TableContainer
          sx={{ border: "1px solid #f0f0f0", borderRadius: 2, overflow: "hidden" }}
        >
          <Table size="small">
            <TableHead>
              <TableRow sx={{ bgcolor: "#fafafa" }}>
                <TableCell
                  sx={{
                    fontSize: 11,
                    fontWeight: 500,
                    color: "#9ca3af",
                    py: 1.25,
                    borderBottom: "1px solid #f0f0f0"
                  }}
                >
                  Caratteristica
                </TableCell>
                {comparison_matrix.map((c, i) => (
                  <TableCell
                    key={i}
                    align="center"
                    sx={{
                      fontSize: 11,
                      fontWeight: 500,
                      color: c.title === winner.title ? "#111827" : "#9ca3af",
                      py: 1.25,
                      borderBottom: "1px solid #f0f0f0"
                    }}
                  >
                    Opzione {i + 1}
                  </TableCell>
                ))}
              </TableRow>
            </TableHead>
            <TableBody>
              <TableRow>
                <TableCell sx={{ fontSize: 12, color: "#374151", borderBottom: "1px solid #f9f9f9" }}>
                  Prezzo
                </TableCell>
                {comparison_matrix.map((c, i) => (
                  <TableCell key={i} align="center" sx={{ borderBottom: "1px solid #f9f9f9" }}>
                    <Typography
                      sx={{
                        fontSize: 12,
                        fontWeight: c.price === minPrice ? 600 : 400,
                        color: c.price === minPrice ? "#16a34a" : "#374151"
                      }}
                    >
                      {formatPrice(c.price, c.currency)}
                    </Typography>
                  </TableCell>
                ))}
              </TableRow>

              <TableRow sx={{ bgcolor: "#fafafa" }}>
                <TableCell sx={{ fontSize: 12, color: "#374151", borderBottom: "1px solid #f9f9f9" }}>
                  Condizioni
                </TableCell>
                {comparison_matrix.map((c, i) => (
                  <TableCell key={i} align="center" sx={{ borderBottom: "1px solid #f9f9f9" }}>
                    <Typography sx={{ fontSize: 11, color: "#6b7280" }}>
                      {c.condition || "—"}
                    </Typography>
                  </TableCell>
                ))}
              </TableRow>

              <TableRow>
                <TableCell sx={{ fontSize: 12, color: "#374151", borderBottom: "1px solid #f9f9f9" }}>
                  Venditore
                </TableCell>
                {comparison_matrix.map((c, i) => (
                  <TableCell key={i} align="center" sx={{ borderBottom: "1px solid #f9f9f9" }}>
                    <Typography sx={{ fontSize: 11, color: "#374151" }}>
                      {c.seller_name || "—"}
                    </Typography>
                    <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.25, mt: 0.25 }}>
                      <VerifiedUserIcon
                        sx={{
                          fontSize: 10,
                          color: (c.trust_score ?? 0) >= 0.9 ? "#16a34a" : "#d1d5db"
                        }}
                      />
                      <Typography sx={{ fontSize: 10, color: "#9ca3af" }}>
                        {Math.round((c.trust_score ?? 0) * 100)}%
                      </Typography>
                    </Box>
                  </TableCell>
                ))}
              </TableRow>

              <TableRow sx={{ bgcolor: "#fafafa" }}>
                <TableCell sx={{ fontSize: 12, color: "#374151", border: "none" }}>
                  AI relevance
                </TableCell>
                {comparison_matrix.map((c, i) => (
                  <TableCell key={i} align="center" sx={{ border: "none" }}>
                    <Typography
                      sx={{
                        fontSize: 12,
                        fontWeight: c.scores?.overall === maxOverall ? 600 : 400,
                        color: c.scores?.overall === maxOverall ? "#6d28d9" : "#9ca3af"
                      }}
                    >
                      {Math.round((c.scores?.overall ?? 0) * 100)}%
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
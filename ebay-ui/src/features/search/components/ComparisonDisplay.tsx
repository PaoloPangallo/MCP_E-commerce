import React, { useRef, useState, useEffect, useMemo, useCallback } from "react"
import {
  Box,
  Table,
  TableBody,
  TableCell,
  TableContainer,
  TableHead,
  TableRow,
  Typography,
  Button,
  IconButton,
  Tooltip,
} from "@mui/material"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import LocalShippingIcon from "@mui/icons-material/LocalShipping"
import WorkspacePremiumIcon from "@mui/icons-material/WorkspacePremium"
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft"
import ChevronRightIcon from "@mui/icons-material/ChevronRight"
import SearchResultCard from "./SearchResultCard"

import type { ComparisonData, SearchItem } from "../types"

// ── Helpers ─────────────────────────────────────────────────────────

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") return "—"
  const formatted = price.toLocaleString("it-IT", {
    minimumFractionDigits: 2,
    maximumFractionDigits: 2,
  })
  return `${formatted} ${currency ?? ""}`.trim()
}

// ── Sub-Components ──────────────────────────────────────────────────

interface ScoreItemProps {
  label: string
  score: number
}

const ScoreIndicator = React.memo(({ label, score }: ScoreItemProps) => {
  const pct = Math.round(score * 100)
  return (
    <Box sx={{ flex: 1, minWidth: 80 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
        <Typography sx={{ fontSize: 9, color: "var(--text-secondary)", fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em" }}>
          {label}
        </Typography>
        <Typography sx={{ fontSize: 10, fontWeight: 800, color: "var(--text-primary)" }}>{pct}%</Typography>
      </Box>
      <Box
        sx={{
          width: "100%",
          height: 3,
          bgcolor: "var(--border-color)",
          borderRadius: 1,
          overflow: "hidden",
        }}
      >
        <Box
          sx={{
            width: `${pct}%`,
            height: "100%",
            bgcolor: "var(--brand-primary)",
            transition: "width 0.8s cubic-bezier(0.4, 0, 0.2, 1)",
          }}
        />
      </Box>
    </Box>
  )
})

interface WinnerHeroProps {
  winner: SearchItem
  winner_reason: string
  minPrice: number
}

const WinnerHero = React.memo(({ winner, winner_reason, minPrice }: WinnerHeroProps) => {
  const scores = (winner as any).scores || (winner as any)._scores || {}

  return (
    <Box
      sx={{
        borderRadius: "20px",
        bgcolor: "var(--bg-primary)",
        border: "1px solid var(--border-color)",
        overflow: "hidden",
        boxShadow: "0 20px 40px -15px rgba(0,0,0,0.05)",
      }}
    >
      <Box
        sx={{
          borderBottom: "1px solid var(--border-color)",
          bgcolor: "var(--bg-secondary)",
          px: 3,
          py: 1.5,
          display: "flex",
          alignItems: "center",
          gap: 1.5,
        }}
      >
        <WorkspacePremiumIcon sx={{ fontSize: 18, color: "var(--brand-primary)" }} />
        <Typography sx={{ fontSize: 11, fontWeight: 800, textTransform: "uppercase", letterSpacing: "0.1em", color: "var(--text-primary)" }}>
          Miglior Scelta dell'Agente
        </Typography>
      </Box>

      <Box sx={{ p: { xs: 3, md: 5 }, display: "grid", gridTemplateColumns: { xs: "1fr", md: "220px 1fr" }, gap: 5, alignItems: "center" }}>
        <Box
          sx={{
            width: "100%",
            height: 220,
            bgcolor: "var(--bg-secondary)",
            borderRadius: "16px",
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            p: 0,
            position: "relative",
            border: "1px solid var(--border-color)",
          }}
        >
          {winner.image_url ? (
            <Box
              component="img"
              src={winner.image_url}
              alt={winner.title}
              sx={{ 
                width: "100%", 
                height: "100%", 
                objectFit: "contain", 
                borderRadius: "16px",
                filter: "drop-shadow(0 4px 12px rgba(0,0,0,0.08))"
              }}
            />
          ) : (
            <Typography sx={{ color: "var(--text-secondary)", opacity: 0.3, fontWeight: 600 }}>NO IMAGE</Typography>
          )}
        </Box>

        <Box sx={{ display: "flex", flexDirection: "column" }}>
          <Typography sx={{ color: "var(--text-primary)", fontWeight: 800, fontSize: { xs: 18, md: 22 }, lineHeight: 1.25, mb: 1, letterSpacing: "-0.02em" }}>
            {winner.title}
          </Typography>

          <Box sx={{ display: "flex", alignItems: "baseline", gap: 1.5, mb: 3 }}>
            <Typography sx={{ fontSize: 26, fontWeight: 800, color: "var(--text-primary)" }}>
              {formatPrice(winner.price, winner.currency)}
            </Typography>
            {winner.price === minPrice && (
              <Typography sx={{ fontSize: 11, fontWeight: 800, color: "var(--success)", textTransform: "uppercase", letterSpacing: "0.05em" }}>
                · Miglior Prezzo
              </Typography>
            )}
          </Box>

          <Typography sx={{ fontSize: 13, color: "var(--text-secondary)", lineHeight: 1.6, fontWeight: 500, mb: 4, fontStyle: "italic" }}>
             "{winner_reason}"
          </Typography>

          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 3, mb: 4 }}>
            <ScoreIndicator label="Affinità" score={scores.overall || 0} />
            <ScoreIndicator label="Valore" score={(winner as any).value_score || 0} />
            <ScoreIndicator label="Trust" score={scores.trust || 0} />
          </Box>

          {winner.url && (
            <Button
              component="a"
              href={winner.url}
              target="_blank"
              variant="contained"
              disableElevation
              sx={{
                bgcolor: "var(--brand-primary)",
                color: "var(--bg-primary)",
                textTransform: "none",
                fontWeight: 800,
                borderRadius: "12px",
                py: 1.25,
                px: 4,
                width: "fit-content",
                "&:hover": { bgcolor: "var(--brand-primary)", opacity: 0.9 }
              }}
            >
              Dettagli su eBay
            </Button>
          )}
        </Box>
      </Box>
    </Box>
  )
})

interface ComparisonTableProps {
  matrix: SearchItem[]
  winner: SearchItem
}

const ComparisonTable = React.memo(({ matrix, winner }: ComparisonTableProps) => (
  <Box sx={{ mt: 2 }}>
    <Typography sx={{ fontSize: 10, fontWeight: 800, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.15em", mb: 2.5, px: 1 }}>
      Specifiche a confronto
    </Typography>

    <TableContainer 
      sx={{ 
        borderRadius: "16px", 
        border: "1px solid var(--border-color)", 
        bgcolor: "var(--bg-primary)", 
        overflow: "hidden",
        boxShadow: "0 4px 12px rgba(0,0,0,0.02)"
      }}
    >
      <Table size="small">
        <TableHead>
          <TableRow sx={{ bgcolor: "var(--bg-secondary)" }}>
            <TableCell sx={{ color: "var(--text-secondary)", fontWeight: 800, fontSize: 9, py: 2.5, pl: 3, width: "150px" }}>PARAMETRO</TableCell>
            {matrix.slice(0, 4).map((item, i) => (
              <TableCell key={i} align="center" sx={{ position: "relative", py: 2 }}>
                <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", gap: 1, minWidth: 100 }}>
                   <Box 
                    sx={{ 
                        width: 52, 
                        height: 52, 
                        borderRadius: "16px",
                        overflow: "hidden",
                        border: item.title === winner.title ? "2px solid var(--brand-primary)" : "1px solid var(--border-color)",
                        bgcolor: "var(--bg-secondary)",
                        display: 'flex',
                        alignItems: 'center',
                        justifyContent: 'center',
                        p: 0,
                        boxShadow: '0 2px 8px rgba(0,0,0,0.05)'
                    }}
                   >
                    <Box 
                        component="img"
                        src={item.image_url} 
                        alt={item.title}
                        sx={{ 
                            width: "100%", 
                            height: "100%", 
                            objectFit: "contain",
                            borderRadius: "12px"
                        }} 
                    />
                   </Box>
                   <Typography sx={{ 
                        fontSize: 9, 
                        fontWeight: 800, 
                        color: "var(--text-primary)", 
                        maxWidth: 100,
                        whiteSpace: "nowrap",
                        overflow: "hidden",
                        textOverflow: "ellipsis"
                    }}>
                     {item.title}
                   </Typography>
                </Box>
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell sx={{ fontSize: 10, fontWeight: 800, color: "var(--text-secondary)", pl: 3 }}>PREZZO</TableCell>
            {matrix.slice(0, 4).map((item, i) => (
              <TableCell key={i} align="center" sx={{ fontSize: 12, fontWeight: 800, color: "var(--text-primary)" }}>
                {formatPrice(item.price, item.currency)}
              </TableCell>
            ))}
          </TableRow>
          
          <TableRow sx={{ bgcolor: "var(--bg-secondary)", opacity: 0.9 }}>
            <TableCell sx={{ fontSize: 10, fontWeight: 800, color: "var(--text-secondary)", pl: 3 }}>RAPPORTO QUALITÀ</TableCell>
            {matrix.slice(0, 4).map((item, i) => {
              const itemScores = (item as any).scores || (item as any)._scores || {}
              const value = Math.round((itemScores.value || 0) * 100)
              return (
                <TableCell key={i} align="center">
                  <Typography sx={{ fontWeight: 800, fontSize: 11, color: value >= 80 ? "var(--success)" : "var(--text-primary)" }}>
                    {value}%
                  </Typography>
                </TableCell>
              )
            })}
          </TableRow>

          <TableRow>
            <TableCell sx={{ fontSize: 10, fontWeight: 800, color: "var(--text-secondary)", pl: 3 }}>SPEDIZIONE</TableCell>
            {matrix.slice(0, 4).map((item, i) => (
              <TableCell key={i} align="center">
                {item.shipping?.free ? (
                   <Tooltip title="Spedizione gratuita">
                     <LocalShippingIcon sx={{ fontSize: 16, color: "var(--success)" }} />
                   </Tooltip>
                ) : (
                  <Typography sx={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)" }}>
                    {item.shipping?.cost ? `${item.shipping.cost}€` : "—"}
                  </Typography>
                )}
              </TableCell>
            ))}
          </TableRow>

          <TableRow sx={{ bgcolor: "var(--bg-secondary)", opacity: 0.9 }}>
            <TableCell sx={{ fontSize: 10, fontWeight: 800, color: "var(--text-secondary)", pl: 3 }}>TRUST VENDITORE</TableCell>
            {matrix.slice(0, 4).map((item, i) => (
              <TableCell key={i} align="center">
                <Box sx={{ display: "inline-flex", alignItems: "center", gap: 0.5 }}>
                  <VerifiedUserIcon sx={{ fontSize: 12, color: (item.trust_score || 0) > 0.8 ? "var(--brand-primary)" : "var(--text-secondary)" }} />
                  <Typography sx={{ fontSize: 11, fontWeight: 800, color: "var(--text-primary)" }}>
                    {Math.round((item.trust_score || 0) * 100)}%
                  </Typography>
                </Box>
              </TableCell>
            ))}
          </TableRow>

          <TableRow>
            <TableCell sx={{ fontSize: 10, fontWeight: 900, color: "var(--brand-primary)", pl: 3 }}>AI SCORE TOTALE</TableCell>
            {matrix.slice(0, 4).map((item, i) => {
              const itemScores = (item as any).scores || (item as any)._scores || {}
              return (
                <TableCell key={i} align="center" sx={{ py: 2 }}>
                  <Box
                    sx={{
                        display: "inline-block",
                        px: 1.5,
                        py: 0.5,
                        borderRadius: "8px",
                        bgcolor: item.title === winner.title ? "var(--brand-primary)" : "transparent",
                        color: item.title === winner.title ? "var(--bg-primary)" : "var(--text-primary)",
                        border: "1px solid var(--border-color)",
                    }}
                  >
                    <Typography sx={{ fontSize: 13, fontWeight: 900 }}>
                        {Math.round((itemScores.overall || 0) * 100)}%
                    </Typography>
                  </Box>
                </TableCell>
              )
            })}
          </TableRow>
        </TableBody>
      </Table>
    </TableContainer>
  </Box>
))

// ── Main Component ──────────────────────────────────────────────────

interface ComparisonDisplayProps {
  data: ComparisonData
}

export default function ComparisonDisplay({ data }: ComparisonDisplayProps) {
  const { winner, comparison_matrix, winner_reason } = data
  const scrollRef = useRef<HTMLDivElement>(null)
  const [showLeftScroll, setShowLeftScroll] = useState(false)
  const [showRightScroll, setShowRightScroll] = useState(true)

  const minPrice = useMemo(() => Math.min(...comparison_matrix.map((c: any) => c.price ?? Infinity)), [comparison_matrix])

  const checkScroll = useCallback(() => {
    if (scrollRef.current) {
      const { scrollLeft, scrollWidth, clientWidth } = scrollRef.current
      setShowLeftScroll(scrollLeft > 5)
      setShowRightScroll(scrollLeft < scrollWidth - clientWidth - 5)
    }
  }, [])

  useEffect(() => {
    checkScroll()
    const current = scrollRef.current
    if (current) {
        current.addEventListener("scroll", checkScroll)
        window.addEventListener("resize", checkScroll)
    }
    return () => {
        if (current) current.removeEventListener("scroll", checkScroll)
        window.removeEventListener("resize", checkScroll)
    }
  }, [checkScroll, comparison_matrix])

  const handleScroll = useCallback((direction: "left" | "right") => {
    if (scrollRef.current) {
      const scrollAmount = scrollRef.current.clientWidth * 0.8
      scrollRef.current.scrollBy({
        left: direction === "left" ? -scrollAmount : scrollAmount,
        behavior: "smooth",
      })
    }
  }, [])

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 5, py: 2 }}>
      {/* 1. Winner Highlight */}
      <WinnerHero winner={winner} winner_reason={winner_reason} minPrice={minPrice} />
      
      {/* 2. Carousel */}
      <Box sx={{ position: "relative" }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.25, mb: 1.5, px: 1 }}>
          <Box sx={{ width: 3, height: 14, bgcolor: "var(--brand-primary)", borderRadius: 4, opacity: 0.8 }} />
          <Typography sx={{ fontSize: 10.5, fontWeight: 800, color: "var(--text-primary)", textTransform: "uppercase", letterSpacing: "0.12em" }}>
            Candidati considerati
          </Typography>
        </Box>

        {showLeftScroll && (
          <IconButton
            onClick={() => handleScroll("left")}
            sx={{
              position: "absolute",
              left: -24,
              top: "50%",
              zIndex: 10,
              bgcolor: "var(--bg-primary)",
              border: "1px solid var(--border-color)",
              boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
              color: "var(--brand-primary)",
              "&:hover": { bgcolor: "var(--bg-secondary)", transform: "scale(1.1)" },
              transition: "all 0.2s ease"
            }}
          >
            <ChevronLeftIcon fontSize="small" />
          </IconButton>
        )}

        <Box
          ref={scrollRef}
          onScroll={checkScroll}
          sx={{
            display: "flex",
            gap: 2.5,
            overflowX: "auto",
            scrollbarWidth: "none",
            "&::-webkit-scrollbar": { display: "none" },
            px: 1,
            py: 1.5,
            maskImage: `linear-gradient(to right, 
              transparent, 
              black 20px, 
              black calc(100% - 20px), 
              transparent 100%)`,
            // Fallback for Safari
            WebkitMaskImage: `linear-gradient(to right, 
              transparent, 
              black 20px, 
              black calc(100% - 20px), 
              transparent 100%)`,
          }}
        >
          {comparison_matrix.map((item, idx) => (
            <SearchResultCard
              key={idx}
              item={item}
              variant="compact"
              index={idx}
            />
          ))}
        </Box>

        {showRightScroll && comparison_matrix.length > 2 && (
          <IconButton
            onClick={() => handleScroll("right")}
            sx={{
              position: "absolute",
              right: -24,
              top: "50%",
              zIndex: 10,
              bgcolor: "var(--bg-primary)",
              border: "1px solid var(--border-color)",
              boxShadow: "0 4px 12px rgba(0,0,0,0.2)",
              color: "var(--brand-primary)",
              "&:hover": { bgcolor: "var(--bg-secondary)", transform: "scale(1.1)" },
              transition: "all 0.2s ease"
            }}
          >
            <ChevronRightIcon fontSize="small" />
          </IconButton>
        )}
      </Box>

      {/* 3. Matrix */}
      <ComparisonTable matrix={comparison_matrix} winner={winner} />
    </Box>
  )
}
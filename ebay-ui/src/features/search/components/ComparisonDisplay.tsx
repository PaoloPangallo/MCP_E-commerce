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
  Chip,
  Button,
  IconButton,
} from "@mui/material"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import LocalShippingIcon from "@mui/icons-material/LocalShipping"
import WorkspacePremiumIcon from "@mui/icons-material/WorkspacePremium"
import TrendingUpIcon from "@mui/icons-material/TrendingUp"
import SpeedIcon from "@mui/icons-material/Speed"
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft"
import ChevronRightIcon from "@mui/icons-material/ChevronRight"
import SearchResultCard from "./SearchResultCard"

import type { ComparisonData, SearchItem } from "../types"

// ── Helpers ─────────────────────────────────────────────────────────

function formatPrice(price?: number, currency?: string) {
  if (typeof price !== "number") return "—"
  const formatted = price.toLocaleString('it-IT', { minimumFractionDigits: 2, maximumFractionDigits: 2 });
  return `${formatted} ${currency ?? ""}`.trim()
}

// ── Sub-Components (Memoized) ───────────────────────────────────────

interface ScoreItemProps {
  label: string
  score: number
  color: string
  icon?: React.ReactNode
}

const ScoreItem = React.memo(({ label, score, color, icon }: ScoreItemProps) => {
  const pct = Math.round(score * 100)
  return (
    <Box sx={{ flex: 1 }}>
      <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mb: 0.5 }}>
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
          {icon}
          <Typography sx={{ fontSize: 10, color: "var(--text-secondary)", fontWeight: 500, textTransform: "uppercase", letterSpacing: "0.02em" }}>
            {label}
          </Typography>
        </Box>
        <Typography sx={{ fontSize: 10, fontWeight: 700, color: color }}>{pct}%</Typography>
      </Box>
      <Box
        sx={{
          width: "100%",
          height: 6,
          bgcolor: "var(--bg-secondary)",
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
        borderRadius: "16px",
        position: "relative",
        bgcolor: "var(--bg-primary)",
        border: "1px solid var(--border-color)",
        boxShadow: "0 10px 25px -5px rgba(0,0,0,0.1), 0 8px 10px -6px rgba(0,0,0,0.05)",
        overflow: "hidden",
      }}
    >
      <Box
        sx={{
          background: "var(--brand-gradient)",
          color: "#fff",
          px: 2.5,
          py: 1,
          display: "flex",
          alignItems: "center",
          gap: 1,
        }}
      >
        <WorkspacePremiumIcon sx={{ fontSize: 18, color: "#fff" }} />
        <Typography sx={{ fontSize: 13, fontWeight: 700, textTransform: "uppercase", letterSpacing: "0.05em", color: "#fff" }}>
          Miglior Scelta dell'Agente
        </Typography>
      </Box>

      <Box sx={{ p: { xs: 2.5, sm: 4 }, display: "grid", gridTemplateColumns: { xs: "1fr", md: "180px 1fr" }, gap: 4 }}>
        <Box
          sx={{
            width: "100%",
            height: 180,
            bgcolor: "var(--bg-primary)",
            borderRadius: 3,
            border: "1px solid var(--border-color)",
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
            <Typography sx={{ color: "var(--text-secondary)", opacity: 0.5 }}>No Image</Typography>
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
          <Typography variant="h6" sx={{ color: "var(--text-primary)", fontWeight: 700, lineHeight: 1.3, mb: 1.5 }}>
            {winner.title}
          </Typography>

          <Box sx={{ display: "flex", alignItems: "baseline", gap: 1.5, mb: 2 }}>
            <Typography sx={{ fontSize: 28, fontWeight: 800, color: "var(--text-primary)" }}>
              {formatPrice(winner.price, winner.currency)}
            </Typography>
            {winner.price === minPrice && (
              <Chip label="Miglior Prezzo" size="small" variant="outlined" sx={{ height: 20, fontSize: 10, fontWeight: 700, color: 'var(--success)', borderColor: 'var(--success)' }} />
            )}
          </Box>

          <Box
            sx={{
              p: 2,
              borderRadius: 2,
              bgcolor: "var(--bg-secondary)",
              borderLeft: "4px solid var(--brand-primary)",
              mb: 3
            }}
          >
            <Typography sx={{ fontSize: 13, color: "var(--text-primary)", lineHeight: 1.6, fontWeight: 500 }}>
              {winner_reason}
            </Typography>
          </Box>

          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 2, mb: 3 }}>
            <ScoreItem label="Match Affinità" score={scores.overall || 0} color="var(--brand-primary)" icon={<TrendingUpIcon sx={{ fontSize: 14 }} />} />
            <ScoreItem label="Value Score" score={(winner as any).value_score || 0} color="var(--success)" icon={<WorkspacePremiumIcon sx={{ fontSize: 14 }} />} />
            <ScoreItem label="Prezzo" score={scores.price || 0} color="var(--brand-primary)" icon={<SpeedIcon sx={{ fontSize: 14 }} />} />
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
                  background: "var(--brand-gradient)",
                  color: "#ffffff",
                  textTransform: "none",
                  fontWeight: 700,
                  borderRadius: "24px", // Rounded as requested
                  py: 1,
                  "&:hover": { opacity: 0.9 }
                }}
              >
                Acquista ora su eBay
              </Button>
            )}
          </Box>
        </Box>
      </Box>
    </Box>
  )
})

interface AlternativesProps {
  matrix: SearchItem[]
  winner: SearchItem
  minPrice: number
  scrollRef: React.RefObject<HTMLDivElement | null>
  showLeft: boolean
  showRight: boolean
  onScroll: () => void
  onScrollClick: (dir: "left" | "right") => void
}

const AlternativesCarousel = React.memo(({ matrix, scrollRef, showLeft, showRight, onScroll, onScrollClick }: AlternativesProps) => (
  <Box sx={{ position: "relative" }}>
    <Typography sx={{ fontSize: 12, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.1em", mb: 2, px: 1 }}>
      Tutte le Alternative
    </Typography>

    {showLeft && (
      <IconButton
        onClick={() => onScrollClick("left")}
        sx={{
          position: "absolute",
          left: -20,
          top: "55%",
          zIndex: 10,
          bgcolor: "var(--bg-primary)",
          border: "1px solid var(--border-color)",
          boxShadow: "var(--card-shadow)",
          color: "var(--brand-primary)",
          "&:hover": { bgcolor: "var(--bg-secondary)", borderColor: "var(--brand-primary)" }
        }}
      >
        <ChevronLeftIcon />
      </IconButton>
    )}

    <Box
      ref={scrollRef}
      onScroll={onScroll}
      sx={{
        display: "flex",
        gap: 2.5,
        overflowX: "auto",
        scrollbarWidth: "none",
        "&::-webkit-scrollbar": { display: "none" },
        px: 1,
        py: 1
      }}
    >
      {matrix.map((item, idx) => (
        <SearchResultCard
          key={idx}
          item={item}
          variant="compact"
          index={idx}
        />
      ))}
    </Box>

    {showRight && matrix.length > 1 && (
      <IconButton
        onClick={() => onScrollClick("right")}
        sx={{
          position: "absolute",
          right: -20,
          top: "55%",
          zIndex: 10,
          bgcolor: "var(--bg-primary)",
          border: "1px solid var(--border-color)",
          boxShadow: "var(--card-shadow)",
          color: "var(--brand-primary)",
          "&:hover": { bgcolor: "var(--bg-secondary)", borderColor: "var(--brand-primary)" }
        }}
      >
        <ChevronRightIcon />
      </IconButton>
    )}
  </Box>
))

interface TableProps {
  matrix: SearchItem[]
  winner: SearchItem
  maxOverall: number
  maxValue: number
}

const ComparisonTable = React.memo(({ matrix, winner, maxOverall, maxValue }: TableProps) => (
  <Box>
    <Typography sx={{ fontSize: 12, fontWeight: 700, color: "var(--text-secondary)", textTransform: "uppercase", letterSpacing: "0.1em", mb: 2, px: 1 }}>
      Tabella Comparativa
    </Typography>

    <TableContainer sx={{ borderRadius: "16px", border: "1px solid var(--border-color)", bgcolor: "var(--bg-primary)", overflow: "hidden" }}>
      <Table size="small">
        <TableHead>
          <TableRow sx={{ bgcolor: "var(--bg-secondary)" }}>
            <TableCell sx={{ color: "var(--text-secondary)", fontWeight: 700, fontSize: 11, py: 2 }}>FEATURE</TableCell>
            {matrix.slice(0, 4).map((item, i) => (
              <TableCell key={i} align="center" sx={{ fontWeight: 800, color: item.title === winner.title ? "var(--brand-primary)" : "var(--text-primary)", fontSize: 11 }}>
                {item.title?.slice(0, 15)}...
              </TableCell>
            ))}
          </TableRow>
        </TableHead>
        <TableBody>
          <TableRow>
            <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)" }}>PREZZO</TableCell>
            {matrix.slice(0, 4).map((item, i) => (
              <TableCell key={i} align="center" sx={{ fontSize: 13, fontWeight: 700, color: "var(--text-primary)" }}>
                {formatPrice(item.price, item.currency)}
              </TableCell>
            ))}
          </TableRow>
          <TableRow sx={{ bgcolor: "var(--bg-secondary)", opacity: 0.8 }}>
            <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)" }}>VALORE / PREZZO</TableCell>
            {matrix.slice(0, 4).map((item, i) => {
              const itemScores = (item as any).scores || (item as any)._scores || {}
              return (
                <TableCell key={i} align="center">
                  <Typography sx={{ fontWeight: 700, fontSize: 12, color: itemScores.value === maxValue && maxValue > 0 ? "var(--success)" : "var(--text-primary)" }}>
                    {Math.round((itemScores.value || 0) * 100)}%
                  </Typography>
                </TableCell>
              )
            })}
          </TableRow>
          <TableRow>
            <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)" }}>SPEDIZIONE</TableCell>
            {matrix.slice(0, 4).map((item, i) => (
              <TableCell key={i} align="center">
                <Typography sx={{ fontSize: 11, fontWeight: 600, color: item.shipping?.free ? "var(--success)" : "var(--text-primary)" }}>
                  {item.shipping?.free ? "GRATIS" : item.shipping?.cost ? `${item.shipping.cost} €` : "—"}
                </Typography>
              </TableCell>
            ))}
          </TableRow>
          <TableRow sx={{ bgcolor: "var(--bg-secondary)", opacity: 0.8 }}>
            <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)" }}>AFFIDABILITÀ</TableCell>
            {matrix.slice(0, 4).map((item, i) => (
              <TableCell key={i} align="center">
                <Box sx={{ display: "flex", alignItems: "center", justifyContent: "center", gap: 0.5 }}>
                  <VerifiedUserIcon sx={{ fontSize: 12, color: (item.trust_score || 0) > 0.9 ? "var(--success)" : "var(--text-secondary)" }} />
                  <Typography sx={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)" }}>{Math.round((item.trust_score || 0) * 100)}%</Typography>
                </Box>
              </TableCell>
            ))}
          </TableRow>
          <TableRow>
            <TableCell sx={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", border: "none" }}>AI SCORE</TableCell>
            {matrix.slice(0, 4).map((item, i) => {
              const itemScores = (item as any).scores || (item as any)._scores || {}
              return (
                <TableCell key={i} align="center" sx={{ border: "none" }}>
                  <Typography sx={{ fontSize: 13, fontWeight: 800, color: itemScores.overall === maxOverall && maxOverall > 0 ? "var(--brand-primary)" : "var(--text-secondary)" }}>
                    {Math.round((itemScores.overall || 0) * 100)}%
                  </Typography>
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
  const maxOverall = useMemo(() => Math.max(...comparison_matrix.map((c: any) => (c.scores || c._scores || {}).overall ?? 0)), [comparison_matrix])
  const maxValue = useMemo(() => Math.max(...comparison_matrix.map((c: any) => (c.scores || c._scores || {}).value ?? 0)), [comparison_matrix])

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
    <Box sx={{ display: "flex", flexDirection: "column", gap: 4, py: 1 }}>
      <WinnerHero winner={winner} winner_reason={winner_reason} minPrice={minPrice} />
      
      <AlternativesCarousel
        matrix={comparison_matrix}
        winner={winner}
        minPrice={minPrice}
        scrollRef={scrollRef}
        showLeft={showLeftScroll}
        showRight={showRightScroll}
        onScroll={checkScroll}
        onScrollClick={handleScroll}
      />

      <ComparisonTable
        matrix={comparison_matrix}
        winner={winner}
        maxOverall={maxOverall}
        maxValue={maxValue}
      />
    </Box>
  )
}
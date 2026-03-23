import { useEffect, useMemo, useState } from "react"
import { Box, Button, Typography } from "@mui/material"
import TrendingUpIcon from "@mui/icons-material/TrendingUp"
import GppGoodIcon from "@mui/icons-material/GppGood"
import ListIcon from "@mui/icons-material/List"

import SearchResultCard from "./SearchResultCard"
import FilterSidebar from "./FilterSidebar"
import type { SearchItem } from "../types"

interface Props {
  results?: SearchItem[]
  aspect_distributions?: any[]
}

function getTopTrust(results: SearchItem[]) {
  const values = results
    .map((item) => item.trust_score)
    .filter((v): v is number => typeof v === "number")
  return values.length ? Math.max(...values) : null
}

export default function SearchResultList({
  results = [],
  aspect_distributions = [],
}: Props) {
  const [visibleCount, setVisibleCount] = useState(5)
  const safeResults = useMemo(() => {
    const seen = new Set<string>()
    return (results || []).filter((item) => {
      if (!item) return false
      if (!item.ebay_id) return true // Keep if no ID (prevent loss of generic results)
      if (seen.has(item.ebay_id)) return false
      seen.add(item.ebay_id)
      return true
    })
  }, [results])

  useEffect(() => {
    setVisibleCount(5)
  }, [safeResults])

  const visibleResults = safeResults.slice(0, visibleCount)
  const topTrust = getTopTrust(safeResults)
  const remaining = safeResults.length - visibleCount

  if (safeResults.length === 0) {
    return (
      <Box sx={{ py: 4, px: 2.5, textAlign: "center" }}>
        <Typography sx={{ fontSize: 13.5, fontWeight: 500, color: "#6b7280", mb: 0.4 }}>
          Nessun risultato trovato
        </Typography>
        <Typography sx={{ fontSize: 12.5, color: "#b0b0b0" }}>
          Prova a cambiare brand, fascia di prezzo o parole chiave.
        </Typography>
      </Box>
    )
  }

  const handleFilterClick = (aspectName: string, value: string) => {
    window.dispatchEvent(
      new CustomEvent("send-chat", {
        detail: `Cerca ${aspectName} ${value} per i risultati correnti`,
      })
    )
  }

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>

      {/* ── Filters ─────────────────────────────────────────────────────── */}
      {aspect_distributions.length > 0 && (
        <Box sx={{ px: 2, pt: 1.5, pb: 1.25, borderBottom: "1px solid #f5f5f5" }}>
          <FilterSidebar
            distributions={aspect_distributions}
            onFilterClick={handleFilterClick}
          />
        </Box>
      )}

      {/* ── Summary line — one quiet row, no heavy chrome ─────────────── */}
      <Box
        sx={{
          px: 2,
          py: 1.25,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          bgcolor: "var(--bg-secondary)",
          borderBottom: "1px solid var(--border-color)",
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <ListIcon sx={{ fontSize: 14, color: "#9ca3af" }} />
            <Typography sx={{ fontSize: 12, fontWeight: 600, color: "#4b5563" }}>
              {safeResults.length} {safeResults.length === 1 ? "risultato" : "risultati"}
            </Typography>
          </Box>

          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <TrendingUpIcon sx={{ fontSize: 14, color: "#8b5cf6" }} />
            <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#6b7280" }}>
              AI Sorted
            </Typography>
          </Box>
        </Box>

        {topTrust !== null && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <GppGoodIcon sx={{ fontSize: 14, color: "#10b981" }} />
            <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#10b981" }}>
              Trust {Math.round(topTrust * 100)}% max
            </Typography>
          </Box>
        )}
      </Box>

      {/* ── Cards ─────────────────────────────────────────────────────── */}
      <Box sx={{ px: 2 }}>
        {visibleResults.map((item, index) => (
          <SearchResultCard
            key={item.ebay_id ?? `${index}-${item.title}`}
            item={item}
          />
        ))}
      </Box>

      {/* ── Load more — plain text link, no button ────────────────────── */}
      {remaining > 0 && (
        <Box
          sx={{
            p: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "center",
            borderTop: "1px solid var(--border-color)",
            bgcolor: "var(--bg-primary)"
          }}
        >
          <Button
            variant="outlined"
            onClick={() => setVisibleCount((prev) => Math.min(prev + 5, safeResults.length))}
            fullWidth
            sx={{
              textTransform: "none",
              borderRadius: "10px",
              borderColor: "#e5e7eb",
              color: "#374151",
              fontSize: 13,
              fontWeight: 600,
              py: 1,
              "&:hover": {
              bgcolor: "var(--bg-secondary)",
              borderColor: "var(--border-color)"
            }
            }}
          >
            Mostra altri {Math.min(remaining, 5)} prodotti ({remaining} rimanenti)
          </Button>
        </Box>
      )}
    </Box>
  )
}
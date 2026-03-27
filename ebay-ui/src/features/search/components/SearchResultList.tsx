import { useState, useMemo, useEffect } from "react"
import { Box, Typography, Button } from "@mui/material"
import TrendingUpIcon from "@mui/icons-material/TrendingUp"
import GppGoodIcon from "@mui/icons-material/GppGood"
import ListIcon from "@mui/icons-material/List"
import CompareArrowsIcon from "@mui/icons-material/CompareArrows"

import SearchResultCard from "./SearchResultCard"
import FilterSidebar from "./FilterSidebar"
import type { SearchItem } from "../types"

interface Props {
  results?: SearchItem[]
  aspect_distributions?: any[]
  selectable?: boolean
  selectedIds?: string[]
  onSelect?: (id: string) => void
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
  selectable = false,
  selectedIds = [],
  onSelect
}: Props) {
  const [visibleCount, setVisibleCount] = useState(5)

  const safeResults = useMemo(() => {
    const seen = new Set<string>()
    return (results || []).filter((item) => {
      if (!item) return false
      if (!item.ebay_id) return true
      if (seen.has(item.ebay_id)) return false
      seen.add(item.ebay_id)
      return true
    })
  }, [results])

  useEffect(() => {
    setVisibleCount(5)
  }, [safeResults])

  const topTrust = getTopTrust(safeResults)
  const visibleResults = safeResults.slice(0, visibleCount)
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
        <Box sx={{ px: 2, pt: 1, pb: 1, borderBottom: "1px solid var(--border-color)" }}>
          <FilterSidebar
            distributions={aspect_distributions}
            onFilterClick={handleFilterClick}
          />
        </Box>
      )}

      {/* ── Summary Row ─────────────────────────────────────────────── */}
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
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75 }}>
            <ListIcon sx={{ fontSize: 14, color: "var(--text-secondary)" }} />
            <Typography sx={{ fontSize: 12, fontWeight: 700, color: "var(--text-primary)" }}>
              {safeResults.length} RISULTATI
            </Typography>
          </Box>

          <Button
            size="small"
            onClick={() => onSelect?.("__TOGGLE_MODE__")}
            startIcon={<CompareArrowsIcon sx={{ fontSize: 14 }} />}
            sx={{
              textTransform: 'none',
              fontSize: 11,
              fontWeight: 700,
              color: selectable ? "var(--brand-primary)" : "var(--text-secondary)",
              bgcolor: selectable ? "var(--brand-soft)" : "transparent",
              borderRadius: "6px",
              px: 1,
              "&:hover": { bgcolor: selectable ? "var(--brand-soft)" : "rgba(0,0,0,0.05)" }
            }}
          >
            {selectable ? "SELEZIONE ATTIVA" : "CONFRONTA"}
          </Button>
        </Box>

        {topTrust !== null && (
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
            <GppGoodIcon sx={{ fontSize: 14, color: "var(--success)" }} />
            <Typography sx={{ fontSize: 12, fontWeight: 700, color: "var(--success)" }}>
              Trust {Math.round(topTrust * 100)}% max
            </Typography>
          </Box>
        )}
      </Box>

      {/* ── Vertical List ─────────────────────────────────────────────── */}
      <Box sx={{ px: 2 }}>
        {visibleResults.map((item, index) => (
          <SearchResultCard
            key={item.ebay_id ?? `${index}-${item.title}`}
            item={item}
            variant="list"
            index={index}
            selectable={selectable}
            isSelected={selectedIds.includes(item.ebay_id || "")}
            onSelect={onSelect}
          />
        ))}
      </Box>

      {/* ── Load More ─────────────────────────────────────────────────── */}
      {remaining > 0 && (
        <Box sx={{ p: 2, display: 'flex', justifyContent: 'center', borderTop: '1px solid var(--border-color)' }}>
           <Button
             variant="outlined"
             fullWidth
             onClick={() => setVisibleCount(prev => prev + 5)}
             sx={{
               textTransform: 'none',
               borderRadius: '10px',
               borderColor: 'var(--border-color)',
               color: 'var(--text-primary)',
               fontWeight: 600,
               py: 1,
               "&:hover": { bgcolor: 'var(--bg-secondary)', borderColor: 'var(--brand-primary)' }
             }}
           >
             Mostra altri {Math.min(remaining, 5)} prodotti ({remaining} rimanenti)
           </Button>
        </Box>
      )}

    </Box>
  )
}
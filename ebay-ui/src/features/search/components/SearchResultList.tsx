import { useEffect, useMemo, useState } from "react"
import { Box, Typography } from "@mui/material"

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
  aspect_distributions = []
}: Props) {
  const [visibleCount, setVisibleCount] = useState(5)
  const safeResults = useMemo(() => results.filter(Boolean), [results])

  useEffect(() => {
    setVisibleCount(5)
  }, [safeResults])

  const visibleResults = safeResults.slice(0, visibleCount)
  const topTrust = getTopTrust(safeResults)

  if (safeResults.length === 0) {
    return (
      <Box sx={{ py: 5, px: 3, textAlign: "center" }}>
        <Typography sx={{ fontSize: 14, fontWeight: 500, color: "#374151", mb: 0.5 }}>
          Nessun risultato trovato
        </Typography>
        <Typography sx={{ fontSize: 13, color: "#9ca3af" }}>
          Prova a cambiare brand, fascia di prezzo o parole chiave.
        </Typography>
      </Box>
    )
  }

  const handleFilterClick = (aspectName: string, value: string) => {
    window.dispatchEvent(
      new CustomEvent("send-chat", {
        detail: `Cerca ${aspectName} ${value} per i risultati correnti`
      })
    )
  }

  return (
    <Box sx={{ display: "flex", flexDirection: "column" }}>
      {/* Filters */}
      {aspect_distributions.length > 0 && (
        <Box sx={{ p: 2, borderBottom: "1px solid #f5f5f5" }}>
          <FilterSidebar
            distributions={aspect_distributions}
            onFilterClick={handleFilterClick}
          />
        </Box>
      )}

      {/* Summary row */}
      <Box
        sx={{
          px: 2,
          py: 1.25,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: "1px solid #f5f5f5"
        }}
      >
        <Typography sx={{ fontSize: 12, color: "#9ca3af" }}>
          {safeResults.length} {safeResults.length === 1 ? "risultato" : "risultati"} · ordinati per AI relevance
        </Typography>
        {topTrust !== null && (
          <Typography sx={{ fontSize: 12, color: "#9ca3af" }}>
            trust max {Math.round(topTrust * 100)}%
          </Typography>
        )}
      </Box>

      {/* Cards */}
      <Box sx={{ px: 2 }}>
        {visibleResults.map((item, index) => (
          <SearchResultCard
            key={item.ebay_id ?? `${index}-${item.title}`}
            item={item}
          />
        ))}
      </Box>

      {/* Pagination */}
      {visibleCount < safeResults.length && (
        <Box
          sx={{
            px: 2,
            py: 1.5,
            borderTop: "1px solid #f5f5f5",
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between"
          }}
        >
          <Typography sx={{ fontSize: 12, color: "#9ca3af" }}>
            {Math.min(visibleCount, safeResults.length)} di {safeResults.length}
          </Typography>
          <Box
            component="button"
            onClick={() =>
              setVisibleCount((prev) => Math.min(prev + 5, safeResults.length))
            }
            sx={{
              background: "none",
              border: "1px solid #e5e7eb",
              borderRadius: "20px",
              px: 1.5,
              py: 0.5,
              fontSize: 12,
              color: "#6b7280",
              cursor: "pointer",
              fontFamily: "inherit",
              "&:hover": { bgcolor: "#f9fafb", borderColor: "#d1d5db" }
            }}
          >
            Mostra altri
          </Box>
        </Box>
      )}
    </Box>
  )
}
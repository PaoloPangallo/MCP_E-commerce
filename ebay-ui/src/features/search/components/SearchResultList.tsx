import { useEffect, useMemo, useState } from "react"
import { Box, Button, Chip, Typography } from "@mui/material"
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

import SearchResultCard from "./SearchResultCard"
import type { SearchItem } from "../types"

interface Props {
  results?: SearchItem[]
}

function getTopTrust(results: SearchItem[]) {
  const values = results.map((item) => item.trust_score).filter((value): value is number => typeof value === "number")
  if (values.length === 0) return null
  return Math.max(...values)
}

export default function SearchResultList({ results = [] }: Props) {
  const [visibleCount, setVisibleCount] = useState(5)
  const safeResults = useMemo(() => results.filter(Boolean), [results])

  useEffect(() => {
    setVisibleCount(5)
  }, [safeResults])

  const visibleResults = safeResults.slice(0, visibleCount)
  const topTrust = getTopTrust(safeResults)

  if (safeResults.length === 0) {
    return (
      <Box sx={{ textAlign: "center", border: "1px solid #e5e7eb", bgcolor: "#ffffff", borderRadius: 4, py: 6, px: 3 }}>
        <Typography sx={{ fontSize: 16, fontWeight: 700, color: "#111827" }}>Nessun risultato trovato</Typography>
        <Typography sx={{ mt: 1, fontSize: 13.5, color: "#6b7280", lineHeight: 1.65 }}>Prova a cambiare brand, fascia di prezzo o parole chiave.</Typography>
      </Box>
    )
  }

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
      <Box sx={{ border: "1px solid #e5e7eb", bgcolor: "#ffffff", borderRadius: 4, px: 2.25, py: 1.75 }}>
        <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 2, flexWrap: "wrap" }}>
          <Box>
            <Typography sx={{ fontSize: 14, fontWeight: 700, color: "#111827" }}>Risultati ordinati per AI relevance</Typography>
            <Typography sx={{ mt: 0.5, fontSize: 13, color: "#6b7280", lineHeight: 1.65 }}>{safeResults.length} {safeResults.length === 1 ? "risultato analizzato" : "risultati analizzati"}</Typography>
          </Box>
          <Box display="flex" gap={1} flexWrap="wrap">
            {safeResults[0]?.ranking_score && safeResults[0].ranking_score > 0.7 ? <Chip icon={<EmojiEventsIcon sx={{ fontSize: 16 }} />} label="best match in testa" size="small" sx={{ bgcolor: "#fff8e6", border: "1px solid #fde68a", color: "#92400e" }} /> : null}
            {typeof topTrust === "number" ? <Chip icon={<AutoAwesomeIcon sx={{ fontSize: 16 }} />} label={`top trust ${Math.round(topTrust * 100)}%`} size="small" sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb", color: "#374151" }} /> : null}
          </Box>
        </Box>
      </Box>

      {visibleResults.map((item, index) => {
        const key = item.ebay_id ?? `${index}-${item.title}`
        return (
          <Box key={key}>
            {index === 0 && (item.ranking_score ?? 0) > 0.7 ? <Box sx={{ mb: 1 }}><Chip icon={<EmojiEventsIcon sx={{ fontSize: 16 }} />} label="AI Best Match" size="small" sx={{ bgcolor: "#fff8e6", border: "1px solid #fde68a", color: "#92400e", fontWeight: 700 }} /></Box> : null}
            <SearchResultCard item={item} />
          </Box>
        )
      })}

      {visibleCount < safeResults.length ? (
        <Box sx={{ display: "flex", justifyContent: "center", pt: 0.5 }}>
          <Button variant="outlined" onClick={() => setVisibleCount((prev) => Math.min(prev + 5, safeResults.length))} sx={{ textTransform: "none", borderRadius: 999, px: 2 }}>
            Mostra altri risultati
          </Button>
        </Box>
      ) : null}

      {safeResults.length > 5 ? <Typography sx={{ textAlign: "center", fontSize: 12.5, color: "#6b7280", pt: 0.25 }}>Mostrati {Math.min(visibleCount, safeResults.length)} di {safeResults.length} articoli</Typography> : null}
    </Box>
  )
}

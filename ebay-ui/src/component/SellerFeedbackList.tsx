import { useEffect, useMemo, useState } from "react"
import { Box, Button, Chip, Divider, Typography } from "@mui/material"
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents"

import SearchResultCard from "./SearchResultCard"
import type { SearchItem } from "../component/searchTypes.ts"

interface Props {
  results?: SearchItem[]
}

export default function SearchResultList({ results = [] }: Props) {
  const [visibleCount, setVisibleCount] = useState(6)

  const safeResults = useMemo(() => results.filter(Boolean), [results])

  useEffect(() => {
    setVisibleCount(6)
  }, [safeResults])

  const visibleResults = safeResults.slice(0, visibleCount)

  if (safeResults.length === 0) {
    return (
      <Box sx={{ mt: 8, textAlign: "center", color: "#6e6e80" }}>
        <Typography sx={{ fontSize: 16, fontWeight: 500 }}>
          Nessun risultato trovato
        </Typography>

        <Typography variant="caption" sx={{ mt: 1, display: "block", color: "#8e8ea0" }}>
          Prova a cambiare brand, fascia di prezzo o parole chiave.
        </Typography>
      </Box>
    )
  }

  return (
    <Box sx={{ mt: 3, display: "flex", flexDirection: "column" }}>
      <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 3 }}>
        <Typography sx={{ fontSize: 14, color: "#6e6e80", fontWeight: 500 }}>
          {safeResults.length} {safeResults.length === 1 ? "risultato" : "risultati"} · ordinati per AI relevance
        </Typography>
      </Box>

      {visibleResults.map((item, index) => {
        const key = item.ebay_id ?? `${index}-${item.title}`
        const ranking = item.ranking_score ?? 0

        return (
          <Box key={key} sx={{ position: "relative" }}>
            {index === 0 && ranking > 0.7 && (
              <Box sx={{ position: "absolute", top: -12, left: 16, zIndex: 1 }}>
                <Chip
                  icon={<EmojiEventsIcon sx={{ fontSize: 16 }} />}
                  label="AI Best Match"
                  size="small"
                  sx={{
                    bgcolor: "#ffd700",
                    color: "#856404",
                    fontWeight: 700,
                    fontSize: 11,
                    height: 24,
                    boxShadow: "0 2px 4px rgba(0,0,0,0.1)",
                    "& .MuiChip-icon": { color: "#856404" }
                  }}
                />
              </Box>
            )}

            <SearchResultCard item={item} />

            {index < visibleResults.length - 1 && <Divider sx={{ my: 1.5, borderColor: "#ececf1" }} />}
          </Box>
        )
      })}

      {visibleCount < safeResults.length && (
        <Box sx={{ textAlign: "center", mt: 2 }}>
          <Button
            variant="outlined"
            onClick={() => setVisibleCount(prev => Math.min(prev + 6, safeResults.length))}
            sx={{ textTransform: "none", borderRadius: 999 }}
          >
            Mostra altri risultati
          </Button>
        </Box>
      )}

      {safeResults.length > 5 && (
        <Box sx={{ mt: 4, pt: 3, borderTop: "1px solid #ececf1", textAlign: "center" }}>
          <Typography sx={{ fontSize: 13, color: "#8e8ea0" }}>
            Mostrati {Math.min(visibleCount, safeResults.length)} di {safeResults.length} articoli analizzati
          </Typography>
        </Box>
      )}
    </Box>
  )
}

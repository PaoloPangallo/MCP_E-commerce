import { Box, Typography, Divider, Chip } from "@mui/material";
import SearchResultCard from "./SearchResultCard";
import EmojiEventsIcon from "@mui/icons-material/EmojiEvents";

interface SearchItem {
  ebay_id?: string
  title?: string
  price?: number
  currency?: string
  image_url?: string
  url?: string
  seller_name?: string
  seller_rating?: number
  trust_score?: number
  _rerank_score?: number
}

export default function SearchResultList({
  results = [],
}: {
  results?: SearchItem[]
}) {

  if (!results || results.length === 0) {
    return (
      <Box
        sx={{
          mt: 8,
          textAlign: "center",
          color: "#6e6e80",
        }}
      >
        <Typography
          sx={{
            fontSize: 16,
            fontWeight: 500,
          }}
        >
          Nessun risultato trovato
        </Typography>

        <Typography
          variant="caption"
          sx={{
            mt: 1,
            display: "block",
            color: "#8e8ea0",
          }}
        >
          Prova a modificare la ricerca
        </Typography>
      </Box>
    )
  }

  return (
    <Box
      sx={{
        mt: 3,
        display: "flex",
        flexDirection: "column",
      }}
    >

      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 3,
        }}
      >

        <Typography
          sx={{
            fontSize: 14,
            color: "#6e6e80",
            fontWeight: 500,
          }}
        >
          {results.length} {results.length === 1 ? "risultato" : "risultati"} · ordinati per AI relevance
        </Typography>

      </Box>


      {/* Lista */}
      {results.map((item, index) => {

        const key = item.ebay_id ?? `${index}-${item.title}`

        return (
          <Box key={key} sx={{ position: "relative" }}>

            {/* AI Best Match */}
            {index === 0 && item._rerank_score && item._rerank_score > 0.7 && (
              <Box
                sx={{
                  position: "absolute",
                  top: -12,
                  left: 16,
                  zIndex: 1,
                }}
              >
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
                    "& .MuiChip-icon": {
                      color: "#856404",
                    },
                  }}
                />
              </Box>
            )}

            <SearchResultCard
              item={item}
              best={index === 0}
            />

            {/* Divider ogni 3 */}
            {index < results.length - 1 && (index + 1) % 3 === 0 && (
              <Divider
                sx={{
                  my: 3,
                  borderColor: "#ececf1",
                }}
              />
            )}

          </Box>
        )

      })}

      {/* Footer */}
      {results.length > 5 && (
        <Box
          sx={{
            mt: 4,
            pt: 3,
            borderTop: "1px solid #ececf1",
            textAlign: "center",
          }}
        >
          <Typography
            sx={{
              fontSize: 13,
              color: "#8e8ea0",
            }}
          >
            Fine risultati · {results.length} articoli analizzati
          </Typography>
        </Box>
      )}

    </Box>
  )
}
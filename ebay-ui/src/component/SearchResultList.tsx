import { Box, Typography } from "@mui/material";
import SearchResultCard from "./SearchResultCard";

interface SearchItem {
  ebay_id?: string;
  title?: string;
  price?: number;
  currency?: string;
  image_url?: string;
  url?: string;
  seller_name?: string;
  seller_rating?: number;
  trust_score?: number;
}

export default function SearchResultList({
  results = [],
}: {
  results?: SearchItem[];
}) {

  if (results.length === 0) {
    return (
      <Box
        sx={{
          mt: 3,
          textAlign: "center",
          color: "#6e6e80",
        }}
      >
        <Typography>Nessun risultato trovato</Typography>
      </Box>
    );
  }

  return (
    <Box
      sx={{
        mt: 2,
        display: "flex",
        flexDirection: "column",
      }}
    >
      {results
        .filter(Boolean)
        .map((item, i) => (
          <SearchResultCard
            key={item.ebay_id ?? i}
            item={item}
          />
        ))}
    </Box>
  );
}
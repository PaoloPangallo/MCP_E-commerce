import { Box, Typography, Chip, Tooltip } from "@mui/material"
import { useSettingsStore } from "./store/settingsStore"
import LocalOfferIcon from "@mui/icons-material/LocalOffer"
import MonetizationOnIcon from "@mui/icons-material/MonetizationOn"

export default function ProfileBadge() {
  const { settings } = useSettingsStore()
  
  const isEmpty = !settings.favoriteBrands && !settings.pricePreference
  
  if (isEmpty) {
    return (
      <Box
        sx={{
          mx: 2,
          mt: 1,
          mb: 2,
          p: 2,
          borderRadius: 2,
          bgcolor: "transparent",
          border: "1px dashed var(--border-color)",
          display: "flex",
          flexDirection: "column",
          alignItems: "center",
          gap: 1,
          opacity: 0.6,
          transition: "all 0.2s ease",
          "&:hover": { bgcolor: "rgba(0,0,0,0.02)", opacity: 1 }
        }}
      >
        <MonetizationOnIcon sx={{ fontSize: 20, color: "var(--text-secondary)" }} />
        <Typography sx={{ fontSize: 11, fontWeight: 600, color: "var(--text-secondary)", textAlign: "center" }}>
          Configura il tuo profilo shopping per ricerche personalizzate
        </Typography>
      </Box>
    )
  }

  const brands = settings.favoriteBrands.split(",").filter(b => b.trim()).slice(0, 3)

  return (
    <Box
      sx={{
        mx: 2,
        mt: 1,
        mb: 2,
        p: 1.5,
        borderRadius: 2,
        bgcolor: "var(--bg-primary)",
        border: "1px solid var(--border-color)",
        display: "flex",
        flexDirection: "column",
        gap: 1,
        boxShadow: "0 1px 3px rgba(0,0,0,0.02)"
      }}
    >
      <Typography
        sx={{
          fontSize: 10,
          fontWeight: 700,
          color: "#9ca3af",
          textTransform: "uppercase",
          letterSpacing: 0.5,
          mb: 0.5
        }}
      >
        Profilo Shopping
      </Typography>

      {/* Brands */}
      {brands.length > 0 && (
        <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
          {brands.map((brand, i) => (
            <Tooltip key={i} title={brand}>
              <Chip
                icon={<LocalOfferIcon sx={{ fontSize: "12px !important" }} />}
                label={brand}
                size="small"
                sx={{
                  height: 20,
                  fontSize: 10,
                  fontWeight: 500,
                  bgcolor: "var(--bg-secondary)",
                  color: "var(--text-primary)",
                  border: "1px solid var(--border-color)",
                  maxWidth: 80,
                  "& .MuiChip-label": { px: 1 }
                }}
              />
            </Tooltip>
          ))}
        </Box>
      )}

      {/* Price */}
      {settings.pricePreference && (
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5 }}>
           <MonetizationOnIcon sx={{ fontSize: 14, color: "#10b981" }} />
           <Typography sx={{ fontSize: 11, fontWeight: 600, color: "var(--text-primary)" }}>
              Budget ~{settings.pricePreference}€
           </Typography>
        </Box>
      )}
    </Box>
  )
}

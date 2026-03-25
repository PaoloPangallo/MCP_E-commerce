import { Box, Typography, Chip, Tooltip, Divider } from "@mui/material"
import { useSettingsStore } from "./store/settingsStore"
import LocalOfferIcon from "@mui/icons-material/LocalOffer"
import MonetizationOnIcon from "@mui/icons-material/MonetizationOn"
import ShoppingBagIcon from "@mui/icons-material/ShoppingBag"
import VerifiedIcon from "@mui/icons-material/Verified"
import CalendarMonthIcon from "@mui/icons-material/CalendarMonth"
import PublicIcon from "@mui/icons-material/Public"
import { useEffect, useState } from "react"
import { apiFetch } from "../../api/apiClient"

interface EbayInfo {
  username?: string
  feedback_score?: string
  watchlist_items?: number
  site?: string
  status?: string
  registration_date?: string
}

const SITE_LABELS: Record<string, string> = {
  Italy: "🇮🇹 Italia",
  US: "🇺🇸 USA",
  Germany: "🇩🇪 Germania",
  France: "🇫🇷 Francia",
  Spain: "🇪🇸 Spagna",
  UK: "🇬🇧 UK",
}

function formatYear(isoDate?: string): string | null {
  if (!isoDate) return null
  try {
    return new Date(isoDate).getFullYear().toString()
  } catch {
    return null
  }
}

export default function ProfileBadge() {
  const { settings } = useSettingsStore()
  const [ebayInfo, setEbayInfo] = useState<EbayInfo | null>(null)

  useEffect(() => {
    apiFetch<EbayInfo>("/auth/ebay/me")
      .then(data => { if (data) setEbayInfo(data) })
      .catch(() => {})
  }, [])

  const isEmpty = !settings.favoriteBrands && !settings.pricePreference
  
  if (isEmpty && !ebayInfo) {
    return (
      <Box
        sx={{
          mx: 2, mt: 1, mb: 2, p: 2,
          borderRadius: 2,
          bgcolor: "transparent",
          border: "1px dashed var(--border-color)",
          display: "flex", flexDirection: "column", alignItems: "center", gap: 1,
          opacity: 0.6, transition: "all 0.2s ease",
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

  const brands = settings.favoriteBrands
    ? settings.favoriteBrands.split(",").filter(b => b.trim()).slice(0, 3)
    : []

  const registrationYear = formatYear(ebayInfo?.registration_date)
  const siteLabel = ebayInfo?.site ? (SITE_LABELS[ebayInfo.site] ?? `🌍 ${ebayInfo.site}`) : null

  return (
    <Box
      sx={{
        mx: 2, mt: 1, mb: 2, p: 1.5,
        borderRadius: 2,
        bgcolor: "var(--bg-primary)",
        border: "1px solid var(--border-color)",
        display: "flex", flexDirection: "column", gap: 1,
        boxShadow: "0 1px 3px rgba(0,0,0,0.02)"
      }}
    >
      <Typography sx={{ fontSize: 10, fontWeight: 700, color: "#9ca3af", textTransform: "uppercase", letterSpacing: 0.5, mb: 0.5 }}>
        Profilo Shopping
      </Typography>

      {/* eBay Identity Card */}
      {ebayInfo?.username && (
        <Box
          sx={{
            p: 1.25,
            borderRadius: "10px",
            background: "linear-gradient(135deg, rgba(0, 100, 210, 0.08) 0%, rgba(0, 100, 210, 0.04) 100%)",
            border: "1px solid rgba(0, 100, 210, 0.15)",
            display: "flex", flexDirection: "column", gap: 0.75
          }}
        >
          {/* Header row */}
          <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between" }}>
            <Box sx={{ display: "flex", alignItems: "center", gap: 0.75 }}>
              <ShoppingBagIcon sx={{ fontSize: 15, color: "#0064d2" }} />
              <Typography sx={{ fontSize: 13, fontWeight: 800, color: "#0064d2", letterSpacing: "-0.02em" }}>
                {ebayInfo.username}
              </Typography>
              {ebayInfo.status === "Confirmed" && (
                <Tooltip title="Account verificato">
                  <VerifiedIcon sx={{ fontSize: 13, color: "#10b981" }} />
                </Tooltip>
              )}
            </Box>

            {ebayInfo.feedback_score && (
              <Tooltip title="Feedback score eBay">
                <Chip
                  label={`⭐ ${ebayInfo.feedback_score}`}
                  size="small"
                  sx={{
                    height: 18, fontSize: 10, fontWeight: 700,
                    bgcolor: "rgba(0, 100, 210, 0.1)",
                    color: "#0064d2", border: "none",
                    "& .MuiChip-label": { px: 0.75 }
                  }}
                />
              </Tooltip>
            )}
          </Box>

          {/* Meta row: site + registration year */}
          {(siteLabel || registrationYear) && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
              {siteLabel && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.4 }}>
                  <PublicIcon sx={{ fontSize: 11, color: "var(--text-secondary)" }} />
                  <Typography sx={{ fontSize: 10, fontWeight: 600, color: "var(--text-secondary)" }}>
                    {siteLabel}
                  </Typography>
                </Box>
              )}
              {registrationYear && (
                <Box sx={{ display: "flex", alignItems: "center", gap: 0.4 }}>
                  <CalendarMonthIcon sx={{ fontSize: 11, color: "var(--text-secondary)" }} />
                  <Typography sx={{ fontSize: 10, fontWeight: 600, color: "var(--text-secondary)" }}>
                    Dal {registrationYear}
                  </Typography>
                </Box>
              )}
              {typeof ebayInfo.watchlist_items === "number" && (
                <Typography sx={{ fontSize: 10, fontWeight: 600, color: "var(--text-secondary)" }}>
                  ♡ {ebayInfo.watchlist_items} osservati
                </Typography>
              )}
            </Box>
          )}
        </Box>
      )}

      {/* Divider if we have both ebay info and prefs */}
      {ebayInfo?.username && (brands.length > 0 || settings.pricePreference) && (
        <Divider sx={{ borderColor: "var(--border-color)", my: 0.25 }} />
      )}

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
                  height: 20, fontSize: 10, fontWeight: 500,
                  bgcolor: "var(--bg-secondary)", color: "var(--text-primary)",
                  border: "1px solid var(--border-color)", maxWidth: 80,
                  "& .MuiChip-label": { px: 1 }
                }}
              />
            </Tooltip>
          ))}
        </Box>
      )}

      {/* Price budget */}
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

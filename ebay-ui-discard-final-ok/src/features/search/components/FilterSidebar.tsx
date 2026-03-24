import { Box, Typography } from "@mui/material"

interface AspectValue {
  localizedValue: string
  matchCount: number
  refinementHref: string
}

interface AspectDistribution {
  localizedAspectName: string
  aspectValues: AspectValue[]
}

interface Props {
  distributions?: AspectDistribution[]
  onFilterClick?: (aspectName: string, value: string) => void
}

export default function FilterSidebar({ distributions = [], onFilterClick }: Props) {
  if (distributions.length === 0) return null

  const visibleAspects = distributions.slice(0, 4)

  return (
    <Box sx={{ display: "flex", flexDirection: "column", gap: 1.5 }}>
      <Typography
        sx={{
          fontSize: 11,
          fontWeight: 500,
          color: "#9ca3af",
          textTransform: "uppercase",
          letterSpacing: "0.06em"
        }}
      >
        Affina la ricerca
      </Typography>

      {visibleAspects.map((aspect) => (
        <Box key={aspect.localizedAspectName}>
          <Typography sx={{ fontSize: 11, fontWeight: 500, color: "#6b7280", mb: 0.6 }}>
            {aspect.localizedAspectName}
          </Typography>

          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
            {aspect.aspectValues.slice(0, 6).map((val) => (
              <Box
                key={val.localizedValue}
                component="button"
                onClick={() => onFilterClick?.(aspect.localizedAspectName, val.localizedValue)}
                sx={{
                  background: "none",
                  border: "1px solid #e5e7eb",
                  borderRadius: "20px",
                  px: 1,
                  py: 0.25,
                  fontSize: 11,
                  color: "#6b7280",
                  cursor: "pointer",
                  fontFamily: "inherit",
                  display: "inline-flex",
                  alignItems: "center",
                  gap: 0.5,
                  transition: "all 0.12s",
                  "&:hover": {
                    bgcolor: "#f9fafb",
                    borderColor: "#d1d5db",
                    color: "#374151"
                  }
                }}
              >
                <span>{val.localizedValue}</span>
                <span style={{ color: "#d1d5db" }}>{val.matchCount}</span>
              </Box>
            ))}
          </Box>
        </Box>
      ))}
    </Box>
  )
}
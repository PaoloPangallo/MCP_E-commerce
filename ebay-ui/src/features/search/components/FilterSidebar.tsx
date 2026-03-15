import { Box, Chip, Typography, Stack } from "@mui/material"
import FilterListIcon from "@mui/icons-material/FilterList"

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

  // We only show the top 4 aspects to avoid clutter
  const visibleAspects = distributions.slice(0, 4)

  return (
    <Box sx={{ border: "1px solid #e5e7eb", bgcolor: "#f9fafb", borderRadius: 4, p: 2, mb: 3 }}>
      <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
        <FilterListIcon sx={{ fontSize: 18, color: "#374151" }} />
        <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#374151", textTransform: "uppercase", letterSpacing: "0.025em" }}>
          Affina la ricerca
        </Typography>
      </Box>

      <Stack spacing={2}>
        {visibleAspects.map((aspect) => (
          <Box key={aspect.localizedAspectName}>
            <Typography sx={{ fontSize: 12, fontWeight: 600, color: "#6b7280", mb: 0.75 }}>
              {aspect.localizedAspectName}
            </Typography>
            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75 }}>
              {aspect.aspectValues.slice(0, 6).map((val) => (
                <Chip
                  key={val.localizedValue}
                  label={`${val.localizedValue} (${val.matchCount})`}
                  size="small"
                  onClick={() => onFilterClick?.(aspect.localizedAspectName, val.localizedValue)}
                  sx={{
                    bgcolor: "#ffffff",
                    border: "1px solid #e5e7eb",
                    fontSize: 11,
                    height: 24,
                    "&:hover": { bgcolor: "#f3f4f6", borderColor: "#d1d5db" }
                  }}
                />
              ))}
            </Box>
          </Box>
        ))}
      </Stack>
    </Box>
  )
}

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
          fontWeight: 700,
          color: "var(--text-secondary)",
          textTransform: "uppercase",
          letterSpacing: "0.06em"
        }}
      >
        Affina la ricerca
      </Typography>

      <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
        {visibleAspects.map((aspect) => (
          <Box key={aspect.localizedAspectName}>
            <Typography sx={{ fontSize: 11, fontWeight: 700, color: "var(--text-primary)", mb: 0.6 }}>
              {aspect.localizedAspectName}
            </Typography>

            <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.5 }}>
              {aspect.aspectValues.slice(0, 8).map((val) => (
                <Box
                  key={val.localizedValue}
                  component="button"
                  onClick={() => onFilterClick?.(aspect.localizedAspectName, val.localizedValue)}
                  sx={{
                    background: "none",
                    border: "1px solid var(--border-color)",
                    borderRadius: "20px",
                    px: 1.25,
                    py: 0.5,
                    fontSize: 11,
                    fontWeight: 600,
                    color: "var(--text-secondary)",
                    cursor: "pointer",
                    fontFamily: "inherit",
                    display: "inline-flex",
                    alignItems: "center",
                    gap: 0.5,
                    transition: "all 0.15s ease",
                    bgcolor: "var(--bg-primary)",
                    "&:hover": {
                      bgcolor: "var(--bg-secondary)",
                      borderColor: "var(--brand-primary)",
                      color: "var(--text-primary)"
                    }
                  }}
                >
                  <span>{val.localizedValue}</span>
                  <span style={{ color: "var(--text-secondary)", opacity: 0.5, marginLeft: 2 }}>
                    {val.matchCount}
                  </span>
                </Box>
              ))}
            </Box>
          </Box>
        ))}
      </Box>
    </Box>
  )
}
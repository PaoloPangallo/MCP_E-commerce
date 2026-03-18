import { useMemo, useState } from "react"
import { Box, Collapse, Typography } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"

import type { IRMetrics, RagContext } from "../../search/types"

interface Props {
  text?: string
  loading?: boolean
  metrics?: IRMetrics
  rag_context?: RagContext
}

function normalizeEvidence(value?: RagContext): string[] {
  if (!value) return []
  if (Array.isArray(value)) return value.filter(Boolean).slice(0, 6)
  return value
    .split(/\n|•|- /g)
    .map((s) => s.trim())
    .filter(Boolean)
    .slice(0, 6)
}

function formatMetric(label: string, value?: number) {
  if (value === undefined || Number.isNaN(value)) return null
  return `${label} ${value.toFixed(2)}`
}

export default function AIAnalysisCard({
  text,
  loading = false,
  metrics,
  rag_context
}: Props) {
  const [expanded, setExpanded] = useState(false)

  const evidence = useMemo(() => normalizeEvidence(rag_context), [rag_context])

  const metricLabels = [
    formatMetric("P@5",    metrics?.["precision@5"]),
    formatMetric("P@10",   metrics?.["precision@10"]),
    formatMetric("R@10",   metrics?.["recall@10"]),
    formatMetric("NDCG@10", metrics?.["ndcg@10"])
  ].filter(Boolean) as string[]

  const hasExtra = metricLabels.length > 0 || evidence.length > 0

  if (!text && !loading && !hasExtra) return null

  return (
    <Box
      sx={{
        border: "1px solid #f0f0f0",
        borderRadius: 3,
        bgcolor: "#fff",
        overflow: "hidden"
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 2,
          py: 1.25,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: text || loading || hasExtra ? "1px solid #f5f5f5" : "none"
        }}
      >
        <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em" }}>
          AI analysis
        </Typography>

        {hasExtra && (
          <Box
            component="button"
            onClick={() => setExpanded((v) => !v)}
            sx={{
              background: "none",
              border: "none",
              p: 0,
              cursor: "pointer",
              display: "inline-flex",
              alignItems: "center",
              gap: 0.25,
              fontFamily: "inherit"
            }}
          >
            <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>
              {expanded ? "nascondi" : "dettagli"}
            </Typography>
            <KeyboardArrowDownIcon
              sx={{
                fontSize: 14,
                color: "#9ca3af",
                transform: expanded ? "rotate(180deg)" : "none",
                transition: "transform 0.2s"
              }}
            />
          </Box>
        )}
      </Box>

      {/* Body */}
      <Box sx={{ px: 2, py: 1.5 }}>
        {loading && !text && (
          <Typography sx={{ fontSize: 12, color: "#d1d5db", fontStyle: "italic" }}>
            generazione analisi…
          </Typography>
        )}

        {text && (
          <Typography
            sx={{
              fontSize: 13,
              color: "#374151",
              lineHeight: 1.7,
              whiteSpace: "pre-wrap"
            }}
          >
            {text}
          </Typography>
        )}

        {/* Expandable details */}
        <Collapse in={expanded} timeout={200}>
          <Box sx={{ mt: 1.5, pt: 1.5, borderTop: "1px solid #f5f5f5", display: "flex", flexDirection: "column", gap: 1.5 }}>

            {metricLabels.length > 0 && (
              <Box>
                <Typography sx={{ fontSize: 11, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.75 }}>
                  Ranking metrics
                </Typography>
                <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap" }}>
                  {metricLabels.map((m) => (
                    <Box
                      key={m}
                      sx={{
                        px: 0.875,
                        py: 0.2,
                        borderRadius: "6px",
                        bgcolor: "#f9fafb",
                        border: "1px solid #e5e7eb"
                      }}
                    >
                      <Typography sx={{ fontSize: 11, color: "#6b7280", fontFamily: "monospace" }}>
                        {m}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            )}

            {evidence.length > 0 && (
              <Box>
                <Typography sx={{ fontSize: 11, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.06em", mb: 0.75 }}>
                  Retrieved evidence
                </Typography>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 0.75 }}>
                  {evidence.map((item, i) => (
                    <Typography
                      key={i}
                      sx={{
                        fontSize: 12,
                        color: "#6b7280",
                        lineHeight: 1.6,
                        borderLeft: "2px solid #f0f0f0",
                        pl: 0.875
                      }}
                    >
                      {item}
                    </Typography>
                  ))}
                </Box>
              </Box>
            )}
          </Box>
        </Collapse>
      </Box>
    </Box>
  )
}
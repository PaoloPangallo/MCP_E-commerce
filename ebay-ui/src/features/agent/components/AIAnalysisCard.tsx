import { useMemo, useState } from "react"
import { Box, Chip, Collapse, IconButton, Typography } from "@mui/material"

import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ExpandLessIcon from "@mui/icons-material/ExpandLess"

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
  return value.split(/\n|•|- /g).map((item) => item.trim()).filter(Boolean).slice(0, 6)
}

function formatMetric(label: string, value?: number) {
  if (value === undefined || Number.isNaN(value)) return null
  return `${label} ${value.toFixed(2)}`
}

export default function AIAnalysisCard({ text, loading = false, metrics, rag_context }: Props) {
  const [expanded, setExpanded] = useState(false)
  const evidence = useMemo(() => normalizeEvidence(rag_context), [rag_context])

  const metricLabels = [
    formatMetric("Precision@5", metrics?.["precision@5"]),
    formatMetric("Precision@10", metrics?.["precision@10"]),
    formatMetric("Recall@10", metrics?.["recall@10"]),
    formatMetric("NDCG@10", metrics?.["ndcg@10"])
  ].filter(Boolean) as string[]

  const hasExtraDetails = metricLabels.length > 0 || evidence.length > 0
  if (!text && !loading && !hasExtraDetails) return null

  return (
    <Box sx={{ border: "1px solid #e5e7eb", bgcolor: "#ffffff", borderRadius: 4, px: 3, py: 2.75 }}>
      <Box sx={{ display: "flex", alignItems: "flex-start", justifyContent: "space-between", gap: 2, mb: text || loading ? 1.25 : 0 }}>
        <Box>
          <Box display="flex" alignItems="center" gap={1}>
            <AutoAwesomeIcon sx={{ fontSize: 18, color: "#111827" }} />
            <Typography sx={{ fontSize: 14, fontWeight: 700, color: "#111827" }}>AI analysis</Typography>
            {hasExtraDetails ? <Chip label="explainable" size="small" sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb", color: "#4b5563" }} /> : null}
          </Box>
          <Typography sx={{ mt: 0.6, fontSize: 12.5, color: "#6b7280", lineHeight: 1.6 }}>
            Sintesi dell’analisi agentica, con eventuali segnali retrieval e metriche di ranking.
          </Typography>
        </Box>
        {hasExtraDetails ? (
          <IconButton size="small" aria-label={expanded ? "Nascondi dettagli" : "Mostra dettagli"} onClick={() => setExpanded((prev) => !prev)} sx={{ color: "#6b7280" }}>
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        ) : null}
      </Box>

      {loading ? <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: text ? 1.2 : 0 }}><Chip size="small" label="generazione analisi" sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb", color: "#4b5563" }} /></Box> : null}
      {text ? <Typography sx={{ fontSize: 14.5, color: "#374151", lineHeight: 1.8, whiteSpace: "pre-wrap" }}>{text}</Typography> : null}

      <Collapse in={expanded}>
        <Box sx={{ mt: 2, pt: 2, borderTop: "1px solid #f0f2f5" }}>
          {metricLabels.length > 0 ? (
            <Box mb={evidence.length > 0 ? 2 : 0}>
              <Typography sx={{ fontSize: 12.5, fontWeight: 700, color: "#111827", mb: 1 }}>Ranking metrics</Typography>
              <Box display="flex" gap={1} flexWrap="wrap">
                {metricLabels.map((metric) => <Chip key={metric} size="small" label={metric} sx={{ bgcolor: "#f9fafb", border: "1px solid #e5e7eb", color: "#4b5563" }} />)}
              </Box>
            </Box>
          ) : null}
          {evidence.length > 0 ? (
            <Box>
              <Typography sx={{ fontSize: 12.5, fontWeight: 700, color: "#111827", mb: 1 }}>Retrieved evidence</Typography>
              <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                {evidence.map((item) => (
                  <Box key={item} sx={{ border: "1px solid #eef1f4", bgcolor: "#fafbfc", borderRadius: 2.5, px: 1.5, py: 1.2 }}>
                    <Typography sx={{ fontSize: 13, color: "#4b5563", lineHeight: 1.65 }}>{item}</Typography>
                  </Box>
                ))}
              </Box>
            </Box>
          ) : null}
        </Box>
      </Collapse>
    </Box>
  )
}

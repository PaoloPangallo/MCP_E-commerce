import { useMemo, useState } from "react"
import {
  Box,
  Typography,
  CircularProgress,
  Collapse,
  IconButton,
  Chip,
  Divider
} from "@mui/material"

import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ExpandLessIcon from "@mui/icons-material/ExpandLess"
import ReactMarkdown from "react-markdown"

import type { IRMetrics, RagContext } from "../component/searchTypes.ts"

interface Props {
  text?: string
  loading?: boolean
  metrics?: IRMetrics
  rag_context?: RagContext
}

function normalizeEvidence(value?: RagContext): string[] {
  if (!value) {
    return []
  }

  if (Array.isArray(value)) {
    return value.filter(Boolean).slice(0, 6)
  }

  return value
    .split(/\n|•|- /g)
    .map(item => item.trim())
    .filter(Boolean)
    .slice(0, 6)
}

function formatMetric(label: string, value?: number) {
  if (value === undefined || Number.isNaN(value)) {
    return null
  }

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
    formatMetric("Precision@5", metrics?.["precision@5"]),
    formatMetric("Precision@10", metrics?.["precision@10"]),
    formatMetric("Recall@10", metrics?.["recall@10"]),
    formatMetric("NDCG@10", metrics?.["ndcg@10"]),
  ].filter(Boolean) as string[]

  const hasExtraDetails = metricLabels.length > 0 || evidence.length > 0

  if (!text && !loading && !hasExtraDetails) {
    return null
  }

  return (
    <Box
      sx={{
        p: 3,
        mb: 3,
        borderRadius: "16px",
        bgcolor: "#ffffff",
        border: "1px solid #e5e5e5",
        boxShadow: "0 1px 2px rgba(0,0,0,0.04)"
      }}
    >
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={2}>
        <Box display="flex" alignItems="center" gap={1} flexWrap="wrap">
          <AutoAwesomeIcon sx={{ fontSize: 18, color: "#10a37f" }} />

          <Typography fontWeight={600} fontSize={14}>
            AI Analysis
          </Typography>

          {hasExtraDetails && (
            <Chip
              label="Explainable AI"
              size="small"
              sx={{
                fontSize: 11,
                bgcolor: "#eef3ff",
                color: "#3b5ccc"
              }}
            />
          )}
        </Box>

        {hasExtraDetails && (
          <IconButton
            size="small"
            aria-label={expanded ? "Nascondi dettagli" : "Mostra dettagli"}
            onClick={() => setExpanded(prev => !prev)}
          >
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        )}
      </Box>

      {loading && (
        <Box display="flex" alignItems="center" gap={1} mb={text ? 2 : 0}>
          <CircularProgress size={16} />
          <Typography sx={{ fontSize: 13, color: "#777" }}>
            Generazione analisi AI...
          </Typography>
        </Box>
      )}

      {text && (
        <Box mb={hasExtraDetails ? 1 : 0} sx={{
          color: "#333",
          fontSize: 15,
          lineHeight: 1.7,
          "& p": { mt: 0, mb: 1.5 },
          "& ul": { mt: 0, mb: 1.5, pl: 2.5 },
          "& li": { mb: 0.5 },
          "& strong": { fontWeight: 600 }
        }}>
          <ReactMarkdown>{text}</ReactMarkdown>
        </Box>
      )}

      <Collapse in={expanded}>
        <Divider sx={{ my: 2 }} />

        {metricLabels.length > 0 && (
          <Box mb={evidence.length > 0 ? 2 : 0}>
            <Typography sx={{ fontSize: 13, fontWeight: 600, mb: 1 }}>
              Ranking Metrics
            </Typography>

            <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
              {metricLabels.map(metric => (
                <Chip
                  key={metric}
                  size="small"
                  label={metric}
                  sx={{
                    bgcolor: "#fafafa",
                    border: "1px solid #e5e5e5",
                    fontSize: 11
                  }}
                />
              ))}
            </Box>
          </Box>
        )}

        {evidence.length > 0 && (
          <Box>
            <Typography sx={{ fontSize: 13, fontWeight: 600, mb: 1 }}>
              Retrieved Evidence
            </Typography>

            <Box sx={{ display: "flex", flexDirection: "column", gap: 0.8 }}>
              {evidence.map(item => (
                <Typography key={item} sx={{ fontSize: 13, color: "#666", lineHeight: 1.6 }}>
                  • {item}
                </Typography>
              ))}
            </Box>
          </Box>
        )}
      </Collapse>
    </Box>
  )
}

import { useMemo, useState } from "react"
import { Box, Collapse, Typography, Button } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"
import PsychologyIcon from "@mui/icons-material/Psychology"
import AssessmentOutlinedIcon from "@mui/icons-material/AssessmentOutlined"
import MenuBookOutlinedIcon from "@mui/icons-material/MenuBookOutlined"

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
        borderRadius: "16px",
        bgcolor: "#fff",
        overflow: "hidden",
        boxShadow: "0 4px 20px rgba(0,0,0,0.04)"
      }}
    >
      {/* Header */}
      <Box
        sx={{
          px: 2.5,
          py: 1.75,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          borderBottom: text || loading || hasExtra ? "1px solid #f5f5f5" : "none",
          bgcolor: "#fafcff"
        }}
      >
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
          <PsychologyIcon sx={{ fontSize: 18, color: "#3b82f6" }} />
          <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#1e3a8a", textTransform: "uppercase", letterSpacing: "0.08em" }}>
            Analisi Agent AI
          </Typography>
        </Box>

        {hasExtra && (
          <Button
            size="small"
            variant="text"
            onClick={() => setExpanded((v) => !v)}
            endIcon={
              <KeyboardArrowDownIcon
                sx={{
                  fontSize: 16,
                  transform: expanded ? "rotate(180deg)" : "none",
                  transition: "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
                }}
              />
            }
            sx={{
              textTransform: "none",
              fontSize: 11,
              fontWeight: 600,
              color: "#9ca3af",
              p: 0,
              "&:hover": { color: "#3b82f6", bgcolor: 'transparent' }
            }}
          >
            {expanded ? "Nascondi dettagli" : "Vedi dettagli"}
          </Button>
        )}
      </Box>

      {/* Body */}
      <Box sx={{ px: 2.5, py: 2 }}>
        {loading && !text && (
          <Typography sx={{ fontSize: 13, color: "#9ca3af", fontStyle: "italic", textAlign: 'center', py: 2 }}>
            Sintesi dell'analisi in corso…
          </Typography>
        )}

        {text && (
          <Typography
            sx={{
              fontSize: 14,
              color: "#374151",
              lineHeight: 1.6,
              whiteSpace: "pre-wrap",
              fontWeight: 400
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
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 1.25 }}>
                   <AssessmentOutlinedIcon sx={{ fontSize: 14, color: "#9ca3af" }} />
                   <Typography sx={{ fontSize: 10, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700 }}>
                    Ranking Performance
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap" }}>
                  {metricLabels.map((m) => (
                    <Box
                      key={m}
                      sx={{
                        px: 1.25,
                        py: 0.5,
                        borderRadius: "8px",
                        bgcolor: "#f8fafc",
                        border: "1px solid #e2e8f0"
                      }}
                    >
                      <Typography sx={{ fontSize: 11, color: "#475569", fontWeight: 700, fontFamily: "'JetBrains Mono', 'Roboto Mono', monospace" }}>
                        {m}
                      </Typography>
                    </Box>
                  ))}
                </Box>
              </Box>
            )}

            {evidence.length > 0 && (
              <Box>
                <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.75, mb: 1.25 }}>
                   <MenuBookOutlinedIcon sx={{ fontSize: 14, color: "#9ca3af" }} />
                   <Typography sx={{ fontSize: 10, color: "#9ca3af", textTransform: "uppercase", letterSpacing: "0.08em", fontWeight: 700 }}>
                    Contesto Rilevato (RAG)
                  </Typography>
                </Box>
                <Box sx={{ display: "flex", flexDirection: "column", gap: 1 }}>
                  {evidence.map((item, i) => (
                    <Box
                      key={i}
                      sx={{
                        p: 1.25,
                        bgcolor: "#fafafa",
                        borderRadius: "8px",
                        border: "1px solid #f0f0f0",
                        borderLeft: "4px solid #3b82f6"
                      }}
                    >
                      <Typography
                        sx={{
                          fontSize: 12,
                          color: "#4b5563",
                          lineHeight: 1.6
                        }}
                      >
                        {item}
                      </Typography>
                    </Box>
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
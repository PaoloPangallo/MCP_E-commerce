import { useMemo, useState } from "react"
import { Box, Chip, Collapse, IconButton, Typography, keyframes } from "@mui/material"

import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline"
import AutorenewIcon from "@mui/icons-material/Autorenew"
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline"
import SearchIcon from "@mui/icons-material/Search"
import StorefrontIcon from "@mui/icons-material/Storefront"
import CloudOutlinedIcon from "@mui/icons-material/CloudOutlined"
import InsightsOutlinedIcon from "@mui/icons-material/InsightsOutlined"
import BuildCircleOutlinedIcon from "@mui/icons-material/BuildCircleOutlined"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ExpandLessIcon from "@mui/icons-material/ExpandLess"
import PsychologyIcon from "@mui/icons-material/Psychology"

import type { AgentStep, PlannedTask } from "../types"

// Animations
const pulse = keyframes`
  0% { transform: scale(0.95); opacity: 0.5; box-shadow: 0 0 0 0 rgba(17, 24, 39, 0.4); }
  70% { transform: scale(1); opacity: 1; box-shadow: 0 0 0 10px rgba(17, 24, 39, 0); }
  100% { transform: scale(0.95); opacity: 0.5; box-shadow: 0 0 0 0 rgba(17, 24, 39, 0); }
`

const glow = keyframes`
  0% { opacity: 0.4; }
  50% { opacity: 0.8; }
  100% { opacity: 0.4; }
`

interface Props {
  agentTrace?: AgentStep[]
  loading?: boolean
  query?: string
  plannedTasks?: PlannedTask[]
}

function humanizeToolName(action?: string, input?: any) {
  const value = (action || "").toLowerCase()
  const q = input?.query || input?.product || ""
  const seller = input?.seller_name || ""

  switch (value) {
    case "search_pipeline":
      return q ? `Cerco "${q}" su eBay...` : "Ricerca prodotti in corso..."
    case "seller_pipeline":
      return seller ? `Analizzo feedback di ${seller}...` : "Analisi affidabilità venditore..."
    case "weather_tool":
      return "Controllo condizioni meteo..."
    case "price_history_tool":
      return "Studio lo storico dei prezzi..."
    case "shipping_tool":
      return "Calcolo stime di spedizione..."
    case "finish":
      return "Sintesi dei risultati..."
    default:
      return action || "Elaborazione passo agente..."
  }
}

function getActionIcon(action?: string) {
  const value = (action || "").toLowerCase()

  switch (value) {
    case "search_pipeline":
      return <SearchIcon sx={{ fontSize: 16 }} />
    case "seller_pipeline":
      return <StorefrontIcon sx={{ fontSize: 16 }} />
    case "weather_tool":
      return <CloudOutlinedIcon sx={{ fontSize: 16 }} />
    case "price_history_tool":
      return <InsightsOutlinedIcon sx={{ fontSize: 16 }} />
    case "finish":
      return <CheckCircleOutlineIcon sx={{ fontSize: 16 }} />
    default:
      return <BuildCircleOutlinedIcon sx={{ fontSize: 16 }} />
  }
}


function getStatus(step: AgentStep) {
  if (step.status === "error") {
    return {
      label: "Interrotto",
      icon: <ErrorOutlineIcon sx={{ fontSize: 14, color: "#e11d48" }} />,
      color: "#e11d48",
      bg: "rgba(255, 241, 242, 0.4)"
    }
  }

  if (step.status === "thinking" || step.status === "running") {
    return {
      label: "Pensando...",
      icon: <AutorenewIcon sx={{ fontSize: 14, color: "#111827" }} />,
      color: "#111827",
      bg: "rgba(249, 250, 251, 0.6)"
    }
  }

  return {
    label: "Completato",
    icon: <CheckCircleOutlineIcon sx={{ fontSize: 14, color: "#059669" }} />,
    color: "#059669",
    bg: "rgba(236, 253, 245, 0.6)"
  }
}

export default function AIThinkingPipeline({
  agentTrace = [],
  loading = false,
  plannedTasks = []
}: Props) {
  const [expanded, setExpanded] = useState(true)

  const normalizedSteps = useMemo(
    () => agentTrace.filter(Boolean).sort((a, b) => (a.step ?? 0) - (b.step ?? 0)),
    [agentTrace]
  )

  if (!loading && normalizedSteps.length === 0 && plannedTasks.length === 0) {
    return null
  }

  return (
    <Box sx={{ mt: 2, mb: 3 }}>
      {/* Dynamic Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          mb: 2,
          px: 1
        }}
      >
        <Box display="flex" alignItems="center" gap={1.5}>
          <Box
            sx={{
              width: 32,
              height: 32,
              borderRadius: "50%",
              bgcolor: "#111827",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              boxShadow: "0 0 15px rgba(17, 24, 39, 0.2)",
              animation: loading ? `${glow} 2s infinite ease-in-out` : "none"
            }}
          >
            <PsychologyIcon sx={{ color: "#fff", fontSize: 20 }} />
          </Box>
          <Box>
            <Typography sx={{ fontSize: 15, fontWeight: 900, color: "#0f172a", display: "flex", alignItems: "center", gap: 1 }}>
              {loading ? "Intelligenza Artificiale al lavoro..." : "Analisi Completata"}
            </Typography>
            <Typography sx={{ fontSize: 12, color: "#64748b", fontWeight: 500 }}>
              {normalizedSteps.length} passaggi logici intrapresi
            </Typography>
          </Box>
        </Box>

        <IconButton
          size="small"
          onClick={() => setExpanded((prev) => !prev)}
          sx={{ bgcolor: "#f1f5f9" }}
        >
          {expanded ? <ExpandLessIcon sx={{ fontSize: 18 }} /> : <ExpandMoreIcon sx={{ fontSize: 18 }} />}
        </IconButton>
      </Box>

      <Collapse in={expanded}>
        <Box sx={{ position: "relative", pl: 2, ml: 1, borderLeft: "2px solid #e2e8f0" }}>
          {normalizedSteps.map((step, index) => {
            const status = getStatus(step)
            const isActive = (step.status === "thinking" || step.status === "running") && loading

            return (
              <Box
                key={`${step.step}-${index}`}
                sx={{
                  mb: 2,
                  position: "relative",
                  transition: "all 0.3s ease",
                  opacity: isActive ? 1 : 0.8,
                  "&:hover": { opacity: 1 }
                }}
              >
                {/* Timeline Connector Dot */}
                <Box
                  sx={{
                    position: "absolute",
                    left: -27.5,
                    top: 14,
                    width: 11,
                    height: 11,
                    borderRadius: "50%",
                    bgcolor: isActive ? "#111827" : step.status === "error" ? "#e11d48" : "#94a3b8",
                    border: "2px solid #fff",
                    boxShadow: isActive ? "0 0 10px rgba(17, 24, 39, 0.3)" : "none",
                    animation: isActive ? `${pulse} 2s infinite` : "none",
                    zIndex: 2
                  }}
                />

                {/* Step Card */}
                <Box
                  sx={{
                    p: 2,
                    borderRadius: 3,
                    bgcolor: isActive ? "rgba(255,255,255,0.8)" : "rgba(255,255,255,0.4)",
                    backdropFilter: "blur(12px)",
                    border: isActive ? "1.5px solid #111827" : "1px solid rgba(226, 232, 240, 0.8)",
                    boxShadow: isActive ? "0 4px 15px rgba(0,0,0,0.05)" : "none"
                  }}
                >
                  <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
                    <Box display="flex" alignItems="center" gap={1}>
                      <Box sx={{ color: "#64748b" }}>{getActionIcon(step.action)}</Box>
                      <Typography sx={{ fontSize: 13, fontWeight: 800, color: "#0f172a" }}>
                        {humanizeToolName(step.action, step.action_input)}
                      </Typography>
                    </Box>
                    <Chip
                      label={status.label}
                      size="small"
                      sx={{
                        height: 20,
                        fontSize: 10,
                        fontWeight: 700,
                        bgcolor: status.bg,
                        color: status.color,
                        "& .MuiChip-label": { px: 1 }
                      }}
                    />
                  </Box>

                  {step.thought && (
                    <Typography sx={{ fontSize: 13.5, color: "#334155", lineHeight: 1.6, fontWeight: 500 }}>
                      {step.thought}
                    </Typography>
                  )}

                  {step.observation_summary && (
                    <Box sx={{ mt: 1.5, p: 1.5, borderRadius: 2, bgcolor: "rgba(0,0,0,0.02)", border: "1px dashed #e2e8f0" }}>
                      <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#64748b", mb: 0.5, textTransform: "uppercase" }}>Risultato</Typography>
                      <Typography sx={{ fontSize: 12.5, color: "#475569", lineHeight: 1.5 }}>
                        {step.observation_summary}
                      </Typography>
                    </Box>
                  )}
                </Box>
              </Box>
            )
          })}

          {/* Planned Tasks if any */}
          {plannedTasks.length > normalizedSteps.length && (
            <Box sx={{ mt: 2, opacity: 0.5 }}>
              <Typography sx={{ fontSize: 12, fontWeight: 700, color: "#94a3b8", ml: 1, mb: 1 }}>PIANO D'AZIONE FUTURO</Typography>
              <Box display="flex" gap={1} flexWrap="wrap">
                {plannedTasks.slice(normalizedSteps.length).map((t, i) => (
                  <Chip key={i} label={humanizeToolName(t.tool)} size="small" variant="outlined" sx={{ height: 24, fontSize: 11, borderColor: "#e2e8f0" }} />
                ))}
              </Box>
            </Box>
          )}

          {/* Active Thinking Loader */}
          {loading && (
            <Box sx={{ mt: 1, display: "flex", alignItems: "center", gap: 2, px: 2, py: 1.5 }}>
              <AutorenewIcon sx={{ fontSize: 18, color: "#64748b", animation: "spin 2s linear infinite" }} />
              <Typography sx={{ fontSize: 13, color: "#64748b", fontStyle: "italic" }}>
                L'IA sta elaborando la strategia ottimale...
              </Typography>
            </Box>
          )}
        </Box>
      </Collapse>

      <style>
        {`
          @keyframes spin {
            from { transform: rotate(0deg); }
            to { transform: rotate(360deg); }
          }
        `}
      </style>
    </Box>
  )
}

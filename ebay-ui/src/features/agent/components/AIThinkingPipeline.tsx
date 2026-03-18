import { useMemo } from "react"
import { Box, Typography, keyframes } from "@mui/material"

import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline"
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline"
import SearchIcon from "@mui/icons-material/Search"
import StorefrontIcon from "@mui/icons-material/Storefront"
import CloudOutlinedIcon from "@mui/icons-material/CloudOutlined"
import InsightsOutlinedIcon from "@mui/icons-material/InsightsOutlined"
import BuildCircleOutlinedIcon from "@mui/icons-material/BuildCircleOutlined"

import type { AgentStep, PlannedTask } from "../types"

const spin = keyframes`
  from { transform: rotate(0deg); }
  to { transform: rotate(360deg); }
`

interface Props {
  agentTrace?: AgentStep[]
  loading?: boolean
  query?: string
  plannedTasks?: PlannedTask[]
}

function humanizeToolName(action?: string, input?: any) {
  const val = (action || "").toLowerCase()
  const q = input?.query || input?.product || ""
  const seller = input?.seller_name || ""

  switch (val) {
    case "search_pipeline":
      return q ? `Cerco "${q}" su eBay` : "Ricerca prodotti"
    case "seller_pipeline":
      return seller ? `Analizzo ${seller}` : "Analisi venditore"
    case "weather_tool":
      return "Condizioni meteo"
    case "price_history_tool":
      return "Storico prezzi"
    case "shipping_tool":
      return "Calcolo spedizione"
    case "finish":
      return "Sintesi risultati"
    default:
      return action || "Elaborazione"
  }
}

function getActionIcon(action?: string) {
  const val = (action || "").toLowerCase()
  const sx = { fontSize: 13, color: "#9ca3af" }
  switch (val) {
    case "search_pipeline":     return <SearchIcon sx={sx} />
    case "seller_pipeline":     return <StorefrontIcon sx={sx} />
    case "weather_tool":        return <CloudOutlinedIcon sx={sx} />
    case "price_history_tool":  return <InsightsOutlinedIcon sx={sx} />
    case "finish":              return <CheckCircleOutlineIcon sx={sx} />
    default:                    return <BuildCircleOutlinedIcon sx={sx} />
  }
}

function StepRow({ step, isLast, loading }: { step: AgentStep; isLast: boolean; loading: boolean }) {
  const isRunning = (step.status === "thinking" || step.status === "running") && loading
  const isError   = step.status === "error"
  return (
    <Box sx={{ display: "flex", gap: 1.25, position: "relative" }}>
      {/* Timeline spine */}
      <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0, width: 16 }}>
        {/* dot */}
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            mt: "4px",
            flexShrink: 0,
            bgcolor: isError ? "#fca5a5" : isRunning ? "#111827" : "#d1fae5",
            border: "1.5px solid",
            borderColor: isError ? "#ef4444" : isRunning ? "#111827" : "#6ee7b7",
            animation: isRunning ? `${spin} 1.8s linear infinite` : "none"
          }}
        />
        {/* vertical line */}
        {!isLast && (
          <Box sx={{ width: "1px", flex: 1, bgcolor: "#f0f0f0", mt: 0.5 }} />
        )}
      </Box>

      {/* Content */}
      <Box sx={{ pb: isLast ? 0 : 2, minWidth: 0, flex: 1 }}>
        {/* Action label */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, mb: 0.25 }}>
          {getActionIcon(step.action)}
          <Typography sx={{ fontSize: 12, fontWeight: 500, color: "#374151" }}>
            {humanizeToolName(step.action, step.action_input)}
          </Typography>
          {isError && (
            <ErrorOutlineIcon sx={{ fontSize: 12, color: "#ef4444", ml: 0.25 }} />
          )}
        </Box>

        {/* Thought */}
        {step.thought && (
          <Typography sx={{ fontSize: 12, color: "#9ca3af", lineHeight: 1.55, mb: 0.5 }}>
            {step.thought}
          </Typography>
        )}

        {/* Observation */}
        {step.observation_summary && (
          <Typography
            sx={{
              fontSize: 11,
              color: "#9ca3af",
              lineHeight: 1.5,
              borderLeft: "2px solid #f0f0f0",
              pl: 0.875,
              fontStyle: "italic"
            }}
          >
            {step.observation_summary}
          </Typography>
        )}
      </Box>
    </Box>
  )
}

export default function AIThinkingPipeline({
  agentTrace = [],
  loading = false,
  plannedTasks = []
}: Props) {
  const steps = useMemo(
    () => agentTrace.filter(Boolean).sort((a, b) => (a.step ?? 0) - (b.step ?? 0)),
    [agentTrace]
  )

  if (!loading && steps.length === 0 && plannedTasks.length === 0) return null

  const pendingTasks = plannedTasks.slice(steps.length)

  return (
    <Box>
      {/* Steps */}
      {steps.map((step, i) => (
        <StepRow
          key={`${step.step}-${i}`}
          step={step}
          isLast={i === steps.length - 1 && !loading && pendingTasks.length === 0}
          loading={loading}
        />
      ))}

      {/* Pending planned tasks */}
      {pendingTasks.length > 0 && (
        <Box sx={{ display: "flex", gap: 0.75, flexWrap: "wrap", mt: 0.5, pl: "20px" }}>
          {pendingTasks.map((t, i) => (
            <Typography key={i} sx={{ fontSize: 11, color: "#d1d5db" }}>
              {humanizeToolName(t.tool)}
              {i < pendingTasks.length - 1 ? " ·" : ""}
            </Typography>
          ))}
        </Box>
      )}

      {/* Loading spinner row */}
      {loading && (
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, pl: "20px", mt: 0.5 }}>
          <Box
            sx={{
              width: 8,
              height: 8,
              borderRadius: "50%",
              border: "1.5px solid #d1d5db",
              borderTopColor: "#9ca3af",
              animation: `${spin} 0.8s linear infinite`,
              flexShrink: 0
            }}
          />
          <Typography sx={{ fontSize: 11, color: "#9ca3af" }}>
            elaborazione in corso…
          </Typography>
        </Box>
      )}
    </Box>
  )
}
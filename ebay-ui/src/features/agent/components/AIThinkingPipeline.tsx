import { useMemo, useState } from "react"
import { Box, Chip, Collapse, IconButton, Typography } from "@mui/material"

import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline"
import AutorenewIcon from "@mui/icons-material/Autorenew"
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline"
import SearchIcon from "@mui/icons-material/Search"
import StorefrontIcon from "@mui/icons-material/Storefront"
import CloudOutlinedIcon from "@mui/icons-material/CloudOutlined"
import InsightsOutlinedIcon from "@mui/icons-material/InsightsOutlined"
import BuildCircleOutlinedIcon from "@mui/icons-material/BuildCircleOutlined"
import PsychologyAltIcon from "@mui/icons-material/PsychologyAlt"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ExpandLessIcon from "@mui/icons-material/ExpandLess"

import type { AgentStep, PlannedTask } from "../types"

interface Props {
  agentTrace?: AgentStep[]
  loading?: boolean
  query?: string
  plannedTasks?: PlannedTask[]
}

function humanizeToolName(action?: string) {
  const value = (action || "").toLowerCase()

  switch (value) {
    case "search_pipeline":
      return "Product search"
    case "seller_pipeline":
      return "Seller analysis"
    case "weather_tool":
      return "Weather lookup"
    case "price_history_tool":
      return "Price history"
    case "shipping_tool":
      return "Shipping estimate"
    case "finish":
      return "Final answer"
    default:
      return action || "Agent step"
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
    default:
      return <BuildCircleOutlinedIcon sx={{ fontSize: 16 }} />
  }
}

function formatInput(input?: Record<string, unknown>) {
  if (!input) return null

  if (typeof input.query === "string" && input.query.trim()) {
    return `query: ${input.query}`
  }

  if (typeof input.seller_name === "string" && input.seller_name.trim()) {
    return `seller: ${input.seller_name}`
  }

  if (typeof input.location === "string" && input.location.trim()) {
    return `location: ${input.location}`
  }

  const keys = Object.keys(input)
  if (keys.length === 0) return null

  try {
    return JSON.stringify(input)
  } catch {
    return null
  }
}

function getStatus(step: AgentStep) {
  if (step.status === "error") {
    return {
      label: "Errore",
      icon: <ErrorOutlineIcon sx={{ fontSize: 16, color: "#b42318" }} />,
      accent: "#fef3f2",
      border: "#fecdca",
      text: "#b42318"
    }
  }

  if (step.status === "thinking" || step.status === "running") {
    return {
      label: "In corso",
      icon: <AutorenewIcon sx={{ fontSize: 16, color: "#475467" }} />,
      accent: "#f8fafc",
      border: "#e5e7eb",
      text: "#475467"
    }
  }

  return {
    label: "Completato",
    icon: <CheckCircleOutlineIcon sx={{ fontSize: 16, color: "#027a48" }} />,
    accent: "#f6fef9",
    border: "#abefc6",
    text: "#027a48"
  }
}

export default function AIThinkingPipeline({
  agentTrace = [],
  loading = false,
  query,
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
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "flex-start",
          justifyContent: "space-between",
          gap: 2,
          mb: 1.5
        }}
      >
        <Box>
          <Typography sx={{ fontSize: 14, fontWeight: 700, color: "#111827" }}>
            Agent trace
          </Typography>

          <Typography
            sx={{ fontSize: 12.5, color: "#6b7280", lineHeight: 1.6, mt: 0.5 }}
          >
            L’agente decide quali tool usare, osserva i risultati e costruisce la risposta finale.
          </Typography>

          {query ? (
            <Typography
              sx={{ fontSize: 12.5, color: "#6b7280", lineHeight: 1.6, mt: 0.75 }}
            >
              Query:{" "}
              <Box component="span" sx={{ color: "#111827", fontWeight: 600 }}>
                {query}
              </Box>
            </Typography>
          ) : null}
        </Box>

        <Box display="flex" alignItems="center" gap={0.75}>
          {loading ? (
            <Chip
              size="small"
              icon={<AutorenewIcon sx={{ fontSize: 14 }} />}
              label="in esecuzione"
              sx={{
                bgcolor: "#f9fafb",
                border: "1px solid #e5e7eb",
                color: "#374151"
              }}
            />
          ) : null}

          <IconButton
            size="small"
            onClick={() => setExpanded((prev) => !prev)}
            sx={{ color: "#6b7280" }}
          >
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        </Box>
      </Box>

      {plannedTasks.length > 0 ? (
        <Box
          sx={{
            border: "1px solid #e5e7eb",
            bgcolor: "#ffffff",
            borderRadius: 3,
            px: 2,
            py: 1.5,
            mb: 1.5
          }}
        >
          <Typography sx={{ fontSize: 12.5, fontWeight: 700, color: "#111827", mb: 1 }}>
            Task pianificati
          </Typography>

          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 0.75 }}>
            {plannedTasks.map((task, index) => (
              <Chip
                key={`${task.tool}-${index}`}
                size="small"
                icon={<PsychologyAltIcon sx={{ fontSize: 14 }} />}
                label={humanizeToolName(task.tool)}
                sx={{
                  bgcolor: "#f9fafb",
                  border: "1px solid #e5e7eb",
                  color: "#374151"
                }}
              />
            ))}
          </Box>
        </Box>
      ) : null}

      {normalizedSteps.length === 0 && loading ? (
        <Box
          sx={{
            border: "1px solid #e5e7eb",
            bgcolor: "#ffffff",
            borderRadius: 3,
            px: 2,
            py: 1.75
          }}
        >
          <Box display="flex" alignItems="center" gap={1}>
            <AutorenewIcon sx={{ fontSize: 16, color: "#6b7280" }} />
            <Typography sx={{ fontSize: 13.5, color: "#374151" }}>
              Sto analizzando la richiesta e decidendo il prossimo passo…
            </Typography>
          </Box>
        </Box>
      ) : null}

      <Collapse in={expanded}>
        <Box
          sx={{
            display: "flex",
            flexDirection: "column",
            gap: 1.25,
            mt: normalizedSteps.length > 0 ? 1.5 : 0
          }}
        >
          {normalizedSteps.map((step, index) => {
            const status = getStatus(step)
            const inputLabel = formatInput(step.action_input)
            const isLast = index === normalizedSteps.length - 1

            return (
              <Box
                key={`${step.step}-${step.action}-${index}`}
                sx={{
                  display: "grid",
                  gridTemplateColumns: "24px 1fr",
                  gap: 1.25
                }}
              >
                <Box
                  sx={{
                    display: "flex",
                    flexDirection: "column",
                    alignItems: "center",
                    pt: 0.5
                  }}
                >
                  <Box
                    sx={{
                      width: 10,
                      height: 10,
                      borderRadius: "50%",
                      bgcolor: step.status === "error" ? "#b42318" : "#111827"
                    }}
                  />
                  {!isLast ? (
                    <Box
                      sx={{
                        width: 1.5,
                        flex: 1,
                        minHeight: 54,
                        mt: 0.75,
                        bgcolor: "#e5e7eb"
                      }}
                    />
                  ) : null}
                </Box>

                <Box
                  sx={{
                    border: "1px solid #e5e7eb",
                    bgcolor: "#ffffff",
                    borderRadius: 3,
                    px: 2,
                    py: 1.5
                  }}
                >
                  <Box
                    sx={{
                      display: "flex",
                      alignItems: "center",
                      justifyContent: "space-between",
                      gap: 1,
                      flexWrap: "wrap",
                      mb: 0.75
                    }}
                  >
                    <Box display="flex" alignItems="center" gap={1}>
                      <Box
                        sx={{
                          color: "#6b7280",
                          display: "flex",
                          alignItems: "center"
                        }}
                      >
                        {getActionIcon(step.action)}
                      </Box>

                      <Typography
                        sx={{
                          fontSize: 13.5,
                          fontWeight: 700,
                          color: "#111827"
                        }}
                      >
                        Step {step.step ?? index + 1} · {humanizeToolName(step.action)}
                      </Typography>
                    </Box>

                    <Chip
                      size="small"
                      icon={status.icon}
                      label={status.label}
                      sx={{
                        bgcolor: status.accent,
                        border: `1px solid ${status.border}`,
                        color: status.text,
                        "& .MuiChip-label": { fontWeight: 600 }
                      }}
                    />
                  </Box>

                  {step.thought ? (
                    <Typography
                      sx={{
                        fontSize: 13.5,
                        color: "#374151",
                        lineHeight: 1.7,
                        mb: inputLabel || step.observation_summary ? 0.8 : 0
                      }}
                    >
                      {step.thought}
                    </Typography>
                  ) : null}

                  {inputLabel ? (
                    <Typography
                      sx={{
                        fontSize: 12.5,
                        color: "#6b7280",
                        lineHeight: 1.6,
                        mb: step.observation_summary ? 0.6 : 0
                      }}
                    >
                      {inputLabel}
                    </Typography>
                  ) : null}

                  {step.observation_summary ? (
                    <Typography
                      sx={{
                        fontSize: 12.8,
                        color: "#4b5563",
                        lineHeight: 1.65
                      }}
                    >
                      {step.observation_summary}
                    </Typography>
                  ) : null}
                </Box>
              </Box>
            )
          })}
        </Box>
      </Collapse>
    </Box>
  )
}
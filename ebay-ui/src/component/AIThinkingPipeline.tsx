import { Box, Chip, Typography } from "@mui/material"
import CheckCircleIcon from "@mui/icons-material/CheckCircle"
import AutorenewIcon from "@mui/icons-material/Autorenew"
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline"
import SearchIcon from "@mui/icons-material/Search"
import StorefrontIcon from "@mui/icons-material/Storefront"
import PsychologyAltIcon from "@mui/icons-material/PsychologyAlt"

import type { AgentStep } from "./searchTypes.ts"

interface Props {
  agentTrace?: AgentStep[]
  loading?: boolean
  query?: string
}

function getActionLabel(action?: string) {
  switch ((action || "").toLowerCase()) {
    case "search_pipeline":
      return "search_pipeline"
    case "seller_pipeline":
      return "seller_pipeline"
    default:
      return action || "agent_step"
  }
}

function getActionIcon(action?: string) {
  switch ((action || "").toLowerCase()) {
    case "search_pipeline":
      return <SearchIcon sx={{ fontSize: 18 }} />
    case "seller_pipeline":
      return <StorefrontIcon sx={{ fontSize: 18 }} />
    default:
      return <PsychologyAltIcon sx={{ fontSize: 18 }} />
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

  return null
}

export default function AIThinkingPipeline({
  agentTrace = [],
  loading = false
}: Props) {
  if (!loading && agentTrace.length === 0) {
    return null
  }

  return (
    <Box
      sx={{
        minWidth: 320,
        p: 2.5,
        borderRadius: 3,
        bgcolor: "#ffffff",
        border: "1px solid #e5e5e5"
      }}
    >
      <Box display="flex" alignItems="center" justifyContent="space-between" mb={1}>
        <Typography sx={{ fontWeight: 700, fontSize: 14 }}>
          ebayGPT agent trace
        </Typography>

        {loading && (
          <Chip
            size="small"
            icon={<AutorenewIcon sx={{ fontSize: 14 }} />}
            label="in esecuzione"
            sx={{
              bgcolor: "#f5f5f5",
              border: "1px solid #e5e5e5"
            }}
          />
        )}
      </Box>

      <Typography sx={{ fontSize: 12, color: "#8e8ea0", mb: 2 }}>
        L’agente decide quali tool usare, raccoglie osservazioni e costruisce la risposta finale.
      </Typography>

      {agentTrace.length === 0 && loading && (
        <Box
          sx={{
            p: 1.5,
            borderRadius: 2,
            bgcolor: "#fcfcfc",
            border: "1px solid #ececf1"
          }}
        >
          <Typography sx={{ fontSize: 13, color: "#4b4b5a" }}>
            Sto analizzando la richiesta e decidendo il prossimo tool da eseguire…
          </Typography>
        </Box>
      )}

      <Box sx={{ display: "flex", flexDirection: "column", gap: 1.25 }}>
        {agentTrace.map((step, index) => {
          const isError = step.status === "error"
          const accent = isError ? "#d14343" : "#10a37f"
          const inputLabel = formatInput(step.action_input)

          return (
            <Box
              key={`${step.step}-${step.action}-${index}`}
              sx={{
                p: 1.5,
                borderRadius: 2,
                border: "1px solid #ececf1",
                borderLeft: `4px solid ${accent}`,
                bgcolor: "#fcfcfc"
              }}
            >
              <Box display="flex" alignItems="center" gap={1} mb={0.75}>
                {isError ? (
                  <ErrorOutlineIcon sx={{ fontSize: 18, color: accent }} />
                ) : (
                  <CheckCircleIcon sx={{ fontSize: 18, color: accent }} />
                )}

                {getActionIcon(step.action)}

                <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#202123" }}>
                  Step {step.step ?? index + 1} · {getActionLabel(step.action)}
                </Typography>
              </Box>

              {step.thought && (
                <Typography sx={{ fontSize: 13, color: "#4b4b5a", lineHeight: 1.6, mb: 0.75 }}>
                  {step.thought}
                </Typography>
              )}

              {inputLabel && (
                <Typography sx={{ fontSize: 12, color: "#7a7a8c", mb: 0.5 }}>
                  {inputLabel}
                </Typography>
              )}

              {step.observation_summary && (
                <Typography sx={{ fontSize: 12.5, color: "#666", lineHeight: 1.6 }}>
                  {step.observation_summary}
                </Typography>
              )}
            </Box>
          )
        })}
      </Box>
    </Box>
  )
}
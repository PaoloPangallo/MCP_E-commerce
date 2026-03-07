import { useEffect, useMemo, useState } from "react"
import { Box, Chip, LinearProgress, Typography } from "@mui/material"
import CheckCircleIcon from "@mui/icons-material/CheckCircle"
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked"
import AutorenewIcon from "@mui/icons-material/Autorenew"
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline"

import type { AgentStep } from "../component/searchTypes.ts"

interface Props {
  agentTrace?: AgentStep[]
  timings?: Record<string, number>
  query?: string
  loading?: boolean
}

interface Step {
  key: string
  label: string
  backendKeys: string[]
}

const PIPELINE: Step[] = [
  {
    key: "parse",
    label: "Analizzo la query",
    backendKeys: ["parse", "parse_query", "parse_rule_based", "parse_llm"]
  },
  {
    key: "search",
    label: "Cerco prodotti su eBay",
    backendKeys: ["ebay_search", "search_items", "ebay"]
  },
  {
    key: "evidence",
    label: "Recupero segnali su seller e contesto",
    backendKeys: ["feedback", "rag", "retrieve", "ingest"]
  },
  {
    key: "ranking",
    label: "Ricalcolo ranking e affidabilità",
    backendKeys: ["rerank", "rank", "fusion"]
  },
  {
    key: "explain",
    label: "Genero analisi finale",
    backendKeys: ["explain", "analysis", "total"]
  }
]

function getCompletedCountFromTimings(timings?: Record<string, number>) {
  if (!timings) {
    return 0
  }

  const keys = Object.keys(timings)

  return PIPELINE.filter(step =>
    step.backendKeys.some(backendKey =>
      keys.some(key => key.toLowerCase().includes(backendKey.toLowerCase()))
    )
  ).length
}

function formatActionLabel(action?: string) {
  switch ((action || "").toLowerCase()) {
    case "search_pipeline":
      return "Search pipeline"
    case "seller_pipeline":
      return "Seller analysis"
    case "finish":
      return "Finalizzazione"
    default:
      return action || "Step"
  }
}

function formatActionInput(input?: Record<string, unknown>) {
  if (!input) {
    return null
  }

  const query = typeof input.query === "string" ? input.query : null
  const seller = typeof input.seller_name === "string" ? input.seller_name : null

  if (query) {
    return `query: ${query}`
  }

  if (seller) {
    return `seller: ${seller}`
  }

  return null
}

export default function AIThinkingPipeline({
  loading = false,
  timings,
  query,
  agentTrace
}: Props) {
  const [simulatedStepIndex, setSimulatedStepIndex] = useState(0)
  const [simulatedProgress, setSimulatedProgress] = useState(8)

  const realTrace = useMemo(
    () => (Array.isArray(agentTrace) ? agentTrace.filter(Boolean) : []),
    [agentTrace]
  )

  useEffect(() => {
    if (!loading || timings || realTrace.length > 0) {
      return
    }

    setSimulatedStepIndex(0)
    setSimulatedProgress(8)

    const interval = window.setInterval(() => {
      setSimulatedProgress(prev => Math.min(prev + 9, 88))
      setSimulatedStepIndex(prev => Math.min(prev + 1, PIPELINE.length - 1))
    }, 950)

    return () => window.clearInterval(interval)
  }, [loading, timings, query, realTrace.length])

  const completedCount = useMemo(
    () => getCompletedCountFromTimings(timings),
    [timings]
  )

  const progress = timings
    ? 100
    : loading
      ? simulatedProgress
      : 0

  if (!loading && !timings && realTrace.length === 0) {
    return null
  }

  if (realTrace.length > 0) {
    const totalTime = timings?.total_s

    return (
      <Box
        sx={{
          minWidth: 320,
          p: 2.5,
          mb: 3,
          borderRadius: 3,
          bgcolor: "#ffffff",
          border: "1px solid #e5e5e5"
        }}
      >
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            gap: 2,
            mb: 1
          }}
        >
          <Typography sx={{ fontWeight: 600, fontSize: 14 }}>
            Agent trace
          </Typography>

          {typeof totalTime === "number" && (
            <Chip
              size="small"
              label={`Total ${totalTime.toFixed(2)}s`}
              sx={{
                bgcolor: "#f5f5f5",
                border: "1px solid #e5e5e5",
                fontSize: 11
              }}
            />
          )}
        </Box>

        <Typography sx={{ fontSize: 12, color: "#8e8ea0", mb: 2 }}>
          Step reali eseguiti dal backend agentico.
        </Typography>

        <Box sx={{ display: "flex", flexDirection: "column", gap: 1.25 }}>
          {realTrace.map(step => {
            const isError = step.status === "error"
            const isFinal = step.status === "final"
            const accent = isError ? "#d14343" : isFinal ? "#3b5ccc" : "#10a37f"
            const inputLabel = formatActionInput(step.action_input)

            return (
              <Box
                key={`${step.step}-${step.action}-${step.observation_summary}`}
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

                  <Typography sx={{ fontSize: 13, fontWeight: 600, color: "#202123" }}>
                    Step {step.step} · {formatActionLabel(step.action)}
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
      <Typography sx={{ fontWeight: 600, fontSize: 14, mb: 0.5 }}>
        {loading ? "AI reasoning in progress" : "AI reasoning completed"}
      </Typography>

      <Typography sx={{ fontSize: 12, color: "#8e8ea0", mb: 2 }}>
        {loading && !timings
          ? "Visualizzazione temporanea mentre il backend agentico completa il loop ReAct."
          : "Step stimati dai timing della search pipeline."}
      </Typography>

      {PIPELINE.map((step, index) => {
        const completed = timings ? index < completedCount : index < simulatedStepIndex
        const running = loading && !timings ? index === simulatedStepIndex : false
        const stepTime = timings
          ? Object.entries(timings).find(([key]) =>
              step.backendKeys.some(backendKey =>
                key.toLowerCase().includes(backendKey.toLowerCase())
              )
            )?.[1]
          : null

        return (
          <Box key={step.key} display="flex" alignItems="center" gap={1.2} mb={1}>
            {completed && <CheckCircleIcon sx={{ fontSize: 18, color: "#10a37f" }} />}

            {running && (
              <AutorenewIcon
                sx={{
                  fontSize: 18,
                  color: "#10a37f",
                  animation: "spin 1s linear infinite",
                  "@keyframes spin": {
                    from: { transform: "rotate(0deg)" },
                    to: { transform: "rotate(360deg)" }
                  }
                }}
              />
            )}

            {!completed && !running && (
              <RadioButtonUncheckedIcon sx={{ fontSize: 18, color: "#bbb" }} />
            )}

            <Typography
              sx={{
                fontSize: 13,
                color: completed ? "#202123" : running ? "#10a37f" : "#999",
                fontWeight: running ? 600 : 400
              }}
            >
              {step.label}
            </Typography>

            {typeof stepTime === "number" && (
              <Typography sx={{ fontSize: 12, color: "#888" }}>
                ({stepTime.toFixed(2)}s)
              </Typography>
            )}
          </Box>
        )
      })}

      <LinearProgress
        variant="determinate"
        value={progress}
        sx={{
          mt: 2,
          height: 5,
          borderRadius: 2,
          backgroundColor: "#ececec",
          "& .MuiLinearProgress-bar": {
            backgroundColor: "#10a37f"
          }
        }}
      />
    </Box>
  )
}
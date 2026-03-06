import { useEffect, useMemo, useState } from "react"
import { Box, LinearProgress, Typography } from "@mui/material"
import CheckCircleIcon from "@mui/icons-material/CheckCircle"
import RadioButtonUncheckedIcon from "@mui/icons-material/RadioButtonUnchecked"
import AutorenewIcon from "@mui/icons-material/Autorenew"

interface Props {
  loading?: boolean
  timings?: Record<string, number>
  query?: string
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

export default function AIThinkingPipeline({ loading = false, timings, query }: Props) {
  const [simulatedStepIndex, setSimulatedStepIndex] = useState(0)
  const [simulatedProgress, setSimulatedProgress] = useState(8)

  useEffect(() => {
    if (!loading || timings) {
      return
    }

    setSimulatedStepIndex(0)
    setSimulatedProgress(8)

    const interval = window.setInterval(() => {
      setSimulatedProgress(prev => Math.min(prev + 9, 88))
      setSimulatedStepIndex(prev => Math.min(prev + 1, PIPELINE.length - 1))
    }, 950)

    return () => window.clearInterval(interval)
  }, [loading, timings, query])

  const completedCount = useMemo(
    () => getCompletedCountFromTimings(timings),
    [timings]
  )

  const progress = timings
    ? 100
    : loading
      ? simulatedProgress
      : 0

  if (!loading && !timings) {
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
      <Typography sx={{ fontWeight: 600, fontSize: 14, mb: 0.5 }}>
        {loading ? "AI reasoning in progress" : "AI reasoning completed"}
      </Typography>

      <Typography sx={{ fontSize: 12, color: "#8e8ea0", mb: 2 }}>
        {loading && !timings
          ? "Visualizzazione dello stato frontend in attesa della risposta finale del backend."
          : "Step confermati dai timing restituiti dalla route search."}
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

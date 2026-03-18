import { useEffect, useState } from "react"
import { Box, Collapse, Paper } from "@mui/material"
import KeyboardArrowDownIcon from "@mui/icons-material/KeyboardArrowDown"

import { AIThinkingPipeline } from "./AIThinkingPipeline"
import type { AgentStep, PlannedTask } from "../types"

interface ThinkingPillProps {
  steps: AgentStep[]
  loading: boolean
  query?: string
  plannedTasks?: PlannedTask[]
  defaultOpen?: boolean
}

export function ThinkingPill({ steps, loading, query, plannedTasks, defaultOpen }: ThinkingPillProps) {
  // Se stiamo attivamente "pensando", apriamo il collapse per far vedere i passi,
  // altrimenti rispettiamo "defaultOpen" (che di solito è false per le chat storiche)
  const [open, setOpen] = useState(defaultOpen ?? loading)

  // Auto-chiudi la pillola quando l'agente finisce, così scompare elegantemente. 
  // Usa useEffect per farlo solo al cambio di stato
  useEffect(() => {
    if (!loading) {
      setOpen(false)
    }
  }, [loading])

  const label = loading
    ? "L'agente sta ragionando…"
    : `Ragionamento completato${steps.length > 0 ? ` · ${steps.length} passi` : ""}`

  if (!loading && steps.length === 0 && (!plannedTasks || plannedTasks.length === 0)) {
    return null
  }

  return (
    <Box sx={{ mb: 1.5 }}>
      <Box
        onClick={() => setOpen((v) => !v)}
        sx={{
          display: "inline-flex",
          alignItems: "center",
          gap: 0.75,
          px: 1.5,
          py: 0.6,
          mb: open ? 1 : 0,
          border: "1px solid #e5e7eb",
          borderRadius: "20px",
          cursor: "pointer",
          bgcolor: "#fafafa",
          transition: "background 0.15s, border-color 0.15s",
          "&:hover": { bgcolor: "#f3f4f6", borderColor: "#d1d5db" },
          userSelect: "none"
        }}
      >
        <Box
          sx={{
            width: 6,
            height: 6,
            borderRadius: "50%",
            bgcolor: loading ? "#f59e0b" : "#10a37f",
            animation: loading ? "dotPulse 1.2s infinite ease-in-out" : "none",
            "@keyframes dotPulse": {
              "0%, 100%": { opacity: 1 },
              "50%": { opacity: 0.35 }
            }
          }}
        />
        <Box component="span" sx={{ fontSize: 12, color: "#6b7280", lineHeight: 1 }}>
          {label}
        </Box>
        <KeyboardArrowDownIcon
          sx={{
            fontSize: 14,
            color: "#9ca3af",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.2s"
          }}
        />
      </Box>

      <Collapse in={open} timeout={250}>
        <Paper
          elevation={0}
          sx={{ p: 2, borderRadius: 3, border: "1px solid #e5e7eb", bgcolor: "#f8fafc" }}
        >
          <AIThinkingPipeline
            agentTrace={steps}
            loading={loading}
            query={query}
            plannedTasks={plannedTasks}
          />
        </Paper>
      </Collapse>
    </Box>
  )
}

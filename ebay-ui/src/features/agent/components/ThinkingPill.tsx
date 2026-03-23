import { useEffect, useState } from "react"
import {Box, Collapse, Paper, Typography} from "@mui/material"
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
  const [open, setOpen] = useState(defaultOpen ?? loading)

  useEffect(() => {
    if (!loading && !defaultOpen) {
      setOpen(false)
    }
  }, [loading, defaultOpen])

  const label = loading
    ? "L'agente sta pianificando…"
    : `Analisi completata${steps.length > 0 ? ` · ${steps.length} passi` : ""}`

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
          gap: 1,
          px: 1.75,
          py: 0.75,
          mb: open ? 1 : 0,
          border: "1px solid var(--border-color)",
          borderRadius: "24px",
          cursor: "pointer",
          bgcolor: loading ? "var(--bg-secondary)" : "var(--bg-primary)",
          boxShadow: loading ? "0 2px 8px rgba(59, 130, 246, 0.08)" : "0 1px 3px rgba(0,0,0,0.02)",
          transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": { 
            bgcolor: "var(--bg-secondary)", 
            borderColor: "var(--text-secondary)",
            transform: "translateY(-1px)"
          },
          userSelect: "none"
        }}
      >
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            bgcolor: loading ? "var(--accent-primary)" : "#10b981",
            animation: loading ? "dotPulse 1.8s infinite ease-in-out" : "none",
            boxShadow: loading ? "0 0 6px rgba(59, 130, 246, 0.4)" : "none",
            "@keyframes dotPulse": {
              "0%, 100%": { transform: "scale(0.8)", opacity: 0.5 },
              "50%": { transform: "scale(1.2)", opacity: 1 }
            }
          }}
        />
        <Typography 
          sx={{ 
            fontSize: 12, 
            fontWeight: 600, 
            color: "var(--text-primary)", 
            lineHeight: 1,
            letterSpacing: '-0.01em'
          }}
        >
          {label}
        </Typography>
        <KeyboardArrowDownIcon
          sx={{
            fontSize: 16,
            color: "var(--text-secondary)",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)"
          }}
        />
      </Box>

      <Collapse in={open} timeout={300}>
        <Paper
          elevation={0}
          sx={{ 
            p: 2.5, 
            border: "1px solid var(--border-color)", 
            bgcolor: "var(--bg-primary)",
            boxShadow: "0 4px 15px rgba(0,0,0,0.03)"
          }}
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

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
  hideTrace?: boolean
}

export function ThinkingPill({ 
  steps, 
  loading, 
  query, 
  plannedTasks, 
  defaultOpen,
  hideTrace
}: ThinkingPillProps) {
  const [open, setOpen] = useState(defaultOpen ?? (loading && !hideTrace))

  useEffect(() => {
    if (!loading && !defaultOpen) {
      setOpen(false)
    }
  }, [loading, defaultOpen])

  const label = loading
    ? "L'agente sta pianificando…"
    : `Analisi completata${steps.length > 0 ? ` · ${steps.length} passi` : ""}`

  if (hideTrace || (!loading && steps.length === 0 && (!plannedTasks || plannedTasks.length === 0))) {
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
          border: "1px solid",
          borderColor: loading ? "rgba(59, 130, 246, 0.3)" : "rgba(16, 185, 129, 0.3)",
          borderRadius: "24px",
          cursor: "pointer",
          bgcolor: loading 
            ? "rgba(59, 130, 246, 0.05)" 
            : "rgba(16, 185, 129, 0.05)",
          backdropFilter: "blur(8px)",
          boxShadow: loading 
            ? "0 0 15px rgba(59, 130, 246, 0.15)" 
            : "0 2px 8px rgba(0,0,0,0.1)",
          transition: "all 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": { 
            bgcolor: loading ? "rgba(59, 130, 246, 0.1)" : "rgba(16, 185, 129, 0.1)", 
            borderColor: loading ? "rgba(59, 130, 246, 0.5)" : "rgba(16, 185, 129, 0.5)",
            transform: "translateY(-1px)",
            boxShadow: loading ? "0 0 20px rgba(59, 130, 246, 0.25)" : "0 4px 12px rgba(0,0,0,0.15)"
          },
          userSelect: "none"
        }}
      >
        <Box
          sx={{
            width: 8,
            height: 8,
            borderRadius: "50%",
            bgcolor: loading ? "#3b82f6" : "#10b981",
            animation: loading ? "dotPulse 1.8s infinite ease-in-out" : "none",
            boxShadow: loading 
              ? "0 0 10px rgba(59, 130, 246, 0.8)" 
              : "0 0 8px rgba(16, 185, 129, 0.6)",
            "@keyframes dotPulse": {
              "0%, 100%": { transform: "scale(0.8)", opacity: 0.6 },
              "50%": { transform: "scale(1.2)", opacity: 1 }
            }
          }}
        />
        <Typography 
          sx={{ 
            fontSize: 12, 
            fontWeight: 800, 
            color: loading ? "#60a5fa" : "#34d399", 
            lineHeight: 1,
            letterSpacing: '0.02em',
            textTransform: 'uppercase',
            textShadow: loading 
              ? "0 0 12px rgba(59, 130, 246, 0.4)" 
              : "0 0 10px rgba(16, 185, 129, 0.3)"
          }}
        >
          {label}
        </Typography>
        <KeyboardArrowDownIcon
          sx={{
            fontSize: 16,
            color: loading ? "#60a5fa" : "#34d399",
            transform: open ? "rotate(180deg)" : "none",
            transition: "transform 0.3s cubic-bezier(0.4, 0, 0.2, 1)",
            opacity: 0.8
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

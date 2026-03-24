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
          border: "1px solid #e567eb", // Subtle magenta hint for agent? No, let's stick to blue/gray
          borderColor: loading ? "#bfdbfe" : "#e5e7eb",
          borderRadius: "24px",
          cursor: "pointer",
          bgcolor: loading ? "#f0f7ff" : "#fff",
          boxShadow: loading ? "0 2px 8px rgba(59, 130, 246, 0.08)" : "0 1px 3px rgba(0,0,0,0.02)",
          transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
          "&:hover": { 
            bgcolor: loading ? "#eff6ff" : "#f9fafb", 
            borderColor: loading ? "#3b82f6" : "#d1d5db",
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
            bgcolor: loading ? "#3b82f6" : "#10b981",
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
            color: loading ? "#1e40af" : "#4b5563", 
            lineHeight: 1,
            letterSpacing: '-0.01em'
          }}
        >
          {label}
        </Typography>
        <KeyboardArrowDownIcon
          sx={{
            fontSize: 16,
            color: "#9ca3af",
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
            borderRadius: "16px", 
            border: "1px solid #f0f0f0", 
            bgcolor: "#fff",
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

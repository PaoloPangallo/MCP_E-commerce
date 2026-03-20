import { useMemo, useState } from "react"
import { Box, Typography, keyframes, Collapse } from "@mui/material"

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

const pulse = keyframes`
  0% { transform: scale(0.95); opacity: 0.5; }
  50% { transform: scale(1.05); opacity: 1; }
  100% { transform: scale(0.95); opacity: 0.5; }
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
  const sx = { fontSize: 13, color: "inherit" }
  switch (val) {
    case "search_pipeline":     return <SearchIcon sx={sx} />
    case "seller_pipeline":     return <StorefrontIcon sx={sx} />
    case "weather_tool":        return <CloudOutlinedIcon sx={sx} />
    case "price_history_tool":  return <InsightsOutlinedIcon sx={sx} />
    case "finish":              return <CheckCircleOutlineIcon sx={sx} />
    default:                    return <BuildCircleOutlinedIcon sx={sx} />
  }
}

function StepRow({
  step,
  isLast,
  loading,
  totalSteps
}: {
  step: AgentStep
  isLast: boolean
  loading: boolean
  totalSteps: number
}) {
  const [showJson, setShowJson] = useState(false)
  const isRunning = (step.status === "thinking" || step.status === "running") && loading
  const isError   = step.status === "error"
  const isDone    = step.status === "ok" || step.status === "final" || !loading

  const stepNum = step.step ?? 1

  // Extract interesting params for badges
  const params = step.action_input || {}
  const badges = [
    params.query && { label: "Query", value: params.query },
    params.seller_name && { label: "Seller", value: params.seller_name },
    params.ebay_id && { label: "ID", value: params.ebay_id },
    params.product && { label: "Prod", value: params.product }
  ].filter(Boolean) as { label: string, value: string }[]

  return (
    <Box sx={{ display: "flex", gap: 2, position: "relative" }}>
      {/* Timeline spine */}
      <Box sx={{ display: "flex", flexDirection: "column", alignItems: "center", flexShrink: 0, width: 24 }}>
        {/* dot/node */}
        <Box
          sx={{
            width: 10,
            height: 10,
            borderRadius: "50%",
            mt: "6px",
            flexShrink: 0,
            bgcolor: isError ? "#ef4444" : isRunning ? "#3b82f6" : "#10b981",
            boxShadow: isRunning ? "0 0 8px rgba(59, 130, 246, 0.5)" : "none",
            animation: isRunning ? `${pulse} 1.5s ease-in-out infinite` : "none",
            border: "2px solid #fff",
            zIndex: 1,
            display: 'flex',
            alignItems: 'center',
            justifyContent: 'center',
            fontSize: 6,
            color: '#fff',
            fontWeight: 700
          }}
        >
          {stepNum}
        </Box>
        {/* vertical line */}
        {!isLast && (
          <Box 
            sx={{ 
              width: "2px", 
              flex: 1, 
              background: isDone ? "linear-gradient(to bottom, #10b981, #f0f0f0)" : "#f0f0f0", 
              mt: 0.5,
              opacity: 0.6 
            }} 
          />
        )}
      </Box>

      {/* Content */}
      <Box sx={{ pb: isLast ? 0 : 2.5, minWidth: 0, flex: 1 }}>
        {/* Action label */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5, flexWrap: 'wrap' }}>
          <Box 
            sx={{ 
              display: 'flex', 
              p: 0.5, 
              borderRadius: '6px', 
              bgcolor: isRunning ? '#eff6ff' : isDone ? '#ecfdf5' : '#f9fafb',
              color: isRunning ? '#3b82f6' : isDone ? '#059669' : '#9ca3af'
            }}
          >
            {getActionIcon(step.action)}
          </Box>
          <Typography sx={{ fontSize: 13, fontWeight: 700, color: isDone ? "#111827" : "#4b5563" }}>
            {humanizeToolName(step.action, step.action_input)}
            <Box component="span" sx={{ ml: 1, fontSize: 10, color: '#9ca3af', fontWeight: 500 }}>
              [{stepNum}/{totalSteps}]
            </Box>
          </Typography>

          {badges.map((b, i) => (
             <Box 
               key={i} 
               sx={{ 
                 px: 0.75, 
                 py: 0.1, 
                 borderRadius: '4px', 
                 bgcolor: '#f3f4f6', 
                 border: '1px solid #e5e7eb',
                 display: 'flex',
                 alignItems: 'center',
                 gap: 0.5
               }}
             >
               <Typography sx={{ fontSize: 9, fontWeight: 700, color: '#6b7280', textTransform: 'uppercase' }}>
                 {b.label}:
               </Typography>
               <Typography sx={{ fontSize: 10, color: '#374151', maxWidth: 100, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
                 {b.value}
               </Typography>
             </Box>
          ))}

          {isRunning && (
             <Box
              sx={{
                width: 4,
                height: 4,
                borderRadius: "50%",
                bgcolor: "#3b82f6",
                animation: `${pulse} 1s infinite`
              }}
            />
          )}

          <Box sx={{ ml: 'auto' }}>
            <Typography 
              onClick={() => setShowJson(!showJson)}
              sx={{ 
                fontSize: 10, 
                color: '#9ca3af', 
                cursor: 'pointer', 
                '&:hover': { color: '#3b82f6', textDecoration: 'underline' },
                userSelect: 'none'
              }}
            >
              {showJson ? "Nascondi JSON" : "Dettagli JSON"}
            </Typography>
          </Box>

          {isError && (
             <ErrorOutlineIcon sx={{ fontSize: 13, color: "#ef4444" }} />
          )}
        </Box>

        {/* JSON Details */}
        <Collapse in={showJson} timeout={200}>
          <Box 
            sx={{ 
              mt: 1, 
              p: 1.5, 
              bgcolor: '#1f2937', 
              borderRadius: '8px', 
              overflowX: 'auto',
              border: '1px solid #374151'
            }}
          >
            <Typography component="pre" sx={{ fontSize: 10, color: '#e5e7eb', fontFamily: 'monospace', m: 0 }}>
              {JSON.stringify(step.action_input, null, 2)}
            </Typography>
          </Box>
        </Collapse>

        {/* Thought */}
        {step.thought && (
          <Typography 
            sx={{ 
              fontSize: 12, 
              color: "#6b7280", 
              lineHeight: 1.6, 
              my: 0.75,
              pl: 1,
              borderLeft: '2px solid #f3f4f6'
            }}
          >
            {step.thought}
          </Typography>
        )}

        {/* Observation */}
        {step.observation_summary && (
          <Box
            sx={{
              p: 1.25,
              bgcolor: "#fafafa",
              borderRadius: "8px",
              border: "1px solid #f0f0f0",
              mt: 0.5,
              position: 'relative',
              '&:before': {
                content: '"OBSERVATION"',
                position: 'absolute',
                top: -6,
                right: 12,
                bgcolor: '#fff',
                px: 0.5,
                fontSize: 8,
                fontWeight: 800,
                color: '#9ca3af',
                letterSpacing: 0.5
              }
            }}
          >
            <Typography
              sx={{
                fontSize: 11,
                color: "#4b5563",
                lineHeight: 1.5,
                fontStyle: "italic"
              }}
            >
              {step.observation_summary}
            </Typography>
          </Box>
        )}
      </Box>
    </Box>
  )
}

export function AIThinkingPipeline({
  agentTrace = [],
  loading = false,
  plannedTasks = []
}: Props) {
  const steps = useMemo(
    () => agentTrace.filter(Boolean).sort((a, b) => (a.step ?? 0) - (b.step ?? 0)),
    [agentTrace]
  )

  const totalPossibleSteps = Math.max(steps.length + plannedTasks.slice(steps.length).length, steps.length)

  if (!loading && steps.length === 0 && plannedTasks.length === 0) return null

  const pendingTasks = plannedTasks.slice(steps.length)

  return (
    <Box sx={{ py: 1 }}>
      {/* Steps */}
      {steps.map((step, i) => (
        <StepRow
          key={`${step.step}-${i}`}
          step={step}
          isLast={i === steps.length - 1 && !loading && pendingTasks.length === 0}
          loading={loading}
          totalSteps={totalPossibleSteps}
        />
      ))}

      {/* Pending planned tasks */}
      {pendingTasks.length > 0 && (
        <Box sx={{ display: "flex", gap: 1, flexWrap: "wrap", mt: 1, pl: "38px" }}>
          {pendingTasks.map((t, i) => (
            <Box 
              key={i} 
              sx={{ 
                display: 'flex', 
                alignItems: 'center', 
                gap: 0.5,
                px: 1,
                py: 0.25,
                bgcolor: '#f9fafb',
                borderRadius: '6px',
                border: '1px dashed #e5e7eb'
              }}
            >
              <Typography sx={{ fontSize: 11, color: "#9ca3af", fontWeight: 500 }}>
                {humanizeToolName(t.tool)}
              </Typography>
            </Box>
          ))}
        </Box>
      )}

      {/* Loading spinner row */}
      {loading && (
        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, pl: "40px", mt: 1.5 }}>
          <Box
            sx={{
              width: 12,
              height: 12,
              borderRadius: "50%",
              border: "2px solid #e5e7eb",
              borderTopColor: "#3b82f6",
              animation: `${spin} 0.8s linear infinite`,
              flexShrink: 0
            }}
          />
          <Typography sx={{ fontSize: 12, color: "#6b7280", fontWeight: 500, letterSpacing: '0.01em' }}>
            generazione risposta in corso…
          </Typography>
        </Box>
      )}
    </Box>
  )
}
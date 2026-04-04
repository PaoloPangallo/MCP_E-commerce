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

const borderFlow = keyframes`
  0% { background-position: 0% 0%; }
  100% { background-position: 0% 100%; }
`

const typeWriter = keyframes`
  0% { opacity: 0.3; transform: translateX(-2px); }
  50% { opacity: 1; transform: translateX(0); }
  100% { opacity: 0.3; transform: translateX(2px); }
`

interface Props {
  agentTrace?: AgentStep[]
  loading?: boolean
  query?: string
  plannedTasks?: PlannedTask[]
}

export function humanizeToolName(action?: string, input?: any) {
  const val = (action || "").toLowerCase()
  const q = input?.query || input?.product || ""
  const seller = input?.seller_name || ""

  switch (val) {
    case "search_products":
      return q ? `Cerco "${q}" su eBay` : "Ricerca prodotti"
    case "compare_products":
      return q ? `Confronto "${q}"` : "Confronto prodotti"
    case "profile_query":
      return "Ricerca potenziata da Profilo"
    case "analyze_seller":
      return seller ? `Analizzo venditore ${seller}` : "Analisi venditore"
    case "get_item_details":
      return "Estrazione dettagli oggetto"
    case "get_similar_items":
      return "Ricerca oggetti simili"
    case "get_shipping_costs":
      return "Calcolo spedizione"
    case "get_marketplace_metadata":
      return "Ricerca metadati marketplace"
    case "get_ebay_deals":
      return "Ricerca offerte eBay"
    case "market_trends":
      return "Analisi trend di mercato"
    case "conversation":
      return "Azione conversazionale"
    case "finish":
      return "Sintesi risultati"
    // Legacy fallbacks
    case "search_pipeline":
      return q ? `Cerco "${q}" su eBay` : "Ricerca prodotti"
    case "seller_pipeline":
      return seller ? `Analizzo ${seller}` : "Analisi venditore"
    case "price_history_tool":
      return "Storico prezzi"
    default:
      return action || "Elaborazione"
  }
}

function getActionIcon(action?: string) {
  const val = (action || "").toLowerCase()
  const sx = { fontSize: 13, color: "inherit" }
  
  if (val.includes("search") || val.includes("profile") || val.includes("similar")) return <SearchIcon sx={sx} />
  if (val.includes("seller")) return <StorefrontIcon sx={sx} />
  if (val.includes("shipping")) return <CloudOutlinedIcon sx={sx} />
  if (val.includes("metadata") || val.includes("compare")) return <InsightsOutlinedIcon sx={sx} />
  if (val === "finish") return <CheckCircleOutlineIcon sx={sx} />
  return <BuildCircleOutlinedIcon sx={sx} />
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
  const displayQuery = step.ebay_query || params.query
  const badges = [
    displayQuery && { label: "Query", value: displayQuery },
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
              background: isRunning 
                ? "linear-gradient(180deg, #3b82f6 0%, rgba(59, 130, 246, 0.1) 50%, #3b82f6 100%)" 
                : isDone 
                  ? "linear-gradient(to bottom, #10b981, rgba(255,255,255,0.05))" 
                  : "rgba(255,255,255,0.05)",
              backgroundSize: "100% 200%",
              animation: isRunning ? `${borderFlow} 1.5s linear infinite` : "none",
              mt: 0.5,
              opacity: isRunning ? 0.8 : 0.4 
            }} 
          />
        )}
      </Box>

      {/* Content */}
      <Box sx={{ pb: isLast ? 0 : 2.5, minWidth: 0, flex: 1, 
         animation: isRunning ? "slideIn 0.3s ease-out forwards" : "none",
         "@keyframes slideIn": {
           "0%": { opacity: 0, transform: "translateX(-10px)" },
           "100%": { opacity: 1, transform: "translateX(0)" }
         }
      }}>
        {/* Action label */}
        <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 0.5, flexWrap: 'wrap' }}>
          <Box 
            sx={{ 
              display: 'flex', 
              p: 0.5, 
              borderRadius: '6px', 
              bgcolor: isRunning ? 'rgba(59, 130, 246, 0.15)' : isDone ? 'rgba(16, 185, 129, 0.1)' : 'rgba(255,255,255,0.05)',
              color: isRunning ? '#60a5fa' : isDone ? '#34d399' : '#9ca3af'
            }}
          >
            {getActionIcon(step.action)}
          </Box>
          <Typography sx={{ fontSize: 13, fontWeight: 700, color: isDone ? "var(--text-primary)" : "var(--text-secondary)" }}>
            {humanizeToolName(step.action, step.action_input)}
            <Box component="span" sx={{ ml: 1, fontSize: 10, color: 'var(--text-secondary)', opacity: 0.6, fontWeight: 500 }}>
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
                 bgcolor: 'rgba(255,255,255,0.03)', 
                 border: '1px solid rgba(255,255,255,0.08)',
                 display: 'flex',
                 alignItems: 'center',
                 gap: 0.5
               }}
             >
               <Typography sx={{ fontSize: 9, fontWeight: 700, color: 'var(--text-secondary)', opacity: 0.7, textTransform: 'uppercase' }}>
                 {b.label}:
               </Typography>
               <Typography sx={{ fontSize: 10, color: 'var(--text-primary)', maxWidth: 180, overflow: 'hidden', textOverflow: 'ellipsis', whiteSpace: 'nowrap' }}>
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
              color: isRunning ? "#60a5fa" : "var(--text-secondary)", 
              lineHeight: 1.6, 
              my: 0.75,
              pl: 1.5,
              borderLeft: isRunning ? '2px solid #3b82f6' : '2px solid rgba(255,255,255,0.05)',
              animation: isRunning ? `${typeWriter} 2s ease-in-out infinite` : "none",
              position: 'relative',
              fontWeight: isRunning ? 500 : 400
            }}
          >
            {step.thought}
          </Typography>
        )}

        {/* Observation */}
        {step.observation_summary && (
          <Box
            sx={{
              p: 1.5,
              bgcolor: "rgba(255,255,255,0.02)",
              borderRadius: "12px",
              border: "1px solid rgba(255,255,255,0.06)",
              mt: 1,
              position: 'relative',
              '&:before': {
                content: '"OBSERVATION"',
                position: 'absolute',
                top: -8,
                right: 16,
                bgcolor: 'var(--bg-primary)',
                px: 1,
                py: 0.2,
                borderRadius: '4px',
                fontSize: 8,
                fontWeight: 900,
                color: isDone ? '#10b981' : '#60a5fa',
                letterSpacing: 1,
                border: '1px solid rgba(255,255,255,0.1)',
                boxShadow: '0 2px 4px rgba(0,0,0,0.2)'
              }
            }}
          >
            <Typography
              sx={{
                fontSize: 11.5,
                color: "var(--text-primary)",
                opacity: 0.9,
                lineHeight: 1.6,
                fontStyle: "italic"
              }}
            >
              {step.observation_summary}
            </Typography>
          </Box>
        )}

        {/* Screenshot (Playwright visual feedback) */}
        {step.observation_data?.screenshot && (
          <Box sx={{ mt: 1.5, position: 'relative', borderRadius: "12px", overflow: "hidden", border: "1px solid rgba(255,255,255,0.1)" }}>
            <img 
              src={`data:image/jpeg;base64,${step.observation_data.screenshot}`} 
              alt="Browser Render" 
              style={{ display: "block", maxWidth: "100%", height: "auto", objectFit: "cover" }} 
            />
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
import { useEffect, useState } from "react"
import { Box, LinearProgress, Typography, Collapse, IconButton } from "@mui/material"
import CheckCircleIcon from "@mui/icons-material/CheckCircle"
import AutorenewIcon from "@mui/icons-material/Autorenew"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ExpandLessIcon from "@mui/icons-material/ExpandLess"

interface Props {
  loading?: boolean
  timings?: Record<string, number>
  query?: string
  trace?: string[]
}

export default function AIThinkingPipeline({ loading = false, timings, query, trace }: Props) {
  const [simulatedProgress, setSimulatedProgress] = useState(8)
  const [expanded, setExpanded] = useState(false)

  useEffect(() => {
    if (!loading || timings || trace) return

    setSimulatedProgress(8)

    const interval = window.setInterval(() => {
      setSimulatedProgress(prev => Math.min(prev + 9, 88))
    }, 950)

    return () => window.clearInterval(interval)
  }, [loading, timings, trace, query])

  if (!loading && !timings && (!trace || trace.length === 0)) {
    return null
  }

  // Se abbiamo una traccia dinamica dal backend, diamo priorità a mostrare quella una volta finita!
  const showDynamicTrace = trace && trace.length > 0;

  return (
    <Box sx={{ minWidth: 320, p: 2.5, borderRadius: 3, bgcolor: "#ffffff", border: "1px solid #e5e5e5" }}>
      <Box display="flex" justifyContent="space-between" alignItems="center" mb={showDynamicTrace ? 0 : 0.5}>
        <Typography sx={{ fontWeight: 600, fontSize: 14 }}>
          {loading ? "Agentic reasoning in progress..." : "Agentic Tracing"}
        </Typography>
        {showDynamicTrace && !loading && (
          <IconButton size="small" onClick={() => setExpanded(!expanded)}>
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        )}
      </Box>

      {(!showDynamicTrace || loading) && (
        <Typography sx={{ fontSize: 12, color: "#8e8ea0", mb: 2 }}>
          {loading && !timings
            ? "L'agente sta pianificando l'estrazione e i tool da utilizzare..."
            : "Step della pipeline confermati."}
        </Typography>
      )}

      {(!showDynamicTrace && loading) ? (
        <Box mt={2}>
          <Box display="flex" alignItems="center" gap={1.2} mb={1}>
            <AutorenewIcon
              sx={{
                fontSize: 18, color: "#10a37f", animation: "spin 1s linear infinite",
                "@keyframes spin": { from: { transform: "rotate(0deg)" }, to: { transform: "rotate(360deg)" } }
              }}
            />
            <Typography sx={{ fontSize: 13, color: "#10a37f", fontWeight: 600 }}>
              Elaborazione del ragionamento in corso...
            </Typography>
          </Box>
        </Box>
      ) : showDynamicTrace && !loading ? (
        <Collapse in={expanded} collapsedSize={40}>
          <Box mt={1}>
            {trace.map((t, idx) => (
              <Box key={idx} display="flex" alignItems="center" gap={1.2} mb={1}>
                <CheckCircleIcon sx={{ fontSize: 16, color: "#10a37f" }} />
                <Typography sx={{ fontSize: 13, color: "#202123" }}>{t}</Typography>
              </Box>
            ))}
          </Box>
        </Collapse>
      ) : null}

      {loading && (
        <LinearProgress variant="determinate" value={simulatedProgress} sx={{ mt: 2, height: 5, borderRadius: 2, backgroundColor: "#ececec", "& .MuiLinearProgress-bar": { backgroundColor: "#10a37f" } }} />
      )}
    </Box>
  )
}

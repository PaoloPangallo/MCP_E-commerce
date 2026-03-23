import { Box, Skeleton, Paper, Typography } from "@mui/material"
import TipsAndUpdatesIcon from "@mui/icons-material/TipsAndUpdates"

interface ToolSkeletonProps {
  type: "search" | "comparison" | "seller" | "details" | "trends" | "deals"
}

export default function ToolSkeleton({ type }: ToolSkeletonProps) {
  const getLabel = () => {
    switch (type) {
      case "search": return "Ricerca prodotti in corso..."
      case "comparison": return "Analisi comparativa in corso..."
      case "seller": return "Analisi affidabilità venditore..."
      case "details": return "Recupero dettagli prodotto..."
      case "trends": return "Analisi trend di mercato..."
      case "deals": return "Ricerca offerte attive..."
      default: return "Elaborazione dati..."
    }
  }

  return (
    <Paper 
      elevation={0}
      sx={{ 
        p: 3, 
        borderRadius: "16px", 
        border: "1px solid #e2e8f0",
        background: "linear-gradient(135deg, #ffffff 0%, #f8fafc 100%)",
        position: "relative",
        overflow: "hidden"
      }}
    >
      <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 3 }}>
        <TipsAndUpdatesIcon sx={{ color: "#7c3aed", fontSize: 20 }} />
        <Typography sx={{ fontSize: 13, fontWeight: 700, color: "#475569", textTransform: "uppercase", letterSpacing: "0.05em" }}>
          {getLabel()}
        </Typography>
      </Box>

      {type === "comparison" ? (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 3 }}>
          <Skeleton variant="rectangular" height={180} sx={{ borderRadius: "12px" }} />
          <Box sx={{ display: "flex", gap: 2 }}>
            <Skeleton variant="rectangular" width="33%" height={120} sx={{ borderRadius: "12px" }} />
            <Skeleton variant="rectangular" width="33%" height={120} sx={{ borderRadius: "12px" }} />
            <Skeleton variant="rectangular" width="33%" height={120} sx={{ borderRadius: "12px" }} />
          </Box>
        </Box>
      ) : (
        <Box sx={{ display: "flex", flexDirection: "column", gap: 2 }}>
          <Skeleton variant="text" width="60%" height={32} />
          <Skeleton variant="rectangular" height={120} sx={{ borderRadius: "12px" }} />
          <Box sx={{ display: "flex", gap: 2 }}>
             <Skeleton variant="circular" width={40} height={40} />
             <Box sx={{ flex: 1 }}>
                <Skeleton variant="text" width="90%" />
                <Skeleton variant="text" width="40%" />
             </Box>
          </Box>
        </Box>
      )}

      {/* Modern Shimmer effect overlay via CSS animation if needed, but MUI Skeleton already has it */}
    </Paper>
  )
}

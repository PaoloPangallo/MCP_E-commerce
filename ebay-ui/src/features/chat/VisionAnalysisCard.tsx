import React from "react"
import { Box, Typography, Chip, Paper, Fade } from "@mui/material"
import VisibilityIcon from "@mui/icons-material/Visibility"
import PsychologyIcon from "@mui/icons-material/Psychology"
import VerifiedUserIcon from "@mui/icons-material/VerifiedUser"
import InventoryIcon from "@mui/icons-material/Inventory"

interface VisionAnalysisCardProps {
  description: string
  tags: string[]
  brand?: string | null
  condition_clues?: string | null
  confidence?: number
}

const VisionAnalysisCard: React.FC<VisionAnalysisCardProps> = ({
  description,
  tags,
  brand,
  condition_clues,
  confidence
}) => {
  const confPercent = confidence !== undefined ? Math.round(confidence * 100) : 95

  return (
    <Fade in timeout={600}>
      <Paper
        elevation={0}
        sx={{
          p: 2.5,
          mb: 3,
          borderRadius: "16px",
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
          position: "relative",
          overflow: "hidden"
        }}
      >

        <Box sx={{ display: "flex", alignItems: "center", justifyContent: "space-between", mb: 2, position: "relative" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.5 }}>
            <Box
              sx={{
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                width: 36,
                height: 36,
                borderRadius: "10px",
                background: "var(--brand-primary)",
                color: "var(--bg-primary)"
              }}
            >
              <VisibilityIcon sx={{ fontSize: 20 }} />
            </Box>
            <Box>
              <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "var(--text-primary)", lineHeight: 1.2 }}>
                Vision Analysis
              </Typography>
              <Typography variant="caption" sx={{ color: "var(--text-secondary)" }}>
                Analisi visuale completata con Qwen 3.5 VL
              </Typography>
            </Box>
          </Box>
          <Box sx={{ display: "flex", flexDirection: "column", alignItems: "flex-end" }}>
            <Typography variant="caption" sx={{ color: "var(--text-secondary)", fontWeight: 600 }}>
              Confidence {confPercent}%
            </Typography>
            <Box sx={{ width: 60, height: 4, bgcolor: "var(--bg-primary)", borderRadius: 2, mt: 0.5, overflow: "hidden" }}>
              <Box sx={{ width: `${confPercent}%`, height: "100%", bgcolor: confPercent > 80 ? "var(--success)" : "var(--warning)" }} />
            </Box>
          </Box>
        </Box>

        <Typography
          variant="body1"
          sx={{
            color: "var(--text-primary)",
            lineHeight: 1.6,
            mb: 2,
            fontFamily: "Inter, sans-serif",
            fontSize: "0.95rem",
            position: "relative"
          }}
        >
          {description}
        </Typography>

        {(brand || condition_clues) && (
          <Box sx={{ display: "flex", gap: 1.5, mb: 2.5, flexWrap: "wrap" }}>
            {brand && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, color: "var(--text-primary)", bgcolor: "var(--bg-primary)", px: 1.5, py: 0.5, borderRadius: 2, border: "1px solid var(--border-color)" }}>
                <VerifiedUserIcon sx={{ fontSize: 16, color: "var(--brand-primary)" }} />
                <Typography variant="caption" sx={{ fontWeight: 600 }}>Brand: {brand}</Typography>
              </Box>
            )}
            {condition_clues && (
              <Box sx={{ display: "flex", alignItems: "center", gap: 0.5, color: "var(--text-primary)", bgcolor: "var(--bg-primary)", px: 1.5, py: 0.5, borderRadius: 2, border: "1px solid var(--border-color)" }}>
                <InventoryIcon sx={{ fontSize: 16, color: "var(--text-secondary)" }} />
                <Typography variant="caption" sx={{ fontWeight: 600 }}>Stato: {condition_clues}</Typography>
              </Box>
            )}
          </Box>
        )}

        {tags && tags.length > 0 && (
          <Box sx={{ display: "flex", flexWrap: "wrap", gap: 1, position: "relative" }}>
            {tags.map((tag, i) => (
              <Chip
                key={i}
                label={tag}
                size="small"
                sx={{
                  background: "var(--bg-primary)",
                  border: "1px solid var(--border-color)",
                  color: "var(--text-secondary)",
                  fontWeight: 500,
                  fontSize: "0.75rem",
                  "&:hover": {
                    background: "var(--bg-secondary)",
                    borderColor: "var(--text-secondary)"
                  }
                }}
              />
            ))}
          </Box>
        )}

        <Box 
          sx={{ 
            mt: 2, 
            pt: 2, 
            borderTop: "1px solid var(--border-color)",
            display: "flex",
            alignItems: "center",
            gap: 1
          }}
        >
          <PsychologyIcon sx={{ fontSize: 16, color: "var(--accent-primary)" }} />
          <Typography variant="caption" sx={{ color: "var(--text-secondary)", fontStyle: "italic" }}>
            Query arricchita e passata al Reasoner (120B)
          </Typography>
        </Box>
      </Paper>
    </Fade>
  )
}

export default VisionAnalysisCard

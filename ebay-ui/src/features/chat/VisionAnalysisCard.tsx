import React from "react"
import { Box, Typography, Chip, Paper, Fade } from "@mui/material"
import VisibilityIcon from "@mui/icons-material/Visibility"
import PsychologyIcon from "@mui/icons-material/Psychology"

interface VisionAnalysisCardProps {
  description: string
  tags: string[]
  confidence?: number
}

const VisionAnalysisCard: React.FC<VisionAnalysisCardProps> = ({
  description,
  tags
}) => {
  return (
    <Fade in timeout={600}>
      <Paper
        elevation={0}
        sx={{
          p: 2.5,
          mb: 3,
          borderRadius: "20px",
          background: "var(--bg-secondary)",
          border: "1px solid var(--border-color)",
          position: "relative",
          overflow: "hidden"
        }}
      >

        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2, position: "relative" }}>
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
              Analisi visuale completata con Qwen-VL
            </Typography>
          </Box>
        </Box>

        <Typography
          variant="body1"
          sx={{
            color: "var(--text-primary)",
            lineHeight: 1.6,
            mb: 2.5,
            fontFamily: "Inter, sans-serif",
            fontSize: "0.95rem",
            position: "relative"
          }}
        >
          {description}
        </Typography>

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

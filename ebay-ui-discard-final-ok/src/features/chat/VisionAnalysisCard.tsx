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
          background: "rgba(255, 255, 255, 0.03)",
          backdropFilter: "blur(10px)",
          border: "1px solid rgba(255, 255, 255, 0.08)",
          boxShadow: "0 8px 32px 0 rgba(0, 0, 0, 0.2)",
          position: "relative",
          overflow: "hidden"
        }}
      >
        {/* Glow effect */}
        <Box
          sx={{
            position: "absolute",
            top: -50,
            right: -50,
            width: 150,
            height: 150,
            background: "radial-gradient(circle, rgba(0, 102, 255, 0.1) 0%, transparent 70%)",
            zIndex: 0
          }}
        />

        <Box sx={{ display: "flex", alignItems: "center", gap: 1.5, mb: 2, position: "relative" }}>
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              width: 36,
              height: 36,
              borderRadius: "10px",
              background: "linear-gradient(135deg, #0066FF 0%, #0044BB 100%)",
              color: "white"
            }}
          >
            <VisibilityIcon sx={{ fontSize: 20 }} />
          </Box>
          <Box>
            <Typography variant="subtitle2" sx={{ fontWeight: 700, color: "grey.300", lineHeight: 1.2 }}>
              Vision Analysis
            </Typography>
            <Typography variant="caption" sx={{ color: "grey.500" }}>
              Analisi visuale completata con Qwen-VL
            </Typography>
          </Box>
        </Box>

        <Typography
          variant="body1"
          sx={{
            color: "grey.100",
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
                  background: "rgba(255, 255, 255, 0.05)",
                  border: "1px solid rgba(255, 255, 255, 0.1)",
                  color: "grey.400",
                  fontWeight: 500,
                  fontSize: "0.75rem",
                  "&:hover": {
                    background: "rgba(255, 255, 255, 0.1)",
                    borderColor: "#0066FF"
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
            borderTop: "1px solid rgba(255, 255, 255, 0.05)",
            display: "flex",
            alignItems: "center",
            gap: 1
          }}
        >
          <PsychologyIcon sx={{ fontSize: 16, color: "#0066FF" }} />
          <Typography variant="caption" sx={{ color: "grey.500", fontStyle: "italic" }}>
            Query arricchita e passata al Reasoner (120B)
          </Typography>
        </Box>
      </Paper>
    </Fade>
  )
}

export default VisionAnalysisCard

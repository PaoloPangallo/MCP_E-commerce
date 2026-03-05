import { useState } from "react"
import {
  Box,
  Typography,
  CircularProgress,
  Collapse,
  IconButton,
  Chip
} from "@mui/material"

import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import ExpandMoreIcon from "@mui/icons-material/ExpandMore"
import ExpandLessIcon from "@mui/icons-material/ExpandLess"


// --------------------------------------------------
// TYPES
// --------------------------------------------------

interface IRMetrics {
  "precision@5"?: number
  "precision@10"?: number
  "recall@10"?: number
  "ndcg@10"?: number
}

interface Props {
  text?: string
  loading?: boolean
  metrics?: IRMetrics
  rag_context?: string
}


// --------------------------------------------------
// COMPONENT
// --------------------------------------------------

export default function AIAnalysisCard({
  text,
  loading = false,
  metrics,
  rag_context
}: Props) {

  const [expanded, setExpanded] = useState(false)

  if (!text && !loading) return null


  return (

    <Box
      sx={{
        p: 2.5,
        mb: 3,
        borderRadius: "16px",
        bgcolor: "#f6f6f7",
        border: "1px solid #e5e5e5"
      }}
    >

      {/* HEADER */}

      <Box
        display="flex"
        alignItems="center"
        justifyContent="space-between"
        mb={1.5}
      >

        <Box display="flex" alignItems="center" gap={1}>

          <AutoAwesomeIcon
            sx={{
              fontSize: 18,
              color: "#10a37f"
            }}
          />

          <Typography
            fontWeight={600}
            fontSize={14}
          >
            AI Analysis
          </Typography>

          {metrics && (
            <Chip
              label="AI Ranked"
              size="small"
              sx={{
                fontSize: 11,
                bgcolor: "#e8f5f0",
                color: "#0a7a5a"
              }}
            />
          )}

        </Box>

        {(metrics || rag_context) && (
          <IconButton
            size="small"
            onClick={() => setExpanded(!expanded)}
          >
            {expanded ? <ExpandLessIcon /> : <ExpandMoreIcon />}
          </IconButton>
        )}

      </Box>


      {/* LOADING */}

      {loading && (

        <Box display="flex" alignItems="center" gap={1}>

          <CircularProgress size={16} />

          <Typography
            sx={{
              fontSize: 13,
              color: "#777"
            }}
          >
            Generazione analisi...
          </Typography>

        </Box>

      )}


      {/* MAIN ANALYSIS */}

      {text && (

        <Typography
          sx={{
            fontSize: 15,
            lineHeight: 1.65,
            whiteSpace: "pre-line"
          }}
        >
          {text}
        </Typography>

      )}


      {/* EXPANDABLE SECTION */}

      <Collapse in={expanded}>

        {/* METRICS */}

        {metrics && (

          <Box mt={2}>

            <Typography
              sx={{
                fontSize: 13,
                fontWeight: 600,
                mb: 1
              }}
            >
              Ranking Metrics
            </Typography>

            <Box
              sx={{
                display: "flex",
                gap: 1,
                flexWrap: "wrap"
              }}
            >

              {metrics["precision@5"] !== undefined && (
                <Chip
                  size="small"
                  label={`Precision@5: ${metrics["precision@5"].toFixed(2)}`}
                />
              )}

              {metrics["precision@10"] !== undefined && (
                <Chip
                  size="small"
                  label={`Precision@10: ${metrics["precision@10"].toFixed(2)}`}
                />
              )}

              {metrics["recall@10"] !== undefined && (
                <Chip
                  size="small"
                  label={`Recall@10: ${metrics["recall@10"].toFixed(2)}`}
                />
              )}

              {metrics["ndcg@10"] !== undefined && (
                <Chip
                  size="small"
                  label={`NDCG@10: ${metrics["ndcg@10"].toFixed(2)}`}
                />
              )}

            </Box>

          </Box>

        )}


        {/* RAG CONTEXT */}

        {rag_context && (

          <Box mt={2}>

            <Typography
              sx={{
                fontSize: 13,
                fontWeight: 600,
                mb: 1
              }}
            >
              Retrieved Evidence
            </Typography>

            <Typography
              sx={{
                fontSize: 13,
                color: "#666",
                lineHeight: 1.6,
                maxHeight: 120,
                overflow: "hidden",
                textOverflow: "ellipsis"
              }}
            >
              {rag_context
                ?.split(" - ")
                .slice(0,5)
                .map((e,i) => (
                  <Typography
                    key={i}
                    sx={{
                      fontSize: 13,
                      color: "#666",
                      lineHeight: 1.6
                    }}
                  >
                    • {e}
                  </Typography>
              ))}
            </Typography>

          </Box>

        )}

      </Collapse>

    </Box>

  )

}
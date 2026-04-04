import { Box, Typography, Chip } from "@mui/material"
import CheckCircleOutlineIcon from "@mui/icons-material/CheckCircleOutline"
import ErrorOutlineIcon from "@mui/icons-material/ErrorOutline"
import SendIcon from "@mui/icons-material/Send"
import type { ContactSellerResult } from "../agent/types"

interface Props {
  data: ContactSellerResult
}

export default function ContactSellerCard({ data }: Props) {
  const success = data?.success === true
  const status = data?.contact_status || "unknown"
  const message = data?.message_sent
  const detail = data?.detail || ""

  return (
    <Box
      sx={{
        border: `1.5px solid ${success ? "var(--brand-primary, #2563eb)" : "var(--danger, #ef4444)"}`,
        borderRadius: "16px",
        overflow: "hidden",
        bgcolor: success
          ? "rgba(37, 99, 235, 0.05)"
          : "rgba(239, 68, 68, 0.05)",
        transition: "all 0.2s ease",
      }}
    >
      {/* Header */}
      <Box
        sx={{
          display: "flex",
          alignItems: "center",
          gap: 1.5,
          px: 2.5,
          py: 1.5,
          borderBottom: "1px solid var(--border-color)",
          bgcolor: success
            ? "rgba(37, 99, 235, 0.08)"
            : "rgba(239, 68, 68, 0.08)",
        }}
      >
        {success ? (
          <CheckCircleOutlineIcon
            sx={{ fontSize: 20, color: "var(--brand-primary, #2563eb)" }}
          />
        ) : (
          <ErrorOutlineIcon
            sx={{ fontSize: 20, color: "var(--danger, #ef4444)" }}
          />
        )}
        <Typography
          sx={{
            fontSize: 13,
            fontWeight: 700,
            color: success
              ? "var(--brand-primary, #2563eb)"
              : "var(--danger, #ef4444)",
            letterSpacing: "-0.01em",
          }}
        >
          {success ? "Messaggio inviato al venditore" : "Invio messaggio fallito"}
        </Typography>
        <Chip
          label={status}
          size="small"
          sx={{
            ml: "auto",
            fontSize: 10,
            height: 20,
            fontWeight: 700,
            bgcolor: success
              ? "rgba(37, 99, 235, 0.12)"
              : "rgba(239, 68, 68, 0.12)",
            color: success
              ? "var(--brand-primary, #2563eb)"
              : "var(--danger, #ef4444)",
            borderRadius: "6px",
          }}
        />
      </Box>

      {/* Body */}
      <Box sx={{ px: 2.5, py: 2, display: "flex", flexDirection: "column", gap: 1.5 }}>
        {/* Detail / error message */}
        {detail && (
          <Typography
            sx={{
              fontSize: 13,
              color: "var(--text-primary)",
              lineHeight: 1.6,
            }}
          >
            {detail}
          </Typography>
        )}

        {/* The actual message sent */}
        {message && (
          <Box
            sx={{
              display: "flex",
              gap: 1,
              alignItems: "flex-start",
              bgcolor: "var(--bg-secondary)",
              borderRadius: "10px",
              px: 2,
              py: 1.5,
            }}
          >
            <SendIcon sx={{ fontSize: 15, color: "var(--text-secondary)", mt: 0.2, flexShrink: 0 }} />
            <Box>
              <Typography
                sx={{ fontSize: 11, fontWeight: 700, color: "var(--text-secondary)", mb: 0.5, textTransform: "uppercase", letterSpacing: "0.06em" }}
              >
                Testo inviato
              </Typography>
              <Typography
                sx={{
                  fontSize: 13,
                  color: "var(--text-primary)",
                  lineHeight: 1.6,
                  fontStyle: "italic",
                }}
              >
                "{message}"
              </Typography>
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  )
}

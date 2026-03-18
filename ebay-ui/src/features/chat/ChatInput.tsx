import { useState } from "react"
import { Box, IconButton, InputBase, Typography } from "@mui/material"
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward"

interface Props {
  onSend: (value: string) => void
  disabled?: boolean
  placeholder?: string
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Chiedi qualcosa…"
}: Props) {
  const [value, setValue] = useState("")

  const handleSend = () => {
    const trimmed = value.trim()
    if (!trimmed || disabled) return
    onSend(trimmed)
    setValue("")
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (disabled) return
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const canSend = !!value.trim() && !disabled

  return (
    <Box>
      <Box
        sx={{
          display: "flex",
          alignItems: "flex-end",
          gap: 0.75,
          px: 1.5,
          py: 1,
          borderRadius: "24px",
          bgcolor: "#ffffff",
          border: "1px solid",
          borderColor: "transparent",
          outline: "1px solid #e5e7eb",
          boxShadow: "0 2px 8px rgba(15, 23, 42, 0.06)",
          transition: "outline-color 0.15s ease, box-shadow 0.15s ease",
          "&:focus-within": {
            outlineColor: "#d1d5db",
            boxShadow: "0 4px 14px rgba(15, 23, 42, 0.08)"
          }
        }}
      >
        <InputBase
          fullWidth
          multiline
          maxRows={8}
          value={value}
          onChange={(e) => setValue(e.target.value)}
          onKeyDown={handleKeyDown}
          placeholder={placeholder}
          disabled={disabled}
          sx={{
            px: 1,
            py: 0.75,
            alignItems: "flex-start",
            fontSize: 14,
            lineHeight: 1.65,
            color: "#111827",
            "& textarea": { resize: "none" },
            "& textarea::placeholder": { color: "#9ca3af" }
          }}
        />

        <IconButton
          aria-label="Invia messaggio"
          onClick={handleSend}
          disabled={!canSend}
          sx={{
            width: 34,
            height: 34,
            mb: 0.25,
            flexShrink: 0,
            bgcolor: canSend ? "#111827" : "#f3f4f6",
            color: canSend ? "#ffffff" : "#d1d5db",
            transition: "all 0.15s ease",
            "&:hover": {
              bgcolor: canSend ? "#1f2937" : "#f3f4f6"
            },
            "&.Mui-disabled": {
              bgcolor: "#f3f4f6",
              color: "#d1d5db"
            }
          }}
        >
          <ArrowUpwardIcon sx={{ fontSize: 17 }} />
        </IconButton>
      </Box>

      <Typography
        sx={{
          mt: 0.75,
          px: 1.75,
          fontSize: 11,
          color: "#c0c4cc",
          userSelect: "none"
        }}
      >
        Enter per inviare · Shift + Enter per andare a capo
      </Typography>
    </Box>
  )
}
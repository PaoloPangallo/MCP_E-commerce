import { useState } from "react"
import {
  Box,
  IconButton,
  InputBase,
  Paper,
  Typography
} from "@mui/material"
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward"

interface Props {
  onSend: (value: string) => void
  disabled?: boolean
  placeholder?: string
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = 'Scrivi una richiesta, ad esempio: "iPhone 13 massimo 700 euro e controlla il venditore"'
}: Props) {
  const [value, setValue] = useState("")

  const handleSend = () => {
    const nextValue = value.trim()

    if (!nextValue || disabled) return

    onSend(nextValue)
    setValue("")
  }

  const handleKeyDown = (event: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (disabled) return

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  return (
    <Box
      sx={{
        px: { xs: 1.5, md: 2.5 },
        pb: { xs: 1.5, md: 2.5 },
        pt: 1.5
      }}
    >
      <Box sx={{ maxWidth: 880, mx: "auto" }}>
        <Paper
          elevation={0}
          sx={{
            display: "flex",
            alignItems: "flex-end",
            gap: 1,
            px: 1.25,
            py: 1,
            borderRadius: "28px",
            bgcolor: "#ffffff",
            border: "1px solid #e5e7eb",
            boxShadow: "0 10px 30px rgba(15, 23, 42, 0.06)",
            transition: "border-color 0.15s ease, box-shadow 0.15s ease",
            "&:focus-within": {
              borderColor: "#cbd5e1",
              boxShadow: "0 12px 34px rgba(15, 23, 42, 0.08)"
            }
          }}
        >
          <InputBase
            fullWidth
            multiline
            maxRows={8}
            value={value}
            onChange={(event) => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder={placeholder}
            disabled={disabled}
            sx={{
              px: 1.25,
              py: 0.8,
              alignItems: "flex-start",
              fontSize: 15,
              lineHeight: 1.7,
              color: "#111827",
              "& textarea": {
                resize: "none"
              }
            }}
          />

          <IconButton
            aria-label="Invia messaggio"
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            sx={{
              width: 40,
              height: 40,
              mb: 0.2,
              bgcolor:
                value.trim() && !disabled ? "#111827" : "transparent",
              color:
                value.trim() && !disabled ? "#ffffff" : "#9ca3af",
              transition: "all 0.15s ease",
              "&:hover": {
                bgcolor:
                  value.trim() && !disabled ? "#0b1220" : "transparent"
              }
            }}
          >
            <ArrowUpwardIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Paper>

        <Typography
          sx={{
            mt: 1,
            px: 1.5,
            fontSize: 12,
            color: "#6b7280"
          }}
        >
          Enter per inviare · Shift + Enter per andare a capo
        </Typography>
      </Box>
    </Box>
  )
}
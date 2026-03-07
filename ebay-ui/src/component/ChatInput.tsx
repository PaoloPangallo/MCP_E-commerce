import { useState } from "react"
import { Box, TextField, IconButton, Paper, Typography } from "@mui/material"
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward"

interface Props {
  onSend: (value: string) => void
  disabled?: boolean
}

export default function ChatInput({ onSend, disabled = false }: Props) {
  const [value, setValue] = useState("")

  const handleSend = () => {
    const nextValue = value.trim()

    if (!nextValue || disabled) {
      return
    }

    onSend(nextValue)
    setValue("")
  }

  const handleKeyDown = (event: React.KeyboardEvent) => {
    if (disabled) {
      return
    }

    if (event.key === "Enter" && !event.shiftKey) {
      event.preventDefault()
      handleSend()
    }
  }

  return (
    <Box sx={{ p: 3, pb: 4, borderTop: "1px solid #ececf1", bgcolor: "#fff" }}>
      <Box sx={{ maxWidth: 768, mx: "auto" }}>
        <Paper
          elevation={0}
          sx={{
            display: "flex",
            alignItems: "flex-end",
            borderRadius: "24px",
            bgcolor: "#f4f4f4",
            border: "1px solid transparent",
            transition: "all 0.15s ease",
            "&:hover": { bgcolor: "#efefef" },
            "&:focus-within": {
              bgcolor: "#fff",
              border: "1px solid #d1d1d6"
            }
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={8}
            value={value}
            onChange={event => setValue(event.target.value)}
            onKeyDown={handleKeyDown}
            placeholder='Scrivi una richiesta, ad esempio: "iPhone 13 massimo 700 euro e controlla il venditore"'
            disabled={disabled}
            variant="standard"
            InputProps={{
              disableUnderline: true,
              sx: {
                px: 3,
                py: 2.25,
                fontSize: 15
              }
            }}
          />

          <IconButton
            aria-label="Invia messaggio"
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            sx={{
              mb: 1.25,
              mr: 1.25,
              bgcolor: value.trim() && !disabled ? "#0d0d0d" : "transparent",
              color: value.trim() && !disabled ? "#fff" : "#b4b4b4",
              width: 36,
              height: 36,
              "&:hover": {
                bgcolor: value.trim() && !disabled ? "#202123" : "transparent"
              }
            }}
          >
            <ArrowUpwardIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Paper>

        <Typography sx={{ mt: 1, ml: 1, fontSize: 12, color: "#8e8ea0" }}>
          Invio con Enter · nuova riga con Shift + Enter
        </Typography>
      </Box>
    </Box>
  )
}
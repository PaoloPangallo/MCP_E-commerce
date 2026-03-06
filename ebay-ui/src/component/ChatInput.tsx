import { Box, TextField, IconButton, Paper } from "@mui/material"
import { useState } from "react"
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward"

export default function ChatInput({
  onSend,
  disabled = false
}: {
  onSend: (value: string) => void
  disabled?: boolean
}) {

  const [value, setValue] = useState("")

  const handleSend = () => {

    if (!value.trim() || disabled) return

    onSend(value.trim())

    setValue("")

  }

  const handleKeyDown = (e: React.KeyboardEvent) => {

    if (disabled) return

    if (e.key === "Enter" && !e.shiftKey) {

      e.preventDefault()

      handleSend()

    }

  }

  return (

    <Box
      sx={{
        p: 3,
        pb: 4
      }}
    >

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

            "&:hover": {
              bgcolor: "#ececec"
            },

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
            onChange={(e) => setValue(e.target.value)}
            onKeyDown={handleKeyDown}
            placeholder="Invia un messaggio..."
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
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            sx={{
              mb: 1.25,
              mr: 1.25,
              bgcolor: value.trim() && !disabled ? "#0d0d0d" : "transparent",
              color: value.trim() && !disabled ? "#fff" : "#b4b4b4",
              width: 32,
              height: 32
            }}
          >

            <ArrowUpwardIcon sx={{ fontSize: 18 }} />

          </IconButton>

        </Paper>

      </Box>

    </Box>

  )

}
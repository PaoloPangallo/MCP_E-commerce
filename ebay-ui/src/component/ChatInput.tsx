import { Box, TextField, IconButton, Paper } from "@mui/material";
import { useState } from "react";
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward";

export default function ChatInput({
  onSend,
  disabled = false,
}: {
  onSend: (value: string) => void;
  disabled?: boolean;
}) {
  const [value, setValue] = useState("");

  const handleSend = () => {
    if (!value.trim() || disabled) return;
    onSend(value.trim());
    setValue("");
  };

  const handleKeyDown = (e: React.KeyboardEvent) => {
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault();
      handleSend();
    }
  };

  return (
    <Box
      sx={{
        p: 2,
        bgcolor: "transparent",
      }}
    >
      <Box sx={{ maxWidth: 800, mx: "auto" }}>
        <Paper
          elevation={0}
          sx={{
            display: "flex",
            alignItems: "flex-end",
            borderRadius: "26px",
            bgcolor: "#f4f4f4",
            transition: "all 0.2s",
            "&:focus-within": {
              bgcolor: "#f4f4f4",
            },
          }}
        >
          <TextField
            fullWidth
            multiline
            maxRows={6}
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
                py: 2,
                fontSize: 16,
                lineHeight: 1.5,
                color: "#0d0d0d",
                "& textarea::placeholder": {
                  color: "#8e8ea0",
                  opacity: 1,
                },
              },
            }}
          />

          <IconButton
            onClick={handleSend}
            disabled={disabled || !value.trim()}
            sx={{
              mb: 1,
              mr: 1,
              bgcolor: value.trim() && !disabled ? "#000" : "#e5e5e5",
              color: value.trim() && !disabled ? "#fff" : "#a3a3a3",
              width: 32,
              height: 32,
              "&:hover": {
                bgcolor: value.trim() && !disabled ? "#333" : "#e5e5e5",
              },
              "&:disabled": {
                bgcolor: "#e5e5e5",
                color: "#f4f4f4", // to match transparent-looking icon
              },
              transition: "all 0.2s",
            }}
          >
            <ArrowUpwardIcon sx={{ fontSize: 20 }} />
          </IconButton>
        </Paper>

        <Box
          sx={{
            mt: 1.5,
            textAlign: "center",
            fontSize: 12,
            color: "#666",
          }}
        >
          L'intelligenza artificiale può commettere errori. Verifica le informazioni importanti.
        </Box>
      </Box>
    </Box>
  );
}

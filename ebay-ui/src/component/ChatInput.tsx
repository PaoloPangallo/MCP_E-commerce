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
        p: 3,
        pb: 4,
        bgcolor: "transparent",
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
              bgcolor: "#ececec",
            },
            "&:focus-within": {
              bgcolor: "#fff",
              border: "1px solid #d1d1d6",
              boxShadow: "0 0 0 2px rgba(0,0,0,0.05)",
            },
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
                fontSize: 15,
                lineHeight: 1.6,
                color: "#0d0d0d",
                fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
                "& textarea": {
                  "&::placeholder": {
                    color: "#8e8ea0",
                    opacity: 1,
                  },
                },
              },
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
              height: 32,
              minWidth: 32,
              minHeight: 32,
              borderRadius: "8px",
              "&:hover": {
                bgcolor: value.trim() && !disabled ? "#2d2d2d" : "#e5e5e5",
              },
              "&:disabled": {
                bgcolor: "transparent",
                color: "#d1d1d6",
              },
              transition: "all 0.2s ease",
            }}
          >
            <ArrowUpwardIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Paper>

        <Box
          sx={{
            mt: 2,
            px: 1,
            textAlign: "center",
            fontSize: 12,
            lineHeight: 1.4,
            color: "#676767",
            fontFamily: "-apple-system, BlinkMacSystemFont, 'Segoe UI', sans-serif",
          }}
        >
          L'intelligenza artificiale può commettere errori. Verifica le informazioni importanti.
        </Box>
      </Box>
    </Box>
  );
}
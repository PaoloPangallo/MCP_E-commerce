import { Box, TextField, IconButton, Paper } from "@mui/material";
import { useState } from "react";
import SendIcon from "@mui/icons-material/Send";

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
        p: 4,
        bgcolor: "#fff",
        borderTop: "1px solid #ececf1",
      }}
    >
      <Box sx={{ maxWidth: 768, mx: "auto" }}>
        <Paper
          elevation={0}
          sx={{
            display: "flex",
            alignItems: "flex-end",
            border: "1px solid #d1d1d1",
            borderRadius: 3,
            bgcolor: "#fff",
            transition: "all 0.2s",
            "&:focus-within": {
              borderColor: "#10a37f",
              boxShadow: "0 0 0 2px rgba(16, 163, 127, 0.1)",
            },
            "&:hover": {
              borderColor: disabled ? "#d1d1d1" : "#b4b4b4",
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
              mb: 1.5,
              mr: 1.5,
              bgcolor: value.trim() && !disabled ? "#10a37f" : "#ececf1",
              color: value.trim() && !disabled ? "#fff" : "#8e8ea0",
              width: 32,
              height: 32,
              "&:hover": {
                bgcolor: value.trim() && !disabled ? "#0d8c6b" : "#ececf1",
              },
              "&:disabled": {
                bgcolor: "#ececf1",
                color: "#8e8ea0",
              },
              transition: "all 0.2s",
            }}
          >
            <SendIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Paper>

        <Box
          sx={{
            mt: 2,
            textAlign: "center",
            fontSize: 12,
            color: "#8e8ea0",
          }}
        >
          AI può commettere errori. Verifica le informazioni importanti.
        </Box>
      </Box>
    </Box>
  );
}

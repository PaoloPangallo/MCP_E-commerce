import { useState, useRef } from "react"
import { Box, IconButton, InputBase, Typography, CircularProgress } from "@mui/material"
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward"
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate"
import CloseIcon from "@mui/icons-material/Close"

interface Props {
  onSend: (value: string, image?: string) => void
  disabled?: boolean
  placeholder?: string
}

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Chiedi qualcosa…"
}: Props) {
  const [value, setValue] = useState("")
  const [image, setImage] = useState<string | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)

  const handleSend = () => {
    const trimmed = value.trim()
    if ((!trimmed && !image) || disabled || isProcessing) return
    onSend(trimmed, image || undefined)
    setValue("")
    setImage(null)
  }

  const handleKeyDown = (e: React.KeyboardEvent<HTMLTextAreaElement>) => {
    if (disabled) return
    if (e.key === "Enter" && !e.shiftKey) {
      e.preventDefault()
      handleSend()
    }
  }

  const handleFileChange = (e: React.ChangeEvent<HTMLInputElement>) => {
    const file = e.target.files?.[0]
    if (!file) return

    setIsProcessing(true)
    const reader = new FileReader()
    reader.onload = (event) => {
      setImage(event.target?.result as string)
      setIsProcessing(false)
    }
    reader.onerror = () => {
      setIsProcessing(false)
    }
    reader.readAsDataURL(file)
    // Reset input so the same file can be selected again
    e.target.value = ""
  }

  const canSend = (!!value.trim() || !!image) && !disabled && !isProcessing

  return (
    <Box>
      {/* Hidden File Input */}
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      {/* Image Preview */}
      {image && (
        <Box sx={{ px: 2, mb: 1, display: "flex" }}>
          <Box
            sx={{
              position: "relative",
              width: 56,
              height: 56,
              borderRadius: "12px",
              overflow: "hidden",
              border: "1px solid #e5e7eb",
              boxShadow: "0 2px 4px rgba(0,0,0,0.1)"
            }}
          >
            <img
              src={image}
              alt="Preview"
              style={{ width: "100%", height: "100%", objectFit: "cover" }}
            />
            <IconButton
              size="small"
              onClick={() => setImage(null)}
              sx={{
                position: "absolute",
                top: 2,
                right: 2,
                p: 0.25,
                bgcolor: "rgba(0,0,0,0.5)",
                color: "#fff",
                "&:hover": { bgcolor: "rgba(0,0,0,0.7)" }
              }}
            >
              <CloseIcon sx={{ fontSize: 12 }} />
            </IconButton>
          </Box>
        </Box>
      )}

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
        <IconButton
          onClick={() => fileInputRef.current?.click()}
          disabled={disabled || isProcessing}
          sx={{
            mb: 0.25,
            color: "#9ca3af",
            "&:hover": { color: "#111827" }
          }}
        >
          {isProcessing ? <CircularProgress size={16} color="inherit" /> : <AddPhotoAlternateIcon sx={{ fontSize: 20 }} />}
        </IconButton>

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
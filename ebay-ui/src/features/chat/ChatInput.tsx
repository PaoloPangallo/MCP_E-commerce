import { useState, useRef } from "react"
import { 
  Box, 
  IconButton, 
  InputBase, 
  Typography, 
  CircularProgress, 
  Menu, 
  MenuItem, 
  ListItemIcon, 
  ListItemText 
} from "@mui/material"
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward"
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate"
import AddIcon from "@mui/icons-material/Add"
import SellIcon from "@mui/icons-material/Sell"
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
  
  // Menu State
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const openMenu = Boolean(anchorEl)

  const handleMenuClick = (event: React.MouseEvent<HTMLElement>) => {
    setAnchorEl(event.currentTarget)
  }

  const handleMenuClose = () => {
    setAnchorEl(null)
  }

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
    handleMenuClose()
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
    e.target.value = ""
  }

  const handleDealsClick = () => {
    handleMenuClose()
    onSend("Mostrami le migliori offerte del giorno su eBay 🏷️", undefined)
  }

  const canSend = (!!value.trim() || !!image) && !disabled && !isProcessing

  return (
    <Box>
      <input
        type="file"
        accept="image/*"
        ref={fileInputRef}
        onChange={handleFileChange}
        style={{ display: "none" }}
      />

      {/* Menu Contextual */}
      <Menu
        anchorEl={anchorEl}
        open={openMenu}
        onClose={handleMenuClose}
        anchorOrigin={{ vertical: 'top', horizontal: 'left' }}
        transformOrigin={{ vertical: 'bottom', horizontal: 'left' }}
        PaperProps={{
          sx: {
mt: -1,
            boxShadow: '0 8px 30px rgba(0,0,0,0.12)',
            border: '1px solid var(--border-color)',
            bgcolor: 'var(--bg-primary)',
            color: 'var(--text-primary)',
            minWidth: 180,
            overflow: 'hidden'
          }
        }}
      >
        <MenuItem onClick={() => fileInputRef.current?.click()} sx={{ py: 1.5, px: 2 }}>
          <ListItemIcon sx={{ color: '#64748b' }}>
            <AddPhotoAlternateIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText 
            primary="Aggiungi foto e file" 
            primaryTypographyProps={{ fontSize: 13, fontWeight: 600, color: 'inherit' }} 
          />
        </MenuItem>
        
        <MenuItem onClick={handleDealsClick} sx={{ py: 1.5, px: 2 }}>
          <ListItemIcon sx={{ color: '#ef4444' }}>
            <SellIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText 
            primary="eBay Deals" 
            primaryTypographyProps={{ fontSize: 13, fontWeight: 600, color: '#ef4444' }} 
          />
        </MenuItem>
      </Menu>

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
          bgcolor: "var(--bg-primary)",
          border: "1px solid var(--border-color)",
          boxShadow: "0 4px 20px rgba(0, 0, 0, 0.05)",
          transition: "all 0.2s ease",
          "&:focus-within": {
            borderColor: "var(--text-primary)",
            boxShadow: "0 8px 30px rgba(0, 0, 0, 0.08)"
          }
        }}
      >
        <IconButton
          onClick={handleMenuClick}
          disabled={disabled || isProcessing}
          sx={{
            mb: 0.25,
            color: "var(--text-secondary)",
            bgcolor: openMenu ? "var(--bg-secondary)" : "transparent",
            "&:hover": { color: "var(--text-primary)", bgcolor: "var(--bg-secondary)" }
          }}
        >
          {isProcessing ? <CircularProgress size={16} color="inherit" /> : <AddIcon sx={{ fontSize: 24 }} />}
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
            color: "var(--text-primary)",
            "& textarea": { resize: "none" },
            "& textarea::placeholder": { color: "var(--text-secondary)", opacity: 0.5 }
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
            bgcolor: canSend ? "var(--text-primary)" : "var(--bg-secondary)",
            color: canSend ? "var(--bg-primary)" : "var(--text-secondary)",
            transition: "all 0.15s ease",
            "&:hover": {
              bgcolor: canSend ? "var(--text-primary)" : "var(--bg-secondary)",
              opacity: canSend ? 0.9 : 1
            },
            "&.Mui-disabled": {
              bgcolor: "var(--bg-secondary)",
              color: "var(--text-secondary)",
              opacity: 0.5
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
          color: "var(--text-secondary)",
          opacity: 0.5,
          userSelect: "none"
        }}
      >
        Enter per inviare · Shift + Enter per andare a capo
      </Typography>
    </Box>
  )
}
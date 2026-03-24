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
  ListItemText,
  Dialog,
  DialogTitle,
  DialogContent,
  Paper
} from "@mui/material"
import { EBAY_CATEGORIES } from "../search/constants"
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward"
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate"
import AddIcon from "@mui/icons-material/Add"
import SellIcon from "@mui/icons-material/Sell"
import CloseIcon from "@mui/icons-material/Close"
import StarsIcon from "@mui/icons-material/Stars"


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

  // Deals Modal State
  const [isCategoryModalOpen, setIsCategoryModalOpen] = useState(false)

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
    setIsCategoryModalOpen(true)
  }

  const handleCategorySelect = (category: typeof EBAY_CATEGORIES[0]) => {
    setIsCategoryModalOpen(false)
    setTimeout(() => {
      onSend(`Cerca le migliori offerte del giorno per la categoria ${category.name} (ID: ${category.id}) 🏷️`, undefined)
    }, 100);
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
        
        <MenuItem 
          onClick={handleDealsClick} 
          sx={{ py: 1.5, px: 2 }}
        >
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

      {/* Deals Category Modal */}
      <Dialog 
        open={isCategoryModalOpen} 
        onClose={() => setIsCategoryModalOpen(false)}
        maxWidth="sm"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: "20px",
            bgcolor: "var(--bg-primary)",
            backgroundImage: "none",
            boxShadow: "0 24px 48px -12px rgba(0,0,0,0.15)",
            border: "1px solid var(--border-color)",
          }
        }}
      >
        <DialogTitle sx={{ 
          pt: 4, 
          pb: 1.5, 
          px: 4, 
          color: "var(--text-primary)",
          display: "flex",
          flexDirection: "column",
          gap: 1
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <StarsIcon sx={{ color: '#ef4444', fontSize: 28 }} />
            <Typography sx={{ fontWeight: 800, fontSize: 24, letterSpacing: '-0.02em' }}>Scegli la categoria</Typography>
          </Box>
          <Typography variant="body2" sx={{ color: "var(--text-secondary)", fontWeight: 500, opacity: 0.8 }}>
            L'agente cercherà le migliori offerte eBay per la categoria selezionata.
          </Typography>
        </DialogTitle>

        <DialogContent sx={{ px: 4, pb: 4, pt: 1 }}>
          <Box sx={{ 
            display: "grid", 
            gridTemplateColumns: { xs: "1fr", sm: "1fr 1fr" }, 
            gap: 2, 
            mt: 2 
          }}>
            {EBAY_CATEGORIES.map((cat) => (
              <Paper
                key={cat.id}
                component="button"
                onClick={() => handleCategorySelect(cat)}
                elevation={0}
                sx={{
                  display: "flex",
                  alignItems: "center",
                  gap: 2,
                  p: 2,
                  textAlign: "left",
                  bgcolor: "var(--bg-secondary)",
                  border: "1px solid var(--border-color)",
                  borderRadius: "16px",
                  cursor: "pointer",
                  transition: "all 0.2s cubic-bezier(0.4, 0, 0.2, 1)",
                  fontFamily: "inherit",
                  "&:hover": {
                    bgcolor: "var(--bg-primary)",
                    borderColor: "#ef4444",
                    transform: "translateY(-2px)",
                    boxShadow: "0 8px 20px -8px rgba(239, 68, 68, 0.3)"
                  }
                }}
              >
                <Box sx={{ fontSize: 24, display: "flex", alignItems: "center", justifyContent: "center" }}>
                  {cat.icon}
                </Box>
                <Box>
                  <Typography sx={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", mb: 0.25 }}>
                    {cat.name}
                  </Typography>
                  <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", lineHeight: 1.2 }}>
                    {cat.desc}
                  </Typography>
                </Box>
              </Paper>
            ))}
          </Box>
        </DialogContent>
      </Dialog>
    </Box>
  )
}
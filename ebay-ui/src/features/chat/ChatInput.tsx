import { useState, useRef, useCallback, useEffect } from "react"
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
  DialogActions,
  Paper,
  TextField,
  Button
} from "@mui/material"
import { EBAY_CATEGORIES } from "../search/constants"
import { useSettingsStore } from "./store/settingsStore"
import ArrowUpwardIcon from "@mui/icons-material/ArrowUpward"
import AddPhotoAlternateIcon from "@mui/icons-material/AddPhotoAlternate"
import AddIcon from "@mui/icons-material/Add"
import SellIcon from "@mui/icons-material/Sell"
import CloseIcon from "@mui/icons-material/Close"
import StarsIcon from "@mui/icons-material/Stars"
import TravelExploreIcon from "@mui/icons-material/TravelExplore"
import MailOutlineIcon from "@mui/icons-material/MailOutline"

interface Props {
  onSend: (value: string, image?: string) => void
  disabled?: boolean
  placeholder?: string
}

// ─── Token helpers ────────────────────────────────────────────────────────────

/** OpenAI vision tile-based token cost for an image of given dimensions. */
function calcImageTokens(width: number, height: number): number {
  const tiles = Math.ceil(width / 512) * Math.ceil(height / 512)
  return Math.min(tiles * 765, 2000)
}

/** Estimate text tokens using the standard ~3.5 chars/token heuristic. */
function calcTextTokens(text: string): number {
  return Math.round(text.length / 3.5)
}

// ─── Component ────────────────────────────────────────────────────────────────

export default function ChatInput({
  onSend,
  disabled = false,
  placeholder = "Chiedi qualcosa…"
}: Props) {
  const [value, setValue] = useState("")
  const [image, setImage] = useState<string | null>(null)
  const [imageTokens, setImageTokens] = useState<number | null>(null)
  const [isProcessing, setIsProcessing] = useState(false)
  const [isDragging, setIsDragging] = useState(false)
  const fileInputRef = useRef<HTMLInputElement>(null)
  const updateLocalSettings = useSettingsStore((s) => s.updateLocalSettings)

  // Menu State
  const [anchorEl, setAnchorEl] = useState<null | HTMLElement>(null)
  const openMenu = Boolean(anchorEl)

  // Deals Modal State
  const [isCategoryModalOpen, setIsCategoryModalOpen] = useState(false)

  // Playwright Modal State
  const [isPlaywrightModalOpen, setIsPlaywrightModalOpen] = useState(false)
  const [playwrightQuery, setPlaywrightQuery] = useState("")

  // Contact Seller Modal State
  const [isContactSellerModalOpen, setIsContactSellerModalOpen] = useState(false)
  const [contactSellerName, setContactSellerName] = useState("")

  // ─── Image processing ──────────────────────────────────────────────────────

  const processImageFile = useCallback((file: File) => {
    if (!file.type.startsWith("image/")) return
    setIsProcessing(true)
    const reader = new FileReader()
    reader.onload = (event) => {
      const dataUrl = event.target?.result as string
      setImage(dataUrl)
      // Measure real pixel dimensions for accurate token estimate
      const img = new Image()
      img.onload = () => {
        setImageTokens(calcImageTokens(img.naturalWidth, img.naturalHeight))
        setIsProcessing(false)
      }
      img.onerror = () => setIsProcessing(false)
      img.src = dataUrl
    }
    reader.onerror = () => setIsProcessing(false)
    reader.readAsDataURL(file)
  }, [])

  // ─── Paste (Ctrl+V) ────────────────────────────────────────────────────────

  useEffect(() => {
    const handlePaste = (e: ClipboardEvent) => {
      const items = e.clipboardData?.items
      if (!items) return
      for (const item of Array.from(items)) {
        if (item.type.startsWith("image/")) {
          const file = item.getAsFile()
          if (file) {
            e.preventDefault()
            processImageFile(file)
            return
          }
        }
      }
    }
    document.addEventListener("paste", handlePaste)
    return () => document.removeEventListener("paste", handlePaste)
  }, [processImageFile])

  // ─── Handlers ──────────────────────────────────────────────────────────────

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
    setImageTokens(null)
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
    processImageFile(file)
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

  const handlePlaywrightClick = () => {
    handleMenuClose()
    setPlaywrightQuery(value.trim())
    setIsPlaywrightModalOpen(true)
  }

  const handlePlaywrightConfirm = () => {
    const query = playwrightQuery.trim()
    if (!query) return

    setIsPlaywrightModalOpen(false)
    onSend(`Cerca su eBay con Playwright (MODALITÀ VISIBILE): ${query} 🌐`, undefined)
    setPlaywrightQuery("")
    if (value.trim() === query) setValue("")
  }

  const handleContactSellerClick = () => {
    handleMenuClose()
    setContactSellerName(value.trim())
    setIsContactSellerModalOpen(true)
  }

  const handleContactSellerConfirm = () => {
    const seller = contactSellerName.trim()
    if (!seller) return
    setIsContactSellerModalOpen(false)
    // Switch to playwright mode so the agent uses contact_seller_playwright.
    // useAgentStream reads mcpMode via getState() at call time, so this update
    // is picked up synchronously before the stream request is sent.
    updateLocalSettings({ mcpMode: 'playwright_browser' })
    onSend(`Contattami il seller ${seller} e chiedi informazioni`, undefined)
    setContactSellerName("")
    if (value.trim() === seller) setValue("")
  }

  // ─── Drag & drop ───────────────────────────────────────────────────────────

  const handleDragOver = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(true)
  }

  const handleDragLeave = (e: React.DragEvent) => {
    // Only clear if leaving the outer Box entirely (not a child)
    if (!e.currentTarget.contains(e.relatedTarget as Node)) {
      setIsDragging(false)
    }
  }

  const handleDrop = (e: React.DragEvent) => {
    e.preventDefault()
    setIsDragging(false)
    const file = e.dataTransfer.files[0]
    if (file) processImageFile(file)
  }

  // ─── Token display ─────────────────────────────────────────────────────────

  const textTokens = calcTextTokens(value)
  const totalTokens = textTokens + (imageTokens ?? 0)
  const showTokens = !!value.trim() || !!image

  const canSend = (!!value.trim() || !!image) && !disabled && !isProcessing

  return (
    <Box
      onDragOver={handleDragOver}
      onDragLeave={handleDragLeave}
      onDrop={handleDrop}
    >
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

        <MenuItem
          onClick={handlePlaywrightClick}
          sx={{ py: 1.5, px: 2 }}
        >
          <ListItemIcon sx={{ color: '#3b82f6' }}>
            <TravelExploreIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="Cerca con Playwright"
            secondary="Mostra il browser (Debug)"
            primaryTypographyProps={{ fontSize: 13, fontWeight: 600, color: '#3b82f6' }}
            secondaryTypographyProps={{ fontSize: 10, color: 'var(--text-secondary)' }}
          />
        </MenuItem>

        <MenuItem
          onClick={handleContactSellerClick}
          sx={{ py: 1.5, px: 2 }}
        >
          <ListItemIcon sx={{ color: '#10b981' }}>
            <MailOutlineIcon fontSize="small" />
          </ListItemIcon>
          <ListItemText
            primary="Contatta venditore"
            secondary="Apre Chrome e invia il messaggio"
            primaryTypographyProps={{ fontSize: 13, fontWeight: 600, color: '#10b981' }}
            secondaryTypographyProps={{ fontSize: 10, color: 'var(--text-secondary)' }}
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
              onClick={() => { setImage(null); setImageTokens(null) }}
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
          borderRadius: "16px",
          bgcolor: "var(--bg-primary)",
          border: isDragging
            ? "2px dashed var(--brand-primary)"
            : "1px solid var(--border-color)",
          transition: "all 0.2s ease",
          "&:focus-within": {
            borderColor: isDragging ? "var(--brand-primary)" : "var(--text-secondary)",
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
            "&:hover": { color: "var(--text-primary)", bgcolor: "var(--bg-secondary)" },
            "&.Mui-disabled": {
              color: "var(--text-secondary)",
              opacity: 0.5
            }
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
          placeholder={isDragging ? "Rilascia l'immagine qui…" : placeholder}
          disabled={disabled}
          sx={{
            px: 1,
            py: 0.75,
            alignItems: "flex-start",
            fontSize: 14,
            lineHeight: 1.65,
            color: "var(--text-primary)",
            "& textarea": { resize: "none" },
            "& textarea:disabled": { WebkitTextFillColor: "var(--text-secondary)", color: "var(--text-secondary)" },
            "& textarea::placeholder": { color: "var(--text-secondary)", opacity: 0.5 }
          }}
        />

        <IconButton
          aria-label="Invia messaggio"
          onClick={handleSend}
          disabled={!canSend}
          sx={{
            width: 32,
            height: 32,
            mb: 0.25,
            flexShrink: 0,
            bgcolor: canSend ? "var(--bg-sidebar)" : "transparent",
            color: canSend ? "var(--text-primary)" : "var(--text-secondary)",
            transition: "all 0.15s ease",
            "&:hover": {
              bgcolor: canSend ? "var(--border-color)" : "transparent",
            },
            "&.Mui-disabled": {
              color: "var(--text-secondary)",
              opacity: 0.5
            }
          }}
        >
          <ArrowUpwardIcon sx={{ fontSize: 17 }} />
        </IconButton>
      </Box>

      {/* Footer: hint + token counter */}
      <Box
        sx={{
          mt: 0.75,
          px: 1.75,
          display: "flex",
          alignItems: "center",
          justifyContent: "space-between",
          userSelect: "none"
        }}
      >
        <Typography
          sx={{
            fontSize: 11,
            color: "var(--text-secondary)",
            opacity: 0.5,
          }}
        >
          Enter per inviare · Shift + Enter per andare a capo
        </Typography>

        {showTokens && (
          <Typography
            sx={{
              fontSize: 11,
              color: "var(--text-secondary)",
              opacity: 0.6,
              display: "flex",
              alignItems: "center",
              gap: 0.5,
              fontVariantNumeric: "tabular-nums"
            }}
          >
            ~{totalTokens} tok
            {imageTokens !== null && (
              <Box
                component="span"
                sx={{
                  ml: 0.25,
                  px: 0.75,
                  py: 0.1,
                  borderRadius: "4px",
                  bgcolor: "var(--bg-secondary)",
                  fontSize: 10,
                  color: "var(--text-secondary)"
                }}
              >
                🖼 +{imageTokens}
              </Box>
            )}
          </Typography>
        )}
      </Box>

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
                    borderColor: "var(--text-secondary)"
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

      {/* Playwright Query Modal */}
      <Dialog
        open={isPlaywrightModalOpen}
        onClose={() => setIsPlaywrightModalOpen(false)}
        maxWidth="xs"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: "20px",
            bgcolor: "var(--bg-primary)",
            backgroundImage: "none",
            boxShadow: "0 24px 48px -12px rgba(0,0,0,0.15)",
            border: "1px solid var(--border-color)",
            p: 1
          }
        }}
      >
        <DialogTitle sx={{
          pt: 3,
          pb: 1,
          px: 3,
          color: "var(--text-primary)",
          display: "flex",
          flexDirection: "column",
          gap: 1
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <TravelExploreIcon sx={{ color: '#3b82f6', fontSize: 28 }} />
            <Typography sx={{ fontWeight: 800, fontSize: 22, letterSpacing: '-0.02em' }}>Ricerca Live Browser</Typography>
          </Box>
          <Typography variant="body2" sx={{ color: "var(--text-secondary)", fontWeight: 500, opacity: 0.8 }}>
            Cosa vuoi cercare con il browser reale in tempo reale?
          </Typography>
        </DialogTitle>

        <DialogContent sx={{ px: 3, pb: 1, pt: 2 }}>
          <TextField
            fullWidth
            autoFocus
            placeholder="Es: iPhone 15 Pro Max..."
            variant="outlined"
            value={playwrightQuery}
            onChange={(e) => setPlaywrightQuery(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handlePlaywrightConfirm()
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: '12px',
                bgcolor: 'var(--bg-secondary)',
                '& fieldset': { borderColor: 'var(--border-color)' },
                '&:hover fieldset': { borderColor: 'var(--text-secondary)' },
                '&.Mui-focused fieldset': { borderColor: '#3b82f6' },
              },
              '& .MuiInputBase-input': {
                color: 'var(--text-primary)',
                fontSize: '15px'
              }
            }}
          />
        </DialogContent>

        <DialogActions sx={{ p: 3, pt: 1 }}>
          <Button
            onClick={() => setIsPlaywrightModalOpen(false)}
            sx={{ color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'none' }}
          >
            Annulla
          </Button>
          <Button
            onClick={handlePlaywrightConfirm}
            variant="contained"
            disableElevation
            disabled={!playwrightQuery.trim()}
            sx={{
              borderRadius: '10px',
              bgcolor: '#3b82f6',
              textTransform: 'none',
              fontWeight: 700,
              px: 3,
              '&:hover': { bgcolor: '#2563eb' }
            }}
          >
            Avvia Ricerca
          </Button>
        </DialogActions>
      </Dialog>

      {/* Contact Seller Modal */}
      <Dialog
        open={isContactSellerModalOpen}
        onClose={() => setIsContactSellerModalOpen(false)}
        maxWidth="xs"
        fullWidth
        PaperProps={{
          sx: {
            borderRadius: "20px",
            bgcolor: "var(--bg-primary)",
            backgroundImage: "none",
            boxShadow: "0 24px 48px -12px rgba(0,0,0,0.15)",
            border: "1px solid var(--border-color)",
            p: 1
          }
        }}
      >
        <DialogTitle sx={{
          pt: 3,
          pb: 1,
          px: 3,
          color: "var(--text-primary)",
          display: "flex",
          flexDirection: "column",
          gap: 1
        }}>
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1.5 }}>
            <MailOutlineIcon sx={{ color: '#10b981', fontSize: 28 }} />
            <Typography sx={{ fontWeight: 800, fontSize: 22, letterSpacing: '-0.02em' }}>Contatta Venditore</Typography>
          </Box>
          <Typography variant="body2" sx={{ color: "var(--text-secondary)", fontWeight: 500, opacity: 0.8 }}>
            Inserisci il nome utente del venditore eBay da contattare
          </Typography>
        </DialogTitle>

        <DialogContent sx={{ px: 3, pb: 1, pt: 2 }}>
          <TextField
            fullWidth
            autoFocus
            placeholder="Es: jjtech2020"
            variant="outlined"
            value={contactSellerName}
            onChange={(e) => setContactSellerName(e.target.value)}
            onKeyDown={(e) => {
              if (e.key === 'Enter') handleContactSellerConfirm()
            }}
            sx={{
              '& .MuiOutlinedInput-root': {
                borderRadius: '12px',
                bgcolor: 'var(--bg-secondary)',
                '& fieldset': { borderColor: 'var(--border-color)' },
                '&:hover fieldset': { borderColor: 'var(--text-secondary)' },
                '&.Mui-focused fieldset': { borderColor: '#10b981' },
              },
              '& .MuiInputBase-input': {
                color: 'var(--text-primary)',
                fontSize: '15px'
              }
            }}
          />
        </DialogContent>

        <DialogActions sx={{ p: 3, pt: 1 }}>
          <Button
            onClick={() => setIsContactSellerModalOpen(false)}
            sx={{ color: 'var(--text-secondary)', fontWeight: 600, textTransform: 'none' }}
          >
            Annulla
          </Button>
          <Button
            onClick={handleContactSellerConfirm}
            variant="contained"
            disableElevation
            disabled={!contactSellerName.trim()}
            sx={{
              borderRadius: '10px',
              bgcolor: '#10b981',
              textTransform: 'none',
              fontWeight: 700,
              px: 3,
              '&:hover': { bgcolor: '#059669' }
            }}
          >
            Invia Messaggio
          </Button>
        </DialogActions>
      </Dialog>
    </Box>
  )
}
import { useState, useEffect } from "react"
import {
  Dialog,
  DialogTitle,
  DialogContent,
  DialogActions,
  Button,
  Box,
  Typography,
  Switch,
  TextField,
  FormControl,
  Select,
  MenuItem,
  CircularProgress,
  Divider,
} from "@mui/material"
import DownloadIcon from "@mui/icons-material/Download"
// @ts-ignore
import html2pdf from "html2pdf.js"

import type { UserSettings } from "./store/settingsStore"
import { useSettingsStore } from "./store/settingsStore"
import { getToken } from "../../auth/authStore"

export default function SettingsModal() {
  const { isOpen, setOpen, settings, updateLocalSettings, saveSettingsToBackend, isSaving } = useSettingsStore()
  const [localSettings, setLocalSettings] = useState<UserSettings>(settings)
  // Quando si apre, copiamo i setting globali
  useEffect(() => {
    if (isOpen) {
      setLocalSettings(settings)
    }
  }, [isOpen, settings])

  const handleClose = () => {
    setOpen(false)
  }

  const handleSave = async () => {
    const token = getToken()
    if (token) {
      try {
        await saveSettingsToBackend(token, localSettings)
        setOpen(false)
      } catch (e) {
        // error handling handled in store or UI
        alert("Errore salvataggio")
      }
    } else {
      // Se ospite, aggiorna solo local
      updateLocalSettings(localSettings)
      setOpen(false)
    }
  }

  const exportPDF = () => {
    const element = document.getElementById("chat-scroll-container")
    if (!element) return

    const opt = {
      margin:       1,
      filename:     'ebayGPT-chat.pdf',
      image:        { type: 'jpeg' as const, quality: 0.98 },
      html2canvas:  { scale: 2, useCORS: true },
      jsPDF:        { unit: 'in', format: 'a4', orientation: 'portrait' as const }
    }
    
    html2pdf().set(opt).from(element).save()
  }

  return (
    <Dialog 
      open={isOpen} 
      onClose={handleClose}
      maxWidth="sm" 
      fullWidth
      slotProps={{
        backdrop: {
          sx: { backdropFilter: 'blur(8px)', bgcolor: 'rgba(0,0,0,0.4)' }
        }
      }}
      PaperProps={{
        sx: {
          borderRadius: 4,
          bgcolor: "var(--bg-primary)",
          backgroundImage: 'none',
          color: "var(--text-primary)",
          boxShadow: '0 24px 48px rgba(0,0,0,0.2), 0 0 0 1px var(--border-color)',
          overflow: 'hidden'
        }
      }}
    >
      <DialogTitle sx={{ fontWeight: 700, pb: 1 }}>
        ⚙️ Impostazioni
      </DialogTitle>
      <DialogContent sx={{ display: 'flex', flexDirection: 'column', gap: 3, pt: 2 }}>
        
        {/* TEMA */}
        <Box sx={{ display: "flex", justifyContent: "space-between", alignItems: "center", mt: 1 }}>
          <Box>
            <Typography fontWeight={600} fontSize={15}>Tema Scuro</Typography>
            <Typography variant="body2" color="var(--text-secondary)">Passa dalla modalità chiara alla scura</Typography>
          </Box>
          <Switch
            checked={localSettings.theme === 'dark'}
            onChange={(e) => setLocalSettings({ ...localSettings, theme: e.target.checked ? 'dark' : 'light' })}
            color="primary"
          />
        </Box>

        <Divider sx={{ borderColor: 'var(--border-color)' }} />

        {/* TONO CONVERSAZIONE */}
        <Box>
          <Typography fontWeight={600} fontSize={15} gutterBottom>Tono Conversazione</Typography>
          <Typography variant="body2" color="var(--text-secondary)" mb={2}>
            Scegli come l'agente deve risponderti
          </Typography>
          <FormControl fullWidth size="small">
            <Select
              value={localSettings.conversationTone}
              onChange={(e) => setLocalSettings({ ...localSettings, conversationTone: e.target.value as any })}
              sx={{ 
                bgcolor: 'var(--bg-secondary)', 
                color: 'var(--text-primary)',
                '.MuiOutlinedInput-notchedOutline': { borderColor: 'var(--border-color)' } 
              }}
            >
              <MenuItem value="neutral">Neutro</MenuItem>
              <MenuItem value="amichevole">Amichevole</MenuItem>
              <MenuItem value="professionale">Professionale</MenuItem>
            </Select>
          </FormControl>
        </Box>

        <Divider sx={{ borderColor: 'var(--border-color)' }} />

        {/* CUSTOM INSTRUCTIONS GEM */}
        <Box>
          <Typography fontWeight={600} fontSize={15} gutterBottom>Istruzioni Personalizzate (GEM)</Typography>
          <Typography variant="body2" color="var(--text-secondary)" mb={2}>
            Regole generali che l'agente deve sempre rispettare (es. "Rispondi sempre in inglese" o "Concentrati solo sui prodotti ricondizionati")
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={3}
            placeholder="Le tue direttive..."
            value={localSettings.customInstructions}
            onChange={(e) => setLocalSettings({ ...localSettings, customInstructions: e.target.value })}
            InputProps={{
              sx: {
                bgcolor: 'var(--bg-secondary)',
                color: 'var(--text-primary)',
                '& fieldset': { borderColor: 'var(--border-color)' }
              }
            }}
          />
        </Box>

        <Divider sx={{ borderColor: 'var(--border-color)' }} />

        {/* EXPORT CHAT */}
        <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
          <Box>
            <Typography fontWeight={600} fontSize={15}>Esporta Chat</Typography>
            <Typography variant="body2" color="var(--text-secondary)">
              Scarica in PDF l'intera conversazione attuale
            </Typography>
          </Box>
          <Button 
            variant="outlined" 
            startIcon={<DownloadIcon />}
            onClick={exportPDF}
            sx={{
              textTransform: 'none',
              borderColor: 'var(--border-color)',
              color: 'var(--text-primary)',
              '&:hover': { bgcolor: 'var(--bg-secondary)', borderColor: 'var(--brand-primary)', color: 'var(--brand-primary)' }
            }}
          >
            PDF
          </Button>
        </Box>

      </DialogContent>
      <DialogActions sx={{ px: 3, pb: 3, pt: 1 }}>
        <Button 
          onClick={handleClose} 
          sx={{ color: "var(--text-secondary)", textTransform: "none", fontWeight: 600 }}
        >
          Annulla
        </Button>
        <Button 
          onClick={handleSave} 
          variant="contained" 
          disabled={isSaving}
          sx={{
            bgcolor: "var(--brand-primary)", 
            color: "#fff", 
            textTransform: "none",
            fontWeight: 600,
            "&:hover": { bgcolor: "var(--brand-primary)", opacity: 0.9 }
          }}
        >
          {isSaving ? <CircularProgress size={20} color="inherit" /> : "Salva Preferenze"}
        </Button>
      </DialogActions>
    </Dialog>
  )
}

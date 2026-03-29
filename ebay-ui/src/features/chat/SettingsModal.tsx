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
  Chip,
} from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import InsightsIcon from "@mui/icons-material/Insights"
import LayersIcon from "@mui/icons-material/Layers"
import HistoryIcon from "@mui/icons-material/History"
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

  // Helpers for AI labels
  const getConditionLabel = (raw?: string) => {
    if (!raw) return "Nessuna preferenza rilevata"
    // raw è formato "new:3,used:1" -> prendiamo il primo
    const match = raw.split(",")[0].split(":")[0]
    const labels: Record<string, string> = {
      new: "Preferisci prodotti Nuovi",
      used: "Preferisci prodotti Usati",
      refurbished: "Preferisci prodotti Ricondizionati"
    }
    return labels[match] || match
  }

  const getDepthLabel = (depth?: string) => {
    const labels: Record<string, { title: string, desc: string }> = {
      browser: { title: "Browser", desc: "Stai esplorando le basi" },
      researcher: { title: "Ricercatore", desc: "Analizzi dettagli e confronti" },
      power_buyer: { title: "Esperto", desc: "Interagisci attivamente con i venditori" }
    }
    return labels[depth || "browser"] || labels.browser
  }

  const getTopCategories = (raw?: string) => {
    if (!raw) return []
    try {
      const data = JSON.parse(raw)
      return Object.entries(data)
        .sort((a: any, b: any) => b[1] - a[1])
        .slice(0, 3)
        .map(([name]) => name)
    } catch {
      return []
    }
  }

  const topCategories = getTopCategories(localSettings.categoryAffinities)
  const depthInfo = getDepthLabel(localSettings.interactionDepth)

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
            Regole generali che l'agente deve sempre rispettare (es. "Rispondi sempre in inglese")
          </Typography>
          <TextField
            fullWidth
            multiline
            rows={2}
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

        {/* SHOPPING PROFILE (BRANDS & PRICE) */}
        <Box sx={{ display: 'flex', gap: 2 }}>
          <Box sx={{ flex: 1 }}>
            <Typography fontWeight={600} fontSize={15} gutterBottom>Brand Preferiti</Typography>
            <TextField
              fullWidth
              size="small"
              placeholder="Nike, Apple, Sony..."
              value={localSettings.favoriteBrands}
              onChange={(e) => setLocalSettings({ ...localSettings, favoriteBrands: e.target.value })}
              InputProps={{
                sx: { 
                    bgcolor: 'var(--bg-secondary)', 
                    color: 'var(--text-primary)',
                    '& fieldset': { borderColor: 'var(--border-color)' }
                }
              }}
            />
          </Box>
          <Box sx={{ width: 120 }}>
            <Typography fontWeight={600} fontSize={15} gutterBottom>Budget (€)</Typography>
            <TextField
              fullWidth
              size="small"
              type="number"
              placeholder="es. 500"
              value={localSettings.pricePreference}
              onChange={(e) => setLocalSettings({ ...localSettings, pricePreference: e.target.value })}
              InputProps={{
                sx: { 
                    bgcolor: 'var(--bg-secondary)', 
                    color: 'var(--text-primary)',
                    '& fieldset': { borderColor: 'var(--border-color)' }
                }
              }}
            />
          </Box>
        </Box>
        <Divider sx={{ borderColor: 'var(--border-color)' }} />

        {/* AI LEARNED PROFILE */}
        <Box 
          sx={{ 
            p: 2, 
            borderRadius: 3, 
            bgcolor: 'rgba(139, 92, 246, 0.05)', 
            border: '1px solid rgba(139, 92, 246, 0.15)',
            position: 'relative',
            overflow: 'hidden'
          }}
        >
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 1, mb: 1.5 }}>
            <AutoAwesomeIcon sx={{ fontSize: 18, color: '#8b5cf6' }} />
            <Typography fontWeight={700} fontSize={14} color="#8b5cf6" sx={{ textTransform: 'uppercase', letterSpacing: 0.5 }}>
              Profilo Appreso dall'AI
            </Typography>
          </Box>
          <Typography variant="body2" color="var(--text-secondary)" mb={2}>
            L'agente impara dalle tue interazioni per offrirti risultati più rilevanti.
          </Typography>

          <Box sx={{ display: 'flex', flexDirection: 'column', gap: 2 }}>
            {/* Condition */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <HistoryIcon sx={{ fontSize: 20, color: 'var(--text-secondary)', opacity: 0.7 }} />
              <Box>
                <Typography fontSize={13} fontWeight={600}>{getConditionLabel(localSettings.conditionPreference)}</Typography>
                <Typography fontSize={11} color="var(--text-secondary)">Basato sui tuoi click e filtri</Typography>
              </Box>
            </Box>

            {/* Depth / Engagement */}
            <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
              <InsightsIcon sx={{ fontSize: 20, color: 'var(--text-secondary)', opacity: 0.7 }} />
              <Box>
                <Typography fontSize={13} fontWeight={600}>Livello: {depthInfo.title}</Typography>
                <Typography fontSize={11} color="var(--text-secondary)">{depthInfo.desc}</Typography>
              </Box>
            </Box>

            {/* Categories */}
            {topCategories.length > 0 && (
              <Box sx={{ display: 'flex', alignItems: 'flex-start', gap: 2 }}>
                <LayersIcon sx={{ fontSize: 20, color: 'var(--text-secondary)', opacity: 0.7, mt: 0.5 }} />
                <Box>
                  <Typography fontSize={13} fontWeight={600} mb={0.5}>Interessi Principali</Typography>
                  <Box sx={{ display: 'flex', flexWrap: 'wrap', gap: 0.5 }}>
                    {topCategories.map((cat, i) => (
                      <Chip 
                        key={i} 
                        label={cat} 
                        size="small" 
                        sx={{ 
                          height: 18, 
                          fontSize: 9, 
                          fontWeight: 600,
                          bgcolor: 'var(--bg-secondary)',
                          color: 'var(--text-primary)',
                          border: '1px solid var(--border-color)'
                        }} 
                      />
                    ))}
                  </Box>
                </Box>
              </Box>
            )}
          </Box>
        </Box>

        <Divider sx={{ borderColor: 'var(--border-color)' }} />

        {/* MONDO MCP */}
        <Box>
          <Typography fontWeight={600} fontSize={15} gutterBottom>Mondo MCP</Typography>
          <Typography variant="body2" color="var(--text-secondary)" mb={2}>
            Scegli come l'agente accede a eBay
          </Typography>
          <Box sx={{ display: 'flex', justifyContent: 'space-between', alignItems: 'center' }}>
            <Box>
              <Typography fontSize={14} fontWeight={500}>
                {localSettings.mcpMode === 'playwright_browser'
                  ? '🌐 Browser Playwright (Visibile)'
                  : '⚡ Standard (API eBay)'}
              </Typography>
              <Typography variant="body2" color="var(--text-secondary)" mt={0.5}>
                {localSettings.mcpMode === 'playwright_browser'
                  ? 'Chromium si aprirà visibile — permette di cercare e contattare venditori'
                  : 'API ufficiali eBay, veloce e affidabile'}
              </Typography>
            </Box>
            <Switch
              checked={localSettings.mcpMode === 'playwright_browser'}
              onChange={(e) =>
                setLocalSettings({
                  ...localSettings,
                  mcpMode: e.target.checked ? 'playwright_browser' : 'standard',
                })
              }
              color="primary"
            />
          </Box>
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

import { useState, useEffect } from "react"
import {
  Dialog,
  DialogContent,
  DialogActions,
  Button,
  TextField,
  Typography,
  CircularProgress,
  Box,
  IconButton
} from "@mui/material"
import CloseIcon from "@mui/icons-material/Close"
import TuneIcon from "@mui/icons-material/Tune"

import { useAuth } from "../useAuth"
import { updateCustomInstructions } from "../authService"

interface CustomInstructionsModalProps {
  open: boolean
  onClose: () => void
}

export function CustomInstructionsModal({ open, onClose }: CustomInstructionsModalProps) {
  const { user } = useAuth()
  const [instructions, setInstructions] = useState(user?.custom_instructions || "")
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (open) {
      setInstructions(user?.custom_instructions || "")
      setError(null)
    }
  }, [open, user?.custom_instructions])

  const handleSave = async () => {
    setLoading(true)
    setError(null)
    try {
      await updateCustomInstructions(instructions)
      if (user) user.custom_instructions = instructions
      onClose()
    } catch (err: any) {
      setError(err.message || "Errore durante il salvataggio")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog
      open={open}
      onClose={loading ? undefined : onClose}
      fullWidth
      maxWidth="sm"
      PaperProps={{
        sx: {
          width: "100%",
          maxWidth: 480,
          borderRadius: "16px",
          border: "1px solid #e5e5e5",
          boxShadow: "0 4px 32px rgba(0,0,0,0.10)",
          p: 0
        }
      }}
    >
      {/* Header */}
      <Box sx={{ px: 4, pt: 4, pb: 0, position: "relative" }}>
        <IconButton
          onClick={onClose}
          disabled={loading}
          size="small"
          sx={{
            position: "absolute",
            top: 16,
            right: 16,
            color: "#999",
            "&:hover": { bgcolor: "#f5f5f5" }
          }}
        >
          <CloseIcon fontSize="small" />
        </IconButton>

        <Box
          sx={{
            width: 38,
            height: 38,
            borderRadius: "10px",
            bgcolor: "#f4f4f5",
            display: "grid",
            placeItems: "center",
            mb: 2.5
          }}
        >
          <TuneIcon sx={{ fontSize: 18, color: "#202123" }} />
        </Box>

        <Typography
          sx={{
            fontSize: 22,
            fontWeight: 700,
            color: "#0d0d0d",
            letterSpacing: "-0.02em",
            lineHeight: 1.2,
            mb: 0.75
          }}
        >
          Istruzioni personalizzate
        </Typography>

        <Typography sx={{ fontSize: 13, color: "#6e6e80", lineHeight: 1.55, mb: 3 }}>
          Definisci le tue preferenze globali. Queste istruzioni guidano il comportamento
          di ebayGPT in ogni ricerca.
        </Typography>
      </Box>

      <DialogContent sx={{ px: 4, pt: 0, pb: 0 }}>
        <TextField
          fullWidth
          multiline
          rows={6}
          placeholder='es. "Cerca solo prodotti Nuovi con spedizione gratuita. Ignora i venditori con meno di 500 feedback."'
          value={instructions}
          onChange={(e) => setInstructions(e.target.value)}
          disabled={loading}
          variant="outlined"
          sx={{
            "& .MuiOutlinedInput-root": {
              borderRadius: "10px",
              bgcolor: "#fafafa",
              fontSize: 13,
              lineHeight: 1.6,
              "& fieldset": { borderColor: "#d9d9e3" },
              "&:hover fieldset": { borderColor: "#b0b0bc" },
              "&.Mui-focused fieldset": { borderColor: "#202123", borderWidth: 1.5 }
            }
          }}
        />

        {error && (
          <Typography color="error" sx={{ mt: 1, fontSize: 12 }}>
            {error}
          </Typography>
        )}
      </DialogContent>

      <DialogActions sx={{ px: 4, pb: 4, pt: 2.5, gap: 1 }}>
        <Button
          onClick={onClose}
          disabled={loading}
          sx={{
            textTransform: "none",
            borderRadius: "10px",
            px: 2.5,
            py: 1,
            fontWeight: 500,
            fontSize: 13,
            color: "#6e6e80",
            "&:hover": { bgcolor: "#f5f5f5" }
          }}
        >
          Annulla
        </Button>
        <Button
          onClick={handleSave}
          disabled={loading}
          variant="contained"
          disableElevation
          startIcon={loading ? <CircularProgress size={14} color="inherit" /> : null}
          sx={{
            textTransform: "none",
            borderRadius: "10px",
            px: 2.5,
            py: 1,
            fontWeight: 600,
            fontSize: 13,
            bgcolor: "#202123",
            boxShadow: "none",
            "&:hover": { bgcolor: "#111214" },
            "&:disabled": { bgcolor: "rgba(32,33,35,0.4)", color: "#fff" }
          }}
        >
          {loading ? "Salvataggio…" : "Salva"}
        </Button>
      </DialogActions>
    </Dialog>
  )
}
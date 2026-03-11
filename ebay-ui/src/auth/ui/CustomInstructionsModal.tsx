import { useState, useEffect } from "react"
import {
    Dialog,
    DialogTitle,
    DialogContent,
    DialogActions,
    Button,
    TextField,
    Typography,
    CircularProgress
} from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"

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

    // Re-sync on open if needed
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
            // Force reload user in auth store if we had a dedicated fetch, 
            // but reloading the page or trusting the optimistic update is fine for now
            if (user) {
                user.custom_instructions = instructions
            }
            onClose()
        } catch (err: any) {
            setError(err.message || "Errore durante il salvataggio")
        } finally {
            setLoading(false)
        }
    }

    return (
        <Dialog open={open} onClose={onClose} fullWidth maxWidth="sm">
            <DialogTitle sx={{ display: 'flex', alignItems: 'center', gap: 1 }}>
                <AutoAwesomeIcon color="primary" />
                Istruzioni Personalizzate
            </DialogTitle>
            <DialogContent dividers>
                <Typography variant="body2" color="text.secondary" sx={{ mb: 2 }}>
                    Queste istruzioni guidano il comportamento di ebayGPT.
                    Puoi usarle per definire le tue preferenze globali (es. "Rispondi in modo molto sintetico",
                    "Cerca sempre e solo prodotti Nuovi", "Ignora i venditori con meno di 500 feedback").
                </Typography>

                <TextField
                    fullWidth
                    multiline
                    rows={6}
                    placeholder="es. Preferisco sempre le spedizioni gratuite e le risposte in formato elenco."
                    value={instructions}
                    onChange={(e) => setInstructions(e.target.value)}
                    disabled={loading}
                    variant="outlined"
                    sx={{
                        "& .MuiOutlinedInput-root": {
                            bgcolor: 'rgba(255, 255, 255, 0.05)',
                            backdropFilter: 'blur(10px)',
                        }
                    }}
                />

                {error && (
                    <Typography color="error" variant="caption" sx={{ mt: 1, display: 'block' }}>
                        {error}
                    </Typography>
                )}
            </DialogContent>
            <DialogActions sx={{ px: 3, pb: 2 }}>
                <Button onClick={onClose} disabled={loading} color="inherit">
                    Annulla
                </Button>
                <Button
                    onClick={handleSave}
                    disabled={loading}
                    variant="contained"
                    disableElevation
                    startIcon={loading ? <CircularProgress size={16} color="inherit" /> : null}
                    sx={{
                        background: "linear-gradient(135deg, #1976d2 0%, #1565c0 100%)",
                        textTransform: 'none',
                        fontWeight: 600,
                    }}
                >
                    {loading ? "Salvataggio..." : "Salva"}
                </Button>
            </DialogActions>
        </Dialog>
    )
}

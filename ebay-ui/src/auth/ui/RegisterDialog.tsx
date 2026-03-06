import {
  Alert,
  Box,
  Button,
  Dialog,
  DialogContent,
  DialogTitle,
  IconButton,
  InputAdornment,
  Stack,
  TextField,
  Typography
} from "@mui/material"
import CloseIcon from "@mui/icons-material/Close"
import VisibilityIcon from "@mui/icons-material/Visibility"
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff"
import { useEffect, useState } from "react"
import { useAuth } from "../useAuth"

export default function RegisterDialog({
  open,
  onClose,
  onLogin
}: {
  open: boolean
  onClose: () => void
  onLogin: () => void
}) {
  const { register } = useAuth()

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [confirmPassword, setConfirmPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) {
      setEmail("")
      setPassword("")
      setConfirmPassword("")
      setShowPassword(false)
      setError(null)
      setLoading(false)
    }
  }, [open])

  async function handleRegister() {
    if (!email.trim() || !password.trim()) {
      setError("Inserisci email e password.")
      return
    }

    if (password.length < 6) {
      setError("La password deve contenere almeno 6 caratteri.")
      return
    }

    if (password !== confirmPassword) {
      setError("Le password non coincidono.")
      return
    }

    try {
      setLoading(true)
      setError(null)

      await register(email.trim(), password)

      onClose()
    } catch (err: any) {
      setError(err?.message || "Registrazione non riuscita.")
    } finally {
      setLoading(false)
    }
  }

  return (
    <Dialog
      open={open}
      onClose={loading ? undefined : onClose}
      maxWidth="xs"
      fullWidth
      PaperProps={{
        sx: {
          borderRadius: 4,
          p: 1,
          boxShadow: "0 20px 60px rgba(0,0,0,0.18)"
        }
      }}
    >
      <DialogTitle sx={{ pb: 0.5 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box>
            <Typography sx={{ fontSize: 24, fontWeight: 700, color: "#202123" }}>
              Crea account
            </Typography>
            <Typography sx={{ fontSize: 13, color: "#6e6e80", mt: 0.5 }}>
              Registrati per salvare preferenze e ottenere risultati più rilevanti.
            </Typography>
          </Box>

          <IconButton onClick={onClose} disabled={loading} size="small">
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ pt: 2 }}>
        <Stack spacing={2}>
          {error && <Alert severity="error">{error}</Alert>}

          <TextField
            label="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            fullWidth
            autoFocus
            disabled={loading}
            autoComplete="email"
          />

          <TextField
            label="Password"
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            fullWidth
            disabled={loading}
            autoComplete="new-password"
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    edge="end"
                    onClick={() => setShowPassword(prev => !prev)}
                    disabled={loading}
                  >
                    {showPassword ? <VisibilityOffIcon /> : <VisibilityIcon />}
                  </IconButton>
                </InputAdornment>
              )
            }}
          />

          <TextField
            label="Conferma password"
            type={showPassword ? "text" : "password"}
            value={confirmPassword}
            onChange={(e) => setConfirmPassword(e.target.value)}
            fullWidth
            disabled={loading}
            autoComplete="new-password"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleRegister()
              }
            }}
          />

          <Button
            variant="contained"
            onClick={handleRegister}
            disabled={loading}
            sx={{
              textTransform: "none",
              borderRadius: 2,
              py: 1.2,
              fontWeight: 600,
              bgcolor: "#202123",
              "&:hover": {
                bgcolor: "#111214"
              }
            }}
          >
            {loading ? "Creazione account..." : "Registrati"}
          </Button>

          <Button
            onClick={onLogin}
            disabled={loading}
            sx={{
              textTransform: "none",
              fontWeight: 600
            }}
          >
            Hai già un account? Accedi
          </Button>
        </Stack>
      </DialogContent>
    </Dialog>
  )
}
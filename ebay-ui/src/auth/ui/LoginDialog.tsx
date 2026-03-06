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

export default function LoginDialog({
  open,
  onClose,
  onRegister
}: {
  open: boolean
  onClose: () => void
  onRegister: () => void
}) {
  const { login } = useAuth()

  const [email, setEmail] = useState("")
  const [password, setPassword] = useState("")
  const [showPassword, setShowPassword] = useState(false)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    if (!open) {
      setEmail("")
      setPassword("")
      setShowPassword(false)
      setError(null)
      setLoading(false)
    }
  }, [open])

  async function handleLogin() {
    if (!email.trim() || !password.trim()) {
      setError("Inserisci email e password.")
      return
    }

    try {
      setLoading(true)
      setError(null)

      await login(email.trim(), password)

      onClose()
    } catch (err: any) {
      setError(err?.message || "Accesso non riuscito.")
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
              Accedi
            </Typography>
            <Typography sx={{ fontSize: 13, color: "#6e6e80", mt: 0.5 }}>
              Continua la tua esperienza di ricerca personalizzata.
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
            autoComplete="current-password"
            onKeyDown={(e) => {
              if (e.key === "Enter") {
                handleLogin()
              }
            }}
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

          <Button
            variant="contained"
            onClick={handleLogin}
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
            {loading ? "Accesso in corso..." : "Accedi"}
          </Button>

          <Button
            onClick={onRegister}
            disabled={loading}
            sx={{
              textTransform: "none",
              fontWeight: 600
            }}
          >
            Non hai un account? Registrati
          </Button>
        </Stack>
      </DialogContent>
    </Dialog>
  )
}
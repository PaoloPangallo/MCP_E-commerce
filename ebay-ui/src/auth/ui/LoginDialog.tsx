import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogContent,
  Divider,
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
      fullWidth
      maxWidth="sm"
      PaperProps={{
        sx: {
          width: "100%",
          maxWidth: 420,
border: "1px solid #e5e5e5",
          boxShadow: "0 4px 32px rgba(0,0,0,0.10)",
          overflow: "hidden",
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

        {/* Logo wordmark */}
        <Typography
          sx={{
            fontSize: 13,
            fontWeight: 600,
            color: "#202123",
            letterSpacing: "0.04em",
            textTransform: "uppercase",
            mb: 3
          }}
        >
          ebayGPT
        </Typography>

        <Typography
          sx={{
            fontSize: 28,
            fontWeight: 700,
            color: "#0d0d0d",
            lineHeight: 1.2,
            letterSpacing: "-0.02em",
            mb: 0.75
          }}
        >
          Bentornato
        </Typography>

        <Typography
          sx={{
            fontSize: 14,
            color: "#6e6e80",
            lineHeight: 1.5,
            mb: 3
          }}
        >
          Accedi al tuo account per continuare.
        </Typography>
      </Box>

      <DialogContent sx={{ px: 4, pt: 0, pb: 4 }}>
        <Stack spacing={2}>
          {error && (
            <Alert
              severity="error"
              sx={{ borderRadius: "10px", fontSize: 13 }}
            >
              {error}
            </Alert>
          )}

          <TextField
            label="Indirizzo email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            fullWidth
            autoFocus
            disabled={loading}
            autoComplete="email"
            variant="outlined"
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: "10px",
                bgcolor: "#fff",
                fontSize: 14,
                "& fieldset": { borderColor: "#d9d9e3" },
                "&:hover fieldset": { borderColor: "#b0b0bc" },
                "&.Mui-focused fieldset": { borderColor: "#202123", borderWidth: 1.5 }
              },
              "& .MuiInputLabel-root": { fontSize: 14 }
            }}
          />

          <TextField
            label="Password"
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            fullWidth
            disabled={loading}
            autoComplete="current-password"
            onKeyDown={(e) => { if (e.key === "Enter") handleLogin() }}
            variant="outlined"
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: "10px",
                bgcolor: "#fff",
                fontSize: 14,
                "& fieldset": { borderColor: "#d9d9e3" },
                "&:hover fieldset": { borderColor: "#b0b0bc" },
                "&.Mui-focused fieldset": { borderColor: "#202123", borderWidth: 1.5 }
              },
              "& .MuiInputLabel-root": { fontSize: 14 }
            }}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    edge="end"
                    onClick={() => setShowPassword((prev) => !prev)}
                    disabled={loading}
                    sx={{ color: "#aaa" }}
                  >
                    {showPassword ? <VisibilityOffIcon sx={{ fontSize: 18 }} /> : <VisibilityIcon sx={{ fontSize: 18 }} />}
                  </IconButton>
                </InputAdornment>
              )
            }}
          />

          <Button
            fullWidth
            variant="contained"
            onClick={handleLogin}
            disabled={loading}
            sx={{
              mt: 0.5,
              textTransform: "none",
              borderRadius: "10px",
              py: 1.4,
              fontWeight: 600,
              fontSize: 14,
              bgcolor: "#202123",
              color: "#fff",
              boxShadow: "none",
              letterSpacing: "0.01em",
              "&:hover": { bgcolor: "#111214", boxShadow: "none" },
              "&:disabled": { bgcolor: "rgba(32,33,35,0.4)", color: "#fff" }
            }}
          >
            {loading ? (
              <Stack direction="row" spacing={1} alignItems="center">
                <CircularProgress size={16} color="inherit" />
                <span>Accesso in corso…</span>
              </Stack>
            ) : "Continua"}
          </Button>

          <Divider sx={{ fontSize: 12, color: "#c5c5d2", my: 0.5 }}>oppure</Divider>

          <Box sx={{ textAlign: "center" }}>
            <Typography sx={{ fontSize: 13, color: "#6e6e80", display: "inline" }}>
              Non hai un account?{" "}
            </Typography>
            <Button
              onClick={onRegister}
              disabled={loading}
              sx={{
                p: 0,
                minWidth: 0,
                textTransform: "none",
                fontWeight: 600,
                fontSize: 13,
                color: "#202123",
                display: "inline",
                verticalAlign: "baseline",
                "&:hover": { bgcolor: "transparent", textDecoration: "underline" }
              }}
            >
              Registrati
            </Button>
          </Box>

          <Typography sx={{ fontSize: 11, color: "#b0b0bc", textAlign: "center", lineHeight: 1.5 }}>
            Continuando, accetti le condizioni della piattaforma.
          </Typography>
        </Stack>
      </DialogContent>
    </Dialog>
  )
}
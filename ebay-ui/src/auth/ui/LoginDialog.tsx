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
  Typography,
  Snackbar
} from "@mui/material"
import CloseIcon from "@mui/icons-material/Close"
import VisibilityIcon from "@mui/icons-material/Visibility"
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff"
import KeyIcon from "@mui/icons-material/Key"
import { useEffect, useState } from "react"
import { useAuth } from "../useAuth"
import { recoverPassword } from "../authService"

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
  
  const [recoveryLoading, setRecoveryLoading] = useState(false)
  const [recoveryResult, setRecoveryResult] = useState<{ open: boolean, message: string } | null>(null)

  useEffect(() => {
    if (!open) {
      setEmail("")
      setPassword("")
      setShowPassword(false)
      setError(null)
      setLoading(false)
      setRecoveryLoading(false)
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

  async function handleRecoverPassword() {
    if (!email.trim()) {
      setError("Inserisci il tuo indirizzo email per il recupero.")
      return
    }
    try {
      setRecoveryLoading(true)
      setError(null)
      const res = await recoverPassword(email.trim())
      setRecoveryResult({ open: true, message: res.message })
    } catch (err: any) {
      setError(err?.message || "Recupero non riuscito.")
    } finally {
      setRecoveryLoading(false)
    }
  }

  return (
    <>
      <Dialog
        open={open}
        onClose={loading ? undefined : onClose}
        fullWidth
        maxWidth="sm"
        PaperProps={{
          sx: {
            width: "100%",
            maxWidth: 420,
            border: "1px solid var(--border-color)",
            boxShadow: "0 8px 32px rgba(0,0,0,0.15)",
            overflow: "hidden",
            bgcolor: "var(--bg-primary)",
            backgroundImage: "none",
            p: 0,
            borderRadius: "20px"
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
              color: "var(--text-secondary)",
              "&:hover": { bgcolor: "var(--bg-secondary)" }
            }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>

          {/* Logo wordmark */}
          <Typography
            sx={{
              fontSize: 12,
              fontWeight: 700,
              color: "var(--accent-primary)",
              letterSpacing: "0.08em",
              textTransform: "uppercase",
              mb: 3
            }}
          >
            ebayGPT
          </Typography>

          <Typography
            sx={{
              fontSize: 26,
              fontWeight: 800,
              color: "var(--text-primary)",
              lineHeight: 1.2,
              letterSpacing: "-0.02em",
              mb: 1
            }}
          >
            Bentornato
          </Typography>

          <Typography
            sx={{
              fontSize: 14,
              color: "var(--text-secondary)",
              lineHeight: 1.5,
              mb: 3
            }}
          >
            Accedi al tuo account per continuare.
          </Typography>
        </Box>

        <DialogContent sx={{ px: 4, pt: 0, pb: 4 }}>
          <Stack spacing={2.5}>
            {error && (
              <Alert
                severity="error"
                sx={{ 
                  borderRadius: "12px", 
                  fontSize: 13,
                  bgcolor: "rgba(239, 68, 68, 0.08)",
                  color: "#ef4444",
                  border: "1px solid rgba(239, 68, 68, 0.15)",
                  "& .MuiAlert-icon": { color: "#ef4444" }
                }}
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
              disabled={loading || recoveryLoading}
              autoComplete="email"
              variant="outlined"
              sx={{
                "& .MuiOutlinedInput-root": {
                  borderRadius: "12px",
                  bgcolor: "var(--bg-secondary)",
                  fontSize: 14,
                  color: "var(--text-primary)",
                  "& fieldset": { borderColor: "var(--border-color)" },
                  "&:hover fieldset": { borderColor: "var(--text-secondary)" },
                  "&.Mui-focused fieldset": { borderColor: "var(--accent-primary)", borderWidth: 2 }
                },
                "& .MuiInputLabel-root": { 
                  fontSize: 14, 
                  color: "var(--text-secondary)",
                  "&.Mui-focused": { color: "var(--accent-primary)" }
                }
              }}
            />

            <Box>
              <TextField
                label="Password"
                type={showPassword ? "text" : "password"}
                value={password}
                onChange={(e) => setPassword(e.target.value)}
                fullWidth
                disabled={loading || recoveryLoading}
                autoComplete="current-password"
                onKeyDown={(e) => { if (e.key === "Enter") handleLogin() }}
                variant="outlined"
                sx={{
                  "& .MuiOutlinedInput-root": {
                    borderRadius: "12px",
                    bgcolor: "var(--bg-secondary)",
                    fontSize: 14,
                    color: "var(--text-primary)",
                    "& fieldset": { borderColor: "var(--border-color)" },
                    "&:hover fieldset": { borderColor: "var(--text-secondary)" },
                    "&.Mui-focused fieldset": { borderColor: "var(--accent-primary)", borderWidth: 2 }
                  },
                  "& .MuiInputLabel-root": { 
                    fontSize: 14, 
                    color: "var(--text-secondary)",
                    "&.Mui-focused": { color: "var(--accent-primary)" }
                  }
                }}
                InputProps={{
                  endAdornment: (
                    <InputAdornment position="end">
                      <IconButton
                        edge="end"
                        onClick={() => setShowPassword((prev) => !prev)}
                        disabled={loading}
                        sx={{ color: "var(--text-secondary)" }}
                      >
                        {showPassword ? <VisibilityOffIcon sx={{ fontSize: 18 }} /> : <VisibilityIcon sx={{ fontSize: 18 }} />}
                      </IconButton>
                    </InputAdornment>
                  )
                }}
              />
              <Box sx={{ display: 'flex', justifyContent: 'flex-end', mt: 0.75 }}>
                <Button
                  size="small"
                  onClick={handleRecoverPassword}
                  disabled={loading || recoveryLoading}
                  sx={{ 
                    textTransform: "none", 
                    fontSize: 11, 
                    fontWeight: 600, 
                    color: "var(--accent-primary)",
                    p: 0,
                    minWidth: 0,
                    "&:hover": { bgcolor: "transparent", textDecoration: "underline" }
                  }}
                >
                  {recoveryLoading ? <CircularProgress size={10} color="inherit" /> : "Password dimenticata?"}
                </Button>
              </Box>
            </Box>

            <Button
              fullWidth
              variant="contained"
              onClick={handleLogin}
              disabled={loading || recoveryLoading}
              sx={{
                mt: 1,
                textTransform: "none",
                borderRadius: "12px",
                py: 1.5,
                fontWeight: 700,
                fontSize: 14,
                bgcolor: "var(--text-primary)",
                color: "var(--bg-primary)",
                boxShadow: "0 4px 12px rgba(0,0,0,0.15)",
                letterSpacing: "0.01em",
                "&:hover": { 
                  bgcolor: "var(--text-primary)", 
                  opacity: 0.9,
                  boxShadow: "0 6px 16px rgba(0,0,0,0.2)" 
                },
                "&:disabled": { bgcolor: "var(--border-color)", color: "var(--text-secondary)" }
              }}
            >
              {loading ? (
                <Stack direction="row" spacing={1} alignItems="center">
                  <CircularProgress size={16} color="inherit" />
                  <span>Accesso in corso…</span>
                </Stack>
              ) : "Continua"}
            </Button>

            <Divider sx={{ 
              fontSize: 11, 
              color: "var(--text-secondary)", 
              my: 1,
              "&::before, &::after": { borderColor: "var(--border-color)" }
            }}>oppure</Divider>

            <Box sx={{ textAlign: "center" }}>
              <Typography sx={{ fontSize: 13, color: "var(--text-secondary)", display: "inline" }}>
                Non hai un account?{" "}
              </Typography>
              <Button
                onClick={onRegister}
                disabled={loading || recoveryLoading}
                sx={{
                  p: 0,
                  minWidth: 0,
                  textTransform: "none",
                  fontWeight: 700,
                  fontSize: 13,
                  color: "var(--text-primary)",
                  display: "inline",
                  verticalAlign: "baseline",
                  "&:hover": { bgcolor: "transparent", textDecoration: "underline" }
                }}
              >
                Registrati
              </Button>
            </Box>

            <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", opacity: 0.7, textAlign: "center", lineHeight: 1.5 }}>
              Continuando, accetti le condizioni della piattaforma.
            </Typography>
          </Stack>
        </DialogContent>
      </Dialog>

      {/* Recovery Result Snackbar */}
      <Snackbar
        open={!!recoveryResult?.open}
        autoHideDuration={8000}
        onClose={() => setRecoveryResult(null)}
        anchorOrigin={{ vertical: 'bottom', horizontal: 'center' }}
      >
        <Alert 
          onClose={() => setRecoveryResult(null)} 
          severity="success" 
          icon={<KeyIcon fontSize="inherit" />}
          sx={{ 
            width: '100%', 
            borderRadius: "12px",
            bgcolor: "#059669", // Premium Emerald Green
            color: "#fff",
            fontWeight: 600,
            fontSize: 14,
            boxShadow: "0 10px 25px -5px rgba(0,0,0,0.3)",
            "& .MuiAlert-icon": { color: "#fff" },
            "& .MuiAlert-action": { color: "#fff" }
          }}
        >
          {recoveryResult?.message}
        </Alert>
      </Snackbar>
    </>
  )
}
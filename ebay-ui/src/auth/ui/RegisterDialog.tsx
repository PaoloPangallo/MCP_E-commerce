import {
  Alert,
  Box,
  Button,
  CircularProgress,
  Dialog,
  DialogContent,
  DialogTitle,
  Divider,
  IconButton,
  InputAdornment,
  Stack,
  TextField,
  Typography
} from "@mui/material"
import { alpha } from "@mui/material/styles"
import CloseIcon from "@mui/icons-material/Close"
import VisibilityIcon from "@mui/icons-material/Visibility"
import VisibilityOffIcon from "@mui/icons-material/VisibilityOff"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
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
      fullWidth
      maxWidth="sm"
      PaperProps={{
        sx: {
          width: "100%",
          maxWidth: 460,
          borderRadius: 6,
          border: "1px solid #e5e7eb",
          boxShadow: "0 28px 90px rgba(0,0,0,0.18)",
          overflow: "hidden"
        }
      }}
    >
      <DialogTitle sx={{ px: 3, pt: 2.5, pb: 1 }}>
        <Box display="flex" justifyContent="space-between" alignItems="flex-start">
          <Box sx={{ width: "100%" }}>
            <Box
              sx={{
                width: 42,
                height: 42,
                borderRadius: "50%",
                bgcolor: "#202123",
                color: "#fff",
                display: "grid",
                placeItems: "center",
                mb: 2
              }}
            >
              <AutoAwesomeIcon sx={{ fontSize: 20 }} />
            </Box>

            <Typography
              sx={{
                fontSize: 30,
                fontWeight: 700,
                color: "#202123",
                lineHeight: 1.15,
                letterSpacing: "-0.02em"
              }}
            >
              Crea il tuo account
            </Typography>

            <Typography
              sx={{
                fontSize: 14,
                color: "#6e6e80",
                mt: 1,
                lineHeight: 1.55,
                maxWidth: 360
              }}
            >
              Registra il tuo profilo per salvare preferenze, istruzioni e risultati
              più pertinenti nel tempo.
            </Typography>
          </Box>

          <IconButton
            onClick={onClose}
            disabled={loading}
            size="small"
            sx={{
              ml: 1,
              color: "#6e6e80"
            }}
          >
            <CloseIcon fontSize="small" />
          </IconButton>
        </Box>
      </DialogTitle>

      <DialogContent sx={{ px: 3, pt: 1.5, pb: 3 }}>
        <Stack spacing={2}>
          {error && (
            <Alert
              severity="error"
              sx={{
                borderRadius: 3
              }}
            >
              {error}
            </Alert>
          )}

          <TextField
            label="Email"
            type="email"
            value={email}
            onChange={(e) => setEmail(e.target.value)}
            fullWidth
            autoFocus
            disabled={loading}
            autoComplete="email"
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: 3,
                bgcolor: "#fff"
              }
            }}
          />

          <TextField
            label="Password"
            type={showPassword ? "text" : "password"}
            value={password}
            onChange={(e) => setPassword(e.target.value)}
            fullWidth
            disabled={loading}
            autoComplete="new-password"
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: 3,
                bgcolor: "#fff"
              }
            }}
            InputProps={{
              endAdornment: (
                <InputAdornment position="end">
                  <IconButton
                    edge="end"
                    onClick={() => setShowPassword((prev) => !prev)}
                    disabled={loading}
                    sx={{ color: "#6e6e80" }}
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
              if (e.key === "Enter") handleRegister()
            }}
            sx={{
              "& .MuiOutlinedInput-root": {
                borderRadius: 3,
                bgcolor: "#fff"
              }
            }}
          />

          <Button
            fullWidth
            variant="contained"
            onClick={handleRegister}
            disabled={loading}
            sx={{
              mt: 0.5,
              textTransform: "none",
              borderRadius: 999,
              py: 1.35,
              fontWeight: 700,
              fontSize: 15,
              bgcolor: "#202123",
              color: "#fff",
              boxShadow: "none",
              "&:hover": {
                bgcolor: "#111214",
                boxShadow: "none"
              },
              "&:disabled": {
                bgcolor: alpha("#202123", 0.45),
                color: "#fff"
              }
            }}
          >
            {loading ? (
              <Stack direction="row" spacing={1} alignItems="center">
                <CircularProgress size={18} color="inherit" />
                <span>Creazione account...</span>
              </Stack>
            ) : (
              "Continua"
            )}
          </Button>

          <Divider sx={{ color: "#a1a1aa", fontSize: 12 }}>oppure</Divider>

          <Box
            sx={{
              borderRadius: 3,
              bgcolor: "#f7f7f8",
              border: "1px solid #ececf1",
              px: 2,
              py: 1.5
            }}
          >
            <Typography
              sx={{
                fontSize: 13,
                color: "#6e6e80",
                lineHeight: 1.55
              }}
            >
              Hai già un account?
            </Typography>

            <Button
              onClick={onLogin}
              disabled={loading}
              sx={{
                mt: 0.5,
                p: 0,
                minWidth: 0,
                textTransform: "none",
                fontWeight: 700,
                color: "#202123",
                "&:hover": {
                  bgcolor: "transparent",
                  textDecoration: "underline"
                }
              }}
            >
              Accedi
            </Button>
          </Box>

          <Typography
            sx={{
              fontSize: 11.5,
              color: "#8e8ea0",
              lineHeight: 1.5,
              textAlign: "center",
              mt: 0.5
            }}
          >
            Creando un account, abiliti un’esperienza più personalizzata nella piattaforma.
          </Typography>
        </Stack>
      </DialogContent>
    </Dialog>
  )
}
import {
  Avatar,
  Box,
  Button,
  Chip,
  Divider,
  Stack,
  Typography
} from "@mui/material"
import LogoutIcon from "@mui/icons-material/Logout"
import TuneIcon from "@mui/icons-material/Tune"
import { useMemo, useState } from "react"

import LoginDialog from "./LoginDialog"
import RegisterDialog from "./RegisterDialog"
import { CustomInstructionsModal } from "./CustomInstructionsModal"
import { useAuth } from "../useAuth"

function getInitials(email?: string) {
  if (!email) return "U"
  return email.slice(0, 2).toUpperCase()
}

export default function AuthPanel() {
  const { user, loggedIn, logout, loadingUser } = useAuth()

  const [loginOpen, setLoginOpen] = useState(false)
  const [registerOpen, setRegisterOpen] = useState(false)
  const [instructionsOpen, setInstructionsOpen] = useState(false)

  const initials = useMemo(() => getInitials(user?.email), [user?.email])

  if (loggedIn && user) {
    return (
      <>
        <Box
          sx={{
            p: 2.5,
            borderRadius: "14px",
            border: "1px solid #e5e5e5",
            bgcolor: "#fff",
            boxShadow: "0 2px 12px rgba(0,0,0,0.06)"
          }}
        >
          {/* User row */}
          <Stack direction="row" spacing={1.5} alignItems="center" mb={2}>
            <Avatar
              sx={{
                width: 38,
                height: 38,
                bgcolor: "#202123",
                color: "#fff",
                fontWeight: 700,
                fontSize: 13,
                borderRadius: "10px"
              }}
            >
              {initials}
            </Avatar>

            <Box sx={{ minWidth: 0 }}>
              <Typography sx={{ fontSize: 12, color: "#999", lineHeight: 1.2 }}>
                Account attivo
              </Typography>
              <Typography
                sx={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: "#202123",
                  lineHeight: 1.35,
                  overflow: "hidden",
                  textOverflow: "ellipsis",
                  whiteSpace: "nowrap"
                }}
              >
                {user.email}
              </Typography>
            </Box>
          </Stack>

          {/* Chips */}
          {(user.favorite_brands || user.price_preference) && (
            <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap" mb={2}>
              {user.favorite_brands && (
                <Chip
                  size="small"
                  label={`Brand: ${user.favorite_brands}`}
                  sx={{
                    height: 24,
                    borderRadius: "6px",
                    bgcolor: "#f4f4f5",
                    color: "#555",
                    fontSize: 11,
                    fontWeight: 500
                  }}
                />
              )}
              {user.price_preference && (
                <Chip
                  size="small"
                  label={`Budget: ${user.price_preference}`}
                  sx={{
                    height: 24,
                    borderRadius: "6px",
                    bgcolor: "#f4f4f5",
                    color: "#555",
                    fontSize: 11,
                    fontWeight: 500
                  }}
                />
              )}
            </Stack>
          )}

          <Divider sx={{ mb: 2, borderColor: "#f0f0f0" }} />

          <Stack spacing={0.75}>
            <Button
              fullWidth
              variant="text"
              startIcon={<TuneIcon sx={{ fontSize: 16 }} />}
              onClick={() => setInstructionsOpen(true)}
              sx={{
                textTransform: "none",
                borderRadius: "8px",
                py: 0.9,
                px: 1.5,
                fontWeight: 500,
                fontSize: 13,
                color: "#202123",
                justifyContent: "flex-start",
                "&:hover": { bgcolor: "#f5f5f5" }
              }}
            >
              Istruzioni personalizzate
            </Button>

            <Button
              fullWidth
              variant="text"
              startIcon={<LogoutIcon sx={{ fontSize: 16 }} />}
              onClick={logout}
              sx={{
                textTransform: "none",
                borderRadius: "8px",
                py: 0.9,
                px: 1.5,
                fontWeight: 500,
                fontSize: 13,
                color: "#888",
                justifyContent: "flex-start",
                "&:hover": { bgcolor: "#f5f5f5", color: "#202123" }
              }}
            >
              Esci
            </Button>
          </Stack>
        </Box>

        <CustomInstructionsModal
          open={instructionsOpen}
          onClose={() => setInstructionsOpen(false)}
        />
      </>
    )
  }

  return (
    <>
      <Box
        sx={{
          p: 2.5,
          borderRadius: "14px",
          border: "1px solid #e5e5e5",
          bgcolor: "#fff",
          boxShadow: "0 2px 12px rgba(0,0,0,0.06)"
        }}
      >
        <Typography
          sx={{
            fontSize: 15,
            fontWeight: 700,
            color: "#0d0d0d",
            letterSpacing: "-0.01em",
            mb: 0.5
          }}
        >
          Accedi a ebayGPT
        </Typography>

        <Typography
          sx={{
            fontSize: 13,
            color: "#6e6e80",
            lineHeight: 1.55,
            mb: 2
          }}
        >
          Salva preferenze e istruzioni per risultati più pertinenti.
        </Typography>

        <Stack spacing={1}>
          <Button
            fullWidth
            variant="contained"
            onClick={() => setLoginOpen(true)}
            disabled={loadingUser}
            sx={{
              textTransform: "none",
              borderRadius: "10px",
              py: 1.1,
              fontWeight: 600,
              fontSize: 13,
              bgcolor: "#202123",
              boxShadow: "none",
              "&:hover": { bgcolor: "#111214", boxShadow: "none" },
              "&:disabled": { bgcolor: "rgba(32,33,35,0.35)" }
            }}
          >
            Accedi
          </Button>

          <Button
            fullWidth
            variant="outlined"
            onClick={() => setRegisterOpen(true)}
            disabled={loadingUser}
            sx={{
              textTransform: "none",
              borderRadius: "10px",
              py: 1,
              fontWeight: 600,
              fontSize: 13,
              color: "#202123",
              borderColor: "#d9d9e3",
              boxShadow: "none",
              "&:hover": { borderColor: "#b0b0bc", bgcolor: "#fafafa", boxShadow: "none" }
            }}
          >
            Registrati
          </Button>
        </Stack>

        <Typography sx={{ mt: 1.5, fontSize: 11, color: "#b0b0bc", lineHeight: 1.45 }}>
          Puoi usare la ricerca anche senza account.
        </Typography>
      </Box>

      <LoginDialog
        open={loginOpen}
        onClose={() => setLoginOpen(false)}
        onRegister={() => {
          setLoginOpen(false)
          setRegisterOpen(true)
        }}
      />

      <RegisterDialog
        open={registerOpen}
        onClose={() => setRegisterOpen(false)}
        onLogin={() => {
          setRegisterOpen(false)
          setLoginOpen(true)
        }}
      />
    </>
  )
}
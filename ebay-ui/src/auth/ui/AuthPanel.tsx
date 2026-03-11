import { Box, Typography, Button, Chip, Stack, Divider } from "@mui/material"
import { useState } from "react"

import LoginDialog from "./LoginDialog"
import RegisterDialog from "./RegisterDialog"
import { CustomInstructionsModal } from "./CustomInstructionsModal"
import { useAuth } from "../useAuth"

export default function AuthPanel() {
  const { user, loggedIn, logout, loadingUser } = useAuth()

  const [loginOpen, setLoginOpen] = useState(false)
  const [registerOpen, setRegisterOpen] = useState(false)
  const [instructionsOpen, setInstructionsOpen] = useState(false)

  if (loggedIn && user) {
    return (
      <Box
        sx={{
          p: 1.5,
          borderRadius: 3,
          border: "1px solid #e5e5e5",
          bgcolor: "#ffffff"
        }}
      >
        <Typography
          sx={{
            fontSize: 12,
            fontWeight: 700,
            letterSpacing: 0.3,
            textTransform: "uppercase",
            color: "#8e8ea0",
            mb: 0.75
          }}
        >
          Account
        </Typography>

        <Typography
          sx={{
            fontSize: 14,
            fontWeight: 600,
            color: "#202123",
            lineHeight: 1.35,
            wordBreak: "break-word"
          }}
        >
          {user.email}
        </Typography>

        <Typography
          sx={{
            fontSize: 12,
            color: "#6e6e80",
            mt: 0.5,
            mb: 1.25,
            lineHeight: 1.45
          }}
        >
          Sessione attiva. La ricerca può usare preferenze e cronologia locale.
        </Typography>

        <Stack direction="row" spacing={1} useFlexGap flexWrap="wrap" mb={1.25}>
          {user.favorite_brands && (
            <Chip
              size="small"
              label={`Brand: ${user.favorite_brands}`}
              sx={{
                height: 24,
                bgcolor: "#f3f4f6",
                fontSize: 11
              }}
            />
          )}

          {user.price_preference && (
            <Chip
              size="small"
              label={`Budget: ${user.price_preference}`}
              sx={{
                height: 24,
                bgcolor: "#f3f4f6",
                fontSize: 11
              }}
            />
          )}
        </Stack>

        <Divider sx={{ my: 1.25 }} />

        <Button
          fullWidth
          variant="outlined"
          size="small"
          onClick={() => setInstructionsOpen(true)}
          sx={{
            mb: 1.25,
            textTransform: "none",
            borderRadius: 2,
            fontWeight: 600,
            color: "#1976d2",
            borderColor: "rgba(25, 118, 210, 0.5)",
            "&:hover": { borderColor: "#1976d2" }
          }}
        >
          Impostazioni Istruzioni (Gems)
        </Button>

        <Button
          fullWidth
          variant="text"
          onClick={logout}
          sx={{
            textTransform: "none",
            borderRadius: 2,
            fontWeight: 600,
            color: "#202123",
            justifyContent: "flex-start",
            px: 1
          }}
        >
          Esci
        </Button>

        <CustomInstructionsModal
          open={instructionsOpen}
          onClose={() => setInstructionsOpen(false)}
        />
      </Box>
    )
  }

  return (
    <Box
      sx={{
        p: 1.5,
        borderRadius: 3,
        border: "1px solid #e5e5e5",
        bgcolor: "#ffffff"
      }}
    >
      <Typography
        sx={{
          fontSize: 12,
          fontWeight: 700,
          letterSpacing: 0.3,
          textTransform: "uppercase",
          color: "#8e8ea0",
          mb: 0.75
        }}
      >
        Account
      </Typography>

      <Typography
        sx={{
          fontSize: 15,
          fontWeight: 700,
          color: "#202123",
          lineHeight: 1.35,
          mb: 0.75
        }}
      >
        Accedi per personalizzare la ricerca
      </Typography>

      <Typography
        sx={{
          fontSize: 13,
          color: "#6e6e80",
          lineHeight: 1.5,
          mb: 1.5
        }}
      >
        Salva preferenze, usa il profilo utente e rendi i risultati più coerenti.
      </Typography>

      <Stack spacing={1}>
        <Button
          fullWidth
          variant="contained"
          onClick={() => setLoginOpen(true)}
          disabled={loadingUser}
          sx={{
            textTransform: "none",
            borderRadius: 2.5,
            py: 1.1,
            fontWeight: 600,
            bgcolor: "#202123",
            boxShadow: "none",
            "&:hover": {
              bgcolor: "#111214",
              boxShadow: "none"
            }
          }}
        >
          Accedi
        </Button>

        <Button
          fullWidth
          variant="text"
          onClick={() => setRegisterOpen(true)}
          disabled={loadingUser}
          sx={{
            textTransform: "none",
            borderRadius: 2.5,
            py: 1,
            fontWeight: 600,
            color: "#202123"
          }}
        >
          Crea account
        </Button>
      </Stack>

      <Typography
        sx={{
          fontSize: 11,
          color: "#8e8ea0",
          mt: 1.25,
          lineHeight: 1.4
        }}
      >
        Puoi usare la ricerca anche senza account.
      </Typography>

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
    </Box>
  )
}
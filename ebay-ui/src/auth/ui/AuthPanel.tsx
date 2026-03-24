import {
  Avatar,
  Box,
  Button,
  Chip,
  Divider,
  Stack,
  Typography,
  Tooltip
} from "@mui/material"
import LogoutIcon from "@mui/icons-material/Logout"
import TuneIcon from "@mui/icons-material/Tune"
import { useMemo, useState } from "react"

import LoginDialog from "./LoginDialog"
import RegisterDialog from "./RegisterDialog"
import { CustomInstructionsModal } from "./CustomInstructionsModal"
import { useAuth } from "../useAuth"
import { useSidebarStore, type SidebarState } from "../../features/chat/store/sidebarStore"

function getInitials(email?: string) {
  if (!email) return "U"
  const parts = email.split(/[\._@]/)
  if (parts.length >= 2 && parts[0].length > 0 && parts[1].length > 1) {
    return (parts[0][0] + parts[1][0]).toUpperCase()
  }
  return email.slice(0, 2).toUpperCase()
}

export default function AuthPanel() {
  const { user, loggedIn, logout } = useAuth()
  const isCollapsed = useSidebarStore((s: SidebarState) => s.isCollapsed)

  const [loginOpen, setLoginOpen] = useState(false)
  const [registerOpen, setRegisterOpen] = useState(false)
  const [instructionsOpen, setInstructionsOpen] = useState(false)

  const initials = useMemo(() => getInitials(user?.email), [user?.email])

  if (loggedIn && user) {
    if (isCollapsed) {
      return (
        <Box sx={{ display: 'flex', justifyContent: 'center', py: 2 }}>
          <Tooltip title={user.email} placement="right">
            <Avatar
              sx={{
                width: 36,
                height: 36,
                bgcolor: "var(--accent-primary)",
                color: "#fff",
                fontWeight: 700,
                fontSize: 12,
                borderRadius: "10px",
                cursor: 'pointer',
                boxShadow: "0 2px 8px rgba(0,0,0,0.2)"
              }}
              onClick={() => setInstructionsOpen(true)}
            >
              {initials}
            </Avatar>
          </Tooltip>
          <CustomInstructionsModal
            open={instructionsOpen}
            onClose={() => setInstructionsOpen(false)}
          />
        </Box>
      )
    }

    return (
      <>
        <Box
          sx={{
            m: 1.5,
            p: 2,
            borderRadius: "16px",
            border: "1px solid var(--border-color)",
            bgcolor: "var(--bg-primary)",
            boxShadow: "0 4px 12px -2px rgba(0,0,0,0.12)",
            transition: 'all 0.2s ease-in-out'
          }}
        >
          {/* User row */}
          <Stack direction="row" spacing={1.5} alignItems="center" mb={1.5}>
            <Avatar
              sx={{
                width: 34,
                height: 34,
                bgcolor: "var(--accent-primary)",
                color: "#fff",
                fontWeight: 700,
                fontSize: 12,
                borderRadius: "10px"
              }}
            >
              {initials}
            </Avatar>

            <Box sx={{ minWidth: 0 }}>
              <Typography sx={{ fontSize: 11, color: "var(--text-secondary)", fontWeight: 500, lineHeight: 1.2 }}>
                Account attivo
              </Typography>
              <Typography
                sx={{
                  fontSize: 13,
                  fontWeight: 600,
                  color: "var(--text-primary)",
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
            <Stack direction="row" spacing={0.75} useFlexGap flexWrap="wrap" mb={1.5}>
              {user.favorite_brands && (
                <Chip
                  size="small"
                  label={user.favorite_brands}
                  sx={{
                    height: 20,
                    borderRadius: "6px",
                    bgcolor: "var(--bg-secondary)",
                    color: "var(--text-secondary)",
                    fontSize: 10,
                    fontWeight: 600
                  }}
                />
              )}
              {user.price_preference && (
                <Chip
                  size="small"
                  label={`€${user.price_preference}`}
                  sx={{
                    height: 20,
                    borderRadius: "6px",
                    bgcolor: "var(--bg-secondary)",
                    color: "var(--text-secondary)",
                    fontSize: 10,
                    fontWeight: 600
                  }}
                />
              )}
            </Stack>
          )}

          <Divider sx={{ mb: 1.5, borderColor: "var(--border-color)" }} />

          <Stack spacing={0.5}>
            <Button
              fullWidth
              variant="text"
              startIcon={<TuneIcon sx={{ fontSize: 16 }} />}
              onClick={() => setInstructionsOpen(true)}
              sx={{
                textTransform: "none",
                borderRadius: "8px",
                py: 0.6,
                px: 1,
                fontWeight: 500,
                fontSize: 12,
                color: "var(--text-primary)",
                justifyContent: "flex-start",
                "&:hover": { bgcolor: "var(--bg-secondary)" }
              }}
            >
              Istruzioni
            </Button>

            <Button
              fullWidth
              variant="text"
              startIcon={<LogoutIcon sx={{ fontSize: 16 }} />}
              onClick={logout}
              sx={{
                textTransform: "none",
                borderRadius: "8px",
                py: 0.6,
                px: 1,
                fontWeight: 500,
                fontSize: 12,
                color: "var(--text-secondary)",
                justifyContent: "flex-start",
                "&:hover": { bgcolor: "rgba(239, 68, 68, 0.08)", color: "#ef4444" }
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
          m: isCollapsed ? 1 : 1.5,
          p: isCollapsed ? 1 : 2,
          borderRadius: "16px",
          border: "1px solid var(--border-color)",
          bgcolor: "var(--bg-primary)",
          boxShadow: "0 4px 12px -2px rgba(0,0,0,0.12)",
          display: 'flex',
          flexDirection: 'column',
          alignItems: isCollapsed ? 'center' : 'stretch',
          transition: 'all 0.2s ease-in-out'
        }}
      >
        {!isCollapsed && (
          <>
            <Typography sx={{ fontSize: 14, fontWeight: 700, color: "var(--text-primary)", mb: 0.5 }}>
              Accedi a ebayGPT
            </Typography>
            <Typography sx={{ fontSize: 12, color: "var(--text-secondary)", lineHeight: 1.4, mb: 2 }}>
              Salva preferenze e istruzioni per risultati migliori.
            </Typography>
          </>
        )}

        <Stack spacing={1} sx={{ width: '100%' }}>
          <Tooltip title={isCollapsed ? "Accedi" : ""} placement="right">
            <Button
              fullWidth
              variant="contained"
              onClick={() => setLoginOpen(true)}

              sx={{
                textTransform: "none",
                borderRadius: "10px",
                py: isCollapsed ? 1 : 1,
                minWidth: isCollapsed ? 40 : 'auto',
                fontWeight: 600,
                fontSize: 13,
                bgcolor: "var(--accent-primary)",
                color: "var(--bg-primary)",
                "&:hover": { bgcolor: "var(--accent-primary)", opacity: 0.9 }
              }}
            >
              {isCollapsed ? initials : "Accedi"}
            </Button>
          </Tooltip>

          {!isCollapsed && (
            <Button
              fullWidth
              variant="outlined"
              onClick={() => setRegisterOpen(true)}

              sx={{
                textTransform: "none",
                borderRadius: "10px",
                py: 1,
                fontWeight: 600,
                fontSize: 13,
                color: "var(--text-primary)",
                borderColor: "var(--border-color)",
                "&:hover": { bgcolor: "var(--bg-secondary)", borderColor: "var(--text-primary)" }
              }}
            >
              Registrati
            </Button>
          )}
        </Stack>
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
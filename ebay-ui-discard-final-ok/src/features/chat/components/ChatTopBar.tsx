import {
  Box,
  IconButton,
  Typography,
  Tooltip,
  Divider,
  Switch
} from "@mui/material"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import MenuIcon from "@mui/icons-material/Menu"
import ShareIcon from "@mui/icons-material/Share"
import ModeEditIcon from "@mui/icons-material/ModeEdit"
import DarkModeIcon from "@mui/icons-material/DarkMode"
import LightModeIcon from "@mui/icons-material/LightMode"

interface ChatTopBarProps {
  isMobile: boolean
  isDarkMode: boolean
  activeSessionTitle: string | undefined
  onMenuClick: () => void
  onThemeToggle: (mode: 'light' | 'dark') => void
  onShare: () => void
  onEditTitle: () => void
}

export function ChatTopBar({
  isMobile,
  isDarkMode,
  activeSessionTitle,
  onMenuClick,
  onThemeToggle,
  onShare,
  onEditTitle
}: ChatTopBarProps) {
  return (
    <Box
      sx={{
        height: 56,
        borderBottom: '1px solid var(--border-color)',
        opacity: 0.95,
        display: 'flex',
        alignItems: 'center',
        px: 2,
        bgcolor: 'var(--bg-primary)',
        transition: 'background-color 0.2s',
        position: 'relative'
      }}
    >
      {/* Left — branding & Title */}
      <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
        {isMobile && (
          <IconButton
            onClick={onMenuClick}
            size="small"
            sx={{ color: "var(--text-secondary)" }}
          >
            <MenuIcon sx={{ fontSize: 20 }} />
          </IconButton>
        )}

        <Box display="flex" alignItems="center" gap={1}>
          <Box
            sx={{
              bgcolor: '#111827',
              color: 'white',
              p: 0.5,
              borderRadius: 1,
              display: 'flex',
              alignItems: 'center'
            }}
          >
            <AutoAwesomeIcon sx={{ fontSize: 16 }} />
          </Box>
          <Typography fontWeight={700} color="var(--text-primary)" sx={{ display: { xs: 'none', sm: 'block' } }}>
            ebayGPT
          </Typography>
        </Box>

        <Divider orientation="vertical" flexItem sx={{ height: 20, alignSelf: 'center', mx: 1, opacity: 0.1, borderColor: 'var(--text-secondary)' }} />

        <Typography
          fontSize={14}
          fontWeight={600}
          color="var(--text-primary)"
          noWrap
          sx={{
            position: 'absolute',
            left: '50%',
            transform: 'translateX(-50%)',
            maxWidth: { xs: 120, sm: 300 },
            textAlign: 'center',
            color: (activeSessionTitle === "Nuova chat" || !activeSessionTitle) ? "var(--text-secondary)" : "var(--text-primary)"
          }}
        >
          {(activeSessionTitle === "Nuova chat" || !activeSessionTitle) ? "Cerca" : activeSessionTitle}
        </Typography>
      </Box>

      {/* Right — actions */}
      <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center', gap: 1.5, mr: 1 }}>
        <Box sx={{ display: 'flex', alignItems: 'center', gap: 0.5 }}>
          <LightModeIcon sx={{ fontSize: 16, color: !isDarkMode ? "var(--text-primary)" : "var(--text-secondary)" }} />
          <Switch
            size="small"
            checked={isDarkMode}
            onChange={(e) => onThemeToggle(e.target.checked ? 'dark' : 'light')}
            color="default"
          />
          <DarkModeIcon sx={{ fontSize: 16, color: isDarkMode ? "var(--text-primary)" : "var(--text-secondary)" }} />
        </Box>

        <Tooltip title="Condividi">
          <IconButton size="small" sx={{ color: "var(--text-secondary)" }} onClick={onShare}>
            <ShareIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>

        <Tooltip title="Modifica">
          <IconButton size="small" sx={{ color: "var(--text-secondary)" }} onClick={onEditTitle}>
            <ModeEditIcon sx={{ fontSize: 18 }} />
          </IconButton>
        </Tooltip>
      </Box>
    </Box>
  )
}

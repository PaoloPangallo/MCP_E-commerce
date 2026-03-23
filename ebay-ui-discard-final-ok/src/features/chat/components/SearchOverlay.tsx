import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  IconButton,
  InputBase
} from "@mui/material"
import SearchIcon from "@mui/icons-material/Search"
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline"
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft"

interface SearchOverlayProps {
  searchQuery: string
  filteredSessions: any[]
  onSearchChange: (query: string) => void
  onClose: () => void
  onSwitchSession: (id: string) => void
}

export function SearchOverlay({
  searchQuery,
  filteredSessions,
  onSearchChange,
  onClose,
  onSwitchSession
}: SearchOverlayProps) {
  return (
    <Box
      sx={{
        position: 'fixed',
        inset: 0,
        zIndex: 9999,
        bgcolor: 'var(--bg-primary)',
        backdropFilter: 'blur(10px)',
        display: 'flex',
        flexDirection: 'column',
        alignItems: 'center',
        pt: { xs: 8, md: 12 },
        px: 2
      }}
      onClick={onClose}
    >
      <Box
        sx={{
          width: '100%',
          maxWidth: 680
        }}
        onClick={(e) => e.stopPropagation()}
      >
        <Typography variant="h4" fontWeight={700} sx={{ mb: 4, textAlign: 'left', color: "var(--text-primary)" }}>
          Cerca
        </Typography>

        <Box
          sx={{
            display: 'flex',
            alignItems: 'center',
            bgcolor: 'var(--bg-secondary)',
            border: '1px solid var(--border-color)',
            borderRadius: '16px',
            px: 2,
            py: 1.5,
            boxShadow: '0 4px 12px rgba(0,0,0,0.05)',
            mb: 6
          }}
        >
          <SearchIcon sx={{ color: 'var(--text-secondary)', mr: 2, fontSize: 24 }} />
          <InputBase
            autoFocus
            fullWidth
            placeholder="Cerca le chat..."
            value={searchQuery}
            onChange={(e) => onSearchChange(e.target.value)}
            sx={{ fontSize: 18, color: 'var(--text-primary)' }}
            onKeyDown={(e) => {
              if (e.key === 'Escape') onClose()
            }}
          />
        </Box>

        <Box sx={{ width: '100%' }}>
          <Typography
            sx={{
              fontSize: 12,
              fontWeight: 600,
              color: "#9ca3af",
              textTransform: "uppercase",
              letterSpacing: 0.7,
              mb: 2
            }}
          >
            Recenti
          </Typography>

          <List sx={{ width: '100%' }}>
            {filteredSessions.map((session) => (
              <ListItem disablePadding key={session.id} sx={{ mb: 1 }}>
                <ListItemButton
                  onClick={() => onSwitchSession(session.id)}
                  sx={{
                    borderRadius: '12px',
                    py: 2,
                    "&:hover": { bgcolor: 'var(--bg-secondary)' }
                  }}
                >
                  <ChatBubbleOutlineIcon sx={{ mr: 2, color: 'var(--text-secondary)' }} />
                  <ListItemText
                    primary={session.title}
                    primaryTypographyProps={{
                      fontSize: 16,
                      fontWeight: 500,
                      color: 'var(--text-primary)'
                    }}
                  />
                  <Typography fontSize={13} color="#9ca3af">
                    Oggi
                  </Typography>
                </ListItemButton>
              </ListItem>
            ))}
          </List>
        </Box>
      </Box>

      <IconButton
        onClick={onClose}
        sx={{
          position: 'absolute',
          top: 20,
          left: 20
        }}
      >
        <ChevronLeftIcon sx={{ fontSize: 32, color: 'var(--text-secondary)' }} />
      </IconButton>
    </Box>
  )
}

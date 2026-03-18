import {
  Box,
  Button,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Typography,
  useMediaQuery,
  Tooltip
} from "@mui/material"
import { useTheme } from "@mui/material/styles"

import AddIcon from "@mui/icons-material/Add"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline"
import MenuIcon from "@mui/icons-material/Menu"
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline"
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep"

import { useChatStore } from "./store/chatStore"
import { useSidebarStore } from "./store/sidebarStore"
import AuthPanel from "../../auth/ui/AuthPanel"

interface Props {
  children: React.ReactNode
  composer?: React.ReactNode
  onNewChat?: () => void
  sidebarTopSlot?: React.ReactNode
}

const SIDEBAR_WIDTH = 260

function SidebarSectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <Typography
      sx={{
        fontSize: 10,
        fontWeight: 600,
        color: "#9ca3af",
        textTransform: "uppercase",
        letterSpacing: 0.7,
        px: 2,
        pb: 0.75
      }}
    >
      {children}
    </Typography>
  )
}

function SessionItem({
  title,
  active,
  onClick,
  onDelete
}: {
  title: string
  active?: boolean
  onClick: () => void
  onDelete: () => void
}) {
  return (
    <ListItem disablePadding sx={{ px: 1, mb: 0.25 }}>
      <ListItemButton
        onClick={onClick}
        sx={{
          borderRadius: 2,
          py: 0.875,
          px: 1.25,
          bgcolor: active ? "#f3f4f6" : "transparent",
          "&:hover": { bgcolor: "#f3f4f6" },
          minHeight: 36
        }}
      >
        <ChatBubbleOutlineIcon
          sx={{
            fontSize: 14,
            color: active ? "#374151" : "#9ca3af",
            mr: 1.25,
            flexShrink: 0
          }}
        />

        <ListItemText
          primary={title}
          primaryTypographyProps={{
            fontSize: 13,
            fontWeight: active ? 500 : 400,
            color: active ? "#111827" : "#6b7280",
            noWrap: true
          }}
        />

        <Tooltip title="Elimina">
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation()
              onDelete()
            }}
            sx={{
              opacity: 0,
              transition: "opacity 0.15s",
              ".MuiListItemButton-root:hover &": { opacity: 1 },
              color: "#9ca3af",
              p: 0.4,
              "&:hover": { color: "#ef4444", bgcolor: "transparent" }
            }}
          >
            <DeleteOutlineIcon sx={{ fontSize: 14 }} />
          </IconButton>
        </Tooltip>
      </ListItemButton>
    </ListItem>
  )
}

export default function ChatLayout({
  children,
  composer,
  onNewChat,
  sidebarTopSlot
}: Props) {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down("md"))

  const sessions = useChatStore((s) => s.sessions)
  const activeSessionId = useChatStore((s) => s.activeSessionId)
  const createSession = useChatStore((s) => s.createSession)
  const switchSession = useChatStore((s) => s.switchSession)
  const deleteSession = useChatStore((s) => s.deleteSession)
  const clearMemory = useChatStore((s) => s.clearMemory)

  const mobileOpen = useSidebarStore((s) => s.mobileOpen)
  const setMobileOpen = useSidebarStore((s) => s.setMobileOpen)

  const activeSession = sessions.find(
    (s) => s.id === (activeSessionId || sessions[0]?.id)
  )

  const handleNewChat = () => {
    createSession()
    onNewChat?.()
    if (isMobile) setMobileOpen(false)
  }

  const handleClearMemory = () => {
    if (
      confirm(
        "Vuoi svuotare la memoria dell'agente? La cronologia e le ricerche verranno cancellate."
      )
    ) {
      clearMemory()
      if (isMobile) setMobileOpen(false)
    }
  }

  const sidebarContent = (
    <Box
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: "#fafafa"
      }}
    >
      {/* HEADER */}
      <Box sx={{ px: 2, pt: 2, pb: 1.5 }}>
        <Box display="flex" alignItems="center" gap={1}>
          <Box
            sx={{
              width: 28,
              height: 28,
              borderRadius: "8px",
              bgcolor: "#111827",
              display: "flex",
              alignItems: "center",
              justifyContent: "center"
            }}
          >
            <AutoAwesomeIcon sx={{ fontSize: 15, color: "#fff" }} />
          </Box>
          <Box>
            <Typography fontSize={13} fontWeight={600} color="#111827">
              ebayGPT
            </Typography>
            <Typography fontSize={11} color="#9ca3af" lineHeight={1.2}>
              shopping assistant
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* NEW CHAT */}
      <Box sx={{ px: 1.5, pb: 1.5 }}>
        <Button
          fullWidth
          variant="outlined"
          startIcon={<AddIcon sx={{ fontSize: 16 }} />}
          onClick={handleNewChat}
          sx={{
            justifyContent: "flex-start",
            textTransform: "none",
            borderRadius: 2,
            fontSize: 13,
            fontWeight: 500,
            color: "#374151",
            borderColor: "#e5e7eb",
            bgcolor: "#fff",
            boxShadow: "none",
            py: 0.875,
            "&:hover": {
              bgcolor: "#f9fafb",
              borderColor: "#d1d5db",
              boxShadow: "none"
            }
          }}
        >
          Nuova chat
        </Button>
      </Box>

      {/* AUTH PANEL */}
      <Box px={1.5} pb={1.5}>
        {sidebarTopSlot ?? <AuthPanel />}
      </Box>

      {/* HISTORY */}
      <Box sx={{ flex: 1, overflowY: "auto", pb: 1 }}>
        {sessions.length > 0 && (
          <>
            <SidebarSectionTitle>Recenti</SidebarSectionTitle>
            <List dense disablePadding>
              {sessions.map((session) => (
                <SessionItem
                  key={session.id}
                  title={session.title}
                  active={(activeSessionId || sessions[0]?.id) === session.id}
                  onClick={() => {
                    switchSession(session.id)
                    if (isMobile) setMobileOpen(false)
                  }}
                  onDelete={() => deleteSession(session.id)}
                />
              ))}
            </List>
          </>
        )}
      </Box>

      {/* FOOTER */}
      <Box
        sx={{ p: 1.5, borderTop: "1px solid #f0f0f0" }}
      >
        <Button
          fullWidth
          variant="text"
          startIcon={<DeleteSweepIcon sx={{ fontSize: 15 }} />}
          onClick={handleClearMemory}
          sx={{
            justifyContent: "flex-start",
            textTransform: "none",
            borderRadius: 2,
            fontSize: 12,
            fontWeight: 500,
            px: 1.25,
            color: "#9ca3af",
            "&:hover": { bgcolor: "#fff1f1", color: "#ef4444" }
          }}
        >
          Svuota memoria server
        </Button>
      </Box>
    </Box>
  )

  return (
    <Box sx={{ display: "flex", minHeight: "100vh" }}>
      {isMobile ? (
        <Drawer
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          variant="temporary"
          ModalProps={{ keepMounted: true }}
          sx={{ "& .MuiDrawer-paper": { width: SIDEBAR_WIDTH } }}
        >
          {sidebarContent}
        </Drawer>
      ) : (
        <Drawer
          variant="permanent"
          sx={{
            width: SIDEBAR_WIDTH,
            flexShrink: 0,
            "& .MuiDrawer-paper": {
              width: SIDEBAR_WIDTH,
              boxSizing: "border-box",
              borderRight: "1px solid #f0f0f0"
            }
          }}
        >
          {sidebarContent}
        </Drawer>
      )}

      {/* MAIN AREA */}
      <Box
        sx={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
          bgcolor: "#ffffff"
        }}
      >
        {/* TOP BAR */}
        <Box
          sx={{
            height: 48,
            px: 2,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            borderBottom: "1px solid #f5f5f5",
            bgcolor: "#fff",
            position: "sticky",
            top: 0,
            zIndex: 10
          }}
        >
          {/* Left — mobile hamburger */}
          <Box sx={{ width: 80 }}>
            {isMobile && (
              <IconButton
                onClick={() => setMobileOpen(true)}
                size="small"
                sx={{ color: "#6b7280" }}
              >
                <MenuIcon sx={{ fontSize: 20 }} />
              </IconButton>
            )}
          </Box>

          {/* Center — session title */}
          <Typography
            fontSize={13}
            fontWeight={500}
            color="#6b7280"
            noWrap
            sx={{ flex: 1, textAlign: "center" }}
          >
            {activeSession?.title || "ebayGPT"}
          </Typography>

          {/* Right — placeholder for balance */}
          <Box sx={{ width: 80 }} />
        </Box>

        {/* CHAT AREA */}
        <Box
          id="chat-scroll-container"
          sx={{
            flex: 1,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column"
          }}
        >
          <Box
            sx={{
              width: "100%",
              maxWidth: 760,
              mx: "auto",
              flex: 1,
              px: { xs: 2, md: 4 }
            }}
          >
            {children}
          </Box>
        </Box>

        {/* COMPOSER */}
        {composer && (
          <Box
            sx={{
              borderTop: "1px solid #f5f5f5",
              px: { xs: 1.5, md: 3 },
              pt: 1,
              pb: { xs: 1.5, md: 2 },
              bgcolor: "#fff"
            }}
          >
            <Box sx={{ maxWidth: 720, mx: "auto" }}>
              {composer}
            </Box>
          </Box>
        )}
      </Box>
    </Box>
  )
}
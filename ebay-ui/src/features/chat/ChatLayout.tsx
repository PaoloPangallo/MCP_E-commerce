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

import { useChatStore } from "./store/chatStore"
import { useSidebarStore } from "./store/sidebarStore"
import AuthPanel from "../../auth/ui/AuthPanel"

interface Props {
  children: React.ReactNode
  composer?: React.ReactNode
  onNewChat?: () => void
  sidebarTopSlot?: React.ReactNode
}

const SIDEBAR_WIDTH = 280

function SidebarSectionTitle({ children }: { children: React.ReactNode }) {
  return (
    <Typography
      sx={{
        fontSize: 11,
        fontWeight: 700,
        color: "#6b7280",
        textTransform: "uppercase",
        letterSpacing: 0.5,
        px: 2,
        pb: 1
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
    <ListItem disablePadding sx={{ px: 1, mb: 0.5 }}>
      <ListItemButton
        onClick={onClick}
        sx={{
          borderRadius: 3,
          py: 1,
          px: 1.25,
          bgcolor: active ? "#eceff3" : "transparent",
          "&:hover": { bgcolor: "#eceff3" }
        }}
      >
        <ChatBubbleOutlineIcon sx={{ fontSize: 16, color: active ? "#111827" : "#6b7280", mr: 1.25 }} />

        <ListItemText
          primary={title}
          primaryTypographyProps={{
            fontSize: 13,
            fontWeight: active ? 600 : 500,
            color: active ? "#111827" : "#4b5563",
            noWrap: true
          }}
        />

        <Tooltip title="Elimina chat">
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation()
              onDelete()
            }}
            sx={{
              opacity: active ? 1 : 0,
              transition: "opacity 0.2s",
              ".MuiListItemButton-root:hover &": { opacity: 1 },
              color: "#9ca3af",
              "&:hover": { color: "#ef4444" }
            }}
          >
            <DeleteOutlineIcon sx={{ fontSize: 16 }} />
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

  const mobileOpen = useSidebarStore((s) => s.mobileOpen)
  const setMobileOpen = useSidebarStore((s) => s.setMobileOpen)

  const handleNewChat = () => {
    createSession()
    onNewChat?.()
  }

  const sidebarContent = (
    <Box
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: "#f7f8fb"
      }}
    >

      {/* HEADER */}
      <Box sx={{ px: 2, pt: 2 }}>
        <Box display="flex" alignItems="center" gap={1.25}>
          <Box
            sx={{
              width: 30,
              height: 30,
              borderRadius: 2,
              bgcolor: "#111827",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff"
            }}
          >
            <AutoAwesomeIcon sx={{ fontSize: 17 }} />
          </Box>

          <Box>
            <Typography fontSize={14} fontWeight={700}>
              ebayGPT
            </Typography>

            <Typography fontSize={12} color="#6b7280">
              chat search + seller trust
            </Typography>
          </Box>
        </Box>
      </Box>

      {/* NEW CHAT */}
      <Box sx={{ px: 2, py: 2 }}>
        <Button
          fullWidth
          variant="contained"
          startIcon={<AddIcon />}
          onClick={handleNewChat}
          sx={{
            justifyContent: "flex-start",
            textTransform: "none",
            borderRadius: 3,
            bgcolor: "#111827",
            boxShadow: "none",
            "&:hover": {
              bgcolor: "#0b1220",
              boxShadow: "none"
            }
          }}
        >
          Nuova chat
        </Button>
      </Box>

      {/* AUTH PANEL */}
      <Box px={2} pb={2}>
        {sidebarTopSlot ?? <AuthPanel />}
      </Box>

      {/* HISTORY */}
      <Box sx={{ flex: 1, overflowY: "auto" }}>
        <SidebarSectionTitle>Le tue Chat</SidebarSectionTitle>

        <List dense disablePadding>
          {sessions.map((session) => (
            <SessionItem
              key={session.id}
              title={session.title}
              active={(activeSessionId || sessions[0]?.id) === session.id}
              onClick={() => switchSession(session.id)}
              onDelete={() => deleteSession(session.id)}
            />
          ))}
        </List>
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
          sx={{
            "& .MuiDrawer-paper": {
              width: SIDEBAR_WIDTH
            }
          }}
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
              boxSizing: "border-box"
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
          flexDirection: "column"
        }}
      >

        {/* TOP BAR */}
        <Box
          sx={{
            height: 56,
            px: 2,
            display: "flex",
            alignItems: "center",
            gap: 1,
            borderBottom: "1px solid #e5e7eb"
          }}
        >
          {isMobile && (
            <IconButton onClick={() => setMobileOpen(true)}>
              <MenuIcon />
            </IconButton>
          )}

          <Typography fontSize={14} fontWeight={600}>
            ebayGPT
          </Typography>
        </Box>

        {/* CHAT AREA */}
        <Box
          sx={{
            flex: 1,
            overflowY: "auto"
          }}
        >
          {children}
        </Box>

        {/* COMPOSER */}
        {composer && (
          <Box
            sx={{
              position: "sticky",
              bottom: 0
            }}
          >
            {composer}
          </Box>
        )}

      </Box>

    </Box>
  )
}
import {
  Box,
  Typography,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Tooltip,
  IconButton,
  Button
} from "@mui/material"
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline"
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline"
import AddIcon from "@mui/icons-material/Add"
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft"
import ChevronRightIcon from "@mui/icons-material/ChevronRight"
import SearchIcon from "@mui/icons-material/Search"
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep"
import { styled } from "@mui/material/styles"
import AuthPanel from "../../../auth/ui/AuthPanel"

const ResizeHandle = styled(Box)(() => ({
  width: "4px",
  height: "100%",
  cursor: "col-resize",
  position: "absolute",
  right: -2,
  top: 0,
  zIndex: 10,
  transition: "background-color 0.2s",
  "&:hover": {
    backgroundColor: "var(--bg-secondary)",
  },
  "&:active": {
    backgroundColor: "var(--border-color)",
  }
}))

function SidebarSectionTitle({ children, collapsed }: { children: React.ReactNode, collapsed?: boolean }) {
  if (collapsed) return <Box sx={{ height: 20 }} />
  return (
    <Typography
      sx={{
        fontSize: 10,
        fontWeight: 600,
        color: "var(--text-secondary)",
        textTransform: "uppercase",
        letterSpacing: 0.7,
        px: 2,
        pb: 0.75,
        whiteSpace: 'nowrap',
        overflow: 'hidden'
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
  onDelete,
  collapsed
}: {
  title: string
  active?: boolean
  onClick: () => void
  onDelete: () => void
  collapsed?: boolean
}) {
  return (
    <ListItem disablePadding sx={{ px: 1, mb: 0.25 }}>
      <Tooltip title={collapsed ? title : ""} placement="right">
        <ListItemButton
          onClick={onClick}
          sx={{
            borderRadius: 2,
            py: 0.875,
            px: collapsed ? 0 : 1.25,
            justifyContent: collapsed ? 'center' : 'flex-start',
            bgcolor: active ? "var(--bg-secondary)" : "transparent",
            "&:hover": {
              bgcolor: "var(--bg-secondary)",
              boxShadow: "0 2px 8px rgba(0,0,0,0.04)"
            },
            minHeight: 36
          }}
        >
          <ChatBubbleOutlineIcon
            sx={{
              fontSize: 14,
              color: active ? "var(--text-primary)" : "var(--text-secondary)",
              mr: collapsed ? 0 : 1.25,
              flexShrink: 0
            }}
          />

          {!collapsed && (
            <ListItemText
              primary={title}
              primaryTypographyProps={{
                fontSize: 13,
                fontWeight: active ? 500 : 400,
                color: active ? "var(--text-primary)" : "var(--text-secondary)",
                noWrap: true,
                textAlign: collapsed ? 'center' : 'left'
              }}
            />
          )}

          {!collapsed && (
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
                  color: "var(--text-secondary)",
                  p: 0.4,
                  "&:hover": { color: "#ef4444", bgcolor: "transparent" }
                }}
              >
                <DeleteOutlineIcon sx={{ fontSize: 14 }} />
              </IconButton>
            </Tooltip>
          )}
        </ListItemButton>
      </Tooltip>
    </ListItem>
  )
}

interface SidebarContentProps {
  isMobile: boolean
  isCollapsed: boolean
  sessions: any[]
  activeSessionId: string | null
  onNewChat: () => void
  onSwitchSession: (id: string) => void
  onDeleteSession: (id: string) => void
  onToggleCollapse: () => void
  onSearchTrigger: () => void
  onClearMemory: () => void
  onResize: (e: React.MouseEvent) => void
  sidebarTopSlot?: React.ReactNode
}

export function SidebarContent({
  isMobile,
  isCollapsed,
  sessions,
  activeSessionId,
  onNewChat,
  onSwitchSession,
  onDeleteSession,
  onToggleCollapse,
  onSearchTrigger,
  onClearMemory,
  onResize,
  sidebarTopSlot
}: SidebarContentProps) {
  return (
    <Box
      sx={{
        height: "100%",
        display: "flex",
        flexDirection: "column",
        bgcolor: 'var(--bg-secondary)',
        position: 'relative',
        transition: 'width 0.2s cubic-bezier(0.4, 0, 0.2, 1), background-color 0.2s',
        overflow: 'hidden',
        border: "none",
        borderRight: isMobile ? "none" : "1px solid var(--border-color)",
        borderRadius: 0
      }}
    >
      {!isMobile && !isCollapsed && <ResizeHandle onMouseDown={onResize} />}

      {/* COLLAPSE TOGGLE (FULL WIDTH VIEW) */}
      {!isMobile && (
        <Box
          sx={{
            display: 'flex',
            justifyContent: isCollapsed ? 'center' : 'space-between',
            alignItems: 'center',
            px: isCollapsed ? 0 : 1.5,
            pt: 1
          }}
        >
          {!isCollapsed && (
            <Tooltip title="Cerca chat">
              <IconButton
                size="small"
                onClick={onSearchTrigger}
                sx={{
                  color: "var(--text-secondary)",
                  "&:hover": { bgcolor: "var(--bg-secondary)" }
                }}
              >
                <SearchIcon sx={{ fontSize: 18 }} />
              </IconButton>
            </Tooltip>
          )}

          <IconButton
            size="small"
            onClick={onToggleCollapse}
            sx={{
              color: "var(--text-secondary)",
              "&:hover": { bgcolor: "var(--bg-secondary)" }
            }}
          >
            {isCollapsed ? <ChevronRightIcon sx={{ fontSize: 18 }} /> : <ChevronLeftIcon sx={{ fontSize: 18 }} />}
          </IconButton>
        </Box>
      )}

      {/* NEW CHAT */}
      <Box sx={{ px: isCollapsed ? 1 : 2, pt: isCollapsed ? 1 : 1, pb: 2 }}>
        <Tooltip title={isCollapsed ? "Nuova chat" : ""} placement="right">
          <Button
            fullWidth
            variant="outlined"
            onClick={onNewChat}
            sx={{
              justifyContent: "center",
              textTransform: "none",
              borderRadius: 2,
              minWidth: isCollapsed ? 0 : "auto",
              fontSize: 13,
              fontWeight: 500,
              color: 'var(--text-primary)',
              borderColor: 'var(--border-color)',
              bgcolor: 'var(--bg-primary)',
              boxShadow: "0 1px 2px rgba(0,0,0,0.05)",
              py: 0.875,
              px: isCollapsed ? 0 : 2,
              "&:hover": {
                bgcolor: 'var(--bg-secondary)',
                borderColor: 'var(--text-secondary)',
                boxShadow: "0 2px 4px rgba(0,0,0,0.05)"
              }
            }}
          >
            <AddIcon sx={{ fontSize: 16, mr: isCollapsed ? 0 : 1 }} />
            {!isCollapsed && "Nuova chat"}
          </Button>
        </Tooltip>
      </Box>

      {/* AUTH PANEL */}
      <Box px={0} pb={0}>
        {sidebarTopSlot ?? <AuthPanel />}
      </Box>

      {/* HISTORY */}
      <Box sx={{ flex: 1, overflowY: "auto", overflowX: 'hidden', pb: 1 }}>
        {sessions.length > 0 && (
          <>
            <SidebarSectionTitle collapsed={isCollapsed}>Recenti</SidebarSectionTitle>
            <List dense disablePadding>
              {sessions.map((session) => (
                <SessionItem
                  key={session.id}
                  title={session.title}
                  active={activeSessionId === session.id}
                  collapsed={isCollapsed}
                  onClick={() => onSwitchSession(session.id)}
                  onDelete={() => onDeleteSession(session.id)}
                />
              ))}
            </List>
          </>
        )}
      </Box>

      <Box sx={{ borderTop: "1px solid var(--border-color)", p: isCollapsed ? 1 : 1 }}>
        <Tooltip title={isCollapsed ? "Svuota memoria" : ""} placement="right">
          <Button
            fullWidth
            variant="text"
            onClick={onClearMemory}
            sx={{
              justifyContent: isCollapsed ? "center" : "flex-start",
              textTransform: "none",
              borderRadius: 2,
              fontSize: 12,
              fontWeight: 500,
              px: isCollapsed ? 0 : 1.25,
              minWidth: isCollapsed ? 0 : "auto",
              color: "var(--text-secondary)",
              "&:hover": { bgcolor: "rgba(239, 68, 68, 0.08)", color: "#ef4444" }
            }}
          >
            <DeleteSweepIcon sx={{ fontSize: 15, mr: isCollapsed ? 0 : 1 }} />
            {!isCollapsed && "Svuota memoria server"}
          </Button>
        </Tooltip>
      </Box>
    </Box>
  )
}

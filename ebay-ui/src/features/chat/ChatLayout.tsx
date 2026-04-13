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
  Tooltip,
  InputBase,
  Divider,
  Badge,
} from "@mui/material"
import { useTheme, styled } from "@mui/material/styles"
import { useCallback, useMemo, useState } from "react"

import AddIcon from "@mui/icons-material/Add"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline"
import MenuIcon from "@mui/icons-material/Menu"
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline"
import DeleteSweepIcon from "@mui/icons-material/DeleteSweep"
import ChevronLeftIcon from "@mui/icons-material/ChevronLeft"
import ChevronRightIcon from "@mui/icons-material/ChevronRight"
import SearchIcon from "@mui/icons-material/Search"
import SettingsIcon from "@mui/icons-material/Settings"
import FavoriteIcon from "@mui/icons-material/Favorite"

import { useChatStore, type ChatSession } from "./store/chatStore"
import { useSidebarStore, type SidebarState } from "./store/sidebarStore"
import { useSettingsStore } from "./store/settingsStore"
import { useScrollContainerProvider } from "./ScrollContainerContext"
import AuthPanel from "../../auth/ui/AuthPanel"
import SettingsModal from "./SettingsModal"
import ProfileBadge from "./ProfileBadge"
import WishlistPanel from "./WishlistPanel"
import { useWishlistStore } from "./store/wishlistStore"

interface Props {
  children: React.ReactNode
  composer?: React.ReactNode
  onNewChat?: () => void
  sidebarTopSlot?: React.ReactNode
}

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
    backgroundColor: "#e5e7eb",
  },
  "&:active": {
    backgroundColor: "#d1d5db",
  }
}))

function SidebarSectionTitle({ children, collapsed }: { children: React.ReactNode, collapsed?: boolean }) {
  if (collapsed) return <Box sx={{ height: 20 }} />
  return (
    <Typography
      sx={{
        fontSize: 10,
        fontWeight: 600,
        color: "#9ca3af",
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
            borderRadius: 1.5,
            py: 0.875,
            px: collapsed ? 0 : 1.25,
            justifyContent: collapsed ? 'center' : 'flex-start',
            bgcolor: active ? "var(--bg-secondary)" : "transparent",
            "&:hover": {
              bgcolor: "var(--bg-secondary)"
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

export default function ChatLayout({
  children,
  composer,
  onNewChat,
  sidebarTopSlot
}: Props) {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down("md"))
  const { ref: scrollRef, Provider: ScrollProvider } = useScrollContainerProvider()

  const sessions = useChatStore((s) => s.sessions)
  const activeSessionId = useChatStore((s) => s.activeSessionId)
  const createSession = useChatStore((s) => s.createSession)
  const switchSession = useChatStore((s) => s.switchSession)
  const deleteSession = useChatStore((s) => s.deleteSession)
  const clearMemory = useChatStore((s) => s.clearMemory)

  const mobileOpen = useSidebarStore((s: SidebarState) => s.mobileOpen)
  const setMobileOpen = useSidebarStore((s: SidebarState) => s.setMobileOpen)
  const isCollapsed = useSidebarStore((s: SidebarState) => s.isCollapsed)
  const setIsCollapsed = useSidebarStore((s: SidebarState) => s.setIsCollapsed)
  const width = useSidebarStore((s: SidebarState) => s.width)
  const setWidth = useSidebarStore((s: SidebarState) => s.setWidth)

  const setSettingsOpen = useSettingsStore((s) => s.setOpen)
  
  const [isSearchOpen, setIsSearchOpen] = useState(false)
  const [searchQuery, setSearchQuery] = useState("")
  const [wishlistOpen, setWishlistOpen] = useState(false)

  const wishlistItems = useWishlistStore((s) => s.items)

  const filteredSessions = useMemo(() => {
    if (!searchQuery) return sessions
    return sessions.filter(s => s.title.toLowerCase().includes(searchQuery.toLowerCase()))
  }, [sessions, searchQuery])

  const groupSessionsByDate = (sessionsToGroup: ChatSession[]) => {
    const groups: { [key: string]: any[] } = {
      "Oggi": [],
      "Ieri": [],
      "Ultimi 7 giorni": [],
      "Più vecchi": []
    }

    const today = new Date()
    today.setHours(0, 0, 0, 0)
    const yesterday = new Date(today)
    yesterday.setDate(yesterday.getDate() - 1)
    const sevenDaysAgo = new Date(today)
    sevenDaysAgo.setDate(sevenDaysAgo.getDate() - 7)

    sessionsToGroup.forEach(session => {
      const date = new Date(session.createdAt || Date.now())
      if (date >= today) groups["Oggi"].push(session)
      else if (date >= yesterday) groups["Ieri"].push(session)
      else if (date >= sevenDaysAgo) groups["Ultimi 7 giorni"].push(session)
      else groups["Più vecchi"].push(session)
    })

    return Object.entries(groups).filter(([_, items]) => items.length > 0)
  }

  const groupedSessions = groupSessionsByDate(filteredSessions)

  const activeSession = sessions.find(
    (s) => s.id === (activeSessionId || sessions[0]?.id)
  )

  const handleNewChat = () => {
    // BUG FIX: If we're already in a fresh session with no messages, don't create a new one.
    if (activeSession && activeSession.chat.length <= 1) {
      onNewChat?.()
      if (isMobile) setMobileOpen(false)
      return
    }

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

  const handleResize = useCallback((e: React.MouseEvent) => {
    if (isCollapsed) return

    const startX = e.clientX
    const startWidth = width

    const onMouseMove = (moveEvent: MouseEvent) => {
      const delta = moveEvent.clientX - startX
      const newWidth = Math.min(Math.max(startWidth + delta, 200), 450)
      setWidth(newWidth)
    }

    const onMouseUp = () => {
      document.removeEventListener("mousemove", onMouseMove)
      document.removeEventListener("mouseup", onMouseUp)
    }

    document.addEventListener("mousemove", onMouseMove)
    document.addEventListener("mouseup", onMouseUp)
  }, [width, setWidth, isCollapsed])

  const currentSidebarWidth = isMobile ? 260 : (isCollapsed ? 64 : width)

  const sidebarContent = (
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
      {!isMobile && !isCollapsed && <ResizeHandle onMouseDown={handleResize} />}

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
          {/* SEARCH TRIGGER (Corner) - Only render if not collapsed to allow centering of toggle */}
          {!isCollapsed && (
            <Tooltip title="Cerca chat">
              <IconButton
                size="small"
                onClick={() => setIsSearchOpen(true)}
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
            onClick={() => setIsCollapsed(!isCollapsed)}
            sx={{
              color: "var(--text-secondary)",
              "&:hover": { bgcolor: "var(--bg-secondary)" }
            }}
          >
            {isCollapsed ? <ChevronRightIcon sx={{ fontSize: 18 }} /> : <ChevronLeftIcon sx={{ fontSize: 18 }} />}
          </IconButton>
        </Box>
      )}

      {/* NEW CHAT (PROMINENT AT TOP) */}
      <Box sx={{ px: isCollapsed ? 1 : 2, pt: isCollapsed ? 1 : 1, pb: 2 }}>
        <Tooltip title={isCollapsed ? "Nuova chat" : ""} placement="right">
          <Button
            fullWidth
            variant="outlined"
            onClick={handleNewChat}
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

      {/* PROFILE BADGE */}
      <ProfileBadge />

      {/* HISTORY */}
      <Box sx={{ flex: 1, overflowY: "auto", overflowX: 'hidden', pb: 1 }}>
        {groupedSessions.length > 0 ? (
          groupedSessions.map(([title, items]) => (
            <Box key={title} sx={{ mb: 1 }}>
              <SidebarSectionTitle collapsed={isCollapsed}>{title}</SidebarSectionTitle>
              <List dense disablePadding>
                {items.map((session) => (
                  <SessionItem
                    key={session.id}
                    title={session.title}
                    active={(activeSessionId || sessions[0]?.id) === session.id}
                    collapsed={isCollapsed}
                    onClick={() => {
                      switchSession(session.id)
                      if (isMobile) setMobileOpen(false)
                    }}
                    onDelete={() => deleteSession(session.id)}
                  />
                ))}
              </List>
            </Box>
          ))
        ) : (
          <Box sx={{ p: 3, textAlign: 'center', opacity: 0.5, mt: 4 }}>
            <ChatBubbleOutlineIcon sx={{ fontSize: 40, mb: 1, color: 'var(--text-secondary)' }} />
            <Typography sx={{ fontSize: 13, color: 'var(--text-secondary)', fontWeight: 500 }}>
              {searchQuery ? "Nessun risultato" : "Nessuna chat ancora"}
            </Typography>
          </Box>
        )}
      </Box>


      <Box sx={{ borderTop: "1px solid var(--border-color)", p: isCollapsed ? 1 : 1 }}>
        <Tooltip title={isCollapsed ? "Svuota memoria" : ""} placement="right">
          <Button
            fullWidth
            variant="text"
            onClick={handleClearMemory}
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

  return (
    <Box sx={{ display: "flex", height: "100dvh", overflow: "hidden" }}>
      {isMobile ? (
        <Drawer
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          variant="temporary"
          ModalProps={{ keepMounted: true }}
          sx={{ "& .MuiDrawer-paper": { width: currentSidebarWidth, border: "none" } }}
        >
          {sidebarContent}
        </Drawer>
      ) : (
        <Drawer
          variant="permanent"
          sx={{
            width: currentSidebarWidth,
            flexShrink: 0,
            transition: 'width 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
            "& .MuiDrawer-paper": {
              width: currentSidebarWidth,
              boxSizing: "border-box",
              border: "none",
              overflow: 'hidden',
              transition: 'width 0.2s cubic-bezier(0.4, 0, 0.2, 1)',
              borderRadius: 0
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
          minHeight: 0,
          display: "flex",
          flexDirection: "column",
          bgcolor: 'var(--bg-primary)',
          transition: 'background-color 0.2s'
        }}
      >
        {/* TOP BAR */}
        <Box
          sx={{
            height: 56,
            display: 'flex',
            alignItems: 'center',
            px: 2,
            bgcolor: 'var(--bg-primary)',
            transition: 'background-color 0.2s',
            position: 'relative' // For centering title
          }}
        >
          {/* Left — branding & Title */}
          <Box sx={{ display: 'flex', alignItems: 'center', gap: 2 }}>
            {isMobile && (
              <IconButton
                onClick={() => setMobileOpen(true)}
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
                color: (activeSession?.title === "Nuova chat" || !activeSession) ? "var(--text-secondary)" : "var(--text-primary)"
              }}
            >
              {(activeSession?.title === "Nuova chat" || !activeSession) ? "Cerca" : activeSession.title}
            </Typography>
          </Box>

          {/* Right — actions */}
          <Box sx={{ ml: 'auto', display: 'flex', alignItems: 'center', gap: 1.5, mr: 1 }}>

            <Tooltip title="Wishlist">
              <IconButton size="small" sx={{ color: "var(--text-secondary)" }} onClick={() => setWishlistOpen(true)}>
                <Badge badgeContent={wishlistItems.length} color="error" sx={{ "& .MuiBadge-badge": { fontSize: 10, height: 16, minWidth: 16 } }}>
                  <FavoriteIcon sx={{ fontSize: 18 }} />
                </Badge>
              </IconButton>
            </Tooltip>

            <Tooltip title="Impostazioni">
              <IconButton size="small" sx={{ color: "var(--text-secondary)" }} onClick={() => setSettingsOpen(true)}>
                <SettingsIcon sx={{ fontSize: 18 }} />
              </IconButton>
            </Tooltip>



          </Box>
        </Box>


        {/* CHAT AREA */}
        <Box
          ref={scrollRef}
          id="chat-scroll-container"
          sx={{
            flex: 1,
            minHeight: 0,
            overflowY: "auto",
            display: "flex",
            flexDirection: "column"
          }}
        >
          <Box
            sx={{
              width: "100%",
              maxWidth: 1000,
              mx: "auto",
              flex: 1,
              px: { xs: 2, md: 4 }
            }}
          >
            <ScrollProvider value={scrollRef}>
              {children}
            </ScrollProvider>
          </Box>
        </Box>

        {/* COMPOSER */}
        {composer && (
          <Box
            sx={{
              px: { xs: 1.5, md: 3 },
              pt: 1,
              pb: { xs: 1.5, md: 2 },
              bgcolor: "var(--bg-primary)",
              transition: 'background-color 0.2s'
            }}
          >
            <Box sx={{ maxWidth: 1000, mx: "auto" }}>
              {composer}
            </Box>
          </Box>
        )}
      </Box>

      {/* SEARCH OVERLAY */}
      {isSearchOpen && (
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
          onClick={() => setIsSearchOpen(false)}
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
                onChange={(e) => setSearchQuery(e.target.value)}
                sx={{ fontSize: 18, color: 'var(--text-primary)' }}
                onKeyDown={(e) => {
                  if (e.key === 'Escape') setIsSearchOpen(false)
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
                      onClick={() => {
                        switchSession(session.id)
                        setIsSearchOpen(false)
                      }}
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
            onClick={() => setIsSearchOpen(false)}
            sx={{
              position: 'absolute',
              top: 20,
              left: 20
            }}
          >
            <ChevronLeftIcon sx={{ fontSize: 32, color: 'var(--text-secondary)' }} />
          </IconButton>
        </Box>
      )}



      <SettingsModal />

      <Drawer
        anchor="right"
        open={wishlistOpen}
        onClose={() => setWishlistOpen(false)}
        SlideProps={{ timeout: 500 }}
        slotProps={{
          backdrop: {
            sx: {
              bgcolor: "rgba(0, 0, 0, 0.1)",
              backdropFilter: "blur(4px)",
            }
          }
        }}
        sx={{
          "& .MuiDrawer-paper": {
            bgcolor: "transparent",
            boxShadow: "none",
            border: "none",
            overflow: "visible"
          }
        }}
      >
        <WishlistPanel onClose={() => setWishlistOpen(false)} />
      </Drawer>
    </Box>
  )
}

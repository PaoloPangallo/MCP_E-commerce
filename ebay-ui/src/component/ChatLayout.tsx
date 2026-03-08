import { useEffect, useMemo, useState } from "react"
import {
  Box,
  Button,
  Divider,
  Drawer,
  IconButton,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Typography,
  useMediaQuery
} from "@mui/material"
import { useTheme } from "@mui/material/styles"

import AddIcon from "@mui/icons-material/Add"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline"
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline"
import MenuIcon from "@mui/icons-material/Menu"
import PushPinOutlinedIcon from "@mui/icons-material/PushPinOutlined"
import SearchIcon from "@mui/icons-material/Search"

interface HistoryItem {
  query: string
  results: number
}

interface Props {
  children: React.ReactNode
  composer?: React.ReactNode
  onSearch?: (query: string) => void
  onNewChat?: () => void
  sidebarTopSlot?: React.ReactNode
}

const HISTORY_KEY = "search_history"
const PINNED_KEY = "pinned_searches"
const SIDEBAR_WIDTH = 280

function readArray<T>(key: string): T[] {
  try {
    const raw = localStorage.getItem(key)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

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

function SidebarItem({
  query,
  onClick,
  onPin,
  isPinned = false
}: {
  query: string
  onClick: () => void
  onPin?: () => void
  isPinned?: boolean
}) {
  return (
    <ListItem disablePadding sx={{ px: 1 }}>
      <ListItemButton
        onClick={onClick}
        sx={{
          borderRadius: 3,
          py: 1.1,
          px: 1.25,
          alignItems: "center",
          "&:hover": {
            bgcolor: "#eceff3"
          }
        }}
      >
        <SearchIcon
          sx={{
            fontSize: 16,
            color: "#6b7280",
            mr: 1.25,
            flexShrink: 0
          }}
        />

        <ListItemText
          primary={query}
          primaryTypographyProps={{
            fontSize: 13,
            color: "#111827",
            noWrap: true
          }}
        />

        {onPin ? (
          <IconButton
            size="small"
            edge="end"
            onClick={(event) => {
              event.stopPropagation()
              onPin()
            }}
            sx={{
              ml: 0.5,
              color: isPinned ? "#111827" : "#9ca3af"
            }}
          >
            <PushPinOutlinedIcon sx={{ fontSize: 16 }} />
          </IconButton>
        ) : null}
      </ListItemButton>
    </ListItem>
  )
}

export default function ChatLayout({
  children,
  composer,
  onSearch,
  onNewChat,
  sidebarTopSlot
}: Props) {
  const theme = useTheme()
  const isMobile = useMediaQuery(theme.breakpoints.down("md"))

  const [mobileOpen, setMobileOpen] = useState(false)
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [pinned, setPinned] = useState<string[]>([])

  useEffect(() => {
    const syncSidebar = () => {
      const nextHistory = readArray<HistoryItem>(HISTORY_KEY).filter(
        (item) =>
          item &&
          typeof item.query === "string" &&
          item.query.trim().length > 0 &&
          typeof item.results === "number"
      )

      const nextPinned = readArray<string>(PINNED_KEY).filter(
        (item) => typeof item === "string" && item.trim().length > 0
      )

      setHistory(nextHistory)
      setPinned(nextPinned)
    }

    syncSidebar()
    window.addEventListener("search_history_updated", syncSidebar)

    return () => {
      window.removeEventListener("search_history_updated", syncSidebar)
    }
  }, [])

  const visibleHistory = useMemo(() => history.slice(0, 20), [history])

  const clearHistory = () => {
    localStorage.removeItem(HISTORY_KEY)
    setHistory([])
    window.dispatchEvent(new Event("search_history_updated"))
  }

  const pinSearch = (query: string) => {
    const updated = [query, ...pinned.filter((item) => item !== query)].slice(
      0,
      10
    )
    localStorage.setItem(PINNED_KEY, JSON.stringify(updated))
    setPinned(updated)
    window.dispatchEvent(new Event("search_history_updated"))
  }

  const unpinSearch = (query: string) => {
    const updated = pinned.filter((item) => item !== query)
    localStorage.setItem(PINNED_KEY, JSON.stringify(updated))
    setPinned(updated)
    window.dispatchEvent(new Event("search_history_updated"))
  }

  const handleSidebarSearch = (query: string) => {
    onSearch?.(query)
    if (isMobile) {
      setMobileOpen(false)
    }
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
      <Box sx={{ px: 2, pt: 2, pb: 1.5 }}>
        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1.25,
            px: 1,
            py: 0.5
          }}
        >
          <Box
            sx={{
              width: 30,
              height: 30,
              borderRadius: 2.5,
              bgcolor: "#111827",
              display: "flex",
              alignItems: "center",
              justifyContent: "center",
              color: "#fff",
              flexShrink: 0
            }}
          >
            <AutoAwesomeIcon sx={{ fontSize: 17 }} />
          </Box>

          <Box>
            <Typography
              sx={{
                fontSize: 14,
                fontWeight: 700,
                color: "#111827",
                lineHeight: 1.15
              }}
            >
              ebayGPT
            </Typography>

            <Typography
              sx={{
                fontSize: 12,
                color: "#6b7280",
                lineHeight: 1.25
              }}
            >
              chat search + seller trust
            </Typography>
          </Box>
        </Box>
      </Box>

      <Box sx={{ px: 2, pb: 1.5 }}>
        <Button
          fullWidth
          variant="contained"
          startIcon={<AddIcon />}
          onClick={onNewChat}
          sx={{
            justifyContent: "flex-start",
            textTransform: "none",
            borderRadius: 3,
            py: 1.15,
            px: 1.5,
            fontWeight: 600,
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

      {sidebarTopSlot ? <Box px={2} pb={1.5}>{sidebarTopSlot}</Box> : null}

      <Box
        sx={{
          flex: 1,
          minHeight: 0,
          overflowY: "auto",
          pb: 1
        }}
      >
        {pinned.length > 0 ? (
          <>
            <SidebarSectionTitle>Pinned</SidebarSectionTitle>

            <List dense disablePadding sx={{ mb: 1 }}>
              {pinned.map((query) => (
                <SidebarItem
                  key={`pinned-${query}`}
                  query={query}
                  onClick={() => handleSidebarSearch(query)}
                  onPin={() => unpinSearch(query)}
                  isPinned
                />
              ))}
            </List>

            <Divider sx={{ mx: 2, mb: 1.5 }} />
          </>
        ) : null}

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between",
            px: 2,
            pb: 1
          }}
        >
          <SidebarSectionTitle>Recenti</SidebarSectionTitle>

          {visibleHistory.length > 0 ? (
            <IconButton
              size="small"
              onClick={clearHistory}
              sx={{ mt: -1, color: "#9ca3af" }}
            >
              <DeleteOutlineIcon sx={{ fontSize: 18 }} />
            </IconButton>
          ) : null}
        </Box>

        {visibleHistory.length > 0 ? (
          <List dense disablePadding>
            {visibleHistory.map((item) => {
              const isPinned = pinned.includes(item.query)

              return (
                <SidebarItem
                  key={`${item.query}-${item.results}`}
                  query={item.query}
                  onClick={() => handleSidebarSearch(item.query)}
                  onPin={() =>
                    isPinned
                      ? unpinSearch(item.query)
                      : pinSearch(item.query)
                  }
                  isPinned={isPinned}
                />
              )
            })}
          </List>
        ) : (
          <Box sx={{ px: 2.25, pt: 0.5 }}>
            <Typography
              sx={{
                fontSize: 13,
                color: "#6b7280",
                lineHeight: 1.6
              }}
            >
              Le tue ricerche recenti compariranno qui.
            </Typography>
          </Box>
        )}
      </Box>

      <Box sx={{ px: 2, py: 1.75 }}>
        <Divider sx={{ mb: 1.5 }} />

        <Box
          sx={{
            display: "flex",
            alignItems: "center",
            gap: 1,
            px: 1
          }}
        >
          <ChatBubbleOutlineIcon sx={{ fontSize: 16, color: "#6b7280" }} />

          <Typography
            sx={{
              fontSize: 12.5,
              color: "#6b7280"
            }}
          >
            UI ispirata a ChatGPT, ottimizzata per search + reasoning.
          </Typography>
        </Box>
      </Box>
    </Box>
  )

  return (
    <Box
      sx={{
        display: "flex",
        minHeight: "100vh",
        bgcolor: "#f7f8fb"
      }}
    >
      {isMobile ? (
        <Drawer
          open={mobileOpen}
          onClose={() => setMobileOpen(false)}
          variant="temporary"
          ModalProps={{ keepMounted: true }}
          sx={{
            "& .MuiDrawer-paper": {
              width: SIDEBAR_WIDTH,
              borderRight: "1px solid #e5e7eb"
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
              boxSizing: "border-box",
              borderRight: "1px solid #e5e7eb"
            }
          }}
        >
          {sidebarContent}
        </Drawer>
      )}

      <Box
        sx={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column"
        }}
      >
        <Box
          sx={{
            height: 56,
            px: { xs: 1.5, md: 2.5 },
            display: "flex",
            alignItems: "center",
            gap: 1,
            borderBottom: "1px solid #e5e7eb",
            bgcolor: "rgba(247, 248, 251, 0.86)",
            backdropFilter: "blur(10px)",
            position: "sticky",
            top: 0,
            zIndex: 10
          }}
        >
          {isMobile ? (
            <IconButton onClick={() => setMobileOpen(true)}>
              <MenuIcon />
            </IconButton>
          ) : null}

          <Typography
            sx={{
              fontSize: 14,
              fontWeight: 600,
              color: "#111827"
            }}
          >
            ebayGPT
          </Typography>

          <Typography
            sx={{
              fontSize: 12.5,
              color: "#6b7280"
            }}
          >
            Conversational search workspace
          </Typography>
        </Box>

        <Box
          sx={{
            flex: 1,
            minHeight: 0,
            overflowY: "auto"
          }}
        >
          {children}
        </Box>

        {composer ? (
          <Box
            sx={{
              position: "sticky",
              bottom: 0,
              zIndex: 8,
              background:
                "linear-gradient(180deg, rgba(247,248,251,0) 0%, rgba(247,248,251,0.9) 24%, rgba(247,248,251,1) 100%)"
            }}
          >
            {composer}
          </Box>
        ) : null}
      </Box>
    </Box>
  )
}
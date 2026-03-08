import { useMemo } from "react"
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
import DeleteOutlineIcon from "@mui/icons-material/DeleteOutline"
import MenuIcon from "@mui/icons-material/Menu"
import PushPinOutlinedIcon from "@mui/icons-material/PushPinOutlined"
import SearchIcon from "@mui/icons-material/Search"

import { useSidebarStore } from "./store/sidebarStore"
import AuthPanel from "../../auth/ui/AuthPanel"

interface Props {
  children: React.ReactNode
  composer?: React.ReactNode
  onSearch?: (query: string) => void
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
          "&:hover": { bgcolor: "#eceff3" }
        }}
      >
        <SearchIcon sx={{ fontSize: 16, color: "#6b7280", mr: 1.25 }} />

        <ListItemText
          primary={query}
          primaryTypographyProps={{
            fontSize: 13,
            color: "#111827",
            noWrap: true
          }}
        />

        {onPin && (
          <IconButton
            size="small"
            onClick={(e) => {
              e.stopPropagation()
              onPin()
            }}
            sx={{
              color: isPinned ? "#111827" : "#9ca3af"
            }}
          >
            <PushPinOutlinedIcon sx={{ fontSize: 16 }} />
          </IconButton>
        )}
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

  const mobileOpen = useSidebarStore((s) => s.mobileOpen)
  const history = useSidebarStore((s) => s.history)
  const pinned = useSidebarStore((s) => s.pinned)

  const setMobileOpen = useSidebarStore((s) => s.setMobileOpen)
  const clearHistory = useSidebarStore((s) => s.clearHistory)
  const pinSearch = useSidebarStore((s) => s.pinSearch)
  const unpinSearch = useSidebarStore((s) => s.unpinSearch)

  const visibleHistory = useMemo(() => history.slice(0, 20), [history])

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
          onClick={onNewChat}
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

        {pinned.length > 0 && (
          <>
            <SidebarSectionTitle>Pinned</SidebarSectionTitle>

            <List dense disablePadding>
              {pinned.map((query) => (
                <SidebarItem
                  key={query}
                  query={query}
                  onClick={() => handleSidebarSearch(query)}
                  onPin={() => unpinSearch(query)}
                  isPinned
                />
              ))}
            </List>

            <Divider sx={{ mx: 2, my: 1 }} />
          </>
        )}

        <Box display="flex" justifyContent="space-between" px={2}>
          <SidebarSectionTitle>Recenti</SidebarSectionTitle>

          {visibleHistory.length > 0 && (
            <IconButton size="small" onClick={clearHistory}>
              <DeleteOutlineIcon sx={{ fontSize: 18 }} />
            </IconButton>
          )}
        </Box>

        {visibleHistory.length > 0 ? (
          <List dense disablePadding>
            {visibleHistory.map((item) => {

              const isPinned = pinned.some(
                (p) => p.toLowerCase() === item.query.toLowerCase()
              )

              return (
                <SidebarItem
                  key={item.query}
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
          <Box px={2}>
            <Typography fontSize={13} color="#6b7280">
              Le tue ricerche compariranno qui.
            </Typography>
          </Box>
        )}

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
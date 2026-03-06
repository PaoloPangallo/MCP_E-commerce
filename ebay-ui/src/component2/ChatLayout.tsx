import { useEffect, useMemo, useState } from "react"
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Typography,
  IconButton,
  Divider,
  Chip
} from "@mui/material"

import AddIcon from "@mui/icons-material/Add"
import DeleteIcon from "@mui/icons-material/Delete"
import StarIcon from "@mui/icons-material/Star"
import PushPinIcon from "@mui/icons-material/PushPin"
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline"

import { useAuth } from "../auth/useAuth.ts"
import AuthPanel from "../auth/ui/AuthPanel.tsx"

interface HistoryItem {
  query: string
  results: number
}

interface Props {
  children: React.ReactNode
  onSearch: (query: string) => void
  onNewChat: () => void
}

const HISTORY_KEY = "search_history"
const PINNED_KEY = "pinned_searches"

function readArray<T>(key: string): T[] {
  try {
    const raw = localStorage.getItem(key)
    const parsed = raw ? JSON.parse(raw) : []
    return Array.isArray(parsed) ? parsed : []
  } catch {
    return []
  }
}

export default function ChatLayout({ children, onSearch, onNewChat }: Props) {
  const { user } = useAuth()
  const [history, setHistory] = useState<HistoryItem[]>([])
  const [pinned, setPinned] = useState<string[]>([])

  useEffect(() => {
    const syncSidebar = () => {
      setHistory(readArray<HistoryItem>(HISTORY_KEY).filter(item => typeof item?.query === "string"))
      setPinned(readArray<string>(PINNED_KEY).filter(item => typeof item === "string"))
    }

    syncSidebar()
    window.addEventListener("search_history_updated", syncSidebar)

    return () => window.removeEventListener("search_history_updated", syncSidebar)
  }, [])

  const clearHistory = () => {
    localStorage.removeItem(HISTORY_KEY)
    setHistory([])
    window.dispatchEvent(new Event("search_history_updated"))
  }

  const pinSearch = (query: string) => {
    const updated = [query, ...pinned.filter(item => item !== query)].slice(0, 10)
    localStorage.setItem(PINNED_KEY, JSON.stringify(updated))
    setPinned(updated)
  }

  const unpinSearch = (query: string) => {
    const updated = pinned.filter(item => item !== query)
    localStorage.setItem(PINNED_KEY, JSON.stringify(updated))
    setPinned(updated)
  }

  const visibleHistory = useMemo(() => history.slice(0, 20), [history])

  return (
    <Box display="flex" height="100vh" bgcolor="#fff">
      <Drawer
        variant="permanent"
        sx={{
          width: 280,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: 280,
            bgcolor: "#f9f9f9",
            borderRight: "1px solid #ececec",
            display: "flex",
            flexDirection: "column"
          }
        }}
      >
        <Box sx={{ display: "flex", flexDirection: "column", p: 2, borderBottom: "1px solid #eee" }}>
          <Box sx={{ display: "flex", alignItems: "center", gap: 1.2, mb: 1 }}>
            <Box
              sx={{
                width: 28,
                height: 28,
                borderRadius: "8px",
                bgcolor: "#10a37f",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fff",
                fontWeight: 700,
                fontSize: 14
              }}
            >
              AI
            </Box>

            <Box>
              <Typography sx={{ fontWeight: 700, fontSize: 14 }}>eBay AI Search</Typography>
              <Typography sx={{ fontSize: 12, color: "#8e8ea0" }}>
                chat, ranking, seller trust
              </Typography>
            </Box>
          </Box>

          <Box mt={1}>
            <AuthPanel />
          </Box>

          {user && (
            <Box mt={1.5} display="flex" gap={1} flexWrap="wrap">
              {user.favorite_brands && (
                <Chip
                  label={`Brand: ${user.favorite_brands}`}
                  size="small"
                  sx={{ fontSize: 11, bgcolor: "#eef3ff", color: "#3b5ccc" }}
                />
              )}

              {user.price_preference && (
                <Chip
                  label={`Budget: ${user.price_preference}`}
                  size="small"
                  sx={{ fontSize: 11, bgcolor: "#f4f4f4" }}
                />
              )}
            </Box>
          )}
        </Box>

        <Box p={2}>
          <ListItemButton
            onClick={onNewChat}
            sx={{
              borderRadius: 2,
              border: "1px solid #e5e5e5",
              bgcolor: "#fff",
              "&:hover": { bgcolor: "#f4f4f4" }
            }}
          >
            <AddIcon sx={{ mr: 1 }} />
            <ListItemText primary="Nuova chat" />
          </ListItemButton>
        </Box>

        <Divider />

        {pinned.length > 0 && (
          <>
            <Typography variant="caption" sx={{ px: 2, pt: 2, color: "#8e8ea0", fontWeight: 700 }}>
              Salvati
            </Typography>

            <List dense>
              {pinned.map(query => (
                <ListItem key={query} disablePadding>
                  <ListItemButton onClick={() => onSearch(query)}>
                    <PushPinIcon sx={{ fontSize: 16, mr: 1, color: "#777" }} />

                    <ListItemText
                      primary={query}
                      primaryTypographyProps={{
                        fontSize: 13,
                        noWrap: true,
                        sx: { maxWidth: 170 }
                      }}
                    />

                    <IconButton
                      size="small"
                      aria-label="Rimuovi ricerca salvata"
                      onClick={event => {
                        event.stopPropagation()
                        unpinSearch(query)
                      }}
                    >
                      <DeleteIcon fontSize="small" />
                    </IconButton>
                  </ListItemButton>
                </ListItem>
              ))}
            </List>

            <Divider />
          </>
        )}

        <Box flex={1} overflow="auto">
          <Typography variant="caption" sx={{ px: 2, pt: 2, color: "#8e8ea0", fontWeight: 700 }}>
            Cronologia
          </Typography>

          {visibleHistory.length === 0 ? (
            <Typography sx={{ px: 2, py: 2, fontSize: 13, color: "#8e8ea0" }}>
              Le tue ricerche recenti compariranno qui.
            </Typography>
          ) : (
            <List dense>
              {visibleHistory.map(item => (
                <ListItem key={`${item.query}-${item.results}`} disablePadding>
                  <ListItemButton onClick={() => onSearch(item.query)}>
                    <ChatBubbleOutlineIcon sx={{ fontSize: 16, mr: 1, color: "#777" }} />

                    <ListItemText
                      primary={item.query}
                      secondary={`${item.results ?? 0} risultati`}
                      primaryTypographyProps={{ fontSize: 13, noWrap: true }}
                      secondaryTypographyProps={{ fontSize: 11 }}
                    />

                    <IconButton
                      size="small"
                      aria-label="Salva ricerca"
                      onClick={event => {
                        event.stopPropagation()
                        pinSearch(item.query)
                      }}
                    >
                      <StarIcon fontSize="small" />
                    </IconButton>
                  </ListItemButton>
                </ListItem>
              ))}
            </List>
          )}
        </Box>

        <Box p={2} borderTop="1px solid #ececec">
          {user && (
            <Box sx={{ display: "flex", alignItems: "center", gap: 1, mb: 1.5 }}>
              <Box
                sx={{
                  width: 28,
                  height: 28,
                  borderRadius: "50%",
                  bgcolor: "#10a37f",
                  display: "flex",
                  alignItems: "center",
                  justifyContent: "center",
                  color: "#fff",
                  fontSize: 12,
                  fontWeight: 600
                }}
              >
                {user.email?.charAt(0).toUpperCase()}
              </Box>

              <Typography sx={{ fontSize: 13, color: "#444" }} noWrap>
                {user.email}
              </Typography>
            </Box>
          )}

          <ListItemButton onClick={clearHistory} sx={{ borderRadius: 2 }}>
            <DeleteIcon sx={{ mr: 1 }} />
            <ListItemText primary="Cancella cronologia" />
          </ListItemButton>
        </Box>
      </Drawer>

      <Box
        component="main"
        sx={{
          flexGrow: 1,
          display: "flex",
          flexDirection: "column",
          height: "100vh",
          minWidth: 0
        }}
      >
        {children}
      </Box>
    </Box>
  )
}

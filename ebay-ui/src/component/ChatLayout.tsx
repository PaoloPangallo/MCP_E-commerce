import { useEffect, useMemo, useState } from "react"
import {
  Box,
  Drawer,
  List,
  ListItem,
  ListItemButton,
  ListItemText,
  Typography,
  Divider,
  Button,
  IconButton
} from "@mui/material"

import AddIcon from "@mui/icons-material/Add"
import DeleteIcon from "@mui/icons-material/Delete"
import PushPinIcon from "@mui/icons-material/PushPin"
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline"
import SearchIcon from "@mui/icons-material/Search"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import {useAuth} from "../auth/useAuth.ts";
import AuthPanel from "../auth/ui/AuthPanel.tsx";

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

export default function ChatLayout({
  children,
  onSearch,
  onNewChat
}: Props) {
  const { loggedIn } = useAuth()

  const [history, setHistory] = useState<HistoryItem[]>([])
  const [pinned, setPinned] = useState<string[]>([])

  useEffect(() => {
    const syncSidebar = () => {
      setHistory(
        readArray<HistoryItem>(HISTORY_KEY).filter(
          (item) =>
            item &&
            true &&
            true
        )
      )

      setPinned(
        readArray<string>(PINNED_KEY).filter(
          (item) => typeof item === "string" && item.trim().length > 0
        )
      )
    }

    syncSidebar()
    window.addEventListener("search_history_updated", syncSidebar)

    return () => {
      window.removeEventListener("search_history_updated", syncSidebar)
    }
  }, [])

  const clearHistory = () => {
    localStorage.removeItem(HISTORY_KEY)
    setHistory([])
    window.dispatchEvent(new Event("search_history_updated"))
  }

  const pinSearch = (query: string) => {
    const updated = [query, ...pinned.filter((item) => item !== query)].slice(0, 10)
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

  const visibleHistory = useMemo(() => history.slice(0, 20), [history])

  return (
    <Box display="flex" height="100vh" bgcolor="#ffffff">
      <Drawer
        variant="permanent"
        sx={{
          width: 296,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: 296,
            bgcolor: "#f7f7f8",
            borderRight: "1px solid #ececf1",
            display: "flex",
            flexDirection: "column"
          }
        }}
      >
        <Box
          sx={{
            px: 2,
            pt: 2,
            pb: 1.5
          }}
        >
          <Box
            sx={{
              display: "flex",
              alignItems: "center",
              gap: 1.25,
              px: 1,
              py: 0.75
            }}
          >
            <Box
              sx={{
                width: 30,
                height: 30,
                borderRadius: "10px",
                bgcolor: "#202123",
                display: "flex",
                alignItems: "center",
                justifyContent: "center",
                color: "#fff"
              }}
            >
              <AutoAwesomeIcon sx={{ fontSize: 18 }} />
            </Box>

            <Box>
              <Typography
                sx={{
                  fontWeight: 700,
                  fontSize: 14,
                  color: "#202123",
                  lineHeight: 1.2
                }}
              >
                ebayGPT
              </Typography>

              <Typography
                sx={{
                  fontSize: 12,
                  color: "#6e6e80",
                  lineHeight: 1.2
                }}
              >
                agent search · ranking · seller trust
              </Typography>
            </Box>
          </Box>
        </Box>

        <Box px={2} pb={1.5}>
          <Button
            fullWidth
            variant="contained"
            startIcon={<AddIcon />}
            onClick={onNewChat}
            sx={{
              justifyContent: "flex-start",
              textTransform: "none",
              borderRadius: 3,
              py: 1.2,
              px: 1.5,
              fontWeight: 600,
              bgcolor: "#202123",
              boxShadow: "none",
              "&:hover": {
                bgcolor: "#111214",
                boxShadow: "none"
              }
            }}
          >
            Nuova chat
          </Button>
        </Box>

        {!loggedIn && (
          <Box px={2} pb={1.5}>
            <AuthPanel />
          </Box>
        )}

        {pinned.length > 0 && (
          <>
            <Box px={2} pb={1}>
              <Typography
                sx={{
                  fontSize: 12,
                  fontWeight: 700,
                  color: "#6e6e80",
                  textTransform: "uppercase",
                  letterSpacing: 0.4
                }}
              >
                Pinned
              </Typography>
            </Box>

            <List dense sx={{ px: 1, pb: 1 }}>
              {pinned.map((query) => (
                <ListItem
                  key={`pinned-${query}`}
                  disablePadding
                  secondaryAction={
                    <IconButton edge="end" size="small" onClick={() => unpinSearch(query)}>
                      <PushPinIcon sx={{ fontSize: 16 }} />
                    </IconButton>
                  }
                >
                  <ListItemButton
                    onClick={() => onSearch(query)}
                    sx={{
                      borderRadius: 2,
                      mx: 1,
                      py: 1
                    }}
                  >
                    <SearchIcon sx={{ fontSize: 16, color: "#6e6e80", mr: 1.2 }} />
                    <ListItemText
                      primary={query}
                      primaryTypographyProps={{
                        fontSize: 13,
                        noWrap: true
                      }}
                    />
                  </ListItemButton>
                </ListItem>
              ))}
            </List>

            <Divider sx={{ mx: 2, mb: 1.5 }} />
          </>
        )}

        <Box
          sx={{
            px: 2,
            pb: 1,
            display: "flex",
            alignItems: "center",
            justifyContent: "space-between"
          }}
        >
          <Typography
            sx={{
              fontSize: 12,
              fontWeight: 700,
              color: "#6e6e80",
              textTransform: "uppercase",
              letterSpacing: 0.4
            }}
          >
            Recenti
          </Typography>

          {visibleHistory.length > 0 && (
            <IconButton size="small" onClick={clearHistory}>
              <DeleteIcon sx={{ fontSize: 16 }} />
            </IconButton>
          )}
        </Box>

        <List
          dense
          sx={{
            px: 1,
            overflowY: "auto",
            flex: 1
          }}
        >
          {visibleHistory.length === 0 ? (
            <Box px={2} py={2}>
              <Typography sx={{ fontSize: 13, color: "#8e8ea0", lineHeight: 1.5 }}>
                Nessuna cronologia ancora. Avvia una ricerca agentica per vedere qui le ultime chat.
              </Typography>
            </Box>
          ) : (
            visibleHistory.map((item) => {
              const isPinned = pinned.includes(item.query)

              return (
                <ListItem
                  key={`${item.query}-${item.results}`}
                  disablePadding
                  secondaryAction={
                    <IconButton
                      edge="end"
                      size="small"
                      onClick={() =>
                        isPinned ? unpinSearch(item.query) : pinSearch(item.query)
                      }
                    >
                      <PushPinIcon
                        sx={{
                          fontSize: 16,
                          color: isPinned ? "#202123" : "#b5b5c3"
                        }}
                      />
                    </IconButton>
                  }
                >
                  <ListItemButton
                    onClick={() => onSearch(item.query)}
                    sx={{
                      borderRadius: 2,
                      mx: 1,
                      py: 1.1,
                      alignItems: "flex-start"
                    }}
                  >
                    <ChatBubbleOutlineIcon
                      sx={{
                        fontSize: 16,
                        color: "#6e6e80",
                        mr: 1.2,
                        mt: 0.15
                      }}
                    />

                    <ListItemText
                      primary={item.query}
                      secondary={`${item.results} risultati`}
                      primaryTypographyProps={{
                        fontSize: 13,
                        noWrap: true,
                        color: "#202123"
                      }}
                      secondaryTypographyProps={{
                        fontSize: 11.5,
                        color: "#8e8ea0"
                      }}
                    />
                  </ListItemButton>
                </ListItem>
              )
            })
          )}
        </List>
      </Drawer>

      <Box
        sx={{
          flex: 1,
          minWidth: 0,
          display: "flex",
          flexDirection: "column",
          bgcolor: "#ffffff"
        }}
      >
        {children}
      </Box>
    </Box>
  )
}
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
  Button
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
          item =>
            item &&
            typeof item.query === "string" &&
            typeof item.results === "number"
        )
      )

      setPinned(
        readArray<string>(PINNED_KEY).filter(
          item => typeof item === "string" && item.trim().length > 0
        )
      )
    }

    syncSidebar()
    window.addEventListener("search_history_updated", syncSidebar)

    return () =>
      window.removeEventListener("search_history_updated", syncSidebar)
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
        {/* Header brand */}
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
                eBay AI Search
              </Typography>

              <Typography
                sx={{
                  fontSize: 12,
                  color: "#6e6e80",
                  lineHeight: 1.2
                }}
              >
                chat, ranking, seller trust
              </Typography>
            </Box>
          </Box>
        </Box>

        {/* New chat */}
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

        {/* Auth block */}
        <Box px={2} pb={2}>
          <AuthPanel />
        </Box>

        <Divider />

        {/* Saved */}
        {pinned.length > 0 && (
          <>
            <Box
              sx={{
                px: 2,
                pt: 2,
                pb: 0.75,
                display: "flex",
                alignItems: "center",
                gap: 1
              }}
            >
              <PushPinIcon sx={{ fontSize: 15, color: "#8e8ea0" }} />
              <Typography
                sx={{
                  fontSize: 12,
                  fontWeight: 700,
                  letterSpacing: 0.3,
                  textTransform: "uppercase",
                  color: "#8e8ea0"
                }}
              >
                Salvati
              </Typography>
            </Box>

            <List dense sx={{ px: 1 }}>
              {pinned.map(query => (
                <ListItem key={query} disablePadding sx={{ mb: 0.5 }}>
                  <ListItemButton
                    onClick={() => onSearch(query)}
                    sx={{
                      borderRadius: 2,
                      minHeight: 42,
                      "&:hover": {
                        bgcolor: "#ececf1"
                      }
                    }}
                  >
                    <PushPinIcon sx={{ fontSize: 16, mr: 1.2, color: "#777" }} />

                    <ListItemText
                      primary={query}
                      primaryTypographyProps={{
                        fontSize: 13,
                        noWrap: true,
                        sx: { maxWidth: 160, color: "#202123" }
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

            <Divider sx={{ mt: 1 }} />
          </>
        )}

        {/* History */}
        <Box flex={1} overflow="auto">
          <Box
            sx={{
              px: 2,
              pt: 2,
              pb: 0.75,
              display: "flex",
              alignItems: "center",
              justifyContent: "space-between"
            }}
          >
            <Box sx={{ display: "flex", alignItems: "center", gap: 1 }}>
              <SearchIcon sx={{ fontSize: 15, color: "#8e8ea0" }} />
              <Typography
                sx={{
                  fontSize: 12,
                  fontWeight: 700,
                  letterSpacing: 0.3,
                  textTransform: "uppercase",
                  color: "#8e8ea0"
                }}
              >
                Cronologia
              </Typography>
            </Box>

            {visibleHistory.length > 0 && (
              <Button
                size="small"
                onClick={clearHistory}
                sx={{
                  minWidth: 0,
                  px: 0.5,
                  textTransform: "none",
                  fontSize: 11,
                  color: "#8e8ea0"
                }}
              >
                Pulisci
              </Button>
            )}
          </Box>

          {visibleHistory.length === 0 ? (
            <Typography
              sx={{
                px: 2,
                py: 2,
                fontSize: 13,
                color: "#8e8ea0",
                lineHeight: 1.5
              }}
            >
              {loggedIn
                ? "Le tue ricerche recenti compariranno qui."
                : "Inizia una ricerca per vedere qui la cronologia recente."}
            </Typography>
          ) : (
            <List dense sx={{ px: 1 }}>
              {visibleHistory.map(item => {
                const isPinned = pinned.includes(item.query)

                return (
                  <ListItem key={item.query} disablePadding sx={{ mb: 0.5 }}>
                    <ListItemButton
                      onClick={() => onSearch(item.query)}
                      sx={{
                        borderRadius: 2,
                        alignItems: "flex-start",
                        py: 1,
                        "&:hover": {
                          bgcolor: "#ececf1"
                        }
                      }}
                    >
                      <ChatBubbleOutlineIcon
                        sx={{
                          fontSize: 16,
                          mr: 1.2,
                          mt: 0.2,
                          color: "#777"
                        }}
                      />

                      <ListItemText
                        primary={item.query}
                        secondary={`${item.results} risultati`}
                        primaryTypographyProps={{
                          fontSize: 13,
                          sx: {
                            maxWidth: 160,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            color: "#202123"
                          }
                        }}
                        secondaryTypographyProps={{
                          fontSize: 11,
                          color: "#8e8ea0"
                        }}
                      />

                      <IconButton
                        size="small"
                        aria-label={
                          isPinned ? "Già salvata" : "Salva ricerca"
                        }
                        onClick={event => {
                          event.stopPropagation()
                          pinSearch(item.query)
                        }}
                      >
                        <PushPinIcon
                          fontSize="small"
                          sx={{
                            color: isPinned ? "#202123" : "#a8a8b3"
                          }}
                        />
                      </IconButton>
                    </ListItemButton>
                  </ListItem>
                )
              })}
            </List>
          )}
        </Box>
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
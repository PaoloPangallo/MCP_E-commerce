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
import ChatBubbleOutlineIcon from "@mui/icons-material/ChatBubbleOutline"
import SearchIcon from "@mui/icons-material/Search"
import AutoAwesomeIcon from "@mui/icons-material/AutoAwesome"
import { useAuth } from "../auth/useAuth.ts";
import AuthPanel from "../auth/ui/AuthPanel.tsx";



// La struttura precedente era basata su localStorage, ora usiamo il DB.
// Manteniamo PINNED per ora come logica locale opzionale.

interface Props {
  children: React.ReactNode
  onNewChat: () => void
  onLoadHistory?: (sid: string) => void
  activeSessionId?: string
}

const HISTORY_KEY = "search_history"


export default function ChatLayout({
  children,
  onNewChat,
  onLoadHistory,
  activeSessionId
}: Props) {
  const { loggedIn } = useAuth()

  const [history, setHistory] = useState<any[]>([])

  const fetchChats = async () => {
    try {
      const { getChats } = await import("../api/searchApi")
      const chats = await getChats()
      setHistory(chats)
    } catch (e) {
      console.error("Error fetching chats", e)
    }
  }

  useEffect(() => {
    fetchChats()

    window.addEventListener("search_history_updated", fetchChats)
    return () => window.removeEventListener("search_history_updated", fetchChats)
  }, [])

  const clearHistory = () => {
    // In un sistema a DB, "pulisci" potrebbe significare cancellare tutto o solo nascondere
    localStorage.removeItem(HISTORY_KEY)
    setHistory([])
  }

  const handleDeleteChat = async (sid: string, e: React.MouseEvent) => {
    e.stopPropagation()
    try {
      const { deleteChat } = await import("../api/searchApi")
      await deleteChat(sid)
      fetchChats()
    } catch (err) {
      console.error(err)
    }
  }

  const visibleHistory = useMemo(() => history.slice(0, 20), [history])

  return (
    <Box display="flex" height="100vh" bgcolor="#ffffff">
      <Drawer
        variant="permanent"
        sx={{
          width: 340,
          flexShrink: 0,
          "& .MuiDrawer-paper": {
            width: 340,
            bgcolor: "#f9f9fb",
            borderRight: "1px solid #e5e5e5",
            display: "flex",
            flexDirection: "column",
            boxShadow: "4px 0 10px rgba(0,0,0,0.02)"
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
                const isActive = activeSessionId === item.id

                return (
                  <ListItem
                    key={item.id}
                    disablePadding
                    sx={{ mb: 0.5 }}
                    secondaryAction={
                      <IconButton
                        edge="end"
                        size="small"
                        onClick={(e) => handleDeleteChat(item.id, e)}
                        sx={{ opacity: 0.6, "&:hover": { opacity: 1 } }}
                      >
                        <DeleteIcon fontSize="inherit" />
                      </IconButton>
                    }
                  >
                    <ListItemButton
                      onClick={() => onLoadHistory?.(item.id)}
                      sx={{
                        borderRadius: 2,
                        alignItems: "flex-start",
                        py: 1.2,
                        bgcolor: isActive ? "#f0f0f5" : "transparent",
                        border: isActive ? "1px solid #e0e0e0" : "1px solid transparent",
                        "&:hover": {
                          bgcolor: isActive ? "#f0f0f5" : "#f7f7f8"
                        }
                      }}
                    >
                      <ChatBubbleOutlineIcon
                        sx={{
                          fontSize: 16,
                          mr: 1.2,
                          mt: 0.3,
                          color: isActive ? "#202123" : "#777"
                        }}
                      />

                      <ListItemText
                        primary={item.title}
                        secondary={new Date(item.updated_at).toLocaleDateString()}
                        primaryTypographyProps={{
                          fontSize: 13,
                          sx: {
                            maxWidth: 180,
                            overflow: "hidden",
                            textOverflow: "ellipsis",
                            whiteSpace: "nowrap",
                            color: "#202123",
                            fontWeight: isActive ? 600 : 400
                          }
                        }}
                        secondaryTypographyProps={{
                          fontSize: 10,
                          color: "#8e8ea0"
                        }}
                      />
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